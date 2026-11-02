"""Collect paired early / observed demonstrations into chunked LeRobot datasets (docs/PIF.md §8.12).

One chunk = (task, case, split): <root>/<task>/<case>/<split>/{early,observed} are two LeRobotDatasets whose i-th
episodes form a pair (same seed, same exogenous conditions = sha256 of the episode variation). A candidate seed is
run with expert.rule early and observed; both must succeed and pass the collection gate, else the pair is rejected
(journal.jsonl keeps every attempt with both sides' reasons) and the next seed is tried. A finished chunk has DONE;
an unfinished chunk is preserved; explicit --archive-incomplete moves it aside before retrying.
Demonstrations only: no policy, no test filtering. The HY-Motion clip split = the data split.
"""
import argparse
import fcntl
import gc
import hashlib
import json
import shutil
import time
from pathlib import Path
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from intent_policy.benchmark.paired_collection import collect_pairs, IncompleteQuota
from intent_policy.benchmark.runner import ExpertAgent
from intent_policy.benchmark.collection_integrity import source_manifest, motion_source_hash
from intent_policy.scenarios.config import _deep_merge
from intent_policy.scenarios.scenario_registry import make_scenario
from intent_policy.utils import controller_config, load_yaml, observation_config, resolve
from scripts.collect_demos import features, record_episode, episode_meta

SPLIT_OFFSET = {'train': 0, 'val': 5000}


def variation_hash(variation) -> str:
    dump = lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)
    return hashlib.sha256(json.dumps(variation, sort_keys=True, default=dump).encode()).hexdigest()


def chunk_overrides(task: dict, case: dict, split: str, mode: str) -> dict:
    """Case overrides + the information rule (task `rules`: expert settings per rule, else expert.rule) + the
    HY-Motion clip split of the data split."""
    rule = (task.get('rules') or {}).get(mode, {'expert': {'rule': mode}})
    o = _deep_merge(_deep_merge(task.get('overrides') or {}, case), rule)
    return _deep_merge(o, {'human_behavior': {'motion_bank': {'split': split}}})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/experiments/ten_task_v8.yaml')
    p.add_argument('--task', required=True)
    p.add_argument('--case', required=True)
    p.add_argument('--split', required=True, choices=sorted(SPLIT_OFFSET))
    p.add_argument('--quota', type=int, help='override data.quota[split] (e.g. a preview)')
    p.add_argument('--root', help='override data.root')
    p.add_argument('--archive-incomplete', action='store_true', help='preserve unfinished chunk under a timestamped name before retrying')
    args = p.parse_args()
    # Keep the descriptor alive for this main() invocation; all entry points share this lock.
    lock_path = resolve('outputs/collect/.collector.lock')
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    collector_lock = lock_path.open('a')
    try:
        fcntl.flock(collector_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Another dataset collector is running')
    exp = load_yaml(args.config)
    data, task = exp['data'], exp['tasks'][args.task]
    case = task['cases'][args.case]
    quota = args.quota or int(data['quota'][args.split])
    root = resolve(args.root or data['root']) / args.task / args.case / args.split
    if (root / 'DONE').exists():
        manifest = root / 'provenance/source_manifest.json'
        saved_config = root / 'provenance/experiment.json'
        if (not manifest.exists() or not saved_config.exists()
                or json.loads(manifest.read_text()) != source_manifest(Path(__file__).resolve().parents[1])
                or json.loads(saved_config.read_text()) != exp):
            raise RuntimeError(f'Completed chunk has different/unknown provenance: {root}; use a new version root')
        print(f'{root}: done (source/config verified)'); return
    if root.exists():
        if not args.archive_incomplete:
            raise RuntimeError(f'Unfinished chunk preserved at {root}; use a new root or --archive-incomplete')
        root.rename(root.with_name(root.name + '.incomplete-' + str(time.time_ns())))
    root.mkdir(parents=True)
    project = Path(__file__).resolve().parents[1]
    frozen = source_manifest(project)
    provenance = root / 'provenance'
    for relative in frozen:
        target = provenance / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project / relative, target)
    (provenance / 'source_manifest.json').write_text(json.dumps(frozen, indent=2))
    (provenance / 'experiment.json').write_text(json.dumps(exp, indent=2))
    obs_cfg, ctrl_cfg = observation_config(exp), controller_config(exp)
    keypoints = list(exp['intent']['keypoints'])
    gate = _deep_merge(_deep_merge({}, exp['gates']['default']), task['gate'])
    gate['scene_coverage_reviewed'] = task['scene_coverage_reviewed']
    tasks, cases = list(exp['tasks']), list(task['cases'])
    seed0 = int(data['seed_base']) + 100000 * tasks.index(args.task) + 10000 * cases.index(args.case) + SPLIT_OFFSET[args.split]
    ds, meta, scen = {}, {'early': [], 'observed': []}, {}
    for mode in ('early', 'observed'):
        ds[mode] = LeRobotDataset.create(repo_id=f"local/{exp['experiment']}_{args.task}_{args.case}_{args.split}_{mode}", fps=20,
                                         features=features(obs_cfg, None, keypoints), root=root / mode,
                                         robot_type='ur3e_susgrip', use_videos=True, image_writer_threads=4)
        overrides = chunk_overrides(task, case, args.split, mode)
        if data.get('paired_source_selection'):
            overrides = _deep_merge(overrides, {'human_behavior': {'motion_bank': {'paired_source_selection': True}}})
        scen[mode] = make_scenario(task['scenario'], overrides)
        scen[f'{mode}_overrides'] = overrides
    journal = (root / 'journal.jsonl').open('a')
    results = {}

    def run(mode, candidate):
        if source_manifest(project) != frozen:
            raise RuntimeError('Source/config changed during collection; chunk preserved, restart in a new version')
        # a new scenario per episode: a reused one carried state between seeds (I2 ended on the left arm, 2026-10-31),
        # and the pilots / gates run every episode from a new one
        scen[mode].close()
        scen[mode] = None
        gc.collect()                 # a closed scenario sits in reference cycles: +120-150 MB per episode otherwise
        sc = scen[mode] = make_scenario(task['scenario'], scen[f'{mode}_overrides'])
        rec, frames, info = record_episode(sc, ExpertAgent(), candidate['seed'], ctrl_cfg, obs_cfg, None, 0, None,
                                           keypoints, None, gate)
        rec['collection_quality'] = info['collection_quality']
        rec['exogenous_condition_sha256'] = variation_hash(rec['variation'])
        rec['motion_source_sha256'] = motion_source_hash(rec)
        results[mode] = (rec, frames, info)
        return rec

    def write_pair(candidate, records):
        for mode in ('early', 'observed'):
            rec, frames, info = results[mode]
            sc = scen[mode]
            for fr in frames:
                ds[mode].add_frame({**fr, 'task': sc.cfg.task})
            ds[mode].save_episode()
            m = episode_meta(len(meta[mode]), task['scenario'], args.split, candidate['seed'], sc, rec, frames, info)
            m.update(pair_id=candidate['pair_id'], task_id=args.task, case=args.case, mode=mode,
                     exogenous_condition_sha256=rec['exogenous_condition_sha256'],
                     motion_source_sha256=rec['motion_source_sha256'])
            meta[mode].append(m)

    def write_attempt(entry):
        entry['t_wall'] = round(time.time() - t0, 1)
        journal.write(json.dumps(entry, default=str) + '\n')
        journal.flush()
        print(f"[{args.task}/{args.case}/{args.split}] {entry['pair_id']} accepted={entry['accepted']} "
              f"{' '.join(entry['reasons'])} kept={sum(1 for _ in meta['early'])} [{entry['t_wall']:.0f}s]", flush=True)

    t0 = time.time()
    n_max = quota * int(data['max_attempts_factor'])
    candidates = ({'pair_id': f'{args.task}_{args.case}_{args.split}_{seed0 + i}', 'seed': seed0 + i} for i in range(n_max))
    status = None
    try:
        status = collect_pairs(candidates, quota, run, write_pair, write_attempt,
                               require_motion_match=data.get('require_matched_motion_sources', False))
    except IncompleteQuota as exc:
        status = dict(complete=False, accepted_pairs=exc.accepted, attempts=exc.attempts, requested=quota)
    finally:
        journal.close()
        for mode in ('early', 'observed'):
            scen[mode].close()
    dump = lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)
    for mode in ('early', 'observed'):
        ds[mode].finalize()
        with open(root / mode / 'meta/hri_episodes.jsonl', 'w') as f:
            for m in meta[mode]:
                f.write(json.dumps(m, default=dump) + '\n')
        (root / mode / 'meta/hri_experiment_config.json').write_text(json.dumps(
            dict(exp, chunk=dict(task=args.task, case=args.case, split=args.split, mode=mode)), indent=2, default=dump))
    (root / 'status.json').write_text(json.dumps(status, indent=2))
    if status.get('complete'):
        (root / 'DONE').write_text(json.dumps(status))
    print(f'{root}: {status}')
    if not status.get('complete'):
        raise SystemExit(2)


if __name__ == '__main__':
    main()

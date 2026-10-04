"""Evaluate an agent (trained checkpoint or scripted expert) on the benchmark scenarios.

Writes one episode record per episode (config, variation, per-step state, logits, probabilities,
selected action, mapped command, oracle given/true, events, metrics) and aggregates the HRIBench-style
metrics per split, scenario and HRI role (no composite score).

Without `--protocol` the experiment's `benchmark` section is used (one split). With a protocol file (Phase 2)
every split is evaluated; `--condition` picks an oracle condition from the protocol (only for checkpoints that
read oracle features). A split with `scenario_list` (configs/scenario_lists/eval_v1.jsonl, the same fixed episodes
for every compared condition, scence_construct.md §5.3) is evaluated per list task (T1 ... T5, T4neg); otherwise
`scenarios` x seeds. `--workers` runs task/split jobs in parallel processes (each loads its own policy).

Output (one folder per run, as LeRobot lays out runs): `--output-dir`, else outputs/eval/<date>/<time>_<experiment>_<agent>,
holding results.json / results.md, episodes/ (records), videos/<task>/ (mp4 recorded while the episodes run: the first
`--videos` episodes of each task and, for checkpoints, up to `--failure-videos` failed ones), log.txt and command.txt.

  ./run.sh -m scripts.evaluate --config configs/experiments/intent_act_late.yaml \
      --checkpoint outputs/train/intent_act/A/checkpoints/010000/pretrained_model --workers 2
  ./run.sh -m scripts.evaluate --config configs/experiments/phase2_oracle.yaml \
      --protocol configs/benchmark/phase2_protocol_v1.yaml --checkpoint <ckpt> --condition delay_200ms \
      --output-dir outputs/eval/phase2/oracle_s0/delay_200ms --workers 3
"""
import argparse
import json
import multiprocessing as mp
from collections import defaultdict
from datetime import datetime, timezone

import torch

from intent_policy.benchmark.metrics import aggregate, ALL_METRICS
from intent_policy.benchmark.runner import ExpertAgent, run_episode, save_record
from intent_policy.benchmark.video import EpisodeVideo
from intent_policy.scenarios.config import _deep_merge, load_scenario_list
from intent_policy.scenarios.scenario_registry import make_scenario, scenario_overrides
from intent_policy.utils import checkpoint_name, controller_config, load_yaml, log_run, observation_config, resolve, run_dir


def fmt(v):
    if v == 'n/a':
        return 'n/a'
    if v['value'] is None:
        return '— (n=0)'
    return f"{v['value']:.3f} ± {v['std']:.3f} (n={v['n']})"


def markdown(results: dict) -> str:
    agent = results['agent']
    lines = [f"# Evaluation — {agent.get('type')} ({agent.get('action_mode', '')})", '',
             f"checkpoint: `{agent.get('checkpoint', '-')}`  ", f"protocol: {results.get('protocol', '-')}  ",
             f"oracle condition: {agent.get('oracle_condition') or agent.get('intent_provider', 'none')}  ",
             f"created: {results['created']}", '']
    for split, sres in results['splits'].items():
        lines += [f"## Split: {split} (seeds {sres['seeds']})", '']
        for role, block in sres['by_role'].items():
            metrics = [m for m in ALL_METRICS if block['metrics'][m] != 'n/a']
            lines += [f'### Role: {role}', '', '| scenario | episodes | ' + ' | '.join(metrics) + ' |',
                      '|---|---|' + '---|' * len(metrics)]
            for sid in block['scenarios']:
                s = sres['by_scenario'][sid]
                lines.append(f"| {sid} | {s['episodes']} | " + ' | '.join(fmt(s['metrics'][m]) for m in metrics) + ' |')
            lines.append(f"| **all {role}** | {block['episodes']} | " + ' | '.join(fmt(block['metrics'][m]) for m in metrics) + ' |')
            lines.append('')
        lines += ['**Failure reasons:**', '']
        for sid, s in sres['by_scenario'].items():
            lines.append(f"- {sid}: {s['failures'] or 'none'}")
    if results.get('videos'):
        lines += ['', '## Videos', '']
        lines += [f"- [{v['task']} #{v['index']} seed {v['seed']}: {'success' if v['success'] else v['failure']}]({v['path']})"
                  for v in results['videos']]
        lines.append('')
    return '\n'.join(lines) + '\n'


def build_agent(args_d: dict, eval_seeds: list[int], ctrl_cfg):
    if args_d['expert']:
        return ExpertAgent()
    from intent_policy.policies.policy_agent import PolicyAgent
    return PolicyAgent(resolve(args_d['checkpoint']), args_d['device'], args_d['oracle_condition'], eval_seeds, ctrl_cfg,
                       args_d['temporal_ensemble_coeff'])


def run_job(job: dict) -> dict:
    """One (split, scenario) block: returns per-episode metrics; records are written to disk."""
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    torch.set_num_threads(max(1, job['threads']))
    out = resolve(job['output_dir'])
    if job['worker']:
        log_run(out)                            # a pool process: its lines go into the run's log.txt as well
    exp = job['exp']
    obs_cfg, ctrl_cfg = observation_config(exp), controller_config(exp)
    agent = build_agent(job['args'], job['seeds'], ctrl_cfg)
    ov = _deep_merge(scenario_overrides(exp, job['scenario']),
                     scenario_overrides({'scenario_overrides': job['split_overrides']}, job['scenario']))
    sc = make_scenario(job['scenario'], ov)
    eps, failures, videos, first, failed = [], defaultdict(int), [], 0, 0
    try:
        for e in job['entries']:
            # videos as lerobot-eval records them, while the episode runs: the first episodes of the task, and failed
            # ones (every episode is recorded while failure slots are left; a success beyond the first is dropped)
            take_first, take_failed = first < job['videos'], failed < job['failure_videos']
            name = f"{job['label']}_{e['index']:03d}__seed{e['seed']}" if e.get('index') is not None else f"{job['label']}__seed{e['seed']}"
            folder = out / 'videos' / job['label']
            video = EpisodeVideo(folder, name) if take_first or take_failed else None
            rec = run_episode(sc, agent, e['seed'], ctrl_cfg, obs_cfg, keep_trace=not job['no_traces'], spec=e['spec'],
                              on_frame=video)
            rec['experiment'], rec['split'], rec['protocol'] = exp['experiment'], job['split'], job['protocol']
            rec['list_task'], rec['list_index'] = job['label'], e.get('index')
            save_record(rec, out / 'episodes' / job['split'] / job['label'] / f"{rec['episode_id']}.json.gz")
            eps.append(rec['metrics'])
            if not rec['success']:
                failures[rec['failure']] += 1
            if video is not None:
                keep = take_first or not rec['success']
                path = folder / f"{name}__{'success' if rec['success'] else rec['failure']}.mp4" if keep else None
                if video.finish(rec, path) is not None:
                    first, failed = (first + 1, failed) if take_first else (first, failed + 1)
                    videos.append(dict(task=job['label'], index=e.get('index'), seed=e['seed'], success=rec['success'],
                                       failure=rec['failure'], path=str(path.relative_to(out))))
            print(f"[{job['split']}] {job['label']} seed={e['seed']} success={rec['success']} failure={rec['failure']} "
                  f"T={rec['duration_s']:.1f}s decisions={rec['n_decisions']}", flush=True)
        role, applicable = sc.cfg.role, sc.cfg.applicable_metrics
    finally:
        sc.close()
    return dict(split=job['split'], scenario=job['label'], role=role, applicable=applicable, metrics=eps,
                failures=dict(failures), agent=agent.describe(), videos=videos)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/experiments/foundation_baseline.yaml')
    p.add_argument('--protocol', help='versioned evaluation protocol (splits, seeds, oracle conditions)')
    p.add_argument('--checkpoint', help='pretrained_model dir; omit with --expert')
    p.add_argument('--expert', action='store_true', help='evaluate the privileged scripted expert (reference)')
    p.add_argument('--condition', default='correct', help='oracle condition name from the protocol')
    p.add_argument('--episodes-per-scenario', type=int)
    p.add_argument('--seed-start', type=int)
    p.add_argument('--scenarios', nargs='*', help='scenario ids or list tasks (T1 ... T5, T4neg)')
    p.add_argument('--splits', nargs='*')
    p.add_argument('--output-dir', help='default: outputs/eval/<date>/<time>_<experiment>_<agent>')
    p.add_argument('--device')
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--no-traces', action='store_true', help='do not keep per-step traces in records')
    p.add_argument('--videos', type=int, default=2, help='record the first N episodes of each task (0: none)')
    p.add_argument('--failure-videos', type=int, default=None,
                   help='also record up to N failed episodes of each task (default 3; 0 with --expert: the expert '
                        'needs no images, so failure videos would render every episode)')
    p.add_argument('--temporal-ensemble-coeff', type=float, default=None,
                   help='inference-time ACT temporal ensembling (default: protocol `inference`, else the checkpoint config)')
    args = p.parse_args()
    exp = load_yaml(args.config)
    out = run_dir('eval', f"{exp['experiment']}_{'expert' if args.expert else checkpoint_name(args.checkpoint)}", args.output_dir)
    log_run(out, 'scripts.evaluate')
    failure_videos = args.failure_videos if args.failure_videos is not None else (0 if args.expert else 3)
    if args.protocol:
        protocol = load_yaml(args.protocol)
        splits = protocol['splits']
        oracle_condition = protocol['conditions'][args.condition]
        protocol_name = protocol['protocol']
        ensemble = protocol.get('inference', {}).get('temporal_ensemble_coeff', 'config')
    else:
        bench = exp['benchmark']
        splits = [dict(name='default', scenario_list=bench['eval_list'])] if bench.get('eval_list') else \
            [dict(name='default', scenarios=bench['scenarios'], seed_start=bench['eval_seed_start'],
                  episodes_per_scenario=bench['eval_episodes_per_scenario'])]
        oracle_condition, protocol_name = {'condition': 'correct'}, exp['experiment']
        ensemble = 'config'
    if args.temporal_ensemble_coeff is not None:
        ensemble = args.temporal_ensemble_coeff
    jobs = []
    for split in splits:
        if args.splits and split['name'] not in args.splits:
            continue
        if split.get('scenario_list'):          # fixed eval episodes, one job per list task
            blocks = defaultdict(list)
            for e in load_scenario_list(split['scenario_list']):
                blocks[(e['task'], e['scenario_id'])].append(e)
            groups = [(task, sid, es[:args.episodes_per_scenario] if args.episodes_per_scenario else es)
                      for (task, sid), es in blocks.items() if not args.scenarios or task in args.scenarios or sid in args.scenarios]
        else:
            n = args.episodes_per_scenario or split['episodes_per_scenario']
            seed0 = split['seed_start'] if args.seed_start is None else args.seed_start
            groups = [(sid, sid, [dict(seed=s, spec=None, index=None) for s in range(seed0, seed0 + n)])
                      for sid in split['scenarios'] if not args.scenarios or sid in args.scenarios]
        for label, sid, entries in groups:
            jobs.append(dict(exp=exp, split=split['name'], scenario=sid, label=label, entries=entries,
                             seeds=[int(e['seed']) for e in entries],
                             split_overrides=split.get('scenario_overrides'), protocol=protocol_name,
                             output_dir=str(out), no_traces=args.no_traces, worker=args.workers > 1,
                             videos=args.videos, failure_videos=failure_videos,
                             threads=max(1, 12 // max(args.workers, 1)),
                             args=dict(expert=args.expert, checkpoint=args.checkpoint, device=args.device,
                                       oracle_condition=oracle_condition, temporal_ensemble_coeff=ensemble)))
    if args.workers > 1:
        with mp.get_context('spawn').Pool(args.workers) as pool:
            blocks = pool.map(run_job, jobs, chunksize=1)
    else:
        blocks = [run_job(j) for j in jobs]
    results = dict(experiment=exp, protocol=protocol_name, agent=blocks[0]['agent'], condition=args.condition,
                   controller=controller_config(exp).to_dict(), observation=observation_config(exp).to_dict(),
                   created=datetime.now(timezone.utc).isoformat(), splits={},
                   videos=sorted((v for b in blocks for v in b['videos']), key=lambda v: (v['task'], v['index'] or 0, v['seed'])))
    for split in dict.fromkeys(b['split'] for b in blocks):
        sblocks, merged = [b for b in blocks if b['split'] == split], {}
        for b in sblocks:                       # a list task can span scenarios (T5: T1-T3 bases): one row per label
            m = merged.setdefault(b['scenario'], dict(b, metrics=[], failures=defaultdict(int)))
            m['metrics'] = m['metrics'] + b['metrics']
            for k, v in b['failures'].items():
                m['failures'][k] += v
        by_scenario = {k: dict(role=b['role'], episodes=len(b['metrics']), metrics=aggregate(b['metrics']),
                               failures=dict(b['failures']), applicable_metrics=b['applicable'],
                               episode_metrics=b['metrics']) for k, b in merged.items()}
        by_role = {}
        for role in dict.fromkeys(b['role'] for b in sblocks):
            rb = [b for b in sblocks if b['role'] == role]
            by_role[role] = dict(scenarios=list(dict.fromkeys(b['scenario'] for b in rb)), episodes=sum(len(b['metrics']) for b in rb),
                                 metrics=aggregate([m for b in rb for m in b['metrics']]))
        seeds = [j['seeds'] for j in jobs if j['split'] == split][0]
        results['splits'][split] = dict(seeds=[seeds[0], seeds[-1]], by_scenario=by_scenario, by_role=by_role,
                                        overall=aggregate([m for b in sblocks for m in b['metrics']]))
    (out / 'results.json').write_text(json.dumps(results, indent=2, default=str))
    (out / 'results.md').write_text(markdown(results))
    print(markdown(results))
    print(f'run folder: {out}')


if __name__ == '__main__':
    main()

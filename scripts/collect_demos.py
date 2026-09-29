"""Record scripted-expert demonstrations as a standard LeRobotDataset (v3.0).

Per frame (20 Hz, one control decision):
  observation.state          [10] joint positions, gripper opening, TCP xyz (world frame, table top z=0.62)
  observation.images.<cam>   video (H, W, 3), LeRobot default encoder (AV1, GOP 2)
  action                     [7]  joint/gripper target of the expert's action (= continuous ACT target)
  restricted_action          [1]  restricted-action id (names in meta/hri_restricted_actions.json)
  task                       the scenario's natural-language task (generic, never the episode's target)
With `intent.provider: oracle` (Phase 2) every frame also stores the oracle fields
(`observation.oracle.*`, intent/representation.py: target identity/validity/position, current and
future hand position/velocity; never an input of the No-Information policy) and `hri.seed`,
`hri.scenario_index`, `hri.noise_injected`.

Only successful episodes are stored. Per-episode ground truth (scenario, seed, variation, expert and
human stage segments, protocol-step times, evidence / commitment / intention-change events, metrics)
goes to meta/hri_episodes.jsonl for inspection. View with `lerobot-dataset-viz` or
`scripts/view_dataset.py`.

Experiment configs may set `scenario_overrides` (deep-merged into the scenario YAMLs), `data.splits`
(e.g. nominal + intention_change episodes, each with its own overrides) and `data.noise` (DART-style
perturbation of a fraction of the episodes; the recorded labels stay the expert's clean actions).
Scenarios with `scene_variation.twin_pairs` are recorded in pairs (seeds 2k, 2k+1 differ only in the
human's choice); a pair is kept only if both episodes succeed.
"""
import argparse
import json
import shutil
import time
from collections import Counter
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import create_lerobot_dataset_card
from benchmark.runner import run_episode, ExpertAgent, NoisyExpertAgent
from controllers.restricted_action import RestrictedAction
from intent.oracle import OracleIntentProvider
from intent.representation import dataset_features as oracle_features, features as oracle_values
from scenarios.config import _deep_merge
from scenarios.scenario_registry import make_scenario, scenario_overrides
from scripts.common import load_yaml, resolve, observation_config, controller_config

JOINTS = ['shoulder_pan', 'shoulder_lift', 'elbow', 'wrist_1', 'wrist_2', 'wrist_3']
STATE_NAMES = [f'{j}.pos' for j in JOINTS] + ['gripper.pos', 'tcp.x', 'tcp.y', 'tcp.z']
ACTION_NAMES = [f'{j}.pos' for j in JOINTS] + ['gripper.pos']
HRI_FIELDS = ('hri.seed', 'hri.scenario_index', 'hri.noise_injected')


def features(obs_cfg, oracle_horizons: tuple | None = None) -> dict:
    f = {'observation.state': {'dtype': 'float32', 'shape': (10,), 'names': STATE_NAMES},
         'action': {'dtype': 'float32', 'shape': (7,), 'names': ACTION_NAMES},
         'restricted_action': {'dtype': 'int64', 'shape': (1,), 'names': ['restricted_action_id']}}
    for cam in obs_cfg.cameras:
        f[f'observation.images.{cam}'] = {'dtype': 'video', 'shape': (obs_cfg.height, obs_cfg.width, 3),
                                          'names': ['height', 'width', 'channels']}
    if oracle_horizons is not None:
        f.update(oracle_features(oracle_horizons))
        f.update({k: {'dtype': 'int64', 'shape': (1,), 'names': [k.removeprefix('hri.')]} for k in HRI_FIELDS})
    return f


def segments(names: list[str], actions: list[int] | None = None) -> list[dict]:
    """Run-length segments [{name, start, end}] (inclusive frame indices), with action counts."""
    out = []
    for i, n in enumerate(names):
        if out and out[-1]['name'] == n:
            out[-1]['end'] = i
        else:
            out.append({'name': n, 'start': i, 'end': i})
    if actions is not None:
        for s in out:
            s['actions'] = dict(Counter(RestrictedAction(a).name for a in actions[s['start']:s['end'] + 1]))
    return out


def frame_of(t: float) -> int:
    return int(round(t * 20))


def record_episode(sc, agent, seed, ctrl_cfg, obs_cfg, provider, scenario_index) -> tuple[dict, list, dict]:
    frames, stages, humans, targets, actions, commits, noise = [], [], [], [], [], [], []
    gt = {}

    def on_frame(obs, decision, target, scenario):
        if not frames:
            gt.update(scenario.get_ground_truth_intention_information())
        label = int(decision.extras.get('expert_action', decision.action_id))
        joint = decision.extras.get('clean_joint_target', target)
        fr = {**obs, 'action': np.asarray(joint, np.float32), 'restricted_action': np.array([label], np.int64)}
        noisy = bool(decision.extras.get('noise', False))
        if provider is not None:
            fr.update(oracle_values(provider.record(scenario), scenario.tcp()))
            fr.update({'hri.seed': np.array([seed], np.int64), 'hri.scenario_index': np.array([scenario_index], np.int64),
                       'hri.noise_injected': np.array([int(noisy)], np.int64)})
        frames.append(fr)
        stages.append(decision.extras['expert_stage'])
        humans.append(scenario.human.stage)
        hs = scenario.human
        targets.append(' -> '.join(str(x) for x in (hs.selected_object, hs.selected_target) if x is not None) or 'none')
        actions.append(label)
        noise.append(noisy)
        if decision.extras.get('expert_commit'):
            commits.append(dict(t=round(scenario.time, 3), frame=len(frames) - 1, target=decision.extras['expert_commit']))

    rec = run_episode(sc, agent, seed, ctrl_cfg, obs_cfg, on_frame=on_frame, keep_trace=False)
    info = dict(gt=gt, stages=stages, humans=humans, targets=targets, actions=actions, commits=commits,
                noise_frames=int(sum(noise)))
    return rec, frames, info


def episode_meta(index, sid, split, seed, sc, rec, frames, info) -> dict:
    ev = rec['events']
    of = lambda kind: [dict(t=round(e['timestamp'], 3), frame=frame_of(e['timestamp']), **e['payload'])
                       for e in ev if e['event_type'] == kind]
    protocol = [dict(step=e['payload']['step'], t=round(e['timestamp'], 3), frame=frame_of(e['timestamp']), by=e['entity_id'])
                for e in ev if e['event_type'] == 'protocol_step_complete']
    cue = [round(e['timestamp'], 3) for e in ev if e['event_type'] == 'human_cue_onset']
    return dict(episode_index=index, scenario_id=sid, role=rec['role'], task=sc.cfg.task, seed=seed, split=split,
                expert=sc.cfg.expert, agent=rec['policy'], twin=rec['variation'].get('twin'),
                n_frames=len(frames), duration_s=round(rec['duration_s'], 3),
                ground_truth_intention_at_start=info['gt'], ground_truth_intention_at_end=sc.get_ground_truth_intention_information(),
                variation=rec['variation'], human_cue_onset_t=cue[0] if cue else None,
                intention_changes=of('human_intention_change'), intention_evident=of('human_intention_evident'),
                robot_target_commits=of('robot_target_commit'), expert_commits=info['commits'],
                noise_frames=info['noise_frames'], human_extras=sc.human.extras,
                protocol_steps=protocol, expert_stages=segments(info['stages'], info['actions']),
                human_stages=segments(info['humans']), human_target_stages=segments(info['targets']),
                action_counts=dict(Counter(RestrictedAction(a).name for a in info['actions'])),
                min_human_robot_distance=round(rec['min_human_robot_distance'], 4), metrics=rec['metrics'])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/experiments/foundation_baseline.yaml')
    p.add_argument('--episodes-per-scenario', type=int)
    p.add_argument('--root', help='override data.root')
    p.add_argument('--repo-id', help='override data.repo_id')
    p.add_argument('--seed-start', type=int, help='override data.seed_start')
    p.add_argument('--scenarios', nargs='*', help='subset of benchmark.scenarios')
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    exp = load_yaml(args.config)
    data = exp['data']
    data.update({k: v for k, v in dict(root=args.root, repo_id=args.repo_id, seed_start=args.seed_start).items()
                 if v is not None})
    splits = data.get('splits') or [dict(name='default', episodes_per_scenario=data['episodes_per_scenario'])]
    if args.episodes_per_scenario is not None:
        splits = [{**s, 'episodes_per_scenario': args.episodes_per_scenario} for s in splits]
    root = resolve(data['root'])
    if root.exists():
        if not args.overwrite:
            raise SystemExit(f'{root} exists; pass --overwrite to replace it')
        shutil.rmtree(root)
    obs_cfg, ctrl_cfg = observation_config(exp), controller_config(exp)
    intent = exp.get('intent') or {}
    horizons = tuple(intent.get('horizons_s', (0.5, 1.0))) if intent.get('provider') == 'oracle' else None
    provider = OracleIntentProvider(horizons) if horizons is not None else None
    noise_cfg = data.get('noise') or {}
    ds = LeRobotDataset.create(repo_id=data['repo_id'], fps=20, features=features(obs_cfg, horizons), root=root,
                               robot_type='ur3e_susgrip', use_videos=True, image_writer_threads=4)
    scenario_ids = list(exp['benchmark']['scenarios'])
    meta, seed = [], int(data['seed_start'])
    t0 = time.time()
    for split in splits:
        for sid in split.get('scenarios') or scenario_ids:
            if args.scenarios and sid not in args.scenarios:
                continue
            split_overrides = scenario_overrides({'scenario_overrides': split.get('scenario_overrides')}, sid)
            sc = make_scenario(sid, _deep_merge(scenario_overrides(exp, sid), split_overrides))
            twin = bool(sc.cfg.scene_variation.get('twin_pairs'))
            if twin and seed % 2:
                seed += 1
            done = rejected = noisy_groups = 0
            while done < int(split['episodes_per_scenario']):
                group = [seed, seed + 1] if twin else [seed]
                base = seed // 2 if twin else seed
                noisy = bool(noise_cfg) and np.random.default_rng(base * 7 + 11).random() < float(noise_cfg['episode_fraction'])
                results = []
                for s in group:
                    agent = NoisyExpertAgent(noise_cfg['burst_prob'], noise_cfg['burst_ticks']) if noisy else ExpertAgent()
                    results.append((s, *record_episode(sc, agent, s, ctrl_cfg, obs_cfg, provider, scenario_ids.index(sid))))
                    if not results[-1][1]['success']:
                        break
                if len(results) == len(group) and all(r[1]['success'] for r in results):
                    for s, rec, frames, info in results:
                        for fr in frames:
                            ds.add_frame({**fr, 'task': sc.cfg.task})
                        ds.save_episode()
                        meta.append(episode_meta(len(meta), sid, split['name'], s, sc, rec, frames, info))
                    done += len(group)
                    noisy_groups += noisy
                else:
                    rejected += len(group)
                    print(f"  {sid} seeds={group}: expert failed ({results[-1][1]['failure']}), group dropped", flush=True)
                seed += len(group)
            print(f"[{split['name']}] {sid}: {done} episodes kept ({noisy_groups} noisy groups), {rejected} rejected  "
                  f'[{time.time() - t0:.0f}s]', flush=True)
            sc.close()
    ds.finalize()
    with open(root / 'meta/hri_episodes.jsonl', 'w') as f:
        for m in meta:
            f.write(json.dumps(m, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)) + '\n')
    (root / 'meta/hri_restricted_actions.json').write_text(json.dumps(dict(
        ids={a.value: a.name for a in RestrictedAction}, controller=ctrl_cfg.to_dict()), indent=2))
    (root / 'meta/hri_experiment_config.json').write_text(json.dumps(exp, indent=2))
    card = create_lerobot_dataset_card(tags=['hri', 'ur3e', 'mujoco', 'scripted-expert', exp['experiment']],
                                       dataset_info=ds.meta.info, repo_id=data['repo_id'])
    card.save(root / 'README.md')
    print(f'dataset: {root}  episodes={ds.meta.total_episodes} frames={ds.meta.total_frames}')


if __name__ == '__main__':
    main()

"""Record chosen episodes as videos (re-runs them; scripts/evaluate.py already records videos while it evaluates).

For a look at particular episodes, e.g. a failure seen in an evaluation's results: same agent, controller,
observation, inference setting and scenario overrides as the evaluation (without `--protocol` the experiment's own
settings, as scripts/evaluate.py does), so a list episode replays as it was evaluated. Same frames as the
evaluation videos (intent_policy/benchmark/video.py). Writes `<task>_<index>__seed<N>__<result>.mp4` with the
episode record next to it and `summary.md`, into `--output-dir`, else outputs/videos/<date>/<time>_<name>.

Episodes come from a scenario list (`--list`, e.g. configs/episode_lists/v5/eval_list.jsonl; `--indices`, or
`--tasks` / `--per-task`) or from `--seeds` x the protocol split's scenarios (random specs).

  ./run.sh -m scripts.record_rollout --config configs/experiments/intent_act_late.yaml \
      --checkpoint outputs/train/intent_act/A/checkpoints/010000/pretrained_model \
      --list configs/episode_lists/v5/eval_list.jsonl --indices 48 144
  ./run.sh -m scripts.record_rollout --config configs/experiments/phase2_oracle.yaml \
      --protocol configs/benchmark/phase2_protocol_v1.yaml \
      --checkpoint outputs/train/phase2/noinfo_s0/checkpoints/last/pretrained_model \
      --list configs/scenario_lists/eval_v1.jsonl --per-task 2 --device cuda
"""
import argparse
import json
from collections import Counter

import torch

from intent_policy.benchmark.runner import ExpertAgent, run_episode, save_record
from intent_policy.benchmark.video import EpisodeVideo
from intent_policy.scenarios.config import _deep_merge, load_scenario_list
from intent_policy.scenarios.scenario_registry import make_scenario, scenario_overrides
from intent_policy.utils import checkpoint_name, controller_config, load_yaml, log_run, observation_config, resolve, run_dir


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/experiments/phase2_oracle.yaml')
    p.add_argument('--protocol', help='versioned evaluation protocol (Phase 2); without it the experiment settings are used')
    p.add_argument('--checkpoint', help='pretrained_model dir; omit with --expert')
    p.add_argument('--expert', action='store_true', help='record the scripted expert (scene / demonstration preview)')
    p.add_argument('--trigger', help='expert trigger override (cue_complete | evidence | cue_onset)')
    p.add_argument('--condition', default='correct', help='oracle condition name from the protocol')
    p.add_argument('--split', default='eval', help='protocol split whose overrides (and scenarios, without --list) are used')
    p.add_argument('--list', help='scenario list (.jsonl); episodes are taken from it')
    p.add_argument('--tasks', nargs='*', help='with --list: tasks to record')
    p.add_argument('--per-task', type=int, default=1, help='with --list: first N entries of each task')
    p.add_argument('--indices', type=int, nargs='*', help='with --list: record these list indices instead')
    p.add_argument('--scenarios', nargs='*')
    p.add_argument('--seeds', type=int, nargs='+')
    p.add_argument('--output-dir', help='default: outputs/videos/<date>/<time>_<experiment>_<agent>')
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    exp = load_yaml(args.config)
    if args.protocol:
        protocol = load_yaml(args.protocol)
        split = next(s for s in protocol['splits'] if s['name'] == args.split)
        condition = protocol['conditions'][args.condition]
        ensemble = protocol.get('inference', {}).get('temporal_ensemble_coeff', 'config')
    else:                                       # as scripts/evaluate.py without a protocol
        split, condition, ensemble = {}, {'condition': 'correct'}, 'config'
    obs_cfg, ctrl_cfg = observation_config(exp), controller_config(exp)
    if args.list:
        per = Counter()
        jobs = []
        for e in load_scenario_list(args.list):
            if args.indices is not None:
                if e['index'] not in args.indices:
                    continue
            elif (args.tasks and e['task'] not in args.tasks) or per[e['task']] >= args.per_task:
                continue
            per[e['task']] += 1
            jobs.append((e['scenario_id'], int(e['seed']), e['spec'], f"{e['task']}_{e['index']:03d}__seed{e['seed']}"))
    else:
        jobs = [(sid, seed, None, f'{sid}__{args.split}__seed{seed}') for sid in args.scenarios or split.get('scenarios', [])
                for seed in args.seeds]
    out = run_dir('videos', f"{exp['experiment']}_{'expert' if args.expert else checkpoint_name(args.checkpoint)}", args.output_dir)
    log_run(out, 'scripts.record_rollout')
    if args.expert:
        agent = ExpertAgent()
    else:
        from intent_policy.policies.policy_agent import PolicyAgent
        agent = PolicyAgent(resolve(args.checkpoint), args.device, condition, sorted({j[1] for j in jobs}), ctrl_cfg, ensemble)
    rows, cache = [], {}
    for scenario, seed, spec, name in jobs:
        if scenario not in cache:
            ov = _deep_merge(scenario_overrides(exp, scenario),
                             scenario_overrides({'scenario_overrides': split.get('scenario_overrides', {})}, scenario))
            if args.trigger:
                ov = _deep_merge(ov, {'expert': {'trigger': args.trigger}})
            cache[scenario] = make_scenario(scenario, ov)
        video = EpisodeVideo(out, name)
        rec = run_episode(cache[scenario], agent, seed, ctrl_cfg, obs_cfg, on_frame=video, keep_trace=True, spec=spec)
        result = 'success' if rec['success'] else rec['failure']
        path = video.finish(rec, out / f'{name}__{result}.mp4')
        save_record(rec, out / f'{name}.json.gz')
        ev = {e['event_type']: e['timestamp'] for e in reversed(rec['events'])}
        rows.append((path.name, result, rec['duration_s'], ev.get('human_intention_evident'), ev.get('robot_target_commit'),
                     json.dumps({k: v for k, v in rec['spec'].items() if k not in ('task', 'layout')})))
        print(f"{name}: {result} T={rec['duration_s']:.1f}s", flush=True)
    for sc in cache.values():
        sc.close()
    lines = [f"# Rollouts — {'scripted expert' if args.expert else f'`{args.checkpoint}`'} ({args.condition}, split {args.split})", '',
             '| video | result | duration s | intention evident s | first robot commit s | spec |', '|---|---|---|---|---|---|']
    fmt = lambda v: '—' if v is None else f'{v:.2f}'
    lines += [f'| [{n}]({n}) | {r} | {d:.1f} | {fmt(e)} | {fmt(c)} | `{sp}` |' for n, r, d, e, c, sp in rows]
    with open(out / 'summary.md', 'a') as f:
        f.write('\n'.join(lines) + '\n\n')
    print(f'run folder: {out}')


if __name__ == '__main__':
    main()

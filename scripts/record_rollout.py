"""Record policy rollouts as videos for visual inspection (front view + the policy's own cameras).

Each frame shows the `high` policy camera (large), the other policy camera images (side panel), the time, the selected
action and the benchmark events logged so far (cue onset, intention evident, target commit, grasp, ...).
Runs the evaluation code path (same agent, controller, observation, protocol inference setting and
scenario overrides). Writes `<scenario>__<split>__seed<N>.mp4` (H.264, plays in any video player), the episode
record next to it, and `summary.md`. Use seeds outside the test/tuning ranges for ad-hoc looks.

Episodes come from a scenario list (`--list`, e.g. configs/scenario_lists/eval_v1.jsonl; `--tasks`, `--per-task`)
or from `--seeds` x the protocol split's scenarios (random specs).

  ./run.sh -m scripts.record_rollout --expert --list configs/scenario_lists/demo_v1.jsonl --per-task 1 \
      --output-dir outputs/videos/expert_preview
  ./run.sh -m scripts.record_rollout --config configs/experiments/phase2_oracle.yaml \
      --checkpoint outputs/train/phase2/noinfo_s0/checkpoints/last/pretrained_model \
      --list configs/scenario_lists/eval_v1.jsonl --per-task 2 --output-dir outputs/videos/noinfo_s0 --device cuda
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image, ImageDraw

from intent_policy.benchmark.runner import ExpertAgent, run_episode, save_record
from intent_policy.sim.restricted_action import RestrictedAction
from intent_policy.scenarios.config import _deep_merge, load_scenario_list
from intent_policy.scenarios.scenario_registry import make_scenario, scenario_overrides
from intent_policy.utils import load_yaml, resolve, observation_config, controller_config

SHOWN_EVENTS = ('human_cue_onset', 'human_intention_evident', 'human_intention_change', 'robot_target_commit',
                'object_grasp', 'object_release', 'disruption_start', 'disruption_end', 'human_robot_contact',
                'safety_distance_violation', 'task_success', 'task_failure')
MAIN_CAMERA = 'high'                       # shown large; the side panel shows the other policy cameras
FRONT_W, FRONT_H, SIDE_W = 640, 480, 256
FPS = 20                                   # one decision per 0.05 s tick, as in the datasets


def event_line(e: dict) -> str:
    p = e.get('payload') or {}
    detail = ' '.join(f'{k}={v}' for k, v in p.items() if k in ('target', 'object', 'region', 'kind', 'new', 'reason'))
    return f"{e['timestamp']:5.2f}s {e['event_type']} {detail}".rstrip()


class VideoWriter:
    def __init__(self, path: Path):
        self.out = av.open(str(path), 'w', options={'movflags': 'faststart'})
        self.stream = self.out.add_stream('libx264', rate=FPS)
        self.stream.width, self.stream.height, self.stream.pix_fmt = FRONT_W + SIDE_W, FRONT_H, 'yuv420p'
        self.stream.options = {'crf': '23', 'preset': 'veryfast'}

    def add(self, frame: np.ndarray) -> None:
        for packet in self.stream.encode(av.VideoFrame.from_ndarray(frame, format='rgb24')):
            self.out.mux(packet)

    def close(self) -> None:
        for packet in self.stream.encode():
            self.out.mux(packet)
        self.out.close()


def compose(front: np.ndarray, obs: dict, header: str, events: list[str], banner: str | None = None) -> np.ndarray:
    canvas = Image.new('RGB', (FRONT_W + SIDE_W, FRONT_H), (20, 20, 20))
    canvas.paste(Image.fromarray(front), (0, 0))
    y = 0
    side = sorted(k for k in obs if k.startswith('observation.images.') and k != f'observation.images.{MAIN_CAMERA}')
    for key in side:
        img = obs[key]
        if y + SIDE_W * 3 // 4 <= FRONT_H:
            img = Image.fromarray(np.asarray(img)).resize((SIDE_W, SIDE_W * 3 // 4))
            canvas.paste(img, (FRONT_W, y))
            y += img.height
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, FRONT_W, 18], fill=(0, 0, 0))
    d.text((4, 3), header, fill=(255, 255, 255))
    for i, line in enumerate(events[-6:]):
        d.text((4, FRONT_H - 16 * (min(len(events), 6) - i) - 2), line, fill=(255, 230, 120))
    d.text((FRONT_W + 4, y + 4), f'policy inputs: {MAIN_CAMERA} (left) + ' + ' / '.join(k.split('.')[-1] for k in side),
           fill=(180, 180, 180))
    if banner:
        ok = banner.startswith('SUCCESS')
        d.rectangle([0, FRONT_H // 2 - 20, FRONT_W, FRONT_H // 2 + 20], fill=(20, 110, 40) if ok else (140, 30, 30))
        d.text((12, FRONT_H // 2 - 6), banner, fill=(255, 255, 255))
    return np.asarray(canvas)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/experiments/phase2_oracle.yaml')
    p.add_argument('--protocol', default='configs/benchmark/phase2_protocol_v1.yaml')
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
    p.add_argument('--output-dir', required=True)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    exp, protocol = load_yaml(args.config), load_yaml(args.protocol)
    split = next(s for s in protocol['splits'] if s['name'] == args.split)
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
            jobs.append((e['scenario_id'], int(e['seed']), e['spec'], f"{e['task']}_{e['index']:03d}__{e['scenario_id']}__seed{e['seed']}"))
    else:
        jobs = [(sid, seed, None, f'{sid}__{args.split}__seed{seed}') for sid in args.scenarios or split.get('scenarios', [])
                for seed in args.seeds]
    if args.expert:
        agent = ExpertAgent()
    else:
        from intent_policy.policies.policy_agent import PolicyAgent
        agent = PolicyAgent(resolve(args.checkpoint), args.device, protocol['conditions'][args.condition],
                            sorted({j[1] for j in jobs}), ctrl_cfg,
                            protocol.get('inference', {}).get('temporal_ensemble_coeff', 'config'))
    out = resolve(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, cache = [], {}
    for scenario, seed, spec, name in jobs:
        if scenario not in cache:
            ov = _deep_merge(scenario_overrides(exp, scenario),
                             scenario_overrides({'scenario_overrides': split.get('scenario_overrides', {})}, scenario))
            if args.trigger:
                ov = _deep_merge(ov, {'expert': {'trigger': args.trigger}})
            cache[scenario] = make_scenario(scenario, ov)
        sc = cache[scenario]
        writer, last = VideoWriter(out / f'{name}.mp4'), {}

        def on_frame(obs, decision, target, s):
            events = [event_line(e) for e in s.get_events() if e['event_type'] in SHOWN_EVENTS]
            header = f'{name}  t={s.time:5.2f}s  action={RestrictedAction(decision.action_id).name}'
            last.update(obs=obs, events=events, header=header)
            writer.add(compose(s.env.render(MAIN_CAMERA, FRONT_W, FRONT_H), obs, header, events))

        rec = run_episode(sc, agent, seed, ctrl_cfg, obs_cfg, on_frame=on_frame, keep_trace=True, spec=spec)
        banner = 'SUCCESS' if rec['success'] else f"FAILURE: {rec['failure']}"
        final = compose(sc.env.render(MAIN_CAMERA, FRONT_W, FRONT_H), last.get('obs', {}), last.get('header', ''),
                        [event_line(e) for e in rec['events'] if e['event_type'] in SHOWN_EVENTS], banner)
        for _ in range(FPS * 2):
            writer.add(final)
        writer.close()
        save_record(rec, out / f'{name}.json.gz')
        ev = {e['event_type']: e['timestamp'] for e in reversed(rec['events'])}
        rows.append((name, banner, rec['duration_s'], ev.get('human_intention_evident'), ev.get('robot_target_commit'),
                     json.dumps({k: v for k, v in rec['spec'].items() if k not in ('task', 'layout')})))
        print(f"{name}: {banner} T={rec['duration_s']:.1f}s", flush=True)
    for sc in cache.values():
        sc.close()
    lines = [f"# Rollouts — {'scripted expert' if args.expert else f'`{args.checkpoint}`'} ({args.condition}, split {args.split})", '',
             '| episode | result | duration s | intention evident s | first robot commit s | spec |', '|---|---|---|---|---|---|']
    fmt = lambda v: '—' if v is None else f'{v:.2f}'
    lines += [f'| {n} | {b} | {d:.1f} | {fmt(e)} | {fmt(c)} | `{sp}` |' for n, b, d, e, c, sp in rows]
    with open(out / 'summary.md', 'a') as f:
        f.write('\n'.join(lines) + '\n\n')


if __name__ == '__main__':
    main()

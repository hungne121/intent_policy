"""Record policy rollouts as videos for visual inspection (front view + the policy's own cameras).

Each frame shows the front camera, the scene/wrist images the policy receives, the time, the selected
action and the benchmark events logged so far (cue onset, intention evident, target commit, grasp, ...).
Runs the evaluation code path (same agent, controller, observation, protocol inference setting and
scenario overrides). Writes `<scenario>__<split>__seed<N>.mp4` (H.264, plays in any video player), the episode
record next to it, and `summary.md`. Use seeds outside the test/tuning ranges for ad-hoc looks.

  ./run.sh -m scripts.record_rollout --config configs/experiments/phase2_oracle.yaml \
      --protocol configs/benchmark/phase2_protocol_v1.yaml \
      --checkpoint outputs/train/phase2/noinfo_s0/checkpoints/last/pretrained_model \
      --split nominal --seeds 300000 300001 --output-dir outputs/videos/noinfo_s0 --device cuda
"""
import argparse
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image, ImageDraw

from intent_policy.benchmark.runner import run_episode, save_record
from intent_policy.sim.restricted_action import RestrictedAction
from intent_policy.scenarios.config import _deep_merge
from intent_policy.scenarios.scenario_registry import make_scenario, scenario_overrides
from intent_policy.utils import load_yaml, resolve, observation_config, controller_config

SHOWN_EVENTS = ('human_cue_onset', 'human_intention_evident', 'human_intention_change', 'robot_target_commit',
                'object_grasp', 'object_release', 'disruption_start', 'disruption_end', 'human_robot_contact',
                'safety_distance_violation', 'task_success', 'task_failure')
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
    for cam in ('scene', 'wrist'):
        img = obs.get(f'observation.images.{cam}')
        if img is not None:
            img = Image.fromarray(np.asarray(img)).resize((SIDE_W, SIDE_W * 3 // 4))
            canvas.paste(img, (FRONT_W, y))
            y += img.height
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, FRONT_W, 18], fill=(0, 0, 0))
    d.text((4, 3), header, fill=(255, 255, 255))
    for i, line in enumerate(events[-6:]):
        d.text((4, FRONT_H - 16 * (min(len(events), 6) - i) - 2), line, fill=(255, 230, 120))
    d.text((FRONT_W + 4, y + 4), 'policy inputs: scene / wrist', fill=(180, 180, 180))
    if banner:
        ok = banner.startswith('SUCCESS')
        d.rectangle([0, FRONT_H // 2 - 20, FRONT_W, FRONT_H // 2 + 20], fill=(20, 110, 40) if ok else (140, 30, 30))
        d.text((12, FRONT_H // 2 - 6), banner, fill=(255, 255, 255))
    return np.asarray(canvas)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/experiments/phase2_oracle.yaml')
    p.add_argument('--protocol', default='configs/benchmark/phase2_protocol_v1.yaml')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--condition', default='correct', help='oracle condition name from the protocol')
    p.add_argument('--split', default='nominal', help='protocol split whose scenarios / overrides are used')
    p.add_argument('--scenarios', nargs='*')
    p.add_argument('--seeds', type=int, nargs='+', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    exp, protocol = load_yaml(args.config), load_yaml(args.protocol)
    split = next(s for s in protocol['splits'] if s['name'] == args.split)
    obs_cfg, ctrl_cfg = observation_config(exp), controller_config(exp)
    from intent_policy.policies.policy_agent import PolicyAgent
    agent = PolicyAgent(resolve(args.checkpoint), args.device, protocol['conditions'][args.condition], args.seeds,
                        ctrl_cfg, protocol.get('inference', {}).get('temporal_ensemble_coeff', 'config'))
    out = resolve(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for scenario in args.scenarios or split['scenarios']:
        ov = _deep_merge(scenario_overrides(exp, scenario),
                         scenario_overrides({'scenario_overrides': split.get('scenario_overrides', {})}, scenario))
        sc = make_scenario(scenario, ov)
        try:
            for seed in args.seeds:
                name = f'{scenario}__{args.split}__seed{seed}'
                writer, last = VideoWriter(out / f'{name}.mp4'), {}

                def on_frame(obs, decision, target, s):
                    events = [event_line(e) for e in s.get_events() if e['event_type'] in SHOWN_EVENTS]
                    header = f'{scenario}  seed {seed}  t={s.time:5.2f}s  action={RestrictedAction(decision.action_id).name}'
                    last.update(obs=obs, events=events, header=header)
                    writer.add(compose(s.env.render('front', FRONT_W, FRONT_H), obs, header, events))

                rec = run_episode(sc, agent, seed, ctrl_cfg, obs_cfg, on_frame=on_frame, keep_trace=True)
                banner = 'SUCCESS' if rec['success'] else f"FAILURE: {rec['failure']}"
                final = compose(sc.env.render('front', FRONT_W, FRONT_H), last.get('obs', {}), last.get('header', ''),
                                [event_line(e) for e in rec['events'] if e['event_type'] in SHOWN_EVENTS], banner)
                for _ in range(FPS * 2):
                    writer.add(final)
                writer.close()
                save_record(rec, out / f'{name}.json.gz')
                ev = {e['event_type']: e['timestamp'] for e in reversed(rec['events'])}
                rows.append((name, banner, rec['duration_s'], ev.get('human_intention_evident'), ev.get('robot_target_commit')))
                print(f"{name}: {banner} T={rec['duration_s']:.1f}s", flush=True)
        finally:
            sc.close()
    lines = [f'# Rollouts — `{args.checkpoint}` ({args.condition}, split {args.split})', '',
             '| episode | result | duration s | intention evident s | first robot commit s |', '|---|---|---|---|---|']
    fmt = lambda v: '—' if v is None else f'{v:.2f}'
    lines += [f'| {n} | {b} | {d:.1f} | {fmt(e)} | {fmt(c)} |' for n, b, d, e, c in rows]
    with open(out / 'summary.md', 'a') as f:
        f.write('\n'.join(lines) + '\n\n')


if __name__ == '__main__':
    main()

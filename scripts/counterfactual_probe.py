"""Fixed-observation counterfactual probe (Phase 2 acceptance item).

For held-out (validation) demonstration frames in the decision window (from the evidence point on),
the camera images and robot state are kept fixed and only the oracle input is changed:
  A            true oracle (target A, true future hand motion)
  B            target B (the other candidate, with its true position) AND reversed motion
  B_spatial    only the target swapped            (branch diagnostic)
  A_reversed   only the motion reversed           (branch diagnostic)
The restricted action distribution of the first chunk step is logged for every variant. Reported:
argmax change rate and total-variation distance A vs B, and whether the chosen move heads for the
target it was given (target-consistent direction). Intruder: P(HOLD) with approaching versus
withdrawing hand. A masked No-Information checkpoint is the control (identical outputs by construction).

  ./run.sh -m scripts.counterfactual_probe --checkpoint outputs/train/phase2/oracle_s0/checkpoints/last/pretrained_model \
      --output outputs/eval/phase2/counterfactual/oracle_s0
"""
import argparse
import json
from collections import defaultdict
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from controllers.restricted_action import RestrictedAction as A
from intent.representation import PREFIX, features, record_from_features
from policies.policy_agent import load_policy
from scripts.common import resolve

STEP = {int(A.MOVE_FORWARD): (0, 1), int(A.MOVE_BACKWARD): (0, -1), int(A.MOVE_LEFT): (1, 1),
        int(A.MOVE_RIGHT): (1, -1), int(A.MOVE_UP): (2, 1), int(A.MOVE_DOWN): (2, -1)}
TABLE_Z = 0.62


def alternative(meta: dict, rec, key: str):
    """The other candidate of the target's group and its (static, pre-grasp) world position."""
    v = meta['variation']
    for group, xy in (('object_xy', v.get('object_xy', {})), ('bowl_xy', v.get('bowl_xy', {})), ('region_xy', v.get('region_xy', {}))):
        if key in xy:
            others = [k for k in xy if k != key]
            alt = others[0]
            z = rec.object_pos[2] if rec.object_pos is not None else TABLE_Z
            return alt, np.array([*xy[alt], z if group != 'region_xy' else TABLE_Z])
    return None, None


def reversed_motion(rec):
    r = rec.copy()
    for h in r.future_pos:
        r.future_pos[h] = r.hand_pos - (r.future_pos[h] - r.hand_pos)
        r.future_vel[h] = -r.future_vel[h]
    r.hand_vel = -r.hand_vel
    return r


def consistent(action: int, tcp: np.ndarray, target: np.ndarray) -> bool | None:
    if action not in STEP:
        return None
    axis, sign = STEP[action]
    return bool(sign * (target - tcp)[axis] > 0)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--root', default='outputs/datasets/hri_phase2_oracle')
    p.add_argument('--repo-id')
    p.add_argument('--output', required=True)
    p.add_argument('--val-every', type=int, default=10)
    p.add_argument('--window-frames', type=int, default=20)
    p.add_argument('--stride', type=int, default=2)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    root = resolve(args.root)
    repo_id = args.repo_id or f'local/{root.name}'
    policy, pre, _ = load_policy(resolve(args.checkpoint), args.device)
    policy.eval()
    cfg = policy.config
    metas = [json.loads(line) for line in (root / 'meta/hri_episodes.jsonl').read_text().splitlines()]
    ds = LeRobotDataset(repo_id, root=root)
    start = {int(e['episode_index']): int(e['dataset_from_index']) for e in ds.meta.episodes}
    out = resolve(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for m in metas:
        if m['episode_index'] % args.val_every != args.val_every - 1:
            continue
        ev = [e for e in m.get('intention_evident', []) if e['kind'] in ('instruction', 'target_object', 'intrusion')]
        if not ev:
            continue
        f0 = ev[0]['frame']
        for f in range(f0, min(f0 + args.window_frames, m['n_frames']), args.stride):
            item = ds[start[m['episode_index']] + f]
            tcp = item['observation.state'][7:10].numpy().astype(float)
            stored = {k: item[k].numpy() for k in item if k.startswith(PREFIX)}
            rec = record_from_features(stored)
            intruder = m['scenario_id'] == 'intruder_pick_place_interruption'
            key = rec.region_key if intruder else rec.object_key
            if key is None:
                continue
            variants = {'A': rec, 'A_reversed': reversed_motion(rec)}
            if not intruder:
                alt, alt_pos = alternative(m, rec, key)
                if alt is None:
                    continue
                b = rec.copy()
                b.object_key, b.object_pos = alt, alt_pos
                variants['B_spatial'] = b
                variants['B'] = reversed_motion(b)
            else:
                b = reversed_motion(rec)
                b.region_pos = rec.region_pos + np.array([0.0, 0.2 if rec.region_pos[1] < 0 else -0.2, 0.0])
                variants['B'] = b
            obs = {k: item[k].unsqueeze(0) for k in cfg.input_features if not k.startswith(PREFIX)}
            probs = {}
            for name, r in variants.items():
                batch = dict(obs)
                batch.update({k: torch.from_numpy(v).unsqueeze(0) for k, v in features(r, tcp).items() if k in cfg.input_features})
                with torch.no_grad():
                    logits = policy.predict_restricted_chunk(pre({k: v.to(args.device) for k, v in batch.items()}))[0, 0]
                probs[name] = logits.softmax(-1).float().cpu().numpy()
            row = dict(episode=m['episode_index'], scenario=m['scenario_id'], split=m['split'], frame=f,
                       frames_after_evidence=f - f0, probs={k: v.round(5).tolist() for k, v in probs.items()})
            pa, pb = probs['A'], probs['B']
            row['tv_AB'] = float(0.5 * np.abs(pa - pb).sum())
            row['argmax_change_AB'] = int(pa.argmax() != pb.argmax())
            for name in probs:
                row[f'tv_A_{name}'] = float(0.5 * np.abs(pa - probs[name]).sum())
            if intruder:
                row['p_hold_A'], row['p_hold_B'] = float(pa[int(A.HOLD)]), float(pb[int(A.HOLD)])
            else:
                row['consistent_A'] = consistent(int(pa.argmax()), tcp, rec.object_pos)
                row['consistent_B'] = consistent(int(pb.argmax()), tcp, variants['B'].object_pos)
            rows.append(row)
    with open(out / 'probe.jsonl', 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    summary = defaultdict(dict)
    for sid in dict.fromkeys(r['scenario'] for r in rows):
        rs = [r for r in rows if r['scenario'] == sid]
        s = dict(frames=len(rs), argmax_change_AB=float(np.mean([r['argmax_change_AB'] for r in rs])),
                 tv_AB=float(np.mean([r['tv_AB'] for r in rs])),
                 tv_spatial_only=float(np.mean([r.get('tv_A_B_spatial', np.nan) for r in rs])),
                 tv_motion_only=float(np.mean([r['tv_A_A_reversed'] for r in rs])))
        if 'p_hold_A' in rs[0]:
            s.update(p_hold_approaching=float(np.mean([r['p_hold_A'] for r in rs])),
                     p_hold_withdrawing=float(np.mean([r['p_hold_B'] for r in rs])))
        else:
            for v in ('A', 'B'):
                c = [r[f'consistent_{v}'] for r in rs if r[f'consistent_{v}'] is not None]
                s[f'target_consistent_move_{v}'] = float(np.mean(c)) if c else None
                s[f'move_fraction_{v}'] = len(c) / len(rs)
        summary[sid] = s
    meta = dict(checkpoint=str(args.checkpoint), intent_mask=cfg.intent_mask, use_intent=cfg.use_intent, summary=summary)
    (out / 'summary.json').write_text(json.dumps(meta, indent=2))
    lines = [f"# Counterfactual probe — `{args.checkpoint}`", '', f"intent_mask={cfg.intent_mask}", '',
             '| scenario | frames | argmax change A↔B | TV(A,B) | TV spatial-only | TV motion-only | target-consistent move A | B | P(HOLD) approaching | withdrawing |',
             '|---|---|---|---|---|---|---|---|---|---|']
    g = lambda d, k: '—' if d.get(k) is None or (isinstance(d.get(k), float) and np.isnan(d[k])) else f'{d[k]:.3f}'
    for sid, s in summary.items():
        lines.append(f"| {sid} | {s['frames']} | {g(s, 'argmax_change_AB')} | {g(s, 'tv_AB')} | {g(s, 'tv_spatial_only')} | "
                     f"{g(s, 'tv_motion_only')} | {g(s, 'target_consistent_move_A')} | {g(s, 'target_consistent_move_B')} | "
                     f"{g(s, 'p_hold_approaching')} | {g(s, 'p_hold_withdrawing')} |")
    (out / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()

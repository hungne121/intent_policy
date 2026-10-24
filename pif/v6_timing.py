"""M5 step 1: how early is the human's target observable in the v6 scenarios (configs/pif/pif.yaml v6_timing)?

Runs v6 episodes (scripted expert, cue_complete) and records at 10 Hz the mannequin keypoints (plus perception noise,
`keypoint_noise_m`), the candidates of each
set and every listed gesture (motion start, end, target). Per set, the PIF v3 filter (pif/filter.py) is fitted on all
its episodes without labels and run causally; per gesture it reports when the true target's belief first reaches each
threshold (s after the motion start, % of the motion, s before its end) and how often another candidate got there
first. Second observer: ManiCast (pif/forecast.py) forecasts the wrist 1 s ahead; its choice is the candidate nearest
the forecast, once unchanged for `manicast_stable_frames`. Late reference: the candidate nearest the hand at the end.
  ./run.sh -m pif.v6_timing  -> outputs/v6_timing/<date_time>/{results.json, report.md}
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict

import numpy as np
import torch

from pif.config import load_config
from pif.filter import batch, belief, fit
from pif.forecast import ManiCastForecaster
from pif.utils import log_run, run_dir

SKIP = 2                                   # 20 Hz control ticks -> 10 Hz frames


def candidates(sc, kind: str) -> list[str]:
    if kind == 'layout':
        return [k for k in sc.variation['spec']['layout'] if k in sc.on_table]
    return [k for k, z in sc.zones.items() if z['kind'] == kind]


def cand_pos(sc, keys: list[str]) -> np.ndarray:
    """Objects where they stood at the start; zones at their nominal centre (never the episode's hidden offset)."""
    from intent_policy.scenarios.base_scenario import BaseScenario
    return np.asarray([sc.start_pos[k] if k in sc.body_id else BaseScenario.zone_pos(sc, k) for k in keys], float)


def record(sid: str, seed: int, sets: dict, kps: list[str], overrides: dict | None = None) -> dict:
    from intent_policy.benchmark.runner import ExpertAgent
    from intent_policy.sim.restricted_action import RestrictedActionConfig, RestrictedActionMapper
    from intent_policy.scenarios.scenario_registry import make_scenario
    sc = record.cache.get(sid) or record.cache.setdefault(sid, make_scenario(sid, overrides or None))
    sc.reset(seed)
    mapper = RestrictedActionMapper(sc.env, RestrictedActionConfig.load())
    ag = ExpertAgent()
    ag.reset(sc, mapper)
    my = {n: s for n, s in sets.items() if s['scenario'] == sid}
    keys = {n: candidates(sc, s['kind']) for n, s in my.items()}
    pos = {n: cand_pos(sc, keys[n]) for n in my}
    reach = {n: np.array([np.linalg.norm(sc.body_model.clamp(q) - q) < 0.01 for q in pos[n]]) for n in my}  # the palm can
    #                                                                       be put there (lean <= max_lean_deg)
    frames, tick = [], 0
    while not sc.done:
        if tick % SKIP == 0:
            frames.append(dict(t=sc.time, kp=sc.human_keypoints(kps), holding=sc.human.holding is not None,
                               pointing=sc.human.hand_shape == 'point'))     # the rendered hand shape (observable)
        sc.step(mapper.map(ag.act(None).action_id).joint_target)
        tick += 1
    gestures = []
    for e in sc.get_events():
        p = e.get('payload') or {}
        if e['event_type'] != 'human_motion_start':
            continue
        for n, s in my.items():
            if p.get('label') in s['labels'] and p.get('target_key') in keys[n]:
                gestures.append(dict(set=n, label=p['label'], target=keys[n].index(p['target_key']),
                                     t0=e['timestamp'], t1=e['timestamp'] + p['duration']))
    T = len(frames)
    ep = dict(human_pos=np.stack([f['kp'] for f in frames]), human_vis=np.ones((T, len(kps)), bool),
              holding=np.array([f['holding'] for f in frames]), t=np.array([f['t'] for f in frames]),
              pointing=np.array([f['pointing'] for f in frames]))
    return dict(sid=sid, seed=seed, ep=ep, keys=keys, pos=pos, reach=reach, gestures=gestures, success=sc.success)


record.cache = {}


def first(cond: np.ndarray) -> int | None:
    i = np.flatnonzero(cond)
    return int(i[0]) if len(i) else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='configs/pif/pif.yaml')
    ap.add_argument('--set', nargs='*', default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    vt, kps = cfg['v6_timing'], cfg['tokens']['keypoints']
    out = run_dir('v6_timing', 'm5')
    log_run(out, 'pif.v6_timing')
    torch.set_num_threads(4)
    recs = []
    for sid, n in vt['episodes'].items():
        for i in range(n):
            recs.append(record(sid, vt['seed0'] + i, vt['sets'], kps, vt.get('scenario_overrides')))
            print(sid, i, len(recs[-1]['gestures']), 'gestures', flush=True)
    rng = np.random.default_rng(0)
    for r in recs:                                  # perception noise (both observers see the same noisy keypoints)
        r['ep']['human_pos'] = r['ep']['human_pos'] + rng.normal(0, vt['keypoint_noise_m'], r['ep']['human_pos'].shape)
    fc = ManiCastForecaster(cfg)
    J = {k: i for i, k in enumerate(kps)}
    need = ['l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist']
    fps = float(cfg['fps'])
    t_hist, t_out = np.arange(-4, 1) / fps, np.array([1.0])
    rows = []
    for name, s in vt['sets'].items():
        rs = [r for r in recs if r['sid'] == s['scenario'] and any(g['set'] == name for g in r['gestures'])]
        if not rs:
            continue
        eps = []
        for r in rs:
            e = dict(r['ep'])
            e['obj_pos'] = np.repeat(r['pos'][name][None], len(e['holding']), 0)
            eps.append(e)
        x = batch(eps, [list(range(len(r['keys'][name]))) for r in rs], kps)
        p, _ = fit(x, cfg['v3']['init'], **cfg['v3']['fit'])
        b = belief(x, p)
        for i, r in enumerate(rs):
            e, K, tt = r['ep'], len(r['keys'][name]), r['ep']['t']
            # ManiCast: wrist 1 s ahead at every frame with 0.4 s of history
            kp = {k: e['human_pos'][:, J[k]] for k in need}
            mk, waist = fc.to_comad(kp)
            pred = np.full((len(tt), 3), np.nan)
            idx = np.arange(4, len(tt))
            H = np.stack([mk[j - 4:j + 1] for j in idx]); W = np.stack([waist[j - 4:j + 1] for j in idx])
            f = np.concatenate([fc.forecast(H[a:a + 256], W[a:a + 256], t_hist, t_out) for a in range(0, len(H), 256)])
            pred[idx] = fc.from_comad(f[:, 0, 6])                         # RWristOut, back to the world frame
            near = np.argmin(np.linalg.norm(pred[:, None] - r['pos'][name][None], axis=2), axis=1)
            for g in r['gestures']:
                if g['set'] != name:
                    continue
                a, z = int(np.searchsorted(tt, g['t0'])), int(np.searchsorted(tt, g['t1']))
                z = min(z, len(tt) - 1)
                row = dict(set=name, label=g['label'], seed=r['seed'], K=K, dur=round(g['t1'] - g['t0'], 2))
                bel = b[i, a:z + 1, 1:K + 1]
                for th in vt['thresholds']:
                    k_ok = first(bel[:, g['target']] >= th)
                    k_any = first((bel >= th).any(1))
                    row[f'b{th}_s'] = None if k_ok is None else round(tt[a + k_ok] - g['t0'], 2)
                    row[f'b{th}_wrong_first'] = bool(k_any is not None and (k_ok is None or k_any < k_ok)
                                                     and int(np.argmax(bel[k_any])) != g['target'])
                row['end_correct'] = bool(np.argmax(b[i, z, 1:K + 1]) == g['target'])
                st, run, k_mc = None, 0, None
                for j in range(a, z + 1):                                  # stable ManiCast choice
                    if np.isnan(pred[j]).any():
                        continue
                    run = run + 1 if st == near[j] else 1
                    st = near[j]
                    if run >= vt['manicast_stable_frames']:
                        k_mc = j
                        break
                row['mc_s'] = None if k_mc is None else round(tt[k_mc] - g['t0'], 2)
                row['mc_correct'] = None if k_mc is None else bool(near[k_mc] == g['target'])
                wr = e['human_pos'][z, J['r_wrist']]
                row['late_correct'] = bool(np.argmin(np.linalg.norm(r['pos'][name] - wr, axis=1)) == g['target'])
                rows.append(row)
    (out / 'results.json').write_text(json.dumps(rows, indent=1))
    lines = ['# v6 timing (M5 step 1)', '', '| set | n | K | gesture s | ' +
             ' | '.join(f'belief>={th}: s after start (median) / reached / wrong first' for th in vt['thresholds']) +
             ' | belief argmax correct at end | ManiCast: s / correct | nearest at end correct |', '|' + '---|' * (6 + len(vt['thresholds']))]
    by = defaultdict(list)
    for r in rows:
        by[r['set']].append(r)
    for name, rs in by.items():
        cells = []
        for th in vt['thresholds']:
            v = [r[f'b{th}_s'] for r in rs if r[f'b{th}_s'] is not None]
            cells.append(f"{np.median(v) if v else float('nan'):.2f} / {len(v)}/{len(rs)} / {sum(r[f'b{th}_wrong_first'] for r in rs)}")
        mc = [r for r in rs if r['mc_s'] is not None]
        lines.append(f"| {name} | {len(rs)} | {rs[0]['K']} | {np.median([r['dur'] for r in rs]):.2f} | " + ' | '.join(cells) +
                     f" | {np.mean([r['end_correct'] for r in rs]):.2f} | {np.median([r['mc_s'] for r in mc]) if mc else float('nan'):.2f} / "
                     f"{np.mean([r['mc_correct'] for r in mc]) if mc else float('nan'):.2f} | {np.mean([r['late_correct'] for r in rs]):.2f} |")
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'-> {out}')


if __name__ == '__main__':
    main()

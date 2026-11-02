"""Causal ManiCast/CUR/CVM diagnostics on scripted motions, not policy evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from intent_policy.intent.manicast_intent import MANICAST, forecast_xi, load_forecaster
from intent_policy.sim.base_env import ROOT
from intent_policy.sim.human_body import HumanBody
from scripts.preview_scenarios_v6 import Track


def sample_motion(cfg, task, fps):
    scene = yaml.safe_load((ROOT / cfg['scene_base']).read_text())
    body = HumanBody({**scene['human'], 'floor_z': scene.get('floor_z', 0.0)})
    rest = np.asarray(scene['human']['hand_rest'], float)
    idle = body.pose(rest)['idle_palm']
    h = Track(task.get('human', []), dict(to=rest, shape='relaxed', palm_up=0., point_at=None))
    hl = Track(task.get('human_l', []), dict(to=idle, shape='relaxed', palm_up=0.))
    objects = {k: np.asarray(o['pos'], float) for k, o in task['objects'].items()}

    def resolve(v):
        if isinstance(v, dict):
            return objects[v['obj']] + np.asarray(v.get('d', [0, 0, 0]))
        return None if v is None else np.asarray(v, float)

    ts = np.arange(int(round(task['duration'] * fps)) + 1) / fps
    kp = []
    for t in ts:
        a, b = h.at(float(t), resolve), hl.at(float(t), resolve)
        pose = body.pose(a['to'], a.get('point_at'), float(a.get('palm_up') or 0.),
                         float(a.get('progress', 1.)), b['to'] if task.get('human_l') else None,
                         float(b.get('palm_up') or 0.))
        kp.append(body.keypoints(pose, MANICAST))
    return ts, np.asarray(kp, np.float32)


def audit_task(cfg, task, fc, out, key):
    fps = cfg['fps']
    ts, kp = sample_motion(cfg, task, fps)
    horizons = (np.arange(8) + 1) / 8
    xi = forecast_xi(kp, fc, fps, horizons)
    predicted = kp[:, None] + xi
    if not np.isfinite(predicted).all():
        raise ValueError(f'{key}: non-finite model forecast')
    actual = np.stack([np.stack([np.interp(ts + dt, ts, kp[:, j, d])
                                for j in range(6) for d in range(3)], -1).reshape(len(ts), 6, 3)
                       for dt in horizons], 1)
    valid = ts[:, None] + horizons[None] <= ts[-1]
    history = max(1, round(.4 * fps))
    velocity = (kp - kp[np.maximum(0, np.arange(len(kp)) - history)]) / \
        np.maximum(1 / fps, np.minimum(np.arange(len(kp)), history) / fps)[:, None, None]
    cvm = kp[:, None] + velocity[:, None] * horizons[None, :, None, None]
    cur = np.broadcast_to(kp[:, None], predicted.shape)
    errors = {}
    for name, p in [('manicast', predicted), ('current_pose', cur), ('constant_velocity', cvm)]:
        e = np.linalg.norm(p - actual, axis=-1)
        wrist = e[:, :, [4, 5]].mean(-1)
        errors[name] = dict(wrist_ade_m=float(wrist[valid].mean()),
                            wrist_fde_1s_m=float(wrist[valid[:, -1], -1].mean()))
    out.mkdir(parents=True, exist_ok=True)
    display = predicted.copy()
    intervention_active = np.zeros(len(ts), bool)
    intervention = task.get('forecast_intervention')
    if intervention:
        intervention_active = (ts >= intervention['t'][0]) & (ts < intervention['t'][1])
        for joint in intervention['joints']:
            if 'offset' in intervention:
                display[intervention_active, :, joint] += np.asarray(intervention['offset'])
            if 'target_x' in intervention:
                display[intervention_active, :, joint, 0] = intervention['target_x']
    np.savez_compressed(out / f'{key}.npz', time_s=ts, keypoints=kp, horizons_s=horizons,
                        predicted=predicted, actual_future=actual, valid=valid, constant_velocity=cvm,
                        display_predicted=display, intervention_active=intervention_active)
    result = dict(clip=key, checkpoint='outputs/forecast_ft/2026-10-24_23-44/best.pt',
                  causal_history_s=.4, forecast_horizon_s=1., evaluation_only=True,
                  errors=errors, scope='scripted human kinematics; no ACT rollout or physics')
    if intervention:
        result['controlled_intervention'] = intervention
        result['not_a_natural_model_error'] = True
    if task.get('forecast_witness'):
        w = task['forecast_witness']
        i = int(np.argmin(abs(ts - w['t_s'])))
        j, h = w['joint'], w['horizon_index']
        result['natural_error_witness'] = dict(t_s=float(ts[i]), joint=MANICAST[j], horizon_s=float(horizons[h]),
            predicted=predicted[i, h, j].tolist(), actual_future=actual[i, h, j].tolist(),
            current=kp[i, j].tolist(), error_m=float(np.linalg.norm(predicted[i, h, j] - actual[i, h, j])))
        if w['kind'] == 'wrong_side':
            side = lambda x: 'left' if x < w['split_x'] else 'right'
            result['natural_error_witness'].update(predicted_side=side(predicted[i, h, j, 0]),
                                                    actual_side=side(actual[i, h, j, 0]), split_x=w['split_x'])
            assert side(predicted[i, h, j, 0]) != side(actual[i, h, j, 0]), key
    (out / f'{key}.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--tasks', nargs='*')
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    torch.set_num_threads(2)
    fc = load_forecaster()
    results = []
    for key in args.tasks or cfg['tasks']:
        result = audit_task(cfg, cfg['tasks'][key], fc, ROOT / args.output_dir, key)
        results.append(result)
        print(key, json.dumps(result['errors'], ensure_ascii=False), flush=True)
    pairs = {}
    for key, task in cfg['tasks'].items():
        if task.get('ambiguity_pair') and (ROOT / args.output_dir / f'{key}.npz').exists():
            pairs.setdefault(task['ambiguity_pair']['group'], []).append(key)
    prefix_checks = {}
    for group, keys in pairs.items():
        if len(keys) != 2:
            continue
        a, b = [np.load(ROOT / args.output_dir / f'{k}.npz') for k in keys]
        branch = cfg['tasks'][keys[0]]['ambiguity_pair']['branch_t']
        n = int(np.searchsorted(a['time_s'], branch + 1e-8, side='right'))
        assert np.allclose(a['time_s'][:n], b['time_s'][:n]), group
        delta = float(np.max(np.abs(a['keypoints'][:n] - b['keypoints'][:n])))
        assert delta < 1e-6, (group, delta)
        pred_delta = float(np.max(np.abs(a['predicted'][:n] - b['predicted'][:n])))
        assert pred_delta < 1e-6, (group, pred_delta)
        prefix_checks[group] = dict(clips=keys, branch_t=branch, max_keypoint_difference_m=delta,
                                    max_forecast_difference_m=pred_delta)
    (ROOT / args.output_dir / 'prefix_checks.json').write_text(json.dumps(prefix_checks, indent=2) + '\n')
    (ROOT / args.output_dir / 'summary.json').write_text(json.dumps(results, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()

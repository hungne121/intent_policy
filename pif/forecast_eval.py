"""Evaluate ManiCast (pif/forecast.py) on its own data (CoMaD) and on other real humans (HABIT tokens).

  python -m pif.forecast_eval comad      CoMaD test split of each task, their loader, base / manicast / manicastW
                                         checkpoints vs current pose and constant velocity (their baselines)
  python -m pif.forecast_eval habit      HABIT S8789 tokens (video keypoints, table frame, 10 Hz): manicast vs current
                                         pose and constant velocity; frames whose 0.4 s history and 1 s future have
                                         valid shoulders / elbows / wrists passing the arm-length gate
Metrics (mm): mean displacement over the 1 s horizon (ADE), at about 0.5 s and at 1 s, wrists and elbows; also on
the 'moving' subset (a wrist moves > forecast.eval.moving_m within the horizon). Writes outputs/forecast_eval/...
"""
from __future__ import annotations
import argparse
import json
import sys

import numpy as np
import torch

from pif.config import load_config
from pif.forecast import ManiCastForecaster
from pif.utils import log_run, resolve, run_dir


def summarise(err: np.ndarray, idx_half: int) -> dict:
    """err [N, T, J] per-frame joint errors in m -> mm summary."""
    return dict(ade=round(float(err.mean()) * 1000, 1), at_half=round(float(err[:, idx_half].mean()) * 1000, 1),
                at_end=round(float(err[:, -1].mean()) * 1000, 1), n=int(len(err)))


# ------------------------------------------------------------------ CoMaD
def run_comad(cfg: dict) -> dict:
    f = cfg['forecast']
    root = resolve(f['repo'])
    sys.path.insert(0, str(root))
    import os
    cwd = os.getcwd()
    os.chdir(root)                                          # their loader reads ./data/...
    from data.utils.comad import CoMaD
    res = {}
    try:
        m = f['model']
        for task in ('handover', 'reactive_stirring', 'table_setting'):
            ds = CoMaD('./data/comad_data', m['input_n'], m['output_n'], sample_rate=m['rate_hz'], split=2,
                       mocap_splits=[[], [], [f'{task}/test']])
            mapping = json.load(open('data/mapping.json'))
            used = np.array([mapping[j] for j in ('BackTop', 'LShoulderBack', 'RShoulderBack', 'LElbowOut', 'RElbowOut',
                                                   'LWristOut', 'RWristOut')])
            seq = np.stack([np.asarray(s)[:, used] for s in ds.data_lst]).astype(np.float32)     # [N, 35, 7, 3]
            hist, gt = seq[:, :m['input_n']], seq[:, m['input_n']:]
            preds = {}
            for ck in ('base', 'manicast', 'manicastW'):
                fc = ManiCastForecaster(dict(cfg, forecast=dict(f, checkpoint=ck)))
                preds[ck] = np.concatenate([fc.predict_comad(hist[i:i + 512]) for i in range(0, len(hist), 512)])
            preds['current'] = np.repeat(hist[:, -1:], m['output_n'], 1)
            vel = (hist[:, -1] - hist[:, 0]) / (m['input_n'] - 1)
            preds['cvm'] = hist[:, -1:] + vel[:, None] * np.arange(1, m['output_n'] + 1)[None, :, None, None]
            moving = np.linalg.norm(gt[:, :, 5:7] - hist[:, -1:, 5:7], axis=-1).max((1, 2)) > cfg['forecast']['eval']['moving_m']
            half = int(round(0.5 * m['rate_hz'])) - 1
            out = {}
            for k, p in preds.items():
                e = np.linalg.norm(p - gt, axis=-1)                                   # [N, 25, 7]
                out[k] = dict(wrist=summarise(e[:, :, 5:7], half), elbow=summarise(e[:, :, 3:5], half),
                              all=summarise(e, half), wrist_moving=summarise(e[moving][:, :, 5:7], half))
            res[task] = out
            print(task, json.dumps({k: v['wrist'] for k, v in out.items()}))
    finally:
        os.chdir(cwd)
    return res


# ------------------------------------------------------------------ HABIT
def sim_episodes(cfg: dict):
    from pif.data.sim import read_dataset
    eps, _ = read_dataset(cfg, None)
    return eps


def run_habit(cfg: dict, task: str = 'S8789', stride: int = 2, episodes=None) -> dict:
    from pif.perception.lift import load_episode
    f = cfg['forecast']
    fc = ManiCastForecaster(cfg)
    kp_names = cfg['tokens']['keypoints']
    J = {n: i for i, n in enumerate(kp_names)}
    need = ['l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist']
    fps, gate = float(cfg['fps']), f['eval']['gate_arm_m']
    n_hist = int(round(0.4 * fps)) + 1                      # frames t-0.4 s .. t
    n_fut = int(round(1.0 * fps))
    t_hist = np.arange(-(n_hist - 1), 1) / fps
    t_out = np.arange(1, n_fut + 1) / fps
    H, W, F, G = [], [], [], []
    eps = episodes if episodes is not None else (load_episode(p) for p in sorted((resolve(cfg['habit']['tokens_dir']) / task).glob('episode_*.npz')))
    for ep in eps:
        pos, vis = ep['human_pos'], ep['human_vis']
        ok = np.all([vis[:, J[n]] for n in need], 0)
        for s in ('l', 'r'):
            ok &= np.linalg.norm(pos[:, J[f'{s}_wrist']] - pos[:, J[f'{s}_shoulder']], axis=1) <= gate
        T = len(pos)
        for t in range(n_hist - 1, T - n_fut, stride):
            if not ok[t - n_hist + 1:t + n_fut + 1].all():
                continue
            kp = {n: pos[t - n_hist + 1:t + n_fut + 1, J[n]] for n in need}
            mk, waist = fc.to_comad(kp)                       # [n_hist + n_fut, 7, 3]
            H.append(mk[:n_hist]); W.append(waist[:n_hist]); F.append(mk[n_hist:])
    H, W, F = np.stack(H), np.stack(W), np.stack(F)
    print(f'{task}: {len(H)} evaluation frames (stride {stride})', flush=True)
    pred = np.concatenate([fc.forecast(H[i:i + 256], W[i:i + 256], t_hist, t_out) for i in range(0, len(H), 256)])
    preds = dict(manicast=pred, current=np.repeat(H[:, -1:], n_fut, 1),
                 cvm=H[:, -1:] + ((H[:, -1] - H[:, 0]) / (t_hist[-1] - t_hist[0]))[:, None] * t_out[None, :, None, None])
    moving = np.linalg.norm(F[:, :, 5:7] - H[:, -1:, 5:7], axis=-1).max((1, 2)) > f['eval']['moving_m']
    half = int(round(0.5 * fps)) - 1
    out = {}
    for k, p in preds.items():
        e = np.linalg.norm(p - F, axis=-1)                                     # [N, 10, 7], rotation keeps lengths
        out[k] = dict(wrist=summarise(e[:, :, 5:7], half), elbow=summarise(e[:, :, 3:5], half),
                      wrist_moving=summarise(e[moving][:, :, 5:7], half))
    out['moving_fraction'] = round(float(moving.mean()), 3)
    print(json.dumps(out, indent=1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset', choices=['comad', 'habit', 'sim'])
    ap.add_argument('--config', default='configs/pif/pif.yaml')
    ap.add_argument('--set', nargs='*', default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    out = run_dir('forecast_eval', args.dataset)
    log_run(out, 'pif.forecast_eval')
    torch.set_num_threads(4)
    if args.dataset == 'comad':
        res = run_comad(cfg)
    elif args.dataset == 'habit':
        res = run_habit(cfg)
    else:                                                    # sim: exact keypoints of the scripted human, 10 Hz
        res = run_habit(cfg, task='sim', episodes=sim_episodes(cfg))
    (out / 'results.json').write_text(json.dumps(res, indent=1))
    print(f'-> {out}')


if __name__ == '__main__':
    main()

"""Fine-tune ManiCast on the v6 sim human motion with the authors' recipe (configs/pif/pif.yaml forecast.finetune,
docs/sim/scenarios_v6_proposal.md M5), then evaluate on the held-out test seeds against the released checkpoint,
the current pose and constant velocity (wrist ADE / at 0.5 s / at 1 s, all windows and transition windows).
Data: pif/v6_motion_data.py (20 Hz arm joints) -> ManiCast markers (pif/forecast.py to_comad) resampled to 25 Hz ->
windows of 10 input + 25 output frames, relative to the waist at the last input frame (as their handover eval).
  ./run.sh -m pif.forecast_finetune  -> outputs/forecast_ft/<date_time>/{best.pt, report.md, results.json, log}
"""
from __future__ import annotations
import argparse
import datetime
import json

import numpy as np
import torch

from pif.config import load_config
from pif.forecast import ManiCastForecaster
from pif.utils import log_run, resolve

ARM = ['l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist']
WRISTS = [5, 6]


def windows(npz, fc, cfg) -> tuple[np.ndarray, np.ndarray]:
    """(all windows, transition windows), each [N, 35, 7, 3] relative to the waist at the last input frame."""
    m, ft = cfg['forecast']['model'], cfg['forecast']['finetune']
    n_in, n_out, rate = m['input_n'], m['output_n'], float(m['rate_hz'])
    L = n_in + n_out
    allw, trans = [], []
    for sid in [k for k in npz.files if k not in ('joints', 'rate_hz', 'seeds') and not k.endswith('__moves')]:
        for kp, moves in zip(npz[sid], npz[f'{sid}__moves']):
            t20 = np.arange(len(kp)) / float(npz['rate_hz'])
            mk, waist = fc.to_comad({j: kp[:, i] for i, j in enumerate(ARM)})
            t25 = np.arange(0, t20[-1], 1.0 / rate)
            mk = np.stack([np.stack([np.interp(t25, t20, mk[:, j, c]) for c in range(3)], -1) for j in range(7)], 1)
            waist = np.stack([np.interp(t25, t20, waist[:, c]) for c in range(3)], -1)
            seq = lambda s: mk[s:s + L] - waist[s + n_in - 1]
            starts = set(range(0, len(t25) - L, int(ft['stride'])))
            tr = set()
            for t0, dur, _ in moves:
                a = int(round(float(t0) * rate)) - n_in
                b = int(round((float(t0) + float(dur) + float(ft['transition_after_s'])) * rate)) - L
                tr |= set(range(max(a, 0), min(b, len(t25) - L) + 1))
            allw += [seq(s) for s in sorted(starts)]
            trans += [seq(s) for s in sorted(tr)]
    return np.asarray(allw, np.float32), np.asarray(trans, np.float32)


def wrist_err(pred: np.ndarray, gt: np.ndarray) -> dict:
    e = np.linalg.norm(pred[:, :, WRISTS] - gt[:, :, WRISTS], axis=-1)        # [N, 25, 2]
    return dict(ade_mm=round(float(e.mean()) * 1000, 1), at_0_5s_mm=round(float(e[:, 11].mean()) * 1000, 1),
                at_1s_mm=round(float(e[:, -1].mean()) * 1000, 1), n=int(len(e)),
                at_1s_left_mm=round(float(e[:, -1, 0].mean()) * 1000, 1), at_1s_right_mm=round(float(e[:, -1, 1].mean()) * 1000, 1))


@torch.no_grad()
def predict(net, x: np.ndarray, dev) -> np.ndarray:
    out = []
    for a in range(0, len(x), 1024):
        h = torch.as_tensor(x[a:a + 1024], device=dev).permute(0, 3, 1, 2)
        out.append(net(h).permute(0, 1, 3, 2).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='configs/pif/pif.yaml')
    ap.add_argument('--set', nargs='*', default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    f, ft = cfg['forecast'], cfg['forecast']['finetune']
    n_in = f['model']['input_n']
    out = resolve(ft['out']) / datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    out.mkdir(parents=True, exist_ok=True)
    log_run(out, 'pif.forecast_finetune')
    torch.manual_seed(0)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fc = ManiCastForecaster(dict(cfg, forecast=dict(f, checkpoint=ft['init'])))
    data = {s: windows(np.load(resolve(f['data']['out']) / f'{s}.npz', allow_pickle=True), fc, cfg)
            for s in ('train', 'val', 'test')}
    for s, (a, t) in data.items():
        print(s, 'windows', a.shape, 'transition', t.shape, flush=True)
    net = fc.net.to(dev)
    w = torch.ones(7, device=dev)
    w[WRISTS] = float(ft['wrist_weight'])
    loss_fn = lambda p, g: (torch.norm(p - g, dim=-1) * w).mean() * 1000         # weighted MPJPE [mm], their scale
    opt = torch.optim.Adam(net.parameters(), lr=float(ft['lr']), weight_decay=float(ft['weight_decay']))
    tr_all, tr_tr = (torch.as_tensor(x) for x in data['train'])
    nb = max(len(tr_all) // int(ft['batch']), 1)
    bt = max(len(tr_tr) // nb, 1)                    # their pairing: as many transition batches as all-window batches
    best, log = (np.inf, None), []
    val_x = data['val'][1]
    for ep in range(int(ft['epochs'])):
        net.train()
        pa, pt = torch.randperm(len(tr_all)), torch.randperm(len(tr_tr))
        tot = 0.0
        for i in range(nb):
            loss = 0.0
            for batch in (tr_all[pa[i * int(ft['batch']):(i + 1) * int(ft['batch'])]], tr_tr[pt[i * bt:(i + 1) * bt]]):
                if not len(batch):
                    continue
                batch = batch.to(dev)
                pred = net(batch[:, :n_in].permute(0, 3, 1, 2)).permute(0, 1, 3, 2)
                loss = loss + loss_fn(pred, batch[:, n_in:])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
        net.eval()
        v = wrist_err(predict(net, val_x[:, :n_in], dev), val_x[:, n_in:])
        log.append(dict(epoch=ep, train_loss=round(tot / nb, 2), val_transition=v))
        print(json.dumps(log[-1]), flush=True)
        if v['ade_mm'] < best[0]:
            best = (v['ade_mm'], ep)
            torch.save(net.state_dict(), out / 'best.pt')
    net.load_state_dict(torch.load(out / 'best.pt', map_location=dev))
    base = ManiCastForecaster(dict(cfg, forecast=dict(f, checkpoint=ft['init']))).net.to(dev).eval()
    res = dict(best_epoch=best[1], log=log)
    rows = []
    for name, x in (('test all', data['test'][0]), ('test transition', data['test'][1])):
        hist, gt = x[:, :n_in], x[:, n_in:]
        vel = (hist[:, -1] - hist[:, 0]) / (n_in - 1)
        preds = {'fine-tuned': predict(net, hist, dev), f"released '{ft['init']}'": predict(base, hist, dev),
                 'current pose': np.repeat(hist[:, -1:], gt.shape[1], 1),
                 'constant velocity': hist[:, -1:] + vel[:, None] * np.arange(1, gt.shape[1] + 1)[None, :, None, None]}
        for path in ft.get('compare') or []:          # earlier fine-tuned weights on the same test windows
            old = ManiCastForecaster(dict(cfg, forecast=dict(f, checkpoint_path=path))).net.to(dev).eval()
            preds[f'previous {path}'] = predict(old, hist, dev)
        res[name] = {k: wrist_err(p, gt) for k, p in preds.items()}
        rows += [f"| {name} | {k} | {v['ade_mm']} | {v['at_0_5s_mm']} | {v['at_1s_mm']} | {v['at_1s_left_mm']} | {v['at_1s_right_mm']} | {v['n']} |"
                 for k, v in res[name].items()]
    (out / 'results.json').write_text(json.dumps(res, indent=1))
    rep = [f"# ManiCast fine-tuned on {f['data']['out']} ({out.name})", '', f"best epoch {best[1]} (val transition wrist ADE {best[0]} mm)",
           '', '| windows | forecaster | wrist ADE mm | at 0.5 s | at 1 s | at 1 s left | at 1 s right | n |',
           '|---|---|---|---|---|---|---|---|'] + rows
    (out / 'report.md').write_text('\n'.join(rep) + '\n')
    print('\n'.join(rep))
    print(f'-> {out}')


if __name__ == '__main__':
    main()

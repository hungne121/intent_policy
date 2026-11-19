"""Offline counterfactual evaluation of a trained checkpoint on the decisive VALIDATION windows (19/QĐ-HL item 9;
measurement of 17/QĐ-HL B). Cache = outputs/diag/m1_p1_1119/cache/windows_cache.pt (build_cache.py of P1: val windows chosen
by the event rule of docs/reports/M1_P1_OVERFIT_2026-11-19.md §1, never by forecast quality; `shuffle_src` = a fixed
within-task partner window of another episode, `wrong_target_src` = a window with a different target).

Per condition the observation is fixed and only the forecast changes:
  true | shuffled (partner's forecast, valid 1) | wrong_target | mask (zeros, valid 0)
Reported: val L1 (normalised action, mean over the window chunk), action sensitivity (mean |delta| vs true, normalised and
TCP mm over the first 10 steps), and counterfactual action correctness: u = goal(wrong target) - goal(own target) from the
mean expert chunk-end TCP of the TRAIN windows; correct = projection of the mean TCP change (wrong_target - true) on u > 0.
A model without forecast input (M0 / M1-noH) only gets the true-condition L1.

    ./run.sh -m tools.eval_counterfactual --ckpt <.../pretrained_model> --label M-star@10000 --out <dir>
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CAMS = ['observation.images.front_oblique', 'observation.images.overhead_oblique', 'observation.images.wrist']
CACHE = ROOT / 'outputs/diag/m1_p1_1119/cache/windows_cache.pt'
STATS = ROOT / 'outputs/diag/m1_collapse_p0_1119/data/hf_dataset_repo/train_sets/train_set_w2fixed/D_forecast/train/meta/stats.json'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', type=Path, required=True)
    p.add_argument('--label', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--cache', type=Path, default=CACHE)
    p.add_argument('--stats', type=Path, default=STATS)
    p.add_argument('--batch', type=int, default=10)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import make_pre_post_processors
    from intent_policy.policies.forecast_token import TcpFK
    from intent_policy.policies.hri_act import HRIACTPolicy

    dev = a.device
    cfg = PreTrainedConfig.from_pretrained(a.ckpt)
    cfg.device = dev
    pol = HRIACTPolicy.from_pretrained(a.ckpt, config=cfg).to(dev).eval()
    pre, _ = make_pre_post_processors(cfg, pretrained_path=str(a.ckpt), preprocessor_overrides={'device_processor': {'device': dev}})
    cache = torch.load(a.cache, weights_only=False)
    TR, VA = cache['train'], cache['val']
    st = json.loads(a.stats.read_text())
    a_mean = torch.tensor(st['action']['mean'], dtype=torch.float32, device=dev)
    a_std = torch.tensor(st['action']['std'], dtype=torch.float32, device=dev)
    has_fc = bool(cfg.use_forecast)
    fk = TcpFK(cfg.forecast_fk_chain).to(dev) if has_fc else None

    def raw_tcp(pred):
        q = pred * (a_std + 1e-8) + a_mean
        b, t = q.shape[:2]
        return fk(q[..., :6].reshape(-1, 6)).reshape(b, t, 3) * 1000.0

    def forecast_for(D, idx, cond):
        if cond == 'true':
            return D['observation.human_forecast'][idx], D['observation.forecast_valid'][idx]
        if cond == 'shuffled':
            return D['observation.human_forecast'][[D['windows'][i]['shuffle_src'] for i in idx]], torch.ones(len(idx), 1)
        if cond == 'wrong_target':
            return D['observation.human_forecast'][[D['windows'][i]['wrong_target_src'] for i in idx]], torch.ones(len(idx), 1)
        return torch.zeros(len(idx), 180), torch.zeros(len(idx), 1)

    @torch.no_grad()
    def predict(D, cond):
        out, tgt = [], []
        for k in range(0, len(D['windows']), a.batch):
            idx = list(range(k, min(k + a.batch, len(D['windows']))))
            b = {c: D[c][idx].float() / 255.0 for c in CAMS}
            for key in ('observation.state', 'action', 'action_is_pad'):
                b[key] = D[key][idx]
            b['observation.human_forecast'], b['observation.forecast_valid'] = forecast_for(D, idx, cond)
            b['task'] = [D['task_str'][i] for i in idx]
            b = pre(b)
            bb = dict(b)
            bb['observation.images'] = [b[c] for c in cfg.image_features]
            ah, _ = pol.model(bb)
            out.append(ah)
            tgt.append(b['action'])
        return torch.cat(out), torch.cat(tgt)

    W = VA['windows']
    tasks = [w['task'] for w in W]
    l1 = lambda x, y: (x - y).abs().mean((1, 2))
    base, tgt = predict(VA, 'true')
    per = l1(base, tgt).cpu().numpy()
    res = dict(label=a.label, ckpt=str(a.ckpt), step=json.loads((a.ckpt / 'run_meta.json').read_text()).get('step') if (a.ckpt / 'run_meta.json').exists() else None,
               n_windows=len(W), has_forecast_input=has_fc, val_l1_true=float(per.mean()),
               val_l1_true_by_task={t: float(per[[i for i, x in enumerate(tasks) if x == t]].mean()) for t in sorted(set(tasks))})
    if has_fc:
        with torch.no_grad():
            te = raw_tcp(TR['action'].to(dev))[:, -1]
        goal = {}
        for t in sorted({w['target'] for w in TR['windows'] if w['target']}):
            goal[t] = te[[i for i, w in enumerate(TR['windows']) if w['target'] == t]].mean(0)
        preds = {c: predict(VA, c)[0] for c in ('shuffled', 'wrong_target', 'mask')}
        for c, q in preds.items():
            res[f'val_l1_{c}'] = float(l1(q, tgt).mean())
            res[f'sens_{c}_norm_median'] = float(np.median((q - base).abs().mean((1, 2)).cpu().numpy()))
            res[f'sens_{c}_tcp10_mm_median'] = float(np.median((raw_tcp(q) - raw_tcp(base)).norm(dim=-1)[:, :10].mean(1).cpu().numpy()))
        d = (raw_tcp(preds['wrong_target']) - raw_tcp(base)).mean(1)
        z = torch.zeros(3, device=dev)
        U = torch.stack([goal.get(W[W[i]['wrong_target_src']]['target'], z) - goal.get(W[i]['target'], z) for i in range(len(W))])
        nu = U.norm(dim=1)
        ok = nu > 1.0
        proj = (d * U).sum(1) / nu.clamp_min(1e-6)
        dd = raw_tcp(preds['wrong_target']) - raw_tcp(base)                           # (B, 40, 3) mm per chunk step
        fr_step = {k: ((dd[:, k] * U).sum(1) / nu.clamp_min(1e-6) ** 2)[ok] for k in (9, 19, 39)}     # share of |U| at chunk step k+1
        res['shift_fraction_by_step'] = {f'step{k + 1}': dict(median=float(v.median()), frac_ge_half=float((v >= 0.5).float().mean())) for k, v in fr_step.items()}
        frac = (proj / nu.clamp_min(1e-6))[ok]                     # share of the goal-to-goal distance the action moved toward the new target
        okidx = torch.nonzero(ok).flatten().tolist()
        res['shift_fraction'] = dict(median=float(frac.median()), mean=float(frac.mean()), p25=float(frac.quantile(0.25)), p75=float(frac.quantile(0.75)),
                                     frac_ge_half=float((frac >= 0.5).float().mean()), frac_ge_quarter=float((frac >= 0.25).float().mean()),
                                     by_task={t: float(frac[[k for k, i in enumerate(okidx) if tasks[i] == t]].median()) for t in sorted(set(tasks))})
        res['correctness'] = dict(n=int(ok.sum()), frac_correct=float((proj[ok] > 0).float().mean()), proj_mm_median=float(proj[ok].median()),
                                  dtcp_mm_median=float(d.norm(dim=1)[ok].median()), u_mm_median=float(nu[ok].median()))
        by = {}
        for t in sorted(set(tasks)):
            ii = [i for i in range(len(W)) if tasks[i] == t and bool(ok[i])]
            by[t] = dict(n=len(ii), frac_correct=float((proj[ii] > 0).float().mean()) if ii else None,
                         proj_mm_median=float(proj[ii].median()) if ii else None)
        res['correctness']['by_task'] = by
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f'{a.label.replace("@", "_").replace("/", "_")}.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ('val_l1_true_by_task',)}, indent=1))


if __name__ == '__main__':
    main()

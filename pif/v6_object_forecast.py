"""M5: future position of an object the human carries (configs/pif/pif.yaml v6_object_forecast). ManiCast forecasts
the arm only; a held object is predicted to move rigidly with the wrist (forecast wrist + the current wrist -> object
offset). Held = the human holds it (in sim the scenario's state; observable as the object moving with the hand).
Compared with: the object stays, constant velocity. Frames where the object is released within the horizon are scored
separately (there the rigid rule must fail: the object stops, the hand goes on).
  ./run.sh -m pif.v6_object_forecast [--set "v6_object_forecast.observers={released: null, fine-tuned: <best.pt>}"]
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict

import numpy as np
import torch

from pif.config import load_config
from pif.forecast import ManiCastForecaster
from pif.utils import log_run, run_dir

NEED = ['l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist']


def record(sc, seed: int) -> dict:
    from intent_policy.benchmark.runner import ExpertAgent
    from intent_policy.sim.restricted_action import RestrictedActionConfig, RestrictedActionMapper
    sc.reset(seed)
    mapper = RestrictedActionMapper(sc.env, RestrictedActionConfig.load())
    ag = ExpertAgent()
    ag.reset(sc, mapper)
    keys = list(sc.on_table)
    kp, obj, held, tick = [], [], [], 0
    while not sc.done:
        if tick % 2 == 0:                                       # 10 Hz
            kp.append(sc.human_keypoints(NEED))
            obj.append([sc.pos(k) for k in keys])
            held.append(keys.index(sc.human.holding) if sc.human.holding in keys else -1)
        sc.step(mapper.map(ag.act(None).action_id).joint_target)
        tick += 1
    return dict(kp=np.asarray(kp), obj=np.asarray(obj), held=np.asarray(held), keys=keys)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='configs/pif/pif.yaml')
    ap.add_argument('--set', nargs='*', default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    vo = cfg['v6_object_forecast']
    out = run_dir('v6_object_forecast', 'm5')
    log_run(out, 'pif.v6_object_forecast')
    torch.set_num_threads(4)
    from intent_policy.scenarios.scenario_registry import make_scenario
    fps = float(cfg['fps'])
    hz = [int(round(h * fps)) for h in vo['horizons_s']]
    t_hist, t_out = np.arange(-4, 1) / fps, np.asarray(vo['horizons_s'], float)
    fcs = {n: ManiCastForecaster(dict(cfg, forecast=dict(cfg['forecast'], checkpoint_path=p))) for n, p in vo['observers'].items()}
    errs = defaultdict(list)                 # (task, kind, observer, horizon) -> errors [m]
    for sid, n in vo['episodes'].items():
        sc = make_scenario(sid)
        for i in range(n):
            r = record(sc, vo['seed0'] + i)
            T = len(r['held'])
            idx = np.arange(4, T - hz[-1])
            idx = idx[r['held'][idx] >= 0]
            if not len(idx):
                continue
            wrist = r['kp'][:, 5]
            k = r['held'][idx]
            o_now = r['obj'][idx, k]
            off = o_now - wrist[idx]
            mk, waist = next(iter(fcs.values())).to_comad({j: r['kp'][:, c] for c, j in enumerate(NEED)})
            H = np.stack([mk[j - 4:j + 1] for j in idx])
            W = np.stack([waist[j - 4:j + 1] for j in idx])
            preds = {'object stays': [o_now] * len(hz)}
            vel = (o_now - r['obj'][idx - 4, k]) / (4 / fps)
            preds['constant velocity'] = [o_now + vel * h for h in vo['horizons_s']]
            t_fine = np.arange(1, 11) / fps                                   # 0.1 .. 1.0 s
            v_obs = np.r_[0.0, np.linalg.norm(np.diff(wrist, axis=0), axis=1) * fps]
            peak = np.array([v_obs[max(t - 20, 0):t + 1].max() for t in idx])   # peak hand speed over the last 2 s
            for name, fc in fcs.items():
                p = np.concatenate([fc.forecast(H[a:a + 256], W[a:a + 256], t_hist, t_fine) for a in range(0, len(H), 256)])
                fw = np.stack([fc.from_comad(p[:, j, 6]) for j in range(len(t_fine))], 1)   # [N, 10, 3] forecast wrist
                preds[f'{name} + held offset'] = [fw[:, h - 1] + off for h in hz]
                # the object follows the forecast hand until the hand is forecast to stop (speed < still_fraction x
                # the recent peak, the movement-end convention), then stays there (put down / held still)
                fv = np.r_['1', np.linalg.norm(fw[:, :1] - wrist[idx][:, None], axis=2),
                           np.linalg.norm(np.diff(fw, axis=1), axis=2)] * fps
                stop = fv < vo['still_fraction'] * np.maximum(peak, 1e-6)[:, None]
                first = np.where(stop.any(1), stop.argmax(1), len(t_fine) - 1)
                preds[f'{name} + held offset, stops'] = [fw[np.arange(len(idx)), np.minimum(first, h - 1)] + off for h in hz]
            for j, h in enumerate(hz):
                gt = r['obj'][idx + h, k]
                released = np.array([(r['held'][t + 1:t + h + 1] != kk).any() for t, kk in zip(idx, k)])
                moving = np.linalg.norm(gt - o_now, axis=1) > vo['moving_m']
                for name, pr in preds.items():
                    e = np.linalg.norm(pr[j] - gt, axis=1)
                    for kind, m in (('held through', ~released), ('held through, moving', ~released & moving),
                                    ('released within', released)):
                        errs[(sid, kind, name, vo['horizons_s'][j])] += e[m].tolist()
        sc.close()
        print(sid, 'done', flush=True)
    names = ['object stays', 'constant velocity'] + [f'{n} + held offset{x}' for n in fcs for x in ('', ', stops')]
    lines = [f'# Held-object position forecast ({out.name}); mean error mm (n frames)', '',
             '| task | frames | horizon s | ' + ' | '.join(names) + ' |', '|' + '---|' * (3 + len(names))]
    res = {}
    for sid in vo['episodes']:
        for kind in ('held through', 'held through, moving', 'released within'):
            for h in vo['horizons_s']:
                cells = []
                for nm in names:
                    e = errs.get((sid, kind, nm, h), [])
                    res[f'{sid}|{kind}|{nm}|{h}'] = dict(mean_mm=round(float(np.mean(e)) * 1000, 1) if e else None, n=len(e))
                    cells.append(f"{np.mean(e) * 1000:.0f} ({len(e)})" if e else '-')
                lines.append(f'| {sid} | {kind} | {h} | ' + ' | '.join(cells) + ' |')
    (out / 'results.json').write_text(json.dumps(res, indent=1))
    (out / 'report.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'-> {out}')


if __name__ == '__main__':
    main()

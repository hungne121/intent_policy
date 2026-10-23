"""P1 report numbers and figures for the sim PIF data (handoff §3 "Báo cáo P1"): samples, K min / max, hidden-keypoint
rate, per task; top-view plots of a few episodes (hand keypoints + object tokens, table frame).

Writes <out>/sim_stats.json and <out>/sim_episode_<task>.png.
"""
import argparse
import json
from collections import defaultdict

import numpy as np
import yaml

from intent_policy.utils import resolve
from pif.data.dataset import PIFWindowDataset
from pif.data.episode_reader import load_meta, read_sim_dataset
from pif.data.labels import TARGET_HUMAN, TARGET_NONE, TARGET_REGION, sim_labels


def plot_episode(ep: dict, lab: dict, path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ep['obj_names']
    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=130)
    kp = ep['human_pos']
    for j, (key, col) in enumerate([('r_wrist', '#1f77b4'), ('r_index_tip', '#6baed6')]):
        k = list(CFG['tokens']['keypoints']).index(key)
        ax.plot(kp[:, k, 0], kp[:, k, 1], '-', color=col, lw=1, label=key)
    types = CFG['tokens']['object_types']
    marker = dict(object='s', container='o', surface='D')
    for k, n in enumerate(names):
        p = ep['obj_pos'][:, k]
        t = types[ep['obj_type'][k]]
        moved = np.ptp(p[:, :2], 0).max() > 0.01
        ax.plot(p[0, 0], p[0, 1], marker[t], ms=8, mfc='none', mec='k')
        if moved:
            ax.plot(p[:, 0], p[:, 1], ':', color='0.4', lw=1)
            ax.plot(p[-1, 0], p[-1, 1], marker[t], ms=8, mfc='0.6', mec='k')
        ax.annotate(f'{k}:{n}', p[0, :2], textcoords='offset points', xytext=(5, 4), fontsize=7)
    tg = [g for g in lab['gestures'] if g['target'] not in (TARGET_NONE,)]
    txt = '; '.join(f"{g['label']}→{g['target'] if g['target'] >= 0 else {TARGET_HUMAN: 'human', TARGET_REGION: 'region'}[g['target']]}"
                    f" [{g['onset']},{g['clear']}]" for g in tg)
    ax.set_title(f"{ep['meta']['task']} ep {ep['meta']['episode_index']} (10 Hz, table frame)\n{txt}", fontsize=7)
    ax.set_xlabel('x (m, away from robot)')
    ax.set_ylabel("y (m, robot's left)")
    ax.set_aspect('equal')
    ax.legend(fontsize=7, loc='lower right')
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    global CFG
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/pif/base.yaml')
    p.add_argument('--dataset', default='outputs/datasets/intent_act_late_v5')
    p.add_argument('--out', default='docs/reports/PIF_P1')
    args = p.parse_args()
    CFG = yaml.safe_load(resolve(args.config).read_text())
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metas = load_meta(resolve(args.dataset))
    eps = read_sim_dataset(args.dataset, CFG['sim']['replay'], CFG)
    labs = [sim_labels(metas[e['meta']['episode_index']], e, CFG) for e in eps]
    ds = PIFWindowDataset(eps, CFG, train=False)
    by = defaultdict(list)
    for e, l in zip(eps, labs):
        by[e['meta']['task']].append((e, l))
    rows = {}
    for task, items in sorted(by.items()):
        K = [len(e['obj_type']) for e, _ in items]
        T = [len(e['holding']) for e, _ in items]
        g = [(e, x) for e, l in items for x in l['gestures'] if x['target'] != TARGET_NONE]
        tgt = defaultdict(int)
        for e, x in g:
            kind = CFG['tokens']['object_types'][e['obj_type'][x['target']]] if x['target'] >= 0 else \
                {TARGET_HUMAN: 'human', TARGET_REGION: 'region'}[x['target']]
            tgt[kind] += 1
        rows[task] = dict(episodes=len(items), frames_10hz=int(sum(T)), seconds_mean=round(float(np.mean(T)) / CFG['fps'], 1),
                          K_min=int(min(K)), K_max=int(max(K)),
                          samples=int(sum(max(1, t - ds.L + 1) for t in T)),
                          holding_frac=round(float(np.mean(np.concatenate([e['holding'] >= 0 for e, _ in items]))), 3),
                          gestures_with_target=len(g), target_kinds=dict(tgt))
    allK = [len(e['obj_type']) for e in eps]
    hidden = float(np.mean(np.concatenate([~e['human_vis'].ravel() for e in eps])))
    stats = dict(episodes=len(eps), frames_10hz=int(sum(len(e['holding']) for e in eps)), samples=len(ds), window_frames=ds.L,
                 horizons_frames=ds.horizons.tolist(), K_min=int(min(allK)), K_max=int(max(allK)),
                 hidden_keypoint_frac=hidden, per_task=rows)
    (out / 'sim_stats.json').write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    for task, items in sorted(by.items()):
        e, l = items[0]
        plot_episode(e, l, out / f'sim_episode_{task}.png')


if __name__ == '__main__':
    main()

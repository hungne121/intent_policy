"""HABIT (configinc/HABIT, `full` config, 10 Hz): download, robot state + gripper events, label rules.

  python -m pif.data.habit download   [--sids ...]   meta, task_details, parquets, camera videos (skips present files)
  python -m pif.data.habit labels                     label rules of every task -> <root>/labels/<sid>.json
  python -m pif.data.habit sheets --n 2               contact sheets to check the rules by eye

Facts this code relies on (PIF_P0, PIF_P1): subtask indices of `sample` and `full` differ (decode with the same
config's meta); robot0 / robot1 positions are each in their own arm base frame (robot0 = left arm), no transform
between them, no camera intrinsics; `episode_index` in episodes.jsonl is a string; gripper thresholds per task
(configs/pif/habit.yaml). Label rules (eval only, never training inputs):
  S8789 target container = where the robot releases the bread: arm of the last grasp + TCP y at the next opening;
        each arm serves two containers (split at the largest y gap); containers 1-4 left to right from the robot.
        Holds for session 260414 (190/208 episodes); the 14 + 4 episodes of 260415 / 260418 are flagged.
  S8791 the container the human closes next = the one the robot just filled; 2 containers one behind the other along
        robot1's x ('near' / 'far', split at the largest gap).
  S6760 sequence robot0 close / open (human has the board) / robot1 close (human hands it back) / open (tray).
"""
from __future__ import annotations
import argparse
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from pif.utils import resolve
from pif.config import load_config

REPO = 'configinc/HABIT'
FPS = 10.0
ARMS = ('robot0', 'robot1')
META = ['info.json', 'episodes.jsonl', 'human_subtasks.jsonl', 'subtasks.jsonl', 'tasks.jsonl', 'modality.json']


# ------------------------------------------------------------------ files and meta
@lru_cache(maxsize=2)
def habit_meta(root: str = 'outputs/habit') -> dict:
    d = resolve(root) / 'full/meta'
    read = lambda f: [json.loads(line) for line in open(d / f)]
    return dict(episodes={int(e['episode_index']): e for e in read('episodes.jsonl')},
                human={x['task_index']: x['task'] for x in read('human_subtasks.jsonl')},
                robot={x['task_index']: x['task'] for x in read('subtasks.jsonl')})


def episode_indices(sid: str, root: str = 'outputs/habit') -> list[int]:
    return sorted(i for i, e in habit_meta(root)['episodes'].items() if e['sid'] == sid)


def session(index: int, root: str = 'outputs/habit') -> str:
    """Recording date (YYMMDD) from the episode's input_key."""
    return re.search(r'/(\d{6})/', ' '.join(habit_meta(root)['episodes'][index]['input_key'])).group(1)


def paths(index: int, root: str = 'outputs/habit', cam: str = 'front_view') -> tuple[Path, Path]:
    base, c = resolve(root) / 'full', index // 1000
    return (base / f'data/chunk-{c:03d}/episode_{index:06d}.parquet',
            base / f'videos/chunk-{c:03d}/observation.images.{cam}/episode_{index:06d}.mp4')


def download(cfg: dict, sids: list[str], cams: list[str], workers: int = 4) -> None:
    from huggingface_hub import hf_hub_download, snapshot_download
    out = resolve(cfg['root'])
    for f in META:
        hf_hub_download(REPO, f'full/meta/{f}', repo_type='dataset', local_dir=out)
    for s in sids:
        hf_hub_download(REPO, f'task_details/{s}.md', repo_type='dataset', local_dir=out)
    habit_meta.cache_clear()
    files = []
    for s in sids:
        for i in episode_indices(s, cfg['root']):
            pq, _ = paths(i, cfg['root'])
            files += [str(pq.relative_to(out))] + [str(paths(i, cfg['root'], c)[1].relative_to(out)) for c in cams]
    todo = [f for f in files if not (out / f).exists()]
    print(f'{len(files)} files for {sids} ({len(files) - len(todo)} present)', flush=True)
    if todo:
        snapshot_download(REPO, repo_type='dataset', local_dir=out, allow_patterns=todo, max_workers=workers)
    print('missing:', [f for f in files if not (out / f).exists()][:5])


# ------------------------------------------------------------------ robot state
def runs(x: np.ndarray) -> list[tuple[int, int, object]]:
    """Run-length segments (start, end inclusive, value) of a 1-D array."""
    cut = np.flatnonzero(np.r_[True, x[1:] != x[:-1]])
    return [(int(a), int(b) - 1, x[a]) for a, b in zip(cut, np.r_[cut[1:], len(x)])]


def debounce(x: np.ndarray, min_frames: int) -> np.ndarray:
    """Runs shorter than `min_frames` take the previous run's value; the first and last run are kept (cut by the
    episode bounds, not flicker)."""
    x = x.copy()
    for a, b, _ in runs(x):
        if 0 < a and b < len(x) - 1 and b - a + 1 < min_frames:
            x[a:b + 1] = x[a - 1]
    return x


def read_episode(index: int, root: str = 'outputs/habit') -> dict:
    meta = habit_meta(root)
    df = pd.read_parquet(paths(index, root)[0])
    out = dict(index=index, sid=meta['episodes'][index]['sid'], t=df['frame_index'].to_numpy() / FPS)
    for arm in ARMS:
        out[f'{arm}_tcp'] = np.stack(df[f'{arm}.observation.state.cartesian_position'].to_numpy())[:, :3]
        out[f'{arm}_grip'] = np.hstack(df[f'{arm}.observation.state.gripper_position'].to_numpy())
    out['human_segments'] = [dict(start=a, end=b, text=meta['human'][int(v)])
                             for a, b, v in runs(df['human_role_subtask_index'].to_numpy())]
    out['robot_segments'] = [dict(start=a, end=b, text=meta['robot'][int(v)])
                             for a, b, v in runs(df['low_level_task_index'].to_numpy())]
    return out


def gripper_closed(ep: dict, arm: str, threshold: float, min_frames: int) -> np.ndarray:
    return debounce(ep[f'{arm}_grip'] > threshold, min_frames)


def gripper_events(ep: dict, threshold: float, min_frames: int = 1) -> list[dict]:
    """Close / open events of each arm: {arm, frame, kind, tcp} at the first frame of the new state."""
    events = []
    for arm in ARMS:
        closed = gripper_closed(ep, arm, threshold, min_frames)
        for f in np.flatnonzero(closed[1:] != closed[:-1]) + 1:
            events.append(dict(arm=arm, frame=int(f), kind='close' if closed[f] else 'open',
                               tcp=ep[f'{arm}_tcp'][f].round(4).tolist()))
    return sorted(events, key=lambda e: e['frame'])


# ------------------------------------------------------------------ label rules
def _events(cfg, sid, i):
    t = cfg['tasks'][sid]
    return gripper_events(read_episode(i, cfg['root']), t['gripper_threshold'], cfg['gripper_min_frames'])


def labels_s8789(cfg: dict) -> dict:
    rows = []
    for i in episode_indices('S8789', cfg['root']):
        ev = _events(cfg, 'S8789', i)
        closes = [e for e in ev if e['kind'] == 'close']
        r = dict(episode=i, session=session(i, cfg['root']), arm=None, grasp=None, release=None, y=None)
        if closes:
            last = closes[-1]
            op = [e for e in ev if e['kind'] == 'open' and e['arm'] == last['arm'] and e['frame'] > last['frame']]
            r.update(arm=last['arm'], grasp=last['frame'], n_grasps=len(closes))
            if op:
                r.update(release=op[0]['frame'], y=op[0]['tcp'][1], tcp=op[0]['tcp'])
        rows.append(r)
    main = max({r['session'] for r in rows}, key=lambda s: sum(r['session'] == s for r in rows))
    split = {}
    for arm, first in (('robot0', 1), ('robot1', 3)):
        ys = np.sort([r['y'] for r in rows if r['arm'] == arm and r['y'] is not None and r['session'] == main])
        k = int(np.argmax(np.diff(ys)))
        split[arm] = dict(threshold=float((ys[k] + ys[k + 1]) / 2), gap=float(ys[k + 1] - ys[k]))
        for r in rows:
            if r['arm'] == arm and r['y'] is not None:
                r['target'] = first if r['y'] > split[arm]['threshold'] else first + 1
                r['trusted'] = r['session'] == main
    ref = {}                                             # metric reference distances (robot kinematics, main session)
    c = {k: np.median([r['tcp'] for r in rows if r.get('target') == k and r['trusted']], 0) for k in (1, 2, 3, 4)}
    ref = dict(d12=float(np.linalg.norm((c[1] - c[2])[:2])), d34=float(np.linalg.norm((c[3] - c[4])[:2])))
    return dict(rule='release position', main_session=main, split=split, reference_m=ref, episodes=rows)


def labels_s8791(cfg: dict) -> dict:
    rows = []
    for i in episode_indices('S8791', cfg['root']):
        rows += [dict(episode=i, arm=e['arm'], frame=e['frame'], x=e['tcp'][0], y=e['tcp'][1])
                 for e in _events(cfg, 'S8791', i) if e['kind'] == 'open']
    xs = np.sort([r['x'] for r in rows])
    k = int(np.argmax(np.diff(xs)))
    thr = float((xs[k] + xs[k + 1]) / 2)
    for r in rows:
        r['target'] = 'near' if r['x'] < thr else 'far'
    return dict(rule='container just filled', split=dict(threshold=thr, gap=float(xs[k + 1] - xs[k])), releases=rows)


def labels_s6760(cfg: dict) -> dict:
    rows = []
    want = [('robot0', 'close'), ('robot0', 'open'), ('robot1', 'close'), ('robot1', 'open')]
    for i in episode_indices('S6760', cfg['root']):
        ev = _events(cfg, 'S6760', i)
        r = dict(episode=i, sequence_ok=[(e['arm'], e['kind']) for e in ev] == want)
        if r['sequence_ok']:
            r.update(t_robot_takes_board=ev[0]['frame'], t_human_has=ev[1]['frame'], t_human_ready_return=ev[2]['frame'],
                     t_in_tray=ev[3]['frame'])
        rows.append(r)
    return dict(rule='gripper event sequence', episodes=rows)


LABEL_RULES = {'S8789': labels_s8789, 'S8791': labels_s8791, 'S6760': labels_s6760}


def load_labels(cfg: dict, sid: str) -> dict:
    return json.loads((resolve(cfg['root']) / 'labels' / f'{sid}.json').read_text())


def contact_sheet(video, frames: list[int], title: str, out) -> None:
    import av
    from PIL import Image, ImageDraw
    want, got = sorted(set(frames)), {}
    with av.open(str(video)) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i in want:
                got[i] = fr.to_image().resize((320, 240))
            if len(got) == len(want):
                break
    rows = (len(want) + 3) // 4
    img = Image.new('RGB', (1280, 240 * rows + 18))
    ImageDraw.Draw(img).text((4, 3), title, fill=(255, 255, 0))
    for j, f in enumerate(want):
        if f in got:
            ImageDraw.Draw(got[f]).text((3, 2), f't={f / FPS:.1f}s', fill=(255, 255, 0))
            img.paste(got[f], ((j % 4) * 320, 18 + (j // 4) * 240))
    img.save(out, quality=85)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command', choices=['download', 'labels', 'sheets'])
    p.add_argument('--config', default='configs/pif/habit.yaml')
    p.add_argument('--sids', nargs='*')
    p.add_argument('--cams', nargs='*', default=['front_view'])
    p.add_argument('--n', type=int, default=2, help='sheets per S8789 container')
    args = p.parse_args()
    cfg = load_config(args.config)['habit']
    sids = args.sids or list(cfg['tasks'])
    if args.command == 'download':
        download(cfg, sids, args.cams)
    elif args.command == 'labels':
        out = resolve(cfg['root']) / 'labels'
        out.mkdir(parents=True, exist_ok=True)
        for s in sids:
            lab = LABEL_RULES[s](cfg)
            (out / f'{s}.json').write_text(json.dumps(lab, indent=1))
            print(s, {k: v for k, v in lab.items() if k not in ('episodes', 'releases')})
    else:
        lab = load_labels(cfg, 'S8789')
        out = resolve(cfg['root']) / 'label_check'
        out.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        for k in (1, 2, 3, 4):
            cand = [r for r in lab['episodes'] if r.get('target') == k]
            for r in rng.permutation(cand)[:args.n] if cand else []:
                fr = np.linspace(max(r['grasp'] - 40, 0), r['release'], 12).round().astype(int).tolist()
                contact_sheet(paths(r['episode'], cfg['root'])[1], fr, f"S8789 ep {r['episode']}: rule -> container {k}",
                              out / f"s8789_c{k}_ep{r['episode']}.jpg")


if __name__ == '__main__':
    main()

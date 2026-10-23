"""Label rules of the HABIT tasks (HANDOFF_PIF_LOCAL.md §4, CORRECTIONS D.2 "kiểm luật suy đích trên video").

S8789 (Supervisor, 4 identical containers in a row): target = where the robot releases the bread: the arm of the last
grasp and its TCP y at the following gripper opening (each arm's own base frame, PIF_P0 §2.2). Each arm serves two
containers; the release positions form two clusters per arm (split at the largest gap of the sorted y). Containers are
numbered 1-4 left to right from the robot (robot0 = left arm; +y = the arm's left).

S8791 (Coworker, robot1 fills two containers, the human closes the lid of the filled one): the container the human is
about to close = the one the robot just released the bread into. The two containers stand one behind the other along
robot1's x (release x clusters 0.35-0.45 / 0.55-0.65 m, split at the largest gap): 'near' / 'far' from robot1.

S6760 (Collaborator, PCB hand-over): every episode has the event sequence robot0 close (PCB from the rack), robot0 open
(the human has taken it), robot1 close (the robot takes it back from the human), robot1 open (in the ESD tray). Times:
t_human_has = robot0 open, t_human_ready_return = robot1 close.

`--sheets N`: contact sheets (front_view) of N episodes per container, frames from 4 s before the grasp to the release,
for checking the rule by eye against the person's pointing. Writes <out>/s8789_targets.json (+ sheets).
"""
import argparse
import json

import av
import numpy as np
import yaml
from PIL import Image, ImageDraw

from intent_policy.utils import resolve
from pif.data.habit import episode_indices, gripper_events, paths, read_habit_episode


def s8789_releases(cfg: dict) -> list[dict]:
    g = cfg['gripper']
    out = []
    for i in episode_indices('S8789'):
        ep = read_habit_episode(i)
        ev = gripper_events(ep, g['threshold']['S8789'], g['min_frames'])
        closes = [e for e in ev if e['kind'] == 'close']
        r = dict(episode=i, T=len(ep['t']), arm=None, grasp=None, release=None, y=None)
        if closes:
            last = closes[-1]
            op = [e for e in ev if e['kind'] == 'open' and e['arm'] == last['arm'] and e['frame'] > last['frame']]
            r.update(arm=last['arm'], grasp=last['frame'], n_grasps=len(closes))
            if op:
                r.update(release=op[0]['frame'], y=op[0]['tcp'][1])
        out.append(r)
    return out


def assign_s8789(rel: list[dict]) -> dict:
    """Split each arm's release y at its largest gap; container = 1/2 (robot0: high y / low y), 3/4 (robot1)."""
    split = {}
    for arm, first in (('robot0', 1), ('robot1', 3)):
        ys = np.sort([r['y'] for r in rel if r['arm'] == arm and r['y'] is not None])
        k = int(np.argmax(np.diff(ys)))
        split[arm] = dict(threshold=float((ys[k] + ys[k + 1]) / 2), gap=float(ys[k + 1] - ys[k]),
                          spread_low=float(ys[k] - ys[0]), spread_high=float(ys[-1] - ys[k + 1]))
        for r in rel:
            if r['arm'] == arm and r['y'] is not None:
                r['target'] = first if r['y'] > split[arm]['threshold'] else first + 1
    return split


def s8791_releases(cfg: dict) -> tuple[list[dict], dict]:
    g = cfg['gripper']
    rel = []
    for i in episode_indices('S8791'):
        ep = read_habit_episode(i)
        ev = gripper_events(ep, g['threshold']['S8791'], g['min_frames'])
        for e in ev:
            if e['kind'] == 'open':
                rel.append(dict(episode=i, arm=e['arm'], frame=e['frame'], x=e['tcp'][0], y=e['tcp'][1]))
    xs = np.sort([r['x'] for r in rel])
    k = int(np.argmax(np.diff(xs)))
    thr = float((xs[k] + xs[k + 1]) / 2)
    for r in rel:
        r['target'] = 'near' if r['x'] < thr else 'far'
    return rel, dict(threshold=thr, gap=float(xs[k + 1] - xs[k]))


def s6760_events(cfg: dict) -> list[dict]:
    g = cfg['gripper']
    out = []
    for i in episode_indices('S6760'):
        ep = read_habit_episode(i)
        ev = gripper_events(ep, g['threshold']['S6760'], g['min_frames'])
        seq = [(e['arm'], e['kind']) for e in ev]
        ok = seq == [('robot0', 'close'), ('robot0', 'open'), ('robot1', 'close'), ('robot1', 'open')]
        r = dict(episode=i, sequence_ok=ok, human_segments=[(s['start'], s['end'], s['text'][:40]) for s in ep['human_segments']])
        if ok:
            r.update(t_robot_takes_pcb=ev[0]['frame'], t_human_has=ev[1]['frame'], t_human_ready_return=ev[2]['frame'],
                     t_in_tray=ev[3]['frame'])
        out.append(r)
    return out


def sheet(video, frames: list[int], title: str, out) -> None:
    want, got = sorted(set(frames)), {}
    with av.open(str(video)) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i in want:
                got[i] = fr.to_image().resize((320, 240))
            if len(got) == len(want):
                break
    cols = 4
    rows = (len(want) + cols - 1) // cols
    img = Image.new('RGB', (320 * cols, 240 * rows + 18))
    ImageDraw.Draw(img).text((4, 3), title, fill=(255, 255, 0))
    for j, f in enumerate(want):
        im = got.get(f)
        if im is None:
            continue
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 70, 16], fill=(0, 0, 0))
        d.text((3, 2), f't={f / 10:.1f}s', fill=(255, 255, 0))
        img.paste(im, ((j % cols) * 320, 18 + (j // cols) * 240))
    img.save(out, quality=85)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/pif/habit_labels.yaml')
    p.add_argument('--out', default='outputs/habit/label_check')
    p.add_argument('--sheets', type=int, default=0)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    cfg = yaml.safe_load(resolve(args.config).read_text())
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rel = s8789_releases(cfg)
    split = assign_s8789(rel)
    counts = {k: sum(r.get('target') == k for r in rel) for k in (1, 2, 3, 4)}
    (out / 's8789_targets.json').write_text(json.dumps(dict(split=split, counts=counts, episodes=rel), indent=1))
    print(json.dumps(dict(split=split, counts=counts, no_target=[r for r in rel if 'target' not in r]), indent=1))
    rel91, split91 = s8791_releases(cfg)
    ev60 = s6760_events(cfg)
    (out / 's8791_targets.json').write_text(json.dumps(dict(split=split91, releases=rel91), indent=1))
    (out / 's6760_events.json').write_text(json.dumps(ev60, indent=1))
    print('S8791 split', split91, {k: sum(r['target'] == k for r in rel91) for k in ('near', 'far')})
    print('S6760 sequence ok', sum(r['sequence_ok'] for r in ev60), '/', len(ev60))
    rng = np.random.default_rng(args.seed)
    for k in (1, 2, 3, 4):
        cand = [r for r in rel if r.get('target') == k and paths(r['episode'])[1].exists()]
        for r in rng.permutation(cand)[:args.sheets] if cand else []:
            fr = np.linspace(max(r['grasp'] - 40, 0), r['release'], 12).round().astype(int).tolist()
            sheet(paths(r['episode'])[1], fr, f"S8789 ep {r['episode']}: rule -> container {k} ({r['arm']}, y={r['y']:.3f}, "
                  f"grasp {r['grasp'] / 10:.1f}s, release {r['release'] / 10:.1f}s)", out / f"s8789_c{k}_ep{r['episode']}.jpg")


if __name__ == '__main__':
    main()

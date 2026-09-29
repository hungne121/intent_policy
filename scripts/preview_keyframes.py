"""Key-frame strips of recorded episodes (PNG) for a quick visual review without rerun.

For each episode: scene + wrist camera at the cue onset, the intention change (if any), shortly
after it, halfway to the end and the last frame, labelled with time, expert stage and the human's
current target.

  ./run.sh -m scripts.preview_keyframes --root outputs/datasets/hri_phase2_scenario_preview \
      --out outputs/viz/hri_phase2_scenario_preview/keyframes
"""
import argparse
import json
import numpy as np
from PIL import Image, ImageDraw
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts.lerobot_dataset_viz import to_hwc_uint8_numpy
from scripts.common import resolve

AFTER_CHANGE_S = 0.6


def stage_at(segments, i):
    return next((s['name'] for s in segments if s['start'] <= i <= s['end']), '')


def key_frames(m) -> list[tuple[int, str]]:
    n = m['n_frames']
    cue = int(round(m['human_cue_onset_t'] * 20)) if m.get('human_cue_onset_t') is not None else 0
    keys = [(min(cue, n - 1), 'cue onset' if m.get('human_cue_onset_t') is not None else 'start')]
    changes = m.get('intention_changes') or []
    if changes:
        c = changes[0]['frame']
        keys += [(max(c - 1, 0), 'just before change'), (min(c + int(AFTER_CHANGE_S * 20), n - 1), f'+{AFTER_CHANGE_S:.1f}s after change')]
        keys.append(((keys[-1][0] + n - 1) // 2, 'later'))
    else:
        keys += [(cue + (n - 1 - cue) // 3, '1/3'), (cue + 2 * (n - 1 - cue) // 3, '2/3')]
    keys.append((n - 1, 'end'))
    return keys


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', required=True)
    p.add_argument('--repo-id')
    p.add_argument('--out', required=True)
    p.add_argument('--episode-index', type=int, nargs='*')
    args = p.parse_args()
    root, out = resolve(args.root), resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    repo_id = args.repo_id or f'local/{root.name}'
    metas = [json.loads(line) for line in (root / 'meta/hri_episodes.jsonl').read_text().splitlines()]
    for m in metas:
        ep = m['episode_index']
        if args.episode_index and ep not in args.episode_index:
            continue
        ds = LeRobotDataset(repo_id, root=root, episodes=[ep])
        cams = ds.meta.camera_keys
        tiles = []
        for i, label in key_frames(m):
            item = ds[i]
            imgs = [to_hwc_uint8_numpy(item[c]) for c in cams]
            tile = Image.fromarray(np.concatenate(imgs, axis=0))
            bar = Image.new('RGB', (tile.width, 58), (20, 20, 20))
            d = ImageDraw.Draw(bar)
            target = stage_at(m.get('human_target_stages') or [], i)
            d.text((4, 2), f"{label}  t={i / 20:.2f}s  frame {i}", fill=(255, 220, 90))
            d.text((4, 20), f"expert: {stage_at(m['expert_stages'], i)}", fill=(150, 200, 255))
            d.text((4, 38), f"human target: {target}", fill=(255, 170, 60))
            canvas = Image.new('RGB', (tile.width, tile.height + bar.height))
            canvas.paste(bar, (0, 0))
            canvas.paste(tile, (0, bar.height))
            tiles.append(canvas)
        strip = Image.new('RGB', (sum(t.width for t in tiles) + 4 * (len(tiles) - 1), tiles[0].height), (255, 255, 255))
        x = 0
        for t in tiles:
            strip.paste(t, (x, 0))
            x += t.width + 4
        split = m.get('split', 'default')
        path = out / f"episode_{ep:03d}_{m['scenario_id']}_{split}_seed{m['seed']}.png"
        strip.save(path)
        print(path, flush=True)


if __name__ == '__main__':
    main()

"""Build the step-2 training sets (07/QĐ-HL B, 03/TTr-HL #3-#5) from collection shards or one collection root.

Input: `--src` paths, each either ONE collection root (holds hrc_observed / hrc_forecast, e.g. the preview root
outputs/diag/preview_d1_1118/ds) or a folder of shard roots (hrc-ds-0.5: one root per shard `<group>_<task>_<split>_<seed>
[_<fault>]`, finished = `DONE` marker; workstation/rent/run_collect_list.sh). Kept: grade-A episodes (66/QĐ-DL, grade
re-derived from the stored summaries) of the groups `--groups` (default N1 N5 N6; N5 exists only in hrc_forecast).
The split is the recorded one, checked against the seed ranges of configs/hrc/collection.yaml; a test seed (300900-
300999) or a split / seed mismatch stops the build.

Output (`--out`), with the LeRobot API of the installed version (`delete_episodes`, `merge_datasets`):
  D_observed/{train,val}   all observed A episodes           (LeRobot roots, meta/hrc_episodes.jsonl in merged order)
  D_forecast/{train,val}   all forecast A episodes incl. N5
  manifest.jsonl           one row per kept episode: dataset, split, merged_episode_index, episode_id, task, variant,
                           group, seed, grade, dataset_version, schema_version, source_shard, source_episode_index, frames
  build_report.json        counts per dataset / split / task / variant / group, dropped episodes with the reason
Normalisation: LeRobot's `delete_episodes` keeps the SOURCE stats (incl. the deleted episodes) and `merge_datasets`
aggregates those (lerobot datasets/dataset_tools.py `_copy_episodes_metadata_and_stats`, aggregate.py), so each output
root gets its stats.json recomputed from its own episodes' per-episode stats (meta/episodes, written at record time by
compute_episode_stats) with LeRobot's `aggregate_stats`; train therefore has train-only statistics
(`meta/stats_source.json` records this).

    ./run.sh -m tools.build_train_set --src outputs/datasets_hrc_pod/shards_d1 --out outputs/datasets_hrc_train/d1
    ./run.sh -m tools.build_train_set --src outputs/diag/preview_d1_1118/ds --out <tmp> --debug-holdout-val 2   (smoke)
Training uses this output only after the user approved the data (CLAUDE.md: BB + QĐ; 03/TTr-HL #7: pilot only).
"""
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from intent_policy.data.episode_quality import observable

ROOT = Path(__file__).resolve().parents[1]
MODES = {'observed': 'D_observed', 'forecast': 'D_forecast'}


def episode_grade(shard, meta):
    """66/QĐ-DL grade of one recorded episode (A as recorded, or a non-A whose only violations are unobservable).
    An N5 episode whose injected forecast failed the injector's own consistency check (reason "fault_checks [...]":
    start_tol / jump_tol / bone_tol of configs/hrc/collection.yaml n5.checks) is never A (lead, 18/11)."""
    if any('fault_checks' in str(r) for r in meta.get('reject_reasons') or []):
        return 'C'
    if meta.get('quality_grade') == 'A':
        return 'A'
    f = shard / 'rejected_episodes' / meta['episode_id'] / 'summary.json'
    if not f.exists():
        return meta.get('quality_grade', 'C')
    s = json.loads(f.read_text())
    bad = (not s['success'] or any('fault_checks' in str(r) for r in s.get('reject_reasons', [])) or not s.get('robot_limits_ok', True)
           or any('start pose' in r or 'frames' in r for r in s.get('reject_reasons', []))
           or any(v.get('violation') and observable(v) for v in s.get('contact_classes', {}).values()))
    return 'C' if bad else 'A'


def split_of_seed(seed: int, splits: dict) -> str | None:
    for name, (lo, hi) in splits.items():
        if lo <= seed <= hi:
            return name
    return None


def is_root(p: Path) -> bool:
    return any((p / f'hrc_{m}' / 'meta' / 'info.json').exists() for m in MODES)


def find_roots(srcs, require_done: bool) -> list[Path]:
    """Collection roots: a --src that is a root itself, else its sub-folders that are roots (shards: DONE required)."""
    out = []
    for s in srcs:
        if is_root(s):
            out.append(s)
            continue
        for d in sorted(p for p in s.iterdir() if p.is_dir()):
            if is_root(d) and (not require_done or (d / 'DONE').exists()):
                out.append(d)
    return out


def read_metas(root: Path) -> dict[int, dict]:
    side = root / 'meta' / 'hrc_episodes.jsonl'
    return {m['episode_index']: m for m in (json.loads(l) for l in side.read_text().splitlines() if l.strip())}


def episode_lengths(meta) -> dict[int, int]:
    ep = meta.episodes
    return {int(i): int(t) - int(f) for i, f, t in zip(ep['episode_index'], ep['dataset_from_index'], ep['dataset_to_index'])}


def episode_stats_from_meta(root: Path, features: dict) -> list[dict]:
    """Per-episode stats {feature: {stat: array}} from the meta/episodes parquet files (columns stats/<feature>/<stat>,
    written at record time by compute_episode_stats; LeRobotDatasetMetadata.episodes does not load them)."""
    import pyarrow.parquet as pq
    out = []
    for f in sorted((root / 'meta' / 'episodes').glob('*/*.parquet')):
        t = pq.read_table(f)
        cols = [c for c in t.column_names if c.startswith('stats/')]
        idx = t.column('episode_index').to_pylist()
        rows = t.select(cols).to_pylist()
        for e, row in sorted(zip(idx, rows), key=lambda x: x[0]):
            st = {}
            for c in cols:
                _, feat, stat = c.split('/', 2)
                if feat in ('index', 'episode_index', 'frame_index', 'timestamp', 'task_index') or row[c] is None:
                    continue
                st.setdefault(feat, {})[stat] = np.asarray(row[c], dtype=np.float64)
            out.append(st)
    if not out or not all(out):
        raise ValueError(f'{root}: per-episode stats missing in meta/episodes')
    return out


def recompute_root_stats(root: Path, repo_id: str) -> dict:
    """stats.json of `root` = aggregate_stats over its own episodes (train: train-only normalisation)."""
    from lerobot.datasets.compute_stats import aggregate_stats
    from lerobot.datasets.io_utils import write_stats
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    meta = LeRobotDatasetMetadata(repo_id, root=root)
    stats = aggregate_stats(episode_stats_from_meta(root, meta.features))
    if 'observation.state' not in stats:
        raise ValueError(f'{root}: recomputed stats lack observation.state')
    write_stats(stats, root)
    (root / 'meta' / 'stats_source.json').write_text(json.dumps(dict(
        method='lerobot.datasets.compute_stats.aggregate_stats over meta/episodes per-episode stats of this root only',
        episodes=meta.total_episodes, frames=meta.total_frames), indent=1))
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src', type=Path, nargs='+', required=True, help='collection roots or folders of shard roots')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--groups', nargs='+', default=['N1', 'N5', 'N6'])
    p.add_argument('--tasks', nargs='*', default=None, help='keep only these tasks (default all)')
    p.add_argument('--splits', nargs='+', default=['train', 'val'], help='never test (03/TTr-HL #4)')
    p.add_argument('--collection-config', type=Path, default=ROOT / 'configs/hrc/collection.yaml')
    p.add_argument('--no-require-done', action='store_true', help='also take shard roots without a DONE marker')
    p.add_argument('--limit-roots', type=int, default=None, help='test on the first N roots only')
    p.add_argument('--debug-holdout-val', type=int, default=0,
                   help='SMOKE ONLY: move the last N kept train episodes of each dataset to val (manifest '
                        'split_note=debug_holdout); for the single-split preview root')
    a = p.parse_args()
    from lerobot.datasets.dataset_tools import delete_episodes, merge_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    if 'test' in a.splits:
        raise SystemExit('the test split is never built for training (03/TTr-HL #4)')
    coll = yaml.safe_load(a.collection_config.read_text())
    seed_splits = {k: tuple(v) for k, v in coll['splits'].items()}
    roots = find_roots(a.src, require_done=not a.no_require_done)[:a.limit_roots]
    if not roots:
        raise SystemExit(f'no collection root under {a.src}')
    if a.out.exists() and any(a.out.iterdir()):
        raise SystemExit(f'{a.out} is not empty')
    tmp = a.out / '_filtered'
    tmp.mkdir(parents=True, exist_ok=True)
    selected = {}                     # (dataset, split) -> [(root, mode, [source episode indices], metas)]
    dropped, errors = [], []
    for root in roots:
        for mode, dname in MODES.items():
            lr = root / f'hrc_{mode}'
            if not (lr / 'meta' / 'info.json').exists():
                continue
            metas = read_metas(lr)
            by_split = {}
            for i, m in sorted(metas.items()):
                why = None
                seed_split = split_of_seed(int(m['seed']), seed_splits)
                if seed_split != m['split']:
                    errors.append(f'{root.name}/{mode} ep {i} {m["episode_id"]}: split {m["split"]} but seed {m["seed"]} is {seed_split}')
                    continue
                if m['split'] not in a.splits:
                    why = f'split {m["split"]}'
                elif m['group'] not in a.groups:
                    why = f'group {m["group"]}'
                elif a.tasks and m['task_name'] not in a.tasks:
                    why = f'task {m["task_name"]}'
                elif m.get('expert_mode', m.get('mode')) != mode:
                    why = f'mode {m.get("expert_mode")} in hrc_{mode}'
                else:
                    g = episode_grade(root, m)
                    m['_grade'] = g
                    if g != 'A':
                        why = f'grade {g}'
                if why:
                    dropped.append(dict(source_shard=root.name, mode=mode, episode_index=i, episode_id=m['episode_id'], reason=why))
                else:
                    by_split.setdefault(m['split'], []).append(i)
            for split, keep in by_split.items():
                selected.setdefault((dname, split), []).append((root, mode, keep, metas))
    if errors:
        raise SystemExit('split / seed mismatch:\n' + '\n'.join(errors[:20]))
    if a.debug_holdout_val:
        for dname in MODES.values():
            items = selected.get((dname, 'train'), [])
            flat = [(k, i) for k, (_, _, keep, _) in enumerate(items) for i in keep]
            move = flat[-a.debug_holdout_val:]
            for k, i in move:
                items[k][2].remove(i)
            for k, i in move:
                r, mode, _, metas = items[k]
                selected.setdefault((dname, 'val'), []).append((r, mode, [i], metas))
                metas[i]['_split_note'] = 'debug_holdout'
            selected[(dname, 'train')] = [x for x in items if x[2]]

    manifest, report = [], {}
    for (dname, split), items in sorted(selected.items()):
        sources, rows = [], []
        for root, mode, keep, metas in items:
            lr = root / f'hrc_{mode}'
            rid = f'local/{root.name}_{mode}'
            ds = LeRobotDataset(rid, root=lr)
            lengths = episode_lengths(ds.meta)
            drop = [i for i in range(ds.meta.total_episodes) if i not in keep]
            if drop:
                out = tmp / f'{root.name}_{mode}_{split}_{len(sources)}'
                shutil.rmtree(out, ignore_errors=True)
                ds = delete_episodes(ds, drop, output_dir=out, repo_id=f'{rid}_{split}_A')
            sources.append(ds)
            for i in sorted(keep):
                m = metas[i]
                rows.append(dict(dataset=dname, split=split, episode_id=m['episode_id'], task=m['task_name'],
                                 variant=m['task_variant'], group=m['group'], seed=int(m['seed']), grade=m['_grade'],
                                 dataset_version=m.get('dataset_version'), schema_version=m.get('schema_version'),
                                 fault_type=m.get('fault_type'), source_shard=root.name, source_mode=mode,
                                 source_episode_index=i, frames=lengths[i],
                                 **({'split_note': m['_split_note']} if m.get('_split_note') else {}), _meta=m))
        out = a.out / dname / split
        rid = f'local/{dname}_{split}'
        merged = merge_datasets(sources, output_repo_id=rid, output_dir=out)
        got = episode_lengths(merged.meta)
        if [got[k] for k in range(len(rows))] != [r['frames'] for r in rows] or len(got) != len(rows):
            raise SystemExit(f'{dname}/{split}: merged episode lengths differ from the sources (episode order?)')
        recompute_root_stats(out, rid)
        with (out / 'meta' / 'hrc_episodes.jsonl').open('w') as f:
            for k, r in enumerate(rows):
                m = r.pop('_meta')
                r['merged_episode_index'] = k
                f.write(json.dumps(dict({kk: v for kk, v in m.items() if not kk.startswith('_')}, episode_index=k,
                                        source_episode_index=r['source_episode_index'], source_shard=r['source_shard'],
                                        merged_episode_index=k, grade=r['grade'])) + '\n')
                manifest.append(r)
        report[f'{dname}/{split}'] = dict(
            episodes=merged.meta.total_episodes, frames=merged.meta.total_frames,
            by_task=dict(Counter(r['task'] for r in rows)), by_group=dict(Counter(r['group'] for r in rows)),
            by_task_variant=dict(Counter(f'{r["task"]}/{r["variant"]}' for r in rows)),
            frames_by_task={t: sum(r['frames'] for r in rows if r['task'] == t) for t in sorted({r['task'] for r in rows})})
    shutil.rmtree(tmp, ignore_errors=True)
    with (a.out / 'manifest.jsonl').open('w') as f:
        for r in manifest:
            f.write(json.dumps(r) + '\n')
    (a.out / 'build_report.json').write_text(json.dumps(dict(
        src=[str(s) for s in a.src], roots=len(roots), groups=a.groups, tasks=a.tasks, splits=a.splits,
        debug_holdout_val=a.debug_holdout_val, outputs=report, dropped=dropped,
        dropped_by_reason=dict(Counter(d['reason'] for d in dropped))), indent=1))
    print(json.dumps(dict(outputs={k: {kk: v[kk] for kk in ('episodes', 'frames', 'by_group')} for k, v in report.items()},
                          dropped_by_reason=dict(Counter(d['reason'] for d in dropped))), indent=1))


if __name__ == '__main__':
    main()

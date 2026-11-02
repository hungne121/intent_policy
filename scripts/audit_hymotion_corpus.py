"""Inventory raw HY-Motion clips; never mark them scene-valid or held-out.

Keeps both hands and content identities. Existing v6 bank may have contributed to
ManiCast fine-tuning, so this command deliberately does not invent a test split.
"""
import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter
import numpy as np
from intent_policy.sim.motion_bank import clip_id, palm_path, segment_ranges, FPS


def inventory(folder):
    rows = []
    for f in sorted(Path(folder).glob('*.npz')):
        with np.load(f, allow_pickle=False) as z:
            if 'keypoints3d' not in z:
                continue
            kp = z['keypoints3d']
            if kp.ndim != 4 or kp.shape[2:] != (52, 3):
                raise ValueError(f'{f}: expected [B,T,52,3], got {kp.shape}')
            fps = float(z['fps']) if 'fps' in z else FPS
            seeds = z['seeds'].tolist() if 'seeds' in z else [None] * len(kp)
            text = str(z['text']) if 'text' in z else None
            model = str(z['model']) if 'model' in z else None
            splits = z['sample_splits'].tolist() if 'sample_splits' in z else [None] * len(kp)
            if len(seeds) != len(kp) or len(splits) != len(kp):
                raise ValueError(f'{f}: seed and split metadata must match sample count')
        digest = hashlib.sha256(f.read_bytes()).hexdigest()
        for i, x in enumerate(kp):
            finite = bool(np.isfinite(x).all())
            metrics = {}
            if finite and len(x) >= 2:
                for hand in ('left', 'right'):
                    p = palm_path(x, hand)
                    speed = np.linalg.norm(np.diff(p, axis=0), axis=1) * fps
                    metrics[hand] = dict(path_length_m=float(speed.sum() / fps),
                        speed_max_m_s=float(speed.max()), speed_p95_m_s=float(np.quantile(speed, .95)),
                        segments=[list(v) for v in segment_ranges(p)])
            rows.append(dict(clip_id=clip_id(x), file=str(f), file_sha256=digest, sample_index=i,
                generation_seed=seeds[i], prompt=text, model=model, fps=fps, frames=len(x),
                sample_split=splits[i],
                finite=finite, hands=metrics, scene_validated=False,
                forecast_training_exposure='unknown_treat_as_exposed'))
    if not rows:
        raise ValueError('No HY-Motion clips found')
    counts = Counter(r['clip_id'] for r in rows)
    return dict(scope='raw_inventory_not_scene_validation', files=len(set(r['file'] for r in rows)),
        samples=len(rows), unique_clip_ids=len(counts), finite_samples=sum(r['finite'] for r in rows),
        duplicate_clip_ids={k: v for k, v in counts.items() if v > 1}, clips=rows)


def split_manifest(report):
    """Use only explicit per-sample assignments; reject ambiguous or invalid clips."""
    assignments = {}
    for row in report['clips']:
        split = row['sample_split']
        if split not in ('train', 'val', 'test') or not row['finite']:
            raise ValueError(f"Invalid split or skeleton for {row['file']} sample {row['sample_index']}")
        previous = assignments.setdefault(row['clip_id'], split)
        if previous != split:
            raise ValueError(f"Clip {row['clip_id']} occurs in both {previous} and {split}")
    return dict(scope='raw_clip_assignments_not_scene_validation', clip_splits=assignments)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--folder', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--split-output', help='Write explicit clip-level split assignments')
    a = ap.parse_args()
    report = inventory(a.folder)
    out = Path(a.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    if a.split_output:
        manifest = split_manifest(report)
        dest = Path(a.split_output)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k != 'clips'}))


if __name__ == '__main__':
    main()

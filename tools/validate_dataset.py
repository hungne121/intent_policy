"""Read an HRC dataset back with LeRobotDataset and check it (QĐ 53/QĐ-DL §8, §9): features match the fixed schema,
the sidecar lists every episode, frame counts, finite values, forecast mask semantics, one decoded video frame.

    ./run.sh -m tools.validate_dataset --mode observed
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from intent_policy.shared.sim_env import ROOT
from intent_policy.data.dataset_schema import features

CFG = yaml.safe_load((ROOT / 'configs/hrc/collection.yaml').read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=CFG['modes'], required=True)
    p.add_argument('--root', type=Path, default=ROOT / CFG['datasets']['root'])
    a = p.parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    name = CFG['datasets'][a.mode]
    ds = LeRobotDataset(f'local/{name}', root=a.root / name)
    errors = []
    expected = features(CFG['cameras'], CFG['image_hw'])
    for k, spec in expected.items():
        if k not in ds.meta.features:
            errors.append(f'missing feature {k}')
        elif tuple(ds.meta.features[k]['shape']) != tuple(spec['shape']):
            errors.append(f'shape {k}: {ds.meta.features[k]["shape"]} != {spec["shape"]}')
    side = [json.loads(l) for l in (a.root / name / 'meta/hrc_episodes.jsonl').read_text().splitlines() if l.strip()]
    if len(side) != ds.meta.total_episodes:
        errors.append(f'sidecar {len(side)} episodes != dataset {ds.meta.total_episodes}')
    for e in side:
        if e['expert_mode'] != a.mode:
            errors.append(f"{e['episode_id']}: mode {e['expert_mode']} in {name}")
        if a.mode == 'observed' and e['forecast_validity_summary']['used_by_expert'] != 0:
            errors.append(f"{e['episode_id']}: forecast handed to the observed expert")
    # 59/QĐ-DL: forecast_provided semantics, sidecar per episode, N5 effective vs raw forecast
    sidecar = a.root / name / 'meta/hrc_sidecar'
    fault_rows = []
    for e in side:
        f = sidecar / f"episode_{e['episode_index']:06d}.npz"
        if not f.exists():
            errors.append(f"{e['episode_id']}: sidecar missing"); continue
        z = np.load(f)
        idx = [i for i in range(ds.meta.episodes[e['episode_index']]['dataset_from_index'],
                                ds.meta.episodes[e['episode_index']]['dataset_to_index'])]
        if len(z['raw_forecast']) != len(idx):
            errors.append(f"{e['episode_id']}: sidecar frames {len(z['raw_forecast'])} != {len(idx)}"); continue
        cols = ds.hf_dataset.select(idx).with_format('numpy')
        eff = np.asarray(cols['observation.human_forecast']).reshape(len(idx), 10, 6, 3)
        prov = np.asarray(cols['observation.forecast_provided']).reshape(-1)
        valid = np.asarray(cols['observation.forecast_valid']).reshape(-1)
        if a.mode == 'observed' and prov.any():
            errors.append(f"{e['episode_id']}: forecast_provided set in the observed dataset")
        if a.mode == 'forecast' and (prov > valid).any():
            errors.append(f"{e['episode_id']}: forecast provided while invalid")
        k = z['fault_strength']; both = (z['raw_forecast_valid'] == 1) & (valid == 1)
        diff = np.abs(eff - z['raw_forecast']).reshape(len(idx), -1).max(1)
        outside = both & (k == 0); inside = both & (k > 0)
        if outside.any() and diff[outside].max() > 1e-5:
            errors.append(f"{e['episode_id']}: effective != raw forecast outside the fault window ({diff[outside].max():.2e})")
        if e.get('fault_type'):
            fault_rows.append({'episode_id': e['episode_id'], 'fault_frames': int(inside.sum()),
                               'max_effective_minus_raw_m': round(float(diff[inside].max()) if inside.any() else 0., 4)})
            if not inside.any():
                errors.append(f"{e['episode_id']}: fault episode without a faulted frame")
    n = len(ds)
    for i in sorted({0, n // 2, n - 1}):
        item = ds[i]
        for k in ('observation.state', 'action', 'observation.human_forecast', 'observation.human_pose'):
            if not np.isfinite(np.asarray(item[k])).all():
                errors.append(f'non-finite {k} at frame {i}')
        if float(item['observation.forecast_valid']) == 0 and np.abs(np.asarray(item['observation.human_forecast'])).sum() != 0:
            errors.append(f'invalid forecast not zero-masked at frame {i}')
        for c in CFG['cameras']:
            img = item[f'observation.images.{c}']
            if tuple(img.shape[-2:]) != tuple(CFG['image_hw']):
                errors.append(f'image {c} shape {tuple(img.shape)}')
    print(json.dumps({'dataset': name, 'episodes': ds.meta.total_episodes, 'frames': n,
                      'tasks': sorted({e['task_name'] for e in side}), 'groups': sorted({e.get('group', '?') for e in side}),
                      'fault_episodes': fault_rows, 'errors': errors}, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == '__main__':
    main()

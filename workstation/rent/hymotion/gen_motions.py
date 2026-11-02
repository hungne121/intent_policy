"""Generate the v6 human motion library with HY-Motion 1.0 (run inside a HY-Motion-1.0 checkout on the rented GPU).

Uses the repo's own runtime (hymotion.utils.t2m_runtime.T2MRuntime, as local_infer.py) without the optional prompt
rewriter; for every prompt of the config saves one npz with the pipeline's own output, which the repo's NPZ export
does not keep: keypoints3d (B, L, 52, 3) SMPL-H joints in metres (Y up, feet on the ground, HY-Motion axes), rot6d,
transl, root rotations, plus the text, seeds and fps (30). Resumable: prompts with an npz are skipped.
  python gen_motions.py --config hymotion_v6.yaml --model ckpts/tencent/HY-Motion-1.0-Lite --out out/
"""
import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import yaml

def main():
    from hymotion.utils.t2m_runtime import T2MRuntime
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rt = None  # a completed resume must not reload GPU weights
    rng = random.Random(int(cfg.get('generation_seed', 0)))
    seed_range = cfg.get('seed_range', [0, 999])
    sample_splits = cfg.get('sample_splits')
    if sample_splits is not None and len(sample_splits) != int(cfg['seeds']):
        raise ValueError('sample_splits must assign every generated sample')
    log = json.loads((out / 'gen_log.json').read_text()) if (out / 'gen_log.json').exists() else []
    for key, p in cfg['prompts'].items():
        f = out / f'{key}.npz'
        seeds = (rng.sample(range(int(seed_range[0]), int(seed_range[1]) + 1), int(cfg['seeds']))
                 if 'seed_range' in cfg else [rng.randint(0, 999) for _ in range(int(cfg['seeds']))])
        batch_size = int(cfg.get('generation_batch_size', len(seeds)))
        if batch_size <= 0:
            raise ValueError('generation_batch_size must be positive')
        plan_hash = hashlib.sha256(json.dumps(dict(prompt=p, seeds=seeds, model=cfg['model'],
            cfg_scale=cfg['cfg_scale'], sample_splits=sample_splits, batch_size=batch_size), sort_keys=True).encode()).hexdigest()
        if f.exists():
            with np.load(f, allow_pickle=False) as existing:
                if str(existing['text']) != p['text'] or existing['seeds'].tolist() != seeds:
                    raise ValueError(f'{f}: existing content does not match this generation plan; use a new output folder')
                if sample_splits is not None and ('sample_splits' not in existing or existing['sample_splits'].tolist() != sample_splits):
                    raise ValueError(f'{f}: split assignment mismatch')
                if sample_splits is not None and ('generation_plan_sha256' not in existing or str(existing['generation_plan_sha256']) != plan_hash):
                    raise ValueError(f'{f}: generation settings mismatch')
            continue
        t0 = time.time()
        if rt is None:
            rt = T2MRuntime(config_path=str(Path(args.model) / 'config.yml'), ckpt_name=str(Path(args.model) / 'latest.ckpt'),
                            disable_prompt_engineering=True)
        generated = {k: [] for k in ('keypoints3d', 'rot6d', 'transl', 'root_rotations_mat')}
        for start in range(0, len(seeds), batch_size):
            _, _, mo = rt.generate_motion(text=p['text'], seeds_csv=','.join(map(str, seeds[start:start + batch_size])),
                duration=float(p['d']), cfg_scale=float(cfg['cfg_scale']), output_format='dict', original_text=p['text'],
                output_dir=str(out / 'repo_npz'), output_filename=f'{key}_{start}')
            for name in generated:
                generated[name].append(mo[name].detach().cpu().numpy().astype(np.float32))
        arrays = {k: np.concatenate(v) for k, v in generated.items()}
        np.savez_compressed(f, **arrays,
                            text=np.array(p['text']), seeds=np.array(seeds), fps=np.array(30), model=np.array(cfg['model']),
                            generation_plan_sha256=np.array(plan_hash),
                            **({'sample_splits': np.array(sample_splits)} if sample_splits is not None else {}))
        row = dict(key=key, seconds=round(time.time() - t0, 1), shape=list(arrays['keypoints3d'].shape), seeds=seeds)
        log.append(row)
        print(json.dumps(row), flush=True)
    (out / 'gen_log.json').write_text(json.dumps(log, indent=1))
    print('GEN_DONE', flush=True)


if __name__ == '__main__':
    main()

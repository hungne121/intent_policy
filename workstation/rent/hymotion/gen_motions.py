"""Generate the v6 human motion library with HY-Motion 1.0 (run inside a HY-Motion-1.0 checkout on the rented GPU).

Uses the repo's own runtime (hymotion.utils.t2m_runtime.T2MRuntime, as local_infer.py) without the optional prompt
rewriter; for every prompt of the config saves one npz with the pipeline's own output, which the repo's NPZ export
does not keep: keypoints3d (B, L, 52, 3) SMPL-H joints in metres (Y up, feet on the ground, HY-Motion axes), rot6d,
transl, root rotations, plus the text, seeds and fps (30). Resumable: prompts with an npz are skipped.
  python gen_motions.py --config hymotion_v6.yaml --model ckpts/tencent/HY-Motion-1.0-Lite --out out/
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import yaml

from hymotion.utils.t2m_runtime import T2MRuntime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rt = T2MRuntime(config_path=str(Path(args.model) / 'config.yml'), ckpt_name=str(Path(args.model) / 'latest.ckpt'),
                    disable_prompt_engineering=True)
    rng = random.Random(0)
    log = []
    for key, p in cfg['prompts'].items():
        f = out / f'{key}.npz'
        seeds = [rng.randint(0, 999) for _ in range(int(cfg['seeds']))]
        if f.exists():
            continue
        t0 = time.time()
        _, _, mo = rt.generate_motion(text=p['text'], seeds_csv=','.join(map(str, seeds)), duration=float(p['d']),
                                      cfg_scale=float(cfg['cfg_scale']), output_format='dict', original_text=p['text'],
                                      output_dir=str(out / 'repo_npz'), output_filename=key)
        np.savez_compressed(f, keypoints3d=mo['keypoints3d'].numpy().astype(np.float32),
                            rot6d=mo['rot6d'].numpy().astype(np.float32), transl=mo['transl'].numpy().astype(np.float32),
                            root_rotations_mat=mo['root_rotations_mat'].numpy().astype(np.float32),
                            text=np.array(p['text']), seeds=np.array(seeds), fps=np.array(30), model=np.array(cfg['model']))
        row = dict(key=key, seconds=round(time.time() - t0, 1), shape=list(mo['keypoints3d'].shape), seeds=seeds)
        log.append(row)
        print(json.dumps(row), flush=True)
    (out / 'gen_log.json').write_text(json.dumps(log, indent=1))
    print('GEN_DONE', flush=True)


if __name__ == '__main__':
    main()

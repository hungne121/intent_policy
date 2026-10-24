"""Keep, per training run, the checkpoint with the lowest val loss (ACT: Zhao et al. 2023) plus checkpoints/last; drop
the optimizer state. Val loss is logged every log_every x 10 steps, checkpoints every --save-every-epochs: a checkpoint
gets the first val loss logged at or after its step (the final step always has one).
  python select_ckpt.py <run folder holding A/ B/ D/ B25/ ...>
"""
import json
import shutil
import sys
from pathlib import Path

run = Path(sys.argv[1])
for r in sorted(p for p in run.iterdir() if p.is_dir() and (p / 'train_log.jsonl').exists()):
    rows = [json.loads(l) for l in (r / 'train_log.jsonl').read_text().splitlines()]
    val = sorted((x['step'], x['val_loss']) for x in rows if 'val_loss' in x)
    ckpts = {int(p.name): p for p in (r / 'checkpoints').glob('[0-9]*')}
    ck_val = {s: next(v for vs, v in val if vs >= s) for s in ckpts if any(vs >= s for vs, _ in val)}
    best = min(ck_val, key=ck_val.get)
    for s, p in ckpts.items():                   # checkpoints/last = the final model (2000 epochs)
        if s != best:
            shutil.rmtree(p)
        else:
            (p / 'training_state.pt').unlink(missing_ok=True)
    (r / 'best.json').write_text(json.dumps(dict(best_step=best, val_loss=ck_val[best], val_by_ckpt=ck_val,
                                                 val_logged=dict(val)), indent=1))
    print(r.name, 'best step', best, 'val', round(ck_val[best], 4), 'last', max(ckpts), 'val', round(ck_val[max(ckpts)], 4))

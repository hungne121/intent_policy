"""12/QĐ-HL: fetch one model's checkpoint at step K for closed-loop eval, from HF (the pod backs up its latest complete
numbered checkpoint as <run>/<M>/checkpoints/resume every 5 min; COMPLETE.json carries the step). Policy weights only,
no optimizer state. Prints FETCHED <step> / NOT_YET <step> / MISSED <step>.
    ./run.sh workstation/rent/fetch_ckpt.py <run> <model> <step> <out_dir>"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

run, model, step, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
REPO = 'Hungdd88/hrc-train-20261118'
tok = Path.home().joinpath('.cache/huggingface/token').read_text().strip()
base = f'{run}/{model}/checkpoints/resume'
with tempfile.TemporaryDirectory(dir='/tmp/claude-1000') as td:
    # the step of each model's `resume` copy is in the run-level backup manifest (COMPLETE.json holds file hashes)
    meta = json.loads(Path(hf_hub_download(REPO, f'{run}/backup_manifest.json', token=tok, local_dir=td,
                                           force_download=True)).read_text())
    s = int(meta['steps'][model])
    if s != step:
        print('NOT_YET' if s < step else 'MISSED', s)
        sys.exit(0)
    snapshot_download(REPO, allow_patterns=[f'{base}/pretrained_model/*', f'{base}/COMPLETE.json',
                                            f'{run}/{model}/run_meta.json'],
                      token=tok, local_dir=td)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(Path(td) / base / 'pretrained_model', out / 'pretrained_model', dirs_exist_ok=True)
    rm = Path(td) / run / model / 'run_meta.json'
    if rm.exists():
        shutil.copy(rm, out.parents[1] / 'run_meta.json')
    shutil.copy(Path(td) / base / 'COMPLETE.json', out / 'COMPLETE.json')   # last: marks the fetch complete
    print('FETCHED', s)

"""Upload a consistent snapshot including resumable optimizer state; read back the commit."""
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from tools.train_checkpoint_io import stage_run, sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--repo', required=True)
    p.add_argument('--final', action='store_true')
    a = p.parse_args()
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get('HF_TOKEN'))
    api.create_repo(a.repo, repo_type='model', private=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='train-backup-') as tmp:
        stage = Path(tmp)
        steps = stage_run(a.run, stage, a.final)
        if not steps:
            raise RuntimeError('no complete resumable checkpoint; refusing to report backup success')
        if a.final and set(steps) != {'M0', 'M1-noH', 'M1'}:
            raise RuntimeError('final backup requires all three models')
        hashes = {str(f.relative_to(stage)): sha256(f) for f in stage.rglob('*') if f.is_file()}
        (stage / 'backup_manifest.json').write_text(json.dumps(dict(steps=steps, sha256=hashes), indent=2))
        commit = api.upload_folder(repo_id=a.repo, repo_type='model', folder_path=stage,
                                   path_in_repo=a.run.name, commit_message=f'{a.run.name} resume {steps}')
        info = api.model_info(a.repo, revision=commit.oid, files_metadata=True)
        files = {f.rfilename: f for f in info.siblings}
        for rel, digest in hashes.items():
            local = stage / rel
            remote = files[f'{a.run.name}/{rel}']
            if remote.size != local.stat().st_size:
                raise RuntimeError(f'remote size mismatch: {rel}')
            lfs = getattr(remote, 'lfs', None)
            if lfs and lfs.sha256 != digest:
                raise RuntimeError(f'remote checksum mismatch: {rel}')
            if not lfs:
                data = local.read_bytes()
                blob = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest()
                if remote.blob_id != blob:
                    raise RuntimeError(f'remote Git checksum mismatch: {rel}')
        receipt = dict(repo=a.repo, revision=commit.oid, steps=steps, files=len(hashes), verified=True)
        (a.run / 'backup_receipt.json').write_text(json.dumps(receipt, indent=2))
        print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()

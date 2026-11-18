"""Pod side (64-66/QĐ-DL): upload every finished shard root (DONE marker) as a tar to a private HF dataset repo, plus the
progress / logs. All files of one round go in ONE commit (HF allows 128 commits / hour; one commit per shard hit the
limit on 17/11). `loop`: every 3 min; `final`: once. Leftover LeRobot temp images of an interrupted episode are skipped."""
import os, sys, tarfile, time
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd

R = Path(os.environ.get('OUT_ROOT', '/root/out/hrc_collect')); PREFIX = os.environ.get('HF_PREFIX', 'shards')
REPO = os.environ['HF_REPO']; api = HfApi(token=os.environ['HF_TOKEN'])
api.create_repo(REPO, repo_type='dataset', private=True, exist_ok=True)


def upload_once():
    ops, done = [], []
    for root in sorted(R.glob('*')):
        if not (root / 'DONE').exists() or (root / '.uploaded').exists():
            continue
        tar = Path('/root/out') / f'{root.name}.tar'
        with tarfile.open(tar, 'w') as t:
            t.add(root, arcname=root.name, filter=lambda ti: None if '/images/' in ti.name else ti)
        ops.append(CommitOperationAdd(path_in_repo=f'{PREFIX}/{root.name}.tar', path_or_fileobj=str(tar))); done.append((root, tar))
    for f in [x for x in Path('/root/out').glob('*.txt')] + [x for x in Path('/root/out').glob('uploader*.log')]:
        ops.append(CommitOperationAdd(path_in_repo=f'meta/{f.name}', path_or_fileobj=str(f)))
    if not done:
        return
    api.create_commit(repo_id=REPO, repo_type='dataset', operations=ops, commit_message=f'{PREFIX}: {len(done)} shards')
    for root, tar in done:
        tar.unlink(); (root / '.uploaded').touch()
    print(time.strftime('%T'), 'uploaded', len(done), 'shards', flush=True)


if sys.argv[1] == 'loop':
    while True:
        try:
            upload_once()
        except Exception as e:
            print('upload error', e, flush=True)
        time.sleep(180)
else:
    upload_once()

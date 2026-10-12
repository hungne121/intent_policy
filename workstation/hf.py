"""Transfers between the laptop and the workstation through the private Hugging Face dataset repo ($HF_REPO).

  python workstation/hf.py pull <path in repo> <local dir>     a folder of the repo -> local dir (replaced)
  python workstation/hf.py push <local dir> <path in repo> [--exclude GLOB ...]
  python workstation/hf.py ls [<prefix>]

The repo holds inputs/ (the replay dataset intent_act_late_v5, DeePoint's released weights) and runs/<job>/<date>/
(each job's outputs). Needs `hf auth login` once with a token that can write to the repo.
"""
import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

REPO = os.environ.get('HF_REPO', 'Hungdd88/intent-policy')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('pull')
    p.add_argument('path')
    p.add_argument('local')
    q = sub.add_parser('push')
    q.add_argument('local')
    q.add_argument('path')
    q.add_argument('--exclude', nargs='*', default=[])
    r = sub.add_parser('ls')
    r.add_argument('prefix', nargs='?', default='')
    args = ap.parse_args()
    api = HfApi()
    if args.cmd == 'ls':
        files = [f for f in api.list_repo_files(REPO, repo_type='dataset') if f.startswith(args.prefix)]
        tops = sorted({'/'.join(f.split('/')[:3]) for f in files})
        print('\n'.join(tops))
    elif args.cmd == 'pull':
        mirror = Path('outputs/hf/mirror')
        snapshot_download(REPO, repo_type='dataset', allow_patterns=[f"{args.path.rstrip('/')}/**", args.path],
                          local_dir=mirror)
        src, dst = mirror / args.path, Path(args.local)
        if not src.exists():
            raise SystemExit(f'{args.path} is not in {REPO}')
        if dst.exists():
            shutil.rmtree(dst) if dst.is_dir() else dst.unlink()
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        print(f'{REPO}/{args.path} -> {dst}')
    else:
        local = Path(args.local)
        if local.is_file():
            api.upload_file(path_or_fileobj=str(local), path_in_repo=args.path, repo_id=REPO, repo_type='dataset')
        else:
            api.upload_folder(folder_path=str(local), path_in_repo=args.path, repo_id=REPO, repo_type='dataset',
                              ignore_patterns=args.exclude or None, commit_message=f'push {args.path}')
        print(f'{local} -> {REPO}/{args.path}')


if __name__ == '__main__':
    main()

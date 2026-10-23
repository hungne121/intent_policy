"""Download the HABIT episodes of the chosen tasks (HANDOFF_PIF_LOCAL.md §2: S8789 Supervisor, S8791 Coworker, S6760
Collaborator) from the public HF dataset configinc/HABIT (CC-BY-4.0): the `full` config's meta, the task_details of
the sids, the episode parquets and the requested camera videos. Files already present are skipped.

Layout (LOCAL §4.1): full/data/chunk-XXX/episode_NNNNNN.parquet, full/videos/chunk-XXX/observation.images.<cam>/
episode_NNNNNN.mp4, XXX = episode_index // 1000; the episodes of a sid come from full/meta/episodes.jsonl.
"""
import argparse
import json

from huggingface_hub import hf_hub_download, snapshot_download

from intent_policy.utils import resolve

REPO = 'configinc/HABIT'
META = ['info.json', 'episodes.jsonl', 'human_subtasks.jsonl', 'subtasks.jsonl', 'tasks.jsonl', 'modality.json']


def episode_files(episodes: list[dict], sids: list[str], cams: list[str]) -> list[str]:
    out = []
    for e in episodes:
        if e['sid'] in sids:
            i = int(e['episode_index'])           # stored as a string in episodes.jsonl
            c = i // 1000
            out.append(f'full/data/chunk-{c:03d}/episode_{i:06d}.parquet')
            out += [f'full/videos/chunk-{c:03d}/observation.images.{cam}/episode_{i:06d}.mp4' for cam in cams]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sids', nargs='+', default=['S8789', 'S8791', 'S6760'])
    p.add_argument('--cams', nargs='*', default=['front_view'])
    p.add_argument('--out', default='outputs/habit')
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    out = resolve(args.out)
    for f in META:
        hf_hub_download(REPO, f'full/meta/{f}', repo_type='dataset', local_dir=out)
    for s in args.sids:
        hf_hub_download(REPO, f'task_details/{s}.md', repo_type='dataset', local_dir=out)
    episodes = [json.loads(line) for line in open(out / 'full/meta/episodes.jsonl')]
    files = episode_files(episodes, args.sids, args.cams)
    todo = [f for f in files if not (out / f).exists()]
    print(f'{len(files)} files for {args.sids} ({len(files) - len(todo)} present)', flush=True)
    if todo:
        snapshot_download(REPO, repo_type='dataset', local_dir=out, allow_patterns=todo, max_workers=args.workers)
    missing = [f for f in files if not (out / f).exists()]
    print(f'done, missing: {len(missing)} {missing[:5]}')


if __name__ == '__main__':
    main()

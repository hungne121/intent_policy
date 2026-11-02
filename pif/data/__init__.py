"""PIF data: one entry point for every domain, chosen by the config (`data.domain`, `data.tasks`).

Every domain yields PIF episodes in the same format (pif/data/tokens.py) plus eval-only labels per episode.
"""
from __future__ import annotations

from pif.utils import resolve


def load_episodes(cfg: dict) -> tuple[list[dict], list[dict]]:
    """(episodes, labels) of `data.tasks` in `data.domain`."""
    dom, tasks = cfg['data']['domain'], cfg['data']['tasks']
    if dom == 'sim':
        from pif.data.sim import read_dataset
        return read_dataset(cfg, tasks)
    if dom == 'habit':
        from pif.data.habit import load_labels
        from pif.perception.lift import load_episode
        eps, labs = [], []
        for sid in tasks:
            files = sorted((resolve(cfg['habit']['tokens_dir']) / sid).glob('episode_*.npz'))
            lab = {r['episode']: r for r in load_labels(cfg['habit'], sid).get('episodes', [])}
            for f in files:
                ep = load_episode(f)
                eps.append(ep)
                labs.append(lab.get(ep['meta']['episode_index'], {}))
        return eps, labs
    raise KeyError(f'unknown data.domain {dom!r}')


def split(eps: list[dict], every: int) -> tuple[list[int], list[int]]:
    """Per task, every `every`-th episode (sorted by index) is held out: (train indices, test indices)."""
    by = {}
    for i, e in enumerate(eps):
        by.setdefault(e['meta']['task'], []).append((e['meta']['episode_index'], i))
    train, test = [], []
    for task in sorted(by):
        for n, (_, i) in enumerate(sorted(by[task])):
            (test if n % every == every - 1 else train).append(i)
    return train, test

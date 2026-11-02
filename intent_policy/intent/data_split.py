"""Explicit episode splits with raw human-motion clip leakage checks."""


def provenance_split(metadata, episodes):
    groups = {'train': [], 'val': [], 'test': []}
    clip_groups = {}
    for e in episodes:
        row = metadata[e]
        split = row.get('split')
        if split not in groups:
            raise ValueError(f'Episode {e}: split must be train, val or test, got {split!r}')
        if split == 'test':
            raise ValueError('Store test episodes in a separate evaluation dataset; dataset statistics must not include test')
        sources = [r.get('source') for r in row.get('human_motion_sources', [])]
        clips = set()
        for source in sources:
            if source and source.get('kind') == 'hymotion':
                cid = source.get('clip_id')
                if not cid:
                    raise ValueError(f'Episode {e}: HY-Motion source lacks clip_id')
                clips.add(cid)
                if source.get('split') != split:
                    raise ValueError(f'Episode {e}: clip manifest split does not match episode split')
        if not clips:
            raise ValueError(f'Episode {e}: no traceable HY-Motion clip')
        for cid in clips:
            previous = clip_groups.setdefault(cid, split)
            if previous != split:
                raise ValueError(f'HY-Motion clip {cid} leaks across {previous}/{split}')
        groups[split].append(e)
    if not groups['train'] or not groups['val']:
        raise ValueError('Explicit split needs nonempty train and validation groups')
    return groups['train'], groups['val']

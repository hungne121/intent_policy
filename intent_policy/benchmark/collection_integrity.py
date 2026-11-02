"""Source identity for paired demos; reactive timing is recorded separately, never hashed as source identity."""
import hashlib
import json
from pathlib import Path

SOURCE_FIELDS = ('kind', 'clip_id', 'file_sha256', 'sample_index', 'start_frame',
                 'end_frame_exclusive', 'fps', 'hand', 'split', 'reason')


def motion_source_hash(record):
    motions = record.get('human_motion_sources')
    if motions is None:
        motions = [dict(label=e['payload'].get('label'), source=e['payload'].get('motion_source'))
                   for e in record.get('events', []) if e['event_type'] == 'human_motion_start']
    if not motions or any(not m.get('source') for m in motions):
        return None
    identity = [(m['label'], {k: m['source'].get(k) for k in SOURCE_FIELDS}) for m in motions]
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def source_manifest(root):
    root = Path(root)
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ('intent_policy', 'scripts', 'configs')
            for p in sorted((root / folder).rglob('*')) if p.is_file() and p.suffix in ('.py', '.yaml', '.sh')}

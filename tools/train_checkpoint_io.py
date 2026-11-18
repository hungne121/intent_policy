"""Publish complete checkpoints and stage immutable, verifiable backups."""
import hashlib
import json
import shutil
import tempfile
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def publish_checkpoint(destination, writer):
    """A failed writer never exposes a completed checkpoint or replaces an old one."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f'.{destination.name}-', dir=destination.parent))
    old = destination.with_name(f'.{destination.name}-previous')
    try:
        writer(temp)
        files = {str(p.relative_to(temp)): sha256(p) for p in sorted(temp.rglob('*')) if p.is_file()}
        (temp / 'COMPLETE.json').write_text(json.dumps(files, indent=2))
        if old.exists():
            shutil.rmtree(old)
        if destination.exists():
            destination.rename(old)
        temp.rename(destination)
        if old.exists():
            shutil.rmtree(old)
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def verify_checkpoint(path):
    path = Path(path)
    files = json.loads((path / 'COMPLETE.json').read_text())
    if not files or any(not (path / name).is_file() or sha256(path / name) != digest
                        for name, digest in files.items()):
        raise ValueError(f'incomplete or corrupt checkpoint: {path}')
    return files


def latest_checkpoint(root):
    candidates = [p for p in Path(root).iterdir() if p.name.isdigit()
                  and (p / 'COMPLETE.json').exists() and (p / 'training_state.pt').exists()]
    return max(candidates, key=lambda p: int(p.name)) if candidates else None


def stage_run(run, stage, final=False):
    """Only numbered checkpoints are copied during training; best/last only after exit."""
    run, stage = Path(run), Path(stage)
    stage.mkdir(parents=True, exist_ok=True)
    for source in run.rglob('*'):
        rel = source.relative_to(run)
        if 'checkpoints' in rel.parts or any(p.startswith('.') for p in rel.parts) or not source.is_file():
            continue
        target = stage / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    checkpoints = {}
    for model in ('M0', 'M1-noH', 'M1'):
        root = run / model / 'checkpoints'
        if not root.exists():
            continue
        latest = latest_checkpoint(root)
        if latest:
            target = stage / model / 'checkpoints' / 'resume'
            shutil.copytree(latest, target)
            verify_checkpoint(target)
            checkpoints[model] = int(latest.name)
        if final:
            for name in ('best', 'last'):
                if (root / name).exists():
                    target = stage / model / 'checkpoints' / name
                    shutil.copytree(root / name, target)
                    verify_checkpoint(target)
    return checkpoints

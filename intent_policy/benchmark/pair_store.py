"""Lưu cặp demo nguyên tử, độc lập với train và rollout đánh giá policy.

Chỉ thư mục đã đổi tên từ staging mới là cặp được công bố. NPZ giữ nguyên
RGB/keypoint/action, không pickle; metadata/checksum phục vụ xuất LeRobot sau đó.
Đây là kho trung gian, không phải LeRobotDataset đã hoàn tất.
"""
from pathlib import Path
import hashlib
import json
import os
import re
import tempfile
import shutil
import numpy as np


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _write_json(path, value):
    with path.open('w') as stream:
        json.dump(value, stream, default=_json, allow_nan=False, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())


def _sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class PairStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def write_pair(self, candidate, records):
        key = str(candidate['pair_id'])
        if not re.fullmatch(r'[A-Za-z0-9_-]+', key):
            raise ValueError('pair_id must be a safe directory name')
        final = self.root / key
        if final.exists():
            raise FileExistsError(final)
        if set(records) != {'early', 'observed'}:
            raise ValueError('Both early and observed are required')
        condition = records['early'].get('exogenous_condition_sha256')
        if not condition or condition != records['observed'].get('exogenous_condition_sha256'):
            raise ValueError('Unmatched exogenous conditions')
        for record in records.values():
            if record.get('success') is not True or (record.get('collection_quality') or {}).get('passed') is not True:
                raise ValueError('Only successful quality-approved demonstrations can be stored')
        stage = Path(tempfile.mkdtemp(prefix='.partial-', dir=self.root))
        try:
            manifest = dict(format='paired_demo_npz_v1', candidate=candidate, sides={})
            for mode, record in records.items():
                frames = record['frames']
                if not frames:
                    raise ValueError('Empty episode')
                keys = set(frames[0])
                if any(set(frame) != keys for frame in frames):
                    raise ValueError('Frame fields changed within episode')
                required = {'action', 'observation.state', 'human.keypoints'}
                if not required <= keys or not any(k.startswith('observation.images.') for k in keys):
                    raise ValueError('Missing training observations/action/human keypoints')
                arrays = {k: np.stack([f[k] for f in frames]) for k in sorted(keys)}
                for k, array in arrays.items():
                    if array.dtype.kind not in 'biuf' or not np.isfinite(array).all():
                        raise ValueError(f'Invalid training array: {k}')
                    if k.startswith('observation.images.') and (array.dtype != np.uint8 or array.ndim != 4 or array.shape[-1] != 3):
                        raise ValueError(f'Expected uint8 RGB frames: {k}')
                path = stage / f'{mode}.npz'
                with path.open('wb') as stream:
                    np.savez_compressed(stream, **arrays)
                    stream.flush()
                    os.fsync(stream.fileno())
                metadata = {k: v for k, v in record.items() if k != 'frames'}
                meta_path = stage / f'{mode}.json'
                _write_json(meta_path, metadata)
                manifest['sides'][mode] = dict(frames=len(frames), arrays=path.name,
                    arrays_sha256=_sha(path), metadata=meta_path.name, metadata_sha256=_sha(meta_path))
            _write_json(stage / 'manifest.json', manifest)
            # Single collector owns this root; never overwrite another pair.
            if final.exists():
                raise FileExistsError(final)
            stage.rename(final)
            fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            return final
        finally:
            if stage.exists():
                shutil.rmtree(stage)

    def verified_pairs(self):
        """Checksum failure aborts resume; incomplete staging is never counted."""
        for directory in sorted(self.root.iterdir()):
            if not directory.is_dir() or directory.name.startswith('.'):
                continue
            manifest = json.loads((directory / 'manifest.json').read_text())
            if manifest.get('format') != 'paired_demo_npz_v1' or set(manifest['sides']) != {'early', 'observed'}:
                raise ValueError(f'Invalid pair manifest: {directory}')
            for mode, side in manifest['sides'].items():
                for kind, suffix in [('arrays', 'npz'), ('metadata', 'json')]:
                    if side[kind] != f'{mode}.{suffix}':
                        raise ValueError('Unexpected pair file path')
                    if _sha(directory / side[kind]) != side[kind + '_sha256']:
                        raise ValueError(f'Checksum mismatch: {directory / side[kind]}')
            yield manifest

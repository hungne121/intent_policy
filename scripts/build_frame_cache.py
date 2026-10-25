"""Decode every camera frame of a LeRobot dataset once and store it as JPEG (quality 95) in one flat buffer, so training
reads frames without seeking in the AV1 videos (the CPU-bound step: 1.8 steps/s per model on a 9-vCPU A40 pod, GPU
~14 %, 2026-10-25). `scripts/train_policy.py --frame-cache` serves the frames from it.
  ./run.sh -m scripts.build_frame_cache --dataset-root outputs/datasets/t2deep_early [--check 200]
  -> <root>/frame_cache_q95.npz: buffer (uint8), offsets / lengths [key, global frame index], keys
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from torchcodec.decoders import VideoDecoder

QUALITY = 95
NAME = f'frame_cache_q{QUALITY}.npz'


def decode_jpeg(buf: np.ndarray) -> torch.Tensor:
    """JPEG bytes -> float CHW in [0, 1] (the layout LeRobot's video decoding returns)."""
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return torch.from_numpy(np.ascontiguousarray(bgr[:, :, ::-1].transpose(2, 0, 1))).float().div_(255.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--dataset-root', required=True)
    ap.add_argument('--check', type=int, default=200, help='frames compared with the decoded video (mean / max |diff|)')
    args = ap.parse_args()
    root = Path(args.dataset_root)
    meta = LeRobotDatasetMetadata('local/cache', root=root)
    keys, fps, n = list(meta.video_keys), float(meta.fps), int(meta.total_frames)
    chunks, offsets, lengths, pos = [], np.zeros((len(keys), n), np.int64), np.zeros((len(keys), n), np.int64), 0
    t0 = time.time()
    for ki, key in enumerate(keys):
        decoders = {}
        for ep in range(meta.total_episodes):
            e = meta.episodes[ep]
            path = root / meta.get_video_file_path(ep, key)
            dec = decoders.setdefault(str(path), VideoDecoder(str(path)))
            first, length = int(e['dataset_from_index']), int(e['length'])
            ts = [float(e[f'videos/{key}/from_timestamp']) + i / fps for i in range(length)]
            frames = dec.get_frames_played_at(seconds=ts).data.numpy()            # [T, C, H, W] uint8
            for i, f in enumerate(frames):
                ok, jpg = cv2.imencode('.jpg', np.ascontiguousarray(f.transpose(1, 2, 0)[:, :, ::-1]),
                                       [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
                assert ok
                b = jpg.ravel()
                chunks.append(b)
                offsets[ki, first + i], lengths[ki, first + i] = pos, len(b)
                pos += len(b)
        print(f'{key}: done ({time.time() - t0:.0f} s)', flush=True)
    buffer = np.concatenate(chunks)
    np.savez(root / NAME, buffer=buffer, offsets=offsets, lengths=lengths, keys=np.array(keys))
    print(f'{root / NAME}: {n} frames x {len(keys)} cameras, {buffer.nbytes / 1e9:.2f} GB, {time.time() - t0:.0f} s')
    if args.check:                              # compare with the video frames LeRobot decodes
        rng = np.random.default_rng(0)
        diffs = []
        for _ in range(args.check):
            ki, gi = int(rng.integers(len(keys))), int(rng.integers(n))
            ep = int(np.searchsorted([meta.episodes[e]['dataset_to_index'] for e in range(meta.total_episodes)], gi, side='right'))
            e = meta.episodes[ep]
            path = root / meta.get_video_file_path(ep, keys[ki])
            ref = VideoDecoder(str(path)).get_frames_played_at(
                seconds=[float(e[f'videos/{keys[ki]}/from_timestamp']) + (gi - int(e['dataset_from_index'])) / fps]).data[0]
            got = decode_jpeg(buffer[offsets[ki, gi]:offsets[ki, gi] + lengths[ki, gi]])
            diffs.append((got - ref.float() / 255.0).abs())
        print(f'check {args.check} frames: mean |diff| {torch.stack([d.mean() for d in diffs]).mean():.4f}, '
              f'max |diff| {max(float(d.max()) for d in diffs):.4f} (in [0, 1])')


if __name__ == '__main__':
    main()

"""Compute time of the measuring device per camera frame (session report 2026-10-18): the streaming DeePoint server
(scripts/deepoint_server.py: OpenPifPaf keypoints + the fine-tuned DeePoint) fed 1280x720 sensor frames, as in the loop
(intent/device.DeePointClient). Wall time per frame, after a warm-up, on recorded sensor videos.

  ./run.sh -m scripts.device_latency --videos outputs/prelim/2026-10-13_slow/device/render/episode_00*_side_15fps.mp4
"""
import argparse
import json
import time

import cv2
import numpy as np

from intent_policy.intent.device import DeePointClient
from intent_policy.utils import resolve


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--videos', nargs='+', required=True)
    ap.add_argument('--folds', default='outputs/device/current/device_folds.json')
    ap.add_argument('--warmup', type=int, default=20)
    ap.add_argument('--out')
    args = ap.parse_args()
    folds = json.loads(resolve(args.folds).read_text())['folds']
    client = DeePointClient.shared(tuple(resolve(folds[k]['checkpoint']) for k in sorted(folds)))
    times = []
    for v in args.videos:
        client.reset(0)
        cap = cv2.VideoCapture(v)
        k = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            t0 = time.perf_counter()
            client.step(rgb)
            dt = time.perf_counter() - t0
            if k >= args.warmup:
                times.append(dt)
            k += 1
    t = np.asarray(times) * 1000
    res = dict(frames=int(len(t)), mean_ms=float(t.mean()), sd_ms=float(t.std()), median_ms=float(np.median(t)),
               p95_ms=float(np.percentile(t, 95)), budget_15fps_ms=1000 / 15)
    print(json.dumps(res, indent=1))
    if args.out:
        resolve(args.out).write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()

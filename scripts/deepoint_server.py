"""Streaming DeePoint for the closed loop (intent_policy/intent/device.py starts it as a subprocess): one camera frame
in, the device's output for the window ending at that frame out.

Each frame goes through DeePoint's own per-frame pipeline exactly as in scripts/deepoint_infer.py (`MovieDataset`'s
per-frame inputs: OpenPifPaf on the half-size BGR frame, the frozen ResNet-34 joint features, the spatial
transformer); the last `tlength` = 15 spatial embeddings feed the temporal transformer and the heads. Frames are
expected at 15 fps (the caller sends every 15 fps frame). The released or fine-tuned checkpoints (> 2 classes: the
multi-gesture classifier of scripts/deepoint_finetune.py) are loaded as in deepoint_infer.load_network; several
checkpoints (the fold models) are held at once and `reset` selects the one of the next episode.

Protocol (binary stdin, one JSON line per reply on stdout; library prints go to stderr):
  request  4-byte little-endian header length, JSON header, payload
           {"cmd": "reset", "model": k}             start a new episode with checkpoint k (clears the window)
           {"cmd": "frame", "h": H, "w": W}         + H*W*3 bytes RGB uint8
  reply    {"ok": true} | {"probs": [C] or null, "direction": [3] or null, "kp": [[x, y, c] x 17]}
           (probs / direction null until 15 frames have been seen; direction is DeePoint's camera-centred unit
           vector: x right, y depth, z up)
"""
import argparse
import json
import os
import struct
import sys
from collections import deque

import numpy as np
import torch
from einops import repeat

sys.path.insert(0, os.path.dirname(__file__))
from deepoint_infer import REPO, SPATIAL_KEYS, dp_dataset, heads, load_network   # noqa: E402


class _Cap:
    """Stands in for MovieDataset's cv2.VideoCapture: returns the frame pushed last."""
    frame = None

    def set(self, *_):
        pass

    def read(self):
        return True, self.frame


class FrameStream(dp_dataset.MovieDataset):
    """MovieDataset fed frame by frame (token_length 1, sequential: no seek), its __getitem__ unchanged."""

    def __init__(self, device: str, lr: str = 'r'):
        import openpifpaf
        self.lr, self.device, self.token_length = lr, device, 1
        self.predictor = openpifpaf.Predictor(visualize_image=True, checkpoint='shufflenetv2k16', json_data=False)
        self.reset()

    def reset(self) -> None:
        self.cap, self.predicted_joints, self.prev_frame, self.i = _Cap(), {}, -1, 0

    def push(self, bgr: np.ndarray) -> dict:
        self.cap.frame = bgr
        i, self.i = self.i, self.i + 1
        self.prev_frame = i
        item = self[i]
        self.predicted_joints.pop(i - 1, None)          # the window lives in the embedding queue, not here
        return item


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ckpt', nargs='+', default=[str(REPO / 'models' / 'weight.ckpt')])
    args = p.parse_args()
    out = os.fdopen(os.dup(1), 'w', buffering=1)       # replies only; everything printed goes to stderr
    os.dup2(2, 1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    nets = [load_network(c, device) for c in args.ckpt]
    if len({int(n.classifier.out_features) for n, _ in nets}) != 1 or len({c.model.tlength for _, c in nets}) != 1:
        raise SystemExit('the checkpoints differ in classes or window length')
    net, cfg = nets[0]
    L = cfg.model.tlength
    stream, window = FrameStream(device), deque(maxlen=L)
    inp = sys.stdin.buffer
    out.write(json.dumps({'ready': True, 'classes': int(net.classifier.out_features)}) + '\n')
    with torch.no_grad():
        while True:
            raw = inp.read(4)
            if len(raw) < 4:
                break
            header = json.loads(inp.read(struct.unpack('<I', raw)[0]))
            if header['cmd'] == 'reset':
                net = nets[int(header.get('model', 0))][0]
                stream.reset()
                window.clear()
                out.write('{"ok": true}\n')
                continue
            h, w = int(header['h']), int(header['w'])
            rgb = np.frombuffer(inp.read(h * w * 3), np.uint8).reshape(h, w, 3)
            it = stream.push(np.ascontiguousarray(rgb[:, :, ::-1]))   # BGR, as cv2 decoded the training videos
            frame = {k: it[k].unsqueeze(0)[:, 0] for k in SPATIAL_KEYS}
            window.append(net.spatial_transformer(frame)[:, -1].detach())
            reply = {'probs': None, 'direction': None, 'kp': it['keypoints'][0].numpy().round(2).tolist()}
            if len(window) == L:
                tokens = list(window)
                if net.use_temporal_special_token:
                    tokens.append(repeat(net.temporal_special_token, 'hidden -> 1 hidden'))
                prob, d = heads(net, net.temporal_transformer(torch.stack(tokens))[:, -1])
                reply['probs'], reply['direction'] = prob[0].round(5).tolist(), d[0].round(5).tolist()
            out.write(json.dumps(reply) + '\n')


if __name__ == '__main__':
    main()

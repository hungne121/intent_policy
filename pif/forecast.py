"""Human motion forecasting with ManiCast (Kedia, Dan, Bhardwaj, Choudhury, CoRL 2023), used as published.

Code + checkpoints: third_party/manicast (github.com/portal-cornell/manicast, branch `release`). Settings:
configs/pif/pif.yaml `forecast`.

ManiCast input / output (their eval/test_comad.py, data/utils/comad.py):
  7 mocap markers [BackTop, LShoulderBack, RShoulderBack, LElbowOut, RElbowOut, LWristOut, RWristOut], 3D [m],
  10 frames at 25 Hz in (0.4 s) -> 25 frames out (1 s); every frame minus WaistLBack at the last input frame.
  CoMaD axes: +y up, the person faces -z, the person's left is -x (measured on handover_0.json, docs in config).
Our table frame: +x from the robot towards the person, +y the robot's left, +z up; the person faces the robot (-x),
so the person's left is -y. The axis map (x_c, y_c, z_c) = (y, z, x) is a rotation (det +1).
Our tokens have shoulders, elbows, wrists (joint centres, not markers) and no back / waist markers: BackTop and
WaistLBack are the shoulder midpoint plus the mean marker offsets measured on CoMaD (config `forecast.offsets`).
Our data are 10 Hz: the 0.4 s history is resampled to 25 Hz by linear interpolation **between observed frames
only** (causal); the 25 Hz output is read at the 10 Hz times we need.
"""
from __future__ import annotations
import sys

import numpy as np
import torch

from pif.utils import resolve

MANICAST_JOINTS = ('BackTop', 'LShoulderBack', 'RShoulderBack', 'LElbowOut', 'RElbowOut', 'LWristOut', 'RWristOut')
OUR_JOINTS = (None, 'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist')   # BackTop from offsets
TO_COMAD = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float)   # (x_c, y_c, z_c) = (y, z, x)


class ManiCastForecaster:
    def __init__(self, cfg: dict):
        f = cfg['forecast']
        root = str(resolve(f['repo']))
        if root not in sys.path:
            sys.path.insert(0, root)
        from model.manicast import ManiCast
        m = f['model']
        self.net = ManiCast(3, m['input_n'], m['output_n'], m['st_gcnn_dropout'], 7, m['n_tcnn_layers'],
                            m['tcnn_kernel_size'], m['tcnn_dropout'])
        ck = resolve(f['checkpoint_path']) if f.get('checkpoint_path') else resolve(f['repo']) / 'model_checkpoints' / f['checkpoint']
        self.net.load_state_dict(torch.load(ck, map_location='cpu'))     # checkpoint_path: our fine-tuned weights
        self.net.eval()
        self.rate, self.input_n, self.output_n = float(m['rate_hz']), m['input_n'], m['output_n']
        self.back_off = np.asarray(f['offsets']['backtop_minus_shoulder_mid'], float)
        self.waist_off = np.asarray(f['offsets']['waist_minus_shoulder_mid'], float)

    # ------------------------------------------------------------------ frames
    def to_comad(self, kp: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """kp: our joints [T, 3] (table frame) -> ManiCast markers [T, 7, 3] and WaistLBack [T, 3], both CoMaD axes."""
        c = {k: v @ TO_COMAD.T for k, v in kp.items()}
        mid = (c['l_shoulder'] + c['r_shoulder']) / 2
        joints = [mid + self.back_off] + [c[j] for j in OUR_JOINTS[1:]]
        return np.stack(joints, 1), mid + self.waist_off

    @staticmethod
    def from_comad(x: np.ndarray) -> np.ndarray:
        return x @ TO_COMAD                          # inverse of a rotation = transpose

    # ------------------------------------------------------------------ forecast
    @torch.no_grad()
    def predict_comad(self, hist: np.ndarray) -> np.ndarray:
        """hist [B, input_n, 7, 3] at the model rate, already relative to the waist -> [B, output_n, 7, 3]."""
        x = torch.as_tensor(hist, dtype=torch.float32).permute(0, 3, 1, 2)
        return self.net(x).permute(0, 1, 3, 2).numpy()

    def forecast(self, markers: np.ndarray, waist: np.ndarray, t_hist: np.ndarray, t_out: np.ndarray) -> np.ndarray:
        """markers [B, n, 7, 3], waist [B, n, 3] observed at times t_hist [n] (s, last = now, any rate);
        returns the forecast markers at future times t_out [k] (s after now), [B, k, 7, 3], CoMaD axes, absolute."""
        dt = 1.0 / self.rate
        ts = t_hist[-1] - dt * np.arange(self.input_n - 1, -1, -1)        # model input times, all <= now (causal)
        flat = markers.reshape(len(markers), len(t_hist), -1)
        res = np.stack([np.stack([np.interp(ts, t_hist, flat[b, :, d]) for d in range(flat.shape[2])], 1)
                        for b in range(len(flat))]).reshape(len(markers), self.input_n, 7, 3)
        ref = waist[:, -1][:, None, None]
        out = self.predict_comad(res - ref) + ref                           # [B, output_n, 7, 3]
        t_model = dt * np.arange(1, self.output_n + 1)
        o = out.reshape(len(out), self.output_n, -1)
        return np.stack([np.stack([np.interp(t_out, t_model, o[b, :, d]) for d in range(o.shape[2])], 1)
                         for b in range(len(o))]).reshape(len(out), len(t_out), 7, 3)

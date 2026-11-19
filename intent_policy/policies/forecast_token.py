"""M1 forecast token (07/QĐ-HL D; 03/TTr-HL #1): z_H = MLP(Flatten(X_H)), X_H = [Ĥ_t, Ĥ_t - p^TCP_t, dĤ_t/dt].

Input (dataset contract hrc-1.2, intent_policy/data/dataset_schema.py):
  * `observation.human_forecast` (B, 180) = the EFFECTIVE ManiCast forecast (N5: the faulty one), 10 horizons
    (0.1 ... 1.0 s) x 6 joints (JOINTS order) x 3, WORLD frame;
  * `observation.forecast_valid` (B, 1), 1 = valid (the recorder stores zeros otherwise);
  * `observation.state` (B, 13) after LeRobot's MEAN_STD normalisation: qpos (6) is un-normalised here with the
    training statistics (buffers) to get p^TCP_t by forward kinematics.
Preprocessing, in the order of 07/QĐ-HL D: (1) one frame: world (as recorded, the TCP from FK is world too);
(2) normalise positions: (x - mu) / sigma, mu / sigma per axis over the valid training forecasts (all horizons, joints);
(3) relative to the robot: Ĥn - p^TCP_n (the TCP normalised with the same mu / sigma);
(4) velocity: finite difference of Ĥn over the horizon steps (dt = 0.1 s): v_k = (Ĥn_k - Ĥn_{k-1}) / dt for k >= 1,
    v_0 = v_1 (one-sided: the forecast alone, no observed pose);
(5) validity mask: an invalid forecast gives zero features and the flag 0, appended to the MLP input.
The MLP is the only trained part (ManiCast stays frozen: its output is a dataset column).

UR3e forward kinematics: the body chain robot_mount -> ... -> site `tcp` of assets/scene.xml with the mount pose of
configs/hrc/station.yaml (robot_base_xy, robot_base_yaw_deg; the same edit as intent_policy/shared/sim_env.apply_station),
read once (`ur3e_tcp_chain`) and stored in the policy config, so a saved model needs no repo file.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch
import yaml
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[2]
ARM_JOINTS = ('shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint')


def _floats(s: str | None, default):
    return [float(v) for v in s.split()] if s else list(default)


def ur3e_tcp_chain(scene_xml: Path = ROOT / 'assets/scene.xml', station_yaml: Path = ROOT / 'configs/hrc/station.yaml',
                   site: str = 'tcp') -> dict:
    """Kinematic chain world -> `site` as plain lists: [{body, pos, quat, joint, axis}], site_pos, joints."""
    root = ET.parse(scene_xml).getroot()
    world = root.find('worldbody')
    path = None

    def walk(el, trail):
        nonlocal path
        for ch in el:
            if ch.tag == 'site' and ch.get('name') == site:
                path = trail + [ch]
            elif ch.tag == 'body':
                walk(ch, trail + [ch])
    walk(world, [])
    if path is None:
        raise KeyError(f'site {site!r} not in {scene_xml}')
    st = yaml.safe_load(Path(station_yaml).read_text())
    links, joints = [], []
    for i, body in enumerate(path[:-1]):
        for attr in ('euler', 'axisangle', 'xyaxes', 'zaxis'):
            if body.get(attr) is not None:
                raise ValueError(f'body {body.get("name")}: orientation {attr} not supported (pos / quat only)')
        pos, quat = _floats(body.get('pos'), (0, 0, 0)), _floats(body.get('quat'), (1, 0, 0, 0))
        if i == 0:                                   # robot_mount: sim_env.apply_station sets xy + yaw, keeps z
            if body.get('name') != 'robot_mount':
                raise ValueError(f'chain starts at {body.get("name")}, expected robot_mount')
            half = np.radians(float(st['robot_base_yaw_deg'])) / 2
            pos = [float(st['robot_base_xy'][0]), float(st['robot_base_xy'][1]), pos[2]]
            quat = [float(np.cos(half)), 0.0, 0.0, float(np.sin(half))]
        js = body.findall('joint')
        if len(js) > 1:
            raise ValueError(f'body {body.get("name")} has {len(js)} joints')
        joint = axis = None
        if js:
            j = js[0]
            if j.get('type', 'hinge') != 'hinge' or any(abs(v) > 0 for v in _floats(j.get('pos'), (0, 0, 0))):
                raise ValueError(f'joint {j.get("name")}: only hinge joints at the body origin')
            joint, axis = j.get('name'), _floats(j.get('axis'), (0, 0, 1))
            joints.append(joint)
        links.append(dict(body=body.get('name'), pos=pos, quat=quat, joint=joint, axis=axis))
    if tuple(joints) != ARM_JOINTS:
        raise ValueError(f'chain joints {joints} != the 6 UR3e arm joints in observation.state order {ARM_JOINTS}')
    return dict(links=links, site_pos=_floats(path[-1].get('pos'), (0, 0, 0)), joints=joints,
                source=f'{Path(scene_xml).name} + {Path(station_yaml).name}')


def _quat_mat(q: Tensor) -> Tensor:
    w, x, y, z = (q / q.norm()).unbind(-1)
    return torch.stack([torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)]),
                        torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)]),
                        torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)])])


def _axis_rot(axis: Tensor, angle: Tensor) -> Tensor:
    """(B, 3, 3) rotations about the unit `axis` by `angle` (B,) (Rodrigues)."""
    k = axis / axis.norm()
    K = torch.zeros(3, 3, dtype=angle.dtype, device=angle.device)
    K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
    s, c = angle.sin()[:, None, None], angle.cos()[:, None, None]
    return torch.eye(3, dtype=angle.dtype, device=angle.device) + s * K + (1 - c) * (K @ K)


class TcpFK(nn.Module):
    """World TCP position (B, 3) from the 6 arm joint angles (B, 6) [rad] (MuJoCo body-frame convention:
    child = parent * (pos, quat) * R(axis, q))."""

    def __init__(self, chain: dict):
        super().__init__()
        self.n = len(chain['links'])
        self.joint_of = []
        k = 0
        for i, l in enumerate(chain['links']):
            self.register_buffer(f'p{i}', torch.tensor(l['pos'], dtype=torch.float64))
            self.register_buffer(f'R{i}', _quat_mat(torch.tensor(l['quat'], dtype=torch.float64)))
            if l['joint'] is not None:
                self.register_buffer(f'a{i}', torch.tensor(l['axis'], dtype=torch.float64))
                self.joint_of.append(k)
                k += 1
            else:
                self.joint_of.append(None)
        self.register_buffer('site', torch.tensor(chain['site_pos'], dtype=torch.float64))

    def forward(self, q: Tensor) -> Tensor:
        dt = q.dtype
        q = q.double()
        b = q.shape[0]
        R = torch.eye(3, dtype=q.dtype, device=q.device).expand(b, 3, 3)
        p = torch.zeros(b, 3, dtype=q.dtype, device=q.device)
        for i in range(self.n):
            p = p + R @ getattr(self, f'p{i}')
            R = R @ getattr(self, f'R{i}')
            j = self.joint_of[i]
            if j is not None:
                R = R @ _axis_rot(getattr(self, f'a{i}'), q[:, j])
        return (p + R @ self.site).to(dt)


def forecast_features(forecast: Tensor, valid: Tensor, tcp: Tensor, mean: Tensor, std: Tensor, dt: float) -> Tensor:
    """X_H (B, H, J, 9) = [Ĥn, Ĥn - p^TCP_n, dĤn/dt] from the world forecast (B, H, J, 3), validity (B,) and world
    TCP (B, 3); zeros where the forecast is invalid. Steps (1)-(5) of the module docstring."""
    h = (forecast - mean) / std                                   # (2) normalise
    t = (tcp - mean) / std
    rel = h - t[:, None, None, :]                                 # (3) relative to the TCP
    v = (h[:, 1:] - h[:, :-1]) / dt                               # (4) velocity over the horizon steps
    v = torch.cat([v[:, :1], v], dim=1)
    x = torch.cat([h, rel, v], dim=-1)
    return x * (valid > 0.5).to(x.dtype)[:, None, None, None]     # (5) mask


class ForecastEncoder(nn.Module):
    """One transformer token z_H (B, 1, D) from the forecast; `set_stats` must be called before training."""
    n_tokens = 1

    def __init__(self, dim_model: int, cfg: dict):
        super().__init__()
        self.H, self.J, self.dt = int(cfg['horizons']), int(cfg['joints']), float(cfg['dt_s'])
        self.key, self.valid_key, self.state_key = cfg['key'], cfg['valid_key'], 'observation.state'
        self.fk = TcpFK(cfg['fk_chain'])
        self.register_buffer('pos_mean', torch.zeros(3))
        self.register_buffer('pos_std', torch.ones(3))
        self.register_buffer('q_mean', torch.zeros(6))
        self.register_buffer('q_std', torch.ones(6))
        self.register_buffer('stats_set', torch.zeros((), dtype=torch.bool))
        n_in = self.H * self.J * 9 + 1
        self.mlp = nn.Sequential(nn.Linear(n_in, int(cfg['hidden_dim'])), nn.ReLU(), nn.Linear(int(cfg['hidden_dim']), dim_model))

    def set_stats(self, pos_mean, pos_std, q_mean, q_std) -> None:
        """Training-split statistics: forecast position mean / std per axis, and the qpos (first 6 dims of
        observation.state) mean / std that LeRobot's normaliser uses (to undo it for FK)."""
        for name, v in (('pos_mean', pos_mean), ('pos_std', pos_std), ('q_mean', q_mean), ('q_std', q_std)):
            getattr(self, name).copy_(torch.as_tensor(np.asarray(v, np.float32)).reshape(-1))
        if (self.pos_std <= 0).any():
            raise ValueError('forecast position std must be > 0')
        self.stats_set.fill_(True)

    def qpos(self, state_norm: Tensor) -> Tensor:
        return state_norm[:, :6].float() * (self.q_std + 1e-8) + self.q_mean    # LeRobot MEAN_STD: (x - m) / (s + eps)

    def features(self, batch: dict) -> tuple[Tensor, Tensor]:
        if not bool(self.stats_set):
            raise RuntimeError('ForecastEncoder statistics not set (train: set_stats from the training split)')
        state = batch[self.state_key]
        b = state.shape[0]
        fc = batch[self.key].float().reshape(b, self.H, self.J, 3)
        valid = batch[self.valid_key].float().reshape(b)
        tcp = self.fk(self.qpos(state))
        return forecast_features(fc, valid, tcp, self.pos_mean, self.pos_std, self.dt), valid

    def forward(self, batch: dict) -> Tensor:
        x, valid = self.features(batch)
        flag = (valid > 0.5).to(x.dtype)[:, None]
        return self.mlp(torch.cat([x.flatten(1), flag], dim=1))[:, None]


class _PreLNLayer(nn.Module):
    """Pre-LN transformer encoder layer with GELU (Xiong et al. 2020 "On Layer Normalization in the Transformer
    Architecture"; Hendrycks & Gimpel 2016 GELU): x + MHA(LN(x)); x + FFN(LN(x)). Records the share of FFN hidden units
    that stay inactive (|post-activation| <= eps over every token of the batch) for the dead-unit diagnostic."""

    def __init__(self, d: int, n_heads: int, ff: int, dropout: float, activation: str):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.lin1, self.lin2 = nn.Linear(d, ff), nn.Linear(ff, d)
        self.act = nn.GELU() if activation == 'gelu' else nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.last_hidden: Tensor | None = None

    def forward(self, x: Tensor) -> Tensor:
        y = self.ln1(x)
        x = x + self.drop(self.attn(y, y, y, need_weights=False)[0])
        h = self.act(self.lin1(self.ln2(x)))
        self.last_hidden = h.detach()
        return x + self.drop(self.lin2(self.drop(h)))


class StructuredForecastEncoder(nn.Module):
    """M* structured ManiCast conditioning (19/QĐ-HL III.1): one token per (horizon k, joint j),
    e_kj = LayerNorm(W [Ĥ_kj, r_kj, v_kj, valid] + E_time[k] + E_joint[j]), then a Pre-LN GELU transformer encoder over
    the H x J tokens; z_H (B, H*J, D) are extra tokens of the ACT encoder. Input features = forecast_features (the
    same normalisation / TCP-relative / velocity as M1), validity flag appended to every token (invalid -> zero
    features + flag 0). Statistics, FK and the diagnostics API are those of ForecastEncoder."""

    def __init__(self, dim_model: int, cfg: dict):
        super().__init__()
        self.H, self.J, self.dt = int(cfg['horizons']), int(cfg['joints']), float(cfg['dt_s'])
        self.n_tokens = self.H * self.J
        self.key, self.valid_key, self.state_key = cfg['key'], cfg['valid_key'], 'observation.state'
        self.fk = TcpFK(cfg['fk_chain'])
        for n, k in (('pos_mean', 3), ('pos_std', 3), ('q_mean', 6), ('q_std', 6)):
            self.register_buffer(n, torch.ones(k) if n.endswith('std') else torch.zeros(k))
        self.register_buffer('stats_set', torch.zeros((), dtype=torch.bool))
        n_layers, n_heads = int(cfg.get('n_layers', 2)), int(cfg.get('n_heads', 4))
        act, p = cfg.get('activation', 'gelu'), float(cfg.get('dropout', 0.1))
        if act not in ('gelu', 'relu'):
            raise ValueError(f'forecast activation {act!r}')
        self.inp = nn.Linear(10, dim_model)
        self.e_time = nn.Parameter(torch.randn(self.H, 1, dim_model) * 0.02)
        self.e_joint = nn.Parameter(torch.randn(1, self.J, dim_model) * 0.02)
        self.ln_in = nn.LayerNorm(dim_model)
        self.layers = nn.ModuleList(_PreLNLayer(dim_model, n_heads, int(cfg.get('hidden_dim', 256)) * 2, p, act)
                                    for _ in range(n_layers))
        self.ln_out = nn.LayerNorm(dim_model)

    set_stats = ForecastEncoder.set_stats
    qpos = ForecastEncoder.qpos

    def features(self, batch: dict) -> tuple[Tensor, Tensor]:
        if not bool(self.stats_set):
            raise RuntimeError('forecast statistics not set (train: set_stats from the training split)')
        state = batch[self.state_key]
        b = state.shape[0]
        fc = batch[self.key].float().reshape(b, self.H, self.J, 3)
        valid = batch[self.valid_key].float().reshape(b)
        return forecast_features(fc, valid, self.fk(self.qpos(state)), self.pos_mean, self.pos_std, self.dt), valid

    def forward(self, batch: dict) -> Tensor:
        x, valid = self.features(batch)
        flag = (valid > 0.5).to(x.dtype)[:, None, None, None].expand(-1, self.H, self.J, 1)
        x = torch.cat([x, flag], dim=-1)                                       # (B, H, J, 10)
        e = self.inp(x) + self.e_time + self.e_joint
        z = self.ln_in(e).flatten(1, 2)
        for layer in self.layers:
            z = layer(z)
        return self.ln_out(z)

    @torch.no_grad()
    def dead_unit_ratio(self) -> float:
        """Share of FFN hidden units (all layers) with |activation| <= 1e-6 on every token of the last forward batch."""
        dead = [(l.last_hidden.abs().amax((0, 1)) <= 1e-6).float().mean().item() for l in self.layers if l.last_hidden is not None]
        return float(np.mean(dead)) if dead else float('nan')


def make_forecast_encoder(dim_model: int, cfg: dict) -> nn.Module:
    arch = cfg.get('arch', 'mlp_flatten')
    if arch == 'structured_transformer':
        return StructuredForecastEncoder(dim_model, cfg)
    if arch == 'mlp_flatten':
        return ForecastEncoder(dim_model, cfg)
    raise ValueError(f'forecast arch {arch!r}')


def forecast_position_stats(forecast: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-axis mean / std of the valid forecasts (N, 180) of the training split."""
    f = np.asarray(forecast, np.float64)[np.asarray(valid).reshape(-1) > 0.5].reshape(-1, 3)
    if len(f) == 0:
        raise ValueError('no valid forecast in the training split')
    return f.mean(0), f.std(0)

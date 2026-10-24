"""D's intent input: the ManiCast forecast in the v2 contract (docs/PIF.md §2.2). One construction for the dataset
columns (scripts/add_manicast_intent.py) and the closed loop (policies/policy_agent.py, OnlineManiCast), so the policy
sees at evaluation exactly what it was trained on.

  xi (M, J, 3)  forecast displacement of the 6 arm joints at (m + 1) / M x horizon_s, from the frozen ManiCast
                fine-tuned on v6 motion (configs/pif/pif.yaml v6_expert.checkpoint_path); causal: the last 0.4 s of
                keypoints (ManiCast's input), frames before the episode start repeat the first one
  the other fields neutral (none / rest / zero occupancy); D is trained and evaluated with intent_keep = motion
"""
import numpy as np

from intent_policy.intent.contract import confidence, one_hot

MANICAST = ['l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist']   # pif/forecast.py OUR_JOINTS[1:]
HISTORY_S = 0.4                                                                        # ManiCast input window


def load_forecaster():
    from pif.config import load_config
    from pif.forecast import ManiCastForecaster
    cfg = load_config('configs/pif/pif.yaml', [])
    return ManiCastForecaster(dict(cfg, forecast=dict(cfg['forecast'], checkpoint_path=cfg['v6_expert']['checkpoint_path'])))


def horizons(schema: dict) -> np.ndarray:
    M = int(schema['n_waypoints'])
    return (np.arange(M) + 1) / M * float(schema['horizon_s'])


def history_len(fps: float) -> int:
    return int(round(HISTORY_S * fps)) + 1


def forecast_xi(kp: np.ndarray, fc, fps: float, t_out: np.ndarray) -> np.ndarray:
    """kp [T, 6, 3] (MANICAST order, world) -> xi [T, M, 6, 3]: forecast joint position - current position."""
    n = history_len(fps)
    pad = np.concatenate([np.repeat(kp[:1], n - 1, 0), kp])
    mk, waist = fc.to_comad({j: pad[:, i] for i, j in enumerate(MANICAST)})
    H = np.stack([mk[t:t + n] for t in range(len(kp))])
    W = np.stack([waist[t:t + n] for t in range(len(kp))])
    t_hist = np.arange(-(n - 1), 1) / fps
    out = np.concatenate([fc.forecast(H[a:a + 512], W[a:a + 512], t_hist, t_out) for a in range(0, len(H), 512)])
    fut = fc.from_comad(out[:, :, 1:7])                             # markers 1..6 = MANICAST joints, [T, M, 6, 3]
    now = fc.from_comad(H[:, -1, 1:7])                              # the same markers now (offsets cancel)
    return (fut - now[:, None]).astype(np.float32)


def neutral_contract(schema: dict, xi: np.ndarray) -> dict:
    """The v2 contract for T frames: xi [T, M, J, 3] and every other field neutral."""
    T = len(xi)
    K, W, P = len(schema['targets']), len(schema['who']), len(schema['phases'])
    p_who, p_target = np.tile(one_hot(0, W), (T, 1)), np.tile(one_hot(0, K), (T, 1))
    return dict(p_who=p_who, p_target=p_target, c_who=p_who.copy(), c_target=p_target.copy(),
                occupancy=np.zeros((T, 7), np.float32), tte=np.zeros((T, 1), np.float32),
                tte_std=np.zeros((T, 1), np.float32), phase=np.tile(one_hot(0, P), (T, 1)), xi=xi,
                confidence=confidence(p_target, p_who))


class OnlineManiCast:
    """Closed-loop source: the scenario's MANICAST keypoints at each policy step (the instant collect_demos records
    them), the forecast of the last 0.4 s, the neutral contract of one frame."""

    name = 'manicast_v6'

    def __init__(self, schema: dict, fps: float = 20.0, forecaster=None):
        if list(schema['keypoints']) != MANICAST:
            raise ValueError(f"schema keypoints {schema['keypoints']} != {MANICAST}")
        self.schema, self.fps = schema, float(fps)
        self.fc = forecaster or load_forecaster()
        self.t_out = horizons(schema)
        self.n = history_len(self.fps)
        self.hist: list[np.ndarray] = []

    def reset(self) -> None:
        self.hist = []

    def observe(self, scenario) -> dict:
        kp = scenario.human_keypoints(MANICAST).astype(np.float32)
        self.hist = (self.hist + [kp])[-self.n:]
        window = np.stack(self.hist)                 # forecast_xi pads the missing start with the first frame
        xi = forecast_xi(window, self.fc, self.fps, self.t_out)[-1:]
        return {k: v[0] for k, v in neutral_contract(self.schema, xi).items()}

"""Causal intent perception (INTENT_ACT_GUIDE_v2.md §3.2-3.3, §9.2-9.3): PerfectPerceptionSource and PredictedSource.

Pipeline per frame t, using keypoints <= t only:
  active hand    the hand farthest from its pose in the first frame (rest)
  kinematics     velocity / acceleration of the active hand (finite differences), causal onset = speed rising above
                 onset_speed after the hand was still
  p_gesture      perfect: the true gesture class, its probability ramping from uniform at the detected onset to p_max
                 after ramp_s (no ground-truth timing); predicted: a small MLP on a keypoint window (cross-fitted)
  p_point        pointing ray index base -> index tip, angle theta_k to every target, exp(-theta^2 / 2 sigma^2)
  p_reach        constant-velocity end point of the hand after horizon_s (shortened to the stopping distance
                 v^2 / 2|a| while it decelerates), softmax(-distance / reach_tau) over the places the hand is getting
                 closer to (distance with the height down-weighted; regions: distance to their box); a still hand
                 within at_place of a place is evidence for the places that close; a still hand away from every place
                 or one moving away from all of them gives no reach evidence (the filter keeps its belief)
  p_target       Bayes filter: T(p_{t-1}) * likelihood, T keeps the target with 1 - switch_eps
  p_who          from p_gesture (point -> robot, reach -> human, palm_up -> joint, rest -> none)
  tte            time to the gesture's goal pose from kinematics; tte_std from target entropy and speed jitter
  phase          rules: rest / prepare (accelerating away) / stroke (decelerating) / hold (still, away) / retract
  xi, occupancy  MotionForecaster (constant velocity; InteRACT later), box around the moving hand
  c_*            IntentTracker on the above
PredictedSource first passes the keypoints through NoiseModel (parameters TODO(user): measured with
scripts/record_real.py); until then noise_scale = 0 and the results are marked "chưa hiệu chỉnh".
"""
from __future__ import annotations
from collections import deque
import numpy as np
import torch
from torch import nn
from intent_policy.intent.contract import one_hot
from intent_policy.intent.labels import hand_indices, waypoint_offsets
from intent_policy.intent.places import target_boxes, target_matrix
from intent_policy.intent.sources import HumanObs, finish
from intent_policy.intent.tracker import IntentTracker, RobotState

WHO_OF_GESTURE = {'rest': 'none', 'point': 'robot', 'reach': 'human', 'palm_up': 'joint', 'palm_forward': 'robot'}
WINDOW = 10                        # frames of keypoint history seen by the gesture classifier
HEIGHT_WEIGHT = 0.3                # reach distances: vertical error weight (places differ mostly in x, y)
APPROACH_M = 0.002                 # a place counts as approached when the hand got this much closer in 2 frames
AT_PLACE_M = 0.08                  # a still wrist this close to a place (weighted distance) is at that place, and a
                                   # moving one gives reach evidence only for places this close to its end point


def gesture_labels(T: int, segments: list[dict], schema: dict) -> np.ndarray:
    """Ground-truth gesture class per frame: the segment's gesture over [t_onset, t_end], rest otherwise."""
    g = np.full(T, schema['gestures'].index('rest'), np.int64)
    for s in sorted(segments, key=lambda s: s['t_onset']):
        g[s['t_onset']:s['t_end'] + 1] = schema['gestures'].index(s['gesture'])
    return g


def angles_deg(origin: np.ndarray, direction: np.ndarray, points: np.ndarray) -> np.ndarray:
    v = points - origin
    c = (v @ direction) / (np.linalg.norm(v, axis=-1) * np.linalg.norm(direction) + 1e-12)
    return np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))


class MotionForecaster:
    """Future keypoint offsets (M, J, 3) from the keypoint history. Interface for InteRACT (TODO: not integrated)."""

    def forecast(self, history: list[np.ndarray], fps: float) -> np.ndarray:
        raise NotImplementedError


class CVMForecaster(MotionForecaster):
    def __init__(self, horizon_s: float, M: int, fps: float):
        self.t = waypoint_offsets(fps, horizon_s, M) / fps                   # (M,) seconds

    def forecast(self, history, fps):
        if len(history) < 2:
            return np.zeros((len(self.t),) + history[-1].shape, np.float32)
        v = (history[-1] - history[-2]) * fps if len(history) < 3 else (history[-1] - history[-3]) * fps / 2
        return (self.t[:, None, None] * v[None]).astype(np.float32)


class NoiseModel:
    """Keypoint noise of a real hand tracker: jitter (m), hand dropout probability (the last value is held), latency
    (s) and a pointing-direction error (deg, rotates the index tip about the index base). Parameters TODO(user):
    measured with scripts/record_real.py; `scale` multiplies them (0 = clean)."""

    def __init__(self, params: dict | None, scale: float, fps: float, seed: int):
        p = params or {}
        self.jitter, self.drop = scale * float(p.get('jitter_m', 0.0)), min(1.0, scale * float(p.get('dropout_prob', 0.0)))
        self.delay = int(round(scale * float(p.get('latency_s', 0.0)) * fps))
        self.angle = np.radians(scale * float(p.get('angle_deg', 0.0)))
        self.rng = np.random.default_rng(seed)

    def apply(self, kp: np.ndarray, hands: dict, schema: dict) -> np.ndarray:
        """kp (T, J, 3) -> noisy keypoints (T, J, 3)."""
        out = kp.copy()
        if self.delay:
            out = np.concatenate([np.repeat(out[:1], self.delay, 0), out[:-self.delay]])
        if self.angle:
            names = schema['keypoints']
            for s in hands:
                b, t = names.index(f'{s}_index_base'), names.index(f'{s}_index_tip')
                axis = self.rng.normal(size=3)
                axis /= np.linalg.norm(axis)
                v = out[:, t] - out[:, b]
                out[:, t] = out[:, b] + v * np.cos(self.angle) + np.cross(axis, v) * np.sin(self.angle)
        if self.jitter:
            out = out + self.rng.normal(0.0, self.jitter, out.shape)
        if self.drop:
            for s, idx in hands.items():
                lost = self.rng.random(len(out)) < self.drop
                for t in np.flatnonzero(lost):
                    if t > 0:
                        out[t, idx] = out[t - 1, idx]
        return out.astype(np.float32)


class GestureMLP(nn.Module):
    def __init__(self, d_in: int, n_cls: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_cls))

    def forward(self, x):
        return self.net(x)


def gesture_features(kp: np.ndarray, hands: dict, fps: float) -> np.ndarray:
    """(T, F) causal features: the active hand's keypoints relative to its first-frame pose over the last WINDOW
    frames and its velocity, plus the pointing direction."""
    T = len(kp)
    rel = kp - kp[:1]
    act = np.array([max(hands, key=lambda s: np.linalg.norm(rel[t, hands[s]])) for t in range(T)])
    feats = []
    for t in range(T):
        idx = hands[act[t]]
        win = [rel[max(t - k, 0), idx].reshape(-1) for k in range(0, WINDOW, 2)]
        vel = (kp[t, idx] - kp[max(t - 1, 0), idx]).reshape(-1) * fps
        tip, base = kp[t, idx[2]], kp[t, idx[1]]
        d = (tip - base) / (np.linalg.norm(tip - base) + 1e-9)
        feats.append(np.concatenate(win + [vel, d]))
    return np.asarray(feats, np.float32)


def train_gesture_classifier(X: np.ndarray, y: np.ndarray, n_cls: int, seed: int = 0, epochs: int = 30) -> tuple:
    """Small MLP (CPU); returns (model, feature mean, feature std)."""
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xt, yt = torch.as_tensor((X - mu) / sd), torch.as_tensor(y)
    model = GestureMLP(X.shape[1], n_cls)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    w = torch.as_tensor(len(y) / (n_cls * np.maximum(np.bincount(y, minlength=n_cls), 1)), dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for b in torch.randperm(len(yt), generator=g).split(512):
            loss = nn.functional.cross_entropy(model(Xt[b]), yt[b], weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval(), mu, sd


@torch.no_grad()
def classifier_logits(clf: tuple, X: np.ndarray) -> np.ndarray:
    model, mu, sd = clf
    return model(torch.as_tensor((X - mu) / sd)).numpy()


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature minimising the NLL of the true classes (grid search)."""
    best, best_t = np.inf, 1.0
    for T in np.exp(np.linspace(np.log(0.25), np.log(8.0), 40)):
        z = logits / T
        z = z - z.max(1, keepdims=True)
        nll = -(z[np.arange(len(y)), y] - np.log(np.exp(z).sum(1))).mean()
        if nll < best:
            best, best_t = nll, float(T)
    return best_t


def softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = z / T
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


class PerfectPerceptionSource:
    """Causal perception on clean keypoints (§3.2 with §9.2-9.3). episode_meta: segments (only for the gesture CLASS),
    keypoints are fed by step(). `gesture_probs` (T, G) replaces the ramped true class (PredictedSource)."""
    name = 'perfect'

    def __init__(self, schema: dict, gesture_probs: np.ndarray | None = None, target_temperature: float = 1.0):
        s = self.schema = schema
        self.cfg = dict(s['perfect'])
        self.fps = float(s['fps'])
        self.targets = target_matrix(s)                                           # (K, 3), row 0 NaN
        boxes = target_boxes(s)
        self.boxes = {s['targets'].index(k) - 1: b for k, b in boxes.items()}       # index into targets[1:]
        self.K, self.W, self.G, self.P = len(s['targets']), len(s['who']), len(s['gestures']), len(s['phases'])
        self.g_idx = {g: i for i, g in enumerate(s['gestures'])}
        self.who_of = np.array([one_hot(s['who'].index(WHO_OF_GESTURE[g]), self.W) for g in s['gestures']])
        self.hands = hand_indices(s)
        self.hand_all = sorted(i for idx in self.hands.values() for i in idx)
        self.forecaster = CVMForecaster(s['horizon_s'], s['n_waypoints'], self.fps)
        self.tracker = IntentTracker.from_schema(s)
        self.gesture_probs = gesture_probs
        self.target_temperature = float(target_temperature)

    def reset(self, episode_meta: dict) -> None:
        self.true_gesture = None if self.gesture_probs is not None else \
            gesture_labels(len(episode_meta['keypoints']), episode_meta['segments'], self.schema)
        self.hist: deque = deque(maxlen=6)
        self.rest = None
        self.prior = one_hot(0, self.K)
        self.onset_t, self.still, self.speeds, self.class_since, self.ramp_start = None, True, deque(maxlen=5), 0, None
        self.tracker.reset()

    # ------------------------------------------------------------------ pieces
    def _p_gesture(self, t: int) -> np.ndarray:
        if self.gesture_probs is not None:
            return self.gesture_probs[t].astype(np.float64)
        true = int(self.true_gesture[t])
        if t == 0 or true != int(self.true_gesture[t - 1]):          # a new gesture: its ramp starts at its own onset
            self.class_since = t
            self.ramp_start = self.onset_t if self.onset_t is not None and self.onset_t >= t - 2 else None
        elif self.ramp_start is None and self.onset_t is not None and self.onset_t >= self.class_since:
            self.ramp_start = self.onset_t                            # first onset detected after the class change
        u = np.full(self.G, 1.0 / self.G)
        if true == self.g_idx['rest']:
            ramp = 1.0
        elif self.ramp_start is None:                                 # no onset detected for this gesture yet
            ramp = 0.0
        else:
            ramp = min(1.0, (t - self.ramp_start) / (self.cfg['ramp_s'] * self.fps))
        p_true = 1.0 / self.G + ramp * (self.cfg['p_max'] - 1.0 / self.G)
        p = np.full(self.G, (1.0 - p_true) / (self.G - 1))
        p[true] = p_true
        return p if ramp > 0 else u

    def _likelihood(self, kp, hand, vel, p_g, accel_along: float) -> np.ndarray:
        """Target likelihood over K: `none` weighted by the rest probability, places by p_point * p_reach."""
        idx = self.hands[hand]
        names = self.schema['keypoints']
        g = int(np.argmax(p_g))
        pts = self.targets[1:]
        if g == self.g_idx['point']:
            b, tip = kp[names.index(f'{hand}_index_base')], kp[names.index(f'{hand}_index_tip')]
            th = angles_deg(tip, tip - b, pts)
            p_point = np.exp(-0.5 * (th / self.cfg['cone_sigma_deg']) ** 2)
        else:
            p_point = np.ones(len(pts))
        p_reach = np.ones(len(pts))
        speed = float(np.linalg.norm(vel))
        reaching = g in (self.g_idx['reach'], self.g_idx['palm_up'])
        wrist = names.index(f'{hand}_wrist')                     # the wrist marks the reached place best (calibration)
        if reaching and speed <= self.cfg['still_speed_m_s']:
            d = self._place_dist(kp[wrist])
            if d.min() <= AT_PLACE_M:
                p_reach = np.exp(-(d - d.min()) / self.cfg['reach_tau_m'])
        elif reaching and len(self.hist) >= 3:
            reach = speed * self.schema['horizon_s']
            if accel_along < -1e-3:
                reach = min(reach, speed ** 2 / (2.0 * -accel_along))
            end = kp[wrist] + vel / (speed + 1e-9) * reach
            d = self._place_dist(end)
            ok = (self._place_dist(kp[wrist]) < self._place_dist(self.hist[-3][wrist]) - APPROACH_M) & (d <= AT_PLACE_M)
            if ok.any():
                p_reach = np.where(ok, np.exp(-(d - d[ok].min()) / self.cfg['reach_tau_m']), 1e-3)
        self.informative = bool(np.ptp(p_point) > 0 or np.ptp(p_reach) > 0)
        lik = p_point ** self.cfg['w_point'] * p_reach ** self.cfg['w_reach']
        lik = lik / lik.sum()
        active = 1.0 - p_g[self.g_idx['rest']]
        return np.r_[1.0 - active, active * lik] + 1e-9

    def _place_dist(self, x: np.ndarray) -> np.ndarray:
        """Distance from point x to every place (targets[1:]), height down-weighted; regions: to their box."""
        w = np.array([1.0, 1.0, HEIGHT_WEIGHT])
        d = np.linalg.norm((self.targets[1:] - x) * w, axis=-1)
        for i, (lo, hi) in self.boxes.items():
            d[i] = np.linalg.norm((x - np.clip(x, lo, hi)) * w)
        return d

    def _phase(self, t, kp, hand, speed, accel_along, away: float) -> int:
        P = {p: i for i, p in enumerate(self.schema['phases'])}
        moving = speed > self.cfg['still_speed_m_s']
        if not moving:
            return P['hold'] if away > 0.05 else P['rest']
        if len(self.hist) >= 2:
            d_prev = np.linalg.norm(self.hist[-2][self.hands[hand]].mean(0) - self.rest[self.hands[hand]].mean(0))
            if away < d_prev - 1e-4:
                return P['retract']
        return P['prepare'] if accel_along > 0 else P['stroke']

    # ------------------------------------------------------------------ step
    def step(self, obs: HumanObs, robot: RobotState) -> dict:
        s, t, kp = self.schema, obs.t, obs.keypoints.astype(np.float64)
        if self.rest is None:
            self.rest = kp.copy()
        self.hist.append(kp)
        dist = {h: float(np.linalg.norm(kp[idx].mean(0) - self.rest[idx].mean(0))) for h, idx in self.hands.items()}
        hand = max(dist, key=dist.get)
        idx = self.hands[hand]
        c = lambda k: self.hist[-1 - k][idx].mean(0) if len(self.hist) > k else self.hist[0][idx].mean(0)
        vel = (c(0) - c(1)) * self.fps
        vel_prev = (c(1) - c(2)) * self.fps
        speed = float(np.linalg.norm(vel))
        accel_along = float(np.dot((vel - vel_prev) * self.fps, vel / (speed + 1e-9)))
        self.speeds.append(speed)
        if speed < self.cfg['still_speed_m_s']:                                    # causal onset detection: armed
            self.still = True                                                       # while still, fires when the
        elif speed > self.cfg['onset_speed_m_s'] and self.still:                   # speed exceeds onset_speed
            self.onset_t, self.still = t, False
        p_g = self._p_gesture(t)
        lik = self._likelihood(kp, hand, vel, p_g, accel_along)
        eps = self.cfg['switch_eps'] if self.informative else 0.0         # no new place evidence: keep the belief
        prior = (1.0 - eps) * self.prior + eps / self.K
        if self.prior[0] > 0.5 and lik[0] < 0.5:                           # leaving `none`: let the places compete again
            prior = np.r_[prior[0], np.maximum(prior[1:], eps / self.K + 1e-6)]
        post = prior * lik
        post = post / post.sum()
        self.prior = post
        p_target = softmax(np.log(post + 1e-12), self.target_temperature) if self.target_temperature != 1.0 else post
        p_who = p_g @ self.who_of
        # tte: time to the gesture's goal pose
        g = int(np.argmax(p_g))
        tmax = float(s['tte_max_s'])
        if g == self.g_idx['rest']:
            tte = tmax
        elif speed < self.cfg['still_speed_m_s']:
            tte = 0.0
        elif g == self.g_idx['point']:
            tte = speed / -accel_along if accel_along < -1e-3 else (t - (self.onset_t or t)) / self.fps + 0.3
        else:
            tgt = int(np.argmax(p_target[1:])) + 1
            tte = float(np.linalg.norm(self.targets[tgt] - kp[idx].mean(0))) / max(speed, 1e-3)
        h = -(p_target * np.log(p_target + 1e-12)).sum() / np.log(self.K)
        jitter = float(np.std(self.speeds) / (np.mean(self.speeds) + 1e-6)) if len(self.speeds) > 1 else 0.0
        tte_std = min(tte, tmax) * (0.5 * h + min(jitter, 1.0))
        xi = self.forecaster.forecast(list(self.hist), self.fps)
        phase = one_hot(self._phase(t, kp, hand, speed, accel_along, dist[hand]), self.P)
        human = float(p_who.max()) if int(np.argmax(p_who)) == s['who'].index('human') else 0.0
        c_out = self.tracker.step(p_target, p_who, p_g, robot, kp[self.hand_all])
        return finish(s, p_who / p_who.sum(), p_target, c_out, kp, xi, tte, tte_std, phase, human)


class PredictedSource(PerfectPerceptionSource):
    """§3.3: noisy keypoints (NoiseModel) + learned gesture probabilities (temperature-scaled), CVM xi."""
    name = 'predicted'

    def __init__(self, schema: dict, noise: NoiseModel, clf: tuple, gesture_T: float, target_T: float):
        super().__init__(schema, gesture_probs=np.zeros((0, len(schema['gestures']))), target_temperature=target_T)
        self.noise, self.clf, self.gesture_T = noise, clf, gesture_T

    def reset(self, episode_meta: dict) -> None:
        kp = self.noise.apply(np.asarray(episode_meta['keypoints'], np.float32), self.hands, self.schema)
        self.noisy = kp
        X = gesture_features(kp, self.hands, self.fps)
        self.gesture_probs = softmax(classifier_logits(self.clf, X), self.gesture_T)
        super().reset(episode_meta)

    def step(self, obs: HumanObs, robot: RobotState) -> dict:
        return super().step(HumanObs(obs.t, self.noisy[obs.t]), robot)


def make_sources(name: str, schema: dict, episodes: dict, segments: dict, train: list[int]) -> dict:
    """Column prefix -> factory(episode index). predicted: 5-fold cross-fitted gesture classifier on the training
    episodes (an episode is labelled by the model that did not see it; other episodes by the all-train model),
    temperatures fitted on the out-of-fold predictions, R noisy replicas."""
    if name == 'perfect':
        return {'intent_pp': lambda e: PerfectPerceptionSource(schema)}
    pc = schema['predicted']
    hands, fps = hand_indices(schema), float(schema['fps'])
    feats = {e: gesture_features(ep['keypoints'], hands, fps) for e, ep in episodes.items()}
    labels = {e: gesture_labels(len(ep['keypoints']), segments.get(e, []), schema) for e, ep in episodes.items()}
    G = len(schema['gestures'])
    if len(train) < 2:
        raise ValueError('the predicted source needs at least 2 training episodes (cross-fitting)')
    folds = [train[i::min(5, len(train))] for i in range(min(5, len(train)))]
    fold_of = {e: k for k, f in enumerate(folds) for e in f}
    clfs = []
    for k, f in enumerate(folds):
        rest = [e for e in train if e not in f]
        clfs.append(train_gesture_classifier(np.concatenate([feats[e] for e in rest]),
                                             np.concatenate([labels[e] for e in rest]), G, seed=k))
    full = train_gesture_classifier(np.concatenate([feats[e] for e in train]), np.concatenate([labels[e] for e in train]), G, seed=5)
    oof = np.concatenate([classifier_logits(clfs[fold_of[e]], feats[e]) for e in train])
    gesture_T = fit_temperature(oof, np.concatenate([labels[e] for e in train]))
    acc = float((oof.argmax(1) == np.concatenate([labels[e] for e in train])).mean())
    print(f'predicted: gesture classifier out-of-fold accuracy {acc:.3f}, temperature {gesture_T:.2f}', flush=True)
    target_T = 1.0                     # p_target is a Bayes posterior; its temperature is fitted by calibrate_intent
    lo, hi = pc.get('noise_scale_range', [0.0, 0.0])
    out = {}
    for r in range(int(pc.get('replicas', 3))):
        def factory(e, r=r):
            rng = np.random.default_rng(1000 * r + e)
            scale = float(rng.uniform(lo, hi)) if hi > lo else float(lo)
            clf = clfs[fold_of[e]] if e in fold_of else full
            return PredictedSource(schema, NoiseModel(pc.get('noise'), scale, fps, 7919 * r + e), clf, gesture_T, target_T)
        out[f'intent_pr{r}'] = factory
    return out

"""Belief over the human's target (docs/requirements/INTENT_ACT_v3.2.md §3-4).

The decision variable of the belief-space early expert (`trigger: belief`) and, from M4 on, the TARGET token of the
v3.2 intent. Causal: frame t uses the keypoints up to t, the human's current gesture CLASS (perfect recognition of the
class, with the onset latency ramp of the perfect source) and the scene layout. Never the human's choice itself.

  HandKinematics   active hand (farthest from its first-frame pose), its speed and the causal motion onset
  GestureRamp      class probabilities: uniform at the onset, rising to p_max after ramp_s
  pointing_likelihood   DeePoint-format pointing (Nakamura et al., ICCV 2023): p_point and the direction index base ->
                   tip; L(g) = p_point * exp(-theta_g^2 / 2 sigma^2) + (1 - p_point)
  ReachLibrary     Pérez-D'Arpino & Shah (ICRA 2015): per target a library of reach motions, one diagonal Gaussian per
                   normalised time step; OnlineDTW aligns the observed prefix to it and scores the mean log-likelihood
  GoalInference    MaxEnt IOC goal inference with a Euclidean cost (Dragan & Srinivasa 2013; Javdani et al. 2015):
                   P(g | S -> Q) ∝ exp(-(path(S -> Q) + |Q - g| - |S - g|) / beta) on the wrist; the goal of g is the
                   wrist's end pose when reaching g (mean over the demonstrations), works from any start
  ReachPredictor   method library_then_ioc: the library when one of its start clusters covers the candidates, else
                   GoalInference; or ioc / library only
  PointingBelief   log b_t = lam * log b_{t-1} + log L_t (forgetting Bayes filter), uniform prior
  ReachBelief      during a reach: b_t = p_g * posterior_t + (1 - p_g) * b_start (b_start: the belief when this reach
                   began; the posterior already integrates the whole prefix, so it is not filtered again); otherwise
                   it forgets towards uniform: normalise(b_{t-1} ** lam)
  OnlineBelief     the above for a running scenario (expert side): candidates from the task, the instruction and the
                   layout; objects / zones / stop probability per frame
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from intent_policy.intent.labels import hand_indices
from intent_policy.intent.perception import angles_deg

# The human motion that selects a target -> its gesture class (perfect class recognition). Holds after a reach
# (lift_block, waiting_cup, take_object, ...) and other motions (move_over_cup, withdrawals) select nothing: `rest`.
SELECTION_CLASS = {'point_object': 'point', 'point_place': 'point', 'reach_out': 'palm_up', 'reach_block': 'reach',
                   'stop_signal': 'palm_forward', 'withdraw_change': 'withdraw'}
# `withdraw`: the hand going back after a change of mind (T3: the cube carried back to its spot; a pointing hand pulled
# back). Not one of the schema's token gestures: the belief's gesture classes are schema gestures + this one.
EXTRA_CLASSES = ['withdraw']
REACH_LABELS = ('reach_block', 'reach_out')       # motions in the reach library (their target = the segment target)
# Holds that follow a selecting reach (schema sim_segments `hold`): the gesture class goes on (its probability keeps
# rising), the reach posterior stays at its value when the motion ended (the hand held out / the cube taken).
SELECTION_HOLD = {'waiting_offer': 'palm_up', 'lift_block': 'reach', 'waiting_cup': 'reach'}


# ---------------------------------------------------------------------- kinematics and gesture class
class HandKinematics:
    """Active hand and causal motion onset (as PerfectPerceptionSource): the onset fires when the hand's speed rises
    above `onset_speed` after it was below `still_speed`."""

    def __init__(self, schema: dict, onset_speed: float, still_speed: float):
        self.hands = hand_indices(schema)
        self.fps = float(schema['fps'])
        self.onset_speed, self.still_speed = float(onset_speed), float(still_speed)
        self.reset()

    def reset(self) -> None:
        self.rest = None
        self.prev: list[np.ndarray] = []
        self.still = True
        self.onset_t: int | None = None
        self.speed = 0.0
        self.hand = next(iter(self.hands))

    def step(self, t: int, kp: np.ndarray) -> None:
        kp = np.asarray(kp, np.float64)
        if self.rest is None:
            self.rest = kp.copy()
        dist = {h: float(np.linalg.norm(kp[i].mean(0) - self.rest[i].mean(0))) for h, i in self.hands.items()}
        self.hand = max(dist, key=dist.get)
        idx = self.hands[self.hand]
        self.speed = float(np.linalg.norm(kp[idx].mean(0) - self.prev[-1][idx].mean(0)) * self.fps) if self.prev else 0.0
        self.prev = (self.prev + [kp])[-3:]
        if self.speed < self.still_speed:
            self.still = True
        elif self.speed > self.onset_speed and self.still:
            self.onset_t, self.still = t, False

    @property
    def moving(self) -> bool:
        return self.speed > self.onset_speed


class GestureRamp:
    """Probability of the current gesture class: uniform at the detected onset of the motion, the true class rising
    linearly to p_max after ramp_s (PerfectPerceptionSource._p_gesture, online). A class change while the hand is
    already moving starts the ramp at once (no new onset can be detected then); so does any motion above still_speed
    within the new class (slow, short gestures never reach onset_speed)."""

    def __init__(self, classes: list[str], ramp_s: float, p_max: float, fps: float):
        self.classes, self.G = list(classes), len(classes)
        self.ramp_frames, self.p_max = float(ramp_s) * float(fps), float(p_max)
        self.reset()

    def reset(self) -> None:
        self.true: int | None = None
        self.class_since = 0
        self.ramp_start: int | None = None

    def step(self, t: int, cls: str, kin: HandKinematics) -> np.ndarray:
        c = self.classes.index(cls)
        if self.true is None or c != self.true:
            self.true, self.class_since = c, t
            onset = kin.onset_t
            self.ramp_start = onset if onset is not None and onset >= t - 2 else (t if kin.moving else None)
        elif self.ramp_start is None and kin.onset_t is not None and kin.onset_t >= self.class_since:
            self.ramp_start = kin.onset_t
        elif self.ramp_start is None and kin.speed > kin.still_speed:
            self.ramp_start = t                 # a small, slow gesture (a 6 cm reach) may never exceed onset_speed
        if cls == 'rest':
            ramp = 1.0
        elif self.ramp_start is None:
            return np.full(self.G, 1.0 / self.G)
        else:
            ramp = min(1.0, (t - self.ramp_start) / self.ramp_frames)
        p_true = 1.0 / self.G + ramp * (self.p_max - 1.0 / self.G)
        p = np.full(self.G, (1.0 - p_true) / (self.G - 1))
        p[c] = p_true
        return p

    @property
    def active_since(self) -> int | None:
        """Frame the current gesture's ramp started (None: not yet)."""
        return self.ramp_start


# ---------------------------------------------------------------------- pointing (DeePoint format)
def pointing_likelihood(base: np.ndarray, tip: np.ndarray, positions: np.ndarray, p_point: float,
                        sigma_deg: float) -> np.ndarray:
    """L(g) = p_point * exp(-theta_g^2 / 2 sigma^2) + (1 - p_point); theta_g = angle between the ray base -> tip
    (origin at the tip) and the direction from the tip to candidate g. 1 everywhere when not pointing."""
    th = angles_deg(np.asarray(tip, float), np.asarray(tip, float) - np.asarray(base, float), np.asarray(positions, float))
    return p_point * np.exp(-0.5 * (th / sigma_deg) ** 2) + (1.0 - p_point)


# ---------------------------------------------------------------------- reach library (Pérez-D'Arpino & Shah 2015)
def reach_features(kp: np.ndarray, rest: np.ndarray, idx: list[int], names: list[str]) -> np.ndarray:
    """(…, 6): the active hand's wrist and index tip relative to their first-frame (rest) position."""
    hand = names[idx[0]].split('_', 1)[0]
    w, tip = names.index(f'{hand}_wrist'), names.index(f'{hand}_index_tip')
    kp, rest = np.asarray(kp, np.float64), np.asarray(rest, np.float64)
    return np.concatenate([kp[..., w, :] - rest[w], kp[..., tip, :] - rest[tip]], axis=-1)


def time_normalise(x: np.ndarray, n: int) -> np.ndarray:
    """(T, D) -> (n, D) by linear interpolation over normalised time [0, 1]."""
    x = np.asarray(x, np.float64)
    if len(x) == 1:
        return np.repeat(x, n, 0)
    src, dst = np.linspace(0.0, 1.0, len(x)), np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(dst, src, x[:, d]) for d in range(x.shape[1])], 1)


def _fit_one(xs: list[np.ndarray], n_steps: int, var_floor_m: float) -> tuple[np.ndarray, np.ndarray]:
    X = np.stack([time_normalise(x, n_steps) for x in xs])                        # (N, n, D)
    var = X.var(0, ddof=1) if len(X) > 1 else np.zeros_like(X[0])
    return X.mean(0), np.maximum(var, var_floor_m ** 2)


def start_clusters(starts: list[np.ndarray], radius: float) -> tuple[list[np.ndarray], list[int]]:
    """Greedy clustering of start positions (wrist, 3-d): a start joins the first centre within `radius`."""
    centres, labels = [], []
    for s in starts:
        k = next((i for i, c in enumerate(centres) if np.linalg.norm(s - c) < radius), None)
        if k is None:
            centres.append(np.asarray(s, float))
            k = len(centres) - 1
        labels.append(k)
    return centres, labels


class ReachLibrary:
    """Library of motions: mean and variance (n_steps, D) of the time-normalised demonstrations, per target `g` (all
    demonstrations) and per (start cluster c, target) `g@c` (Pérez-D'Arpino & Shah define a motion class by its initial
    position and its target: one Gaussian per time step only fits motions from one start). `start[g@c]`: the
    cluster's wrist start (3,)."""

    def __init__(self, mean: dict[str, np.ndarray], var: dict[str, np.ndarray], start: dict[str, np.ndarray] | None = None,
                 radius: float = 0.06, temperature: dict[str, float] | None = None, goal: dict[str, np.ndarray] | None = None):
        self.mean, self.var, self.start, self.radius = mean, var, dict(start or {}), float(radius)
        self.temperature = dict(temperature or {})          # '+'.join(sorted(classes)) -> T; 'ioc:<same>' -> beta [m]
        self.goal = dict(goal or {})                        # target -> mean world wrist position at the reach's end

    @classmethod
    def fit(cls, demos: dict[str, list[np.ndarray]], n_steps: int, var_floor_m: float, families: list[list[str]] = (),
            radius: float = 0.06, min_demos: int = 3, ends: dict[str, list[np.ndarray]] | None = None) -> 'ReachLibrary':
        """demos: target -> feature trajectories; ends: target -> world wrist positions at the reaches' ends."""
        mean, var, start = {}, {}, {}
        goal = {g: np.mean(np.asarray(e, float), 0) for g, e in (ends or {}).items() if len(e)}
        for g, xs in demos.items():
            mean[g], var[g] = _fit_one(xs, n_steps, var_floor_m)
        for fam in families:
            items = [(g, x) for g in fam for x in demos.get(g, [])]
            if not items:
                continue
            centres, labels = start_clusters([x[0, :3] for _, x in items], radius)
            for c, centre in enumerate(centres):
                for g in fam:
                    xs = [x for (h, x), lab in zip(items, labels) if h == g and lab == c]
                    if len(xs) >= min_demos:
                        key = f'{g}@{c}'
                        mean[key], var[key] = _fit_one(xs, n_steps, var_floor_m)
                        start[key] = centre
        return cls(mean, var, start, radius, goal=goal)

    @property
    def classes(self) -> list[str]:
        return sorted(k for k in self.mean if '@' not in k)

    def cluster_for(self, classes: list[str], x0: np.ndarray) -> str | None:
        """The nearest start cluster (within 2 radius of x0, features with the wrist first) that has every class."""
        best, best_d = None, 2.0 * self.radius
        clusters = {k.split('@')[1] for k in self.start}
        for c in clusters:
            if all(f'{g}@{c}' in self.mean for g in classes):
                d = float(np.linalg.norm(np.asarray(x0)[:3] - self.start[f'{classes[0]}@{c}']))
                if d < best_d:
                    best, best_d = c, d
        return best

    def keys_for(self, classes: list[str], x0: np.ndarray) -> list[str]:
        """Library keys to score `classes` with: the matching start cluster's, else the all-starts libraries."""
        best = self.cluster_for(classes, x0)
        return [g if best is None else f'{g}@{best}' for g in classes]

    def nll(self, key: str, x: np.ndarray) -> np.ndarray:
        """Negative log-likelihood of one sample x (D,) at every library step (n,)."""
        m, v = self.mean[key], self.var[key]
        return 0.5 * (((x - m) ** 2) / v + np.log(2 * np.pi * v)).sum(-1)


class OnlineDTW:
    """Online alignment of an observed prefix to each class's library (steps: stay, +1, +2 library steps; the prefix
    starts at library step 0-2). score_g = min_j D_g[j] / k = mean negative log-likelihood along the best path, the
    log posterior is -score / T (uniform prior, eq. 7 of Pérez-D'Arpino & Shah; T = the library's calibrated temperature
    for these classes, 1 if none). The start cluster is chosen from the first sample."""

    def __init__(self, library: ReachLibrary, classes: list[str], temperature: float | None = None):
        self.lib, self.classes = library, list(classes)
        self.T = float(temperature if temperature is not None else library.temperature.get('+'.join(sorted(classes)), 1.0))
        self.reset()

    def reset(self) -> None:
        self.D: dict[str, np.ndarray] = {}
        self.k = 0
        self.keys: list[str] = list(self.classes)

    def step(self, x: np.ndarray) -> np.ndarray:
        """Add sample x (D,); returns the posterior over `classes`."""
        if self.k == 0:
            self.keys = self.lib.keys_for(self.classes, x)
        score = []
        for g, key in zip(self.classes, self.keys):
            c = self.lib.nll(key, x)
            if self.k == 0:
                D = np.full_like(c, np.inf)
                D[:3] = c[:3]
            else:
                prev = self.D[g]
                best = np.minimum.reduce([prev, np.r_[np.inf, prev[:-1]], np.r_[np.inf, np.inf, prev[:-2]]])
                D = c + best
            self.D[g] = D
            score.append(D.min() / (self.k + 1))
        self.k += 1
        self.logits = -np.asarray(score)                        # uncalibrated (temperature 1), for calibration
        z = self.logits / self.T
        z = np.exp(z - z.max())
        return z / z.sum()


class GoalInference:
    """MaxEnt IOC goal inference with a Euclidean cost (Dragan & Srinivasa 2013 'Legibility and predictability of
    robot motion'; Javdani et al. 2015): a human moving from S to Q is taken to be noisily optimal towards the goal g,
    P(g | S -> Q) ∝ exp(-(C(S -> Q) + C*(Q -> g) - C*(S -> g)) / beta), C = path length travelled (wrist), C* = straight
    line. `beta`: calibrated temperature (metres). Works for any start; needs no demonstrations."""

    def __init__(self, targets: dict[str, np.ndarray], classes: list[str], beta: float):
        self.G = np.asarray([targets[g] for g in classes], float)
        self.beta = float(beta)
        self.reset()

    def reset(self) -> None:
        self.S = self.prev = None
        self.path = 0.0

    def step(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, float)
        if self.S is None:
            self.S = self.prev = q
        self.path += float(np.linalg.norm(q - self.prev))
        self.prev = q
        self.logits = -(self.path + np.linalg.norm(self.G - q, axis=1) - np.linalg.norm(self.G - self.S, axis=1))
        z = self.logits / self.beta
        z = np.exp(z - z.max())
        return z / z.sum()


class ReachPredictor:
    """Posterior over the candidate targets of one reach: the library (Pérez-D'Arpino & Shah) when one of its start
    clusters covers every candidate from where this reach began, else MaxEnt IOC goal inference. `mode`: which one."""

    METHODS = ('library_then_ioc', 'ioc', 'library')

    def __init__(self, library: ReachLibrary, classes: list[str], method: str = 'library_then_ioc'):
        if method not in self.METHODS:
            raise ValueError(f'reach method must be one of {self.METHODS}, got {method!r}')
        self.lib, self.classes, self.method = library, list(classes), method
        key = '+'.join(sorted(classes))
        self.ioc = GoalInference(library.goal, classes, library.temperature.get(f'ioc:{key}', 0.001))
        self.dtw = OnlineDTW(library, classes)
        self.reset()

    def reset(self) -> None:
        self.mode: str | None = None
        self.dtw.reset()
        self.ioc.reset()

    def step(self, x: np.ndarray, wrist: np.ndarray) -> np.ndarray:
        """x: reach_features (6,); wrist: world position of the active wrist (3,)."""
        if self.mode is None:
            covered = self.lib.cluster_for(self.classes, x) is not None
            self.mode = 'ioc' if self.method == 'ioc' or (self.method == 'library_then_ioc' and not covered) else 'library'
        return self.dtw.step(x) if self.mode == 'library' else self.ioc.step(wrist)


def fold_of(seed: int, folds: int) -> int:
    """Cross-fitting fold of an episode: by its list cell (seeds are cell base + attempt, the bases are multiples of
    10; the late and the early episode of a cell share the seed, hence the fold)."""
    return (int(seed) // 10) % int(folds)


def save_libraries(path: str | Path, folds: dict[int, ReachLibrary], meta: dict) -> None:
    arrays = {}
    for f, lib in folds.items():
        for g in lib.mean:
            arrays[f'f{f}/{g}/mean'], arrays[f'f{f}/{g}/var'] = lib.mean[g], lib.var[g]
            if g in lib.start:
                arrays[f'f{f}/{g}/start'] = lib.start[g]
            if g in lib.goal:
                arrays[f'f{f}/{g}/goal'] = lib.goal[g]
        arrays[f'r{f}'] = np.array(lib.radius)
        for k, T in lib.temperature.items():
            arrays[f'T{f}/{k}'] = np.array(T)
    arrays['meta'] = np.array(repr(meta))
    np.savez_compressed(path, **arrays)


def load_library(path: str | Path, fold: int) -> ReachLibrary:
    z = np.load(path)
    parts: dict[str, dict] = {'mean': {}, 'var': {}, 'start': {}, 'goal': {}}
    temperature = {}
    for key in z.files:
        if key.startswith(f'f{fold}/'):
            _, g, k = key.split('/')
            parts[k][g] = z[key]
        elif key.startswith(f'T{fold}/'):
            temperature[key.split('/', 1)[1]] = float(z[key])
    if not parts['mean']:
        raise KeyError(f'fold {fold} not in {path}')
    return ReachLibrary(parts['mean'], parts['var'], parts['start'], float(z[f'r{fold}']), temperature, parts['goal'])


def n_folds(path: str | Path) -> int:
    return len({k.split('/')[0] for k in np.load(path).files if k.startswith('f')})


# ---------------------------------------------------------------------- beliefs over one candidate set
def _normalise_log(z: np.ndarray) -> np.ndarray:
    return z - (z.max() + np.log(np.exp(z - z.max()).sum()))


class PointingBelief:
    """Forgetting Bayes filter over candidate keys: log b_t = lam * log b_{t-1} + log L_t, normalised."""

    def __init__(self, keys: list[str], lam: float):
        self.keys, self.lam = list(keys), float(lam)
        self.logb = _normalise_log(np.zeros(len(self.keys)))

    def update(self, lik: np.ndarray) -> None:
        self.logb = _normalise_log(self.lam * self.logb + np.log(np.maximum(lik, 1e-300)))

    @property
    def b(self) -> dict[str, float]:
        return dict(zip(self.keys, np.exp(self.logb).tolist()))


class ReachBelief:
    """During a reach: b = p_g * posterior + (1 - p_g) * b_start. The reach-library posterior already integrates every
    frame of the reach so far, so it is mixed (weighted by the probability that the motion is that reach), never
    multiplied in again frame after frame (that would count the same evidence many times). Between reaches the belief
    forgets towards uniform: normalise(b ** lam)."""

    def __init__(self, keys: list[str], lam: float):
        self.keys, self.lam = list(keys), float(lam)
        self.p = np.full(len(self.keys), 1.0 / len(self.keys))
        self.start: np.ndarray | None = None

    def update(self, posterior: np.ndarray | None, p_g: float, new_reach: bool = False) -> None:
        if posterior is None:
            self.start = None
            self.p = np.exp(_normalise_log(self.lam * np.log(np.maximum(self.p, 1e-300))))
            return
        if new_reach or self.start is None:
            self.start = self.p.copy()
        self.p = p_g * np.asarray(posterior) + (1.0 - p_g) * self.start

    @property
    def b(self) -> dict[str, float]:
        return dict(zip(self.keys, self.p.tolist()))


# ---------------------------------------------------------------------- online belief of a running scenario
class OnlineBelief:
    """Per-frame belief of a running scenario for the belief-space expert.

    Candidates (fixed per episode, from what the policy can see as well: task, instruction, layout):
      T1  pointing over the instructed kind's objects + the place zones P1 / P2 (one joint set: a pointing at a zone
          cannot raise an object)
      T2  pointing over the instructed kind's objects; reach (palm up) over the hand zones H1 / H2
      T3  reach over the cubes' spots U1 / U2 -> their paired cups
      T4  none
    `objects()` / `zones()`: candidate -> probability; `p_stop`: probability of the 'wait' gesture."""

    def __init__(self, scenario, schema: dict, cfg: dict, library_path: str | Path | None = None):
        self.sc, self.schema, self.cfg = scenario, schema, dict(cfg)
        self.names = list(schema['keypoints'])
        self.fps = float(schema['fps'])
        self.kin = HandKinematics(schema, self.cfg['onset_speed_m_s'], self.cfg['still_speed_m_s'])
        self.classes = list(schema['gestures']) + EXTRA_CLASSES
        self.ramp = GestureRamp(self.classes, self.cfg['ramp_s'], self.cfg['p_max'], self.fps)
        self.library_path = library_path
        self.code = scenario.cfg.task_code

    def reset(self) -> None:
        sc, var = self.sc, self.sc.variation
        spec = var['spec']
        lam = float(self.cfg['lambda'])
        self.kin.reset()
        self.ramp.reset()
        self.t = 0
        self.onset_seen = False
        self.p_gesture = np.full(len(self.classes), 1.0 / len(self.classes))
        self.point = self.reach = self.dtw = None
        self.object_keys: list[str] = []
        self.zone_keys: list[str] = []
        self.reach_map: dict[str, str] = {}             # reach class (U1 / H1 ...) -> object or zone key
        desc = self.schema['instruction']['object_desc']
        if self.code in ('T1', 'T2'):
            kind = desc[spec['target']]
            self.object_keys = [k for k in spec['layout'] if desc.get(k) == kind]
        if self.code == 'T1':
            self.zone_keys = [k for k, z in sc.zones.items() if z['kind'] == 'place']
        if self.object_keys or self.zone_keys:
            self.point_keys = self.object_keys + self.zone_keys
            self.point = PointingBelief(self.point_keys, lam)
        if self.code == 'T2':
            self.zone_keys = [k for k, z in sc.zones.items() if z['kind'] == 'hand']
            self.reach_map = {k: k for k in self.zone_keys}
            self.reach_kind = 'zone'
        elif self.code == 'T3':
            cup_of = var['pairs']                                     # cube -> cup
            self.reach_map = {f'U{i + 1}': cup_of[b] for i, b in enumerate(spec['u_blocks'])}
            self.object_keys = list(self.reach_map.values())
            self.reach_kind = 'object'
        if self.reach_map:
            fold = fold_of(self.sc.seed, n_folds(self.library_path))   # out-of-fold: late / early share seeds
            self.library = load_library(self.library_path, fold)
            self.reach_classes = list(self.reach_map)
            self.reach = ReachBelief(self.reach_classes, lam)
            self.dtw = ReachPredictor(self.library, self.reach_classes, self.cfg['reach'].get('method', 'library_then_ioc'))
            self.dtw_start: int | None = None
            self.last_post: np.ndarray | None = None

    # -- positions
    def _point_positions(self) -> np.ndarray:
        sc = self.sc
        pos = [sc.start_pos[k] if k in sc.start_pos else np.r_[sc.zone_pos(k)[:2], sc.table_z] for k in self.point_keys]
        return np.asarray(pos, float)

    # -- per frame
    def step(self) -> None:
        sc, t = self.sc, self.t
        kp = sc.human_keypoints(self.names)
        self.kin.step(t, kp)
        self.onset_seen = self.onset_seen or self.kin.onset_t is not None
        stage = sc.human.stage
        cls = SELECTION_CLASS.get(stage) or SELECTION_HOLD.get(stage, 'rest')
        self.p_gesture = self.ramp.step(t, cls, self.kin)
        G = self.classes
        idx = self.kin.hands[self.kin.hand]
        if self.point is not None:
            base = kp[self.names.index(f'{self.kin.hand}_index_base')]
            tip = kp[self.names.index(f'{self.kin.hand}_index_tip')]
            self.point.update(pointing_likelihood(base, tip, self._point_positions(), float(self.p_gesture[G.index('point')]),
                                                  float(self.cfg['cone_sigma_deg'])))
        if self.reach is not None:
            reach_cls = 'palm_up' if self.code == 'T2' else 'reach'
            running = cls == reach_cls and self.ramp.active_since is not None
            post, new = None, False
            if running:
                if self.dtw_start != self.ramp.active_since:
                    self.dtw_start, new = self.ramp.active_since, True
                    self.dtw.reset()
                    self.last_post = None
                if stage in REACH_LABELS:                       # the reach motion itself: the next sample
                    wrist = kp[self.names.index(f'{self.kin.hand}_wrist')]
                    self.last_post = self.dtw.step(reach_features(kp, self.kin.rest, idx, self.names), wrist)
                post = self.last_post                           # its hold: the posterior at the motion's end
                running = post is not None
            self.reach.update(post, float(self.p_gesture[G.index(reach_cls)]) if running else 0.0, new)
        self.t += 1

    # -- outputs
    def objects(self) -> dict[str, float]:
        if self.code == 'T3' and self.reach is not None:
            return {self.reach_map[k]: p for k, p in self.reach.b.items()}
        if self.point is None:
            return {}
        b = self.point.b
        return {k: b[k] for k in self.object_keys}

    def zones(self) -> dict[str, float]:
        if self.code == 'T2' and self.reach is not None:
            return {self.reach_map[k]: p for k, p in self.reach.b.items()}
        if self.point is None:
            return {}
        b = self.point.b
        return {k: b[k] for k in self.zone_keys}

    @property
    def p_stop(self) -> float:
        return float(self.p_gesture[self.classes.index('palm_forward')])

    @property
    def p_withdraw(self) -> float:
        return float(self.p_gesture[self.classes.index('withdraw')])

    def snapshot(self) -> dict:
        """Compact per-frame record (expert stage overlay, analysis)."""
        r = lambda d: {k: round(v, 3) for k, v in d.items()}
        return dict(objects=r(self.objects()), zones=r(self.zones()), p_stop=round(self.p_stop, 3),
                    p_withdraw=round(self.p_withdraw, 3),
                    p_gesture=np.round(self.p_gesture, 3).tolist(),
                    reach_mode=None if self.dtw is None else self.dtw.mode)

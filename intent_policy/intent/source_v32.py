"""Intent contract v3.2 (docs/requirements/INTENT_ACT_v3.2.md §5; intent/contract.py FIELDS_V32), causal.

`IntentV32` computes one frame from a scenario: the live one (closed-loop evaluation, the same object the belief expert
reads) or `ReplayScenario` (labels of a recorded episode: its keypoints, human stages and layout). Both run the same
`OnlineBelief`, so the labels a policy trains on equal the belief the early expert acted on.

  p_target   OnlineBelief.distribution() mapped onto the schema targets (objects -> their slots); T4: none
  p_gesture  the belief's gesture classes (schema gestures + withdraw); p_who = p_gesture @ who-of-gesture
  c_who, c_target   IntentTracker on p_target / p_who (memory of what was asked and not done yet)
  phase, tte, tte_std, xi, occupancy   kinematic rules / constant-velocity forecast of the active hand (as the v2 perfect
             source); point_dir: unit ray index base -> tip + its origin (DeePoint format)
  min_dist   closest hand keypoint to the robot TCP [m]; hand_hist: the last HAND_HIST keypoint frames minus the current
  human_present   1 (the simulated hand is always tracked; a real tracker may lose it)
"""
from __future__ import annotations
from collections import deque
from types import SimpleNamespace
import numpy as np
from intent_policy.intent.belief import EXTRA_CLASSES, OnlineBelief
from intent_policy.intent.contract import HAND_HIST, one_hot
from intent_policy.intent.labels import hand_indices
from intent_policy.intent.perception import CVMForecaster
from intent_policy.intent.places import target_positions
from intent_policy.intent.sources import occupancy
from intent_policy.intent.tracker import HoldingDetector, IntentTracker, RobotState
from intent_policy.utils import load_yaml

WHO_OF_CLASS = {'rest': 'none', 'point': 'robot', 'reach': 'human', 'palm_up': 'joint', 'palm_forward': 'robot',
                'withdraw': 'human'}


class ReplayScenario:
    """What OnlineBelief reads from a scenario, rebuilt from a recorded episode (meta/hri_episodes.jsonl entry +
    human.keypoints). `set_frame(t)` selects the frame."""

    def __init__(self, meta: dict, keypoints: np.ndarray, schema: dict):
        scene = load_yaml('configs/scene/common.yaml')
        layout = load_yaml(scene['layout'])
        self.variation, self.seed = meta['variation'], int(meta['seed'])
        self.cfg = SimpleNamespace(task_code=meta['task_code'])
        self.zones, self.table_z = layout['zones'], float(scene['table_top_z'])
        objs = scene['objects_catalog']
        half = lambda o: float(o['size'][1]) / 2 if o['shape'] == 'cup' else float(o['size'][-1])
        self.start_pos = {k: np.array([*self.variation['object_xy'][k], self.table_z + half(objs[k])], float)
                          for k in self.variation['on_table']}
        self.kp = np.asarray(keypoints, np.float64)
        self.stages = [s['name'] for s in meta['human_stages'] for _ in range(s['end'] - s['start'] + 1)]
        self.human = SimpleNamespace(stage=self.stages[0])
        self.t = 0

    def set_frame(self, t: int) -> None:
        self.t = t
        self.human.stage = self.stages[min(t, len(self.stages) - 1)]

    def zone_pos(self, key: str) -> np.ndarray:
        z = self.zones[key]
        return np.array([*z['xy'], self.table_z + float(z.get('height', 0.0))])

    def human_keypoints(self, names) -> np.ndarray:
        return self.kp[self.t]


class IntentV32:
    """One v3.2 contract frame per call to `step(state)` (observation.state [10]: joints, gripper opening, TCP xyz)."""
    name = 'belief'

    def __init__(self, scenario, schema: dict, belief_cfg: dict, library_path):
        self.sc, self.schema = scenario, schema
        self.fps, self.tmax = float(schema['fps']), float(schema['tte_max_s'])
        self.belief = OnlineBelief(scenario, schema, belief_cfg, library_path)
        self.tracker = IntentTracker.from_schema(schema)
        self.forecaster = CVMForecaster(schema['horizon_s'], schema['n_waypoints'], self.fps)
        self.hands = hand_indices(schema)
        self.hand_all = sorted(i for idx in self.hands.values() for i in idx)
        self.targets = list(schema['targets'])
        self.K, self.W, self.P = len(self.targets), len(schema['who']), len(schema['phases'])
        self.classes = list(schema['gestures']) + EXTRA_CLASSES
        self.who_of = np.array([one_hot(schema['who'].index(WHO_OF_CLASS[c]), self.W) for c in self.classes])
        self.pos = target_positions(schema)
        self.names = list(schema['keypoints'])
        self.ph = {p: i for i, p in enumerate(schema['phases'])}

    def reset(self) -> None:
        self.belief.reset()
        self.tracker.reset()
        self.holding = HoldingDetector(**self.schema['holding'])
        self.hist: deque = deque(maxlen=max(HAND_HIST, 3))
        self.speeds: deque = deque(maxlen=5)
        layout = self.sc.variation['spec'].get('layout') or {}
        self.slot = dict(layout)                                    # object key -> its slot (target name)

    # -- pieces
    def _p_target(self) -> np.ndarray:
        d = self.belief.distribution()
        p = np.zeros(self.K)
        for k, v in d.items():
            p[self.targets.index(self.slot.get(k, k))] += v
        if p.sum() <= 0:
            p[0] = 1.0                                              # T4: no choice to make
        return p / p.sum()

    def _phase(self, speed: float, accel_along: float, away: float, d_prev: float) -> int:
        still = float(self.schema['belief']['still_speed_m_s'])
        if speed <= still:
            return self.ph['hold'] if away > 0.05 else self.ph['rest']
        if away < d_prev - 1e-4:
            return self.ph['retract']
        return self.ph['prepare'] if accel_along > 0 else self.ph['stroke']

    def step(self, state: np.ndarray) -> dict:
        b = self.belief
        b.step()
        kp = b.kp
        self.hist.append(kp)
        state = np.asarray(state, np.float64)
        tcp, grip = state[7:10], float(state[6])
        robot = RobotState(tcp=tcp, gripper=grip, holding=self.holding.step(grip))
        p_g = np.asarray(b.p_gesture, np.float64)
        p_who = p_g @ self.who_of
        p_who = p_who / p_who.sum()
        p_target = self._p_target()
        p_g5 = p_g[:len(self.schema['gestures'])].copy()
        p_g5[0] += p_g[len(self.schema['gestures']):].sum()           # the tracker knows the schema gestures only
        c = self.tracker.step(p_target, p_who, p_g5, robot, kp[self.hand_all])
        # kinematics of the active hand
        idx = self.hands[b.kin.hand]
        cen = lambda k: self.hist[-1 - k][idx].mean(0) if len(self.hist) > k else self.hist[0][idx].mean(0)
        vel, vel_prev = (cen(0) - cen(1)) * self.fps, (cen(1) - cen(2)) * self.fps
        speed = float(np.linalg.norm(vel))
        accel_along = float(np.dot((vel - vel_prev) * self.fps, vel / (speed + 1e-9)))
        self.speeds.append(speed)
        rest = b.kin.rest[idx].mean(0)
        away, d_prev = float(np.linalg.norm(cen(0) - rest)), float(np.linalg.norm(cen(1) - rest))
        phase = one_hot(self._phase(speed, accel_along, away, d_prev), self.P)
        g = self.classes[int(np.argmax(p_g))]
        if g == 'rest':
            tte = self.tmax
        elif speed < float(self.schema['belief']['still_speed_m_s']):
            tte = 0.0
        elif g == 'point':
            start = b.ramp.active_since
            tte = speed / -accel_along if accel_along < -1e-3 else (b.t - (start if start is not None else b.t)) / self.fps + 0.3
        else:
            goal = self.targets[int(np.argmax(p_target))]
            tte = float(np.linalg.norm(self.pos[goal] - kp[idx].mean(0))) / max(speed, 1e-3) if goal != 'none' else self.tmax
        tte = float(np.clip(tte, 0.0, self.tmax))
        h = -(p_target * np.log(p_target + 1e-12)).sum() / np.log(self.K)
        jitter = float(np.std(self.speeds) / (np.mean(self.speeds) + 1e-6)) if len(self.speeds) > 1 else 0.0
        tte_std = float(np.clip(tte * (0.5 * h + min(jitter, 1.0)), 0.0, self.tmax))
        xi = self.forecaster.forecast(list(self.hist), self.fps)
        human = float(p_who.max()) if int(np.argmax(p_who)) == self.schema['who'].index('human') else 0.0
        occ = occupancy(kp, xi, self.hands, float(self.schema.get('occupancy_margin_m', 0.05)), human)
        hand = b.kin.hand
        base, tip = kp[self.names.index(f'{hand}_index_base')], kp[self.names.index(f'{hand}_index_tip')]
        ray = (tip - base) / max(float(np.linalg.norm(tip - base)), 1e-9)
        hist = np.stack([self.hist[max(len(self.hist) - 1 - k, 0)] - kp for k in range(HAND_HIST)])
        f32 = lambda x: np.asarray(x, np.float32)
        return dict(p_who=f32(p_who), p_target=f32(p_target), c_who=f32(c['c_who']), c_target=f32(c['c_target']),
                    phase=f32(phase), p_gesture=f32(p_g / p_g.sum()), point_dir=f32(np.r_[ray, tip]),
                    occupancy=f32(occ), min_dist=f32([np.linalg.norm(kp[self.hand_all] - tcp, axis=1).min()]),
                    tte=f32([tte]), tte_std=f32([tte_std]), xi=f32(xi), hand_hist=f32(hist), human_present=f32([1.0]))


def label_episode(meta: dict, keypoints: np.ndarray, state: np.ndarray, schema: dict, belief_cfg: dict,
                  library_path) -> dict[str, np.ndarray]:
    """v3.2 fields (T, ...) of a recorded episode, frame by frame as the live expert / policy would see them."""
    sc = ReplayScenario(meta, keypoints, schema)
    src = IntentV32(sc, schema, belief_cfg, library_path)
    src.reset()
    frames = []
    for t in range(len(keypoints)):
        sc.set_frame(t)
        frames.append(src.step(state[t]))
    return {k: np.stack([f[k] for f in frames]).astype(np.float32) for k in frames[0]}

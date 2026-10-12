"""Intent contract v3.2 (docs/requirements/INTENT_ACT_v3.2.md §5; intent/contract.py FIELDS_V32) from the measuring
device, causal.

`IntentV32` computes one frame from a running scenario and the device's belief (intent/device.DeviceBelief):
  - data collection: the very belief the expert acts on (shared, stepped by the expert); scripts/collect_demos.py saves
    the frames and scripts/build_intent_labels.py writes them as the dataset columns `intent_pd`;
  - closed-loop evaluation (policies/policy_agent.py): its own belief, stepped here, with the device settings the
    dataset was collected with (rule, fold models).
So the labels a policy trains on equal the belief the expert acted on, and the policy sees the same device when tested.

  p_target   DeviceBelief.distribution() mapped onto the schema targets (objects -> their slots); T4: none
  p_gesture  the device's class probabilities (its `none` = the schema's `rest`; schema gestures + withdraw);
             p_who = p_gesture @ who-of-class
  c_who, c_target   IntentTracker on p_target / p_who (memory of what was asked and not done yet)
  phase, tte, tte_std, xi, occupancy   kinematic rules / constant-velocity forecast of the active hand (thresholds of the
             schema's `perfect` source); point_dir: unit ray index base -> tip + its origin (DeePoint format)
  min_dist   closest hand keypoint to the robot TCP [m]; hand_hist: the last HAND_HIST keypoint frames minus the current
  human_present   1 (the simulated hand is always tracked; a real tracker may lose it)
"""
from __future__ import annotations
from collections import deque
import numpy as np
from intent_policy.intent.contract import HAND_HIST, one_hot
from intent_policy.intent.kinematics import HandKinematics
from intent_policy.intent.labels import hand_indices
from intent_policy.intent.perception import CVMForecaster
from intent_policy.intent.places import target_positions
from intent_policy.intent.sources import occupancy
from intent_policy.intent.tracker import HoldingDetector, IntentTracker, RobotState

WHO_OF_CLASS = {'rest': 'none', 'point': 'robot', 'reach': 'human', 'palm_up': 'joint', 'palm_forward': 'robot',
                'withdraw': 'human'}
EXTRA_CLASSES = ['withdraw']
DEVICE_FPS = 15.0                    # the device sees every 15 fps camera frame (intent/device.py)


class IntentV32:
    """One v3.2 contract frame per call to `step(state)` (observation.state [10]: joints, gripper opening, TCP xyz).
    `belief`: a DeviceBelief stepped by its owner (the expert); None: an own DeviceBelief from `device_cfg`."""
    name = 'device'

    def __init__(self, scenario, schema: dict, belief=None, device_cfg: dict | None = None):
        self.sc, self.schema = scenario, schema
        self.owns = belief is None
        if belief is None:
            from intent_policy.intent.device import DeviceBelief
            belief = DeviceBelief(scenario, schema, device_cfg if device_cfg is not None else schema['device'])
        self.belief = belief
        self.fps, self.tmax = float(schema['fps']), float(schema['tte_max_s'])
        perf = schema['perfect']
        self.still = float(perf['still_speed_m_s'])
        self.kin = HandKinematics(schema, perf['onset_speed_m_s'], perf['still_speed_m_s'])
        self.tracker = IntentTracker.from_schema(schema)
        self.forecaster = CVMForecaster(schema['horizon_s'], schema['n_waypoints'], self.fps)
        self.hands = hand_indices(schema)
        self.hand_all = sorted(i for idx in self.hands.values() for i in idx)
        self.targets = list(schema['targets'])
        self.K, self.W, self.P = len(self.targets), len(schema['who']), len(schema['phases'])
        self.classes = list(schema['gestures']) + EXTRA_CLASSES
        dev = ['rest' if c == 'none' else c for c in belief.classes]
        if dev != self.classes:
            raise ValueError(f'device classes {belief.classes} do not match the schema gestures + withdraw {self.classes}')
        self.who_of = np.array([one_hot(schema['who'].index(WHO_OF_CLASS[c]), self.W) for c in self.classes])
        self.pos = target_positions(schema)
        self.names = list(schema['keypoints'])
        self.ph = {p: i for i, p in enumerate(schema['phases'])}

    def reset(self) -> None:
        if self.owns:
            self.belief.reset()
        self.kin.reset()
        self.t = 0
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
        if speed <= self.still:
            return self.ph['hold'] if away > 0.05 else self.ph['rest']
        if away < d_prev - 1e-4:
            return self.ph['retract']
        return self.ph['prepare'] if accel_along > 0 else self.ph['stroke']

    def step(self, state: np.ndarray) -> dict:
        b = self.belief
        if self.owns:
            b.step()
        kp = np.asarray(self.sc.human_keypoints(self.names), np.float64)
        self.kin.step(self.t, kp)
        self.t += 1
        self.hist.append(kp)
        state = np.asarray(state, np.float64)
        tcp, grip = state[7:10], float(state[6])
        robot = RobotState(tcp=tcp, gripper=grip, holding=self.holding.step(grip))
        p_g = np.asarray(b.p_gesture, np.float64)
        if not np.all(np.isfinite(p_g)):                         # no device output yet (no pre-roll): at rest
            p_g = one_hot(0, len(self.classes)).astype(np.float64)
        p_who = p_g @ self.who_of
        p_who = p_who / p_who.sum()
        p_target = self._p_target()
        p_g5 = p_g[:len(self.schema['gestures'])].copy()
        p_g5[0] += p_g[len(self.schema['gestures']):].sum()           # the tracker knows the schema gestures only
        c = self.tracker.step(p_target, p_who, p_g5, robot, kp[self.hand_all])
        # kinematics of the active hand
        idx = self.hands[self.kin.hand]
        cen = lambda k: self.hist[-1 - k][idx].mean(0) if len(self.hist) > k else self.hist[0][idx].mean(0)
        vel, vel_prev = (cen(0) - cen(1)) * self.fps, (cen(1) - cen(2)) * self.fps
        speed = float(np.linalg.norm(vel))
        accel_along = float(np.dot((vel - vel_prev) * self.fps, vel / (speed + 1e-9)))
        self.speeds.append(speed)
        rest = self.kin.rest[idx].mean(0)
        away, d_prev = float(np.linalg.norm(cen(0) - rest)), float(np.linalg.norm(cen(1) - rest))
        phase = one_hot(self._phase(speed, accel_along, away, d_prev), self.P)
        g = self.classes[int(np.argmax(p_g))]
        if g == 'rest':
            tte = self.tmax
        elif speed < self.still:
            tte = 0.0
        elif g == 'point':                                       # elapsed: device frames since the pointing was detected
            tte = speed / -accel_along if accel_along < -1e-3 else b.class_run.get('point', 0) / DEVICE_FPS + 0.3
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
        hand = self.kin.hand
        base, tip = kp[self.names.index(f'{hand}_index_base')], kp[self.names.index(f'{hand}_index_tip')]
        ray = (tip - base) / max(float(np.linalg.norm(tip - base)), 1e-9)
        hist = np.stack([self.hist[max(len(self.hist) - 1 - k, 0)] - kp for k in range(HAND_HIST)])
        f32 = lambda x: np.asarray(x, np.float32)
        return dict(p_who=f32(p_who), p_target=f32(p_target), c_who=f32(c['c_who']), c_target=f32(c['c_target']),
                    phase=f32(phase), p_gesture=f32(p_g / p_g.sum()), point_dir=f32(np.r_[ray, tip]),
                    occupancy=f32(occ), min_dist=f32([np.linalg.norm(kp[self.hand_all] - tcp, axis=1).min()]),
                    tte=f32([tte]), tte_std=f32([tte_std]), xi=f32(xi), hand_hist=f32(hist), human_present=f32([1.0]))

"""Non-privileged belief for the v6 experts (docs/PIF.md §2.2, §2.4): the human's target from what the policy D also
gets - the mannequin keypoints (as rendered), the frozen fine-tuned ManiCast forecast, the visible candidates.

Readout = pif/v6_goal.py: inverse planning on the observed (+ forecast) wrist path (Dragan, Lee, Srinivasa 2013;
Baker, Saxe, Tenenbaum 2009), the pointing ray while the hand points, the stop term for goals within reach.
Same interface as intent/device.py DeviceBelief (objects / zones / complete / thresholds / ...), so the expert's
belief branch (scripted_expert._follow_belief: hedge, commit, redirect) is reused unchanged.

Commit level (no hand-set threshold): commit to the most probable candidate when the expected cost of a wrong commit
is not larger than the cost of waiting, b* = C_wrong / (C_wrong + C_wait) (Bayes decision with two actions;
Javdani et al. 2018 act on the goal distribution before that). C_wrong = (2 travel_dz + nearest-neighbour spacing) /
TCP speed: the descent made for nothing, the rise, the move across; C_wait = the measured time from the motion start
until the target stays the most probable (configs/pif/pif.yaml v6_expert). Late rule (model A): the same belief, but
only once the gesture is complete (the hand stopped after a motion).
"""
from __future__ import annotations

import numpy as np

from pif.config import load_config
from pif.forecast import ManiCastForecaster
from pif.v6_goal import posterior
from intent_policy.sim.base_env import ROOT

NEED = ['l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist']
KPS = NEED + ['r_index_base', 'r_index_tip']


class GoalBelief:
    def __init__(self, scenario, rule: str, speed: float, travel_dz: float):
        self.sc = scenario
        cfg = load_config(str(ROOT / 'configs/pif/pif.yaml'), [])
        self.cfg, ve = cfg, cfg['v6_expert']
        self.ve = ve
        if rule not in ('early', 'late'):
            raise ValueError(f'GoalBelief rule must be early | late, got {rule!r}')
        self.rule = rule
        self.fps = float(cfg['fps'])
        self.every = int(round(1.0 / (self.fps * scenario.dt)))       # control ticks per belief frame (20 Hz -> 10 Hz)
        self.beta = 1000.0 / float(ve['beta_from_error_mm'])
        self.kappa = 1.0 / np.radians(float(ve['pointing_sigma_deg'])) ** 2
        self.fc = ManiCastForecaster(dict(cfg, forecast=dict(cfg['forecast'], checkpoint_path=ve['checkpoint_path'])))
        self.t_out = np.arange(1, 11) / 10.0
        self.speed, self.travel_dz = float(speed), float(travel_dz)
        self.approach_star = 0.0 if rule == 'early' else None    # early: act on the distribution from the start
        self.pause_timeout_s = 0.0
        self.last_point_t = None
        self.top, self.probs = None, None                        # (device API, unused here)

    # -- candidates of the running v6 scenario
    def reset(self) -> None:
        sc = self.sc
        plan, spec = getattr(sc, 'v6_plan', None), sc.variation['spec']
        kind_of = {'coaster': 'coaster', 'fetch_return': 'hand'}
        if sc.cfg.id == 't1_choose_item':
            self.obj_keys = [k for k in spec['layout'] if k in sc.on_table]
            self.zone_keys = [k for k, z in sc.zones.items() if z['kind'] == 'place']
        elif plan in kind_of:
            self.obj_keys = [sc.variation['target_object']]           # one object, always needed (workflow)
            self.zone_keys = [k for k, z in sc.zones.items() if z['kind'] == kind_of[plan]]
        else:                                                        # T4: no target choice (yield on the forecast)
            self.obj_keys, self.zone_keys = [], []
        self.obj_pos = np.asarray([sc.start_pos[k] for k in self.obj_keys], float).reshape(-1, 3)
        from intent_policy.scenarios.base_scenario import BaseScenario
        self.zone_pos = np.asarray([BaseScenario.zone_pos(sc, k) for k in self.zone_keys], float).reshape(-1, 3)
        reach = lambda P: np.array([np.linalg.norm(sc.body_model.clamp(q) - q) < 0.01 for q in P], bool)
        self.obj_reach, self.zone_reach = reach(self.obj_pos), reach(self.zone_pos)
        self.b_star_obj = self._b_star(self.obj_pos)
        self.b_star_zone = self._b_star(self.zone_pos)
        self.kp, self.pointing, self.tick = [], [], 0
        self.b_obj = np.full(len(self.obj_keys), 1.0 / max(len(self.obj_keys), 1))
        self.b_zone = np.full(len(self.zone_keys), 1.0 / max(len(self.zone_keys), 1))
        self.forecast = None                                         # [10, 3] wrist 0.1 .. 1.0 s ahead (world)
        self.moving, self.peak, self.moved_from, self.onset = False, 0.0, None, None
        self.after_motion = False                                    # the hand stopped after its latest motion
        self.evidence = {}                                           # kind -> a gesture towards the candidates was seen

    def _b_star(self, P: np.ndarray) -> float:
        """b* = C_wrong / (C_wrong + C_wait) (module doc); one candidate: 0 (nothing to choose)."""
        if len(P) < 2:
            return 0.0
        d = np.linalg.norm(P[:, None, :2] - P[None, :, :2], axis=2)
        np.fill_diagonal(d, np.inf)
        c_wrong = (2 * self.travel_dz + float(d.min())) / self.speed
        c_wait = float(self.ve['c_wait_s'])
        return c_wrong / (c_wrong + c_wait)

    def thresholds(self) -> tuple[float, int, float]:
        return self.b_star_obj, 1, 0.5

    def commit_star(self, kind: str) -> float:
        """b* while the gesture goes on (waiting still brings evidence, C_wait = c_wait_s); once it is complete waiting
        brings nothing more (C_wait -> inf, b* -> 0): commit to the most probable candidate if there was evidence.
        Late rule (A): only the complete case."""
        if len(self.obj_keys if kind == 'object' else self.zone_keys) < 2:
            return 0.0                       # one candidate: nothing to choose
        if self.after_motion and self.evidence.get(kind):
            return 0.0
        if self.rule == 'late':
            return 2.0                       # never before the gesture is complete
        return self.b_star_obj if kind == 'object' else self.b_star_zone

    # -- per control tick
    def step(self) -> None:
        sc = self.sc
        self.tick += 1
        if (self.tick - 1) % self.every:
            return
        self.kp.append(sc.human_keypoints(KPS))
        self.pointing.append(sc.human.hand_shape == 'point')        # the rendered hand shape (observable)
        kp = np.asarray(self.kp)
        T = len(kp)
        wrist = kp[:, 5]
        v = float(np.linalg.norm(wrist[-1] - wrist[-2])) * self.fps if T > 1 else 0.0
        if not self.moving and self.moved_from is None:
            self.moved_from = wrist[-1].copy()                       # where the hand rests: a motion starts from here
        if not self.moving and np.linalg.norm(wrist[-1] - self.moved_from) > float(self.ve['moving_m']):
            self.moving, self.after_motion = True, False
            self.onset = self.moved_from                             # S of inverse planning = the motion's start
        self.peak = max(self.peak, v) if self.moving else 0.0
        stopped = self.moving and v < float(self.ve['still_fraction']) * self.peak
        f_stop = self._forecast_stopped(wrist[-1])                   # before the peak is reset below
        if stopped:                                                  # a motion ended: the gesture is complete
            self.after_motion = True
            self.moving, self.peak, self.moved_from = False, 0.0, None
        if T >= 5:
            mk, waist = self.fc.to_comad({k: kp[-5:, i] for i, k in enumerate(NEED)})
            p = self.fc.forecast(mk[None], waist[None], np.arange(-4, 1) / self.fps, self.t_out)
            self.forecast = self.fc.from_comad(p[0, :, 6])
        if not self.moving and not stopped:
            return                                                   # at rest: keep the belief of the last gesture
        S = self.onset[None]                                         # (Dragan 2013: the path from the motion's start)
        Q_obs = wrist[-1][None]
        Q_fc = self.forecast[-1][None] if self.forecast is not None else Q_obs
        ray = (kp[-1:, 6], kp[-1:, 7], np.array([self.pointing[-1]]))
        for kind, keys, pos, reach in (('object', self.obj_keys, self.obj_pos, self.obj_reach),
                                       ('zone', self.zone_keys, self.zone_pos, self.zone_reach)):
            if len(keys) < 2 or not self._open(kind):
                continue
            Q = Q_fc if self.ve['use_forecast'][kind] else Q_obs
            prog = np.linalg.norm(S - pos, axis=1) - np.linalg.norm(Q - pos, axis=1)
            if prog.max() < float(self.ve['moving_m']) and not self.pointing[-1]:
                continue                     # not heading for any candidate (withdrawing): no evidence, keep the belief
            st = np.array([stopped or (f_stop and self.ve['use_forecast'][kind])])   # observed stop always counts
            b = posterior(S, Q, pos, self.beta, ray, self.kappa, st, reach)[0]
            if kind == 'object':
                self.b_obj = b
            else:
                self.b_zone = b
            self.evidence[kind] = True

    def _open(self, kind: str) -> bool:
        """Task-protocol prior (observable), as the v3.2 device: T1's items are open while the robot holds nothing, its
        trays while it holds the item (the robot's own grasp); T3's spots once the coaster has left its place; T2's hand
        zones while the hand is an open palm (the v3.2 device's palm_up class)."""
        sc = self.sc
        if getattr(sc, 'v6_plan', None) == 'coaster' and kind == 'zone':   # T3: the spot is open once the coaster has
            c = sc.coaster                                                 #   left its place in U (it moves with the
            return bool(np.linalg.norm(sc.pos(c)[:2] - sc.start_pos[c][:2]) > float(self.ve['moving_m']))   # hand)
        if getattr(sc, 'v6_plan', None) == 'fetch_return' and kind == 'zone':   # T2: the hand zone is open while the
            return sc.human.hand_shape == 'offer'                             #   hand is an open, upturned palm (rendered)
        if sc.cfg.id != 't1_choose_item':
            return True
        holding = any(sc.grasp_state.values())
        return not holding if kind == 'object' else holding

    def _forecast_stopped(self, now: np.ndarray) -> bool:
        if self.forecast is None or self.peak <= 0:
            return False
        v_end = float(np.linalg.norm(self.forecast[-1] - self.forecast[-2])) / (self.t_out[-1] - self.t_out[-2])
        return v_end < float(self.ve['still_fraction']) * self.peak

    # -- device API used by the expert
    def objects(self) -> dict[str, float]:
        return {k: float(b) for k, b in zip(self.obj_keys, self.b_obj)} if self._open('object') else {}

    def zones(self) -> dict[str, float]:
        return {k: float(b) for k, b in zip(self.zone_keys, self.b_zone)} if self._open('zone') else {}

    def complete(self, kind: str) -> bool:
        """Late rule: the current gesture has ended (the hand stopped after its latest motion and has not moved off
        again). A single, always-needed object (T2 can, T3 cup) needs no gesture."""
        if kind == 'object' and len(self.obj_keys) == 1:
            return True
        return self.after_motion and bool(self.evidence.get(kind))

    @property
    def p_stop(self) -> float:
        return 0.0                                                   # v6 has no 'wait' gesture

    @property
    def p_withdraw(self) -> float:
        return 0.0

    def snapshot(self) -> dict:
        r = lambda d: {k: round(v, 3) for k, v in d.items()}
        return dict(objects=r(self.objects()), zones=r(self.zones()), rule=self.rule,
                    forecast_end=None if self.forecast is None else np.round(self.forecast[-1], 3).tolist())

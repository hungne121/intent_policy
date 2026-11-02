"""v7 C3 demonstrator (scenarios/c3_cloth_bowl.py): lift the bowl of the tray the cloth goes to, hold it aside, set it
on the cloth once the cloth is down and the hand has left. Two information rules on one workflow (PIF.md §8.12):

  observed: the tray is known once the carried cloth (visible) has stopped (still_fraction / hold_min_s of
            configs/pif/pif.yaml v6_expert, the stop rule of the v6 belief) within commit_radius_m of a tray's waiting
            point; no forecast.
  early:    the tray nearest to the predicted cloth position (ManiCast wrist forecast at +1 s plus the observed
            cloth-wrist offset) once it is within commit_radius_m (switching needs `switch_margin_m` more); the observed
            stop overrides the prediction. The controlled forecast fault of case C3 turns the forecast arm.

A new tray after a commit: before the grasp, go to the new bowl; holding the old bowl, put it back on its tray first
(the correction work of C4 after). Nothing reads the human's chosen tray: the cloth pose, the mannequin keypoints and
the forecast only. Separation monitoring as experts/separation.py (back away, never towards the human).
"""
from __future__ import annotations
import numpy as np
from intent_policy.experts.scripted_expert import ScriptedExpert, Step, GUARD_OCCUPIED_XY


class C3Expert(ScriptedExpert):
    RULES = ('observed', 'early')

    def reset(self) -> None:
        super().reset()
        sc, e = self.sc, self.sc.cfg.expert
        self.rule = e.get('rule', 'observed')
        if self.rule not in self.RULES:
            raise ValueError(f'expert.rule must be one of {self.RULES}')
        self.anticipate = True
        self.belief = None
        cont = getattr(self.mapper, 'continuous', False)
        self._speed = float(self.mapper.cfg.continuous['tcp_speed']) if cont else float(np.max(self.mapper.step)) / sc.dt
        self._gb = None
        if self.rule == 'early':
            from intent_policy.intent.goal_belief import GoalBelief
            self._gb = GoalBelief(sc, 'early', self._speed, self.travel - self.z0)
            self._gb.reset()
        from pif.config import load_config
        from intent_policy.sim.base_env import ROOT
        ve = load_config(str(ROOT / 'configs/pif/pif.yaml'), [])['v6_expert']
        self._moving_m, self._still, self._hold_s = float(ve['moving_m']), float(ve['still_fraction']), float(ve['hold_min_s'])
        self.commit_r, self.switch_m = float(e['commit_radius_m']), float(e['switch_margin_m'])
        self.hold_offset = np.asarray(e['hold_offset_xy'], float)
        self.source = np.asarray(sc.start_pos[sc.cloth][:2], float).copy()
        self._prev, self._peak, self._calm = None, 0.0, 0.0
        self.stopped_at = None
        self.predicted = None
        self.tray = None                     # committed tray
        self.carrying = None                 # tray whose bowl the robot holds (lifted from)
        self._make_way = None
        self._set_plan([Step('grip', 'open', until=lambda: sc.env.data.qpos[sc.env.gid] > 0.075, ticks=25, name='open'),
                        Step('hold', ticks=10 ** 6, name='await_tray')])

    # -- observation
    def _wait_point(self, tray):
        b = self.sc.cfg.human_behavior
        return np.asarray(self.sc.trays[tray], float) + np.asarray(b['wait_offset_xy'], float)

    def _nearest(self, p):
        d = {t: float(np.linalg.norm(p - self._wait_point(t))) for t in self.sc.trays}
        best = min(d, key=d.get)
        return best, d

    def _observe(self) -> None:
        sc = self.sc
        if self._gb is not None:
            self._apply_forecast_fault()
            self._gb.step()
        p = sc.pos(sc.cloth)[:2]
        if np.linalg.norm(p - self.source) < self._moving_m or sc.coaster_down():
            self._prev = p.copy()
            return
        v = float(np.linalg.norm(p - self._prev)) / sc.dt if self._prev is not None else 0.0
        self._prev = p.copy()
        self._peak = max(self._peak, v)
        self._calm = self._calm + sc.dt if v < self._still * self._peak else 0.0
        if self._calm >= self._hold_s - 1e-9:
            self.stopped_at = p.copy()
        elif v > self._still * self._peak and self.stopped_at is not None and np.linalg.norm(p - self.stopped_at) > self._moving_m:
            self.stopped_at, self._peak = None, v            # moving on (C4): the stop no longer tells the tray
        if self._gb is not None and self._gb.forecast is not None and self._gb.kp:
            wrist = np.asarray(self._gb.kp[-1][5], float)
            self.predicted = np.asarray(self._gb.forecast[-1][:2], float) + (p - wrist[:2])

    def _decide(self):
        """The tray the evidence points to now (None: not yet)."""
        if self.stopped_at is not None:
            t, d = self._nearest(self.stopped_at)
            return t if d[t] <= self.commit_r else self.tray
        if self.rule == 'early' and self.predicted is not None:
            t, d = self._nearest(self.predicted)
            if d[t] > self.commit_r:
                return self.tray
            if self.tray is not None and t != self.tray and d[self.tray] - d[t] < self.switch_m:
                return self.tray
            return t
        return self.tray

    def _apply_forecast_fault(self) -> None:
        sc, gb = self.sc, self._gb
        spec = sc.variation['c3'].get('forecast_fault')
        if spec is None or sc.realized_forecast_fault is not None:
            return
        seg = sc.human.segment
        if seg is None or seg.label != 'carry_cloth':
            return
        from intent_policy.intent.forecast_fault import ForecastFault
        fault = dict(start_s=round(float(sc.time), 4), end_s=round(float(seg.end_t + spec['extra_s']), 4),
                     yaw_deg=float(spec['yaw_deg']), arm='right', ramp_s=float(spec['ramp_s']))
        gb.forecast_fault = ForecastFault(**fault)
        sc.realized_forecast_fault = fault

    # -- plans
    def _hold_point(self, tray):
        return np.r_[np.asarray(self.sc.trays[tray], float) + self.hold_offset + self.tcp_obj_offset[:2], self.travel]

    def _lift(self):
        return Step('goto', lambda: np.r_[self.mapper.setpoint[:2], self.travel], name='lift')

    def _put_down(self, bowl, xy_fn, z_fn, name):
        sc = self.sc
        return [Step('goto', lambda: np.r_[xy_fn() + self.tcp_obj_offset[:2], self.travel], name=f'{name}_transport', guard=True),
                Step('goto', lambda: np.r_[xy_fn(), z_fn()] + self.tcp_obj_offset, name=f'{name}_lower', guard=True),
                Step('settle', ticks=15, name='settle_before_release'),
                Step('grip', 'open', until=lambda: not sc.grasp_state[bowl] and sc.env.data.qpos[sc.env.gid] > 0.06,
                     ticks=25, name='release'),
                Step('goto', lambda: np.r_[self.mapper.setpoint[:2], self.travel], name='retreat')]

    def _serve_plan(self, tray):
        sc = self.sc
        bowl = sc.bowls[tray]
        hh = lambda: float(sc.rest_height[bowl] - sc.table_z - float(sc.cfg.human_behavior['tray_top_m']))  # half height
        on_cloth_z = lambda: sc.pos(sc.cloth)[2] + sc.cloth_top / 2 + hh() + 0.003
        grasp = [g for g in self._grasp_steps(bowl) if g.name != 'open']
        return [Step('grip', 'open', until=lambda: sc.env.data.qpos[sc.env.gid] > 0.075, ticks=25, name='open')] + grasp + [
            Step('call', target=lambda: setattr(self, 'carrying', tray), name='carry'),
            Step('goto', lambda: self._hold_point(tray), name='hold_aside', guard=True),
            Step('wait', until=lambda: self._cloth_down_and_clear(tray), name='await_cloth')] + \
            self._put_down(bowl, lambda: sc.pos(sc.cloth)[:2], on_cloth_z, 'onto_cloth') + \
            [Step('call', target=lambda: setattr(self, 'carrying', None), name='done_carry')] + self._home_steps()

    def _put_back_plan(self, old, new):
        sc = self.sc
        bowl = sc.bowls[old]
        z = lambda: sc.rest_height[bowl] + 0.003
        return self._put_down(bowl, lambda: np.asarray(sc.trays[old], float), z, 'put_back') + \
            [Step('call', target=lambda: setattr(self, 'carrying', None), name='put_back_done')] + self._serve_plan(new)

    def _cloth_down_and_clear(self, tray) -> bool:
        sc = self.sc
        if ('cloth_on_' + tray) not in sc.protocol_done:
            return False
        return float(np.linalg.norm(sc.human.pos[:2] - np.asarray(sc.trays[tray], float))) >= GUARD_OCCUPIED_XY

    def _commit_tray(self, tray) -> None:
        sc = self.sc
        old, self.tray = self.tray, tray
        sc.robot_bowl = sc.bowls[tray]
        if self.carrying is not None and self.carrying != tray:
            self._commit(sc.bowls[tray], self._put_back_plan(self.carrying, tray))
        else:
            self._commit(sc.bowls[tray], self._serve_plan(tray))

    def _follow_intention_change(self) -> None:
        """No reading of the human's change-of-mind event (privileged)."""

    def _known_zone(self):
        return None                          # never the human's chosen tray (privileged); see _decide

    def _yield(self) -> bool:
        return False

    def _back_away(self):
        sc = self.sc
        tcp, hand = sc.tcp(), sc.human.pos
        u = tcp[:2] - hand[:2]
        u = u / max(float(np.linalg.norm(u)), 1e-9)
        return np.r_[tcp[:2] + 0.08 * u, min(tcp[2] + 0.03, self.travel + 0.05)]

    def act(self) -> int:
        from intent_policy.experts.separation import separation_hold
        from intent_policy.sim.restricted_action import RestrictedAction as A
        self._observe()
        busy = self.stage in ('onto_cloth_lower', 'settle_before_release', 'release', 'put_back_lower')
        tray = self._decide()
        if tray is not None and tray != self.tray and not busy and not self.sc.coaster_down():
            self._commit_tray(tray)
        if separation_hold(self):
            a = self._toward(self._back_away())
            return int(A.HOLD) if a is None else int(a)
        return super().act()

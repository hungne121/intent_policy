"""I1 (three coasters in a row) demonstrator: two information rules on one workflow (PIF.md §8.12).

Workflow per round: the robot takes the next cup (the order C1, C2, C3 is the task protocol: target-agnostic
preparation, allowed for both rules), lifts it, approaches a stand-off point next to where the human puts the coaster,
and sets the cup on the coaster once it is down and the hand has left it.

  observed: the destination is known once the coaster itself (visible, out of its source) has stopped moving: its
            horizontal speed below still_fraction of its peak for hold_min_s (configs/pif/pif.yaml v6_expert, the
            same stop rule as the v6 goal belief). No forecast.
  early:    while the human still carries the coaster, the robot follows the stand-off point of the predicted spot:
            the frozen ManiCast wrist forecast at +1 s plus the currently observed coaster-wrist offset (forecast from
            the 0.4 s history up to t only; intent/goal_belief.py, including the controlled forecast fault of C3);
            once the coaster is down it moves in when the forecast shows the withdrawing hand clear of the spot by
            the time the robot gets there (observed: when the hand is clear).

Stand-off: GUARD_APPROACH_XY (0.20 m) from the spot, away from where the human stands, at travel height; the guarded moves of
ScriptedExpert hold whenever the hand occupies the destination or is next to the arm. Nothing reads the human's
chosen spot: only the coaster pose, the mannequin keypoints and the forecast (all observable).
"""
from __future__ import annotations
import numpy as np
from intent_policy.experts.scripted_expert import ScriptedExpert, Step, GUARD_APPROACH_XY, GUARD_OCCUPIED_XY
from intent_policy.benchmark.events import EventType as E


class I1Expert(ScriptedExpert):
    RULES = ('observed', 'early')

    def reset(self) -> None:
        super().reset()
        sc, e = self.sc, self.sc.cfg.expert
        self.rule = e.get('rule', 'observed')
        if self.rule not in self.RULES:
            raise ValueError(f'expert.rule must be one of {self.RULES}')
        self.anticipate = True                   # guarded moves are active
        self.belief = None
        self.standoff = float(e.get('standoff_m', GUARD_APPROACH_XY))
        cont = getattr(self.mapper, 'continuous', False)
        self._speed = float(self.mapper.cfg.continuous['tcp_speed']) if cont else float(np.max(self.mapper.step)) / sc.dt
        if not hasattr(self, '_gb'):             # one forecaster history for the whole episode (continuous observation)
            self._gb = None
            if self.rule == 'early':
                from intent_policy.intent.goal_belief import GoalBelief
                cont = getattr(self.mapper, 'continuous', False)
                speed = float(self.mapper.cfg.continuous['tcp_speed']) if cont else float(np.max(self.mapper.step)) / sc.dt
                self._gb = GoalBelief(sc, 'early', speed, self.travel - self.z0)
                self._gb.reset()
            from pif.config import load_config
            from intent_policy.sim.base_env import ROOT
            ve = load_config(str(ROOT / 'configs/pif/pif.yaml'), [])['v6_expert']
            self._moving_m, self._still, self._hold_s = float(ve['moving_m']), float(ve['still_fraction']), float(ve['hold_min_s'])
        self.coaster = sc.coaster
        self.source = np.asarray(sc.start_pos[self.coaster][:2], float).copy()
        self._prev = None
        self._peak = 0.0
        self._calm = 0.0
        self.spot = None                         # observed destination (xy)
        self._make_way = None                    # a point away from a waiting hand (_approach_point)
        self.predicted = None                    # early: predicted destination (xy)
        self.spot_t = None
        self.obj, self.zone = sc.robot_target(), sc.spot
        self._commit(self.obj, self._grasp_steps(self.obj) + self._placing_plan())

    def _placing_plan(self, back_off: bool = False):
        sc = self.sc
        last = sc.sequence_index == len(sc.rounds) - 1
        tail = [Step('goto', lambda: np.r_[self.mapper.setpoint[:2], self.travel], name='clear_up')] + \
            (self._home_steps() if last else [Step('hold', ticks=10 ** 6, name='round_done')])
        up = [Step('goto', lambda: np.r_[self.mapper.setpoint[:2], self.travel], name='back_off')] if back_off else []
        return up + [
            Step('track', self._approach_point, name='approach_spot', guard=True, skip=self._placed_and_clear),
            Step('wait', until=self._placed_and_clear, name='await_coaster')] + self._place_steps(self.obj) + tail

    def _back_off_due(self) -> bool:
        """Setting the cup down (not yet released) while the coaster is taken again or the hand comes back to the spot:
        back off to the stand-off point and start over (C4 after; the same rule for both information rules)."""
        sc = self.sc
        if self.stage not in ('transport', 'lower', 'settle_before_release') or not sc.grasp_state.get(self.obj):
            return False
        if ('coaster_placed_' + self.coaster) not in sc.protocol_done:
            return True
        return float(np.linalg.norm(sc.human.pos[:2] - sc.zone_pos(self.zone)[:2])) < GUARD_OCCUPIED_XY

    # -- observation of the destination
    def _observe(self) -> None:
        sc = self.sc
        p = sc.pos(self.coaster)[:2]
        if self._gb is not None:
            self._gb.step()
        self._apply_forecast_fault()
        placed = ('coaster_placed_' + self.coaster) in sc.protocol_done
        if placed:
            self.spot = p.copy()
            return
        if self.spot is not None and np.linalg.norm(p - self.spot) > self._moving_m:
            self.spot, self._peak, self._calm = None, 0.0, 0.0      # the coaster moves on (taken again): observe anew
        if np.linalg.norm(p - self.source) < self._moving_m:
            self._prev = p.copy()
            return
        v = float(np.linalg.norm(p - self._prev)) / sc.dt if self._prev is not None else 0.0
        self._prev = p.copy()
        self._peak = max(self._peak, v)
        self._calm = self._calm + sc.dt if v < self._still * self._peak else 0.0
        if self._calm >= self._hold_s - 1e-9:
            if self.spot is None:
                self.spot_t = sc.time
            self.spot = p.copy()
        if self._gb is not None and self._gb.forecast is not None and self._gb.kp:
            wrist = np.asarray(self._gb.kp[-1][5], float)
            self.predicted = np.asarray(self._gb.forecast[-1][:2], float) + (p - wrist[:2])

    def _apply_forecast_fault(self) -> None:
        """C3 (controlled): from the start of the human's carry in the chosen round until the carry ends + extra_s, the
        forecast arm is turned about the shoulder (intent/forecast_fault.py). The realised window is stored on the
        scenario (episode metadata), so the offline tokens of the dataset get exactly the same fault."""
        sc, gb = self.sc, self._gb
        if gb is None or getattr(sc, 'realized_forecast_fault', None) is not None:
            return
        spec = sc.rounds[sc.human.human_round].get('forecast_fault') if sc.human.human_round < len(sc.rounds) else None
        seg = sc.human.segment
        if spec is None or seg is None or seg.label != 'carry_coaster' or sc.human.holding != self._round_coaster():
            return
        from intent_policy.intent.forecast_fault import ForecastFault
        fault = dict(start_s=round(float(sc.time), 4), end_s=round(float(seg.end_t + spec['extra_s']), 4),
                     yaw_deg=float(spec['yaw_deg']), arm='right', ramp_s=float(spec['ramp_s']))
        gb.forecast_fault = ForecastFault(**fault)
        sc.realized_forecast_fault = fault

    def _round_coaster(self):
        sc = self.sc
        return sc.rounds[sc.human.human_round]['coaster']

    def _destination(self):
        if self.spot is not None:
            return self.spot
        if self.rule == 'early' and self.predicted is not None:
            return self.predicted
        return None

    def _approach_point(self):
        """The stand-off point of the destination; but while a still hand waits within 0.12 m of the robot (hand mesh vs
        robot geoms: its way is blocked by the robot), make way: 0.10 m further from the hand, held until the hand is
        0.25 m away."""
        import mujoco
        sc = self.sc
        m, d = sc.env.model, sc.env.data
        hand, tcp = sc.human.pos, sc.tcp()
        near = min(mujoco.mj_geomDistance(m, d, r, sc.body_model.hand_geom, 0.3, None) for r in sc.robot_geoms)
        if self._make_way is None and near < 0.12 and float(np.linalg.norm(sc.human.vel)) < 0.05:
            u = tcp[:2] - hand[:2]
            u = u / max(float(np.linalg.norm(u)), 1e-9)
            self._make_way = np.r_[tcp[:2] + 0.10 * u, self.travel]
        if self._make_way is not None:
            if near > 0.25:
                self._make_way = None
            else:
                return self._make_way
        return self._standoff_point()

    def _standoff_point(self):
        d = self._destination()
        if d is None:                            # nothing known yet: keep the lifted cup where it is
            return np.r_[self.mapper.setpoint[:2], self.travel]
        # away from where the human stands (towards the base folded the arm onto the held cup, pilot 2026-10-30)
        u = d - np.asarray(self.sc.scene['human']['stand_xy'], float)
        u = u / max(float(np.linalg.norm(u)), 1e-9)
        return np.r_[d + self.standoff * u + self.tcp_obj_offset[:2], self.travel]

    def _placed_and_clear(self) -> bool:
        """The coaster is down and the hand has left the spot (GUARD_OCCUPIED_XY). Early rule: or the forecast shows
        the withdrawing hand clear of the spot by the time the robot would get there (travel at TCP speed, forecast
        horizon 1 s); the guarded moves still hold if the observed hand is near."""
        sc = self.sc
        if ('coaster_placed_' + self.coaster) not in sc.protocol_done:
            return False
        spot = sc.zone_pos(self.zone)[:2]
        if float(np.linalg.norm(sc.human.pos[:2] - spot)) >= GUARD_OCCUPIED_XY:
            return True
        gb = self._gb
        if self.rule != 'early' or gb is None or gb.forecast is None or not gb.kp:
            return False
        above = np.r_[spot + self.tcp_obj_offset[:2], self.travel]
        t_travel = float(np.linalg.norm(above - sc.tcp())) / self._speed
        k = int(np.clip(np.ceil(t_travel / 0.1), 1, len(gb.forecast))) - 1
        palm_off = sc.human.pos[:2] - np.asarray(gb.kp[-1][5], float)[:2]
        moving_away = float(np.linalg.norm(gb.forecast[k][:2] - spot)) > float(np.linalg.norm(gb.kp[-1][5][:2] - spot))
        return moving_away and float(np.linalg.norm(gb.forecast[k][:2] + palm_off - spot)) >= GUARD_OCCUPIED_XY

    def _yield(self) -> bool:
        return False                             # separation is handled in act(): back away instead of freezing

    def _back_away(self):
        """Too close to the human (experts/separation.py): never towards them; move away from the hand (8 cm, 3 cm up,
        at most 5 cm above travel height) so a hand that waits for the robot can go on. Freezing in place left both
        waiting until the human went on into the gripper (pilot 2026-10-30)."""
        sc = self.sc
        tcp, hand = sc.tcp(), sc.human.pos
        u = tcp[:2] - hand[:2]
        u = u / max(float(np.linalg.norm(u)), 1e-9)
        return np.r_[tcp[:2] + 0.08 * u, min(tcp[2] + 0.03, self.travel + 0.05)]

    def _follow_intention_change(self) -> None:
        """No reading of the human's change-of-mind event (privileged): a change shows only through the coaster moving
        on / being taken again (_observe, _back_off_due)."""

    def act(self) -> int:
        from intent_policy.experts.separation import separation_hold
        from intent_policy.sim.restricted_action import RestrictedAction as A
        self._observe()
        if self._back_off_due():
            self._set_plan(self._placing_plan(back_off=True))
        if separation_hold(self):
            a = self._toward(self._back_away())
            return int(A.HOLD) if a is None else int(a)
        return super().act()

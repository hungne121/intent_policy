"""v7 I2 demonstrator (scenarios/i2_side_request.py): an item from the side whose arm reaches out, into the tray.

A request of side s = the wrist of s reaching forward >= ext_m from where it hangs at rest (along the human's
facing direction); a hand moving to the chest (adjusting clothes) does not reach that far.
  observed: the wrists as seen now;
  early:    also the ManiCast forecast of both wrists at +1 s (the controlled fault of C3 turns the forecast arm).
After serving a request the robot waits for that arm to come back (below ext_m / 2) before taking a new one.
Nothing reads the human's requested side: the wrists and the forecast only. Separation monitoring as W1.
"""
from __future__ import annotations
import numpy as np
from intent_policy.experts.w1_expert import W1Expert
from intent_policy.experts.scripted_expert import Step


class I2Expert(W1Expert):
    def reset(self) -> None:
        super().reset()
        sc, e = self.sc, self.sc.cfg.expert
        self.ext_m = float(e['extend_m'])
        yaw = np.radians(float(sc.scene['human'].get('facing_deg', 180.0)))
        self.fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        self.rest_w = None
        self.side = None
        self.armed = True
        self.served_side = None
        if self._gb is None:                     # observed: the same keypoint history, no forecast
            from intent_policy.intent.goal_belief import GoalBelief
            self._kp_only = GoalBelief(sc, 'observed', self._speed, self.travel - self.z0)
            self._kp_only.reset()
        else:
            self._kp_only = None

    def _belief(self):
        return self._gb if self._gb is not None else self._kp_only

    def _ext(self):
        """Forward reach of each wrist (now, and at +1 s for the early rule) from its rest position."""
        gb = self._belief()
        if not gb.kp:
            return None
        w = {'l': np.asarray(gb.kp[-1][4], float), 'r': np.asarray(gb.kp[-1][5], float)}
        if self.rest_w is None:
            self.rest_w = {s: v.copy() for s, v in w.items()}
        now = {s: float((w[s] - self.rest_w[s]) @ self.fwd) for s in w}
        ahead = dict(now)
        if self.rule == 'early' and gb.forecast is not None and gb.forecast_left is not None:
            ahead = {'l': max(now['l'], float((np.asarray(gb.forecast_left[-1]) - self.rest_w['l']) @ self.fwd)),
                     'r': max(now['r'], float((np.asarray(gb.forecast[-1]) - self.rest_w['r']) @ self.fwd))}
        return now, ahead

    def _remaining_side(self, side):
        sc = self.sc
        return [k for k in sc.variation['i2']['items'][side] if not sc.in_box(k, sc.robot_box) and k not in sc.served]

    def _served_done(self):
        self.served_side, self.side = self.side, None
        self.item = self.sc.robot_item = None
        self._set_plan([Step('goto', lambda: np.r_[self.mapper.setpoint[:2], self.travel], name='clear_up'),
                        Step('goto', lambda: self.sc.rest_tcp, name='to_rest'),
                        Step('hold', ticks=10 ** 6, name='await_request')])     # waits for a request at rest

    def _item_plan(self, key):
        plan = super()._item_plan(key)
        return [s for s in plan if s.name != 'next'] + [Step('call', target=self._served_done, name='served')]

    def act(self) -> int:
        from intent_policy.experts.separation import separation_hold
        from intent_policy.sim.restricted_action import RestrictedAction as A
        sc = self.sc
        self._belief().step()
        ext = self._ext()
        if ext is not None:
            now, ahead = ext
            if not self.armed and self.served_side is not None and now[self.served_side] < self.ext_m / 2:
                self.armed = True
            if self.armed and self.side is None:
                cand = [s for s in ('l', 'r') if ahead[s] >= self.ext_m and self._remaining_side(s)]
                if cand:
                    s = max(cand, key=lambda q: ahead[q])
                    tcp = sc.tcp()
                    key = min(self._remaining_side(s), key=lambda k: float(np.linalg.norm(sc.pos(k)[:2] - tcp[:2])))
                    self.side, self.armed, self.item, sc.robot_item = s, False, key, key
                    self._commit(key, self._item_plan(key))
        if separation_hold(self):
            a = self._toward(self._back_away())
            return int(A.HOLD) if a is None else int(a)
        return super(W1Expert, self).act()

    def _remaining(self):
        sc = self.sc
        return [k for k in sc.robot_items if k not in sc.served]

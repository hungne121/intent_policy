"""v10 I3 demonstrator (scenarios/i3_direct_sort.py): lift the nearest item of the pile and hold it up above the pile
(the common preparation of both sides), then put it into the basket of the side the right arm is swept to.

A direction = the right wrist displaced sideways (world x: the human faces +y, its left is -x) >= lateral_m from where
it rested when the item was held up (the start of the instruction, as the motion start S of inverse planning, Dragan
et al. 2013; the hand does not come back to one fixed rest: it stopped ~0.095 m off after a sweep, trace 02/11); a hand
moving to the chest (adjusting clothes) does not move that far sideways.
  observed: the wrist as seen now;
  early:    also the ManiCast forecast path of that wrist at D's waypoints (expert.decision.path_features: endpoint =
            the +1 s point only, path = the furthest sideways point of the path; docs/PIF.md §8.12 "Khai thác hết
            ManiCast").
Recovery: a direction to the other side seen / forecast before the release turns the robot to the other basket (cheap);
seen / forecast after the release, before the next item is held up, the robot takes the item out again and puts it
into the other basket (the after-commit cost, as C4's wrong bin put back). One direction per item: the reference is taken anew for the next item, so the arm coming back is never read as one.
Nothing reads the human's side: the wrist and the forecast only. Separation monitoring as W1.
"""
from __future__ import annotations
import numpy as np
from intent_policy.experts.w1_expert import W1Expert
from intent_policy.experts.scripted_expert import Step

HOLDING = ('lift', 'hold_item', 'choose')
CARRYING = ('to_box', 'lower_box')


class I3Expert(W1Expert):
    def reset(self) -> None:
        super().reset()
        sc, e = self.sc, self.sc.cfg.expert
        self.lat_m = float(e['lateral_m'])
        self.hold_dz = float(e['hold_dz_m'])
        d = e.get('decision') or {}
        self.path_features = str(d.get('path_features', 'endpoint'))
        if self._gb is not None and d.get('forecast_waypoints'):
            m = int(d['forecast_waypoints'])
            self._gb.t_out = (np.arange(m) + 1) / m * float(d.get('forecast_horizon_s', 1.0))
        if self._gb is None:                     # observed: the same keypoint history, no forecast
            from intent_policy.intent.goal_belief import GoalBelief
            self._kp_only = GoalBelief(sc, 'observed', self._speed, self.travel - self.z0)
            self._kp_only.reset()
        else:
            self._kp_only = None
        self.pile = np.asarray(sc.cfg.scene_variation['i3']['pile_xy'], float)
        self.ref_x = None
        self.side = None
        self.prev = None                         # the item just released: (item, side, reference), until the next is up
        self._next()

    def _belief(self):
        return self._gb if self._gb is not None else self._kp_only

    def _lateral(self):
        """Sideways displacement of the right wrist from rest: (now, ahead); ahead = now for observed."""
        gb = self._belief()
        if not gb.kp:
            return None
        w = np.asarray(gb.kp[-1][5], float)
        sc = self.sc
        if (self.ref_x is None and self.side is None and self.stage in HOLDING and sc.grasp_state.get(self.item)
                and not gb.moving):
            self.ref_x = float(w[0])                 # the hand at rest while the item is held up: the reference
        if self.ref_x is None:
            return None
        return self._lat_from(self.ref_x)

    def _lat_from(self, ref):
        gb = self._belief()
        w = np.asarray(gb.kp[-1][5], float)
        now = ahead = float(w[0]) - ref
        if self.rule == 'early' and gb.forecast is not None:
            X = np.asarray(gb.forecast, float)[:, 0] - ref
            f = float(X[np.argmax(np.abs(X))]) if self.path_features == 'path' else float(X[-1])
            ahead = f if abs(f) > abs(now) else now
        return now, ahead

    def _remaining(self):
        """Items not yet in a basket. Over a basket's footprint counts as in it: an item just released is still
        falling / settling there (taken again as the next item otherwise, smoke 02/11)."""
        sc = self.sc
        over = lambda k: any(np.all(np.abs(sc.pos(k)[:2] - np.asarray(b['xy'], float)) <= np.asarray(b['half'], float))
                             for b in sc.baskets.values())
        return [k for k in sc.items if 'sorted_' + k not in sc.protocol_done and not over(k)]

    def _pick_plan(self, key):
        sc = self.sc
        grasp = [g for g in self._grasp_steps(key) if g.name not in ('open', 'confirm_target')]
        return [Step('grip', 'open', until=lambda: sc.env.data.qpos[sc.env.gid] > 0.075, ticks=25, name='open')] + grasp + [
            Step('goto', lambda: np.r_[self.pile + self.tcp_obj_offset[:2], self.travel + self.hold_dz], name='hold_item'),
            Step('hold', ticks=10 ** 6, name='choose')]

    def _box_plan(self, side):
        sc = self.sc
        key = self.item
        b = sc.baskets[side]
        n = sum(1 for k in sc.items if k != key and sc.basket_of(k) == side)   # items in that basket (seen): the next slot
        bx = np.asarray(b['xy'], float) + np.asarray(sc.variation['w1']['drop_slots'][n], float)
        drop_z = lambda: sc.table_z + float(b['height']) + 0.02 + sc.rest_height[key] - sc.table_z
        return [Step('goto', lambda: np.r_[bx + self.tcp_obj_offset[:2], self.travel], name='to_box', guard=True),
                Step('goto', lambda: np.r_[bx + self.tcp_obj_offset[:2], drop_z() + self.tcp_obj_offset[2]], name='lower_box'),
                Step('grip', 'open', until=lambda: not sc.grasp_state[key] and sc.env.data.qpos[sc.env.gid] > 0.06, ticks=25,
                     name='release'),
                Step('goto', lambda: np.r_[self.mapper.setpoint[:2], self.travel], name='retreat'),
                Step('call', target=self._next, name='next')]

    def _next(self):
        sc = self.sc
        self.prev = (self.item, self.side, self.ref_x) if self.side is not None else None
        self.side = self.ref_x = None
        left = self._remaining()
        if not left:
            self.item = sc.robot_item = None
            self._set_plan(self._home_steps())
            return
        tcp = sc.tcp()
        self.item = sc.robot_item = min(left, key=lambda k: float(np.linalg.norm(sc.pos(k)[:2] - tcp[:2])))
        self._set_plan(self._pick_plan(self.item))

    def act(self) -> int:
        from intent_policy.experts.separation import separation_hold
        from intent_policy.sim.restricted_action import RestrictedAction as A
        sc = self.sc
        self._belief().step()
        self._apply_forecast_fault()
        if self.prev is not None and self._belief().kp:
            item, side, ref = self.prev
            if self.stage in HOLDING and sc.grasp_state.get(self.item):
                self.prev = None                     # the next item is up: its own direction from now on
            else:
                _, ahead = self._lat_from(ref)
                s = 'l' if ahead <= -self.lat_m else ('r' if ahead >= self.lat_m else None)
                if s is not None and s != side and sc.basket_of(item) == side:
                    # recovery after the release: the item out of the basket again, into the other one
                    sc.log_recovery = getattr(sc, 'log_recovery', []) + [dict(t=sc.time, frm=side, to=s, released=True)]
                    self.prev = None
                    self.item, self.side, self.ref_x = item, s, ref
                    sc.robot_item = item
                    grasp = [g for g in self._grasp_steps(item) if g.name not in ('open', 'confirm_target')]
                    self._commit(item, [self._lift_step('lift_redirect'),
                                        Step('grip', 'open', until=lambda: sc.env.data.qpos[sc.env.gid] > 0.075, ticks=25,
                                             name='open')] + grasp + self._box_plan(s))
        lat = self._lateral()
        if lat is not None:
            now, ahead = lat
            s = 'l' if ahead <= -self.lat_m else ('r' if ahead >= self.lat_m else None)
            if self.side is None and s is not None and self.stage in HOLDING and sc.grasp_state.get(self.item):
                self.side = s
                self._commit(self.item, self._box_plan(s))
            elif self.side is not None and s is not None and s != self.side and self.stage in CARRYING:
                # recovery: the direction seen / forecast changed before the release: the other basket
                sc.log_recovery = getattr(sc, 'log_recovery', []) + [dict(t=sc.time, frm=self.side, to=s)]
                self.side = s
                self._commit(self.item, [self._lift_step('lift_redirect')] + self._box_plan(s))
        if separation_hold(self):
            a = self._toward(self._back_away())
            return int(A.HOLD) if a is None else int(a)
        return super(W1Expert, self).act()

    def _apply_forecast_fault(self) -> None:
        """forecast_fault case: the forecast arm turned about the shoulder during the sweep of the fault round
        (intent/forecast_fault.py, as C3); the realised window on the scenario (never an input of ACT)."""
        sc, gb = self.sc, self._gb
        spec = sc.variation['i3'].get('forecast_fault')
        if gb is None or spec is None or sc.realized_forecast_fault is not None or sc.human.k != spec['round']:
            return
        seg = sc.human.segment
        if seg is None or seg.label != 'direct_out':
            return
        from intent_policy.intent.forecast_fault import ForecastFault
        fault = dict(start_s=round(float(sc.time), 4), end_s=round(float(seg.end_t + spec['extra_s']), 4),
                     yaw_deg=float(spec['yaw_deg']), arm='right', ramp_s=float(spec['ramp_s']))
        gb.forecast_fault = ForecastFault(**fault)
        sc.realized_forecast_fault = fault

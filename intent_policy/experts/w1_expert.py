"""v7 W1 demonstrator (scenarios/w1_sorting.py): put the blue items into the robot's box, one at a time, never where
the human is about to work. An item is in conflict when the human's hand (or what it holds) is within conflict_m of it
(horizontally, the hand below conflict_z above the table):

  observed: the hand as seen now (palm, held item);
  early:    also the ManiCast wrist forecast over the next 1 s (0.1 ... 1.0 s; the controlled fault of C3 turns it).

The robot takes the nearest free item; it switches to another free item if its item becomes in conflict before the
grasp, and waits at `wait_xy` when none is free. Separation monitoring as experts/separation.py (back away).
Nothing reads the human's plan (item order / timing): the hand, the forecast and the items only.
"""
from __future__ import annotations
import numpy as np
from intent_policy.experts.scripted_expert import ScriptedExpert, Step


class W1Expert(ScriptedExpert):
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
        self.conflict_m, self.conflict_z = float(e['conflict_m']), float(e['conflict_z_m'])
        self.wait_xy = np.asarray(e['wait_xy'], float)
        self.item = None
        self._waiting = False
        self._set_plan([Step('grip', 'open', until=lambda: sc.env.data.qpos[sc.env.gid] > 0.075, ticks=25, name='open'),
                        Step('hold', ticks=10 ** 6, name='choose')])

    def _footprint(self, key, centre):
        """Points over a held box-shaped object (its whole extent, e.g. W2's tray), at `centre` with its current yaw."""
        obj = self.sc.scene['objects'][key]
        if obj['shape'] != 'box':
            return [centre]
        hx, hy = float(obj['size'][0]), float(obj['size'][1])
        q = self.sc.quat(key)
        yaw = 2 * np.arctan2(q[3], q[0])
        c, s = np.cos(yaw), np.sin(yaw)
        out = []
        for u in np.linspace(-hx, hx, 5):
            for v in np.linspace(-hy, hy, 3):
                out.append(np.asarray(centre, float) + [c * u - s * v, s * u + c * v, 0.0])
        return out

    def _human_points(self):
        sc = self.sc
        held = sc.human.holding
        pts = [sc.human.pos]
        if held:
            pts += self._footprint(held, sc.pos(held))
        if self._gb is not None and self._gb.forecast is not None:
            pts += [np.asarray(q, float) for q in self._gb.forecast]
            if held and self._gb.kp:            # the held object where the forecast hand will be (rigid hold)
                off = sc.pos(held) - np.asarray(self._gb.kp[-1][5], float)
                for q in self._gb.forecast:
                    pts += self._footprint(held, np.asarray(q, float) + off)
        if getattr(sc.human, 'bimanual', None):
            pts.append(sc.human.idle_palm())
        return np.asarray(pts, float)

    def _conflict(self, key) -> bool:
        sc = self.sc
        p = sc.pos(key)
        h = self._human_points()
        near = np.linalg.norm(h[:, :2] - p[:2], axis=1) < self.conflict_m
        low = h[:, 2] < sc.table_z + self.conflict_z
        return bool(np.any(near & low))

    def _remaining(self):
        sc = self.sc
        return [k for k in sc.robot_items if ('robot_boxed_' + k) not in sc.protocol_done and not sc.in_box(k, sc.robot_box)]

    def _choose(self):
        sc = self.sc
        free = [k for k in self._remaining() if not self._conflict(k)]
        if not free:
            return None
        tcp = sc.tcp()
        return min(free, key=lambda k: float(np.linalg.norm(sc.pos(k)[:2] - tcp[:2])))

    def _item_plan(self, key):
        sc = self.sc
        box = sc.robot_box
        drop_z = lambda: sc.table_z + float(box['height']) + 0.02 + sc.rest_height[key] - sc.table_z
        n = len(sc.robot_items) - len(self._remaining())        # items already in the box (seen): the next slot
        bx = np.asarray(box['xy'], float) + np.asarray(sc.variation['w1']['drop_slots'][n], float)
        grasp = [g for g in self._grasp_steps(key) if g.name != 'open']
        return [Step('grip', 'open', until=lambda: sc.env.data.qpos[sc.env.gid] > 0.075, ticks=25, name='open')] + grasp + [
            Step('goto', lambda: np.r_[bx + self.tcp_obj_offset[:2], self.travel], name='to_box', guard=True),
            Step('goto', lambda: np.r_[bx + self.tcp_obj_offset[:2], drop_z() + self.tcp_obj_offset[2]], name='lower_box'),
            Step('grip', 'open', until=lambda: not sc.grasp_state[key] and sc.env.data.qpos[sc.env.gid] > 0.06, ticks=25,
                 name='release'),
            Step('goto', lambda: np.r_[self.mapper.setpoint[:2], self.travel], name='retreat'),
            Step('call', target=self._next, name='next')]

    def _wait_plan(self):
        return [Step('goto', lambda: np.r_[self.wait_xy, self.travel], name='wait_aside', guard=True),
                Step('hold', ticks=10 ** 6, name='choose')]

    def _next(self):
        self.item = None
        self.sc.robot_item = None
        self._waiting = False
        if not self._remaining():
            self._set_plan(self._home_steps())
            return
        self._set_plan([Step('hold', ticks=10 ** 6, name='choose')])

    def _follow_intention_change(self) -> None:
        """No reading of the human's events (privileged)."""

    def _known_zone(self):
        return None

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
        sc = self.sc
        if self._gb is not None:
            self._gb.step()
        before_grasp = self.stage in ('choose', 'wait_aside', 'open', 'above_grasp', 'settle_above_grasp', 'confirm_target')
        if before_grasp:
            if self.item is not None and self._conflict(self.item):
                self.item = sc.robot_item = None     # the human is about to work there: drop it before the grasp
            if self.item is None:
                new = self._choose()
                if new is not None:
                    self.item, sc.robot_item, self._waiting = new, new, False
                    self._commit(new, self._item_plan(new))
                elif not self._waiting:
                    self._waiting = True
                    self._set_plan(self._wait_plan())
        if separation_hold(self):
            a = self._toward(self._back_away())
            return int(A.HOLD) if a is None else int(a)
        return super().act()

"""v10 C2 demonstrator (scenarios/c2_shelf_wipe.py): wipe the section whose item the human is taking, once the item
and the hand have left it; wait above the middle of the sections otherwise (the best common point, v9 G2).

The section = the pad of the item the hand goes for, chosen as in experts/c4_expert.py (belief over the items, Fang et
al. 2026; commit at tau, Lekeufack et al. 2024). Nothing reads the human's
order: the wrist, the forecast and the visible items only.
"""
from __future__ import annotations
import numpy as np
from intent_policy.experts.c4_expert import C4Expert
from intent_policy.experts.w1_expert import W1Expert
from intent_policy.experts.scripted_expert import Step

WIPING = ('approach_pad', 'over_pad', 'descend_wipe', 'wipe', 'lift_wipe')


class C2Expert(C4Expert):
    def reset(self) -> None:
        W1Expert.reset(self)
        sc, e = self.sc, self.sc.cfg.expert
        self._decision_params(e)
        self.wipe_dz, self.stroke_m, self.strokes = float(e['wipe_dz']), float(e['stroke_m']), int(e['strokes'])
        self.clear_m = float(sc.cfg.human_behavior['clear_m'])
        self.staging = np.asarray(list(sc.pads.values()), float).mean(0) + np.asarray(e['staging_offset'], float)
        self.bin = self.item = None              # self.bin = the pad being served (C4Expert's bookkeeping)
        if self._gb is None:
            from intent_policy.intent.goal_belief import GoalBelief
            self._kp_only = GoalBelief(sc, 'observed', self._speed, self.travel - self.z0)
            self._kp_only.reset()
        else:
            self._kp_only = None
        self._set_plan(self._staging_plan())

    def _group(self, item):
        return self.sc.pad_of[item]

    def _staging_plan(self):
        sc = self.sc
        return [Step('grip', 'close', until=lambda: sc.env.data.qpos[sc.env.gid] < 0.01, ticks=25, name='close_hand'),
                Step('goto', lambda: np.r_[self.staging, self.travel], name='staging', guard=True),
                Step('hold', ticks=10 ** 6, name='choose')]

    def _left(self):
        sc = self.sc
        return [k for k in sc.items if 'wiped_' + sc.pad_of[k] not in sc.protocol_done]

    def _cleared(self):
        """An item already taken off its section (seen: put elsewhere, W3) whose section is not wiped yet."""
        sc = self.sc
        for k in self._left():
            if sc.human.holding != k and float(np.linalg.norm(sc.pos(k)[:2] - sc.pads[sc.pad_of[k]])) > self.clear_m:
                return k
        return None

    def _wipe_plan(self, pad):
        sc = self.sc
        c = sc.pads[pad]
        z = lambda: sc.table_z + self.wipe_dz
        item = next(k for k in sc.items if sc.pad_of[k] == pad)
        # the section's item is off it (held up, or put elsewhere: W3) and the hand and the item are clear of it
        clear = lambda: (float(np.linalg.norm(sc.pos(item)[:2] - c)) > self.clear_m
                         and float(np.linalg.norm(sc.human.pos[:2] - c)) > self.clear_m
                         and (getattr(sc.human, 'bimanual', None) is None or float(np.linalg.norm(sc.human.idle_palm()[:2] - c)) > self.clear_m))
        r_max = float(sc.cfg.expert.get('reach_m', np.inf))
        def at(dx):                                # a stroke end, kept within the robot's reach (configs/layout.yaml)
            q = c + [dx, 0.0]
            n = float(np.linalg.norm(q))
            return q * (r_max / n) if n > r_max else q
        strokes = []
        for j in range(self.strokes):
            s = self.stroke_m * (1 if j % 2 == 0 else -1)
            strokes.append(Step('goto', lambda s=s: np.r_[at(s), z()], name='wipe'))
        # wait at a standoff on the robot's side of the section (not over it: the hands work there) until it is clear
        stand = c + np.asarray(sc.cfg.expert.get('standoff_offset', [0.0, 0.0]), float)
        return [Step('goto', lambda: np.r_[stand, self.travel], name='approach_pad', guard=True),
                Step('wait', until=clear, name='await_clear'),
                Step('goto', lambda: np.r_[c, self.travel], name='over_pad', guard=True),
                Step('goto', lambda: np.r_[at(-self.stroke_m), z()], name='descend_wipe')] + strokes + [
                Step('goto', lambda: np.r_[c, self.travel], name='lift_wipe'),
                Step('call', target=self._served, name='served')]

    def act(self) -> int:
        from intent_policy.experts.separation import separation_hold
        from intent_policy.sim.restricted_action import RestrictedAction as A
        sc = self.sc
        self._belief().step()
        held = sc.human.holding if sc.human.holding in sc.items else None
        if held is not None and held != self._last_held and held in self._left():
            self._round_outcome(held)
        self._last_held = held
        if self.bin is None and self.stage in ('close_hand', 'staging', 'choose'):
            k = self._target_item() or self._cleared()
            if k is not None:
                self.item, self.bin = k, sc.pad_of[k]
                self._round_groups.add(self.bin)
                sc.robot_item = k
                self._commit(k, self._wipe_plan(self.bin))
        elif (self.bin is not None and sc.human.holding in sc.items and sc.pad_of[sc.human.holding] != self.bin
              and self.stage in ('approach_pad', 'await_clear', 'over_pad')):
            # the item taken is another section's: go there instead (nothing wiped yet)
            self.item, self.bin = sc.human.holding, sc.pad_of[sc.human.holding]
            self._round_groups.add(self.bin)
            sc.robot_item = self.item
            self._set_plan(self._wipe_plan(self.bin))
        if sc.cfg.expert.get('yield_while_wiping', False) and sc.human_distance < float(sc.cfg.expert['yield_distance_m']):
            # Cowork (W3): any part of the human arm closing in on the robot (its whole body, not only the hand): rise
            # straight up out of the shared space, then carry on with the plan
            a = self._toward(np.r_[sc.tcp()[:2], self.travel + float(sc.cfg.expert['yield_rise_m'])])
            return int(A.HOLD) if a is None else int(a)
        exempt = () if sc.cfg.expert.get('yield_while_wiping', False) else WIPING + ('await_clear',)
        if self.stage not in exempt and separation_hold(self):
            a = self._toward(self._back_away())
            return int(A.HOLD) if a is None else int(a)
        return super(W1Expert, self).act()

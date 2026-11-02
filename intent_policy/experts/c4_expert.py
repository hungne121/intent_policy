"""v10 C4 demonstrator (scenarios/c4_waste_sort.py): fetch the bin of the item the human is taking, hold it out at the
presentation point until the item is in and the hand is away, put it back.

Which item the hand goes for: a belief over the pile's items (Fang et al. 2026: distance and heading of the wrist,
seen for observed, the ManiCast forecast at +1 s for early); the robot commits to the bin of the most likely kind once
that kind's belief reaches tau, calibrated per rule by Conformal Decision Theory (Lekeufack et al. 2024) to a target
rate of wrong commitments (docs/PIF.md §8.12 "Công thức quyết định và phục hồi").
While waiting the robot stays above the middle of the two bins (the best common point, v9 G2). A bin that turns out
wrong (the held item is of the other kind) is put back before the right one is fetched. Nothing reads the human's
order: the wrist, the forecast and the visible items only.
"""
from __future__ import annotations
import numpy as np
from intent_policy.experts.w1_expert import W1Expert
from intent_policy.experts.scripted_expert import Step

TAU = {}                    # tau while calibrating, kept across the episodes of one process (expert.decision.calibrate)
PRESENT = ('to_present', 'lower_present', 'hold_present', 'lift_present')
BACK = ('to_home', 'lower_home', 'settle_before_release', 'release', 'retreat', 'served')


class C4Expert(W1Expert):
    def reset(self) -> None:
        super().reset()
        sc, e = self.sc, self.sc.cfg.expert
        self._decision_params(e)
        self.present_xy = np.asarray(sc.cfg.human_behavior['present_xy'], float)
        self.present_dz = float(sc.cfg.human_behavior['present_dz_m'])
        homes = np.asarray([v[:2] for v in sc.bin_home.values()], float)
        self.staging = homes.mean(0)
        self.bin = None
        self.item = None
        if self._gb is None:                     # observed: the same keypoint history, no forecast
            from intent_policy.intent.goal_belief import GoalBelief
            self._kp_only = GoalBelief(sc, 'observed', self._speed, self.travel - self.z0)
            self._kp_only.reset()
        else:
            self._kp_only = None
        self._set_plan(self._staging_plan())

    def _decision_params(self, e):
        d = e['decision']
        self.alpha, self.beta = 1.0 / float(d['sigma_m']), float(d['beta'])
        self.still_speed, self.vel_window_s = float(d['still_speed']), float(d['velocity_window_s'])
        self.eps, self.eta = float(d['eps']), float(d['eta'])
        self.calibrate = bool(d.get('calibrate', False))
        tau = d['tau'][self.rule] if isinstance(d['tau'], dict) else d['tau']
        self.tau = TAU.get((self.sc.cfg.id, self.rule), float(tau)) if self.calibrate else float(tau)
        # which part of the forecast path the belief reads (docs/PIF.md §8.12 "Khai thác hết ManiCast", ablation):
        # endpoint (the +1 s point) | closest (closest approach along the path) | stop (the predicted stopping point,
        # else closest approach); the path at D's waypoints so that teacher and student see the same forecast
        self.path_features = str(d.get('path_features', 'endpoint'))
        if self._gb is not None and d.get('forecast_waypoints'):
            m = int(d['forecast_waypoints'])
            self._gb.t_out = (np.arange(m) + 1) / m * float(d.get('forecast_horizon_s', 1.0))
        self._item_held, self._redir, self._wrist, self._rest_w = False, 0, [], None
        self._round_groups, self.losses, self._last_held = set(), [], None

    def _belief(self):
        return self._gb if self._gb is not None else self._kp_only

    def _staging_plan(self):
        return [Step('grip', 'open', until=lambda: self.sc.env.data.qpos[self.sc.env.gid] > 0.075, ticks=25, name='open'),
                Step('goto', lambda: np.r_[self.staging, self.travel], name='staging', guard=True),
                Step('hold', ticks=10 ** 6, name='choose')]

    def _left(self):
        sc = self.sc
        return [k for k in sc.items if 'binned_' + k not in sc.protocol_done and not any(
            sc.in_bin(k, b) for b in sc.bins.values())]

    def _group(self, item):
        """What the robot serves for an item: its kind's bin (C2 / W3: its section)."""
        return self.sc.bins[self.sc.kind[item]]

    def _beliefs(self):
        """Belief over the remaining items the right hand may be going for (docs/PIF.md §8.12 "Công thức quyết định"):
        Fang et al. 2026 Eq. 4, p(i) ∝ exp(-alpha d_i + beta c_i), d_i the distance from the wrist (observed: as seen;
        early: the ManiCast forecast at +1 s) to item i, c_i the cosine between the hand's direction of motion (seen
        velocity / now -> forecast) and the direction to item i (0 for a still hand). None while the (forecast) hand is
        no nearer to any item than to its rest pose: nothing is being reached for yet."""
        sc, gb = self.sc, self._belief()
        left = self._left()
        if not gb.kp or not left:
            return None
        w = np.asarray(gb.kp[-1][5], float)
        if self._rest_w is None:
            self._rest_w = w.copy()
        self._wrist.append((sc.time, w))
        self._wrist = [x for x in self._wrist if sc.time - x[0] <= self.vel_window_s + 1e-9]
        t0, w0 = self._wrist[0]
        q = np.asarray([sc.decision_point(k) if hasattr(sc, 'decision_point') else sc.pos(k) for k in left], float)
        if self.rule == 'early' and gb.forecast is not None:
            X = np.asarray(gb.forecast, float)
            x = X[-1]
            v = (x - w) / float(gb.t_out[-1])
            D = np.linalg.norm(X[:, None, :2] - q[None, :, :2], axis=2)      # [waypoint, item]
            k = int(np.argmin(D.min(1)))                                    # closest approach to any item
            if self.path_features == 'stop':
                sp = np.linalg.norm(np.diff(np.vstack([w, X]), axis=0), axis=1) / np.diff(np.r_[0.0, gb.t_out])
                slow = np.nonzero(sp < float(gb.ve['still_fraction']) * max(gb.peak, 1e-9))[0] if gb.peak > 0 else []
                k = int(slow[0]) if len(slow) else k                         # the hand is forecast to stop here
            if self.path_features in ('closest', 'stop'):
                x = X[k]
        else:
            x = w
            v = (w - w0) / (sc.time - t0) if sc.time > t0 else np.zeros(3)
        d = np.linalg.norm(q[:, :2] - x[:2], axis=1)
        if d.min() >= float(np.linalg.norm(x[:2] - self._rest_w[:2])):
            return None
        c = np.zeros(len(left))
        if float(np.linalg.norm(v[:2])) > self.still_speed:
            u = q[:, :2] - w[:2]
            c = (u @ v[:2]) / (np.linalg.norm(u, axis=1) * np.linalg.norm(v[:2]) + 1e-12)
        e = -self.alpha * d + self.beta * c
        p = np.exp(e - e.max())
        p /= p.sum()
        return dict(zip(left, p))

    def _target_item(self):
        """The item served next: the held one (seen) if it still needs serving, else the most likely item of the most
        likely group once that group's belief reaches tau (Fang et al. 2026 §II-B3)."""
        sc = self.sc
        if sc.human.holding in sc.items:          # (an item already served: nothing to do until it is let go)
            return sc.human.holding if sc.human.holding in self._left() else None
        b = self._beliefs()
        if b is None:
            return None
        groups = {}
        for k, pk in b.items():
            groups[self._group(k)] = groups.get(self._group(k), 0.0) + pk
        g = max(groups, key=groups.get)
        if groups[g] < self.tau:
            return None
        return max((k for k in b if self._group(k) == g), key=b.get)

    def _round_outcome(self, held):
        """Conformal Decision Theory (Lekeufack et al. 2024, Eq. 5) on the wrong-commitment loss of a round: tau_{k+1} =
        tau_k + eta (l_k - eps), l_k = 1 if a group other than the taken item's was committed to in the round. Only
        while calibrating (expert.calibrate_tau); tau is frozen otherwise."""
        loss = float(any(g != self._group(held) for g in self._round_groups))
        self._round_groups = set()
        self.losses.append(loss)
        if self.calibrate:
            self.tau = float(np.clip(self.tau + self.eta * (loss - self.eps), 0.5, 0.999))
            TAU[(self.sc.cfg.id, self.rule)] = self.tau

    def _go(self, item):
        sc = self.sc
        self.item, self.bin, self._item_held, self._redir = item, self._group(item), False, 0
        self._round_groups.add(self.bin)
        sc.robot_item = self.bin
        self._commit(self.bin, self._bin_plan(self.bin))

    def _bin_plan(self, b):
        sc = self.sc
        h = sc.rest_height[b] - sc.table_z
        above_p = lambda: np.r_[self.present_xy + self.tcp_obj_offset[:2], self.travel]
        at_p = lambda: np.r_[self.present_xy, sc.table_z + self.present_dz + h] + self.tcp_obj_offset
        home = sc.bin_home[b]
        above_h = lambda: np.r_[home[:2] + self.tcp_obj_offset[:2], self.travel]
        at_h = lambda: np.r_[home[:2], sc.rest_height[b] + 0.003] + self.tcp_obj_offset
        done = lambda: (self.item is not None and sc.in_bin(self.item, b)
                        and float(np.linalg.norm(sc.human.pos[:2] - sc.pos(b)[:2])) > 0.10)
        return self._grasp_steps(b) + [
            Step('goto', above_p, name='to_present', guard=True),
            Step('goto', at_p, name='lower_present'),
            Step('wait', until=done, name='hold_present'),
            Step('goto', above_p, name='lift_present'),
        ] + self._put_back(b)

    def _put_back(self, b):
        sc = self.sc
        home = sc.bin_home[b]
        return [Step('goto', lambda: np.r_[home[:2] + self.tcp_obj_offset[:2], self.travel], name='to_home', guard=True),
                Step('goto', lambda: np.r_[home[:2], sc.rest_height[b] + 0.003] + self.tcp_obj_offset, name='lower_home'),
                Step('settle', ticks=15, name='settle_before_release'),
                Step('grip', 'open', until=lambda: not sc.grasp_state[b] and sc.env.data.qpos[sc.env.gid] > 0.06, ticks=25,
                     name='release'),
                Step('goto', lambda: np.r_[self.mapper.setpoint[:2], self.travel], name='retreat'),
                Step('call', target=self._served, name='served')]

    def _served(self):
        self.bin = self.item = self.sc.robot_item = None
        if not self._left():
            self._set_plan(self._home_steps())
            return
        self._set_plan(self._staging_plan())

    def act(self) -> int:
        from intent_policy.experts.separation import separation_hold
        from intent_policy.sim.restricted_action import RestrictedAction as A
        sc = self.sc
        self._belief().step()
        held = sc.human.holding if sc.human.holding in sc.items else None
        if held is not None and held != self._last_held and held in self._left():
            self._round_outcome(held)
        self._last_held = held
        if self.bin is None and self.stage in ('open', 'staging', 'choose'):
            k = self._target_item()
            if k is not None:
                self._go(k)
        elif self.bin is not None and self.stage not in BACK:
            grasped = bool(sc.grasp_state.get(self.bin))
            if held is not None and sc.bins[sc.kind[held]] == self.bin:
                self.item, self._item_held = held, True     # the item actually taken (of this bin's kind)
            want = held if held is not None else (self._target_item() if not grasped else None)
            other = want is not None and sc.bins[sc.kind[want]] != self.bin
            self._redir = self._redir + 1 if other else 0
            if other:
                # recovery: the intention seen / forecast changed to the other kind. Before the grasp: turn to the other
                # bin (cheap); with the bin in the gripper (only on what is seen held): put it back, then the other
                wrong = self.bin
                if not grasped:
                    self._go(want)
                else:
                    self.bin, self.item = None, want
                    self._set_plan([Step('goto', lambda: np.r_[self.mapper.setpoint[:2], self.travel], name='lift_present')]
                                   + self._put_back(wrong))
                sc.log_recovery = getattr(sc, 'log_recovery', []) + [dict(t=sc.time, frm=wrong, grasped=grasped)]
            elif (self._item_held and held is None and not sc.in_bin(self.item, self.bin)
                  and sc.pos(self.item)[2] < sc.table_z + 0.03 and sc.speed(self.item) < 0.05):
                # the taken item was put back on the pile (a change of mind after taking it): this bin is not needed now
                wrong = self.bin
                self.bin = None
                if grasped:
                    self._set_plan([Step('goto', lambda: np.r_[self.mapper.setpoint[:2], self.travel], name='lift_present')]
                                   + self._put_back(wrong))
                else:
                    self._set_plan([Step('call', target=self._served, name='served')])
                sc.log_recovery = getattr(sc, 'log_recovery', []) + [dict(t=sc.time, frm=wrong, grasped=grasped, cancel=True)]
        if self.stage not in PRESENT and separation_hold(self):
            a = self._toward(self._back_away())
            return int(A.HOLD) if a is None else int(a)
        return super(W1Expert, self).act()

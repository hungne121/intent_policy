"""Local I1 preview rules requested by the user; no changes to the stable expert."""
import numpy as np
from intent_policy.experts.i1_expert import I1Expert
from intent_policy.experts.scripted_expert import ScriptedExpert, Step, GUARD_OCCUPIED_XY
from intent_policy.experts.separation import separation_hold
from intent_policy.sim.restricted_action import RestrictedAction as A


class SmoothI1Expert(I1Expert):
    def reset(self):
        self.filtered_goal = None
        self.clear_since = None
        self.preview_safety_holds = 0
        super().reset()

    def _placing_plan(self, back_off=False):
        last = self.sc.sequence_index == len(self.sc.rounds)-1
        tail = self._home_steps() if last else [Step('hold', ticks=10**6, name='round_done')]
        prep = ([Step('track', self._smooth_forecast_goal, name='forecast_follow',
                      guard=True, skip=self._placed_and_clear)] if self.rule == 'early' else [])
        return prep + [Step('wait', until=self._placed_and_clear, name='wait_coaster_ready')] + self._place_steps(self.obj) + tail

    def _observe(self):
        # Reuse real causal ManiCast inference, but never use momentary slowing as
        # permission for Observed to transport a cup before placement is finished.
        super()._observe()
        if ('coaster_placed_'+self.coaster) not in self.sc.protocol_done:
            self.spot = None

    def _placed_and_clear(self):
        sc=self.sc
        placed=('coaster_placed_'+self.coaster) in sc.protocol_done
        clear=placed and np.linalg.norm(sc.human.pos[:2]-sc.pos(self.coaster)[:2]) >= GUARD_OCCUPIED_XY
        stable=placed and sc.speed(self.coaster) < .02 and sc.coaster_bottom_gap(self.coaster) <= .001
        if not (clear and stable):
            self.clear_since=None
            return False
        if self.clear_since is None:self.clear_since=sc.time
        return sc.time-self.clear_since >= .15-1e-8

    def _smooth_forecast_goal(self):
        destination=self._destination()
        desired=(np.r_[np.asarray(destination)[:2]+self.tcp_obj_offset[:2],self.travel]
                 if destination is not None else np.r_[self.mapper.setpoint[:2],self.travel])
        if self.filtered_goal is None:self.filtered_goal=self.mapper.setpoint.copy()
        delta=np.asarray(desired)-self.filtered_goal
        step=delta*(1-np.exp(-self.sc.dt/.14))
        norm=np.linalg.norm(step); limit=.30*self.sc.dt
        if norm>limit:step*=limit/norm
        self.filtered_goal+=step
        # Vertical travel height is maintained after grasp/lift, never following
        # a downward wrist forecast into a hand that is placing the coaster.
        self.filtered_goal[2]=self.travel
        return self.filtered_goal.copy()

    def act(self):
        self._observe()
        # One safety authority: hold rather than competing retreat/replan goals.
        # Existing guarded moves and human clearance remain active.
        if separation_hold(self):
            self.preview_safety_holds+=1
            return int(A.HOLD)
        return ScriptedExpert.act(self)

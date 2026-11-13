"""Local I1 preview rules requested by the user; no changes to the stable expert."""
import numpy as np
from intent_policy.experts.human_paced import HumanPace, CartesianFollower, work_speed, WORK_ACCEL
from intent_policy.experts.i1_expert import I1Expert
from intent_policy.experts.scripted_expert import ScriptedExpert, Step, GUARD_OCCUPIED_XY
from intent_policy.experts.separation import separation_hold
from intent_policy.sim.restricted_action import RestrictedAction as A


HAND_CLEAR_M = 0.16   # robot keeps this xy distance from the observed and the predicted hand while following


class SmoothI1Expert(I1Expert):
    def reset(self):
        self.filtered_goal = None
        self.human_pace = HumanPace(self.sc.dt)
        self.forecast_follower = CartesianFollower(self.sc.dt, acceleration=WORK_ACCEL)
        self.clear_since = None
        self.preview_safety_holds = 0
        super().reset()

    def _grasp_steps(self, key):
        # 13/11: the cup is taken before the human starts; the 3 s settle pauses of the stable expert are not needed
        steps = super()._grasp_steps(key)
        for st in steps:
            if st.name in ('settle_above_grasp', 'settle_before_grasp'):
                st.ticks = 8
        return steps

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
        # 13/11 (user): chase the predicted spot itself as soon as ManiCast predicts it; the only
        # correction is a safety margin around the predicted hand (forecast at +1 s, palm offset).
        if destination is not None:
            goal_xy=np.asarray(destination)[:2]+self.tcp_obj_offset[:2]
            if self._gb is not None and self._gb.forecast is not None and self._gb.kp:
                wrist=np.asarray(self._gb.kp[-1][5],float)[:2]
                hand_pred=np.asarray(self._gb.forecast[-1][:2],float)+(self.sc.human.pos[:2]-wrist)
                for hand in (hand_pred, self.sc.human.pos[:2]):
                    away=goal_xy-hand; dist=float(np.linalg.norm(away))
                    if dist<HAND_CLEAR_M:
                        u=away/dist if dist>1e-6 else (self.mapper.setpoint[:2]-hand)/max(float(np.linalg.norm(self.mapper.setpoint[:2]-hand)),1e-6)
                        goal_xy=hand+HAND_CLEAR_M*u
            desired=np.r_[goal_xy,self.travel]
        else:
            desired=np.r_[self.mapper.setpoint[:2],self.travel]
        if self.filtered_goal is None:self.filtered_goal=self.mapper.setpoint.copy()
        predicted_speed = 0.
        if self._gb is not None and self._gb.forecast is not None:
            future = np.asarray(self._gb.forecast, float)
            if len(future) > 1:
                predicted_speed = float(np.median(np.linalg.norm(np.diff(future, axis=0), axis=-1) / .1))
        self.human_pace.update(np.linalg.norm(self.sc.human.vel), predicted_speed)
        # Human-paced only while the coaster is still being carried; once it is down the robot
        # closes in on the stand-off point at independent work speed (user, 13/11).
        placed = ('coaster_placed_' + self.coaster) in self.sc.protocol_done
        speed = work_speed('move') if placed else self.human_pace.speed(.24, following=True)
        self.filtered_goal = self.forecast_follower.step(self.mapper.setpoint, desired, speed)
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

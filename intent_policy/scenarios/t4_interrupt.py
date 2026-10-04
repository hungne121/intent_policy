"""T4 interrupt (intruder): the robot starts by itself (no instruction) and puts the single cube into the fixed
place zone P1; the human puts a hand into the robot zone (configs/layout.yaml `robot_zone`) in one of three phases
and holds it there. While the hand is inside the zone the robot must be stationary (after a reaction grace time);
it then resumes and completes the task. A robot that keeps moving is detected whatever path it takes. T4-neg: the
hand moves near but outside the zone and the robot must not stop (scence_construct.md §4.4, simplified 2026-09-30)."""
import numpy as np
from intent_policy.benchmark.events import EventType as E
from intent_policy.scenarios.t1_pick_place import PickPlaceTask


class InterruptTask(PickPlaceTask):
    def on_reset(self):
        super().on_reset()
        spec = self.variation['spec']
        self.negative = bool(spec.get('negative'))
        sc = self.cfg.safety_constraints
        self.max_speed = float(sc['max_tcp_speed_in_zone_m_s'])
        self.grace = float(sc.get('yield_grace_s', 0.5))
        self.zone_entry_t = None
        self.in_zone = False
        self.disrupted = self.disruption_over = False
        self.ever_in_zone = False                       # the hand was inside the robot zone at some point (plan v5)
        self.yield_violations = 0
        self.recovering = self.recovered = False
        self.still_since = None
        self.longest_stop = 0.0

    def oracle_slots(self, evidence):
        """The cube and P1 are known from the start (no instruction); the interaction region is the intrusion point
        from its evidence until the hand withdraws."""
        slots = {'object': self.robot_target(), 'region': self.place_zone}
        if any(ev['kind'] == 'intrusion' for ev in evidence) and self.human.stage in ('intrude', 'dwelling'):
            slots['region'] = 'intrusion_point'
        return slots

    def oracle_position(self, key):
        if key == 'intrusion_point':
            return np.asarray(self.human.extras['intrusion_point'], float)
        return super().oracle_position(key)

    def _update_positive(self, t):
        self.in_zone = self.in_robot_zone(self.human.pos)
        self.zone_entry_t = (self.zone_entry_t if self.zone_entry_t is not None else t) if self.in_zone else None
        if not self.disrupted and self.in_zone:
            self.disrupted = True
            self.log(E.DISRUPTION_START, 'human', hand_position=self.human.pos.round(4).tolist(), phase=self.human.phase,
                     hand_tcp_distance=float(np.linalg.norm(self.human.pos - self.tcp())))
        if self.disrupted and not self.disruption_over:
            if self.in_zone and t - self.zone_entry_t > self.grace and self.tcp_speed > self.max_speed:
                self.yield_violations += 1
                if self.yield_violations == 1:
                    self.log(E.SAFETY_DISTANCE_VIOLATION, 'robot', kind='yield', tcp_speed=self.tcp_speed,
                             hand_tcp_distance=float(np.linalg.norm(self.human.pos - self.tcp())))
                    if self.cfg.success_conditions.get('require_yield', True):
                        self.fail('did_not_yield')
            if 'withdrawn_time' in self.human.extras and not self.in_zone:
                self.disruption_over = True
                self.log(E.DISRUPTION_END, 'human')
        elif not self.disrupted and 'withdrawn_time' in self.human.extras and not self.disruption_over:
            # (random intrusion timing) the hand stopped short of the zone, the arm being right there: nothing to yield to
            self.disruption_over = True
            self.protocol('disruption_handled', 'robot', entered_zone=False)
        if self.disruption_over and not self.recovering and self.robot_moving:
            self.recovering = True
            self.log(E.RECOVERY_START, 'robot')
            self.protocol('disruption_handled', 'robot', yield_violations=self.yield_violations)
        if self.recovering and not self.recovered and 'object_placed' in self.protocol_done:
            self.recovered = True
            self.log(E.RECOVERY_COMPLETE, 'robot')

    def _update_negative(self, t):
        """The robot may not stop (longer than negative_stop_max_s) while the hand moves near the zone."""
        active = self.human.stage in ('reach_near', 'dwelling', 'withdraw') and self.human.intruded \
            and 'withdrawn_time' not in self.human.extras and 'object_placed' not in self.protocol_done
        if active and not self.moving_now:
            self.still_since = t if self.still_since is None else self.still_since
            self.longest_stop = max(self.longest_stop, t - self.still_since)
            if self.longest_stop > float(self.cfg.success_conditions['negative_stop_max_s']):
                self.fail('unnecessary_stop')
        else:
            self.still_since = None
        if 'withdrawn_time' in self.human.extras and not self.disruption_over:
            self.disruption_over = True
            self.protocol('disruption_handled', 'robot', negative=True, longest_stop_s=round(self.longest_stop, 3))

    def update_task(self):
        t = self.time
        self.ever_in_zone = self.ever_in_zone or bool(self.in_robot_zone(self.human.pos))
        if self.negative:
            self._update_negative(t)
        else:
            self._update_positive(t)
        super().update_task()

    def success_predicate(self):
        ok_yield = self.yield_violations == 0 or not self.cfg.success_conditions.get('require_yield', True)
        return super().success_predicate() and self.disruption_over and ok_yield

    def ground_truth_intention_information(self):
        info = super().ground_truth_intention_information()
        info.update(intrusion_point=self.human.extras.get('intrusion_point'), negative=self.negative,
                    robot_zone=dict(lo=self.robot_zone_lo.tolist(), hi=self.robot_zone_hi.tolist()),
                    human_in_robot_zone=self.in_zone)
        return info

"""Intruder: nominal pick-and-place interrupted by a human hand entering the workspace.

Danger region = a fixed box around the robot workspace (config `scene.danger_zone`). While the
hand is inside it the robot must be stationary (after a reaction grace time); the task can
only be completed after the hand has left. A robot that keeps moving is therefore detected
whatever path it takes.
"""
import numpy as np
from benchmark.events import EventType as E
from scenarios.base_scenario import BaseScenario


class IntruderPickPlaceInterruption(BaseScenario):
    def on_reset(self):
        self.obj = self.variation['requested_object']
        self.target = self.variation['requested_target']
        zone = self.scene['danger_zone']
        self.zone_lo = np.array([zone[a][0] for a in 'xyz'], float)
        self.zone_hi = np.array([zone[a][1] for a in 'xyz'], float)
        self.zone_entry_t = None
        self.max_speed = float(self.cfg.safety_constraints['max_tcp_speed_in_danger_m_s'])
        self.grace = float(self.cfg.safety_constraints.get('yield_grace_s', 0.5))
        self.disrupted = False
        self.disruption_t = None
        self.disruption_over = False
        self.in_danger = False
        self.yield_violations = 0
        self.recovering = False
        self.recovered = False
        self.steps_before_recovery = None

    def oracle_slots(self, evidence):
        """Interaction region = the intrusion point, from its evidence until the hand withdraws."""
        active = any(ev['kind'] == 'intrusion' for ev in evidence) and self.human.stage in ('approaching', 'dwelling')
        return {'object': None, 'region': 'intrusion_point' if active else None}

    def oracle_position(self, key):
        if key == 'intrusion_point':
            return np.asarray(self.human.extras['intrusion_point'], float)
        return super().oracle_position(key)

    def robot_target(self):
        return self.obj

    def _resting(self):
        sc = self.cfg.success_conditions
        return (not self.grasp_state[self.obj] and self.speed(self.obj) < sc['object_speed_max']
                and self.pos(self.obj)[2] < self.table_z + sc['place_height_max_m'])

    def hand_in_zone(self) -> bool:
        return bool(np.all(self.human.pos >= self.zone_lo) and np.all(self.human.pos <= self.zone_hi))

    def update_task(self):
        t = self.time
        stage = self.human.stage
        self.in_danger = self.hand_in_zone()
        if self.in_danger and self.zone_entry_t is None:
            self.zone_entry_t = t
        elif not self.in_danger:
            self.zone_entry_t = None
        if not self.disrupted and self.in_danger:
            self.disrupted, self.disruption_t = True, t
            self.steps_before_recovery = list(self.protocol_done)
            self.log(E.DISRUPTION_START, 'human', hand_position=self.human.pos.round(4).tolist(),
                     hand_tcp_distance=float(np.linalg.norm(self.human.pos - self.tcp())))
        if self.disrupted and not self.disruption_over:
            if self.in_danger and t - self.zone_entry_t > self.grace and self.tcp_speed > self.max_speed:
                self.yield_violations += 1
                if self.yield_violations == 1:
                    self.log(E.SAFETY_DISTANCE_VIOLATION, 'robot', kind='yield', tcp_speed=self.tcp_speed,
                             hand_tcp_distance=float(np.linalg.norm(self.human.pos - self.tcp())))
                    if self.cfg.success_conditions.get('require_yield', True):
                        self.fail('did_not_yield')
            if stage in ('withdrawing', 'done') and not self.in_danger:
                self.disruption_over = True
                self.log(E.DISRUPTION_END, 'human')
        if self.disruption_over and not self.recovering and self.robot_moving:
            self.recovering = True
            self.log(E.RECOVERY_START, 'robot')
        if self.robot_lifted == self.obj:
            self.protocol('object_grasped', 'robot', object=self.obj)
        if self.recovering and not self.recovered and len(self.protocol_done) > len(self.steps_before_recovery or []):
            self.recovered = True
            self.protocol('disruption_handled', 'robot', yield_violations=self.yield_violations)
            self.log(E.RECOVERY_COMPLETE, 'robot')
        if self.obj in self.ever_lifted and self._resting():
            if self.inside_region(self.pos(self.obj), self.target, margin=0.01):
                if not self.disruption_over:
                    return      # placed before the intrusion finished: wait, no success yet
                if not self.recovered:
                    self.recovered = True
                    self.log(E.RECOVERY_COMPLETE, 'robot')
                self.protocol('disruption_handled', 'robot', yield_violations=self.yield_violations)
                self.protocol('object_released_in_target', 'robot', object=self.obj, target=self.target)
            else:
                self.fail('released_outside_target')

    def success_predicate(self):
        ok_yield = self.yield_violations == 0 or not self.cfg.success_conditions.get('require_yield', True)
        return ('object_released_in_target' in self.protocol_done and self.disruption_over and ok_yield
                and self._resting() and self.inside_region(self.pos(self.obj), self.target, margin=0.01))

    def ground_truth_intention_information(self):
        return dict(target_object=self.obj, target_region=self.target,
                    target_position=self.region_pos(self.target).round(5).tolist(),
                    intrusion_point=self.human.extras.get('intrusion_point'),
                    danger_zone=dict(lo=self.zone_lo.tolist(), hi=self.zone_hi.tolist()),
                    human_in_danger_region=self.in_danger)

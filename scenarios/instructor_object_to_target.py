"""Instructor: the human names an object and a target region; robot must place exactly that."""
import numpy as np
from scenarios.base_scenario import BaseScenario


class InstructorObjectToTarget(BaseScenario):
    @property
    def req_target(self) -> str:
        """The currently instructed region (the human may revise it once while the robot carries the object)."""
        return self.human.selected_target

    def on_reset(self):
        self.instruction_visible = False
        self.req_obj = self.variation['requested_object']

    def robot_target(self):
        return self.req_obj

    def oracle_slots(self, evidence):
        slots = {'object': None, 'region': None}
        for ev in evidence:
            if ev['kind'] == 'instruction':
                slots = {'object': ev['payload']['object'], 'region': ev['payload']['region']}
            elif ev['kind'] == 'target_region':
                slots['region'] = ev['payload']['region']
        return slots

    def commit_stage(self):
        if self.robot_lifted == self.req_obj:
            return 'region', {r: self.region_pos(r) for r in self.regions}, self.req_target
        return self._pre_grasp_stage('object', self.objects, self.req_obj)

    def _resting(self, key):
        sc = self.cfg.success_conditions
        return (not self.grasp_state[key] and self.speed(key) < sc['object_speed_max']
                and self.pos(key)[2] < self.table_z + sc['place_height_max_m'])

    def update_task(self):
        obj = self.req_obj
        if self.robot_lifted == obj:
            self.protocol('correct_object_grasped', 'robot', object=obj)
        if obj in self.ever_lifted and self._resting(obj):
            p = self.pos(obj)
            if self.inside_region(p, self.req_target, margin=0.01):
                self.protocol('object_released_in_target', 'robot', object=obj, target=self.req_target)
            elif any(self.inside_region(p, r) for r in self.regions if r != self.req_target):
                self.fail('wrong_target_region')
            else:
                self.fail('released_outside_target')

    def success_predicate(self):
        return 'object_released_in_target' in self.protocol_done and self._resting(self.req_obj) \
            and self.inside_region(self.pos(self.req_obj), self.req_target, margin=0.01)

    def ground_truth_intention_information(self):
        return dict(target_object=self.req_obj, target_object_position=self.pos(self.req_obj).round(5).tolist(),
                    target_region=self.req_target, target_position=self.region_pos(self.req_target).round(5).tolist(),
                    interaction_region=self.region_pos(self.req_target).round(5).tolist())

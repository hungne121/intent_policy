"""T2 handover (collaborator): the human points at an object and holds an open hand out, palm up, in a hand zone H
(early / on time / late); the robot must put exactly that object into the waiting palm, hold still and release only
when the human has closed the hand on it and pulls. T5 on T2: the human switches objects before the robot grasps
(scence_construct.md §4.4)."""
import numpy as np
from intent_policy.scenarios.base_scenario import BaseScenario


class HandoverTask(BaseScenario):
    def on_reset(self):
        self.hand_zone = self.variation['target_zone']

    def robot_target(self):
        return self.human.selected_object

    def present_point(self) -> np.ndarray:
        """Where the human expects the object (resting in the upturned, waiting palm)."""
        obj = self.robot_target()
        return self.zone_pos(self.hand_zone) + np.asarray(self.cfg.human_behavior['present_offset'], float) \
            + [0.0, 0.0, self.rest_height[obj] - self.table_z]

    def oracle_slots(self, evidence):
        slots = {'object': None, 'region': None}
        for ev in evidence:
            if ev['kind'] == 'target_object':
                slots['object'] = ev['payload']['object']
            elif ev['kind'] == 'target_withdrawn':        # T5: the pointing hand is withdrawn, target unknown again
                slots['object'] = None
            elif ev['kind'] == 'target_zone':
                slots['region'] = ev['payload']['zone']
        return slots

    def commit_stage(self):
        return self._pre_grasp_stage('object', list(self.on_table), self.robot_target())

    def update_task(self):
        obj = self.robot_target()
        if self.robot_lifted == obj:
            self.protocol('object_grasped', 'robot', object=obj)
            if np.linalg.norm(self.pos(obj) - self.present_point()) < self.cfg.human_behavior['accept_radius_m'] \
                    and self.tcp_speed < 0.02:
                self.protocol('robot_at_handover', 'robot', object=obj)
        held_by_human = self.human.holding == obj
        if obj in self.ever_lifted and not held_by_human and not self.grasp_state[obj] and self.speed(obj) < 0.02 \
                and self.pos(obj)[2] < self.table_z + 0.1:
            self.fail('object_dropped')          # also: released before the human held it

    def success_predicate(self):
        obj = self.robot_target()
        return ('object_transferred' in self.protocol_done and self.human.holding == obj
                and not self.grasp_state[obj] and not self.env.finger_contacts(self.body_id[obj]))

    def ground_truth_intention_information(self):
        obj = self.robot_target()
        hand = self.zone_pos(self.hand_zone).round(5).tolist()
        return dict(target_object=obj, target_object_position=self.pos(obj).round(5).tolist(),
                    target_region=self.hand_zone, interaction_region=hand, target_position=hand)

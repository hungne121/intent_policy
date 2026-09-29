"""Collaborator handover: pick the object the human selected and hand it over."""
import numpy as np
from intent_policy.scenarios.base_scenario import BaseScenario


class CollaboratorObjectHandover(BaseScenario):
    @property
    def sel(self) -> str:
        """The human's current choice (may change once before the robot grasps, see `intention_change`)."""
        return self.human.selected_object

    def on_reset(self):
        self.receive_pose = np.array(self.variation['human']['receive_pose'])
        self.accept = self.cfg.human_behavior['accept_radius_m']

    def robot_target(self):
        return self.sel

    def oracle_slots(self, evidence):
        objs = [ev['payload']['object'] for ev in evidence if ev['kind'] == 'target_object']
        return {'object': objs[-1], 'region': 'receive_pose'} if objs else {'object': None, 'region': None}

    def oracle_position(self, key):
        return self.receive_pose.copy() if key == 'receive_pose' else super().oracle_position(key)

    def commit_stage(self):
        return self._pre_grasp_stage('object', self.objects, self.sel)

    def update_task(self):
        obj = self.sel
        if self.robot_lifted == obj:
            self.protocol('correct_object_grasped', 'robot', object=obj)
            if np.linalg.norm(self.pos(obj) - self.receive_pose) < self.accept + 0.02:
                self.protocol('robot_at_handover', 'robot', object=obj)
        held_by_human = self.human.holding == obj
        if obj in self.ever_lifted and not held_by_human and not self.grasp_state[obj] and self.speed(obj) < 0.02 \
                and self.pos(obj)[2] < self.rest_height[obj] + 0.01:
            self.fail('object_dropped')

    def success_predicate(self):
        obj = self.sel
        return ('object_transferred' in self.protocol_done and self.human.holding == obj
                and not self.grasp_state[obj] and not self.env.finger_contacts(self.body_id[obj]))

    def ground_truth_intention_information(self):
        return dict(target_object=self.sel, target_object_position=self.pos(self.sel).round(5).tolist(),
                    interaction_region=self.receive_pose.round(5).tolist(), target_position=self.receive_pose.round(5).tolist())

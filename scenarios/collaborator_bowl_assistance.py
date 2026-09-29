"""Collaborator bowl assistance: human picks an object; robot must deliver the matching bowl."""
import numpy as np
from scipy.spatial.transform import Rotation
from scenarios.base_scenario import BaseScenario


class CollaboratorBowlAssistance(BaseScenario):
    def on_reset(self):
        self.sel = self.variation['selected_object']
        self.bowl_for = {v['pairs_with']: k for k, v in self.scene['bowls'].items()}
        self.bowl = self.bowl_for[self.sel]
        self.delivery = np.array(self.scene['delivery_spot'], float)
        self.tol = float(self.scene['delivery_tolerance_m'])

    def robot_target(self):
        return self.bowl

    def oracle_slots(self, evidence):
        objs = [ev['payload']['object'] for ev in evidence if ev['kind'] == 'target_object']
        return {'object': self.bowl_for[objs[-1]], 'region': 'delivery_spot'} if objs else {'object': None, 'region': None}

    def oracle_position(self, key):
        return np.r_[self.delivery, self.table_z] if key == 'delivery_spot' else super().oracle_position(key)

    def commit_point(self, key):
        """Bowls are grasped at the handle on the robot side of the rim."""
        return self.pos(key) - [self.scene['bowls'][key].get('radius', 0.05) + 0.03, 0.0, 0.0]

    def commit_stage(self):
        return self._pre_grasp_stage('bowl', self.bowls, self.bowl)

    def _upright(self, key):
        w, x, y, z = self.quat(key)
        return Rotation.from_quat([x, y, z, w]).apply([0, 0, 1])[2] > 0.9

    def bowl_ready(self, key):
        p = self.pos(key)
        return bool(np.linalg.norm(p[:2] - self.delivery) < self.tol and not self.grasp_state[key]
                    and self.speed(key) < 0.02 and p[2] < self.table_z + 0.02 and self._upright(key))

    def human_context_extras(self):
        return dict(bowl_ready={k: self.bowl_ready(k) for k in self.bowls})

    def object_in_bowl(self):
        o, b = self.pos(self.sel), self.pos(self.bowl)
        return bool(np.linalg.norm(o[:2] - b[:2]) < 0.045 and 0.0 < o[2] - b[2] < 0.06 and self.speed(self.sel) < 0.04
                    and self.human.holding is None)

    def update_task(self):
        if self.robot_lifted == self.bowl:
            self.protocol('correct_bowl_grasped', 'robot', bowl=self.bowl)
        if self.bowl in self.ever_lifted and self.bowl_ready(self.bowl):
            self.protocol('bowl_delivered', 'robot', bowl=self.bowl)
        for k in self.bowls:
            if not self._upright(k):
                self.fail('bowl_upset')
        if self.human.stage in ('withdrawing', 'done') and self.speed(self.sel) < 0.02:
            if self.object_in_bowl():
                self.protocol('object_placed_in_bowl', 'human', object=self.sel, bowl=self.bowl)
            else:
                self.fail('object_missed_bowl')

    def success_predicate(self):
        return ('object_placed_in_bowl' in self.protocol_done and self.object_in_bowl()
                and np.linalg.norm(self.pos(self.bowl)[:2] - self.delivery) < self.tol)

    def ground_truth_intention_information(self):
        return dict(target_object=self.bowl, human_selected_object=self.sel,
                    target_object_position=self.pos(self.bowl).round(5).tolist(),
                    target_position=np.r_[self.delivery, self.table_z].round(5).tolist(),
                    interaction_region=np.r_[self.delivery, self.table_z].round(5).tolist())

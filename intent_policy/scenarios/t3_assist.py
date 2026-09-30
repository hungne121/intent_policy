"""T3 assist (collaborator), a learned workflow: two cubes lie in U, each paired with a cup on a slot. The human
picks up one cube (no pointing); the robot must anticipate which cup belongs to it, stand that cup in the cup spot
of U (U_cup, the former support zone A) and return to rest; the human then drops the cube into it. T5 on T3: the
human switches cubes (and so cups) before the robot grasps (scence_construct.md §4.4, reworked 2026-09-30)."""
import numpy as np
from scipy.spatial.transform import Rotation
from intent_policy.scenarios.base_scenario import BaseScenario


class AssistTask(BaseScenario):
    def on_reset(self):
        self.support = self.variation['target_zone']

    @property
    def block(self) -> str:
        """The cube the human (currently) wants to put into its cup."""
        return self.human.block

    def robot_target(self):
        return self.human.selected_object

    def oracle_slots(self, evidence):
        cup = None
        for ev in evidence:
            if ev['kind'] in ('target_object', 'target_withdrawn'):    # T5: withdrawn hand -> unknown again
                cup = ev['payload'].get('object')
        return {'object': cup, 'region': self.support} if cup else {'object': None, 'region': None}

    def commit_stage(self):
        return self._pre_grasp_stage('object', list(self.variation['spec']['layout']), self.robot_target())

    def upright(self, key) -> bool:
        w, x, y, z = self.quat(key)
        return Rotation.from_quat([x, y, z, w]).apply([0, 0, 1])[2] > self.cfg.success_conditions['upright_min']

    def cup_ready(self) -> bool:
        cup = self.robot_target()
        return bool(cup in self.ever_lifted and self.resting(cup) and self.upright(cup)
                    and self.inside_zone(self.pos(cup), self.support, self.cfg.success_conditions['zone_margin_m']))

    def human_context_extras(self):
        return dict(cup_ready=self.cup_ready())

    def block_in_cup(self) -> bool:
        o, c = self.pos(self.block), self.pos(self.robot_target())
        r, h = self.scene['objects'][self.robot_target()]['size']
        return bool(np.linalg.norm(o[:2] - c[:2]) < r and o[2] - c[2] < h / 2 and self.speed(self.block) < 0.04
                    and self.human.holding is None)

    def update_task(self):
        cup = self.robot_target()
        if self.robot_lifted == cup:
            self.protocol('cup_grasped', 'robot', cup=cup)
        if self.cup_ready():
            self.protocol('cup_at_support', 'robot', cup=cup)
        elif cup in self.ever_lifted and self.resting(cup) and 'cup_at_support' not in self.protocol_done:
            self.fail('cup_not_at_support' if self.upright(cup) else 'cup_upset')
        if 'cup_at_support' in self.protocol_done and self.robot_at_rest():
            self.protocol('robot_at_rest', 'robot')
        for k in self.variation['spec']['layout']:
            if not self.upright(k):
                self.fail('cup_upset')
        if self.human.stage in ('withdraw', 'done') and self.speed(self.block) < 0.02:
            if self.block_in_cup():
                self.protocol('block_in_cup', 'human', block=self.block, cup=cup)
            else:
                self.fail('block_missed_cup')

    def success_predicate(self):
        return 'block_in_cup' in self.protocol_done and self.block_in_cup() and self.cup_ready() and self.robot_at_rest()

    def ground_truth_intention_information(self):
        cup = self.robot_target()
        a = self.zone_pos(self.support).round(5).tolist()
        return dict(target_object=cup, human_block=self.block, target_object_position=self.pos(cup).round(5).tolist(),
                    target_region=self.support, target_position=a, interaction_region=a)

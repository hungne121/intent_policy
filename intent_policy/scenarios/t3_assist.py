"""T3 assist (collaborator), a learned workflow: two cubes lie in U, each paired with a cup on a slot. The human
picks up one cube (no pointing); the robot must anticipate which cup belongs to it, stand that cup in the cup spot
of U (U_cup, the former support zone A) and return to rest; the human then drops the cube into it. T5 on T3: the
human switches cubes (and so cups) before the robot grasps (scence_construct.md §4.4, reworked 2026-09-30).

T3 new (human_behavior.handover, 2026-10-18): after the drop the human holds an open hand out, palm up, in the hand zone
variation.hand_zone; the robot picks the cup (with the cube) up again and puts it into the palm, released on the pull
as in T2. Success = the human holds the cup with the cube inside, the robot has let go."""
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from intent_policy.scenarios.base_scenario import BaseScenario
from intent_policy.scenarios.t2_handover import OBJECT_CONTYPE, PALM_SUPPORT_CLEAR_M, PALM_SUPPORT_STAGES

RECEIVE_STAGES = ('reach_out', 'waiting_offer', 'take_object', 'pulling', 'wait_release', 'released_by_robot',
                  'withdraw_with_object', 'done')


class AssistTask(BaseScenario):
    def on_reset(self):
        self.support = self.variation['target_zone']
        self.handover = bool(self.cfg.human_behavior.get('handover'))
        self.hand_zone = self.variation.get('hand_zone')
        self._drop_checked = False
        gid = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_GEOM, 'human_palm_support')
        self.palm_support = gid if gid >= 0 and self.handover else None
        self._set_palm_support(False)

    # -- T3 new: the cup with the cube into the open hand (as t2_handover.HandoverTask)
    def _set_palm_support(self, on: bool) -> None:
        if self.palm_support is not None:
            m = self.env.model
            m.geom_conaffinity[self.palm_support] = m.body_conaffinity[m.geom_bodyid[self.palm_support]] = \
                OBJECT_CONTYPE if on else 0

    def _palm_support_wanted(self, obj: str) -> bool:
        if self.palm_support is None or self.human.stage not in PALM_SUPPORT_STAGES:
            return False
        m, d = self.env.model, self.env.data
        gap = (self.pos(obj)[2] - (self.rest_height[obj] - self.table_z)) - (d.geom_xpos[self.palm_support][2] + m.geom_size[self.palm_support][2])
        return gap >= PALM_SUPPORT_CLEAR_M or not self.grasp_state[obj]

    def present_point(self) -> np.ndarray:
        """Where the human expects the cup: resting in the upturned, waiting palm (T2's present_offset + half height)."""
        cup = self.robot_target()
        return self.zone_pos(self.hand_zone) + np.asarray(self.cfg.human_behavior['present_offset'], float) \
            + [0.0, 0.0, self.rest_height[cup] - self.table_z]

    def cube_in_cup(self) -> bool:
        """The cube lies inside the cup (wherever the cup is)."""
        o, c = self.pos(self.block), self.pos(self.robot_target())
        r, h = self.scene['objects'][self.robot_target()]['size']
        return bool(np.linalg.norm(o[:2] - c[:2]) < r and -h / 2 < o[2] - c[2] < h / 2)

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
        if self.handover:
            return self._update_handover()
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

    def _update_handover(self):
        cup = self.robot_target()
        self._set_palm_support(self._palm_support_wanted(cup))
        if self.robot_lifted == cup:
            self.protocol('cup_grasped', 'robot', cup=cup)
        if 'block_in_cup' not in self.protocol_done:
            if self.cup_ready():
                self.protocol('cup_at_support', 'robot', cup=cup)
            elif cup in self.ever_lifted and self.resting(cup) and 'cup_at_support' not in self.protocol_done:
                self.fail('cup_not_at_support' if self.upright(cup) else 'cup_upset')
        for k in self.variation['spec']['layout']:
            if not self.upright(k):
                self.fail('cup_upset')
        if not self._drop_checked and self.human.stage in RECEIVE_STAGES and self.speed(self.block) < 0.02:
            self._drop_checked = True             # the cube was dropped and has settled
            if self.block_in_cup():
                self.protocol('block_in_cup', 'human', block=self.block, cup=cup)
            else:
                self.fail('block_missed_cup')
        if 'block_in_cup' in self.protocol_done and self.robot_lifted == cup and self.grasp_state[cup] \
                and np.linalg.norm(self.pos(cup) - self.present_point()) < self.cfg.human_behavior['accept_radius_m'] \
                and self.tcp_speed < 0.02:
            self.protocol('robot_at_handover', 'robot', object=cup)
        if 'block_in_cup' in self.protocol_done and self.human.holding != cup and not self.grasp_state[cup] \
                and self.speed(cup) < 0.02 and not self.cup_ready() and self.pos(cup)[2] < self.table_z + 0.1 \
                and self.human.stage not in ('take_object',):
            self.fail('object_dropped')       # let go of before the human held it, away from U_cup and the palm

    def success_predicate(self):
        if self.handover:
            cup = self.robot_target()
            return ('object_transferred' in self.protocol_done and self.human.holding == cup and not self.grasp_state[cup]
                    and not self.env.finger_contacts(self.body_id[cup]) and self.cube_in_cup())
        return 'block_in_cup' in self.protocol_done and self.block_in_cup() and self.cup_ready() and self.robot_at_rest()

    def ground_truth_intention_information(self):
        cup = self.robot_target()
        a = self.zone_pos(self.support).round(5).tolist()
        return dict(target_object=cup, human_block=self.block, target_object_position=self.pos(cup).round(5).tolist(),
                    target_region=self.support, target_position=a, interaction_region=a)

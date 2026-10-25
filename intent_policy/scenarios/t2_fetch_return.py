"""v6 T2 fetch and return (collaborator; HABIT 'Retrieve and hand over the desired beverage to the person', S7045,
and the give-back of S6760; docs/sim/scenarios_v6_proposal.md): the human works on the piece in U, then holds the open
hand out (a hand zone plus an offset); the robot must put the can into that hand as in T2, wait while the human uses
it, take it back where the human holds it out (another hand zone plus an offset), put it back on its slot and return
to rest. When (work progress) and where (the open hand) are the intention; there is one object."""
import numpy as np
from intent_policy.scenarios.t2_handover import HandoverTask


HAND_OUT = ('waiting_offer', 'take_object', 'pulling', 'wait_release', 'released_by_robot', 'withdraw_with_object',
            'use', 'using', 'wait_robot_take', 'react')   # the receiving / giving-back hand is out (or holds the can)


class FetchReturnTask(HandoverTask):
    v6_plan = 'fetch_return'                    # the expert's v6 plan (scripted_expert._task_steps)
    cue_complete_step = 'hand_ready'            # expert trigger cue_complete: the open hand is out

    def zone_pos(self, key: str) -> np.ndarray:
        """Hand zones: where the human's open hand actually is (the zone plus the episode's offset; observable once the
        hand is out). The give-back zone the same with its own offset."""
        p = super().zone_pos(key)
        if self.zones.get(key, {}).get('kind') != 'hand' or self.human is None:
            return p
        spec, prm = self.variation['spec'], self.human.params
        if self.human.stage not in HAND_OUT:          # the actual hand is known only once it is out and still
            return p                                  #   (observable); before that the nominal zone (non-privileged)
        if key == spec['return_hand'] and self.human.selected_target == key and 'object_transferred' in self.protocol_done:
            return p + np.r_[prm['return_offset'], float(self.cfg.human_behavior['return_dz_m'])]
        return p + np.r_[prm['hand_offset'], 0.0]

    def human_context(self) -> dict:
        """The human aims at the nominal zones and adds its own offsets (sim/scripted_human.WorkerReceiver); the robot
        sees the offset hand through zone_pos only once the hand is out."""
        from intent_policy.scenarios.base_scenario import BaseScenario
        ctx = super().human_context()
        ctx['zone_positions'] = {k: BaseScenario.zone_pos(self, k) for k in self.zones}
        return ctx

    def handover_only(self) -> bool:
        """`success_conditions.handover_only`: the episode is the hand-over alone (docs/PIF.md §4, C1 first; 2026-10-25):
        success once the human holds the can (T2's predicate), no give-back."""
        return bool(self.cfg.success_conditions.get('handover_only', False))

    def returning(self) -> bool:
        return 'object_transferred' in self.protocol_done and not self.handover_only()

    def update_task(self):
        obj = self.robot_target()
        if not self.returning():
            super().update_task()                # the handover (T2): grasp, at the hand, dropped
            return
        self._set_palm_support(False)
        if self.grasp_state[obj] and self.human.returned:
            self.protocol('object_taken_back', 'robot', object=obj)
        if 'object_taken_back' in self.protocol_done and not self.grasp_state[obj] and self.speed(obj) < 0.02 \
                and 'object_put_back' not in self.protocol_done:
            back = np.linalg.norm(self.pos(obj)[:2] - self.start_pos[obj][:2]) <= self.cfg.success_conditions['put_back_max_m']
            if back and abs(self.pos(obj)[2] - self.rest_height[obj]) < 0.01:
                self.protocol('object_put_back', 'robot', object=obj)
            elif self.pos(obj)[2] < self.table_z + 0.1:
                self.fail('object_dropped')
        if self.human.returned and not self.grasp_state[obj] and 'object_taken_back' not in self.protocol_done \
                and self.speed(obj) < 0.02 and self.pos(obj)[2] < self.table_z + 0.1:
            self.fail('object_dropped')         # let go by the human before the robot held it
        if 'object_put_back' in self.protocol_done and self.robot_at_rest():
            self.protocol('robot_at_rest', 'robot')

    def success_predicate(self):
        if self.handover_only():
            return super().success_predicate()
        obj = self.robot_target()
        return 'robot_at_rest' in self.protocol_done and 'object_put_back' in self.protocol_done \
            and not self.grasp_state[obj] and self.robot_at_rest()

    def ground_truth_intention_information(self):
        info = super().ground_truth_intention_information()
        info['return_region'] = self.variation['spec']['return_hand']
        return info

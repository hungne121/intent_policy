"""T1 pick and place (instructor): the human points at a cube and at a place zone P; the robot must put exactly that
cube into that zone without touching the other objects and return to rest. T5 on T1: the human switches cubes
before the robot grasps (scence_construct.md §4.4)."""
from intent_policy.scenarios.base_scenario import BaseScenario


class PickPlaceTask(BaseScenario):
    def on_reset(self):
        self.place_zone = self.variation['target_zone']
        self.places = [k for k, z in self.zones.items() if z['kind'] == 'place']

    def robot_target(self):
        """The cube the human currently wants (changes once in T5)."""
        return self.human.selected_object

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
        target = self.robot_target()
        if self.robot_lifted == target:
            return 'region', {z: self.zone_pos(z) for z in self.places}, self.place_zone
        return self._pre_grasp_stage('object', [k for k in self.on_table], target)

    def placed(self) -> bool:
        obj = self.robot_target()
        return bool(obj in self.ever_lifted and self.resting(obj)
                    and self.inside_zone(self.pos(obj), self.place_zone, self.cfg.success_conditions['zone_margin_m']))

    def update_task(self):
        obj = self.robot_target()
        if self.robot_lifted == obj:
            self.protocol('object_grasped', 'robot', object=obj)
        if obj in self.ever_lifted and self.resting(obj) and 'object_placed' not in self.protocol_done:
            p = self.pos(obj)
            if self.placed():
                self.protocol('object_placed', 'robot', object=obj, zone=self.place_zone)
            elif any(self.inside_zone(p, z) for z in self.places if z != self.place_zone):
                self.fail('wrong_place_zone')
            else:
                self.fail('released_outside_zone')
        if 'object_placed' in self.protocol_done and self.robot_at_rest():
            self.protocol('robot_at_rest', 'robot')

    def success_predicate(self):
        return 'robot_at_rest' in self.protocol_done and self.placed() and self.robot_at_rest()

    def ground_truth_intention_information(self):
        obj = self.robot_target()
        return dict(target_object=obj, target_object_position=self.pos(obj).round(5).tolist(),
                    target_region=self.place_zone, target_position=self.zone_pos(self.place_zone).round(5).tolist(),
                    interaction_region=self.zone_pos(self.place_zone).round(5).tolist())

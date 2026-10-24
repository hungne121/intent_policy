"""v6 T3 set the cup on the coaster (collaborator; HABIT 'Place coasters on the table and set cups on them',
docs/sim/scenarios_v6_proposal.md): the human takes the coaster from U and puts it down at a spot of their choice in
the shared area (K1-K3 plus a small offset); the robot must stand the cup on that coaster, without touching the human
or the coaster, and return to rest. The spot is the intention; the carrying motion reveals it."""
import numpy as np
from intent_policy.scenarios.base_scenario import BaseScenario
from intent_policy.sim.scene_builder import object_half_height


class CoasterCupTask(BaseScenario):
    v6_plan = 'coaster'                         # the expert's v6 plan (scripted_expert._task_steps)

    def on_reset(self):
        self.coaster = self.variation['spec']['u_blocks'][0]
        self.spot = self.variation['target_zone']
        self.spots = [k for k, z in self.zones.items() if z['kind'] == 'coaster']
        self.coaster_top = 2 * object_half_height(self.scene['objects'][self.coaster])
        self.placed_xy: np.ndarray | None = None      # where the human put the coaster down

    def coaster_down(self) -> bool:
        return 'coaster_placed' in self.protocol_done

    def zone_pos(self, key: str) -> np.ndarray:
        """Coaster spots: the coaster itself once it has been put down there (observable), else the nominal spot."""
        if key == self.spot and self.zones[key]['kind'] == 'coaster' and self.coaster_down():
            return np.r_[self.pos(self.coaster)[:2], self.table_z]
        return super().zone_pos(key)

    def place_dz(self, key: str) -> float:
        """The cup stands on the coaster, this much above its table rest height."""
        return self.coaster_top if self.zones.get(key, {}).get('kind') == 'coaster' else 0.0

    def robot_target(self):
        return self.variation['target_object']

    def oracle_slots(self, evidence):
        slots = {'object': self.robot_target(), 'region': None}
        for ev in evidence:
            if ev['kind'] == 'target_zone':
                slots['region'] = ev['payload']['zone']
        return slots

    def commit_stage(self):
        cup = self.robot_target()
        if self.robot_lifted == cup:            # commit points: where the cup would stand on each spot
            up = np.array([0.0, 0.0, self.rest_height[cup] - self.table_z])
            return 'region', {z: np.r_[super(CoasterCupTask, self).zone_pos(z)[:2], self.table_z] + up for z in self.spots}, self.spot
        return None

    def on_coaster(self) -> bool:
        cup, sc = self.robot_target(), self.cfg.success_conditions
        p, c = self.pos(cup), self.pos(self.coaster)
        return bool(np.linalg.norm(p[:2] - c[:2]) <= sc['on_coaster_xy_m']
                    and abs(p[2] - (self.rest_height[cup] + self.coaster_top)) <= 0.005)

    def update_task(self):
        cup = self.robot_target()
        if self.robot_lifted == cup:
            self.protocol('cup_grasped', 'robot', object=cup)
        if cup in self.ever_lifted and not self.grasp_state[cup] and self.speed(cup) < 0.02 \
                and 'cup_on_coaster' not in self.protocol_done:
            if self.on_coaster():
                self.protocol('cup_on_coaster', 'robot', object=cup, zone=self.spot)
            elif self.pos(cup)[2] < self.rest_height[cup] + self.coaster_top + 0.01:
                self.fail('cup_not_on_coaster')
        if 'cup_on_coaster' in self.protocol_done and self.robot_at_rest():
            self.protocol('robot_at_rest', 'robot')
        if self.coaster_down() and self.human.holding is None:
            if self.placed_xy is None:
                self.placed_xy = self.pos(self.coaster)[:2].copy()
            elif np.linalg.norm(self.pos(self.coaster)[:2] - self.placed_xy) > 0.02:
                self.fail('coaster_moved')              # pushed by the robot or the cup

    def success_predicate(self):
        return 'robot_at_rest' in self.protocol_done and self.on_coaster() and self.robot_at_rest()

    def ground_truth_intention_information(self):
        cup = self.robot_target()
        spot = self.zone_pos(self.spot).round(5).tolist()
        return dict(target_object=cup, target_object_position=self.pos(cup).round(5).tolist(),
                    target_region=self.spot, target_position=spot, interaction_region=spot)

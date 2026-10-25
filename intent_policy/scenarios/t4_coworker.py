"""v6 T4 coworker (HABIT 'Place the bread into the airtight container', S8791; docs/sim/scenarios_v6_proposal.md): the
robot puts the two rolls into the two open boxes by itself (the roll on box A's side -> box A first, the other -> box B;
the rolls look alike, so the order is set by what the cameras see). The human closes the lid of
each box once its roll is in and the gripper has moved away, and at a random moment reaches into the robot zone for
the spice box. While the hand is in the robot zone the robot must be stationary (T4's yield rule, InterruptTask); it
must never touch the human; it completes the task and returns to rest."""
import numpy as np
from intent_policy.scenarios.t4_interrupt import InterruptTask
from intent_policy.sim.scene_builder import object_half_height


class CoworkerTask(InterruptTask):
    v6_plan = 'coworker'                        # the expert's v6 plan (scripted_expert._task_steps)

    def on_reset(self):
        super().on_reset()
        cw = self.variation['coworker']
        # The rolls look identical and their slots are random, so the order must be observable (2026-10-25,
        # docs/reports/ACT_v6_failure_analysis.md): the roll on box A's side (smaller x, as box A) goes into box A,
        # first; the other into box B. Slot x values differ by >= 3 cm (layout.yaml), the jitter is 0.8 cm.
        xy = self.variation['object_xy']
        breads = sorted(cw['breads'], key=lambda b: float(xy[b][0]))
        self.pairs = list(zip(breads, cw['boxes']))
        self.lid_of = dict(zip(cw['boxes'], cw['lids']))
        self.item = cw['item']
        self.watch = []                         # rolls become targets in turn; lids / the item are the human's
        self.box_wall = float(self.cfg.human_behavior['box_wall_m'])
        self.box_height = float(self.cfg.human_behavior['box_height_m'])

    def robot_target(self):
        """The roll the robot is working on: the first one not yet in its box."""
        for bread, box in self.pairs:
            if f'bread_in_{box}' not in self.protocol_done:
                return bread
        return self.pairs[-1][0]

    def place_dz(self, key: str) -> float:
        return self.box_wall if self.zones.get(key, {}).get('kind') == 'box' else 0.0

    def placed(self) -> bool:
        return all(f'bread_in_{box}' in self.protocol_done for _, box in self.pairs)

    def lid_closed(self, box: str) -> bool:
        lid = self.lid_of[box]
        p = self.pos(lid)
        top = self.table_z + self.box_height + object_half_height(self.scene['objects'][lid])
        return bool(self.human.holding != lid and np.linalg.norm(p[:2] - self.zone_pos(box)[:2]) <= 0.02
                    and abs(p[2] - top) <= 0.006)

    def commit_stage(self):
        return None

    def update_task(self):
        t = self.time
        self.ever_in_zone = self.ever_in_zone or bool(self.in_robot_zone(self.human.pos))
        self._update_positive(t)
        margin = float(self.cfg.success_conditions['in_box_margin_m'])
        for bread, box in self.pairs:
            step = f'bread_in_{box}'
            if self.robot_lifted == bread:
                self.protocol(f'bread_grasped_{bread}', 'robot', object=bread)
            if bread in self.ever_lifted and step not in self.protocol_done and self.resting(bread):
                if self.inside_zone(self.pos(bread), box, margin):
                    self.protocol(step, 'robot', object=bread, zone=box)
                else:
                    self.fail('bread_outside_box')
        if self.placed():
            self.protocol('object_placed', 'robot')     # InterruptTask's recovery bookkeeping
        if self.placed() and self.robot_at_rest():
            self.protocol('robot_at_rest', 'robot')

    def resting(self, key: str) -> bool:
        """Released, still, and not lifted (a roll in a box stands on its 4 mm bottom)."""
        sc = self.cfg.success_conditions
        return (not self.grasp_state[key] and self.speed(key) < sc.get('object_speed_max', 0.04)
                and self.pos(key)[2] < self.rest_height[key] + self.box_wall + 0.01)

    def success_predicate(self):
        ok_yield = self.yield_violations == 0 or not self.cfg.success_conditions.get('require_yield', True)
        return ('robot_at_rest' in self.protocol_done and self.placed() and self.robot_at_rest() and ok_yield
                and self.disruption_over and all(self.lid_closed(box) for _, box in self.pairs))

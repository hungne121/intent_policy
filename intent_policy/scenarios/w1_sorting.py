"""v7 W1 (Cowork, docs/PIF.md §8.6 / §8.11; HABIT S8186 adapted): mixed small items in a shared area; the human
puts the red items into their box one by one (random order, own pace), the robot puts the blue items into its box.
Each works on their own task; the robot must keep working without getting in the human's way: reorder its picks when
the human is about to reach near one, wait at a safe point when none is free (experts/w1_expert.py). Safety, fluency
and progress are the targets (PIF §8.6 Cowork), not a shared completion time.
"""
import numpy as np
from intent_policy.scenarios.base_scenario import BaseScenario
from intent_policy.scenarios.config import InfeasibleVariation, sample_variation
from intent_policy.sim.scene_builder import object_half_height
from intent_policy.sim.scripted_human import CoasterPlacer, Gesture


class SortingHuman(CoasterPlacer):
    """CoasterPlacer's take / carry / put, one item after another into the human's box."""

    def __init__(self, scene, behavior, variation):
        super().__init__(scene, behavior, variation)
        w = variation['w1']
        self.order, self.box = list(w['human_order']), w['human_box']
        self.pauses = list(w['human_pauses'])
        self.slots = w['drop_slots']
        self.k = 0
        self.coaster = self.order[0]
        self.selected_target = 'HBOX'

    def _box_xy(self):
        return np.asarray(self.box['xy'], float) + np.asarray(self.slots[self.k], float)   # the k-th item: k-th slot

    def placing(self):
        b, p = self.behavior, self.params
        item = self.coaster = self.order[self.k]
        on_item = lambda ctx: np.asarray(ctx['object_positions'][item], float) + [0.0, 0.0, self.grip_dz]
        # set down inside the box: the item's bottom 1 cm above the box floor (wall thickness), not dropped from the rim
        floor = lambda ctx: ctx['table_top_z'] + float(self.box.get('wall', 0.004)) + 0.01
        hold = lambda ctx: floor(ctx) + ctx['object_half_height'][item] + self.grip_dz
        # the palm aims so that the held item (not the palm) ends over the slot: the item sits off the palm by the grasp
        aim = lambda ctx: self._box_xy() - (np.asarray(ctx['object_positions'][item], float)[:2] - self.pos[:2])
        steps = [Gesture('sleep', 'between_items', duration=float(self.pauses[self.k])) if self.pauses[self.k] > 0 else None,
                 Gesture('move', 'reach_item', target=on_item, duration=p['take_reach_s'], shape='grasp', key=item),
                 self.G_call(self._take),
                 Gesture('move', 'lift_item', target=lambda ctx: self.pos + [0.0, 0.0, float(b['lift_m'])],
                         duration=p['lower_s'], shape='grasp'),
                 Gesture('move', 'carry_item', target=lambda ctx: np.r_[aim(ctx), ctx['table_top_z'] + float(self.box['height'])
                                                                      + 0.01 + 2 * ctx['object_half_height'][item] + self.grip_dz
                                                                      + float(b['carry_height_m'])],
                         duration=p['carry_s'], shape='grasp', key='HBOX'),
                 Gesture('move', 'lower_item', target=lambda ctx: np.r_[aim(ctx), hold(ctx)], duration=p['lower_s'],
                         shape='grasp', key='HBOX'),
                 self.G_call(self._put),
                 Gesture('move', 'withdraw_item', target=lambda ctx: self.pos + [0.0, 0.0, float(b['lift_m'])],
                         duration=p['lower_s'], shape='relaxed'),
                 self.G_call(self._next)]
        return [g for g in steps if g is not None]

    def _put(self, t, ctx):
        self.requests.append(('detach', self.coaster))
        self.holding = None
        self.emit('object_release', object=self.coaster, actor='human')
        self.emit('protocol_step_complete', step='human_dropped_' + self.coaster, object=self.coaster)

    def _next(self, t, ctx):
        self.k += 1
        if self.k == len(self.order):
            self.queue = [self.G_rest('withdraw'), self.G_call(lambda t, ctx: self.goto('done', t))]
            return
        self.queue = self.placing()


class SortingTask(BaseScenario):
    v6_plan = 'sorting'

    @property
    def expert_class(self):
        from intent_policy.experts.w1_expert import W1Expert
        return W1Expert

    def sample_episode_variation(self, seed, spec=None):
        var = sample_variation(self.cfg, seed, spec)
        cfg = self.cfg.scene_variation['w1']
        rng = np.random.default_rng([int(seed), 404])
        items = list(cfg['robot_items']) + list(cfg['human_items'])
        lo, hi = np.asarray(cfg['area_min_xy'], float), np.asarray(cfg['area_max_xy'], float)
        for restart in range(int(cfg['sampling_restarts'])):
            pos = []
            for key in items:
                reach = float(cfg['robot_reach_m']) if key in cfg['robot_items'] else np.inf   # robot items within R_eff
                for _ in range(int(cfg['sampling_attempts'])):
                    q = rng.uniform(lo, hi)
                    clear_of_boxes = all(np.any(np.abs(q - np.asarray(bx['xy'], float)) > np.asarray(bx['half'], float)
                                                + float(cfg['box_clearance_m'])) for bx in (cfg['robot_box'], cfg['human_box']))
                    if float(cfg['min_base_distance_m']) <= np.linalg.norm(q) <= reach and clear_of_boxes and all(
                            np.linalg.norm(q - r) >= float(cfg['min_spacing_m']) for r in pos):
                        pos.append(q)
                        break
            if len(pos) == len(items):
                break
        else:
            raise InfeasibleVariation('No item layout found')
        var['object_xy'] = {k: q.tolist() for k, q in zip(items, pos)}
        var['object_yaw'] = {k: float(rng.uniform(-1, 1) * float(cfg['yaw_rad'])) for k in items}
        var['on_table'] = items
        order = [cfg['human_items'][i] for i in rng.permutation(len(cfg['human_items']))]
        pauses = [0.0] + [float(rng.uniform(*cfg['human_pause_s'])) for _ in order[1:]]
        var['w1'] = dict(robot_items=list(cfg['robot_items']), human_items=list(cfg['human_items']),
                         human_order=order, human_pauses=pauses, robot_box=dict(cfg['robot_box']),
                         human_box=dict(cfg['human_box']), drop_slots=[list(q) for q in cfg['drop_slots']])
        var['target_object'] = cfg['robot_items'][0]
        var['target_zone'] = None
        return var

    def on_reset(self):
        w = self.variation['w1']
        self.robot_items, self.human_items = w['robot_items'], w['human_items']
        self.robot_box, self.human_box = w['robot_box'], w['human_box']
        self.robot_item = None                    # the item the robot works on now (set by the expert)
        self.realized_forecast_fault = None
        self.watch = []
        if self.human_items:                      # (W2 has no human items: its own human)
            old = self.human
            self.human = SortingHuman(self.scene, self.cfg.human_behavior, self.variation)
            for key in ('bank', 'bank_rng', 'bank_table_top', 'reach', 'point_pose', 'shoulder'):
                setattr(self.human, key, getattr(old, key))

    def robot_target(self):
        held = [k for k in self.robot_items if self.grasp_state.get(k)]
        return held[0] if held else (self.robot_item or self.robot_items[0])

    def in_box(self, key, box) -> bool:
        p = self.pos(key)
        return bool(np.all(np.abs(p[:2] - np.asarray(box['xy'], float)) <= np.asarray(box['half'], float))
                    and p[2] < self.table_z + float(box['height']) and self.speed(key) < 0.05
                    and not self.grasp_state.get(key) and self.human.holding != key)

    def update_task(self):
        for k in self.robot_items:
            if self.in_box(k, self.robot_box):
                self.protocol('robot_boxed_' + k, 'robot', object=k)
        for k in self.human_items:
            if self.in_box(k, self.human_box):
                self.protocol('human_boxed_' + k, 'human', object=k)
        for k in self.robot_items + self.human_items:
            if self.pos(k)[2] < self.table_z - 0.05:
                self.fail('item_dropped_off_table')
        if all('robot_boxed_' + k in self.protocol_done for k in self.robot_items) and self.robot_at_rest():
            self.protocol('robot_at_rest', 'robot')

    def success_predicate(self):
        return ('robot_at_rest' in self.protocol_done and self.human.stage == 'done'
                and all(self.in_box(k, self.robot_box) for k in self.robot_items)
                and all(self.in_box(k, self.human_box) for k in self.human_items))

    def ground_truth_intention_information(self):
        return dict(target_object=self.robot_target(), target_region=None)

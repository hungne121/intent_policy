"""v10 C2 (Collaborator, docs/PIF.md §8.12 "5 task bổ sung theo HABIT"; HABIT S6218 'Shelf cleaning' adapted): three
shelf sections (pads) side by side, an item on each. The human lifts the item of a section of their choice and holds
it up while the robot wipes that section, then puts it back; section after section. Waiting with the item in the
hand makes the wiping the human's critical path. The robot wipes with its closed gripper (no tool: prototype).
"""
import numpy as np
from intent_policy.scenarios.w1_sorting import SortingTask
from intent_policy.scenarios.config import sample_variation
from intent_policy.sim.scripted_human import CoasterPlacer, Gesture


class ShelfHuman(CoasterPlacer):
    """Take the item of the chosen section, hold it up beside the section until it is wiped, put it back."""

    def __init__(self, scene, behavior, variation):
        super().__init__(scene, behavior, variation)
        c = variation['c2']
        self.order, self.pauses, self.pad_of, self.home = list(c['order']), list(c['pauses']), c['pad_of'], c['item_home']
        self.k = 0
        self.coaster = self.order[0]
        self.selected_target = None

    def placing(self):
        b, p = self.behavior, self.params
        item = self.coaster = self.order[self.k]
        pad = self.pad_of[item]
        self.selected_target = pad
        on_item = lambda ctx: np.asarray(ctx['object_positions'][item], float) + [0.0, 0.0, self.grip_dz]
        hold = lambda ctx: np.r_[np.asarray(self.home[item][:2], float) + np.asarray(b['hold_offset'][:2], float),
                                 ctx['table_top_z'] + float(b['hold_offset'][2])]
        def back(ctx):
            off = np.asarray(ctx['object_positions'][item], float)[:2] - self.pos[:2]
            return np.r_[np.asarray(self.home[item][:2], float) - off,
                         ctx['table_top_z'] + float(b['pad_top_m']) + ctx['object_half_height'][item] + 0.002 + self.grip_dz]
        wiped = lambda t, ctx: 'wiped_' + pad in ctx['protocol_done'] and ctx['tcp_xy_distance'](self.home[item]) > float(b['clear_m'])
        start = lambda t, ctx: self.emit('human_cue_onset', modality='lift_item', selected_zone=pad, item=item)
        steps = [Gesture('sleep', 'between_items', duration=float(self.pauses[self.k])) if self.pauses[self.k] > 0 else None,
                 self.G_call(start),
                 Gesture('move', 'reach_item', target=on_item, duration=p['take_reach_s'], shape='grasp', key=item),
                 self.G_call(self._take),
                 Gesture('move', 'lift_item', target=hold, duration=p['carry_s'], shape='grasp'),
                 Gesture('wait', 'wait_wiped', until=wiped),
                 Gesture('move', 'put_back', target=back, duration=p['carry_s'], shape='grasp', key=pad),
                 self.G_call(self._put),
                 Gesture('move', 'withdraw_item', target=lambda ctx: self.rest, duration=p['rest_s'], shape='relaxed'),
                 self.G_call(self._next)]
        return [g for g in steps if g is not None]

    def _take(self, t, ctx):
        self.requests.append(('attach', self.coaster))
        self.holding = self.coaster
        self.emit('object_grasp', object=self.coaster, actor='human')
        self.emit('protocol_step_complete', step='lifted_' + self.coaster, object=self.coaster)

    def _put(self, t, ctx):
        self.requests.append(('detach', self.coaster))
        self.holding = None
        self.emit('object_release', object=self.coaster, actor='human')
        self.emit('protocol_step_complete', step='put_back_' + self.coaster, object=self.coaster)

    def _next(self, t, ctx):
        self.k += 1
        if self.k == len(self.order):
            self.goto('done', t)
            return
        self.queue = self.placing()


class ShelfWipeTask(SortingTask):

    @property
    def expert_class(self):
        from intent_policy.experts.c2_expert import C2Expert
        return C2Expert

    def sample_episode_variation(self, seed, spec=None):
        var = sample_variation(self.cfg, seed, spec)
        cfg = self.cfg.scene_variation['c2']
        rng = np.random.default_rng([int(seed), 505])
        pads = dict(cfg['pads'])
        items = list(cfg['items'])                     # item i stands on pad i
        pad_of = {k: p for k, p in zip(items, pads)}
        xy = {k: (np.asarray(pads[pad_of[k]], float) + rng.uniform(-1, 1, 2) * float(cfg['jitter_m'])).tolist()
              for k in items}
        var['object_xy'] = xy
        var['object_yaw'] = {k: float(rng.uniform(-1, 1) * float(cfg['yaw_rad'])) for k in items}
        var['on_table'] = list(xy)
        order = [items[i] for i in rng.permutation(len(items))]
        pauses = [0.0] + [float(rng.uniform(*cfg['pause_s'])) for _ in order[1:]]
        var['c2'] = dict(order=order, pauses=pauses, pad_of=pad_of, pads={k: list(v) for k, v in pads.items()},
                         item_home={k: list(v) for k, v in xy.items()})
        var['w1'] = dict(robot_items=[], human_items=[], human_order=[], human_pauses=[], robot_box=None, human_box=None,
                         drop_slots=[])
        var['target_object'] = order[0]
        var['target_zone'] = None
        return var

    def on_reset(self):
        super().on_reset()
        old = self.human
        self.human = ShelfHuman(self.scene, self.cfg.human_behavior, self.variation)
        for key in ('bank', 'bank_rng', 'bank_table_top', 'reach', 'point_pose', 'shoulder'):
            setattr(self.human, key, getattr(old, key))
        c = self.variation['c2']
        self.items, self.pad_of = c['order'], c['pad_of']
        self.pads = {k: np.asarray(v, float) for k, v in c['pads'].items()}
        self.wipe_len = {k: 0.0 for k in self.pads}
        self._last_tcp = None

    def update_task(self):
        cfg = self.cfg.scene_variation['c2']
        tcp = self.tcp()
        if self._last_tcp is not None:
            step = float(np.linalg.norm(tcp[:2] - self._last_tcp[:2]))
            for k, c in self.pads.items():
                held_away = all(self.pad_of[i] != k or self.human.holding == i or float(np.linalg.norm(
                    self.pos(i)[:2] - c)) > 2 * float(cfg['pad_half_m']) for i in self.items)   # its item taken off
                over = np.all(np.abs(tcp[:2] - c) <= float(cfg['pad_half_m'])) and tcp[2] < self.table_z + float(cfg['wipe_z_max_m'])
                if over and held_away and 'wiped_' + k not in self.protocol_done:
                    self.wipe_len[k] += step
                    if self.wipe_len[k] >= float(cfg['wipe_path_m']):
                        self.protocol('wiped_' + k, 'robot', zone=k)
        self._last_tcp = tcp
        for k in self.items:
            if 'put_back_' + k in self.protocol_done and self.speed(k) < 0.04 and float(np.linalg.norm(
                    self.pos(k)[:2] - self.pads[self.pad_of[k]])) < float(cfg['restore_m']):
                self.protocol('restored_' + k, 'human', object=k)
            if self.pos(k)[2] < self.table_z - 0.05:
                self.fail('item_dropped_off_table')
        if all('restored_' + k in self.protocol_done for k in self.items) and self.robot_at_rest():
            self.protocol('robot_at_rest', 'robot')

    def success_predicate(self):
        return ('robot_at_rest' in self.protocol_done and self.human.stage == 'done'
                and all('wiped_' + k in self.protocol_done for k in self.pads)
                and all('restored_' + k in self.protocol_done for k in self.items))

    def robot_target(self):
        return self.robot_item or self.items[0]

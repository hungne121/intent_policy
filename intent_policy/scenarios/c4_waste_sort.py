"""v10 C4 (Collaborator, docs/PIF.md §8.12 "5 task bổ sung theo HABIT"; HABIT S7025 'Waste Sorting' adapted): mixed
waste items (red = cans, blue = plastic) lie in a small pile in front of the human. The human takes one, holds it and
waits for the bin of its kind; the robot takes that bin (left: cans, right: plastic), holds it out at the presentation
point, the human drops the item in and withdraws; the robot puts the bin back. Item after item, in the human's order.

The kind of the item the hand is reaching for is the intention: the robot can fetch the right bin while the hand is
still on its way (early) or once the item is taken (observed). A wrong bin must be put back first (recovery cost).
"""
import numpy as np
from intent_policy.scenarios.w1_sorting import SortingTask
from intent_policy.scenarios.config import sample_variation
from intent_policy.sim.scripted_human import CoasterPlacer, Gesture


class WasteHuman(CoasterPlacer):
    """CoasterPlacer's take / lift, then hold the item until its bin is held out, drop it in, withdraw."""

    def __init__(self, scene, behavior, variation):
        super().__init__(scene, behavior, variation)
        c = variation['c4']
        self.order, self.pauses, self.kind, self.bins = list(c['order']), list(c['pauses']), c['kind'], c['bins']
        self.variation_c4 = c
        self.k = 0
        self.coaster = self.order[0]
        self.selected_target = None

    def _bin_ready(self, ctx):
        """The bin of the held item's kind is held out at the presentation point (the robot holds it there)."""
        b = self.bins[self.kind[self.coaster]]
        p = np.asarray(ctx['object_positions'][b], float)
        want = np.asarray(self.behavior['present_xy'], float)
        z0 = ctx['table_top_z'] + float(self.behavior['present_dz_m']) + ctx['object_half_height'][b]
        return (ctx['robot_grasping'].get(b, False) and float(np.linalg.norm(p[:2] - want)) < 0.03
                and abs(float(p[2] - z0)) < 0.03)

    def placing(self):
        b, p = self.behavior, self.params
        item = self.coaster = self.order[self.k]
        bin_ = self.bins[self.kind[item]]
        self.selected_target = bin_
        on_item = lambda ctx: np.asarray(ctx['object_positions'][item], float) + [0.0, 0.0, self.grip_dz]
        hold = np.asarray(b['hold_xyz'], float)
        # the held item (not the palm: it sits off the palm by the grasp) over the bin, drop_dx_m towards +x (the gripper
        # holds the -x rim), its bottom drop_dz above the rim
        def over(ctx):
            off = np.asarray(ctx['object_positions'][item], float)[:2] - self.pos[:2]
            q = np.asarray(ctx['object_positions'][bin_], float)
            return np.r_[q[:2] + [float(b['drop_dx_m']), 0.0] - off,
                         q[2] + ctx['object_half_height'][bin_] + float(b['drop_dz_m']) + ctx['object_half_height'][item]
                         + self.grip_dz]
        start = lambda t, ctx: self.emit('human_cue_onset', modality='take_item', selected_zone=bin_, item=item)
        c = self.variation_c4
        sw = c.get('switch') if c.get('switch') and c['switch']['round'] == self.k else None
        reach = 'reach_item_f' if c.get('fault_round') == self.k else 'reach_item'   # C3: the forecast turned here
        pre = []
        if sw is not None:
            d = sw['decoy']
            on_d = lambda ctx: np.asarray(ctx['object_positions'][d], float) + [0.0, 0.0, self.grip_dz]
            change = lambda t, ctx: self.emit('human_intention_change', kind='waste_item', previous=d, current=item,
                                              timing=sw['kind'])
            if sw['kind'] == 'before':
                # C4 early: towards the decoy (other kind), turned to the real item part of the way
                f = float(sw['progress'])
                part = lambda ctx: self.pos + f * (on_d(ctx) - self.pos)
                pre = [Gesture('move', 'reach_decoy', target=part, duration=f * p['take_reach_s'], shape='grasp', key=d),
                       self.G_call(change)]
            else:
                # C4 late: the decoy taken and held up, then put back on its place, then the real item
                home = np.asarray(c['home'][d], float)
                def back(ctx):
                    off = np.asarray(ctx['object_positions'][d], float)[:2] - self.pos[:2]
                    return np.r_[home[:2] - off, ctx['table_top_z'] + ctx['object_half_height'][d] + 0.002 + self.grip_dz]
                pre = [Gesture('move', 'reach_decoy', target=on_d, duration=p['take_reach_s'], shape='grasp', key=d),
                       self.G_call(lambda t, ctx, d=d: self._take_key(d)),
                       Gesture('move', 'lift_decoy', target=lambda ctx: hold + [0.0, 0.0, ctx['table_top_z']],
                               duration=p['carry_s'], shape='grasp'),
                       Gesture('sleep', 'decoy_hold', duration=float(sw['delay_s'])),
                       self.G_call(change),
                       Gesture('move', 'return_decoy', target=back, duration=p['carry_s'], shape='grasp', key=d),
                       self.G_call(lambda t, ctx, d=d: self._return_key(d))]
        steps = [Gesture('sleep', 'between_items', duration=float(self.pauses[self.k])) if self.pauses[self.k] > 0 else None,
                 self.G_call(start), *pre,
                 Gesture('move', reach, target=on_item, duration=p['take_reach_s'], shape='grasp', key=item),
                 self.G_call(self._take),
                 Gesture('move', 'lift_item', target=lambda ctx: hold + [0.0, 0.0, ctx['table_top_z']],
                         duration=p['carry_s'], shape='grasp'),
                 Gesture('wait', 'wait_bin', until=lambda t, ctx: self._bin_ready(ctx)),
                 Gesture('move', 'to_bin', target=over, duration=p['lower_s'], shape='grasp', key=bin_),
                 self.G_call(self._put),
                 Gesture('move', 'withdraw_item', target=lambda ctx: self.rest, duration=p['rest_s'], shape='relaxed'),
                 self.G_call(self._next)]
        return [g for g in steps if g is not None]

    def _take_key(self, key):
        self.requests.append(('attach', key))
        self.holding = key
        self.emit('object_grasp', object=key, actor='human')

    def _return_key(self, key):
        self.requests.append(('detach', key))
        self.holding = None
        self.emit('object_release', object=key, actor='human')
        self.emit('protocol_step_complete', step='returned_' + key, object=key)

    def _take(self, t, ctx):
        self.requests.append(('attach', self.coaster))
        self.holding = self.coaster
        self.emit('object_grasp', object=self.coaster, actor='human')
        self.emit('protocol_step_complete', step='taken_' + self.coaster, object=self.coaster)

    def _put(self, t, ctx):
        self.requests.append(('detach', self.coaster))
        self.holding = None
        self.emit('object_release', object=self.coaster, actor='human')
        self.emit('protocol_step_complete', step='dropped_' + self.coaster, object=self.coaster)

    def _next(self, t, ctx):
        self.k += 1
        if self.k == len(self.order):
            self.goto('done', t)
            return
        self.queue = self.placing()


class WasteSortTask(SortingTask):

    @property
    def expert_class(self):
        from intent_policy.experts.c4_expert import C4Expert
        return C4Expert

    def sample_episode_variation(self, seed, spec=None):
        var = sample_variation(self.cfg, seed, spec)
        cfg = self.cfg.scene_variation['c4']
        rng = np.random.default_rng([int(seed), 707])
        items = list(cfg['items'])
        slots = [np.asarray(cfg['pile_xy'], float) + np.asarray(o, float) for o in cfg['pile_offsets']]
        place = rng.permutation(len(slots))[:len(items)]           # the kinds mixed over the pile
        xy = {k: (slots[j] + rng.uniform(-1, 1, 2) * float(cfg['jitter_m'])).tolist() for k, j in zip(items, place)}
        for b, q in cfg['bins'].items():
            xy[cfg['bin_objects'][b]] = list(q)
        var['object_xy'] = xy
        var['object_yaw'] = {k: (float(rng.uniform(-1, 1) * float(cfg['yaw_rad'])) if k in items else 0.0) for k in xy}
        var['on_table'] = list(xy)
        order = [items[i] for i in rng.permutation(len(items))]
        pauses = [0.0] + [float(rng.uniform(*cfg['pause_s'])) for _ in order[1:]]
        var['c4'] = dict(order=order, pauses=pauses, kind=dict(cfg['kind']), bins=dict(cfg['bin_objects']),
                         bin_home={cfg['bin_objects'][b]: list(q) for b, q in cfg['bins'].items()})
        var['c4']['home'] = {k: list(xy[k]) for k in items}
        # recovery cases (PIF §8.12 "Hướng tối ưu expert ManiCast"): one round by seed whose item has a decoy of the other
        # kind still on the pile (taken later); design values, no source (as I1 / C3)
        rounds = [r for r in range(len(order)) if any(cfg['kind'][d] != cfg['kind'][order[r]] for d in order[r + 1:])]
        sw = cfg.get('switch')
        if sw and rounds:
            r = int(rng.choice(rounds))
            dec = [d for d in order[r + 1:] if cfg['kind'][d] != cfg['kind'][order[r]]]
            var['c4']['switch'] = dict(kind=sw['kind'], round=r, decoy=str(rng.choice(dec)))
            if sw['kind'] == 'before':
                var['c4']['switch']['progress'] = float(rng.uniform(*sw['progress']))
            else:
                var['c4']['switch']['delay_s'] = float(rng.uniform(*sw['delay_s']))
        if cfg.get('forecast_fault_round'):
            var['c4']['fault_round'] = int(rng.integers(len(order)))
        # SortingTask's bookkeeping: no robot items of its own, no human box
        var['w1'] = dict(robot_items=[], human_items=[], human_order=[], human_pauses=[], robot_box=None, human_box=None,
                         drop_slots=[])
        var['target_object'] = order[0]
        var['target_zone'] = None
        return var

    def on_reset(self):
        super().on_reset()
        old = self.human
        self.human = WasteHuman(self.scene, self.cfg.human_behavior, self.variation)
        for key in ('bank', 'bank_rng', 'bank_table_top', 'reach', 'point_pose', 'shoulder'):
            setattr(self.human, key, getattr(old, key))
        c = self.variation['c4']
        self.items, self.kind, self.bins = c['order'], c['kind'], c['bins']
        self.bin_home = {k: np.asarray(v, float) for k, v in c['bin_home'].items()}

    def in_bin(self, item, bin_) -> bool:
        p, b = self.pos(item), self.pos(bin_)
        r, h = self.scene['objects'][bin_]['size']
        return bool(float(np.linalg.norm(p[:2] - b[:2])) < r - 0.005 and b[2] - h / 2 < p[2] < b[2] + h / 2   # not under a bin carried over it
                    and self.human.holding != item and self.speed(item) < 0.05)

    def bin_home_ok(self, bin_) -> bool:
        return (float(np.linalg.norm(self.pos(bin_)[:2] - self.bin_home[bin_][:2])) < 0.03
                and not self.grasp_state.get(bin_) and self.speed(bin_) < 0.04)

    def update_task(self):
        for k in self.items:
            right = self.bins[self.kind[k]]
            if self.in_bin(k, right):
                self.protocol('binned_' + k, 'human', object=k, bin=right)
            for b in self.bins.values():
                if b != right and self.in_bin(k, b):
                    self.fail('item_in_wrong_bin')
            if self.pos(k)[2] < self.table_z - 0.05:
                self.fail('item_dropped_off_table')
        if (all('binned_' + k in self.protocol_done for k in self.items)
                and all(self.bin_home_ok(b) for b in self.bins.values()) and self.robot_at_rest()):
            self.protocol('robot_at_rest', 'robot')

    def success_predicate(self):
        return ('robot_at_rest' in self.protocol_done and self.human.stage == 'done'
                and all(self.in_bin(k, self.bins[self.kind[k]]) for k in self.items)
                and all(self.bin_home_ok(b) for b in self.bins.values()))

    def robot_target(self):
        held = [k for k in self.bins.values() if self.grasp_state.get(k)]
        return held[0] if held else (self.robot_item or list(self.bins.values())[0])

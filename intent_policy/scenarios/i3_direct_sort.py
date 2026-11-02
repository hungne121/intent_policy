"""v10 I3 (Instructor, docs/PIF.md §8.12 "I3 mới: phân loại đồ theo hướng tay"; HABIT S5334 'Clothes Organizing',
Supervisor role, adapted): folded items lie in a pile in front of the robot, a basket on each side (the human's left /
right). The robot lifts one item and holds it up; the human directs it with the right arm, swept to the left or to the
right; the robot puts it into the basket of that side. Item after item.

The side of the sweep is the intention: the robot can set off while the arm is still on its way (early) or once the
sweep is far enough to be seen (observed). Sometimes the hand first goes to the chest (adjusting clothes, as I2), so
a moving arm alone is not yet a direction. Recovery cases: the sweep turns to the other side part of the way (before)
or the arm sweeps again to the other side after a delay on the human's clock (after: the robot is on its way).
"""
import numpy as np
from intent_policy.scenarios.i2_side_request import SideRequester, SideRequestTask
from intent_policy.scenarios.config import sample_variation
from intent_policy.sim.scripted_human import Gesture


class DirectingHuman(SideRequester):
    """The right arm only: wait until the robot holds an item up, then sweep towards the side's basket."""

    def __init__(self, scene, behavior, variation):
        super().__init__(scene, behavior, variation)
        self.v3 = variation['i3']
        self.reqs = self.v3['rounds']

    def _lifted(self, t, ctx):
        """The robot holds an item up (grasped, its bottom clear of the pile)."""
        for k in self.v3['items']:
            if ctx['robot_grasping'].get(k, False) and 'sorted_' + k not in ctx['protocol_done']:   # (not one taken out
                                                                                                     #  again to re-sort)
                p = np.asarray(ctx['object_positions'][k], float)
                if p[2] > ctx['table_top_z'] + float(self.behavior['lifted_dz_m']):
                    return True
        return False

    def round(self):
        b, r = self.behavior, self.reqs[self.k]
        s = r['side']
        sw = self.v3.get('switch') if self.v3.get('switch') and self.v3['switch']['round'] == self.k else None
        k_done = self.k + 1
        g = [Gesture('wait', 'wait_lifted', until=self._lifted),
             Gesture('sleep', 'react', duration=r['pause_s'])]
        if r.get('distractor'):
            g += [Gesture('move', 'adjust', target=lambda ctx: self._chest('r'), duration=0.8, shape='relaxed'),
                  Gesture('sleep', 'adjusting', duration=float(r['adjust_s'])),
                  Gesture('move', 'adjust_back', target=lambda ctx: self.rest, duration=0.8, shape='relaxed')]
        first = sw['decoy'] if sw is not None else s
        start = lambda t, ctx: (setattr(self, 'selected_target', 'side_' + first),
                                self.emit('human_cue_onset', modality='arm_sweep', selected_zone='side_' + first, round=self.k))
        change = lambda t, ctx: (setattr(self, 'selected_target', 'side_' + s),
                                 self.emit('human_intention_change', kind='sweep_side', previous='side_' + first,
                                           current='side_' + s, timing=sw['kind']))
        g.append(self.G_call(start))
        out = float(r['out_s'])
        if sw is not None and sw['kind'] == 'before':
            # early change: towards the decoy side, turned to the real side part of the way
            f = float(sw['progress'])
            part = lambda ctx: self.rest + f * (self._signal(first) - self.rest)
            g += [Gesture('move', 'direct_decoy', target=part, duration=f * out, shape='relaxed', key=lambda: 'side_' + first),
                  self.G_call(change),
                  Gesture('move', 'direct_out', target=lambda ctx: self._signal(s), duration=out, shape='relaxed',
                          key=lambda: 'side_' + s)]
        elif sw is not None:
            # late change: the full sweep to the decoy side, held, then swept again to the real side (human's clock)
            g += [Gesture('move', 'direct_out', target=lambda ctx: self._signal(first), duration=out, shape='relaxed',
                          key=lambda: 'side_' + first),
                  Gesture('sleep', 'decoy_hold', duration=float(sw['delay_s'])),
                  self.G_call(change),
                  Gesture('move', 'redirect', target=lambda ctx: self._signal(s), duration=out, shape='relaxed',
                          key=lambda: 'side_' + s)]
        else:
            g.append(Gesture('move', 'direct_out', target=lambda ctx: self._signal(s), duration=out, shape='relaxed',
                             key=lambda: 'side_' + s))
        g += [self.G_protocol('direct_' + s),
              Gesture('sleep', 'signal_hold', duration=float(r['signal_hold_s'])),
              Gesture('move', 'direct_back', target=lambda ctx: self.rest, duration=out, shape='relaxed'),
              Gesture('wait', 'wait_sorted', until=lambda t, ctx: sum(
                  1 for q in self.v3['items'] if 'sorted_' + q in ctx['protocol_done']) >= k_done),
              self.G_call(self._next)]
        return g


class DirectSortTask(SideRequestTask):
    """Robot side: each lifted item into the basket of the side swept to; human side: the right arm's directions."""

    @property
    def expert_class(self):
        from intent_policy.experts.i3_expert import I3Expert
        return I3Expert

    def sample_episode_variation(self, seed, spec=None):
        var = sample_variation(self.cfg, seed, spec)
        cfg = self.cfg.scene_variation['i3']
        rng = np.random.default_rng([int(seed), 808])
        items = list(cfg['items'])
        slots = [np.asarray(cfg['pile_xy'], float) + np.asarray(o, float) for o in cfg['pile_offsets']]
        place = rng.permutation(len(slots))[:len(items)]
        xy = {k: (slots[j] + rng.uniform(-1, 1, 2) * float(cfg['jitter_m'])).tolist() for k, j in zip(items, place)}
        var['object_xy'] = xy
        var['object_yaw'] = {k: float(rng.uniform(-1, 1) * float(cfg['yaw_rad'])) for k in xy}
        var['on_table'] = list(xy)
        rounds = []
        for i in range(len(items)):                # the side of each round: the item's kind (tops / bottoms), random
            r = dict(side='l' if rng.random() < 0.5 else 'r', pause_s=float(rng.uniform(*cfg['react_s'])),
                     out_s=float(rng.uniform(*cfg['out_s'])), signal_hold_s=float(rng.uniform(*cfg['signal_hold_s'])))
            if rng.random() < float(cfg['distractor_p']):
                r['distractor'] = 'r'
                r['adjust_s'] = float(rng.uniform(*cfg['adjust_s']))
            rounds.append(r)
        var['i3'] = dict(items=items, rounds=rounds, baskets={s: dict(b) for s, b in cfg['baskets'].items()})
        # recovery cases: one round by seed (design values, no source; as C3 / C4)
        sw = cfg.get('switch')
        if sw:
            r = int(rng.integers(len(items)))
            var['i3']['switch'] = dict(kind=sw['kind'], round=r, decoy='r' if rounds[r]['side'] == 'l' else 'l')
            if sw['kind'] == 'before':
                var['i3']['switch']['progress'] = float(rng.uniform(*sw['progress']))
            else:
                var['i3']['switch']['delay_s'] = float(rng.uniform(*sw['delay_s']))
        ff = cfg.get('forecast_fault')
        if ff:
            # the forecast arm turned about the shoulder towards the OTHER side's sweep point, by the angle between the
            # two sweep points seen from the shoulder x U(factor) (as C3)
            r = int(rng.integers(len(items)))
            sh = np.asarray(self.scene['human']['stand_xy'], float)
            sig = self.cfg.human_behavior['signal_xyz']
            ang = lambda s: np.arctan2(*(np.asarray(sig[s][:2], float) - sh)[::-1])
            s, o = rounds[r]['side'], 'r' if rounds[r]['side'] == 'l' else 'l'
            yaw = np.degrees(ang(o) - ang(s)) * float(rng.uniform(*ff['factor']))
            var['i3']['forecast_fault'] = dict(round=r, yaw_deg=float(yaw), extra_s=float(ff['extra_s']),
                                               ramp_s=float(ff['ramp_s']))
        var['i2'] = dict(requests=rounds, items={'l': [], 'r': []})     # SideRequester / SideRequestTask bookkeeping
        var['w1'] = dict(robot_items=items, human_items=[], human_order=[], human_pauses=[], robot_box=None,
                         human_box=None, drop_slots=[list(q) for q in cfg['drop_slots']])
        var['target_object'] = items[0]
        var['target_zone'] = None
        return var

    def on_reset(self):
        super().on_reset()
        old = self.human
        self.human = DirectingHuman(self.scene, self.cfg.human_behavior, self.variation)
        for key in ('bank', 'bank_rng', 'bank_table_top', 'reach', 'point_pose', 'shoulder', 'rest_by_side'):
            setattr(self.human, key, getattr(old, key))
        self.human.rest = old.rest_by_side['r'].copy()
        self.human.pos = self.human.rest.copy()
        v = self.variation['i3']
        self.items, self.baskets = v['items'], v['baskets']
        self.landed = []                          # the items in the order they first landed in a basket (= the rounds)

    def basket_of(self, key):
        for s, b in self.baskets.items():
            if self.in_box(key, b):
                return s
        return None

    def update_task(self):
        for k in self.items:
            s = self.basket_of(k)
            if s is not None and k not in self.landed:  # a wrong basket is not a failure yet: it can be re-sorted
                self.landed.append(k)
                self.protocol('sorted_' + k, 'robot', object=k, side=s)
            if self.pos(k)[2] < self.table_z - 0.05:
                self.fail('item_dropped_off_table')
        if len(self.landed) == len(self.items) and self.robot_at_rest():
            self.protocol('robot_at_rest', 'robot')

    def success_predicate(self):
        want = [r['side'] for r in self.variation['i3']['rounds']]
        return ('robot_at_rest' in self.protocol_done and self.human.stage == 'done'
                and [self.basket_of(k) for k in self.landed] == want)

    def robot_target(self):
        held = [k for k in self.items if self.grasp_state.get(k)]
        return held[0] if held else (self.robot_item or self.items[0])

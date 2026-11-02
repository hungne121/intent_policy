"""v7 I2 (Instructor, docs/PIF.md §8.6 / §8.11): two clusters of the same items, one on each side of the human. To
ask for an item the human holds the hand of that side out (left hand: an item from the left cluster); the robot
takes one from that side and puts it into the tray. Sometimes a hand first moves for something else (to the chest:
adjusting clothes), so a moving hand alone is not yet a request; the arm reaching out is (experts/i2_expert.py).
Both arms hang relaxed at rest, so the moving arm can change without a jump (human_body.set_active).
"""
import numpy as np
from intent_policy.scenarios.w1_sorting import SortingTask
from intent_policy.sim.scripted_human import GestureHuman, Gesture
from intent_policy.sim.motion_bank import MotionBank


class SideRequester(GestureHuman):
    def __init__(self, scene, behavior, variation):
        super().__init__(scene, behavior, variation)
        self.reqs = variation['i2']['requests']
        self.k = 0
        self.rest_by_side = None                 # set by the task after reset (relaxed hanging palms)
        self.side = 'r'

    def _switch(self, side):
        def fn(t, ctx):
            if side != self.side:
                self.requests.append(('switch_arm', side))
                self.side = side
        return [self.G_call(fn), Gesture('sleep', 'switch_arm', duration=0.05)]

    def _signal(self, side):
        return np.asarray(self.behavior['signal_xyz'][side], float)

    def _chest(self, side):
        return np.asarray(self.behavior['chest_xyz'][side], float)

    def _served(self, side):
        def until(t, ctx):
            items = self.behavior['items'][side]
            return any(ctx['robot_grasping'].get(k, False) for k in items)
        return until

    def round(self):
        b, r = self.behavior, self.reqs[self.k]
        s = r['side']
        g = []
        if r['pause_s'] > 0:
            g.append(Gesture('sleep', 'pause', duration=r['pause_s']))
        if r.get('distractor'):
            d = r['distractor']
            g += self._switch(d) + [Gesture('move', 'adjust', target=lambda ctx, d=d: self._chest(d), duration=0.8,
                                            shape='relaxed'),
                                    Gesture('sleep', 'adjusting', duration=float(r['adjust_s'])),
                                    Gesture('move', 'adjust_back', target=lambda ctx, d=d: self.rest_by_side[d], duration=0.8,
                                            shape='relaxed')]
        start = lambda t, ctx: (setattr(self, 'selected_target', 'side_' + s),
                                self.emit('human_cue_onset', modality='arm_out', selected_zone='side_' + s, round=self.k))
        # v8: lower the signal on a human clock; acknowledgement only gates the NEXT request.
        seen = set()
        original_start = start
        def start(t, ctx):
            seen.update(ctx['protocol_done'])
            return original_start(t, ctx)
        wait_served = Gesture('wait', 'wait_served', until=lambda t, ctx: any(
            'served_' + k in ctx['protocol_done'] and 'served_' + k not in seen for k in b['items'][s]))
        timed = 'signal_hold_s' in r
        g += self._switch(s) + [self.G_call(start),
              Gesture('move', 'request_out', target=lambda ctx: self._signal(s), duration=float(r['out_s']), shape='offer',
                      palm_up=0.3, key=lambda: 'side_' + s),
              self.G_protocol('request_' + s),
              (Gesture('sleep', 'signal_hold', duration=float(r['signal_hold_s'])) if timed
               else Gesture('wait', 'wait_served', until=self._served(s))),
              Gesture('move', 'request_back', target=lambda ctx: self.rest_by_side[s], duration=float(r['out_s']),
                      shape='relaxed'),
              *([wait_served] if timed else []), self.G_call(self._next)]
        return g

    def _next(self, t, ctx):
        self.k += 1
        if self.k == len(self.reqs):
            self.goto('done', t)
            return
        self.queue = self.round()

    def behave(self, t, ctx):
        if not self.started and t >= self.params['cue_onset_s']:
            self.started = True
            self.queue = self.round()
        self.run_queue(t, ctx)


class SideRequestTask(SortingTask):
    """Robot side: items of the requested side into the tray (its box); human side: requests with either arm."""

    @property
    def expert_class(self):
        from intent_policy.experts.i2_expert import I2Expert
        return I2Expert

    def sample_episode_variation(self, seed, spec=None):
        var = super().sample_episode_variation(seed, spec)       # (W1 layout code: overwritten below)
        cfg = self.cfg.scene_variation['i2']
        rng = np.random.default_rng([int(seed), 606])
        xy = {}
        for side, centre in cfg['clusters'].items():
            for j, k in enumerate(cfg['items'][side]):
                xy[k] = (np.asarray(centre, float) + np.asarray(cfg['cluster_offsets'][j], float)
                         + rng.uniform(-1, 1, 2) * float(cfg['jitter_m'])).tolist()
        var['object_xy'] = xy
        var['object_yaw'] = {k: 0.0 for k in xy}
        var['on_table'] = list(xy)
        reqs = []
        left = {s: len(v) for s, v in cfg['items'].items()}      # a side is not asked for more items than it has
        for i in range(int(cfg['requests'])):
            side = 'l' if rng.random() < 0.5 else 'r'
            if left[side] == 0:
                side = 'r' if side == 'l' else 'l'
            left[side] -= 1
            r = dict(side=side, pause_s=0.0 if i == 0 else float(rng.uniform(*cfg['pause_s'])),
                     out_s=float(rng.uniform(*cfg['out_s'])))
            if 'signal_hold_s' in cfg:
                r['signal_hold_s'] = float(rng.uniform(*cfg['signal_hold_s']))
            if rng.random() < float(cfg['distractor_p']):
                r['distractor'] = 'l' if rng.random() < 0.5 else 'r'
                r['adjust_s'] = float(rng.uniform(*cfg['adjust_s']))
            reqs.append(r)
        var['i2'] = dict(requests=reqs, items={s: list(v) for s, v in cfg['items'].items()})
        var['w1']['robot_items'] = [k for v in cfg['items'].values() for k in v]
        var['w1']['human_items'] = []
        return var

    def on_reset(self):
        super().on_reset()
        bm = self.body_model
        bm.set_active('r')
        rest = {}
        for side, other in (('r', 'l'), ('l', 'r')):    # the relaxed hanging palm of each side (idle pose)
            bm.set_active(other)
            rest[side] = np.asarray(bm.pose(self.human.rest)['idle_palm'], float)
        bm.set_active('r')
        old = self.human
        self.human = SideRequester(self.scene, self.cfg.human_behavior, self.variation)
        for key in ('bank', 'bank_rng', 'bank_table_top', 'reach', 'point_pose', 'shoulder'):
            setattr(self.human, key, getattr(old, key))
        self.human.rest_by_side = rest
        self.human.rest = rest['r'].copy()
        self.human.pos = rest['r'].copy()
        mb = self.cfg.human_behavior['motion_bank']
        self.banks = {'r': self.human.bank}
        if self.human.bank is not None:            # the left palm's paths of the same clips, loaded once
            if getattr(self, '_bank_left', None) is None:
                J = bm.J
                height = J['Head'][2] - min(J[k][2] for k in ('L_Ankle', 'R_Ankle', 'L_Foot', 'R_Foot'))
                self._bank_left = MotionBank(mb['folder'], J['Pelvis'], float(self.scene['human'].get('facing_deg', 180.0)),
                                             float(self.scene.get('floor_z', 0.0)), float(height), hand='left',
                                             split_manifest=mb.get('split_manifest'), split=mb.get('split'))
            self.banks['l'] = self._bank_left
        self.served = []

    def _switch_arm(self, side):
        self.body_model.set_active(side)
        self.human.pos = self.human.rest_by_side[side].copy()
        self.human.rest = self.human.rest_by_side[side].copy()
        if side in self.banks:
            self.human.bank = self.banks[side]

    def update_task(self):
        for k in self.robot_items:
            if self.in_box(k, self.robot_box) and k not in self.served:
                self.served.append(k)
                self.protocol('served_' + k, 'robot', object=k)
        if len(self.served) == len(self.variation['i2']['requests']) and self.robot_at_rest():
            self.protocol('robot_at_rest', 'robot')

    def success_predicate(self):
        want = [r['side'] for r in self.variation['i2']['requests']]
        items = self.variation['i2']['items']
        got = ['l' if k in items['l'] else 'r' for k in self.served]
        return ('robot_at_rest' in self.protocol_done and self.human.stage == 'done'
                and sorted(got) == sorted(want) and all(self.in_box(k, self.robot_box) for k in self.served))

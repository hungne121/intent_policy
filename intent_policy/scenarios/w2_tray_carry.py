"""v7 W2 (Cowork, docs/PIF.md §8.6 "W2 thay thế"): the human carries a long tray with both hands across the shared area
(to the other side, a pause, back); the robot independently puts its cubes into its box (the W1 robot task) and must
yield the swept area of the whole tray, not only of the wrists. The image gives the tray's size / pose and that it is
held; the forecast gives the hands' short-term motion (experts/w1_expert.py with the tray as the human's footprint).
"""
import numpy as np
from intent_policy.scenarios.w1_sorting import SortingTask
from intent_policy.sim.scripted_human import CoasterPlacer, Gesture


class TrayCarrier(CoasterPlacer):
    uses_bimanual = True

    def __init__(self, scene, behavior, variation):
        super().__init__(scene, behavior, variation)
        w = variation['w2']
        self.coaster = w['tray']
        self.spots, self.pauses = [np.asarray(q, float) for q in w['spots']], list(w['pauses'])
        self.k = 0
        self.extras['bimanual_provenance'] = dict(primary_hand='right',
            primary_motion='retargeted_hymotion_or_logged_fallback',
            secondary_hand='rigid_offset_from_primary', tray_orientation='horizontal_constraint',
            raw_two_hand_replay=False)
        self.selected_target = 'SPOT1'

    def placing(self):
        b, p, tray = self.behavior, self.params, self.coaster
        hold = lambda ctx: ctx['table_top_z'] + ctx['object_half_height'][tray] + 0.001 + self.grip_dz
        # the right hand holds the tray's right end: its palm is `grip_offset_xy` from the tray centre
        end = np.asarray(b['grip_offset_xy'], float)
        on_tray = lambda ctx: np.r_[np.asarray(ctx['object_positions'][tray], float)[:2] + end, hold(ctx)]
        dest = self.spots[self.k + 1]
        moves = [Gesture('sleep', 'pause', duration=float(self.pauses[self.k])) if self.pauses[self.k] > 0 else None,
                 Gesture('move', 'reach_tray', target=on_tray, duration=p['take_reach_s'], shape='grasp', key=tray),
                 self.G_call(self._grab),
                 Gesture('move', 'lift_tray', target=lambda ctx: self.pos + [0.0, 0.0, float(b['carry_height_m'])],
                         duration=p['lower_s'], shape='grasp'),
                 Gesture('move', 'carry_tray', target=lambda ctx: np.r_[dest + end, hold(ctx) + float(b['carry_height_m'])],
                         duration=p['carry_s'], shape='grasp', key='SPOT%d' % (self.k + 2)),
                 Gesture('move', 'lower_tray', target=lambda ctx: np.r_[dest + end, hold(ctx)], duration=p['lower_s'],
                         shape='grasp'),
                 self.G_call(self._drop),
                 Gesture('move', 'withdraw', target=lambda ctx: self.rest, duration=p['take_reach_s'], shape='relaxed'),
                 self.G_call(self._next)]
        return [g for g in moves if g is not None]

    def _grab(self, t, ctx):
        self._take(t, ctx)
        self.bimanual = dict(width_m=float(self.behavior['hands_apart_m']), shape='grasp')
        self.requests.append(('two_hand', self.coaster))      # the scenario records the pose in the two-hand frame

    def _drop(self, t, ctx):
        self._put(t, ctx)
        self.bimanual = None
        self.requests.append(('one_hand', self.coaster))
        self.emit('protocol_step_complete', step='tray_carried_%d' % (self.k + 1))

    def _next(self, t, ctx):
        self.k += 1
        if self.k + 1 >= len(self.spots):
            self.goto('done', t)
            return
        self.queue = self.placing()


class TrayCarryTask(SortingTask):
    """The robot side is W1's (its cubes into its box); the human carries the tray between the spots."""

    def sample_episode_variation(self, seed, spec=None):
        var = super().sample_episode_variation(seed, spec)
        cfg = self.cfg.scene_variation['w2']
        rng = np.random.default_rng([int(seed), 505])
        tray = cfg['tray']
        a, b = np.asarray(cfg['spot_left'], float), np.asarray(cfg['spot_right'], float)
        first = [a, b] if rng.random() < 0.5 else [b, a]
        spots = [first[i % 2] + rng.uniform(-1, 1, 2) * float(cfg['spot_jitter_m']) for i in range(int(cfg['carries']) + 1)]
        var['object_xy'][tray] = spots[0].tolist()
        var['object_yaw'][tray] = 0.0
        var['on_table'] = list(var['on_table']) + [tray]
        var['w1']['human_items'] = []
        var['w2'] = dict(tray=tray, spots=[q.tolist() for q in spots],
                         pauses=[float(rng.uniform(*cfg['pause_s'])) for _ in range(int(cfg['carries']))])
        return var

    def on_reset(self):
        super().on_reset()
        self.tray = self.variation['w2']['tray']
        old = self.human
        self.human = TrayCarrier(self.scene, self.cfg.human_behavior, self.variation)
        for key in ('bank', 'bank_rng', 'bank_table_top', 'reach', 'point_pose', 'shoulder'):
            setattr(self.human, key, getattr(old, key))

    def update_task(self):
        super().update_task()

    def success_predicate(self):
        return ('robot_at_rest' in self.protocol_done and self.human.stage == 'done'
                and all(self.in_box(k, self.robot_box) for k in self.robot_items))

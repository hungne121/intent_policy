"""v10 W3 (Cowork, docs/PIF.md §8.12 "5 task bổ sung theo HABIT"; HABIT S7236 'Lint Rolling' Coworker adapted): two
trays stand side by side in the shared area. The human takes them away one at a time, in an order of their own,
with both hands, to their side of the table, and works on each there (wiping the dishes) before the next. The robot
cleans the place a tray stood on once the tray and the hands have left it, without getting in the human's way.
Nobody waits for the other by design: safety, fluency and progress are the targets (PIF §8.6 Cowork).
"""
import numpy as np
from intent_policy.scenarios.c2_shelf_wipe import ShelfWipeTask
from intent_policy.scenarios.config import sample_variation
from intent_policy.sim.scripted_human import CoasterPlacer, Gesture


class TrayClearer(CoasterPlacer):
    uses_bimanual = True

    def __init__(self, scene, behavior, variation):
        super().__init__(scene, behavior, variation)
        w = variation['w3']
        self.order, self.dest, self.work = list(w['order']), w['dest'], list(w['work_s'])
        self.pauses = list(w['pauses'])
        self.k = 0
        self.coaster = self.order[0]
        self.selected_target = None
        self.extras['bimanual_provenance'] = dict(primary_hand='right',
            primary_motion='retargeted_hymotion_or_logged_fallback',
            secondary_hand='rigid_offset_from_primary', tray_orientation='horizontal_constraint',
            raw_two_hand_replay=False)

    def placing(self):
        b, p = self.behavior, self.params
        tray = self.coaster = self.order[self.k]
        self.selected_target = tray
        hold = lambda ctx: ctx['table_top_z'] + ctx['object_half_height'][tray] + 0.001 + self.grip_dz
        end = np.asarray(b['grip_offset_xy'], float)
        on_tray = lambda ctx: np.r_[np.asarray(ctx['object_positions'][tray], float)[:2] + end, hold(ctx)]
        dest = np.asarray(self.dest[tray], float)
        start = lambda t, ctx: self.emit('human_cue_onset', modality='take_tray', selected_zone=tray)
        moves = [Gesture('sleep', 'pause', duration=float(self.pauses[self.k])) if self.pauses[self.k] > 0 else None,
                 self.G_call(start),
                 Gesture('move', 'reach_tray', target=on_tray, duration=p['take_reach_s'], shape='grasp', key=tray),
                 self.G_call(self._grab),
                 Gesture('move', 'lift_tray', target=lambda ctx: self.pos + [0.0, 0.0, float(b['carry_height_m'])],
                         duration=p['lower_s'], shape='grasp'),
                 Gesture('move', 'carry_tray', target=lambda ctx: np.r_[dest + end, hold(ctx) + float(b['carry_height_m'])],
                         duration=p['carry_s'], shape='grasp', key=tray),
                 Gesture('move', 'lower_tray', target=lambda ctx: np.r_[dest + end, hold(ctx)], duration=p['lower_s'],
                         shape='grasp'),
                 self.G_call(self._drop),
                 Gesture('move', 'withdraw', target=lambda ctx: self.rest, duration=p['take_reach_s'], shape='relaxed'),
                 Gesture('sleep', 'own_work', duration=float(self.work[self.k])),     # wiping the dishes (own work)
                 self.G_call(self._next)]
        return [g for g in moves if g is not None]

    def _grab(self, t, ctx):
        self._take(t, ctx)
        self.bimanual = dict(width_m=float(self.behavior['hands_apart_m']), shape='grasp')
        self.requests.append(('two_hand', self.coaster))

    def _drop(self, t, ctx):
        self._put(t, ctx)
        self.bimanual = None
        self.requests.append(('one_hand', self.coaster))
        self.emit('protocol_step_complete', step='tray_cleared_' + self.coaster, object=self.coaster)

    def _next(self, t, ctx):
        self.k += 1
        if self.k == len(self.order):
            self.goto('done', t)
            return
        self.queue = self.placing()


class LintRollTask(ShelfWipeTask):
    """C2's wiping robot (experts/c2_expert.py) on the places of the trays the human takes away."""

    def sample_episode_variation(self, seed, spec=None):
        var = sample_variation(self.cfg, seed, spec)
        cfg = self.cfg.scene_variation['w3']
        rng = np.random.default_rng([int(seed), 909])
        trays = list(cfg['trays'])
        pads = dict(cfg['pads'])
        pad_of = {k: p for k, p in zip(trays, pads)}
        xy = {k: (np.asarray(pads[pad_of[k]], float) + rng.uniform(-1, 1, 2) * float(cfg['jitter_m'])).tolist()
              for k in trays}
        var['object_xy'] = xy
        var['object_yaw'] = {k: 0.0 for k in trays}
        var['on_table'] = list(xy)
        order = [trays[i] for i in rng.permutation(len(trays))]
        dest = {k: (np.asarray(cfg['dest'][k], float) + rng.uniform(-1, 1, 2) * float(cfg['jitter_m'])).tolist()
                for k in trays}
        var['w3'] = dict(order=order, dest=dest, pauses=[0.0] + [0.0] * (len(order) - 1),
                         work_s=[float(rng.uniform(*cfg['work_s'])) for _ in order])
        var['w3']['pauses'][0] = float(rng.uniform(*cfg['first_pause_s']))
        var['c2'] = dict(order=order, pauses=[0.0] * len(order), pad_of=pad_of, pads={k: list(v) for k, v in pads.items()},
                         item_home={k: list(v) for k, v in xy.items()})
        var['w1'] = dict(robot_items=[], human_items=[], human_order=[], human_pauses=[], robot_box=None, human_box=None,
                         drop_slots=[])
        var['target_object'] = order[0]
        var['target_zone'] = None
        return var

    def on_reset(self):
        super().on_reset()
        old = self.human
        self.human = TrayClearer(self.scene, self.cfg.human_behavior, self.variation)
        for key in ('bank', 'bank_rng', 'bank_table_top', 'reach', 'point_pose', 'shoulder'):
            setattr(self.human, key, getattr(old, key))
        self.dest = {k: np.asarray(v, float) for k, v in self.variation['w3']['dest'].items()}

    def decision_point(self, tray):
        """Where the right hand takes a tray (its right end)."""
        return self.pos(tray) + np.r_[np.asarray(self.cfg.human_behavior['grip_offset_xy'], float), 0.0]

    def update_task(self):
        super().update_task()            # wiping (C2); its 'restored' never applies here
        if (all('wiped_' + k in self.protocol_done for k in self.pads) and self.human.stage == 'done'
                and self.robot_at_rest()):
            self.protocol('robot_at_rest', 'robot')

    def success_predicate(self):
        return ('robot_at_rest' in self.protocol_done and self.human.stage == 'done'
                and all('wiped_' + k in self.protocol_done for k in self.pads)
                and all(float(np.linalg.norm(self.pos(k)[:2] - self.dest[k][:2])) < 0.04 for k in self.items))

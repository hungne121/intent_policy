"""v7 C3 (docs/PIF.md §8.5 / §8.10 / §8.11; HABIT S6002 adapted to a one-arm robot): two trays, a bowl on each. The
human takes the folded cloth and brings it to the tray of their choice; the robot lifts that tray's bowl and holds it
aside; the human puts the cloth on the tray and withdraws; the robot sets the bowl on the cloth. The human waits with
the cloth near the tray while its bowl is still on it: lifting the right bowl early saves the human's waiting time.

Cases (variation, exogenous timing, PIF §8.10): C4 before = the cloth is carried towards the other tray first and
turned to the real one part of the way; v8 C4 after (clock=human_arrival) = after arriving at the first
tray and a sampled human delay, turn to the other tray; legacy v7 waits for bowl clearance (the robot must put the first bowl back and lift the other one). C3 is a
forecast intervention of the early expert (experts/c3_expert.py), not a human behaviour.
"""
import numpy as np
from intent_policy.scenarios.base_scenario import BaseScenario
from intent_policy.scenarios.config import sample_variation
from intent_policy.sim.scene_builder import object_half_height
from intent_policy.sim.scripted_human import CoasterPlacer, Gesture


class ClothSpreader(CoasterPlacer):
    """The cloth is the 'coaster' of CoasterPlacer (take / carry / put); the spot is the chosen tray."""

    def __init__(self, scene, behavior, variation):
        super().__init__(scene, behavior, variation)
        c3 = variation['c3']
        self.trays, self.bowls = c3['trays'], c3['bowls']
        self.selected_target = c3['target_tray']
        self.switch = c3.get('switch')
        self.decoy = self.switch['decoy_tray'] if self.switch else None

    def _tray(self, ctx, tray):
        return np.asarray(self.trays[tray], float)

    def _hold_z(self, ctx):
        """Palm height carrying the cloth level at its put height + carry height."""
        return ctx['table_top_z'] + float(self.behavior['tray_top_m']) + ctx['object_half_height'][self.coaster] + 0.001 \
            + self.grip_dz

    def _wait_point(self, ctx, tray):
        """Next to the tray on the human's side, at carry height (never over a tray the robot still works at)."""
        return np.r_[self._tray(ctx, tray) + np.asarray(self.behavior['wait_offset_xy'], float),
                     self._hold_z(ctx) + float(self.behavior['carry_height_m'])]

    def _tray_clear(self, tray, ctx) -> bool:
        """The bowl of `tray` has been lifted off it (seen: it is up, or away from the tray)."""
        bowl = self.bowls[tray]
        p = np.asarray(ctx['object_positions'][bowl], float)
        return bool(np.linalg.norm(p[:2] - self._tray(ctx, tray)) > float(self.behavior['tray_radius_m'])
                    or p[2] > ctx['table_top_z'] + float(self.behavior['tray_top_m']) + ctx['object_half_height'][bowl] + 0.03)

    def spot_xy(self, ctx):
        return self._tray(ctx, self.decoy if self.decoy is not None else self.selected_target)

    def _switch(self, t, ctx):
        old = self.decoy
        self.decoy = None
        self.emit('human_intention_change', kind='target_tray', previous=old, current=self.selected_target,
                  timing=self.switch['kind'], clock=self.switch.get('clock', 'robot_clear'),
                  delay_s=self.switch.get('delay_s'))

    def placing(self):
        b, p, c = self.behavior, self.params, self.coaster
        on_cloth = lambda ctx: np.asarray(ctx['object_positions'][c], float) + [0.0, 0.0, self.grip_dz]
        put = lambda ctx: np.r_[self._tray(ctx, self.selected_target), self._hold_z(ctx)]
        tray_now = lambda: self.decoy if self.decoy is not None else self.selected_target
        reach = Gesture('move', 'reach_cloth', target=on_cloth, duration=p['take_reach_s'], shape='grasp', key=c)
        carry = lambda tray_fn, frac=1.0, label='carry_cloth': Gesture(
            'move', label, target=lambda ctx: self.pos + frac * (self._wait_point(ctx, tray_fn()) - self.pos),
            duration=p['carry_s'] * frac, shape='grasp', key=lambda: tray_fn())
        wait_clear = lambda tray_fn: Gesture('wait', 'wait_tray_clear', until=lambda t, ctx: self._tray_clear(tray_fn(), ctx))
        over = Gesture('move', 'over_tray', target=lambda ctx: put(ctx) + [0.0, 0.0, float(b['carry_height_m'])],
                       duration=p['lower_s'], shape='grasp', key=lambda: self.selected_target)
        lower = Gesture('move', 'lower_cloth', target=put, duration=p['lower_s'], shape='grasp',
                        key=lambda: self.selected_target)
        tail = [over, lower, self.G_call(self._put), self.G_rest('withdraw'), self.G_call(lambda t, ctx: self.goto('done', t))]
        moves = [reach, self.G_call(self._take)]
        sw = self.switch
        if sw is None:
            return moves + [carry(tray_now), wait_clear(tray_now)] + tail
        if sw['kind'] == 'before':                  # towards the decoy, turned at `progress` of the way
            return moves + [carry(tray_now, float(sw['progress']), 'carry_towards_decoy'), self.G_call(self._switch),
                            carry(tray_now), wait_clear(tray_now)] + tail
        # after: wait at the decoy tray until its bowl is lifted, then decide (delay) and go to the real tray
        # Legacy v7 is retained for reproducibility; v8 never waits for the robot to trigger a change.
        before_decision = [] if sw.get('clock') == 'human_arrival' else [wait_clear(tray_now)]
        return moves + [carry(tray_now)] + before_decision + [Gesture('sleep', 'decide', duration=float(sw['delay_s'])),
                        self.G_call(self._switch), carry(tray_now, label='carry_cloth'), wait_clear(tray_now)] + tail

    def _put(self, t, ctx):
        super()._put(t, ctx)
        self.emit('protocol_step_complete', step='cloth_on_' + self.selected_target, tray=self.selected_target)


class ClothBowlTask(BaseScenario):
    v6_plan = 'cloth_bowl'

    @property
    def expert_class(self):
        from intent_policy.experts.c3_expert import C3Expert
        return C3Expert

    def sample_episode_variation(self, seed, spec=None):
        var = sample_variation(self.cfg, seed, spec)
        cfg = self.cfg.scene_variation['c3']
        rng = np.random.default_rng([int(seed), 303])
        trays = {t: list(xy) for t, xy in cfg['trays'].items()}
        bowls = dict(cfg['bowls'])
        for t, bowl in bowls.items():
            var['object_xy'][bowl] = (np.asarray(trays[t]) + rng.uniform(-1, 1, 2) * float(cfg['bowl_jitter_m'])).tolist()
            var['object_yaw'][bowl] = 0.0
        cl = cfg['cloth']
        var['object_xy'][cl] = (np.asarray(cfg['cloth_xy']) + rng.uniform(-1, 1, 2) * float(cfg['cloth_jitter_m'])).tolist()
        var['object_yaw'][cl] = 0.0
        var['on_table'] = list(bowls.values()) + [cl]
        var['object_xy'] = {k: var['object_xy'][k] for k in var['on_table']}
        var['object_yaw'] = {k: var['object_yaw'][k] for k in var['on_table']}
        var['object_dz'] = {b: float(cfg['tray_top_m']) for b in bowls.values()}
        names = sorted(trays)
        target = names[int(rng.integers(len(names)))]
        c3 = dict(trays=trays, bowls=bowls, cloth=cl, target_tray=target)
        sw = cfg.get('switch')
        if sw:
            other = next(t for t in names if t != target)
            c3['switch'] = dict(kind=sw['kind'], decoy_tray=other, clock=sw.get('clock', 'robot_clear'))
            if sw['kind'] == 'before':
                c3['switch']['progress'] = float(rng.uniform(*sw['progress']))
            else:
                c3['switch']['delay_s'] = float(rng.uniform(*sw['delay_s']))
        ff = cfg.get('forecast_fault')
        if ff:
            # C3 (two targets): the forecast arm is turned about the shoulder towards the OTHER tray, by the angle
            # between the two trays seen from the shoulder x U(factor) - "a point far from A" is the other tray here
            # (45-75 deg swung the prediction past both trays: no wrong action, pilot 2026-10-30)
            sh = np.asarray(self.scene['human']['stand_xy'], float)
            other = next(t for t in names if t != target)
            ang = lambda t: np.arctan2(*(np.asarray(trays[t]) - sh)[::-1])
            yaw = np.degrees(ang(other) - ang(target)) * float(rng.uniform(*ff['factor']))
            c3['forecast_fault'] = dict(yaw_deg=float(yaw), extra_s=float(ff['extra_s']), ramp_s=float(ff['ramp_s']))
        var['c3'] = c3
        var['target_object'] = bowls[target]
        var['target_zone'] = target
        return var

    def on_reset(self):
        c3 = self.variation['c3']
        self.trays, self.bowls, self.cloth = c3['trays'], c3['bowls'], c3['cloth']
        self.coaster = self.cloth
        self.target_tray = c3['target_tray']
        self.cloth_top = 2 * object_half_height(self.scene['objects'][self.cloth])
        self.robot_bowl = None                      # the bowl the robot works on now (set by the expert)
        old = self.human                            # the cloth spreader instead of the generic coaster placer
        self.human = ClothSpreader(self.scene, self.cfg.human_behavior, self.variation)
        for key in ('bank', 'bank_rng', 'bank_table_top', 'reach', 'point_pose', 'shoulder'):
            setattr(self.human, key, getattr(old, key))
        self.realized_forecast_fault = None
        self.watch = []                             # bowls / cloth are checked here and by the collection gate

    def zone_pos(self, key):
        if key in getattr(self, 'trays', {}):
            return np.r_[self.trays[key], self.table_z + float(self.cfg.human_behavior['tray_top_m'])]
        return super().zone_pos(key)

    def robot_target(self):
        held = [b for b in self.bowls.values() if self.grasp_state.get(b)]
        return held[0] if held else (self.robot_bowl or self.bowls[self.target_tray])

    def coaster_down(self):
        return ('cloth_on_' + self.target_tray) in self.protocol_done

    def bowl_on_tray(self, bowl, tray, on_cloth=False) -> bool:
        sc = self.cfg.success_conditions
        p = self.pos(bowl)
        z = self.table_z + float(self.cfg.human_behavior['tray_top_m']) + (self.cloth_top if on_cloth else 0.0) \
            + object_half_height(self.scene['objects'][bowl])
        ref = self.pos(self.cloth)[:2] if on_cloth else np.asarray(self.trays[tray], float)
        return bool(np.linalg.norm(p[:2] - ref) <= sc['on_place_xy_m'] and abs(p[2] - z) <= 0.006
                    and not self.grasp_state[bowl] and self.speed(bowl) < sc['object_speed_max'])

    def update_task(self):
        for tray, bowl in self.bowls.items():
            if self.grasp_state[bowl] and self.robot_lifted == bowl:
                self.protocol('bowl_lifted_' + tray, 'robot', object=bowl)
        t = self.target_tray
        if self.coaster_down():
            cl = self.pos(self.cloth)
            if np.linalg.norm(cl[:2] - np.asarray(self.trays[t])) > self.cfg.success_conditions['cloth_on_tray_xy_m']:
                self.fail('cloth_not_on_tray')
            if self.bowl_on_tray(self.bowls[t], t, on_cloth=True):
                self.protocol('bowl_on_cloth', 'robot', object=self.bowls[t])
        if 'bowl_on_cloth' in self.protocol_done and self.robot_at_rest():
            self.protocol('robot_at_rest', 'robot')
        for bowl in self.bowls.values():
            if bowl in self.ever_lifted and not self.grasp_state[bowl] and self.pos(bowl)[2] < self.table_z + 0.02 \
                    and self.speed(bowl) < 0.02:
                self.fail('bowl_dropped')

    def success_predicate(self):
        t = self.target_tray
        others = [(tr, b) for tr, b in self.bowls.items() if tr != t]
        return ('robot_at_rest' in self.protocol_done and self.human.stage == 'done'
                and self.bowl_on_tray(self.bowls[t], t, on_cloth=True)
                and all(self.bowl_on_tray(b, tr) for tr, b in others))

    def ground_truth_intention_information(self):
        return dict(target_object=self.bowls[self.target_tray], target_region=self.target_tray,
                    target_position=self.zone_pos(self.target_tray).round(5).tolist())

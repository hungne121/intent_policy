"""I1: nhiều đế trong một scene vật lý; pilot chưa chứng nhận demo train."""
import numpy as np
from intent_policy.scenarios.t3_coaster_cup import CoasterCupTask
from intent_policy.scenarios.config import InfeasibleVariation, sample_variation
from intent_policy.sim.scene_builder import object_half_height
from intent_policy.sim.scripted_human import CoasterPlacer, Gesture


class SequentialCoasterHuman(CoasterPlacer):
    def __init__(self, scene, behavior, variation):
        super().__init__(scene, behavior, variation)
        self.rounds = variation['coaster_sequence']
        self.human_round = 0
        self._select_round()

    def _select_round(self):
        item = self.rounds[self.human_round]
        self.coaster = item['coaster']
        self.selected_target = item['zone']

    def spot_xy(self, ctx):
        if getattr(self, 'decoy_xy', None) is not None:
            return np.asarray(self.decoy_xy, float)
        return np.asarray(self.rounds[self.human_round]['destination_xy'])

    def _switch(self, t, ctx):
        """C4: the human's goal for this coaster changes from the decoy spot A to the real spot B (logged for the
        evaluation only; the robot sees the motion)."""
        sw = self.rounds[self.human_round]['switch']
        self.decoy_xy = None
        self.emit('human_intention_change', kind='coaster_spot', previous=np.round(sw['decoy_xy'], 4).tolist(),
                  current=np.round(self.rounds[self.human_round]['destination_xy'], 4).tolist(), timing=sw['kind'],
                  round=self.human_round)

    def _retake(self, t, ctx):
        self.requests.append(('attach', self.coaster))
        self.holding = self.coaster
        self.emit('object_grasp', object=self.coaster, actor='human')
        self.emit('protocol_step_complete', step='coaster_retaken_' + self.coaster, object=self.coaster,
                  round=self.human_round)

    def placing(self):
        sw = self.rounds[self.human_round].get('switch')
        self.decoy_xy = np.asarray(sw['decoy_xy'], float) if sw else None
        moves = super().placing()
        if sw is not None:
            moves = self._switch_moves(moves, sw)
        c = self.coaster
        clearance = float(self.behavior['source_approach_height_m'])
        lift = Gesture('move', 'source_raise',
            target=lambda ctx: np.r_[self.pos[:2], ctx['table_top_z'] + clearance],
            duration=float(self.behavior['source_raise_s']), shape='relaxed')
        above = Gesture('move', 'source_approach',
            target=lambda ctx: np.r_[ctx['object_positions'][c][:2], ctx['table_top_z'] + clearance],
            duration=float(self.behavior['source_approach_s']), shape='grasp')
        # Optionally lift before returning to rest so the withdrawal can clear
        # nearby source objects. This is a pilot path, not a collision waiver.
        if 'withdraw_raise_height_m' in self.behavior:
            raise_hand = Gesture('move', 'withdraw_raise',
                target=lambda ctx: np.r_[self.pos[:2], ctx['table_top_z'] +
                                         float(self.behavior['withdraw_raise_height_m'])],
                duration=float(self.behavior['withdraw_raise_s']), shape='relaxed')
            clear_sources = Gesture('move', 'withdraw_clear_sources',
                target=lambda ctx: np.r_[self.rest[:2], ctx['table_top_z'] +
                                         float(self.behavior['withdraw_raise_height_m'])],
                duration=float(self.behavior['withdraw_raise_s']), shape='relaxed')
            moves = moves[:-2] + [raise_hand, clear_sources] + moves[-2:]
        # Không chờ robot: đặt đế tiếp theo sau khi rút tay xong.
        return [lift, above] + moves[:-1] + [self.G_call(self._next_round)]

    def _switch_moves(self, moves, sw):
        """moves = [reach, take, carry, lower, put, withdraw, done] (CoasterPlacer.placing)."""
        b = self.behavior
        reach, take, carry, lower, put, withdraw, done = moves
        hold = lambda ctx: ctx['table_top_z'] + ctx['object_half_height'][self.coaster] + 0.001 + self.grip_dz
        if sw['kind'] == 'before':
            # towards A, redirected at `progress` of the way (the hand slows and turns), then on to B
            frac = float(sw['progress'])
            to_a = Gesture('move', 'carry_coaster',
                target=lambda ctx: self.pos + frac * (np.r_[self.decoy_xy, hold(ctx) + b['carry_height_m']] - self.pos),
                duration=self.params['carry_s'] * frac, shape='grasp', key=lambda: self.selected_target)
            return [reach, take, to_a, self.G_call(self._switch), carry, lower, put, withdraw, done]
        # after: put down at A, lift the hand, decide (exogenous delay), take it again and move it to B
        lift = Gesture('move', 'withdraw_raise', target=lambda ctx: self.pos + [0.0, 0.0, float(sw['lift_m'])],
                       duration=float(sw['lift_s']), shape='grasp')
        again = Gesture('move', 'reach_coaster', target=lambda ctx: np.asarray(ctx['object_positions'][self.coaster], float)
                        + [0.0, 0.0, self.grip_dz], duration=float(sw['lift_s']), shape='grasp', key=self.coaster)
        carry_b = Gesture('move', 'carry_coaster',
            target=lambda ctx: np.r_[np.asarray(self.rounds[self.human_round]['destination_xy'], float),
                                     hold(ctx) + b['carry_height_m']],
            duration=self.params['carry_s'], shape='grasp', key=lambda: self.selected_target)
        return [reach, take, carry, lower, put, lift, Gesture('sleep', 'decide', duration=float(sw['delay_s'])),
                self.G_call(self._switch), again, self.G_call(self._retake), carry_b, lower, put, withdraw, done]

    def _put(self, t, ctx):
        super()._put(t, ctx)
        self.emit('protocol_step_complete', step='coaster_placed_' + self.coaster,
                  object=self.coaster, round=self.human_round)

    def _next_round(self, t, ctx):
        self.human_round += 1
        if self.human_round == len(self.rounds):
            self.goto('done', t)
            return
        self._select_round()
        pause = float(self.rounds[self.human_round].get('pause_before_s', 0.0))
        start = lambda t, ctx: self.emit('human_cue_onset', modality='human_place', selected_zone=self.selected_target,
                                         round=self.human_round)
        # other table work between two coasters (hand at rest): the human's pace varies (HABIT: no fixed schedule)
        self.queue = ([Gesture('sleep', 'between_coasters', duration=pause)] if pause > 0 else []) + \
            [self.G_call(start)] + self.placing()


class CoasterSequenceTask(CoasterCupTask):
    @property
    def expert_class(self):
        # expert.rule observed | early: the I1 demonstrator (experts/i1_expert.py); else the workflow-check expert
        if 'rule' in self.cfg.expert:
            from intent_policy.experts.i1_expert import I1Expert
            return I1Expert
        return None

    def reserved_obstacles(self):
        """A coaster that is down but still waits for its cup: the cup will stand there (the robot may set it down
        while the human carries the next coaster past), so the hand keeps clear of a cup on it."""
        out = []
        for item in self.rounds:
            cup, coaster = item['cup'], item['coaster']
            if 'coaster_placed_' + coaster in self.protocol_done and 'cup_on_coaster_' + cup not in self.protocol_done:
                p = self.pos(coaster)
                top = p[2] + object_half_height(self.scene['objects'][coaster]) + \
                    2 * object_half_height(self.scene['objects'][cup])
                out.append((p[:2], max(float(self.radius[coaster]), float(self.radius[cup])) + 0.01, float(top)))
        return out

    def round_released(self, cup) -> bool:
        """The robot has let go of the cup on its coaster and lifted the gripper clear of the rim (3 cm)."""
        if 'rule' not in self.cfg.expert:
            return self.robot_at_rest()
        top = self.pos(cup)[2] + object_half_height(self.scene['objects'][cup]) + 0.03
        return not self.grasp_state[cup] and self.tcp()[2] >= top

    @property
    def cue_complete_step(self):
        return 'coaster_placed_' + self.coaster

    def sample_episode_variation(self, seed, spec=None):
        var = sample_variation(self.cfg, seed, spec)
        cfg = self.cfg.scene_variation['sequence']
        rng = np.random.default_rng([int(seed), 101])
        for restart in range(int(cfg.get('sampling_restarts', 1))):   # greedy draws can corner themselves: redraw all
            positions = []
            for _ in cfg['items']:
                for attempt in range(cfg['sampling_attempts']):
                    p = rng.uniform(cfg['destination_min_xy'], cfg['destination_max_xy'])
                    if np.linalg.norm(p) < float(cfg.get('min_base_distance_m', 0.0)):
                        continue
                    if all(np.linalg.norm(p - q) >= cfg['minimum_spacing_m'] for q in positions):
                        positions.append(p)
                        break
                else:
                    break
            if len(positions) == len(cfg['items']):
                break
        else:
            raise InfeasibleVariation('No separated coaster placement found within configured attempts')
        rounds = []
        for i, (item, p) in enumerate(zip(cfg['items'], positions)):
            cup, coaster = item['cup'], item['coaster']
            for key, xy in [(cup, item['cup_xy']), (coaster, item['coaster_xy'])]:
                var['object_xy'][key] = list(xy)
                var['object_yaw'][key] = 0.0
            pause = float(rng.uniform(*cfg['pause_before_s'])) if i > 0 and 'pause_before_s' in cfg else 0.0
            rounds.append(dict(cup=cup, coaster=coaster, zone=f'K{i+1}', destination_xy=p.tolist(), pause_before_s=pause))
        for kind in ('switch', 'forecast_fault'):
            spec = cfg.get(kind)
            if not spec:
                continue
            r = int(rng.integers(len(rounds)))
            if kind == 'forecast_fault':
                rounds[r]['forecast_fault'] = dict(yaw_deg=float(rng.choice([-1, 1]) * rng.uniform(*spec['yaw_deg'])),
                                                   extra_s=float(spec['extra_s']), ramp_s=float(spec['ramp_s']))
                continue
            # decoy A: a feasible spot apart from the real spot B and from the coasters already down (earlier rounds);
            # later rounds' spots are still free when the switch happens. The rounds are tried in a seeded order and
            # the first with room for a decoy is the switched one (the last round often has none: 3 spots fill it).
            # 'before': the carry towards A and the turn to B must both be >= 0.10 m and A >= 30 deg off the way to B
            # (a visible redirect, and long enough for a HY-Motion segment; shorter ones fell back to minimum-jerk)
            a, frac = None, None
            for r in rng.permutation(len(rounds)).tolist():
                taken = positions[:r + 1]
                src = np.asarray(cfg['items'][r]['coaster_xy'], float)
                for _ in range(cfg['sampling_attempts']):
                    cand = rng.uniform(cfg['destination_min_xy'], cfg['destination_max_xy'])
                    if np.linalg.norm(cand) < float(cfg.get('min_base_distance_m', 0.0)) or any(
                            np.linalg.norm(cand - q) < cfg['minimum_spacing_m'] for q in taken):
                        continue
                    if spec['kind'] == 'before':
                        f = float(rng.uniform(*spec['progress']))
                        mid = src + f * (cand - src)
                        da, db = cand - src, positions[r] - src
                        cos = float(da @ db / (np.linalg.norm(da) * np.linalg.norm(db) + 1e-9))
                        if np.linalg.norm(mid - src) < 0.10 or np.linalg.norm(positions[r] - mid) < 0.10 or cos > np.cos(np.radians(30)):
                            continue
                        frac = f
                    a = cand
                    break
                if a is not None:
                    break
            if a is None:
                raise InfeasibleVariation('No decoy spot found')
            sw = dict(kind=spec['kind'], decoy_xy=a.tolist())
            if spec['kind'] == 'before':
                sw['progress'] = frac
            else:
                sw.update(delay_s=float(rng.uniform(*spec['delay_s'])), lift_m=float(spec['lift_m']),
                          lift_s=float(spec['lift_s']))
            rounds[r]['switch'] = sw
        var['on_table'] = list(var['object_xy'])
        var['coaster_sequence'] = rounds
        var['target_object'] = rounds[0]['cup']
        var['target_zone'] = rounds[0]['zone']
        return var

    def on_reset(self):
        super().on_reset()
        self.realized_forecast_fault = None          # C3: the fault window the early expert applied (episode metadata)
        self.rounds = self.variation['coaster_sequence']
        self.sequence_index = 0
        self.completed = []
        self.placed_positions = {}
        old = self.human
        self.human = SequentialCoasterHuman(self.scene, self.cfg.human_behavior, self.variation)
        for key in ('bank', 'bank_rng', 'bank_table_top', 'reach', 'point_pose', 'shoulder'):
            setattr(self.human, key, getattr(old, key))
        self._select_round()

    def _select_round(self):
        item = self.rounds[self.sequence_index]
        self.coaster, self.spot = item['coaster'], item['zone']
        self.variation['target_object'] = item['cup']
        self.variation['target_zone'] = item['zone']
        self.watch = [r['cup'] for r in self.rounds if r['cup'] != item['cup']]

    def protocol(self, step, entity_id='scene', **payload):
        if step.startswith('coaster_retaken_'):
            coaster = step.removeprefix('coaster_retaken_')
            placed = 'coaster_placed_' + coaster
            if placed in self.protocol_done:              # C4 after: the coaster is moved again, not placed any more
                self.protocol_done.remove(placed)
            self.placed_positions.pop(coaster, None)
            self.log('protocol_step_complete', entity_id, step=step, **payload)
            return
        super().protocol(step, entity_id, **payload)

    def coaster_down(self):
        return self.cue_complete_step in self.protocol_done

    def zone_pos(self, key):
        # Nominal locations remain the public layout until a coaster is placed.
        for item in getattr(self, 'rounds', []):
            if key == item['zone'] and 'coaster_placed_' + item['coaster'] in self.protocol_done:
                return np.r_[self.pos(item['coaster'])[:2], self.table_z]
        return super().zone_pos(key)

    def update_task(self):
        cup = self.robot_target()
        if self.robot_lifted == cup:
            self.protocol('cup_grasped_' + cup, 'robot', object=cup)
        mark = 'cup_on_coaster_' + cup
        if cup in self.ever_lifted and not self.grasp_state[cup] and self.speed(cup) < .02 and mark not in self.protocol_done:
            if self.on_coaster():
                self.protocol(mark, 'robot', object=cup, coaster=self.coaster)
            elif self.pos(cup)[2] < self.rest_height[cup] + self.coaster_top + .01:
                self.fail('cup_not_on_coaster')
        for item in self.rounds:
            coaster = item['coaster']
            if 'coaster_placed_' + coaster in self.protocol_done:
                if coaster not in self.placed_positions:
                    self.placed_positions[coaster] = self.pos(coaster)[:2]
                elif np.linalg.norm(self.pos(coaster)[:2] - self.placed_positions[coaster]) > .02:
                    self.fail('coaster_moved')
        if mark in self.protocol_done and self.round_released(cup) and cup not in self.completed:
            self.completed.append(cup)
            # Cốc đã hoàn thành phải giữ nguyên tại đế trong các lượt sau.
            self.start_pos[cup] = self.pos(cup)
            self.protocol('sequence_round_complete_' + cup, 'robot', object=cup)
            if len(self.completed) < len(self.rounds):
                self.sequence_index += 1
                self._select_round()

    def success_predicate(self):
        if len(self.completed) != len(self.rounds) or not self.robot_at_rest() or self.human.stage != 'done':
            return False
        for item in self.rounds:
            cup, coaster = item['cup'], item['coaster']
            if np.linalg.norm(self.pos(cup)[:2] - self.pos(coaster)[:2]) > self.cfg.success_conditions['on_coaster_xy_m']:
                return False
            if abs(self.pos(cup)[2] - self.rest_height[cup] - self.coaster_top) > .005:
                return False
        return True

"""I1 pilot: one simple sequence, with three independently welded coasters in hand."""
import numpy as np
from intent_policy.scenarios.i1_coaster_sequence import SequentialCoasterHuman
from intent_policy.sim.scripted_human import Gesture


class StackedCoasterHuman(SequentialCoasterHuman):
    def placing(self):
        p, b, key = self.params, self.behavior, self.coaster
        def put(ctx):
            return np.r_[self.spot_xy(ctx),
                         ctx['table_top_z'] + ctx['object_half_height'][key] + .001 + self.grip_dz]
        def set_down(ctx):
            return self.pos + [0., 0., b['set_down_aim_m'] - float(ctx['object_positions'][key][2] - ctx['object_half_height'][key] - ctx['table_top_z'])]
        return [Gesture('move', 'carry_coaster', target=lambda ctx: put(ctx) + [0, 0, b['carry_height_m']],
                        duration=p['carry_s'], shape='grasp', key=lambda: self.selected_target),
                Gesture('move', 'lower_coaster', target=put, duration=p['lower_s'],
                        shape='grasp', key=lambda: self.selected_target),
                Gesture('move', 'set_down_coaster', target=set_down, duration=b['set_down_s'],
                        shape='grasp', key=key),
                Gesture('move', 'set_down_coaster', target=set_down, duration=b['set_down_s'],
                        shape='grasp', key=key),
                Gesture('move', 'set_down_coaster', target=set_down, duration=b['set_down_s'],
                        shape='grasp', key=key),
                self.G_call(self._put),
                self.G_rest('withdraw', duration=p['rest_s'], shape='grasp'),
                self.G_call(self._next_round)]

    def free_pause(self):
        # 13/11 (user): once the robot is prepared above the cup, the human places the coaster whenever they like
        rng = getattr(self, 'pause_rng', None) or np.random.default_rng(917)
        return float(rng.uniform(0.0, 2.0))

    def _put(self, t, ctx):
        super()._put(t, ctx)
        if self.human_round + 1 < len(self.rounds):
            self.holding = self.rounds[self.human_round + 1]['coaster']

    def _next_round(self, t, ctx):
        previous_cup = self.rounds[self.human_round]['cup']
        self.human_round += 1
        if self.human_round == len(self.rounds):
            self.goto('done', t)
            return
        self._select_round()
        self.queue = [Gesture('wait', 'wait_robot_place_cup',
                              until=lambda tt, cc, cup=previous_cup:
                                  'cup_on_coaster_' + cup in cc['protocol_done']),
                      # 13/11 (user): the robot already holds the lifted cup before the human starts
                      Gesture('wait', 'wait_robot_holds_next_cup',
                              until=lambda tt, cc, cup=self.rounds[self.human_round]['cup']:
                                  'cup_grasped_' + cup in cc['protocol_done']),
                      Gesture('sleep', 'free_pause', duration=self.free_pause(), scale=False),
                      self.G_call(lambda tt, cc: self.emit('human_cue_onset',
                      modality='human_place', round=self.human_round))] + self.placing()

    def behave(self, t, ctx):
        if not self.started:
            self.started = True
            self.queue = [Gesture('sleep', 'initial_still', duration=self.initial_still_s,
                                  scale=False),
                          Gesture('wait', 'wait_robot_holds_first_cup',
                                  until=lambda tt, cc:
                                      'cup_grasped_' + self.rounds[0]['cup'] in cc['protocol_done']),
                          Gesture('sleep', 'free_pause', duration=self.free_pause(), scale=False),
                          self.G_call(lambda tt, cc: self.emit('human_cue_onset',
                                      modality='human_place', round=self.human_round))] + self.placing()
        self.run_queue(t, ctx)

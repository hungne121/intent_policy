"""I2 corrective expert (PILOT, 28/QĐ-HL Phase 3). Same FSM as i2_expert up to the grasp, then:
  * dwell after attach: random extra hold AFTER the observed attach (valid timing variation: the lift never starts before
    the attach, so the demonstration never teaches 'lift at a fixed time');
  * variant `recover`: the close is cut at a sampled hold < 1 s and the arm starts to lift without the cake (the failure
    the policies show); the expert then detects it from the object state (cake not attached to the gripper), opens,
    re-descends on the cake's CURRENT position, closes until attach and carries on (up to `max_attempts`).
Frames of the injected segment have phase `injected_early_lift` (to be masked from the training loss, never imitated);
recovery frames have phases `recover_*`. Config: configs/hrc/i2_corrective.yaml."""
import numpy as np
import yaml

from intent_policy.experts.i2_expert import Expert as I2Expert
from intent_policy.scenarios.i2 import PLATES
from intent_policy.shared.sim_env import ROOT, GRIP_OPEN, GRIP_CLOSED

CFG = yaml.safe_load((ROOT / 'configs/hrc/i2_corrective.yaml').read_text())
DT = 0.05


def make(variant):
    spec = CFG['variants'][variant]

    class Expert(I2Expert):
        corrective_variant = variant

        def __init__(self, ep):
            super().__init__(ep)
            rng = np.random.default_rng(int(ep.seed) + 7919)
            lo, hi = spec['dwell_after_attach_s']
            self.dwell = float(rng.uniform(lo, hi))
            h = spec['early_lift_hold_s']
            self.early_hold = None if h is None else float(rng.uniform(*h))
            self.attempts = 0

        def attached(self):
            return self.ep.attached_name() == 'cake'

        def run(self):
            ep = self.ep
            src = ep.pos('cake')
            yield from self.goto(np.r_[src[:2], .80], 'above_cake', GRIP_OPEN)
            if self.forecast_mode:
                yield from self.wait(lambda: self.plan.active() or (self.selected is not None and self.ready >= 3), 'wait_cue', GRIP_OPEN)
                ep.event('pick_start', basis='forecast' if self.selected is None else 'observed')
            else:
                yield from self.wait(lambda: self.selected is not None and self.ready >= 3, 'wait_cue', GRIP_OPEN)
                ep.event('pick_start', basis='observed')
            yield from self.goto(np.r_[src[:2], .667], 'descend', GRIP_OPEN)
            if self.early_hold is not None:
                ep.event('injected_early_lift', hold_s=round(self.early_hold, 2), dwell_s=round(self.dwell, 2))
                ep.phase = 'close'
                for _ in range(int(round(self.early_hold / DT))):
                    self.observe(); yield ep.hold(GRIP_CLOSED)
                yield from self.goto(np.r_[src[:2], CFG['perturb_lift_z']], 'injected_early_lift', GRIP_CLOSED)
                while not self.attached() and self.attempts < CFG['max_attempts']:
                    self.attempts += 1
                    ep.event('grasp_not_attached', attempt=self.attempts)
                    for cmd in ep.open_gripper('recover_open', ticks=CFG['recover_open_ticks']):
                        self.observe(); yield cmd
                    cake = ep.pos('cake')
                    yield from self.goto(np.r_[cake[:2], .72], 'recover_above', GRIP_OPEN)
                    yield from self.goto(np.r_[cake[:2], .667], 'recover_descend', GRIP_OPEN)
                    for cmd in ep.close_on('cake', phase='recover_close'):
                        self.observe(); yield cmd
            else:
                for cmd in ep.close_on('cake'):
                    self.observe(); yield cmd
            for _ in range(int(round(self.dwell / DT))):
                self.observe(); yield ep.hold(GRIP_CLOSED)
            yield from self.goto(np.r_[src[:2], .80], 'lift', GRIP_CLOSED, follow=self.forecast_mode)
            yield from self.wait(lambda: self.selected is not None and np.linalg.norm(ep.wrist() - PLATES[self.selected]) > .22,
                                 'wait_target', GRIP_CLOSED)
            plate = PLATES[self.selected]; ep.event('transport_start', plate=self.selected)
            yield from self.goto(np.r_[plate[:2], .80], 'move_plate', GRIP_CLOSED)
            yield from self.goto(np.r_[plate[:2], .674], 'lower_plate', GRIP_CLOSED)
            for cmd in ep.open_gripper('release', ticks=20):
                self.observe(); yield cmd
            yield from self.goto(np.r_[plate[:2], .80], 'retreat', GRIP_OPEN)
            ep.phase = 'done'

        def extra(self):
            return {**super().extra(), 'corrective_variant': variant, 'early_hold_s': self.early_hold,
                    'dwell_s': self.dwell, 'recover_attempts': self.attempts}

    return Expert

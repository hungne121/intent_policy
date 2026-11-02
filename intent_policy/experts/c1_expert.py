"""v7 C1 demonstrator: the T2 goal-belief expert (scripted_expert, trigger goal, goal_rule early | observed) plus the
v7 separation monitoring outside the hand-over itself (experts/separation.py): too close to the human, the robot moves
away instead of on. The final hand-over (down into the still, open palm, waiting for the pull, release) is the
interaction itself and keeps T2's own rules.
"""
from __future__ import annotations
import numpy as np
from intent_policy.experts.scripted_expert import ScriptedExpert

HANDOVER = ('over_hand', 'settle_over_hand', 'to_handover', 'time_descent', 'wait_pull', 'release',
            'wait_human_withdraw')


class C1Expert(ScriptedExpert):
    def _back_away(self):
        sc = self.sc
        tcp, hand = sc.tcp(), sc.human.pos
        u = tcp[:2] - hand[:2]
        u = u / max(float(np.linalg.norm(u)), 1e-9)
        return np.r_[tcp[:2] + 0.08 * u, min(tcp[2] + 0.03, self.travel + 0.05)]

    def act(self) -> int:
        from intent_policy.experts.separation import separation_hold
        from intent_policy.sim.restricted_action import RestrictedAction as A
        # Only the stationary open-palm transfer is exempt. A visible withdrawal must
        # restore separation immediately, rather than waiting for the belief debounce.
        withdrawing = (self.sc.cfg.expert.get('guard_withdrawal', False)
                       and self.sc.human.hand_shape == 'relaxed'
                       and self.sc.human.holding is None)
        if (self.stage not in HANDOVER or withdrawing) and separation_hold(self):
            if self.belief is not None:
                self._follow_belief()            # keep the belief current while backing away
            a = self._toward(self._back_away())
            return int(A.HOLD) if a is None else int(a)
        return super().act()

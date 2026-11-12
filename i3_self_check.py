"""I3 self checks: existing arm/body slabs plus opposite-arm mesh pairs."""
from intent_policy.sim.human_self_check import HumanSelfCheck
from intent_policy.sim.demo_contact_audit import DemoContactAudit, PairRule


class I3SelfCheck(HumanSelfCheck):
    def __init__(self, model):
        super().__init__(model)
        arms = {}
        for side in ('l', 'r'):
            arms[side] = [g for g in range(model.ngeom)
                          if model.body(model.geom_bodyid[g]).name in
                          {f'human_seg_{side}_{part}' for part in ('upperarm', 'forearm', 'hand')}]
        self.opposite = DemoContactAudit(model, [PairRule(a, b, 0., 0.)
                                                for a in arms['l'] for b in arms['r']])

    def measure(self, data):
        result = super().measure(data)
        self.opposite.observe(self.model, data)
        for index, distance, _, _ in self.opposite.step_violations:
            pair = self.opposite.rules[index]
            names = (self.model.geom(pair.geom_a).name, self.model.geom(pair.geom_b).name)
            result[names] = max(result.get(names, 0.), -float(distance))
        return result

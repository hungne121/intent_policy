"""Fail-fast geometry diagnostic of existing executable scenarios, no demo export.

Checks human visual hulls, objects and robot collision geoms at physics rate.
No contact exemptions: a flagged intended contact needs review, not automatic
acceptance. This is deliberately NOT the full dataset quality gate.
"""
import argparse
import json
from pathlib import Path
from itertools import combinations, product
import mujoco
from intent_policy.benchmark.runner import ExpertAgent, run_episode
from intent_policy.scenarios.scenario_registry import make_scenario
from intent_policy.sim.demo_contact_audit import DemoContactAudit, PairRule
from intent_policy.sim.restricted_action import RestrictedActionConfig


class GeometryExpert(ExpertAgent):
    def __init__(self, snapshot=None):
        self.snapshot = snapshot

    def reset(self, scenario, mapper):
        super().reset(scenario, mapper)
        self.sc = sc = scenario
        m = sc.env.model
        human = [g for g in range(m.ngeom) if m.geom(g).name.startswith('human_seg_')]
        objects = [g for k in sc.on_table for g in sc.object_geom_ids[k]]
        pairs = set(product(sc.robot_geoms, human)) | set(product(human, objects)) | set(product(sc.robot_geoms, objects))
        for a, b in combinations(sc.on_table, 2):
            pairs.update(product(sc.object_geom_ids[a], sc.object_geom_ids[b]))
        # Table contacts are expected for resting objects; this diagnostic does not
        # certify them or any unsupported swept human-pose path.
        self.audit = DemoContactAudit(m, [PairRule(a, b, 0, 0) for a, b in sorted(pairs)])
        self.first = None
        self.observe(m, sc.env.data)
        sc.env.physics_observers.append(self.observe)
        sc.env.pre_physics_observers.append(self.observe)

    def observe(self, model, data):
        if self.first is not None:
            return
        self.audit.observe(model, data)
        if self.audit.first_violation:
            v = self.audit.first_violation
            self.first = {**v, 'human_stage': self.sc.human.stage, 'robot_stage': self.expert.stage}
            if 'geom_a' in v and 'geom_b' in v:
                self.first.update(
                    geom_a_name=model.geom(v['geom_a']).name,
                    geom_b_name=model.geom(v['geom_b']).name,
                    body_a=model.body(model.geom_bodyid[v['geom_a']]).name,
                    body_b=model.body(model.geom_bodyid[v['geom_b']]).name)
            if self.snapshot is not None:
                from PIL import Image
                self.snapshot.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(self.sc.env.render('high', 960, 720)).save(self.snapshot)
                self.first['snapshot'] = str(self.snapshot)

    def act(self, obs):
        if self.first:
            self.sc.fail('geometry_diagnostic_contact_requires_review')
        return super().act(obs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scenario', default='t2_fetch_return')
    ap.add_argument('--seeds', type=int, nargs='+', default=[300000, 300001, 300002, 300003])
    ap.add_argument('--output', required=True)
    ap.add_argument('--snapshots', action='store_true')
    a = ap.parse_args()
    rows = []
    for seed in a.seeds:
        sc = make_scenario(a.scenario, {'expert': {'trigger': 'goal', 'goal_rule': 'observed'},
                                      'success_conditions': {'handover_only': True}})
        try:
            ag = GeometryExpert(Path(a.output).with_suffix('') / f'{seed}.png' if a.snapshots else None)
            rec = run_episode(sc, ag, seed, RestrictedActionConfig.load('configs/controller/continuous_motion.yaml'),
                              keep_trace=False)
            sources = [e['payload'].get('motion_source') for e in rec['events'] if e['event_type'] == 'human_motion_start']
            rows.append(dict(seed=seed, scenario=a.scenario, success=rec['success'], failure=rec['failure'],
                first_contact=ag.first, samples=ag.audit.samples, pairs=len(ag.audit.rules), motion_sources=sources))
            print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
        finally:
            sc.close()
    p = Path(a.output)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(scope='diagnostic_only_no_contact_exemptions', episodes=rows), indent=2))


if __name__ == '__main__':
    main()

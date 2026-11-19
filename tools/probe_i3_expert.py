"""Per-tick probe of the I3 expert (28/30 QĐ-HL): first tick where a NON-selected cake moves, with the phase, selected cake,
TCP, finger contacts. Diagnosis only.
    ./run.sh -m tools.probe_i3_expert --variant switch_before_commit --variants-expert original --seeds 300826 ...
"""
import argparse, json
from pathlib import Path
import numpy as np
from intent_policy.shared.manicast_adapter import load_predictor
from intent_policy.shared.episode import Episode
from intent_policy.scenarios import scenario
from intent_policy.scenarios.i3 import CAKES
from intent_policy.experts import expert_class
from tools.preview import CFG


def run(seed, variant, mode, pred, rest_lean_free=True):
    sc = scenario('I3'); scene, positions, xml, info = sc.build(seed, variant); scene['human']['rest_lean_free'] = rest_lean_free
    ep = Episode('I3', mode, seed, scene, positions, xml, pred, out=None, max_ticks=sc.MAX_TICKS, review_camera=CFG['review_camera'],
                 review_hw=CFG['review_hw'], variant=variant)
    ex = expert_class('I3')(ep); env = ep.env
    ids = {c: ep.m.body(c).id for c in CAKES}; start = {c: ep.pos(c).copy() for c in CAKES}
    first, log, orig = {}, [], env.step

    def step(a):
        s = orig(a)
        for c in CAKES:
            d = float(np.linalg.norm(ep.pos(c)[:2] - start[c][:2]))
            if d > .004 and c not in first and c != ex.selected:
                first[c] = dict(t=round(ep.tick * .05, 2), phase=ep.phase, selected=ex.selected, moved_mm=round(d * 1000, 1),
                                tcp=[round(float(x), 3) for x in env.data.site_xpos[env.tcp]], touch=sorted(env.finger_contacts(ids[c])),
                                attached=None if env.attached is None else ep.m.body(env.attached).name)
        return s
    env.step = step
    ep.event('task_active')
    ep.run(sc.human(ep, info), ex.run(), done=lambda: sc.human_done(ep))
    ok, reason = ex.success()
    if ok and hasattr(sc, 'check'): ok, reason = sc.check(ep, info)
    return dict(seed=seed, A=info['A'], B=info['B'], success=bool(ok), reason=reason, last_phase=ex.phase, first_move=first)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--variant', default='switch_before_commit')
    p.add_argument('--seeds', type=int, nargs='+', required=True); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); a.out.parent.mkdir(parents=True, exist_ok=True); pred = load_predictor()
    for s in a.seeds:
        r = run(s, a.variant, 'forecast', pred)
        with open(a.out, 'a') as f: f.write(json.dumps(r) + '\n')
        print(json.dumps(r), flush=True)


if __name__ == '__main__':
    main()

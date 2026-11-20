"""Re-score captured I3 episodes (tools.audit_policy_reason output) with the old evaluator (v1) and the retry-aware one (v2)
(31/QĐ-HL). Also reports the A-return error and tolerance sensitivity (STAY_TOL stays 10 mm; sensitivity is information only).
    ./run.sh -m tools.rescore_retry outputs/diag/phase3_1120/audit_policy/Mstar_after ...
"""
import json, sys, collections
from pathlib import Path
import numpy as np
from intent_policy.scenarios import i3


class Ep:
    def __init__(self, r):
        self._f = {k: np.array(v) for k, v in r['final'].items()}
        self.initial = {k: np.array(v) for k, v in r['initial'].items()}
        self.events = [{'event': 'robot_grasp', 'object': g[0]} for g in r['grasps']]

    def pos(self, k):
        return self._f[k]


def main():
    for d in sys.argv[1:]:
        R = [json.loads(l) for l in open(Path(d) / 'audit_reasons.jsonl') if l.strip()]
        v1 = v2 = rec = agree = 0
        only_seq = []; ret_err = []
        for r in R:
            info = dict(variant=r['variant'], A=r['info']['A'], B=r['info']['B'])
            ep = Ep(r)
            ok1, _ = i3.check(ep, info, retry_ok=False); ok2, why2 = i3.check(ep, info, retry_ok=True)
            v1 += ok1; v2 += ok2; rec += bool(r['check_ok']); agree += (ok1 == bool(r['check_ok']))
            if ok2 and not ok1: only_seq.append(r['seed'])
            tgt = info['B'] or info['A']
            if info['variant'] == 'switch_after_commit' and info['A'] != tgt and info['A'] in ep.initial:
                ret_err.append(1000 * float(np.linalg.norm(ep._f[info['A']][:2] - ep.initial[info['A']][:2])))
        print(f"{Path(d).name}: n={len(R)} recorded_ok={rec} v1_recomputed={v1} (agrees with record {agree}/{len(R)}) v2_retry={v2}; flipped by retry rule: {only_seq}")
        if ret_err:
            e = np.array(ret_err)
            print('   A-return error mm: median %.0f p75 %.0f max %.0f; share within 10 mm %.2f; within 15 %.2f; 25 %.2f' % (np.median(e), np.percentile(e, 75), e.max(), (e <= 10).mean(), (e <= 15).mean(), (e <= 25).mean()))


if __name__ == '__main__':
    main()

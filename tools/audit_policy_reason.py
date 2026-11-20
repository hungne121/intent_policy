"""Audit wrapper (30/QĐ-HL items 4-5): run tools.eval_policy unchanged, but capture, before `time_limit` overwrites it, the real
`check` reason and the final cake positions of every episode. Writes <out>/audit_reasons.jsonl. Evaluation results are not changed.
    ./run.sh -m tools.audit_policy_reason <same arguments as tools.eval_policy>
"""
import json, sys
from pathlib import Path
import numpy as np
from intent_policy.envs import hrc_env

_orig = hrc_env.HRCEnv._finish
_orig_step = hrc_env.HRCEnv.step
OUT = []
TICKS = {}      # (task, variant, seed) -> per-tick rows: t, tcp, gripper, attached, cake positions (Rx etc.), 31/QĐ-HL


def _finish(self, truncated):
    ok, reason = self._expert_success()
    ep = self.ep
    try:
        pos = {k: [round(float(x), 4) for x in ep.pos(k)] for k in ep.positions}
        init = {k: [round(float(x), 4) for x in ep.initial[k]] for k in ep.positions}
    except Exception:
        pos, init = {}, {}
    OUT.append(dict(task=self.task, variant=self.variant, seed=ep.seed, check_ok=bool(ok), check_reason=reason, truncated=bool(truncated),
                    info={k: v for k, v in self.info.items() if isinstance(v, (str, int, float, type(None)))}, final=pos, initial=init,
                    grasps=[(e['object'], round(e['t'], 2)) for e in ep.events if e['event'] == 'robot_grasp']))
    return _orig(self, truncated)


def _step(self, action):
    out = _orig_step(self, action)
    ep = self.ep
    try:
        att = ep.attached_name()
        row = dict(t=round(ep.tick * 0.05, 2), tcp=[round(float(x), 4) for x in ep.tcp()], grip=round(float(ep.env.data.qpos[ep.env.gid]), 4),
                   att=att, pos={k: [round(float(x), 4) for x in ep.pos(k)] for k in ep.positions})
        TICKS.setdefault((self.task, self.variant, ep.seed), []).append(row)
    except Exception:
        pass
    return out


hrc_env.HRCEnv._finish = _finish
hrc_env.HRCEnv.step = _step
if __name__ == '__main__':
    import runpy
    out = Path(sys.argv[sys.argv.index('--out') + 1]) if '--out' in sys.argv else Path('.')
    sys.argv_orig = list(sys.argv)
    sys.argv = ['tools.eval_policy'] + [a for a in sys.argv[1:] if a != '--ticks']
    try:
        runpy.run_module('tools.eval_policy', run_name='__main__')
    finally:
        out.mkdir(parents=True, exist_ok=True)
        (out / 'audit_reasons.jsonl').write_text('\n'.join(json.dumps(r) for r in OUT) + '\n')
        if '--ticks' in sys.argv_orig:
            (out / 'audit_ticks.json').write_text(json.dumps({f'{k[0]}|{k[1]}|{k[2]}': v for k, v in TICKS.items()}))

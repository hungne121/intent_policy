"""Diagnosis only (not a demonstrator): the G3 gate, upper bound of the forecast benefit per task (docs/PIF.md §8.12).
Rules: observed | cv (constant velocity: the seen wrist velocity over the expert's velocity window, extrapolated) |
early (ManiCast) | oracle1 (true future wrists, 0.1..1.0 s) | oracle_onset (true future of the motion under way only) |
oracle2 (true future, 0.2..2.0 s). Recovered 02/11/2026 from the session log (the /tmp copy was lost on a reboot).
  ORACLE_CONFIG=configs/experiments/g3_extra_tasks.yaml ./run.sh scripts/diag/oracle_bound.py out.jsonl C4 300100:300110
The early expert logic is unchanged; only GoalBelief's forecaster is swapped for a stub that returns the true future
wrist positions recorded in a previous run of the same seed (iterated: early -> oracle pass 1 -> oracle pass 2, the
human reacts to the robot, so the future is taken from the latest pass; residual mismatch reported)."""
import json, sys, time, gc
import numpy as np
from intent_policy.benchmark.runner import run_episode, ExpertAgent
from intent_policy.scenarios.scenario_registry import make_scenario
from intent_policy.utils import controller_config, load_yaml
from intent_policy.intent import goal_belief as G
from scripts.collect_pairs import chunk_overrides
from intent_policy.scenarios.config import _deep_merge
import os
CONFIG = os.environ.get('ORACLE_CONFIG', 'configs/experiments/ten_task_v7.yaml')
CASE = os.environ.get('ORACLE_CASE', 'nominal')
EXTRA = json.loads(os.environ.get('ORACLE_EXTRA', '{}'))

STATE = dict(table=None, H=1.0, rec=None, onset=False, cv=False)
_orig_init, _orig_step = G.GoalBelief.__init__, G.GoalBelief.step


class Stub:
    def __init__(self, sc): self.sc = sc
    def to_comad(self, d): return np.zeros((5, 7, 3)), np.zeros((5, 3))
    def from_comad(self, x): return x
    def forecast(self, mk, waist, t_in, t_out):
        T, L, R, segs = STATE['table']
        now = self.sc.time
        t = now + np.asarray(t_out) * STATE['H']
        if STATE.get('onset'):            # only the motion under way: at rest the forecast is the current pose
            act = [b for a, b in segs if a <= now + 1e-9 and now <= b + 1e-9]
            t = np.minimum(t, act[0]) if act else np.full_like(t, now)
        p = np.zeros((1, len(t_out), 7, 3))
        for j in range(3):
            p[0, :, 6, j] = np.interp(t, T, R[:, j])
            p[0, :, 5, j] = np.interp(t, T, L[:, j])
        return p


class CVStub:
    """Constant velocity (Schöller et al. 2020; the CVM baseline of ManiCast, Kedia et al. 2023): each wrist moves on
    at its seen velocity over the expert's velocity window (expert.decision.velocity_window_s, else 0.2 s)."""
    def __init__(self, gb):
        self.gb = gb
        d = gb.sc.cfg.expert.get('decision') or {}
        self.n = max(1, int(round(float(d.get('velocity_window_s', 0.2)) * gb.fps)))
    def to_comad(self, d): return np.zeros((5, 7, 3)), np.zeros((5, 3))
    def from_comad(self, x): return x
    def forecast(self, mk, waist, t_in, t_out):
        kp = self.gb.kp
        n = min(self.n, len(kp) - 1)
        p = np.zeros((1, len(t_out), 7, 3))
        for m, j in ((6, 5), (5, 4)):
            now = np.asarray(kp[-1][j], float)
            v = (now - np.asarray(kp[-1 - n][j], float)) * self.gb.fps / n if n > 0 else np.zeros(3)
            p[0, :, m] = now + np.asarray(t_out)[:, None] * v
        return p


def init(self, sc, rule, speed, travel_dz):
    _orig_init(self, sc, rule, speed, travel_dz)
    if rule == 'early' and STATE['cv']:
        self.fc = CVStub(self)
    elif rule == 'early' and STATE['table'] is not None:
        self.fc = Stub(sc)


def step(self):
    _orig_step(self)
    if STATE['rec'] is not None and self.kp and (self.tick - 1) % self.every == 0:
        STATE['rec'].append((self.sc.time, self.kp[-1][4].copy(), self.kp[-1][5].copy()))


G.GoalBelief.__init__, G.GoalBelief.step = init, step


def run(exp, tid, seed, rule, table=None, H=1.0, onset=False, cv=False):
    task = exp['tasks'][tid]
    ov = chunk_overrides(task, task['cases'][CASE], 'train', rule)
    if exp['data'].get('paired_source_selection'):
        ov = _deep_merge(ov, {'human_behavior': {'motion_bank': {'paired_source_selection': True}}})
    ov = _deep_merge(ov, EXTRA)
    sc = make_scenario(task['scenario'], ov)
    STATE.update(table=table, H=H, rec=[], onset=onset, cv=cv)
    try:
        rec = run_episode(sc, ExpertAgent(), seed, controller_config(exp), keep_trace=False)
    finally:
        sc.close()
        del sc
        gc.collect()
    r = STATE['rec']
    segs, open_ = [], {}
    for e in rec['events']:
        if e['event_type'] == 'human_motion_start': open_[e['payload'].get('label')] = e['timestamp']
        elif e['event_type'] == 'human_motion_end' and e['payload'].get('label') in open_:
            segs.append((open_.pop(e['payload'].get('label')), e['timestamp']))
    tab = (np.array([x[0] for x in r]), np.array([x[1] for x in r]), np.array([x[2] for x in r]), segs) if r else None
    ev = []
    for e in rec['events']:
        e = dict(e); pl = dict(e.get('payload') or {}); pl.pop('motion_source', None); e['payload'] = pl; ev.append(e)
    return dict(task=tid, seed=seed, rule=rule, H=H, success=rec['success'], dur=rec['duration_s'], events=ev), tab


def mismatch(a, b):
    if a is None or b is None: return None
    t = b[0]
    return float(max(np.abs(np.interp(t, a[0], a[2][:, j]) - b[2][:, j]).max() for j in range(3)))


out, tasks, seeds = sys.argv[1], sys.argv[2].split(','), (range(*map(int, sys.argv[3].split(':'))) if ':' in sys.argv[3] else [int(s) for s in sys.argv[3].split(',')])
exp = load_yaml(CONFIG)
with open(out, 'a') as f:
    for tid in tasks:
        for seed in seeds:
            t0 = time.time()
            r, _ = run(exp, tid, seed, 'observed'); f.write(json.dumps(r, default=str) + '\n')
            r, _ = run(exp, tid, seed, 'early', cv=True); r['rule'] = 'cv'; f.write(json.dumps(r, default=str) + '\n')
            r, tab = run(exp, tid, seed, 'early'); f.write(json.dumps(r, default=str) + '\n')
            line = [tid, seed, 'early', r['dur']]
            for name, H, onset in (('oracle1', 1.0, False), ('oracle_onset', 1.0, True), ('oracle2', 2.0, False)):
                cur = tab
                for k in range(3):
                    r, new = run(exp, tid, seed, 'early', table=cur, H=H, onset=onset)
                    mm = mismatch(cur, new); cur = new
                    if mm is not None and mm < 0.005: break
                r['rule'] = name; r['passes'] = k + 1; r['mismatch_m'] = mm
                f.write(json.dumps(r, default=str) + '\n'); f.flush()
                line += [r['rule'], r['dur'], r['success'], 'mm', None if mm is None else round(mm, 3)]
            print(*line, round(time.time() - t0), flush=True)

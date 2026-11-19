"""The 8 evaluation metrics of 10/QĐ-KT with the event semantics of 11/QĐ-KT (18/11/2026).

HRIBench-style metrics adapted for this project; not an exact reproduction of HRIBench.

Formulas (verbatim, 10/QĐ-KT; source: Slide_HRI_English_v10_Learning.pdf, slide 10): TSync = |t_R,ready − t_H,ready|,
CT = t_success − t_start, HCS = N_safe / N, CSR = N_success / N, Rsp = t_response − t_trigger,
OC = n_ordered / n_required, DSR = N_recovered / N_disrupted, CFR = N_contactfree / N.

Input: one metric record per episode, produced ONLY by agent H's event detector (intent_policy/shared/events.py,
`metric_record`) and stored by intent_policy/envs/hrc_env.py under `metrics` in episodes.jsonl. This module never
builds events itself (11/QĐ-KT, integration): it reads the record. Record schema (METRIC_SPEC_2026-11-18.md §5):

  detector_version, schema_version          refused for official output if absent / not agent H's (check_official)
  t_start, t_success (None if failed), success
  rounds: [{round_id, t_R_ready, t_H_ready, t_trigger, t_response}]      C1: paired by explicit round id
  steps_required[], steps_done[[name, t]], step_predecessors {name: [names]} (partial order of the scenario script)
      or steps_ordered (True = total order in steps_required; False = independent steps)            C3
  disrupted, recovered, human_robot_contact (bool)
  safety_audit: {min_distance_m, closest_pair, interaction_phase, exemption_reason, contact_force_n, penetration_m,
                 violations: [{kind: contact|clearance, t, pair, phase, distance_m, exempt, exemption_reason}]}   C2
  anticipation (optional, e.g. useful-action onset, AM): copied to the episode output, never mixed into Rsp       C4

Rules: 10/QĐ-KT 3-6 (n/a never 0; time metrics averaged over applicable episodes; ratios as sums of numerator /
denominator with both reported; failed episodes stay in CSR, HCS, CFR; DSR denominator = disrupted episodes only) and
11/QĐ-KT C1-C6:
  C1  TSync, Rsp per interaction round (matched by round_id), episode value = mean over valid rounds; valid / missing
      rounds are reported.
  C2  N_safe = no human-robot contact AND no non-exempt violation in the detector's safety audit; no distance rule
      is computed here. Sensitivity: `hcs_under(...)` recomputes HCS under alternative exemption definitions given as
      config (clearance violations only; contacts are never exempted).
  C3  OC on the dependency graph (12/QĐ-KT C21): total order = longest correct prefix of the completed sequence;
      partial order = a step is valid iff done and every predecessor is valid and done earlier; no LCS.
  C4  Rsp only from the trigger onward: t_response < t_trigger is an invalid round (flag), never a negative value;
      no response -> n/a + missing_response flag.
  C5  no search-radius shortcut: HCS is n/a without the detector's audit.
  C6  groups I = Instructor, C = Collaborator, W = Cowork (project role; NOT HRIBench's Intruder).
Aggregation: episode -> case -> task -> group / evaluation set (N1/N5/N6) -> overall; macro over tasks = main,
pooled = secondary; 95 % percentile bootstrap resampling scenarios (task, variant, seed) within each task;
paired differences over matched scenarios with a paired bootstrap of the same structure.
"""
from __future__ import annotations

import fnmatch
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HRIBENCH_NOTE = 'HRIBench-style metrics adapted for this project; not an exact reproduction of HRIBench.'
ROLE_GROUPS = {'I': 'Instructor', 'C': 'Collaborator', 'W': 'Cowork'}   # 11/QĐ-KT C6; Cowork is not HRIBench's Intruder

SLIDE10 = (     # 10/QĐ-KT verbatim: formula, unit (None = ratio), better direction
    dict(name='TSync', full='Human–robot readiness gap', formula='TSync = |t_R,ready − t_H,ready|', unit='s',
         better='lower', kind='time'),
    dict(name='CT', full='Completion time', formula='CT = t_success − t_start', unit='s', better='lower', kind='time'),
    dict(name='HCS', full='Human contact safety (no contact, no safety-distance violation)',
         formula='HCS = N_safe / N', unit=None, better='higher', kind='ratio'),
    dict(name='CSR', full='Task success rate', formula='CSR = N_success / N', unit=None, better='higher', kind='ratio'),
    dict(name='Rsp', full='Response latency to a human cue', formula='Rsp = t_response − t_trigger', unit='s',
         better='lower', kind='time'),
    dict(name='OC', full='Steps completed in the right order', formula='OC = n_ordered / n_required', unit=None,
         better='higher', kind='ratio'),
    dict(name='DSR', full='Disruption recovery rate', formula='DSR = N_recovered / N_disrupted', unit=None,
         better='higher', kind='ratio'),
    dict(name='CFR', full='Contact-free rate', formula='CFR = N_contactfree / N', unit=None, better='higher',
         kind='ratio'),
)
METRICS = tuple(m['name'] for m in SLIDE10)
KIND = {m['name']: m['kind'] for m in SLIDE10}
REQUIRED_KEYS = ('detector_version', 'schema_version', 't_start', 't_success', 'success', 'rounds', 'steps_required',
                 'steps_done', 'disrupted', 'recovered', 'human_robot_contact', 'safety_audit')


# ---------------------------------------------------------------------------------------------- input / schema
def metric_input(rec: dict) -> dict:
    """The metric record of an episodes.jsonl line: under `metrics` (hrc_env) or `metric_record`, else the line."""
    for k in ('metrics', 'metric_record'):
        v = rec.get(k)
        if isinstance(v, dict) and 't_start' in v:
            return v
    return rec


def expected_versions():
    """(DETECTOR_VERSION, schema) published by agent H in intent_policy/shared/events.py; None if not published."""
    try:
        from intent_policy.shared import events
    except Exception:
        return None, None
    return getattr(events, 'DETECTOR_VERSION', None), getattr(events, 'METRIC_SCHEMA_VERSION', None)


def check_official(recs, expected_detector=None, expected_schema=None) -> list[str]:
    """Problems that forbid an OFFICIAL benchmark output (11/QĐ-KT integration); [] = accepted."""
    if expected_detector is None and expected_schema is None:
        expected_detector, expected_schema = expected_versions()
    probs = []
    if not expected_detector:
        probs.append('agent H has not published DETECTOR_VERSION in intent_policy/shared/events.py')
    seen = set()
    for i, rec in enumerate(recs):
        r = metric_input(rec)
        miss = [k for k in REQUIRED_KEYS if k not in r]
        if 'step_predecessors' not in r and 'steps_ordered' not in r:
            miss.append('step_predecessors|steps_ordered')
        if miss:
            probs.append(f'record {i} ({rec.get("task")}/{rec.get("seed")}): missing {miss}')
            continue
        seen.add((r['detector_version'], r['schema_version']))
        if expected_detector and r['detector_version'] != expected_detector:
            probs.append(f'record {i}: detector_version {r["detector_version"]} != {expected_detector}')
        if expected_schema and r['schema_version'] != expected_schema:
            probs.append(f'record {i}: schema_version {r["schema_version"]} != {expected_schema}')
        if safe(r) is None:
            probs.append(f'record {i}: safety unresolved; cannot publish official HCS')
    if len(seen) > 1:
        probs.append(f'mixed detector/schema versions {sorted(seen)}')
    return probs


def _num(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def rounds_of(r) -> tuple[list[dict], list[str]]:
    """The episode's interaction rounds (C1) and flags. Legacy v1 records (parallel lists, no round ids) are read
    with round_id = list index and flagged `legacy_rounds_by_index` (never accepted as official)."""
    flags = []
    if 'rounds' in r:
        rounds = list(r['rounds'] or [])
    else:
        keys = ('t_R_ready', 't_H_ready', 't_trigger', 't_response')
        n = max((len(r.get(k) or []) for k in keys), default=0)
        rounds = [{'round_id': i, **{k: (r.get(k) or [])[i] if i < len(r.get(k) or []) else None for k in keys}}
                  for i in range(n)]
        if n:
            flags.append('legacy_rounds_by_index')
    ids = [x.get('round_id') for x in rounds]
    if any(i is None for i in ids) or len(set(map(str, ids))) != len(ids):
        flags.append('round_id_missing_or_duplicate')
    return rounds, flags


# ---------------------------------------------------------------------------------------------- per-metric
def tsync(r) -> dict:
    """TSync = |t_R,ready − t_H,ready| per round; episode = mean over valid rounds (C1)."""
    rounds, flags = rounds_of(r)
    if 'round_id_missing_or_duplicate' in flags:
        return dict(value=None, valid_rounds=[], missing_rounds=[x.get('round_id') for x in rounds], flags=flags)
    vals, ok, miss = [], [], []
    for x in rounds:
        tr, th = _num(x.get('t_R_ready')), _num(x.get('t_H_ready'))
        if tr is None or th is None:
            miss.append(x['round_id'])
        else:
            vals.append(abs(tr - th)); ok.append(x['round_id'])
    return dict(value=float(np.mean(vals)) if vals else None, valid_rounds=ok, missing_rounds=miss, flags=flags)


def rsp(r) -> dict:
    """Rsp = t_response − t_trigger per round with a trigger, from the trigger onward only (C4)."""
    rounds, flags = rounds_of(r)
    if 'round_id_missing_or_duplicate' in flags:
        return dict(value=None, valid_rounds=[], missing_response=[], invalid_rounds=[], flags=flags)
    vals, ok, missing, invalid = [], [], [], []
    for x in rounds:
        tt, tresp = _num(x.get('t_trigger')), _num(x.get('t_response'))
        if tt is None:
            continue                                     # no human cue in this round: Rsp not defined
        if tresp is None:
            missing.append(x['round_id'])
        elif tresp < tt:
            invalid.append(x['round_id'])                # detector contract broken; never a negative Rsp
        else:
            vals.append(tresp - tt); ok.append(x['round_id'])
    return dict(value=float(np.mean(vals)) if vals else None, valid_rounds=ok, missing_response=missing,
                invalid_rounds=invalid, flags=flags)


def ct(r):
    """CT = t_success − t_start, successful episodes only."""
    if not r.get('success'):
        return None
    t0, t1 = _num(r.get('t_start')), _num(r.get('t_success'))
    return None if t0 is None or t1 is None else t1 - t0


def predecessors(r) -> dict | None:
    """Direct required predecessors of each required step (the dependency graph): step_predecessors (scenario
    partial order), else a chain = total order in steps_required (steps_ordered True), or none (steps_ordered
    False). None = order not declared."""
    req = list(r.get('steps_required') or [])
    if r.get('step_predecessors') is not None:
        return {s: list(r['step_predecessors'].get(s, [])) for s in req}
    if r.get('steps_ordered') is True:
        return {s: req[i - 1:i] for i, s in enumerate(req)}
    if r.get('steps_ordered') is False:
        return {s: [] for s in req}
    return None


def ordered_steps(r) -> dict:
    """(n_ordered, n_required) for OC on the dependency graph (11/QĐ-KT C3, 12/QĐ-KT C21), no LCS.
    Total order (steps_ordered True, no step_predecessors): the completed sequence (by first completion time) is
    checked from the start; n_ordered = length of its longest prefix equal to steps_required[:k] with strictly
    increasing times, so after the first out-of-order / missing step nothing counts.
    Partial order (step_predecessors) / independent steps: a step is valid iff done and every direct predecessor is
    itself valid and was done strictly earlier.
    Same tick (14/QĐ-KT item 3): a violation only between (transitively) dependent steps with no recorded substep
    order (`steps_done` entry [name, t, seq] / {name, t, seq}); independent steps may share a tick."""
    req = list(r.get('steps_required') or [])
    if not req:
        return dict(num=0, den=0, flags=[])
    if len(set(req)) != len(req):
        return dict(num=0, den=0, flags=['duplicate_step_names'])
    pred = predecessors(r)
    if pred is None:
        return dict(num=0, den=0, flags=['step_order_not_declared'])
    t = {}                                     # name -> (first completion time, substep order or None)
    for s in r.get('steps_done') or []:
        if isinstance(s, dict):
            name, ts, seq = s.get('name'), s.get('t'), s.get('seq')
        else:
            name, ts, seq = s[0], s[1], (s[2] if len(s) > 2 else None)
        if _num(ts) is not None and name not in t:
            t[name] = (float(ts), _num(seq))

    def before(p, s):
        """p done before s: earlier tick, or the same tick with a recorded substep order p < s (14/QĐ-KT item 3)."""
        (tp, qp), (ts_, qs) = t[p], t[s]
        return tp < ts_ or (tp == ts_ and qp is not None and qs is not None and qp < qs)

    if r.get('step_predecessors') is None and r.get('steps_ordered') is True:
        k, prev = 0, None
        seq = sorted((n for n in t if n in req), key=lambda n: (t[n][0], math.inf if t[n][1] is None else t[n][1]))
        for name in seq:
            if k < len(req) and name == req[k] and (prev is None or before(prev, name)):
                k, prev = k + 1, name
            else:
                break
        return dict(num=k, den=len(req), flags=[])
    valid = {}

    def ok(s, stack=()):
        if s not in valid:
            if s in stack:
                raise ValueError(f'cyclic step dependencies at {s}')
            valid[s] = s in t and all(p in t and before(p, s) and ok(p, stack + (s,)) for p in pred.get(s, []))
        return valid[s]
    n = sum(ok(s) for s in req)
    return dict(num=n, den=len(req), flags=[])


def _match(pair, phase, rule) -> bool:
    pairs = rule.get('pairs')
    phases = rule.get('phases')
    ok_pair = pairs is None or (pair is not None and any(
        fnmatch.fnmatch(str(pair[0]), a) and fnmatch.fnmatch(str(pair[1]), b) for a, b in pairs))
    ok_phase = phases is None or phase in phases
    return ok_pair and ok_phase


def safe(r, exemptions: dict | None = None):
    """N_safe indicator (C2): no human-robot contact and no non-exempt violation of the detector's safety audit.
    `exemptions` (sensitivity analysis) replaces the detector's own exemption decision for CLEARANCE violations:
    {exempt: [{pairs: [[robot_glob, human_glob]], phases: [...]}], max_distance_m: optional}. None = unknown."""
    c, audit = r.get('human_robot_contact'), r.get('safety_audit')
    if c is None or not isinstance(audit, dict) or 'violations' not in audit:
        return None
    if c:
        return False
    unresolved = False
    for v in audit['violations'] or []:
        if v.get('kind') == 'contact':
            return False                               # contacts are never exempted (11/QĐ-KT C2)
        if exemptions is None:
            if not v.get('exempt'):
                if v.get('verified') is False:
                    unresolved = True
                else:
                    return False
        else:
            d = _num(v.get('distance_m'))
            lim = exemptions.get('max_distance_m')
            if lim is not None and d is not None and d >= lim:
                continue                               # not a violation under this (stricter-only) threshold
            if not any(_match(v.get('pair'), v.get('phase'), rule) for rule in exemptions.get('exempt', [])):
                if v.get('verified') is False:
                    unresolved = True
                else:
                    return False
    return None if unresolved else True


def contact_consistent(r):
    """HCS / CFR must agree with the contact detector: contact flag <-> a contact entry in the audit."""
    audit = r.get('safety_audit')
    if not isinstance(audit, dict) or r.get('human_robot_contact') is None:
        return None
    has = any(v.get('kind') == 'contact' for v in audit.get('violations') or [])
    return bool(r['human_robot_contact']) == has


# ---------------------------------------------------------------------------------------------- episode level
def episode_metrics(rec: dict) -> dict:
    """The 8 metrics of one episode as {value, num, den} (time: num=value, den=1 when applicable; den=0 -> n/a)
    plus the C1-C4 diagnostics."""
    r = metric_input(rec)
    out = {}

    def put(name, num, den):
        out[name] = dict(value=None if not den else num / den, num=float(num) if den else 0., den=den)

    ts, rs, oc = tsync(r), rsp(r), ordered_steps(r)
    put('TSync', ts['value'] or 0., int(ts['value'] is not None))
    v = ct(r); put('CT', v or 0., int(v is not None))
    s = safe(r); put('HCS', float(bool(s)), int(s is not None))
    put('CSR', float(bool(r.get('success'))), int(r.get('success') is not None))
    put('Rsp', rs['value'] or 0., int(rs['value'] is not None))
    put('OC', oc['num'], oc['den'])
    if r.get('disrupted'):
        ok = bool(r.get('recovered')) and bool(r.get('success')) and r.get('human_robot_contact') is False
        put('DSR', float(ok), 1)
    else:
        put('DSR', 0., 0)                               # no disruption (or unknown): outside the DSR denominator
    c = r.get('human_robot_contact'); put('CFR', float(c is False), int(c is not None))
    diag = dict(tsync_valid_rounds=len(ts['valid_rounds']), tsync_missing_rounds=ts['missing_rounds'],
                rsp_valid_rounds=len(rs['valid_rounds']), missing_response=rs['missing_response'],
                rsp_invalid_rounds=rs['invalid_rounds'], round_flags=sorted(set(ts['flags'] + rs['flags'])),
                oc_flags=oc['flags'], contact_audit_consistent=contact_consistent(r))
    return dict(metrics=out, diagnostics=diag)


def episode_row(rec: dict) -> dict:
    """One line of results_per_episode.jsonl: identification, safety audit (C2), anticipation (C4), metrics."""
    r = metric_input(rec)
    task = str(rec.get('task'))
    row = dict(run=rec.get('run'), model=rec.get('model'), step=rec.get('step'), n_action_steps=rec.get('n_action_steps'),
               task=task, variant=rec.get('variant', 'nominal'), task_group=task[:1],
               role=ROLE_GROUPS.get(task[:1]), eval_set=rec.get('group'), seed=rec.get('seed'),
               detector_version=r.get('detector_version'), schema_version=r.get('schema_version'),
               success=r.get('success'), human_robot_contact=r.get('human_robot_contact'),
               disrupted=r.get('disrupted'), recovered=r.get('recovered'),
               safety_audit=r.get('safety_audit'), anticipation=r.get('anticipation'), safety_detail=safety_detail(r))
    row.update(episode_metrics(rec))
    return row


def hcs_under(rows, variants: dict) -> dict:
    """Sensitivity analysis (C2): HCS per variant {name: exemption definition}, pooled and per task, numerator /
    denominator. Variants come from a config declared before evaluation; never tuned on a model's results."""
    out = {}
    for name, ex in variants.items():
        per = {}
        for row in rows:
            s = safe(row, ex)
            t = per.setdefault(row['task'], [0, 0])
            if s is not None:
                t[0] += int(s); t[1] += 1
        num, den = sum(v[0] for v in per.values()), sum(v[1] for v in per.values())
        task_vals = [v[0] / v[1] for v in per.values() if v[1]]
        out[name] = dict(pooled=num / den if den else None, numerator=num, denominator=den,
                         macro=float(np.mean(task_vals)) if task_vals else None,
                         per_task={k: dict(numerator=v[0], denominator=v[1]) for k, v in sorted(per.items())})
    return out


# ---------------------------------------------------------------------------------------------- safety detail (15/QĐ-KT D21-D24)
SAFETY_CLASSES = ('collision', 'proximity_violation', 'unverified_distance', 'clear')
_DT_DEFAULT = 0.05


def _unverified_windows(audit):
    """[t, t_end] of non-exempt clearance violations that are NOT verified (only a lower bound of the distance)."""
    return [(float(v['t']), float(v.get('t_end', v['t']))) for v in audit.get('violations') or []
            if v.get('kind') == 'clearance' and not v.get('exempt') and v.get('verified') is False and v.get('t') is not None]


def _span_ticks(v, dt):
    return int(round((float(v.get('t_end', v['t'])) - float(v['t'])) / dt)) + 1


def safety_detail(r) -> dict:
    """Extra reading of the detector's safety audit (15/QĐ-KT D21-D22); HCS itself is unchanged (`safe`).
    cls (priority order): collision = a contact violation or the contact flag; proximity_violation = a VERIFIED
    non-exempt clearance violation (d < S_p) without contact; unverified_distance = only non-exempt violations whose
    distance is a lower bound (verified False); clear = none (exempt close-interaction ticks do not count).
    Fields (None = not available in the record, never 0): n_ticks, dt_s, violation_ticks (verified, non-exempt),
    unverified_ticks, exempt_ticks, violation_duration_s = violation_ticks * dt, violation_ratio = violation_ticks /
    n_ticks, min_verified_clearance_m (min d_nonexempt over ticks that are not unverified), max_clearance_deficit_m
    (max (S_p - d) over verified violation ticks), contact_force_n / penetration_m (max over contact violations;
    clearance violations carry no force: ISO/TS 15066 force / pressure is not modelled), worst_pair (verified
    non-exempt pair with the most violating ticks; contact pair first), source = 'ticks' | 'violations'.
    Without `ticks` (stripped records) counts come from the grouped violations (t..t_end); ratio and min clearance
    are n/a and the deficit is the group's closest value."""
    audit = r.get('safety_audit')
    if not isinstance(audit, dict) or 'violations' not in audit or r.get('human_robot_contact') is None:
        return dict(cls=None)
    viol = audit.get('violations') or []
    contacts = [v for v in viol if v.get('kind') == 'contact']
    cl = [v for v in viol if v.get('kind') == 'clearance' and not v.get('exempt')]
    cl_ver = [v for v in cl if v.get('verified') is not False]
    cols = audit.get('ticks_columns') or ['t', 'd_nonexempt_m', 'S_p_m', 'close_interaction', 'd_all_m']
    ticks = audit.get('ticks')
    ix = {c: i for i, c in enumerate(cols)}
    out = dict(cls=None, source='ticks' if ticks else 'violations', n_ticks=None, dt_s=None, violation_ticks=None,
               unverified_ticks=None, exempt_ticks=audit.get('exempt_clearance_ticks'), violation_duration_s=None,
               violation_ratio=None, min_verified_clearance_m=None, max_clearance_deficit_m=None, contact_force_n=None,
               penetration_m=None, contact_force_missing=False, worst_pair=None, n_contacts=len(contacts))
    dt = _DT_DEFAULT
    unv = _unverified_windows(audit)
    if ticks:
        ts = [float(x[ix['t']]) for x in ticks]
        gaps = sorted(b - a for a, b in zip(ts, ts[1:]) if b > a)
        dt = gaps[len(gaps) // 2] if gaps else _DT_DEFAULT
        vt = ut = 0
        defi, mind = [], []
        for x in ticks:
            t, d, sp = float(x[ix['t']]), x[ix['d_nonexempt_m']], x[ix['S_p_m']]
            if d is None:
                continue
            in_unv = any(a - 1e-6 <= t <= b + 1e-6 for a, b in unv)
            if not in_unv:
                mind.append(float(d))
            if sp is not None and d < sp:
                if in_unv:
                    ut += 1
                else:
                    vt += 1; defi.append(float(sp) - float(d))
        out.update(n_ticks=len(ticks), violation_ticks=vt, unverified_ticks=ut, violation_ratio=vt / len(ticks),
                   min_verified_clearance_m=min(mind) if mind else None,
                   max_clearance_deficit_m=max(defi) if defi else None)
    else:
        vt = sum(_span_ticks(v, dt) for v in cl_ver)
        ut = sum(int(round((b - a) / dt)) + 1 for a, b in unv)
        defi = [float(v['threshold_m']) - float(v['distance_m']) for v in cl_ver
                if v.get('threshold_m') is not None and v.get('distance_m') is not None]
        out.update(violation_ticks=vt, unverified_ticks=ut, max_clearance_deficit_m=max(defi) if defi else None)
    out.update(dt_s=dt, violation_duration_s=out['violation_ticks'] * dt)
    pair_ticks = {}
    for v in cl_ver:
        k = tuple(v.get('pair') or [])
        pair_ticks[k] = pair_ticks.get(k, 0) + _span_ticks(v, dt)
    if contacts:
        forces = [_num(v.get('force_n')) for v in contacts]
        pens = [_num(v.get('penetration_m')) for v in contacts]
        out['contact_force_n'] = max((f for f in forces if f is not None), default=None)
        out['penetration_m'] = max((p for p in pens if p is not None), default=None)
        out['contact_force_missing'] = out['contact_force_n'] is None
        out['worst_pair'] = list(contacts[0].get('pair') or [])
    elif pair_ticks:
        out['worst_pair'] = list(max(pair_ticks, key=pair_ticks.get))
    if r.get('human_robot_contact') or contacts:
        out['cls'] = 'collision'
    elif cl_ver:
        out['cls'] = 'proximity_violation'
    elif cl:
        out['cls'] = 'unverified_distance'
    else:
        out['cls'] = 'clear'
    return out


def _agg_safety(rows) -> dict:
    """Per group of episode rows: class counts, durations, ratios, clearances, force, worst pairs, n/a, tasks, cases.
    `pooled` = over the episodes where the field exists; `macro` = mean of the per-task values (same reducer)."""
    rows = list(rows)
    have = [(r, r['safety_detail']) for r in rows if (r.get('safety_detail') or {}).get('cls')]
    out = dict(n_episodes=len(rows), n_with_audit=len(have), n_na_audit=len(rows) - len(have),
               n_tasks=len({r['task'] for r in rows}), n_cases=len({(r['task'], r['variant']) for r in rows}),
               classes={c: sum(d['cls'] == c for _, d in have) for c in SAFETY_CLASSES})

    def stat(key, fn=np.mean):
        vals = [(r['task'], r['variant'], d[key]) for r, d in have if d.get(key) is not None]
        per = {}
        for t, _, v in vals:
            per.setdefault(t, []).append(v)
        return dict(pooled=float(fn([v for *_, v in vals])) if vals else None,
                    macro=float(np.mean([fn(v) for v in per.values()])) if per else None,
                    n=len(vals), n_na=len(rows) - len(vals), n_tasks=len(per),
                    n_cases=len({(t, c) for t, c, _ in vals}))
    out['violation_duration_s'] = stat('violation_duration_s')
    out['violation_ratio'] = stat('violation_ratio')
    out['max_clearance_deficit_m'] = stat('max_clearance_deficit_m', np.max)     # worst episode (pooled) / mean of task worst
    out['min_verified_clearance_m'] = stat('min_verified_clearance_m', np.min)   # closest episode / mean of task closest
    out['contact_force_n'] = dict(stat('contact_force_n', np.max), n_contact_episodes=sum(d['cls'] == 'collision' for _, d in have),
                                  n_force_missing=sum(bool(d.get('contact_force_missing')) for _, d in have))
    out['penetration_m'] = stat('penetration_m', np.max)
    pairs = {}
    for _, d in have:
        if d.get('worst_pair'):
            k = ' x '.join(d['worst_pair']); pairs[k] = pairs.get(k, 0) + 1
    out['worst_pairs'] = dict(sorted(pairs.items(), key=lambda kv: -kv[1])[:5])
    return out


def safety_detail_summary(rows) -> dict:
    """15/QĐ-KT D21-D24 for ONE run: overall, per task, per case (task/variant). Never replaces HCS / CFR / the 8 metrics."""
    rows = list(rows)
    return dict(overall=_agg_safety(rows),
                task={k: _agg_safety(v) for k, v in _group(rows, lambda r: r['task']).items()},
                case={k: _agg_safety(v) for k, v in _group(rows, lambda r: f"{r['task']}/{r['variant']}").items()})


def _rule_safe(row, name):
    """HCS indicator of ONE declared variant of configs/hrc/events_sensitivity.yaml, from the audit ticks (15/QĐ-KT D19:
    thresholds / variants unchanged). None = not computable (no ticks / lower-bound column absent)."""
    r = metric_input(row)
    a = r.get('safety_audit')
    if not isinstance(a, dict) or r.get('human_robot_contact') is None:
        return None
    contact = bool(r['human_robot_contact']) or any(v.get('kind') == 'contact' for v in a.get('violations') or [])
    if name == 'contact_only':
        return not contact
    if name in ('ssm_exempt_verified', 'ssm_exempt'):
        return safe(r)
    ticks = a.get('ticks')
    if not ticks:
        return None
    cols = a.get('ticks_columns') or ['t', 'd_nonexempt_m', 'S_p_m', 'close_interaction', 'd_all_m']
    ix = {c: i for i, c in enumerate(cols)}
    if contact:
        return False
    if name == 'ssm_exempt_lowerbound':
        col = 'd_nonexempt_lower_bound_m'
        return None if col not in ix else not any(x[ix[col]] is not None and x[ix[col]] < x[ix['S_p_m']] for x in ticks)
    if name == 'ssm_no_exemption':
        return not any(x[ix['d_all_m']] is not None and x[ix['d_all_m']] < x[ix['S_p_m']] for x in ticks)
    if name.startswith('static_'):
        lim = float(name.split('_', 1)[1])
        return not any(x[ix['d_nonexempt_m']] is not None and x[ix['d_nonexempt_m']] < lim for x in ticks)
    return None


def hcs_rule_variants(rows, names) -> dict:
    """HCS per declared rule variant (events_sensitivity.yaml), pooled and per task, numerator / denominator, n/a."""
    out = {}
    for name in names:
        per = {}
        for row in rows:
            s = _rule_safe(row, name)
            t = per.setdefault(row['task'], [0, 0])
            if s is not None:
                t[0] += int(s); t[1] += 1
        num, den = sum(v[0] for v in per.values()), sum(v[1] for v in per.values())
        tv = [v[0] / v[1] for v in per.values() if v[1]]
        out[name] = dict(pooled=num / den if den else None, numerator=num, denominator=den, n_na=len(rows) - den,
                         macro=float(np.mean(tv)) if tv else None,
                         per_task={k: dict(numerator=v[0], denominator=v[1]) for k, v in sorted(per.items())})
    return out


# ---------------------------------------------------------------------------------------------- aggregation
def condition(r) -> str:
    """Evaluation condition reported separately (12/QĐ-KT E28): forecast_fault (N5), switch_before_commit /
    switch_after_commit (N6 variants), nominal (N1; other N1 variants keep their name)."""
    return 'forecast_fault' if str(r.get('eval_set')) == 'N5' else str(r.get('variant'))


def _scenario(r):
    return (r['task'], r['variant'], r['seed'])


def _clusters(rows):
    """Bootstrap structure: tasks (strata) -> scenarios (task, variant, seed; resampled units) -> row indices."""
    strata = {}
    for i, r in enumerate(rows):
        strata.setdefault(str(r['task']), {}).setdefault(_scenario(r), []).append(i)
    return {t: list(sc.values()) for t, sc in sorted(strata.items())}


def _draw(clusters, rng):
    """One bootstrap draw: per task, scenarios resampled with replacement -> {task: row indices}."""
    out = {}
    for t, sc in clusters.items():
        pick = rng.integers(0, len(sc), len(sc))
        out[t] = np.array([i for k in pick for i in sc[k]], int)
    return out


def _stats(num, den, by_task):
    pooled_idx = np.concatenate(list(by_task.values())) if by_task else np.zeros(0, int)
    d = den[pooled_idx].sum()
    pooled = np.nan if d == 0 else num[pooled_idx].sum() / d
    vals = [num[ix].sum() / den[ix].sum() for ix in by_task.values() if den[ix].sum() > 0]
    return pooled, (float(np.mean(vals)) if vals else np.nan), len(vals)


def summarize(rows, n_boot=2000, seed=0) -> dict:
    """Macro over tasks (main) and pooled (secondary) value of each metric, numerator / denominator, eligible
    episodes, n/a, contributing tasks, 95 % percentile bootstrap CI (scenarios resampled within tasks)."""
    rows = list(rows)
    out = dict(n_episodes=len(rows), n_success=sum(bool(r['success']) for r in rows),
               n_tasks=len({r['task'] for r in rows}), n_cases=len({(r['task'], r['variant']) for r in rows}))
    if not rows:
        return out
    clusters = _clusters(rows)
    full = {t: np.array([i for sc in c for i in sc], int) for t, c in clusters.items()}
    rng = np.random.default_rng(seed)
    draws = [_draw(clusters, rng) for _ in range(n_boot)]
    for m in METRICS:
        num = np.array([r['metrics'][m]['num'] for r in rows], float)
        den = np.array([r['metrics'][m]['den'] for r in rows], float)
        pooled, macro, n_t = _stats(num, den, full)
        n_app = int((den > 0).sum())
        n_c = len({(r['task'], r['variant']) for r, dn in zip(rows, den) if dn > 0})
        e = dict(macro=None if np.isnan(macro) else macro, pooled=None if np.isnan(pooled) else float(pooled),
                 n_eligible=n_app, n_na=len(rows) - n_app, n_tasks_contributing=n_t, n_cases_contributing=n_c,
                 macro_ci95=None, pooled_ci95=None)
        if KIND[m] == 'ratio':
            e.update(numerator=float(num.sum()), denominator=float(den.sum()))
        if draws and n_app >= 2:
            bs = np.array([_stats(num, den, dr)[:2] for dr in draws], float)
            for j, how in enumerate(('pooled', 'macro')):
                col = bs[:, j][~np.isnan(bs[:, j])]
                if len(col) and e[how] is not None:
                    e[f'{how}_ci95'] = [float(np.percentile(col, 2.5)), float(np.percentile(col, 97.5))]
        out[m] = e
    return out


def _group(rows, key):
    g = {}
    for r in rows:
        g.setdefault(key(r), []).append(r)
    return dict(sorted(g.items()))


def aggregate(rows, n_boot=2000, seed=0) -> dict:
    """Rows of ONE run -> case / task / role group / evaluation set / evaluation set x role group / overall."""
    rows = list(rows)
    S = lambda rs: summarize(rs, n_boot, seed)
    return dict(
        case={k: S(v) for k, v in _group(rows, lambda r: f"{r['task']}/{r['variant']}").items()},
        task={k: S(v) for k, v in _group(rows, lambda r: r['task']).items()},
        role_group={k: S(v) for k, v in _group(rows, lambda r: r['task_group']).items()},
        condition={k: S(v) for k, v in _group(rows, condition).items()},
        eval_set={k: S(v) for k, v in _group(rows, lambda r: str(r['eval_set'])).items()},
        eval_set_role={k: S(v) for k, v in _group(rows, lambda r: f"{r['eval_set']}/{r['task_group']}").items()},
        overall=S(rows))


# ---------------------------------------------------------------------------------------------- paired comparison
def _ep_value(r, m):
    e = r['metrics'][m]
    return None if not e['den'] else e['num'] / e['den']


def paired(rows_a, rows_b, n_boot=2000, seed=0) -> dict:
    """d_i = m_i^A − m_i^B over scenarios (task, variant, seed) present in both runs (one episode per scenario and
    run); mean d and paired bootstrap 95 % CI resampling scenario pairs within tasks. Counts reported."""
    a, b = {_scenario(r): r for r in rows_a}, {_scenario(r): r for r in rows_b}
    common = sorted(set(a) & set(b), key=str)
    out = dict(n_pairs=len(common), n_only_a=len(set(a) - set(b)), n_only_b=len(set(b) - set(a)),
               csr_a=float(np.mean([bool(a[k]['success']) for k in common])) if common else None,
               csr_b=float(np.mean([bool(b[k]['success']) for k in common])) if common else None)
    rng = np.random.default_rng(seed)
    for m in METRICS:
        dd = [(k[0], _ep_value(a[k], m) - _ep_value(b[k], m)) for k in common
              if _ep_value(a[k], m) is not None and _ep_value(b[k], m) is not None]
        e = dict(n=len(dd), n_na=len(common) - len(dd), n_tasks_contributing=len({t for t, _ in dd}),
                 n_cases_contributing=len({(k[0], k[1]) for k in common if _ep_value(a[k], m) is not None
                                           and _ep_value(b[k], m) is not None}),
                 mean_diff=None, ci95=None)
        if dd:
            d = np.array([x for _, x in dd])
            e['mean_diff'] = float(d.mean())
            strata = [np.array([i for i, (t, _) in enumerate(dd) if t == tk]) for tk in sorted({t for t, _ in dd})]
            if n_boot and len(dd) >= 2:
                bs = [d[np.concatenate([s[rng.integers(0, len(s), len(s))] for s in strata])].mean()
                      for _ in range(n_boot)]
                e['ci95'] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        out[m] = e
    return out


# ---------------------------------------------------------------------------------------------- secondary / guards
def anticipation_summary(rows) -> dict:
    """Secondary indicators (12/QĐ-KT C20: AM, useful-action onset, ...; NOT among the 8 metrics): every numeric key
    of the episodes' `anticipation` dict, mean over applicable episodes (pooled) and over tasks (macro), with n,
    n/a, contributing tasks / cases. Empty when the detector provides none."""
    keys = sorted({k for r in rows if isinstance(r.get('anticipation'), dict) for k, v in r['anticipation'].items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool)})
    out = {}
    for k in keys:
        vals = [(r['task'], r['variant'], float(r['anticipation'][k])) for r in rows
                if isinstance(r.get('anticipation'), dict) and _num(r['anticipation'].get(k)) is not None]
        per_task = {}
        for t, _, v in vals:
            per_task.setdefault(t, []).append(v)
        out[k] = dict(pooled=float(np.mean([v for *_, v in vals])) if vals else None,
                      macro=float(np.mean([np.mean(v) for v in per_task.values()])) if per_task else None,
                      n_eligible=len(vals), n_na=len(rows) - len(vals), n_tasks_contributing=len(per_task),
                      n_cases_contributing=len({(t, c) for t, c, _ in vals}))
    return out


def test_seed_range():
    """configs/hrc/collection.yaml splits.test (03/TTr-HL #4), read without the simulator."""
    import yaml
    lo, hi = yaml.safe_load((ROOT / 'configs/hrc/collection.yaml').read_text())['splits']['test']
    return int(lo), int(hi)


test_seed_range.__test__ = False        # not a pytest test


def check_selection(rows, seed_range=None) -> list[str]:
    """12/QĐ-KT E29: a run used for SELECTION (checkpoint / execution horizon / threshold) must not contain final-test
    seeds; returns the offending (run, task, variant, seed)."""
    lo, hi = seed_range or test_seed_range()
    return [f"{r.get('run')} {r['task']}/{r['variant']} seed {r['seed']} is a test seed ({lo}-{hi})"
            for r in rows if r.get('seed') is not None and lo <= int(r['seed']) <= hi]


def check_paired_set(rows_by_run: dict, models=('m0', 'm1_noh', 'm1')) -> tuple[dict, list[str]]:
    """12/QĐ-KT E26: for each n_action_steps, exactly one run of each model on IDENTICAL scenarios. Returns
    ({n_action_steps: {model: run}}, problems)."""
    by_n, probs = {}, []
    for run, rows in rows_by_run.items():
        if rows and rows[0]['model'] in models:
            ns = {r['n_action_steps'] for r in rows}
            if len(ns) != 1:
                probs.append(f'{run}: mixed n_action_steps {sorted(ns, key=str)}')
                continue
            slot = by_n.setdefault(ns.pop(), {})
            if rows[0]['model'] in slot:
                probs.append(f"n_action_steps {rows[0]['n_action_steps']}: two runs of {rows[0]['model']} "
                             f"({slot[rows[0]['model']]}, {run}); pass one")
            slot[rows[0]['model']] = run
    if not by_n:
        probs.append(f'no run of {list(models)}')
    for n, slot in by_n.items():
        miss = [m for m in models if m not in slot]
        if miss:
            probs.append(f'n_action_steps {n}: missing {miss}')
        sets = {m: {_scenario(r) for r in rows_by_run[run]} for m, run in slot.items()}
        ref = next(iter(sets.values()), set())
        for m, sc in sets.items():
            if sc != ref:
                probs.append(f'n_action_steps {n}: {m} scenarios differ ({len(sc ^ ref)} not shared)')
    return by_n, probs

"""8 metrics of 10/QĐ-KT (intent_policy/benchmark/metrics.py) on hand-made metric records."""
import json
import re
from pathlib import Path

import pytest

from intent_policy.benchmark import metrics as M

ROOT = Path(__file__).resolve().parents[1]
QD10 = ROOT / 'docs/ho_so/04_policy_test/QD-KT-10_khoa_8_metric_slide10.md'
D = 0.08


def rec(**kw):
    """A metric record in the schema of METRIC_SPEC_2026-11-18 §5 (agent H's detector output)."""
    base = dict(task='I1', variant='nominal', group='N1', seed=1, model='m0', step=1, n_action_steps=10,
                detector_version='det-test', schema_version='schema-test',
                t_start=0., t_success=10., success=True, rounds=[], steps_required=[], steps_done=[],
                steps_ordered=True, disrupted=False, recovered=False, human_robot_contact=False,
                safety_audit=dict(min_distance_m=0.2, closest_pair=['wrist_3', 'forearm'], violations=[]))
    base.update(kw)
    return base


def rnd(i, R=None, H=None, T=None, S=None):
    return dict(round_id=i, t_R_ready=R, t_H_ready=H, t_trigger=T, t_response=S)


def val(r, m):
    return M.episode_metrics(r)['metrics'][m]['value']


def diag(r):
    return M.episode_metrics(r)['diagnostics']


# --------------------------------------------------------------------------------- slide-10 consistency
def test_slide10_table_matches_decision_verbatim():
    text = QD10.read_text()
    blocks = re.split(r'\n\*\*(\d)\. ', text)[1:]
    qd = {}
    for i in range(0, len(blocks), 2):
        head, body = blocks[i + 1].split('**', 1)
        name, unit = re.match(r'(\w+)(?: \((\w+)\))?', head).groups()
        better = 'lower' if 'Càng thấp càng tốt' in body else 'higher' if 'Càng cao càng tốt' in body else None
        qd[name] = dict(order=int(blocks[i]), unit=unit, better=better, body=body)
    assert [m['name'] for m in M.SLIDE10] == sorted(qd, key=lambda k: qd[k]['order'])
    expected = {  # (formula, unit, direction) as in 10/QĐ-KT
        'TSync': ('TSync = |t_R,ready − t_H,ready|', 's', 'lower'),
        'CT': ('CT = t_success − t_start', 's', 'lower'),
        'HCS': ('HCS = N_safe / N', None, 'higher'),
        'CSR': ('CSR = N_success / N', None, 'higher'),
        'Rsp': ('Rsp = t_response − t_trigger', 's', 'lower'),
        'OC': ('OC = n_ordered / n_required', None, 'higher'),
        'DSR': ('DSR = N_recovered / N_disrupted', None, 'higher'),
        'CFR': ('CFR = N_contactfree / N', None, 'higher'),
    }
    for m in M.SLIDE10:
        f, unit, better = expected[m['name']]
        assert (m['formula'], m['unit'], m['better']) == (f, unit, better)
        assert f in qd[m['name']]['body'].splitlines()          # verbatim line of the decision
        assert (qd[m['name']]['unit'], qd[m['name']]['better']) == (unit, better)
    assert M.HRIBENCH_NOTE in text


# --------------------------------------------------------------------------------- formulas (11/QĐ-KT C1-C6)
def test_tsync_per_round_and_na():
    r = rec(rounds=[rnd('r0', 2., 3.), rnd('r1', 5., 4.5), rnd('r2', None, 7.)])
    assert val(r, 'TSync') == pytest.approx((1. + .5) / 2)
    assert diag(r)['tsync_valid_rounds'] == 2 and diag(r)['tsync_missing_rounds'] == ['r2']
    assert val(rec(), 'TSync') is None                             # no round -> n/a, never 0
    dup = rec(rounds=[rnd('a', 1., 2.), rnd('a', 3., 3.)])         # ambiguous rounds are never paired
    assert val(dup, 'TSync') is None and 'round_id_missing_or_duplicate' in diag(dup)['round_flags']


def test_legacy_parallel_lists_flagged():
    r = rec(t_R_ready=[1.], t_H_ready=[1.5]); del r['rounds']
    assert val(r, 'TSync') == pytest.approx(.5) and 'legacy_rounds_by_index' in diag(r)['round_flags']


def test_ct_success_only():
    assert val(rec(t_start=1., t_success=9.5), 'CT') == pytest.approx(8.5)
    assert val(rec(success=False, t_success=None), 'CT') is None
    assert val(rec(success=True, t_success=None), 'CT') is None


def test_rsp_from_trigger_only():
    r = rec(rounds=[rnd(0, T=1., S=1.5), rnd(1, T=4., S=4.2), rnd(2, T=6.), rnd(3, T=8., S=7.9), rnd(4)])
    assert val(r, 'Rsp') == pytest.approx((.5 + .2) / 2)
    d = diag(r)
    assert d['rsp_valid_rounds'] == 2 and d['missing_response'] == [2] and d['rsp_invalid_rounds'] == [3]
    assert val(rec(rounds=[rnd(0, T=1.)]), 'Rsp') is None         # no response -> n/a + flag
    assert val(rec(rounds=[rnd(0, T=1., S=0.5)]), 'Rsp') is None   # never negative
    assert val(rec(rounds=[rnd(0, T=1., S=1.)]), 'Rsp') == 0.


def viol(kind='clearance', exempt=False, pair=('wrist_3', 'hand'), phase='transport', d=0.05):
    return dict(kind=kind, t=1., pair=list(pair), phase=phase, distance_m=d, exempt=exempt,
                exemption_reason='declared handover pair' if exempt else None)


def audit(*v):
    return dict(min_distance_m=0.01, closest_pair=['wrist_3', 'hand'], violations=list(v))


def test_hcs_cfr_from_detector_audit():
    assert val(rec(success=False, t_success=None), 'HCS') == 1.    # failed episode stays in HCS / CFR
    assert val(rec(success=False, t_success=None), 'CFR') == 1.
    c = rec(human_robot_contact=True, safety_audit=audit(viol('contact')))
    assert val(c, 'HCS') == 0. and val(c, 'CFR') == 0. and diag(c)['contact_audit_consistent'] is True
    v = rec(safety_audit=audit(viol()))
    assert val(v, 'HCS') == 0. and val(v, 'CFR') == 1.             # clearance violation only hurts HCS
    assert val(rec(safety_audit=audit(viol(exempt=True))), 'HCS') == 1.   # declared close-interaction exemption
    assert val(rec(safety_audit=None), 'HCS') is None               # no audit -> n/a (no own distance rule)
    assert val(rec(human_robot_contact=None), 'HCS') is None and val(rec(human_robot_contact=None), 'CFR') is None
    bad = rec(human_robot_contact=False, safety_audit=audit(viol('contact', exempt=True)))
    assert val(bad, 'HCS') == 0. and diag(bad)['contact_audit_consistent'] is False   # contacts never exempt


def test_hcs_sensitivity_variants():
    rows = [M.episode_row(rec(seed=1, safety_audit=audit(viol(exempt=True, phase='handover')))),
            M.episode_row(rec(seed=2, safety_audit=audit(viol(exempt=False, phase='transport', d=0.07)))),
            M.episode_row(rec(seed=3))]
    out = M.hcs_under(rows, {
        'none_exempt': {'exempt': []},
        'handover_phase': {'exempt': [{'phases': ['handover'], 'pairs': [['wrist_*', 'hand']]}]},
        'handover_06': {'exempt': [{'phases': ['handover']}], 'max_distance_m': 0.06}})
    assert (out['none_exempt']['numerator'], out['none_exempt']['denominator']) == (1, 3)
    assert out['handover_phase']['numerator'] == 2
    assert out['handover_06']['numerator'] == 3                    # 0.07 m is not below the 0.06 m variant


def test_unverified_clearance_is_unknown_not_collision():
    uncertain = dict(viol(), verified=False)
    r = rec(safety_audit=audit(uncertain))
    assert val(r, 'HCS') is None
    assert any('safety unresolved' in p for p in M.check_official([r], 'det-test', 'schema-test'))
    assert val(rec(safety_audit=audit(uncertain, viol())), 'HCS') == 0.
    assert val(rec(safety_audit=audit(dict(uncertain, exempt=True))), 'HCS') == 1.
    assert val(rec(human_robot_contact=True, safety_audit=audit(uncertain)), 'HCS') == 0.


def test_csr():
    assert val(rec(), 'CSR') == 1. and val(rec(success=False), 'CSR') == 0.


def test_oc_predecessors():
    req = ['a', 'b', 'c']
    done = lambda *x: [[n, t] for n, t in x]
    assert val(rec(steps_required=req, steps_done=done(('a', 1), ('b', 2), ('c', 3))), 'OC') == 1.
    # total order = completed sequence correct from the start (12/QĐ-KT C21): nothing counts after the first error
    assert val(rec(steps_required=req, steps_done=done(('b', 1), ('a', 2), ('c', 3))), 'OC') == 0.
    assert val(rec(steps_required=req, steps_done=done(('a', 1), ('c', 2), ('b', 3))), 'OC') == pytest.approx(1 / 3)
    assert val(rec(steps_required=req, steps_done=done(('a', 1), ('b', 1), ('c', 3))), 'OC') == pytest.approx(1 / 3)


def test_oc_same_tick_rule():
    """14/QĐ-KT item 3: same tick = violation only for dependent steps without a recorded substep order."""
    req = ['a', 'b', 'c']
    done = lambda *x: [[n, t] for n, t in x]
    assert val(rec(steps_required=req, steps_done=[['a', 1, 0], ['b', 1, 1], ['c', 3]]), 'OC') == 1.
    assert val(rec(steps_required=req, steps_done=[['b', 1, 0], ['a', 1, 1], ['c', 3]]), 'OC') == 0.
    free = rec(steps_required=req, steps_ordered=False, steps_done=[['a', 1], ['b', 1], ['c', 1]])
    assert val(free, 'OC') == 1.                                    # independent steps may share a tick
    po = dict(steps_required=['a', 'b', 'c'], step_predecessors={'c': ['a']})
    assert val(rec(**po, steps_done=[['a', 1], ['b', 1], ['c', 2]]), 'OC') == 1.           # a, b independent
    assert val(rec(**po, steps_done=[['a', 1], ['b', 1], ['c', 1]]), 'OC') == pytest.approx(2 / 3)   # c needs a
    assert val(rec(**po, steps_done=[{'name': 'a', 't': 1, 'seq': 0}, {'name': 'b', 't': 1},
                                     {'name': 'c', 't': 1, 'seq': 1}]), 'OC') == 1.
    assert val(rec(steps_required=req, steps_done=done(('a', 1), ('c', 3))), 'OC') == pytest.approx(1 / 3)
    assert val(rec(steps_required=req, steps_done=done(('c', 1), ('b', 2), ('a', 3))), 'OC') == 0.
    free = rec(steps_required=req, steps_ordered=False, steps_done=done(('c', 1), ('a', 2)))
    assert val(free, 'OC') == pytest.approx(2 / 3)                 # independent steps (Cowork)
    # partial order from the scenario script: c needs a and b (any order between a, b); d needs c (=> a, b too)
    po = dict(steps_required=['a', 'b', 'c', 'd'], step_predecessors={'c': ['a', 'b'], 'd': ['c']})
    assert val(rec(**po, steps_done=done(('b', 1), ('a', 2), ('c', 3), ('d', 4))), 'OC') == 1.
    # c before b: c invalid, and so d (its predecessor c is not valid) — 12/QĐ-KT C21
    assert val(rec(**po, steps_done=done(('a', 1), ('c', 2), ('b', 3), ('d', 4))), 'OC') == pytest.approx(2 / 4)
    assert val(rec(**po, steps_done=[{'name': 'a', 't': 1}, {'name': 'b', 't': 1}]), 'OC') == pytest.approx(2 / 4)
    assert val(rec(**po, steps_done=done(('a', 1), ('b', 2), ('c', 2), ('d', 3))), 'OC') == pytest.approx(2 / 4)  # strict
    nd = rec(steps_required=req); del nd['steps_ordered']
    assert val(nd, 'OC') is None and diag(nd)['oc_flags'] == ['step_order_not_declared']
    assert val(rec(), 'OC') is None                                 # no required step -> n/a
    e = M.episode_metrics(rec(steps_required=req, steps_done=done(('a', 1))))['metrics']['OC']
    assert (e['num'], e['den']) == (1., 3)


def test_dsr_denominator_and_conditions():
    assert val(rec(disrupted=False), 'DSR') is None                 # outside the denominator
    assert val(rec(disrupted=True, recovered=True), 'DSR') == 1.
    assert val(rec(disrupted=True, recovered=True, success=False, t_success=None), 'DSR') == 0.
    assert val(rec(disrupted=True, recovered=True, human_robot_contact=True), 'DSR') == 0.
    assert val(rec(disrupted=True, recovered=False), 'DSR') == 0.
    rows = [M.episode_row(r) for r in (rec(disrupted=True, recovered=True, seed=1),
                                       rec(disrupted=True, recovered=False, seed=2), rec(seed=3), rec(seed=4))]
    s = M.summarize(rows, n_boot=0)['DSR']
    assert (s['numerator'], s['denominator'], s['pooled'], s['n_na']) == (1., 2., .5, 2)


def test_record_location_and_roles():
    r = dict(task='C4', variant='nominal', group='N1', seed=5, model='m1', metrics=rec(success=False))
    row = M.episode_row(r)
    assert row['role'] == 'Collaborator' and row['metrics']['CSR']['value'] == 0.
    assert M.ROLE_GROUPS == {'I': 'Instructor', 'C': 'Collaborator', 'W': 'Cowork'}
    assert row['safety_audit']['closest_pair'] == ['wrist_3', 'forearm']    # audit kept in results_per_episode


def test_official_check():
    good = [rec(seed=i) for i in range(2)]
    assert M.check_official(good, 'det-test', 'schema-test') == []
    assert M.check_official(good, 'det-other', 'schema-test')                 # wrong detector version
    old = rec(); del old['detector_version']
    assert any('missing' in p for p in M.check_official([old], 'det-test', 'schema-test'))
    mixed = [rec(), rec(detector_version='det-2')]
    assert any('mixed' in p for p in M.check_official(mixed, None, 'schema-test'))


# --------------------------------------------------------------------------------- aggregation, CI, pairing
def _rows():
    rs = []
    for i in range(6):
        rs.append(rec(task='I1', seed=i, success=i < 4, t_success=10. + i if i < 4 else None,
                      rounds=[rnd(0, 1., 1. + .1 * i)]))
    for i in range(2):
        rs.append(rec(task='W2', seed=i, success=True, t_success=20., human_robot_contact=bool(i),
                      safety_audit=audit(viol('contact')) if i else audit()))
    return [M.episode_row(r) for r in rs]


def test_macro_main_pooled_secondary_and_counts():
    s = M.summarize(_rows(), n_boot=0)
    assert s['CSR']['macro'] == pytest.approx((4 / 6 + 1) / 2) and s['CSR']['pooled'] == pytest.approx(6 / 8)
    assert (s['CSR']['numerator'], s['CSR']['denominator'], s['CSR']['n_tasks_contributing']) == (6., 8., 2)
    assert s['CT']['n_eligible'] == 6 and s['CT']['n_na'] == 2
    assert s['CT']['pooled'] == pytest.approx((10 + 11 + 12 + 13 + 20 + 20) / 6)
    assert s['CT']['macro'] == pytest.approx((11.5 + 20) / 2)
    assert s['TSync']['n_tasks_contributing'] == 1 and s['TSync']['macro'] == pytest.approx(s['TSync']['pooled'])
    assert s['CFR']['pooled'] == pytest.approx(7 / 8) and s['CFR']['macro'] == pytest.approx((1 + .5) / 2)


def test_bootstrap_deterministic_scenario_grouped():
    rows = _rows()
    a, b = M.aggregate(rows, n_boot=300, seed=7), M.aggregate(rows, n_boot=300, seed=7)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    c = M.aggregate(rows, n_boot=300, seed=8)
    assert a['overall']['CT']['pooled_ci95'] != c['overall']['CT']['pooled_ci95']
    lo, hi = a['overall']['CT']['macro_ci95']
    assert lo <= a['overall']['CT']['macro'] <= hi
    assert set(a) == {'case', 'task', 'role_group', 'condition', 'eval_set', 'eval_set_role', 'overall'}
    assert a['overall']['CSR']['n_cases_contributing'] == 2 and a['overall']['TSync']['n_cases_contributing'] == 1
    assert set(a['role_group']) == {'I', 'W'} and 'I1/nominal' in a['case'] and 'N1/I' in a['eval_set_role']
    # scenario grouping: two episodes of one scenario are always drawn together
    twin = [M.episode_row(rec(seed=1, t_success=5.)), M.episode_row(rec(seed=1, t_success=5.)),
            M.episode_row(rec(seed=2, t_success=9.))]
    cl = M._clusters(twin)
    assert sorted(map(len, cl['I1'])) == [1, 2]


def test_paired_differences():
    ra = [M.episode_row(rec(seed=i, model='m1', t_success=8. + i)) for i in range(4)]
    rb = [M.episode_row(rec(seed=i, model='m0', t_success=10. + i, success=i != 3)) for i in range(5)]
    p = M.paired(ra, rb, n_boot=200, seed=0)
    assert p['n_pairs'] == 4 and p['n_only_b'] == 1
    assert p['CT']['n'] == 3 and p['CT']['n_na'] == 1 and p['CT']['mean_diff'] == pytest.approx(-2.)
    assert p['CSR']['mean_diff'] == pytest.approx(.25)
    assert p['CT']['ci95'] == M.paired(ra, rb, n_boot=200, seed=0)['CT']['ci95']
    assert p['TSync']['mean_diff'] is None and p['TSync']['n'] == 0


def test_report_tool(tmp_path):
    from tools import benchmark_report as R
    for model in ('m0', 'm1', 'm1_noh'):
        d = tmp_path / 'ev' / f'{model}_step1_n10'; d.mkdir(parents=True)
        d.joinpath('episodes.jsonl').write_text('\n'.join(json.dumps(dict(
            task='I1', variant='nominal', group='N1', seed=s, model=model, step=1, n_action_steps=10,
            metrics=rec(t_success=9. + s))) for s in range(3)) + '\n')
    sens = tmp_path / 'sens.json'
    sens.write_text(json.dumps({'none_exempt': {'exempt': []}}))
    R.main([str(tmp_path / 'ev'), '--out', str(tmp_path / 'out'), '--n-boot', '50', '--hcs-sensitivity', str(sens)])
    for f in ('results_per_episode.jsonl', 'results_per_task.json', 'results_summary.json', 'REPORT.md'):
        assert (tmp_path / 'out' / f).exists()
    md = (tmp_path / 'out' / 'REPORT.md').read_text()
    assert M.HRIBENCH_NOTE in md and 'DIAGNOSTIC' in md and 'HCS sensitivity' in md
    assert 'm1_step1_n10 vs m0_step1_n10' in md and 'm1_step1_n10 vs m1_noh_step1_n10' in md
    with pytest.raises(SystemExit):                                # records not from H's published detector
        R.main([str(tmp_path / 'ev'), '--out', str(tmp_path / 'out2'), '--official'])


# --------------------------------------------------------------------------------- 12/QĐ-KT C20, E26, E28, E29
def test_conditions_reported_separately():
    rows = [M.episode_row(rec(task='C4', variant=v, group=g, seed=i))
            for i, (v, g) in enumerate([('nominal', 'N1'), ('nominal', 'N5'), ('switch_before_commit', 'N6'),
                                        ('switch_after_commit', 'N6')])]
    assert set(M.aggregate(rows, n_boot=0)['condition']) == {'nominal', 'forecast_fault', 'switch_before_commit',
                                                             'switch_after_commit'}


def test_anticipation_secondary_only():
    rows = [M.episode_row(rec(seed=1, anticipation={'AM_s': 0.4, 'useful_onset_s': 1.0})),
            M.episode_row(rec(seed=2, task='C4', anticipation={'AM_s': None})), M.episode_row(rec(seed=3))]
    a = M.anticipation_summary(rows)
    assert a['AM_s']['n_eligible'] == 1 and a['AM_s']['n_na'] == 2 and a['AM_s']['n_tasks_contributing'] == 1
    assert 'AM_s' not in M.METRICS and set(rows[0]['metrics']) == set(M.METRICS)


def test_selection_refuses_test_seeds():
    lo, hi = M.test_seed_range()
    assert (lo, hi) == (300900, 300999)                             # configs/hrc/collection.yaml splits.test
    rows = [M.episode_row(rec(seed=300850)), M.episode_row(rec(seed=300901))]
    bad = M.check_selection(rows)
    assert len(bad) == 1 and '300901' in bad[0]
    assert M.check_selection(rows[:1]) == []


def test_paired_set_check():
    mk = lambda model, seeds, n=10: [M.episode_row(rec(model=model, seed=s, n_action_steps=n)) for s in seeds]
    ok = {'m0_a': mk('m0', [1, 2]), 'noh_a': mk('m1_noh', [1, 2]), 'm1_a': mk('m1', [1, 2])}
    slots, probs = M.check_paired_set(ok)
    assert probs == [] and slots == {10: {'m0': 'm0_a', 'm1_noh': 'noh_a', 'm1': 'm1_a'}}
    assert M.check_paired_set({**ok, 'm1_a': mk('m1', [1, 3])})[1]           # different scenarios
    assert M.check_paired_set({'m0_a': ok['m0_a'], 'm1_a': ok['m1_a']})[1]     # M1-noH missing
    assert M.check_paired_set({**ok, 'm1_b': mk('m1', [1, 2])})[1]           # two runs of one model, same n
    assert M.check_paired_set({**ok, 'm1_a': mk('m1', [1, 2], n=1)})[1]      # n_action_steps differ


def test_report_selection_and_paired_validation(tmp_path):
    from tools import benchmark_report as R
    for model, seed0 in (('m0', 300900), ('m1', 300900), ('m1_noh', 300900)):
        d = tmp_path / 'ev' / f'{model}_step1_n10'; d.mkdir(parents=True)
        d.joinpath('episodes.jsonl').write_text('\n'.join(json.dumps(dict(
            task='C4', variant=v, group=g, seed=seed0 + i, model=model, step=1, n_action_steps=10, metrics=rec()))
            for i, (v, g) in enumerate([('nominal', 'N1'), ('nominal', 'N5')])) + '\n')
    with pytest.raises(SystemExit):
        R.main([str(tmp_path / 'ev'), '--out', str(tmp_path / 'o1'), '--selection', '--n-boot', '0'])
    R.main([str(tmp_path / 'ev'), '--out', str(tmp_path / 'o2'), '--paired-validation', '--n-boot', '20'])
    md = (tmp_path / 'o2' / 'REPORT.md').read_text()
    assert 'm1_noh_step1_n10 vs m0_step1_n10 [forecast_fault]' in md and 'condition forecast_fault' in md
    assert 'Secondary (not slide-10 metrics)' in md


# --------------------------------------------------------------------------------- safety detail (15/QĐ-KT D21-D24)
def _ticks(d_list, sp=0.06, dt=0.05):
    return [[round(i * dt, 3), d, d, sp, 0, d] for i, d in enumerate(d_list)]


COLS = ['t', 'd_nonexempt_m', 'd_nonexempt_lower_bound_m', 'S_p_m', 'close_interaction', 'd_all_m']


def clr(t, t_end, pair, d, sp=0.06, verified=True, exempt=False):
    return dict(kind='clearance', t=t, t_end=t_end, pair=pair, phase='non_contact', distance_m=d, threshold_m=sp,
                verified=verified, exempt=exempt)


def sec(r, ticks=None, **kw):
    a = dict(min_distance_m=0.01, closest_pair=['a', 'b'], violations=kw.pop('violations', []))
    if ticks is not None:
        a.update(ticks=ticks, ticks_columns=COLS)
    a.update(kw)
    return rec(safety_audit=a, **r)


def test_safety_detail_proximity_fields_and_hcs_unchanged():
    r = sec({}, ticks=_ticks([0.2, 0.05, 0.03, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
            violations=[clr(0.05, 0.1, ['finger', 'hand'], 0.03), clr(0.0, 0.0, ['x', 'y'], 0.2, exempt=True)])
    d = M.safety_detail(r)
    assert d['cls'] == 'proximity_violation' and val(r, 'HCS') == 0.
    assert d['n_ticks'] == 10 and d['violation_ticks'] == 2 and d['dt_s'] == pytest.approx(0.05)
    assert d['violation_duration_s'] == pytest.approx(0.1) and d['violation_ratio'] == pytest.approx(0.2)
    assert d['min_verified_clearance_m'] == 0.03 and d['max_clearance_deficit_m'] == pytest.approx(0.03)
    assert d['worst_pair'] == ['finger', 'hand'] and d['contact_force_n'] is None and not d['contact_force_missing']


def test_safety_detail_collision_force_and_missing_force():
    ct = dict(kind='contact', t=1., t_end=1.1, pair=['link', 'hand'], distance_m=-0.002, force_n=12.5,
              penetration_m=0.002, verified=True, exempt=False)
    r = sec(dict(human_robot_contact=True), violations=[ct])
    d = M.safety_detail(r)
    assert d['cls'] == 'collision' and d['contact_force_n'] == 12.5 and d['penetration_m'] == 0.002
    assert d['worst_pair'] == ['link', 'hand'] and d['violation_ratio'] is None and d['source'] == 'violations'
    assert val(r, 'HCS') == 0. and val(r, 'CFR') == 0.
    ct2 = dict(ct, force_n=None, penetration_m=None)
    d2 = M.safety_detail(sec(dict(human_robot_contact=True), violations=[ct2]))
    assert d2['contact_force_n'] is None and d2['contact_force_missing'] is True


def test_safety_detail_unverified_is_not_evidence_and_hcs_is_na():
    r = sec({}, ticks=_ticks([0.2, 0.0, 0.0, 0.2]), violations=[clr(0.05, 0.1, ['mesh', 'box'], 0.0, verified=False)])
    d = M.safety_detail(r)
    assert d['cls'] == 'unverified_distance' and d['violation_ticks'] == 0 and d['unverified_ticks'] == 2
    assert d['max_clearance_deficit_m'] is None and d['min_verified_clearance_m'] == 0.2 and d['worst_pair'] is None
    assert val(r, 'HCS') is None                         # same as safe(): unresolved -> n/a, never a pass


def test_safety_detail_exempt_only_is_clear_and_class_matches_safe():
    r = sec({}, ticks=_ticks([0.2] * 4), violations=[clr(0.0, 0.1, ['finger', 'palm'], 0.01, exempt=True)])
    assert M.safety_detail(r)['cls'] == 'clear' and val(r, 'HCS') == 1.
    cases = [sec({}, violations=[]), r, sec({}, violations=[clr(0., 0.1, ['a', 'b'], 0.01)]),
             sec({}, violations=[clr(0., 0.1, ['a', 'b'], 0.0, verified=False)]),
             sec(dict(human_robot_contact=True), violations=[])]
    for c in cases:
        cls, hcs = M.safety_detail(c)['cls'], val(c, 'HCS')
        assert (hcs == 1.) == (cls == 'clear') and (hcs is None) == (cls == 'unverified_distance')


def test_safety_detail_na_without_audit_and_summary_counts():
    r_no = rec(human_robot_contact=None, safety_audit=None)
    assert M.safety_detail(r_no)['cls'] is None
    rows = [M.episode_row(x) for x in (
        sec(dict(task='I1', seed=1), ticks=_ticks([0.2, 0.03, 0.2]), violations=[clr(0.05, 0.05, ['f', 'h'], 0.03)]),
        sec(dict(task='I1', seed=2), ticks=_ticks([0.2] * 3), violations=[]),
        sec(dict(task='C1', seed=3), violations=[clr(0.0, 0.0, ['m', 'b'], 0.0, verified=False)]),
        r_no)]
    s = M.safety_detail_summary(rows)['overall']
    assert s['classes'] == dict(collision=0, proximity_violation=1, unverified_distance=1, clear=1)
    assert s['n_with_audit'] == 3 and s['n_na_audit'] == 1 and s['n_tasks'] == 2
    assert s['violation_ratio']['n'] == 2 and s['violation_ratio']['pooled'] == pytest.approx((1 / 3 + 0) / 2)
    assert s['worst_pairs'] == {'f x h': 1}
    assert set(M.safety_detail_summary(rows)['task']) == {'I1', 'C1'}


def test_safety_detail_v10_ticks_without_lower_bound_column():
    ticks = [[0., 0.2, 0.06, 0, 0.2], [0.05, 0.02, 0.06, 0, 0.02]]
    a = dict(violations=[clr(0.05, 0.05, ['a', 'b'], 0.02)], ticks=ticks, ticks_columns=['t', 'd_nonexempt_m', 'S_p_m', 'close_interaction', 'd_all_m'])
    d = M.safety_detail(rec(safety_audit=a))
    assert d['violation_ticks'] == 1 and d['max_clearance_deficit_m'] == pytest.approx(0.04)


def test_rule_variants_from_ticks():
    r_clear = sec({}, ticks=_ticks([0.2] * 3), violations=[])
    r_prox = sec({}, ticks=_ticks([0.2, 0.03, 0.2]), violations=[clr(0.05, 0.05, ['a', 'b'], 0.03)])
    rows = [dict(M.episode_row(x), task='I1') for x in (r_clear, r_prox)]
    out = M.hcs_rule_variants(rows, ['ssm_exempt_verified', 'ssm_exempt_lowerbound', 'static_0.04', 'static_0.08', 'contact_only'])
    assert out['ssm_exempt_verified']['numerator'] == 1 and out['ssm_exempt_lowerbound']['numerator'] == 1
    assert out['static_0.04']['numerator'] == 1 and out['contact_only']['numerator'] == 2

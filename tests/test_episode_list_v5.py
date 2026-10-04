"""Episode lists v5 (docs/specs/EPISODE_PLAN_v5.md §5): checks on configs/episode_lists/v5 (skipped until generated
by scripts/make_episode_list_v5.py)."""
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from intent_policy.scenarios.config import ScenarioConfig, cup_crowded, sample_variation
from scripts.make_episode_list_v5 import CHANGE_STRATA, HOLD_STRATA, REACH_STRATA, T4_STRATA, T4NEG_STRATA, SID

ROOT = Path(__file__).resolve().parents[1] / 'configs/episode_lists/v5'
DEMO = {'T1': 120, 'T2': 144, 'T3': 88, 'T4': 72, 'T4neg': 48, 'T5': 108}
EVAL = {'T1': 48, 'T2': 48, 'T3': 48, 'T4': 48, 'T4neg': 24, 'T5': 48}

pytestmark = pytest.mark.skipif(not (ROOT / 'demo_list.jsonl').exists(), reason='v5 lists not generated')


def load(split):
    return [json.loads(l) for l in (ROOT / f'{split}_list.jsonl').read_text().splitlines()]


def rows(split):
    with open(ROOT / f'{split}_list.csv') as f:
        return list(csv.DictReader(f))


def test_counts_per_task():
    for split, want in (('demo', DEMO), ('eval', EVAL)):
        eps = load(split)
        got = Counter(e['task'] for e in eps if not e['extra'])
        assert dict(got) == want, (split, dict(got))
        assert len(rows(split)) == len(eps)
    extra = [e for e in load('demo') if e['extra']]
    assert all(e['task'] == 'T4' for e in extra) and not any(e['extra'] for e in load('eval'))


def test_t1_full_pairs_and_colours():
    t1 = [e for e in load('demo') if e['task'] == 'T1']
    combos = Counter((e['axes']['target_slot'], e['axes']['twin_slot'], e['axes']['place_zone'], e['dart']) for e in t1)
    assert len(combos) == 120 and set(combos.values()) == {1}
    assert dict(Counter(e['axes']['pair_color'] for e in t1)) == {'red': 40, 'blue': 40, 'yellow': 40}


def test_t2_cells_lowering_crowding_and_pairs():
    scene = ScenarioConfig.load('t2_handover').scene
    t2 = [e for e in load('demo') if e['task'] == 'T2']
    cells = Counter((e['axes']['pair_type'], e['axes']['hand_zone'], e['axes']['target_slot'], e['axes']['timing_stratum'])
                    for e in t2)
    assert len(cells) == 72 and set(cells.values()) == {2}
    lowered = Counter((e['axes']['pair_type'], e['axes']['hand_zone'], e['axes']['target_slot'], e['axes']['timing_stratum'],
                       e['axes']['hand_lowered_first']) for e in t2)
    assert set(lowered.values()) == {1}
    assert not any(cup_crowded(scene, e['spec']['layout']) for e in t2)
    pairs = Counter((e['axes']['pair_type'], e['axes']['target_slot'], e['axes']['twin_slot']) for e in t2)
    assert len(pairs) == 50 and min(pairs.values()) >= 2


def test_t3_full_combinations():
    t3 = [e for e in load('demo') if e['task'] == 'T3']
    combos = Counter(json.dumps([e['spec']['target'], e['spec']['layout'], e['spec']['u_blocks']], sort_keys=True) for e in t3)
    assert len(combos) == 88 and set(combos.values()) == {1}


def test_t4_slot_stratum_dart_and_colours():
    t4 = [e for e in load('demo') if e['task'] == 'T4' and not e['extra']]
    cells = Counter((e['axes']['target_slot'], e['axes']['timing_stratum'], e['dart']) for e in t4)
    assert len(cells) == 72 and set(cells.values()) == {1}
    assert dict(Counter(e['axes']['pair_color'] for e in t4)) == {'red': 24, 'blue': 24, 'yellow': 24}


def test_stratified_values_fall_in_their_strata():
    cfgs = {b: ScenarioConfig.load(sid) for b, sid in SID.items()}
    for split in ('demo', 'eval'):
        for e in load(split):
            ax, h = e['axes'], sample_variation(cfgs[e['base']], e['seed'], e['spec'])['human']
            if e['task'] == 'T2':
                lo, hi = REACH_STRATA[ax['timing_stratum']]
                assert lo <= h['reach_delay_s'] <= hi and h['lower_first'] == ax['hand_lowered_first']
            elif e['task'] in ('T4', 'T4neg'):
                lo, hi = (T4_STRATA if e['task'] == 'T4' else T4NEG_STRATA)[ax['timing_stratum']]
                assert lo <= h['intrusion_time_s'] <= hi
                if e['task'] == 'T4':
                    lo, hi = HOLD_STRATA[ax['hold_stratum']]
                    assert lo <= h['intrusion_hold_s'] <= hi
            elif e['task'] == 'T5':
                lo, hi = CHANGE_STRATA[ax['change_stratum']]
                assert lo <= h['change_delay_s'] <= hi


def test_seeds_disjoint_between_demo_and_eval():
    demo, ev = {e['seed'] for e in load('demo')}, {e['seed'] for e in load('eval')}
    assert not demo & ev


def test_both_experts_succeeded_with_the_kept_seed():
    for split in ('demo', 'eval'):
        pilot = json.loads((ROOT / f'{split}_pilot.json').read_text())
        for e in load(split):
            last = pilot[e['cell_id']]['tries'][-1]
            assert last['seed'] == e['seed'] and last['late']['success'] and last['early']['success'], e['cell_id']
            if e['task'] == 'T5':
                assert all(last[w]['changed'] and not last[w]['cancelled'] for w in ('late', 'early'))


def test_no_consecutive_target_slot():
    for split in ('demo', 'eval'):
        slots = [e['spec']['layout'][e['spec']['target']] for e in load(split)]
        assert not any(a == b for a, b in zip(slots, slots[1:])), split

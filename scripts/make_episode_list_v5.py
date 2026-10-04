"""Episode lists v5 (docs/specs/EPISODE_PLAN_v5.md): design cells -> seed selection -> demo / eval lists.

  1. Design cells per task (plan §3): trajectory cells crossed in full, perception factors cycled round-robin in a
     fixed order inside the trajectory cells, stratified timings stored as ranges in the spec (drawn uniformly
     within the stratum by the episode seed, intent_policy/scenarios/config.py), DART fixed per cell.
  2. Seed selection (plan §2.4): per cell try the seeds s0, s1, s2 and keep the first one for which BOTH the late
     and the early expert (configs/experiments/intent_act_{late,early}.yaml, with the cell's DART) succeed; T5 also
     needs the change of mind to happen (not dropped) in both. Cells failing every seed go to failed_cells.csv.
     T4 (demo): strata with > 25 % episodes whose hand never entered the robot zone get extra cells until 12 entered.
  3. Lists: within each task the order avoids consecutive episodes on the same target slot.

Output (configs/episode_lists/v5/): demo_list.csv / eval_list.csv (plan §4 columns), demo_list.jsonl /
eval_list.jsonl (the pipeline's scenario-list format, + cell_id / dart / optional_t5), failed_cells.csv, pilot.json
(per-cell expert outcomes), summary.md.

  ./run.sh -m scripts.make_episode_list_v5 --cells-only        # step 1 only (no simulation)
  ./run.sh -m scripts.make_episode_list_v5 --workers 4         # steps 1-3
"""
import argparse
import csv
import itertools
import json
import multiprocessing as mp
import time
from collections import Counter, defaultdict

import numpy as np

from intent_policy.scenarios.config import ScenarioConfig, cup_crowded, validate_spec
from intent_policy.utils import load_yaml, resolve

OUT = 'configs/episode_lists/v5'
SLOTS = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6']
PLACES, HANDS = ['P1', 'P2'], ['H1', 'H2']
COLORS = {'red': ('B1', 'B1p'), 'blue': ('B2', 'B2p'), 'yellow': ('B3', 'B3p')}
COLOR_OF = {k: c for c, pair in COLORS.items() for k in pair}
T2_PAIRS = {'cube': ('B1', 'B1p'), 'cup': ('C1', 'C2')}
T2_THIRD = {'cube': ['B2', 'B3', 'C1', 'C3'], 'cup': ['B1', 'B2', 'B3', 'C3']}
T3_BLOCK_OF = {'C3': 'B1', 'C1': 'B2'}                       # learned workflow: red B1 -> green C3, blue B2 -> white C1
REACH_STRATA = [[0.0, 2.0], [2.0, 4.0], [4.0, 6.0]]          # T2 hand-out delay after the pointing hold
T4_STRATA = [[0.5 + 1.25 * i, 0.5 + 1.25 * (i + 1)] for i in range(6)]
T4NEG_STRATA = [[0.5 + 1.875 * i, 0.5 + 1.875 * (i + 1)] for i in range(4)]
HOLD_STRATA = [[0.5, 2.25], [2.25, 4.0]]
HOLD_FULL = [0.5, 4.0]
CHANGE_STRATA = [[0.3, 2.2], [2.2, 4.1], [4.1, 6.0]]
SID = {'T1': 't1_pick_place', 'T2': 't2_handover', 'T3': 't3_assist', 'T4': 't4_interrupt'}
SEED_BASE = {'demo': 1_000_000, 'eval': 2_000_000}
ATTEMPTS = 3


class RR:
    """Round-robin over `options` in a fixed order; `next(ok)` skips options failing `ok` (keeps the cycle going)."""
    def __init__(self, options):
        self.options, self.i = list(options), 0

    def next(self, ok=lambda x: True):
        for _ in range(len(self.options)):
            x = self.options[self.i % len(self.options)]
            self.i += 1
            if ok(x):
                return x
        raise ValueError(f'no valid option among {self.options}')


class RRMap(dict):
    """A round-robin per context key, created on first use with options_fn(key)."""
    def __init__(self, options_fn):
        super().__init__()
        self.options_fn = options_fn

    def __missing__(self, key):
        self[key] = RR(self.options_fn(key))
        return self[key]


# ---------------------------------------------------------------------------------------------------- design cells
def crowded(scene, layout) -> bool:
    return cup_crowded(scene, layout)


def t1_cells(split, scene):
    color, third_color = RR(COLORS), RRMap(lambda c: [x for x in COLORS if x != c])
    third_slot = RR(SLOTS)
    out = []
    if split == 'demo':
        grid = [(ts, tw, pl, dart) for ts in SLOTS for tw in SLOTS if tw != ts for pl in PLACES for dart in (False, True)]
    else:
        twin = RRMap(lambda ts: [s for s in SLOTS if s != ts])
        grid = [(ts, twin[ts].next(), pl, False) for ts, pl in itertools.product(SLOTS, PLACES) for _ in range(4)]
    for ts, tw, pl, dart in grid:
        c = color.next()
        tc = third_color[c].next()
        target, tw_key, third = COLORS[c][0], COLORS[c][1], COLORS[tc][0]
        s3 = third_slot.next(lambda s: s not in (ts, tw))
        spec = dict(task='T1', target=target, layout={target: ts, tw_key: tw, third: s3}, pair=True, place=pl)
        out.append(dict(task='T1', base='T1', spec=spec, dart=dart,
                        axes=dict(target_slot=ts, twin_slot=tw, pair_color=c, pair_type='cube', third_obj=third,
                                  third_slot=s3, place_zone=pl)))
    return out


def t2_layout_options(scene, kind, ts):
    a, b = T2_PAIRS[kind]
    return [tw for tw in SLOTS if tw != ts and not crowded(scene, {a: ts, b: tw})
            and any(not crowded(scene, {a: ts, b: tw, o: s}) for o in T2_THIRD[kind] for s in SLOTS if s not in (ts, tw))]


def t2_third_rr(scene):
    return RRMap(lambda key: [(o, s) for o in T2_THIRD[key[0]] for s in SLOTS])


def t2_cell(scene, kind, ts, tw, hand, third_rr, extra_spec, axes, dart, task='T2'):
    a, b = T2_PAIRS[kind]
    o, s3 = third_rr[(kind, ts)].next(lambda os: os[1] not in (ts, tw) and not crowded(scene, {a: ts, b: tw, os[0]: os[1]}))
    spec = dict(task='T2', target=a, layout={a: ts, b: tw, o: s3}, pair=True, hand=hand, timing='free', **extra_spec)
    return dict(task=task, base='T2', spec=spec, dart=dart,
                axes=dict(target_slot=ts, twin_slot=tw, pair_type=kind, pair_color='red' if kind == 'cube' else 'white',
                          third_obj=o, third_slot=s3, hand_zone=hand, **axes))


def t2_cells(split, scene):
    twin = RRMap(lambda key: t2_layout_options(scene, *key))
    third = t2_third_rr(scene)
    out = []
    if split == 'demo':
        dart_k = Counter()
        for kind, ts, hand, st, lower in itertools.product(['cube', 'cup'], SLOTS, HANDS, range(3), (True, False)):
            dart = dart_k[ts] % 2 == 1
            dart_k[ts] += 1
            out.append(t2_cell(scene, kind, ts, twin[(kind, ts)].next(), hand, third,
                               dict(reach_delay_range=REACH_STRATA[st], lower_first=lower),
                               dict(timing_stratum=st, hand_lowered_first=lower), dart))
    else:
        slot_rr = RRMap(lambda g: SLOTS)
        for kind, hand, st in itertools.product(['cube', 'cup'], HANDS, range(3)):
            for k in range(4):
                ts = slot_rr[(kind, hand)].next()
                lower = k % 2 == 0
                out.append(t2_cell(scene, kind, ts, twin[(kind, ts)].next(), hand, third,
                                   dict(reach_delay_range=REACH_STRATA[st], lower_first=lower),
                                   dict(timing_stratum=st, hand_lowered_first=lower), False))
    return out


def t3_valid_others(scene, cup, ts):
    other = 'C1' if cup == 'C3' else 'C3'
    return [s for s in SLOTS if s != ts and not crowded(scene, {cup: ts, other: s})]


def t3_cell(cup, ts, os_, order, dart, task='T3', change=None, axes=None):
    other = 'C1' if cup == 'C3' else 'C3'
    spec = dict(task='T3', target=cup, layout={cup: ts, other: os_}, u_blocks=list(order))
    if change:
        spec['change'] = change
    return dict(task=task, base='T3', spec=spec, dart=dart,
                axes=dict(target_slot=ts, twin_slot=os_, block_order='-'.join(order), **(axes or {})))


def t3_cells(split, scene):
    orders = [('B1', 'B2'), ('B2', 'B1')]
    out = []
    if split == 'demo':
        dart_k = Counter()
        for cup in ('C1', 'C3'):
            for ts in SLOTS:
                for os_ in t3_valid_others(scene, cup, ts):
                    for order in orders:
                        out.append(t3_cell(cup, ts, os_, order, False))
        for c in sorted(out, key=lambda c: c['axes']['target_slot']):
            c['dart'] = dart_k[c['axes']['target_slot']] % 2 == 1
            dart_k[c['axes']['target_slot']] += 1
    else:
        other_rr = RRMap(lambda key: t3_valid_others(scene, *key))
        order_rr = RR(orders)
        for ts, cup in itertools.product(SLOTS, ('C1', 'C3')):
            for _ in range(4):
                out.append(t3_cell(cup, ts, other_rr[(cup, ts)].next(), order_rr.next(), False))
    return out


def t4_cell(task, slot, color, time_range, hold_range, dart, axes, negative=False, goal=None):
    cube = COLORS[color][0]
    spec = dict(task='T4', target=cube, layout={cube: slot}, pair=False, place='P1', phase='free',
                hold_s=float(np.mean(hold_range)), negative=negative, neg_target=goal,
                intrusion_time_range=list(time_range), hold_range=list(hold_range))
    return dict(task=task, base='T4', spec=spec, dart=dart,
                axes=dict(target_slot=slot, pair_color=color, hand_goal=goal, **axes))


def t4_cells(split, scene, extra_from: int = 0):
    colors = list(COLORS)
    out = []
    if split == 'demo':
        hold = RR(range(2))
        for i, slot in enumerate(SLOTS):
            for st in range(6):
                for dart in (False, True):
                    h = hold.next()
                    out.append(t4_cell('T4', slot, colors[(i + st) % 3], T4_STRATA[st], HOLD_STRATA[h], dart,
                                       dict(timing_stratum=st, hold_stratum=h)))
    else:
        slot_rr, color_rr, hold = RR(SLOTS), RR(colors), RR(range(2))
        for st in range(6):
            for _ in range(8):
                h = hold.next()
                out.append(t4_cell('T4', slot_rr.next(), color_rr.next(), T4_STRATA[st], HOLD_STRATA[h], False,
                                   dict(timing_stratum=st, hold_stratum=h)))
    return out


def t4neg_cells(split, scene):
    color_rr = RR(COLORS)
    out = []
    if split == 'demo':
        for goal in ('U', 'edge'):
            for slot in SLOTS:
                for st in range(4):
                    out.append(t4_cell('T4neg', slot, color_rr.next(), T4NEG_STRATA[st], HOLD_FULL, st % 2 == 1,
                                       dict(timing_stratum=st), negative=True, goal=goal))
    else:
        st_rr = RR(range(4))
        for goal in ('U', 'edge'):
            for slot in SLOTS:
                for _ in range(2):
                    st = st_rr.next()
                    out.append(t4_cell('T4neg', slot, color_rr.next(), T4NEG_STRATA[st], HOLD_FULL, False,
                                       dict(timing_stratum=st), negative=True, goal=goal))
    return out


def t5_cells(split, scene, t1_scene):
    """Base T1 / T2 / T3; `target_slot` = the INITIAL (old) target, `twin_slot` = the new target (T3: the other cup)."""
    out = []
    change = lambda old, st: dict(timing='free', old=old, delay_range=CHANGE_STRATA[st])
    # base T1
    twin = RRMap(lambda ts: [s for s in SLOTS if s != ts])
    place, color, third_color, third_slot = RR(PLACES), RR(COLORS), RRMap(lambda c: [x for x in COLORS if x != c]), RR(SLOTS)
    if split == 'demo':
        t1_grid = [(ts, st, dart) for ts in SLOTS for st in range(3) for dart in (False, True)]
    else:
        t1_grid = [(SLOTS[k % 6], k % 3, False) for k in range(16)]
    for ts, st, dart in t1_grid:
        tw, c = twin[ts].next(), color.next()
        tc = third_color[c].next()
        old, new, third = COLORS[c][0], COLORS[c][1], COLORS[tc][0]
        s3 = third_slot.next(lambda s: s not in (ts, tw))
        pl = place.next()
        spec = dict(task='T1', target=new, layout={old: ts, new: tw, third: s3}, pair=True, place=pl, change=change(old, st))
        out.append(dict(task='T5', base='T1', spec=spec, dart=dart,
                        axes=dict(target_slot=ts, twin_slot=tw, pair_color=c, pair_type='cube', third_obj=third,
                                  third_slot=s3, place_zone=pl, change_stratum=st, base_task='T1')))
    # base T2
    twin2 = RRMap(lambda key: t2_layout_options(scene, *key))
    third2 = t2_third_rr(scene)
    hand = RR(HANDS)
    if split == 'demo':
        t2_grid = [(kind, ts, st) for kind in ('cube', 'cup') for ts in SLOTS for st in range(3)]
    else:
        t2_grid = [(('cube', 'cup')[k % 2], SLOTS[k % 6], k % 3) for k in range(16)]
    dart_k = Counter()
    for kind, ts, st in t2_grid:
        dart = split == 'demo' and dart_k[ts] % 2 == 1
        dart_k[ts] += 1
        a, b = T2_PAIRS[kind]
        cell = t2_cell(scene, kind, ts, twin2[(kind, ts)].next(), hand.next(), third2, {}, dict(change_stratum=st, base_task='T2'),
                       dart, task='T5')
        spec = cell['spec']
        spec['target'] = b                      # the twin is the new target, a (on target_slot) the old one
        spec['change'] = change(a, st)
        out.append(cell)
    # base T3
    cup_rr, order_rr = RR(['C1', 'C3']), RR([('B1', 'B2'), ('B2', 'B1')])
    other_rr = RRMap(lambda key: t3_valid_others(scene, *key))
    if split == 'demo':
        t3_grid = [(ts, st, dart) for ts in SLOTS for st in range(3) for dart in (False, True)]
    else:
        t3_grid = [(SLOTS[k % 6], k % 3, False) for k in range(16)]
    for ts, st, dart in t3_grid:
        old = cup_rr.next()
        new = 'C1' if old == 'C3' else 'C3'
        os_ = other_rr[(old, ts)].next()
        cell = t3_cell(new, os_, ts, order_rr.next(), dart, task='T5', change=change(old, st),
                       axes=dict(change_stratum=st, base_task='T3'))
        cell['spec']['layout'] = {old: ts, new: os_}
        cell['axes'].update(target_slot=ts, twin_slot=os_)
        out.append(cell)
    for c in out:
        c['optional_t5'] = True
    return out


def build_cells(split: str, cfgs: dict) -> list[dict]:
    scene = cfgs['T2'].scene
    cells = t1_cells(split, scene) + t2_cells(split, scene) + t3_cells(split, scene) + t4_cells(split, scene) \
        + t4neg_cells(split, scene) + t5_cells(split, scene, cfgs['T1'].scene)
    count = Counter()
    for c in cells:
        validate_spec(cfgs[c['base']], c['spec'])
        c['scenario_id'] = SID[c['base']]
        c.setdefault('optional_t5', False)
        c['extra'] = False
        c['cell_id'] = f"{split}-{c['task']}-{count[c['task']]:03d}"
        count[c['task']] += 1
    return cells


# ---------------------------------------------------------------------------------------------------- seed selection
_SC: dict = {}


def _scenario(which: str, sid: str):
    from intent_policy.scenarios.scenario_registry import make_scenario, scenario_overrides
    if (which, sid) not in _SC:
        exp = load_yaml(f'configs/experiments/intent_act_{which}.yaml')
        _SC[(which, sid)] = (make_scenario(sid, scenario_overrides(exp, sid)), exp)
    return _SC[(which, sid)]


def run_expert(which: str, cell: dict, seed: int) -> dict:
    from intent_policy.benchmark.runner import ExpertAgent, NoisyExpertAgent, run_episode
    from intent_policy.utils import controller_config
    sc, exp = _scenario(which, cell['scenario_id'])
    noise = exp['data'].get('noise') or {}
    agent = NoisyExpertAgent(noise['burst_prob'], noise['burst_ticks']) if cell['dart'] else ExpertAgent()
    rec = run_episode(sc, agent, seed, controller_config(exp), keep_trace=False, spec=cell['spec'])
    out = dict(success=bool(rec['success']), failure=rec['failure'], duration_s=round(rec['duration_s'], 2))
    if cell['base'] == 'T4':
        out.update(entered_zone=bool(getattr(sc, 'ever_in_zone', False)), phase=sc.human.extras.get('phase'))
    if cell['spec'].get('change'):
        out.update(change_state=sc.change_robot_state, cancelled='change_dropped' in sc.human.extras,
                   changed=sc.change_t is not None)
    return out


def select_seed(cell: dict) -> dict:
    tries = []
    for a in range(ATTEMPTS):
        seed = cell['seed_base'] + a
        res = {w: run_expert(w, cell, seed) for w in ('late', 'early')}
        ok = all(r['success'] for r in res.values())
        if ok and cell['spec'].get('change'):
            ok = all(r['changed'] and not r['cancelled'] for r in res.values())
        tries.append(dict(seed=seed, **{w: r for w, r in res.items()}))
        if ok:
            return dict(cell_id=cell['cell_id'], seed=seed, ok=True, tries=tries)
    return dict(cell_id=cell['cell_id'], seed=None, ok=False, tries=tries)


def run_selection(cells: list[dict], workers: int) -> dict:
    t0, out = time.time(), {}
    with mp.get_context('spawn').Pool(workers) as pool:
        for n, r in enumerate(pool.imap_unordered(select_seed, cells, chunksize=2)):
            out[r['cell_id']] = r
            if (n + 1) % 25 == 0 or n + 1 == len(cells):
                bad = sum(not x['ok'] for x in out.values())
                print(f'  [{n + 1}/{len(cells)}] failed cells {bad}  [{time.time() - t0:.0f}s]', flush=True)
    return out


def t4_extra_cells(split, cells, pilot, base_index: int) -> list[dict]:
    """Plan §3 T4: a stratum with > 25 % episodes whose hand never entered the zone (late or early) gets extra cells
    (same stratum, slots / DART / colours cycled) until 12 episodes entered in both."""
    extra = []
    for st in range(6):
        mine = [c for c in cells if c['task'] == 'T4' and c['axes']['timing_stratum'] == st and pilot[c['cell_id']]['ok']]
        entered = lambda c: all(pilot[c['cell_id']]['tries'][-1][w].get('entered_zone') for w in ('late', 'early'))
        n_in = sum(entered(c) for c in mine)
        if not mine or (len(mine) - n_in) / len(mine) <= 0.25:
            continue
        need = 12 - n_in
        slot_rr, color_rr, dart_rr, hold = RR(SLOTS), RR(COLORS), RR((False, True)), RR(range(2))
        for k in range(max(need, 0) * 3):        # up to 3x the missing count
            h = hold.next()
            extra.append(t4_cell('T4', slot_rr.next(), color_rr.next(), T4_STRATA[st], HOLD_STRATA[h], dart_rr.next(),
                                 dict(timing_stratum=st, hold_stratum=h)))
            extra[-1]['need_stratum'] = (st, need)
    for i, c in enumerate(extra):
        c.update(scenario_id=SID['T4'], optional_t5=False, extra=True, cell_id=f'{split}-T4-x{i:03d}',
                 seed_base=SEED_BASE[split] + 10 * (base_index + i))
    return extra


# ---------------------------------------------------------------------------------------------------- output
COLUMNS = ['episode_idx', 'task', 'cell_id', 'seed', 'dart', 'optional_t5', 'extra', 'target_slot', 'twin_slot',
           'pair_color', 'pair_type', 'third_obj', 'third_slot', 'place_zone', 'hand_zone', 'block_order', 'hand_goal',
           'base_task', 'timing_stratum', 'hand_lowered_first', 'hold_stratum', 'change_stratum']
TASK_ORDER = ('T1', 'T2', 'T3', 'T4', 'T4neg', 'T5')


def slot_of(c) -> str:
    return c['spec']['layout'][c['spec']['target']]


def order_block(cells, rng, prev: str | None = None, tries: int = 2000):
    """Shuffle a task block so that no two consecutive episodes (also across the previous block's last one, `prev`)
    share the target slot."""
    slot = slot_of
    for _ in range(tries):
        pool, out = [cells[i] for i in rng.permutation(len(cells))], []
        while pool:
            last = slot(out[-1]) if out else prev
            j = next((k for k, c in enumerate(pool) if slot(c) != last), None)
            if j is None:
                break
            out.append(pool.pop(j))
        if not pool:
            return out
    print(f'  warning: {cells[0]["task"]} cannot avoid consecutive target slots ({len(cells)} cells), kept as drawn')
    return [cells[i] for i in rng.permutation(len(cells))]


def write_outputs(split: str, kept: list[dict], out_dir):
    rng = np.random.default_rng(SEED_BASE[split] + 7)
    rows = []
    for task in TASK_ORDER:
        block = [c for c in kept if c['task'] == task]
        if block:
            rows += order_block(block, rng, slot_of(rows[-1]) if rows else None)
    with open(out_dir / f'{split}_list.jsonl', 'w') as f:
        for i, c in enumerate(rows):
            f.write(json.dumps(dict(index=i, task=c['task'], base=c['base'], scenario_id=c['scenario_id'], seed=c['seed'],
                                    spec=c['spec'], cell_id=c['cell_id'], dart=c['dart'], optional_t5=c['optional_t5'],
                                    extra=c['extra'], axes=c['axes'])) + '\n')
    with open(out_dir / f'{split}_list.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, COLUMNS)
        w.writeheader()
        for i, c in enumerate(rows):
            row = dict(episode_idx=i, task=c['task'], cell_id=c['cell_id'], seed=c['seed'], dart=c['dart'],
                       optional_t5=c['optional_t5'], extra=c['extra'], base_task=c['base'])
            row.update({k: v for k, v in c['axes'].items() if k in COLUMNS})
            w.writerow(row)
    return rows


def summary(split: str, rows: list[dict], failed: list[dict], pilot: dict) -> str:
    L = [f'## {split}: {len(rows)} episodes', '', '| task | episodes | DART | details |', '|---|---|---|---|']
    for task in TASK_ORDER:
        rs = [r for r in rows if r['task'] == task]
        if not rs:
            continue
        ax = lambda k: dict(sorted(Counter(r['axes'].get(k) for r in rs).items(), key=lambda kv: str(kv[0])))
        det = {'T1': f"target slot {ax('target_slot')}; place {ax('place_zone')}; pair colour {ax('pair_color')}",
               'T2': f"pair {ax('pair_type')}; hand {ax('hand_zone')}; stratum {ax('timing_stratum')}; lowered {ax('hand_lowered_first')}",
               'T3': f"target slot {ax('target_slot')}; block order {ax('block_order')}",
               'T4': f"stratum {ax('timing_stratum')}; slot {ax('target_slot')}; colour {ax('pair_color')}; extra {sum(r['extra'] for r in rs)}",
               'T4neg': f"goal {ax('hand_goal')}; stratum {ax('timing_stratum')}",
               'T5': f"base {ax('base_task')}; change stratum {ax('change_stratum')}"}[task]
        L.append(f"| {task} | {len(rs)} | {sum(r['dart'] for r in rs)} | {det} |")
    L += ['', f'Failed cells: {len(failed)}', '']
    # pilot statistics (the selected seed's late / early runs)
    t4 = [r for r in rows if r['task'] == 'T4']
    if t4:
        L += ['T4 (selected seeds): entered the robot zone / phase at intrusion, per stratum:', '',
              '| stratum | late entered | early entered | late phases | early phases |', '|---|---|---|---|---|']
        for st in range(6):
            rs = [r for r in t4 if r['axes']['timing_stratum'] == st]
            last = lambda r, w: pilot[r['cell_id']]['tries'][-1][w]
            L.append(f"| {st} | {sum(last(r, 'late')['entered_zone'] for r in rs)}/{len(rs)} | "
                     f"{sum(last(r, 'early')['entered_zone'] for r in rs)}/{len(rs)} | "
                     f"{dict(Counter(last(r, 'late')['phase'] for r in rs))} | {dict(Counter(last(r, 'early')['phase'] for r in rs))} |")
        L.append('')
    t5 = [r for r in rows if r['task'] == 'T5']
    if t5:
        L += ['T5 (selected seeds): robot state at the change of mind, per base and stratum:', '',
              '| base | stratum | late | early |', '|---|---|---|---|']
        for b, st in itertools.product(('T1', 'T2', 'T3'), range(3)):
            rs = [r for r in t5 if r['base'] == b and r['axes']['change_stratum'] == st]
            last = lambda r, w: pilot[r['cell_id']]['tries'][-1][w]['change_state']
            L.append(f"| {b} | {st} | {dict(Counter(last(r, 'late') for r in rs))} | {dict(Counter(last(r, 'early') for r in rs))} |")
        L.append('')
    return '\n'.join(L)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default=OUT)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--splits', nargs='*', default=['demo', 'eval'])
    p.add_argument('--cells-only', action='store_true', help='write the design cells (no seed selection)')
    p.add_argument('--limit-per-task', type=int, help='first N cells per task (trial runs)')
    p.add_argument('--reorder-only', action='store_true',
                   help='re-order the existing <split>_list.jsonl (seeds kept) and rewrite the list files')
    args = p.parse_args()
    cfgs = {k: ScenarioConfig.load(v) for k, v in SID.items()}
    out_dir = resolve(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = []
    if args.reorder_only:
        for split in args.splits:
            kept = [json.loads(l) for l in (out_dir / f'{split}_list.jsonl').read_text().splitlines()]
            write_outputs(split, kept, out_dir)
            print(f'{split}: re-ordered {len(kept)} episodes')
        return
    for split in args.splits:
        cells = build_cells(split, cfgs)
        if args.limit_per_task:
            per = Counter()
            cells = [c for c in cells if (per.update([c['task']]) or per[c['task']] <= args.limit_per_task)]
        for i, c in enumerate(cells):
            c['seed_base'] = SEED_BASE[split] + 10 * i
        print(f'{split}: {len(cells)} cells {dict(Counter(c["task"] for c in cells))}', flush=True)
        if args.cells_only:
            (out_dir / f'{split}_cells.json').write_text(json.dumps(cells, indent=1))
            continue
        pilot = run_selection(cells, args.workers)
        if split == 'demo' and not args.limit_per_task:
            extra = t4_extra_cells(split, cells, pilot, len(cells))
            if extra:
                print(f'  T4: {len(extra)} extra cells for strata with too few zone entries', flush=True)
                pilot.update(run_selection(extra, args.workers))
                kept_extra, have = [], Counter()
                for c in extra:                  # keep only as many extra cells as each stratum needs
                    st, need = c['need_stratum']
                    tr = pilot[c['cell_id']]
                    if tr['ok'] and have[st] < need and all(tr['tries'][-1][w]['entered_zone'] for w in ('late', 'early')):
                        kept_extra.append(c)
                        have[st] += 1
                cells += kept_extra
        failed = [c for c in cells if not pilot[c['cell_id']]['ok']]
        kept = []
        for c in cells:
            if pilot[c['cell_id']]['ok']:
                c['seed'] = pilot[c['cell_id']]['seed']
                kept.append(c)
        rows = write_outputs(split, kept, out_dir)
        (out_dir / f'{split}_pilot.json').write_text(json.dumps(pilot, indent=1))
        with open(out_dir / f'{split}_failed_cells.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['task', 'cell_id', 'seed', 'late_failure', 'early_failure', 'note'])
            for c in failed:
                for t in pilot[c['cell_id']]['tries']:
                    note = '' if not c['spec'].get('change') else \
                        f"late changed={t['late'].get('changed')} cancelled={t['late'].get('cancelled')}; " \
                        f"early changed={t['early'].get('changed')} cancelled={t['early'].get('cancelled')}"
                    w.writerow([c['task'], c['cell_id'], t['seed'], t['late']['failure'], t['early']['failure'], note])
        report.append(summary(split, rows, failed, pilot))
        print(report[-1], flush=True)
    if report:
        (out_dir / 'summary.md').write_text('# Episode lists v5 (docs/specs/EPISODE_PLAN_v5.md)\n\n' + '\n'.join(report) + '\n')


if __name__ == '__main__':
    main()

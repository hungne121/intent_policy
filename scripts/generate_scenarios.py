"""Generate the balanced demo / eval scenario lists (docs/requirements/scence_construct.md §5).

Every line of `<out>/<split>_v<version>.jsonl` is one episode: {index, task, base, scenario_id, seed, spec}. The spec
is the discrete episode definition read by the scenarios (intent_policy/scenarios/config.py); the seed drives the
continuous variation (position jitter, human timing). Rules (§5.2):
  1. the main variant combinations of each task appear equally often (difference <= 1):
     T1 target slot x P, T2 target slot x H (timing 1/3 each), T3 target cup x target slot (cube order in U 1/2 each),
     T4 phase x hold time (target slot balanced; one cube, place zone P1), T4-neg negative target x phase,
     T5 base task x change timing;
  2. the other objects are shuffled into the free slots (never a cup next to another object in a row: the open
     gripper would hit the cup wall);
  3. T1 / T2 (and T5 on T1 / T2): exactly 1/3 of the episodes have an identical pair on the table and the target
     is one of the pair;
  4. no two consecutive episodes share the target slot;
  5. one human (no participant_id, user decision 2026-09-29);
  6. fixed, stored seeds (configs/scenario_lists/generator.yaml).
The eval list uses another seed; the optional `holdout` combinations never appear in the demo list.

  ./run.sh -m scripts.generate_scenarios            # writes configs/scenario_lists/{demo,eval}_v1.jsonl + summary
"""
import argparse
import itertools
import json
from collections import Counter
import numpy as np
from intent_policy.scenarios.config import ScenarioConfig, cup_crowded, validate_spec, zones_of
from intent_policy.scenarios.scenario_registry import SCENARIO_OF_TASK
from intent_policy.utils import load_yaml, resolve

CUBES, CUPS = ['B1', 'B2', 'B3'], ['C1', 'C3']          # objects used when no identical pair is on the table
PAIRS = {'cube': ('B1', 'B1p'), 'cup': ('C1', 'C2')}


def balanced(combos: list, n: int, rng) -> list:
    """n items cycling through the combos in shuffled rounds: every combo count differs by at most 1."""
    out = []
    while len(out) < n:
        out += [combos[i] for i in rng.permutation(len(combos))]
    return out[:n]


def flags(n: int, k: int, rng) -> list[bool]:
    """Exactly k True among n, spread evenly (every n/k-th position, random phase)."""
    idx = set(np.floor((np.arange(k) + rng.random()) * n / max(k, 1)).astype(int)) if k else set()
    return [i in idx for i in range(n)]


class Builder:
    def __init__(self, cfgs: dict, rng, pair_fraction: float):
        self.cfgs, self.rng, self.pf = cfgs, rng, pair_fraction
        scene = cfgs['T1'].scene
        self.slots, self.places, self.hands = zones_of(scene, 'slot'), zones_of(scene, 'place'), zones_of(scene, 'hand')

    def pick(self, options):
        return options[int(self.rng.integers(len(options)))]

    def layout(self, target: str, target_slot: str, others: list[str]) -> dict:
        """Target on its slot, the others shuffled into the free slots (no cup next to another object in a row)."""
        free = [s for s in self.slots if s != target_slot]
        while True:
            chosen = [free[i] for i in self.rng.permutation(len(free))[:len(others)]]
            layout = {target: target_slot, **dict(zip(others, chosen))}
            if not cup_crowded(self.cfgs['T2'].scene, layout):
                return layout

    def objects(self, pair: str | None, kind: str, i: int, mixed: bool) -> tuple[str, list[str]]:
        """(target, other objects) for T1 / T2 / T4. With an identical pair (`pair` = cube | cup) the target is one
        of it (alternating); `kind` = target kind without a pair; `mixed`: cups may stand on the table (T2)."""
        pool = CUBES + CUPS if mixed else CUBES
        if pair:
            a, b = PAIRS[pair]
            target, twin = (a, b) if i % 2 == 0 else (b, a)
            return target, [twin, self.pick([k for k in pool if k not in (a, b)])]
        target = self.pick(CUPS if kind == 'cup' else CUBES)
        rest = [k for k in pool if k != target]
        return target, [rest[j] for j in self.rng.permutation(len(rest))[:2]]

    def t1(self, n: int, holdout: set, place_combos=True) -> list[dict]:
        combos = [c for c in itertools.product(self.slots, self.places) if f'T1:{c[0]}-{c[1]}' not in holdout]
        pairs = flags(n, round(n * self.pf), self.rng)
        out = []
        for i, (slot, place) in enumerate(balanced(combos, n, self.rng)):
            target, others = self.objects('cube' if pairs[i] else None, 'cube', i, mixed=False)
            out.append(dict(task='T1', target=target, layout=self.layout(target, slot, others), pair=pairs[i], place=place))
        return out

    def t2(self, n: int, holdout: set) -> list[dict]:
        combos = [c for c in itertools.product(self.slots, self.hands) if f'T2:{c[0]}-{c[1]}' not in holdout]
        timings = balanced(['early', 'on_time', 'late'], n, self.rng)
        pairs = flags(n, round(n * self.pf), self.rng)
        kinds = balanced(['cube', 'cup'], n, self.rng)
        out = []
        for i, (slot, hand) in enumerate(balanced(combos, n, self.rng)):
            target, others = self.objects(kinds[i] if pairs[i] else None, kinds[i], i, mixed=True)
            out.append(dict(task='T2', target=target, layout=self.layout(target, slot, others), pair=pairs[i],
                            hand=hand, timing=timings[i]))
        return out

    def t3(self, n: int) -> list[dict]:
        """Two cubes in U (order balanced), their paired cups on two slots; the target cup x its slot balanced."""
        pairs = self.cfgs['T3'].scene_variation['pairs']
        blocks, cups = list(pairs), list(pairs.values())
        order = balanced([0, 1], n, self.rng)
        out = []
        for i, (cup, slot) in enumerate(balanced(list(itertools.product(cups, self.slots)), n, self.rng)):
            out.append(dict(task='T3', target=cup, layout=self.layout(cup, slot, [c for c in cups if c != cup]),
                            u_blocks=blocks if order[i] == 0 else blocks[::-1]))
        return out

    def t4(self, n: int, negative: bool) -> list[dict]:
        """One cube (colour shuffled) on a balanced slot, placed in the fixed zone P1 (no instruction)."""
        main = list(itertools.product(['U', 'edge'], ['approach', 'carry', 'place'])) if negative else \
            list(itertools.product(['approach', 'carry', 'place'], [1.0, 2.0, 4.0]))
        slots = balanced(self.slots, n, self.rng)
        holds = balanced([1.0, 2.0, 4.0], n, self.rng)
        place = self.cfgs['T4'].scene_variation['place_zone']
        out = []
        for i, combo in enumerate(balanced(main, n, self.rng)):
            target = self.pick(CUBES)
            spec = dict(task='T4', target=target, layout={target: slots[i]}, pair=False, place=place)
            if negative:
                spec.update(phase=combo[1], hold_s=holds[i], negative=True, neg_target=combo[0])
            else:
                spec.update(phase=combo[0], hold_s=combo[1], negative=False, neg_target=None)
            out.append(spec)
        return out

    def t5(self, n: int, holdout: set) -> list[dict]:
        """Change of mind on T1 / T2 / T3 (1/3 each), early / late balanced; old / new pair shuffled. On T2 the
        hand-over timing is `on_time`, so the pointing hand is free when the human changes their mind."""
        combos = balanced(list(itertools.product(['T1', 'T2', 'T3'], ['early', 'late'])), n, self.rng)
        per = Counter(b for b, _ in combos)
        made = {'T1': self.t1(per['T1'], holdout), 'T2': self.t2(per['T2'], holdout), 'T3': self.t3(per['T3'])}
        out = []
        for base, timing in combos:
            spec = made[base].pop(0)
            if base == 'T2':
                spec['timing'] = 'on_time'
            old = self.pick([k for k in spec['layout'] if k != spec['target']])
            spec['change'] = dict(timing=timing, old=old)
            out.append((base, spec))
        return out


def no_repeat_slots(items: list, rng, tries: int = 2000) -> list:
    """Reorder so that no two consecutive episodes share the target slot (§5.2 rule 4)."""
    slot = lambda it: it[1]['layout'][it[1]['target']]
    for _ in range(tries):
        order = [items[i] for i in rng.permutation(len(items))]
        out, pool = [], list(order)
        while pool:
            j = next((k for k, it in enumerate(pool) if not out or slot(it) != slot(out[-1])), None)
            if j is None:
                break
            out.append(pool.pop(j))
        if not pool:
            return out
    raise RuntimeError('could not order the list without consecutive target slots')


def build(split: str, gcfg: dict, cfgs: dict) -> list[dict]:
    rng = np.random.default_rng(int(gcfg['seeds'][split]))
    b = Builder(cfgs, rng, float(gcfg['pair_fraction']))
    holdout = set(gcfg.get('holdout') or []) if split == 'demo' else set()
    counts = gcfg['counts'][split]
    blocks = [('T1', [('T1', s) for s in b.t1(counts['T1'], holdout)]),
              ('T2', [('T2', s) for s in b.t2(counts['T2'], holdout)]),
              ('T3', [('T3', s) for s in b.t3(counts['T3'])]),
              ('T4', [('T4', s) for s in b.t4(counts['T4'], negative=False)]),
              ('T4neg', [('T4', s) for s in b.t4(counts['T4neg'], negative=True)]),
              ('T5', b.t5(counts['T5'], holdout))]
    episodes, seed0 = [], int(gcfg['episode_seed_start'][split])
    for task, items in blocks:
        for base, spec in no_repeat_slots(items, rng):
            validate_spec(cfgs[base], spec)
            episodes.append(dict(index=len(episodes), task=task, base=base, scenario_id=SCENARIO_OF_TASK[base],
                                 seed=seed0 + len(episodes), spec=spec))
    for a, c in zip(episodes, episodes[1:]):     # a repeat across block boundaries: swap with a later episode of the block
        if slot_of(a['spec']) == slot_of(c['spec']):
            j = next(k for k in range(c['index'] + 2, len(episodes)) if episodes[k]['task'] == c['task']
                     and slot_of(episodes[k]['spec']) not in (slot_of(a['spec']), slot_of(episodes[k - 1]['spec']))
                     and (k + 1 >= len(episodes) or slot_of(episodes[k + 1]['spec']) != slot_of(c['spec'])))
            for f in ('base', 'scenario_id', 'spec'):
                c[f], episodes[j][f] = episodes[j][f], c[f]
    return episodes


def summary(name: str, episodes: list[dict], gcfg: dict) -> str:
    slot = lambda e: e['spec']['layout'][e['spec']['target']]
    lines = [f'# Scenario list `{name}`', '', f"{len(episodes)} episodes; generator seed {gcfg['seeds'][name.split('_')[0]]}.", '']
    lines += ['| task | episodes | main combination counts (min-max) | identical pair | other |', '|---|---|---|---|---|']
    for task in ('T1', 'T2', 'T3', 'T4', 'T4neg', 'T5'):
        eps = [e for e in episodes if e['task'] == task]
        if not eps:
            continue
        s = [e['spec'] for e in eps]
        key = {'T1': lambda x: (slot_of(x), x['place']), 'T2': lambda x: (slot_of(x), x['hand']),
               'T3': lambda x: (x['target'], slot_of(x)), 'T4': lambda x: (x['phase'], x['hold_s']),
               'T4neg': lambda x: (x['neg_target'], x['phase'])}.get(task, None)
        if task == 'T5':
            main = Counter((e['base'], e['spec']['change']['timing']) for e in eps)
        else:
            main = Counter(key(x) for x in s)
        pairs = sum(bool(x.get('pair')) for x in s)
        other = ''
        if task == 'T2':
            other = f"timing {dict(Counter(x['timing'] for x in s))}; cup targets {sum(x['target'].startswith('C') for x in s)}"
        elif task == 'T3':
            other = f"cube order in U {dict(Counter('-'.join(x['u_blocks']) for x in s))}"
        elif task in ('T4', 'T4neg'):
            other = f"hold {dict(Counter(x['hold_s'] for x in s))}; slots {dict(sorted(Counter(slot_of(x) for x in s).items()))}"
        elif task == 'T5':
            other = f"base {dict(Counter(e['base'] for e in eps))}"
        lines.append(f"| {task} | {len(eps)} | {len(main)} combos, {min(main.values())}-{max(main.values())} | "
                     f"{pairs} ({pairs / len(eps):.0%}) | {other} |")
    repeats = sum(slot(a) == slot(b) for a, b in zip(episodes, episodes[1:]))
    lines += ['', f'Consecutive episodes with the same target slot: {repeats}.',
              f"Holdout combinations (never in the demo list): {gcfg.get('holdout') or 'none'}.", '']
    return '\n'.join(lines)


def slot_of(spec: dict) -> str:
    return spec['layout'][spec['target']]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/scenario_lists/generator.yaml')
    p.add_argument('--out', default='configs/scenario_lists')
    args = p.parse_args()
    gcfg = load_yaml(args.config)
    cfgs = {code: ScenarioConfig.load(sid) for code, sid in SCENARIO_OF_TASK.items()}
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for split in ('demo', 'eval'):
        name = f"{split}_v{gcfg['version']}"
        episodes = build(split, gcfg, cfgs)
        with open(out / f'{name}.jsonl', 'w') as f:
            for e in episodes:
                f.write(json.dumps(e) + '\n')
        text = summary(name, episodes, gcfg)
        (out / f'{name}_summary.md').write_text(text)
        print(text)


if __name__ == '__main__':
    main()

"""G0 audit for transition-aware sampling (docs/PIF.md §8.13): per task / case / split / mode of a paired dataset
(scripts/collect_pairs.py layout <root>/<task>/<case>/<split>/<mode>/), what the episode metadata
(meta/hri_episodes.jsonl) records of the transitions: the human's cue onset, intention changes, forecast faults, the
expert's commits, and the human motion labels; episode / frame counts; then the transition index of
intent_policy/benchmark/transition_index.py and its share of the frames.
  ./run.sh scripts/audit_transitions.py [--root outputs/datasets/ten_task_v9] [--out outputs/diag/transitions_v9.json]
"""
from __future__ import annotations
import argparse
import collections
import json
from pathlib import Path

from intent_policy.benchmark.transition_index import index_episode, load_settings


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='outputs/datasets/ten_task_v9')
    p.add_argument('--config', default='configs/experiments/act_tais.yaml')
    p.add_argument('--out', default='outputs/diag/transitions_v9.json')
    args = p.parse_args()
    cfg = load_settings(args.config)
    rows, labels = [], collections.defaultdict(collections.Counter)
    for meta in sorted(Path(args.root).glob('*/*/*/*/meta/hri_episodes.jsonl')):
        mode_dir = meta.parent.parent
        task, case, split, mode = mode_dir.parts[-4:]
        if '.incomplete' in split or not (mode_dir.parent / 'DONE').exists():
            continue
        fps = float(json.loads((mode_dir / 'meta' / 'info.json').read_text())['fps'])
        r = collections.Counter(task=0)
        r.update(episodes=0, frames=0, cue=0, changes=0, faults=0, commits=0, tr_frames=0, dropped_leak=0)
        kinds = collections.Counter()
        for line in meta.open():
            m = json.loads(line)
            r['episodes'] += 1
            r['frames'] += int(m['n_frames'])
            r['cue'] += m.get('human_cue_onset_t') is not None
            r['changes'] += len(m.get('intention_changes') or [])
            r['faults'] += m.get('forecast_fault') is not None
            r['commits'] += len([c for c in m.get('expert_commits') or [] if c.get('frame') is not None])
            for x in m.get('human_motions') or []:
                labels[m['scenario_id']][x['label']] += 1
            ix = index_episode(m, fps, cfg)
            r['tr_frames'] += int(ix['mask'].sum())
            r['dropped_leak'] += ix['dropped_leak']
            kinds.update(k for k, _ in ix['transitions'])
        del r['task']
        rows.append(dict(task=task, case=case, split=split, mode=mode, **r, kinds=dict(kinds)))
    print('task case split mode | episodes frames | cue changes faults commits | transitions by kind | '
          'frames in P_T (share) | dropped by the leak check')
    for r in rows:
        print(f"{r['task']} {r['case']} {r['split']} {r['mode']} | {r['episodes']} {r['frames']} | {r['cue']} "
              f"{r['changes']} {r['faults']} {r['commits']} | {r['kinds']} | {r['tr_frames']} "
              f"({r['tr_frames'] / max(r['frames'], 1):.1%}) | {r['dropped_leak']}")
    print('\nhuman motion labels per scenario:')
    for s, c in labels.items():
        print(f'  {s}: {dict(c)}')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(dict(rows=rows, labels={s: dict(c) for s, c in labels.items()}, settings=cfg),
                                         indent=1))
    print('->', args.out)


if __name__ == '__main__':
    main()

"""Expert check of the v6 scenarios (docs/sim/scenarios_v6_proposal.md §7, M4): random-spec episodes per scenario with
the scripted expert, success / failure per episode, optional episode videos (recorded while it runs).
  ./run.sh -m scripts.check_v6 --scenarios t1_choose_item --seeds 0 10 --videos 3
  -> outputs/v6_check/<date_time>/{summary.md, results.json, <scenario>_seed<k>.mp4}
"""
import argparse
import datetime
import json
from collections import Counter

from intent_policy.benchmark.runner import ExpertAgent, run_episode
from intent_policy.benchmark.video import EpisodeVideo
from intent_policy.sim.base_env import ROOT
from intent_policy.scenarios.scenario_registry import make_scenario


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scenarios', nargs='+', required=True)
    p.add_argument('--seeds', nargs=2, type=int, default=[0, 10], help='seed range [a, b)')
    p.add_argument('--videos', type=int, default=0, help='record the first N episodes of each scenario')
    p.add_argument('--set', nargs='*', default=[], help='scenario overrides key.sub=value (yaml values)')
    args = p.parse_args()
    import yaml
    overrides: dict = {}
    for kv in args.set:
        k, v = kv.split('=', 1)
        d = overrides
        *path, last = k.split('.')
        for q in path:
            d = d.setdefault(q, {})
        d[last] = yaml.safe_load(v)
    out = ROOT / 'outputs/v6_check' / datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    out.mkdir(parents=True, exist_ok=True)
    results, lines = [], [f'# v6 expert check {out.name}', '', f'overrides: {overrides}', '']
    for sid in args.scenarios:
        sc = make_scenario(sid, overrides or None)
        outcomes = Counter()
        for i, seed in enumerate(range(*args.seeds)):
            video = EpisodeVideo(out, f'{sid}_seed{seed}') if i < args.videos else None
            rec = run_episode(sc, ExpertAgent(), seed, on_frame=video, keep_trace=False)
            name = f'{sid}_seed{seed}.mp4'
            if video is not None:
                video.finish(rec, out / name)
            ok = rec['success']
            outcomes['success' if ok else rec['failure']] += 1
            spec = rec.get('variation', {}).get('spec', {})
            gesture = rec.get('variation', {}).get('human', {}).get('object_gesture')
            row = dict(scenario=sid, seed=seed, success=ok, failure=rec['failure'], t=round(rec.get('duration_s') or 0, 2),
                       spec=spec, gesture=gesture)
            results.append(row)
            print(json.dumps(row), flush=True)
        sc.close()
        lines.append(f"- {sid}: {outcomes['success']}/{sum(outcomes.values())} success; " +
                     ', '.join(f'{k} {v}' for k, v in outcomes.items() if k != 'success'))
    (out / 'results.json').write_text(json.dumps(results, indent=1))
    (out / 'summary.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print(f'-> {out}')


if __name__ == '__main__':
    main()

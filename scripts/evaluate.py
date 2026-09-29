"""Evaluate an agent (trained checkpoint or scripted expert) on the benchmark scenarios.

Writes one episode record per episode (config, variation, per-step state, logits, probabilities,
selected action, mapped command, oracle given/true, events, metrics) and aggregates the HRIBench-style
metrics per split, scenario and HRI role (no composite score).

Without `--protocol` the experiment's `benchmark` section is used (Phase 1: one split). With a
protocol file (Phase 2) every split is evaluated; `--condition` picks an oracle condition from the
protocol (only for checkpoints that read oracle features). `--workers` runs scenario/split jobs in
parallel processes (each loads its own policy).

  ./run.sh -m scripts.evaluate --checkpoint <ckpt>/pretrained_model --output-dir outputs/eval/x
  ./run.sh -m scripts.evaluate --config configs/experiments/phase2_oracle.yaml \
      --protocol configs/benchmark/phase2_protocol_v1.yaml --checkpoint <ckpt> --condition delay_200ms \
      --output-dir outputs/eval/phase2/oracle_s0/delay_200ms --workers 3
"""
import argparse
import json
import multiprocessing as mp
from collections import defaultdict
from datetime import datetime, timezone

import torch

from benchmark.metrics import aggregate, ALL_METRICS
from benchmark.runner import ExpertAgent, run_episode, save_record
from scenarios.config import _deep_merge
from scenarios.scenario_registry import make_scenario, scenario_overrides
from scripts.common import load_yaml, resolve, observation_config, controller_config


def fmt(v):
    if v == 'n/a':
        return 'n/a'
    if v['value'] is None:
        return '— (n=0)'
    return f"{v['value']:.3f} ± {v['std']:.3f} (n={v['n']})"


def markdown(results: dict) -> str:
    agent = results['agent']
    lines = [f"# Evaluation — {agent.get('type')} ({agent.get('action_mode', '')})", '',
             f"checkpoint: `{agent.get('checkpoint', '-')}`  ", f"protocol: {results.get('protocol', '-')}  ",
             f"oracle condition: {agent.get('oracle_condition') or agent.get('intent_provider', 'none')}  ",
             f"created: {results['created']}", '']
    for split, sres in results['splits'].items():
        lines += [f"## Split: {split} (seeds {sres['seeds']})", '']
        for role, block in sres['by_role'].items():
            metrics = [m for m in ALL_METRICS if block['metrics'][m] != 'n/a']
            lines += [f'### Role: {role}', '', '| scenario | episodes | ' + ' | '.join(metrics) + ' |',
                      '|---|---|' + '---|' * len(metrics)]
            for sid in block['scenarios']:
                s = sres['by_scenario'][sid]
                lines.append(f"| {sid} | {s['episodes']} | " + ' | '.join(fmt(s['metrics'][m]) for m in metrics) + ' |')
            lines.append(f"| **all {role}** | {block['episodes']} | " + ' | '.join(fmt(block['metrics'][m]) for m in metrics) + ' |')
            lines.append('')
        lines += ['**Failure reasons:**', '']
        for sid, s in sres['by_scenario'].items():
            lines.append(f"- {sid}: {s['failures'] or 'none'}")
        lines.append('')
    return '\n'.join(lines) + '\n'


def build_agent(args_d: dict, eval_seeds: list[int], ctrl_cfg):
    if args_d['expert']:
        return ExpertAgent()
    from policies.policy_agent import PolicyAgent
    return PolicyAgent(resolve(args_d['checkpoint']), args_d['device'], args_d['oracle_condition'], eval_seeds, ctrl_cfg,
                       args_d['temporal_ensemble_coeff'])


def run_job(job: dict) -> dict:
    """One (split, scenario) block: returns per-episode metrics; records are written to disk."""
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    torch.set_num_threads(max(1, job['threads']))
    exp = job['exp']
    obs_cfg, ctrl_cfg = observation_config(exp), controller_config(exp)
    agent = build_agent(job['args'], job['seeds'], ctrl_cfg)
    ov = _deep_merge(scenario_overrides(exp, job['scenario']),
                     scenario_overrides({'scenario_overrides': job['split_overrides']}, job['scenario']))
    sc = make_scenario(job['scenario'], ov)
    out = resolve(job['output_dir'])
    eps, failures = [], defaultdict(int)
    try:
        for seed in job['seeds']:
            rec = run_episode(sc, agent, seed, ctrl_cfg, obs_cfg, keep_trace=not job['no_traces'])
            rec['experiment'], rec['split'], rec['protocol'] = exp['experiment'], job['split'], job['protocol']
            save_record(rec, out / 'episodes' / job['split'] / job['scenario'] / f"{rec['episode_id']}.json.gz")
            eps.append(rec['metrics'])
            if not rec['success']:
                failures[rec['failure']] += 1
            print(f"[{job['split']}] {job['scenario']} seed={seed} success={rec['success']} failure={rec['failure']} "
                  f"T={rec['duration_s']:.1f}s decisions={rec['n_decisions']}", flush=True)
        role, applicable = sc.cfg.role, sc.cfg.applicable_metrics
    finally:
        sc.close()
    return dict(split=job['split'], scenario=job['scenario'], role=role, applicable=applicable, metrics=eps,
                failures=dict(failures), agent=agent.describe())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/experiments/foundation_baseline.yaml')
    p.add_argument('--protocol', help='versioned evaluation protocol (splits, seeds, oracle conditions)')
    p.add_argument('--checkpoint', help='pretrained_model dir; omit with --expert')
    p.add_argument('--expert', action='store_true', help='evaluate the privileged scripted expert (reference)')
    p.add_argument('--condition', default='correct', help='oracle condition name from the protocol')
    p.add_argument('--episodes-per-scenario', type=int)
    p.add_argument('--seed-start', type=int)
    p.add_argument('--scenarios', nargs='*')
    p.add_argument('--splits', nargs='*')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--device')
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--no-traces', action='store_true', help='do not keep per-step traces in records')
    p.add_argument('--temporal-ensemble-coeff', type=float, default=None,
                   help='inference-time ACT temporal ensembling (default: protocol `inference`, else the checkpoint config)')
    args = p.parse_args()
    exp = load_yaml(args.config)
    if args.protocol:
        protocol = load_yaml(args.protocol)
        splits = protocol['splits']
        oracle_condition = protocol['conditions'][args.condition]
        protocol_name = protocol['protocol']
        ensemble = protocol.get('inference', {}).get('temporal_ensemble_coeff', 'config')
    else:
        bench = exp['benchmark']
        splits = [dict(name='default', scenarios=bench['scenarios'], seed_start=bench['eval_seed_start'],
                       episodes_per_scenario=bench['eval_episodes_per_scenario'])]
        oracle_condition, protocol_name = {'condition': 'correct'}, exp['experiment']
        ensemble = 'config'
    if args.temporal_ensemble_coeff is not None:
        ensemble = args.temporal_ensemble_coeff
    jobs = []
    for split in splits:
        if args.splits and split['name'] not in args.splits:
            continue
        n = args.episodes_per_scenario or split['episodes_per_scenario']
        seed0 = split['seed_start'] if args.seed_start is None else args.seed_start
        for sid in split['scenarios']:
            if args.scenarios and sid not in args.scenarios:
                continue
            jobs.append(dict(exp=exp, split=split['name'], scenario=sid, seeds=list(range(seed0, seed0 + n)),
                             split_overrides=split.get('scenario_overrides'), protocol=protocol_name,
                             output_dir=args.output_dir, no_traces=args.no_traces,
                             threads=max(1, 12 // max(args.workers, 1)),
                             args=dict(expert=args.expert, checkpoint=args.checkpoint, device=args.device,
                                       oracle_condition=oracle_condition, temporal_ensemble_coeff=ensemble)))
    out = resolve(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.workers > 1:
        with mp.get_context('spawn').Pool(args.workers) as pool:
            blocks = pool.map(run_job, jobs, chunksize=1)
    else:
        blocks = [run_job(j) for j in jobs]
    results = dict(experiment=exp, protocol=protocol_name, agent=blocks[0]['agent'], condition=args.condition,
                   controller=controller_config(exp).to_dict(), observation=observation_config(exp).to_dict(),
                   created=datetime.now(timezone.utc).isoformat(), splits={})
    for split in dict.fromkeys(b['split'] for b in blocks):
        sblocks = [b for b in blocks if b['split'] == split]
        by_scenario = {b['scenario']: dict(role=b['role'], episodes=len(b['metrics']), metrics=aggregate(b['metrics']),
                                           failures=b['failures'], applicable_metrics=b['applicable'],
                                           episode_metrics=b['metrics']) for b in sblocks}
        by_role = {}
        for role in dict.fromkeys(b['role'] for b in sblocks):
            rb = [b for b in sblocks if b['role'] == role]
            by_role[role] = dict(scenarios=[b['scenario'] for b in rb], episodes=sum(len(b['metrics']) for b in rb),
                                 metrics=aggregate([m for b in rb for m in b['metrics']]))
        seeds = [j['seeds'] for j in jobs if j['split'] == split][0]
        results['splits'][split] = dict(seeds=[seeds[0], seeds[-1]], by_scenario=by_scenario, by_role=by_role,
                                        overall=aggregate([m for b in sblocks for m in b['metrics']]))
    (out / 'results.json').write_text(json.dumps(results, indent=2, default=str))
    (out / 'results.md').write_text(markdown(results))
    print(markdown(results))


if __name__ == '__main__':
    main()

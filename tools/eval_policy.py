"""Closed-loop evaluation of trained checkpoints (12/QĐ-HL; 08/QĐ-KT: our runner, lerobot only as model library).

Checkpoints x execution horizons x tasks x variants x seeds in the collection episodes (HRCEnv). Seeds must lie in
the val split 300800-300899 (03/TTr-HL #4); test seeds 300900-300999 are refused. One process, one model on the GPU
at a time (laptop rules). LeRobot-style run folder:

  <out>/command.txt, eval.log, summary.json, summary.md
  <out>/<model>_step<N>_n<k>/episodes.jsonl, eval_info.json, videos/<task>_<variant>_<seed>.mp4

    choom -n 500 -- ./run.sh -m tools.eval_policy --checkpoint <ckpt dir> [--checkpoint ...] --tasks I2 C3 \
        --seeds 300800 300801 --n-action-steps 1 10 --video 1
"""
import argparse
import gc
import json
import logging
import shlex
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = yaml.safe_load((ROOT / 'configs/hrc/collection.yaml').read_text())


def markdown(rows):
    head = ('| model | step | n_action_steps | task | n | success % | CT mean s (n) | grade A/B/C | observable viol. ep | '
            'min robot–person m | onset s | wall s/ep |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n')
    f = lambda x, d=2: '–' if x is None else f'{x:.{d}f}'
    lines = []
    for r in rows:
        a = r['metrics']
        g = a['grade_counts']
        lines.append(f"| {r['model']} | {r['step']} | {r['n_action_steps']} | {r['task']} | {a['n_episodes']} | "
                     f"{a['pc_success']:.0f} | {f(a['avg_ct_s'], 1)} ({a['n_ct']}) | {g['A']}/{g['B']}/{g['C']} | "
                     f"{a['episodes_with_observable_violation']} | {f(a['min_robot_person_m'], 3)} | "
                     f"{f(a['avg_robot_onset_s'])} | {f(a['eval_ep_s'], 0)} |")
    return head + '\n'.join(lines) + '\n'


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--checkpoint', action='append', required=True, help='checkpoint or pretrained_model folder')
    p.add_argument('--tasks', nargs='+', required=True)
    p.add_argument('--variants', nargs='+', default=['nominal'])
    p.add_argument('--group', default='N1', help='collection group of the variation (N1 nominal; N6 for switch_*)')
    p.add_argument('--fault-type', default=None, choices=[None, *CFG['groups']['N5']['fault_types']],
                   help='with --group N5 (14/QĐ-HL P3): forecast fault injected into the policy forecast tensor')
    p.add_argument('--seeds', nargs='+', type=int, default=None)
    p.add_argument('--start-seed', type=int, default=CFG['splits']['val'][0])
    p.add_argument('--n-seeds', type=int, default=1)
    p.add_argument('--n-action-steps', nargs='+', type=int, default=[1])
    p.add_argument('--forecast-condition', nargs='+', default=['true'], choices=['true', 'masked'],
                   help='20/QĐ-HL: run every episode under each condition (paired by seed); masked = forecast 0, valid 0')
    p.add_argument('--trace', action='store_true', help='20/QĐ-HL 5: save raw chunks (true / masked / executed), executed actions and TCP per episode')
    p.add_argument('--video', type=int, default=0, help='overhead review video for the first K seeds per task')
    p.add_argument('--device', default=None)
    p.add_argument('--out', type=Path, default=None)
    a = p.parse_args()

    if (a.group == 'N5') != (a.fault_type is not None):
        raise SystemExit('--group N5 needs --fault-type, and --fault-type needs --group N5')
    seeds = a.seeds or list(range(a.start_seed, a.start_seed + a.n_seeds))
    lo, hi = CFG['splits']['val']
    bad = [s for s in seeds if not lo <= s <= hi]
    if bad:
        raise SystemExit(f'seeds {bad} outside the val split {lo}-{hi} (12/QĐ-HL, 03/TTr-HL #4)')
    out = a.out or ROOT / 'outputs/eval_closed_loop' / time.strftime('%Y-%m-%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'command.txt').write_text(shlex.join([sys.executable, '-m', 'tools.eval_policy', *sys.argv[1:]]) + '\n')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler(out / 'eval.log'), logging.StreamHandler(sys.stdout)])
    log = logging.getLogger('eval').info
    log(f'out {out}; seeds {seeds}; tasks {a.tasks}; variants {a.variants}; n_action_steps {a.n_action_steps}')

    import torch
    from intent_policy.envs import HRCEnv, HRCEnvConfig
    from intent_policy.eval.runner import load_policy, evaluate
    env = HRCEnv(a.tasks[0], a.variants[0], HRCEnvConfig(group=a.group, fault_type=a.fault_type))
    rows, summary = [], []
    for ck in a.checkpoint:
        for n in a.n_action_steps:
            policy, pre, post, meta = load_policy(ck, n, a.device)
            tags = dict(model=meta['model'], step=meta['step'], n_action_steps=meta['n_action_steps'],
                        checkpoint=meta['path'], best_val_l1=meta['best_val_l1'])
            log(f'loaded {tags}')
            run0 = out / f"{meta['model']}_step{meta['step']}_n{meta['n_action_steps']}"
            for cond in a.forecast_condition:
                run = run0 if cond == 'true' and a.forecast_condition == ['true'] else run0 / f'forecast_{cond}'
                info = evaluate(policy, pre, post, env, a.tasks, a.variants, seeds, run, dict(tags, forecast_condition=cond), a.video, log,
                                a.fault_type, cond, (run / 'trajectories') if a.trace else None)
                summary.append(dict(tags, forecast_condition=cond, per_task=info['per_task'], aggregated=info['aggregated']))
                for key, m in info['per_task'].items():
                    rows.append(dict(model=meta['model'] + ('' if len(a.forecast_condition) == 1 else f'[{cond}]'), step=meta['step'], n_action_steps=n, task=key, metrics=m))
            del policy, pre, post
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    env.close()
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    (out / 'summary.md').write_text(f'# Closed-loop eval {time.strftime("%Y-%m-%d %H:%M")}\n\nseeds {seeds}\n\n'
                                    + markdown(rows))
    log('\n' + markdown(rows))


if __name__ == '__main__':
    main()

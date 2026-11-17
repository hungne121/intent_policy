"""Collection runner (QĐ 53/QĐ-DL §4, §7, §8; 59/QĐ-DL §E-§G). Runs the observed and the forecast rollout of the
same seed / human scenario as two separate rollouts, records them with the shared LeRobot recorder, validates each
episode, writes accepted episodes to datasets/hrc_<mode>, rejected ones (summary + trace) to rejected_episodes/, and
a JSONL report. Resumable: an episode id already accepted or rejected is skipped. Never starts on its own: the
coordinator picks the group, tasks, variants, split and the number of seeds.

Groups (configs/hrc/collection.yaml `groups`, 59/QĐ-DL):
  N1       nominal, diverse: scenario draws + generic N1 variation (speed, start shift, hesitation, robot start)
  N5       forecast fault (spatial / temporal / direction) injected right after ManiCast; forecast mode only
  N6       change of mind, only the previewed cases (I3, C4 switch_before / switch_after_commit)
  PREVIEW  the 17/11 preview cases without generic variation (regression check)

    ./run.sh -m tools.collect --group N1 --tasks I1 --split train --n-seeds 2
    ./run.sh -m tools.collect --group N5 --tasks C4 --fault-types spatial_fault --split train --n-seeds 1
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import yaml

from intent_policy.shared.sim_env import ROOT
from intent_policy.shared.manicast_adapter import load_predictor
from intent_policy.shared.variation import sample_variation
from intent_policy.shared.forecast_fault import sample_fault
from intent_policy.experts import TASKS
from intent_policy.data.lerobot_recorder import LeRobotRecorder
from intent_policy.data.episode_validator import validate
from tools.preview import run_episode, variants

CFG = yaml.safe_load((ROOT / 'configs/hrc/collection.yaml').read_text())


def split_seeds(split, n, start=None):
    lo, hi = CFG['splits'][split]
    lo = lo if start is None else max(lo, start)
    return list(range(lo, min(hi + 1, lo + n)))


def task_cfg(block, task):
    """A config block with its per-task overrides applied."""
    c = copy.deepcopy({k: v for k, v in CFG[block].items() if k != 'per_task'})
    for k, v in (CFG[block].get('per_task') or {}).get(task, {}).items():
        c[k] = v
    return c


def group_variants(group, task):
    g = CFG['groups'][group]['variants']
    v = g.get(task, g['default'])
    return list(variants(task)) if v == 'all' else [x for x in v if x in variants(task)]


def forecast_source(mode, fault_type):
    if mode == 'observed':
        return 'manicast_recorded_not_used_by_expert'
    return 'manicast' if fault_type is None else f'manicast+{fault_type}'


def jsonable(x):
    return json.loads(json.dumps(x, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--group', choices=list(CFG['groups']), required=True)
    p.add_argument('--tasks', nargs='+', default=list(TASKS))
    p.add_argument('--modes', nargs='+', default=None, help='default: the group modes')
    p.add_argument('--variants', nargs='*', default=None, help='default: the group variants of each task')
    p.add_argument('--fault-types', nargs='*', default=None, help='N5 only; default: all three')
    p.add_argument('--split', choices=list(CFG['splits']), required=True)
    p.add_argument('--n-seeds', type=int, required=True)
    p.add_argument('--first-seed', type=int, default=None)
    p.add_argument('--root', type=Path, default=ROOT / CFG['datasets']['root'])
    p.add_argument('--review-video', action='store_true', help='also write the top review video per episode')
    p.add_argument('--record-rejected', action='store_true',
                   help='PILOT DIAGNOSTIC ONLY: also write rejected episodes into the dataset with accepted=false, to '
                        'test LeRobot write / read-back / replay while no episode passes the physics gate. Refused '
                        'on the main dataset root.')
    a = p.parse_args()
    if a.record_rejected and a.root.resolve() == (ROOT / CFG['datasets']['root']).resolve():
        raise SystemExit('--record-rejected is refused on the main dataset root')
    grp = CFG['groups'][a.group]
    modes = a.modes or grp['modes']
    faults = [None] if a.group != 'N5' else (a.fault_types or grp['fault_types'])
    use_variation = grp.get('variation', True)
    ds = CFG['datasets']
    recorders = {m: LeRobotRecorder(a.root / ds[m], f'local/{ds[m]}', CFG['fps'], CFG['cameras'], CFG['image_hw'],
                                    CFG['robot_type']) for m in modes}
    rejected_root = a.root / ds['rejected']; reports = a.root / ds['reports']
    reports.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    report = reports / f'collect_{stamp}_{a.group}.jsonl'
    review_root = reports / f'review_{stamp}_{a.group}'
    done = {m: recorders[m].recorded_ids() for m in modes}
    done_rejected = {p.parent.name for p in rejected_root.glob('*/summary.json')} if rejected_root.exists() else set()
    predictor = load_predictor()
    try:
        for task in a.tasks:
            var_cfg = task_cfg('n1', task); fault_cfg = task_cfg('n5', task)
            for variant in (a.variants or group_variants(a.group, task)):
                if variant not in variants(task):
                    continue
                for fault_type in faults:
                    for seed in split_seeds(a.split, a.n_seeds, a.first_seed):
                        variation = sample_variation(seed, var_cfg) if use_variation else None
                        spec = None if fault_type is None else sample_fault(fault_type, seed, fault_cfg)
                        for mode in modes:
                            tag = '' if fault_type is None else f'-{fault_type}'
                            eid = f'{a.group}-{task}-{variant}{tag}-{seed}-{mode}'
                            if eid in done[mode] or eid in done_rejected:
                                continue
                            rec = recorders[mode]
                            out = review_root if a.review_video else None
                            t0 = time.time()
                            s = run_episode(task, mode, seed, predictor, out=out, recorder=rec, review=a.review_video,
                                            variant=variant, variation=variation, fault=spec,
                                            name_suffix=f'{a.group}_{seed}_' + ('' if fault_type is None else fault_type + '_'))
                            arrays = s.pop('_arrays', None)
                            ok, reasons = validate(s, rec.frames, round(s['duration_s'] / 0.05))
                            fault = s.get('fault') or {}
                            meta = jsonable({
                                'episode_id': eid, 'group': a.group, 'task_name': task, 'task_variant': variant,
                                'expert_mode': mode, 'seed': seed, 'split': a.split, 'success': s['success'],
                                'failure_reason': s['failure_reason'], 'duration': s['duration_s'],
                                'physics_valid': s['physics_valid'], 'forecast_validity_summary': s['forecast_validity_summary'],
                                'forecast_source': forecast_source(mode, fault_type),
                                'fault_type': fault_type, 'fault_start': fault.get('fault_start'), 'fault_end': fault.get('fault_end'),
                                'fault_parameters': fault.get('fault_parameters'), 'fault_checks': {k: fault.get(k) for k in (
                                    'fault_applied', 'active_ticks', 'max_wrist_shift_1s_m', 'max_bone_dev_m',
                                    'max_start_err_m', 'max_jump_m', 'clamped_ticks')} if fault else None,
                                'variation': s.get('variation'), 'variation_log': s.get('variation_log'),
                                'natural_forecast_error': s.get('natural_forecast_error'),
                                'expert_version': CFG['versions']['expert'], 'manicast_version': CFG['versions']['manicast'],
                                'controller_version': CFG['versions']['controller'], 'schema_version': CFG['schema_version'],
                                'dataset_version': CFG['dataset_version'], 'wall_s': round(time.time() - t0, 1),
                                'final_positions': s['final_positions'], 'events': s['events']})
                            meta['accepted'] = ok; meta['reject_reasons'] = reasons
                            if ok or a.record_rejected:
                                idx = rec.accept(meta, arrays)
                            else:
                                rec.reject(); idx = None
                            if not ok:
                                rej = rejected_root / eid
                                rej.mkdir(parents=True, exist_ok=True)
                                (rej / 'summary.json').write_text(json.dumps(jsonable({**s, 'reject_reasons': reasons}), indent=2) + '\n')
                            row = {'episode_id': eid, 'accepted': ok, 'episode_index': idx, 'reasons': reasons,
                                   'success': s['success'], 'duration_s': s['duration_s'], 'wall_s': meta['wall_s'],
                                   'frames': rec.frames if ok else None}
                            with report.open('a') as f:
                                f.write(json.dumps(row) + '\n')
                            print(json.dumps(row), flush=True)
    finally:
        for r in recorders.values():
            r.finalize()


if __name__ == '__main__':
    main()

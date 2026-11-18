"""Data check before step-2 training (07/QĐ-HL B; docs/lo_trinh/BUOC_2_baseline_M0_M1.md checklist B).

Reads every LeRobot root under `--root` (a collection root with hrc_observed / hrc_forecast, a folder of shards, or a
train set of tools/build_train_set.py with D_observed|D_forecast/{train,val}) with the LeRobot API and checks:
  cameras        image keys = configs/hrc/collection.yaml cameras in that order, 256 x 256 x 3, decoded sample shape
  robot_state    observation.state names q0-5, qd0-5, gripper; joint order verified by UR3e FK(qpos) vs the recorded
                 sim TCP (observation.tcp_pose), max error < 1 mm
  action         7 = 6 joint targets [rad] (|target - qpos| small) + gripper [m] within the slide range 0-0.13
  time           fps 20, frame_index 0..n-1, timestamp = frame_index / fps, global index contiguous
  forecast       180 = 10 x 6 x 3; validity; invalid -> zeros; joint order (horizon 0.1 s near the observed joints,
                 not the left/right swap); forecast_provided 0 in observed
  boundaries     episode index ranges contiguous; an action chunk (chunk_size) never takes frames of the next episode
                 (action_is_pad from LeRobot's delta_timestamps)
  split          seeds within the collection.yaml ranges, no test seed, no seed / episode id in two splits
  balance        episodes / frames per task, variant, group
  policy_inputs  the exact keys each model consumes (configs/policy/m0|m1_noh|m1.yaml); none is a task id, FSM phase,
                 hidden expert state or future ground truth
  stats          train set only: stats.json mean / std of observation.state and action equal a direct computation
Writes <out>/check_train_data.json + .md.
    ./run.sh -m tools.check_train_data --root outputs/diag/preview_d1_1118/ds --out outputs/diag/step2_1118/check_preview
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
STATE_NAMES = [f'q{i}' for i in range(6)] + [f'qd{i}' for i in range(6)] + ['gripper']
ACTION_NAMES = [f'q{i}_target' for i in range(6)] + ['gripper_cmd']
GRIPPER_RANGE_M = (0.0, 0.13)          # assets/scene.xml gripper_joint range
# frame features no model may read: task id, expert hidden state / FSM-related, future ground truth, privileged sim state
FORBIDDEN = {'task_index': 'task id', 'observation.tcp_reference': 'expert controller reference (hidden state)',
             'observation.tcp_reference_valid': 'expert controller state', 'observation.forecast_provided':
             'expert used the forecast (hidden state)', 'observation.forecast_age': 'recorder bookkeeping',
             'observation.human_pose': 'privileged sim keypoints (not a step-2 input)',
             'observation.human_history': 'ManiCast input history (not a step-2 input)',
             'observation.human_history_valid': 'ManiCast history flag', 'observation.tcp_pose': 'sim TCP (FK check only)'}
MODELS = {'M0': 'configs/policy/m0.yaml', 'M1-noH': 'configs/policy/m1_noh.yaml', 'M1': 'configs/policy/m1.yaml'}


def lerobot_roots(root: Path) -> list[Path]:
    if (root / 'meta' / 'info.json').exists():
        return [root]
    return sorted(p.parent.parent for p in root.glob('**/meta/info.json') if '_filtered' not in str(p))


def model_inputs(path: Path) -> dict:
    y = yaml.safe_load((ROOT / path).read_text())
    keys = [f'observation.images.{c}' for c in y['train_data']['cameras']] + ['observation.state']
    if y.get('use_forecast'):
        keys += [y['forecast_key'], y['forecast_valid_key']]
    return dict(dataset=y['train_data']['dataset'], inputs=keys, outputs=['action'])


class Report:
    def __init__(self):
        self.checks = []

    def add(self, scope, name, ok, detail):
        self.checks.append(dict(scope=scope, check=name, ok=bool(ok), detail=detail))
        return ok


def check_root(lr: Path, rep: Report, coll: dict, chunk: int, n_fk: int, rng) -> dict:
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from intent_policy.policies.forecast_token import TcpFK, ur3e_tcp_chain
    scope = str(lr)
    info = json.loads((lr / 'meta/info.json').read_text())
    feats = info['features']
    fps = info['fps']
    cams = [f'observation.images.{c}' for c in coll['cameras']]
    img_keys = [k for k, v in feats.items() if v['dtype'] in ('video', 'image')]
    hw = tuple(coll['image_hw'])
    rep.add(scope, 'cameras', img_keys == cams and all(tuple(feats[k]['shape']) == (*hw, 3) for k in img_keys),
            dict(order=img_keys, expected=cams, shapes={k: feats[k]['shape'] for k in img_keys},
                 codec={k: feats[k].get('info', {}).get('video.codec') for k in img_keys}))
    rep.add(scope, 'state_names', feats['observation.state']['names'] == STATE_NAMES, feats['observation.state']['names'])
    rep.add(scope, 'action_names', feats['action']['names'] == ACTION_NAMES and feats['action']['shape'] == [7],
            feats['action']['names'])
    rep.add(scope, 'fps', fps == coll['fps'], dict(fps=fps, expected=coll['fps']))
    rep.add(scope, 'forecast_shape', feats['observation.human_forecast']['shape'] == [180]
            and feats['observation.forecast_valid']['shape'] == [1], '180 = 10 horizons x 6 joints x 3')

    ds = LeRobotDataset(f'local/check_{lr.name}', root=lr, delta_timestamps={'action': [i / fps for i in range(chunk)]})
    cols = ['episode_index', 'frame_index', 'timestamp', 'index', 'observation.state', 'action', 'observation.tcp_pose',
            'observation.human_forecast', 'observation.forecast_valid', 'observation.human_pose',
            'observation.forecast_provided']
    d = ds.hf_dataset.with_format('numpy', columns=cols)[:]
    d = {k: np.asarray(v) for k, v in d.items()}
    ep, fr, ts, ix = d['episode_index'], d['frame_index'], d['timestamp'], d['index']
    st, act = d['observation.state'].astype(np.float64), d['action'].astype(np.float64)

    # time
    bad_fr, max_dt = [], 0.0
    for e in np.unique(ep):
        m = ep == e
        if not np.array_equal(fr[m], np.arange(m.sum())):
            bad_fr.append(int(e))
        max_dt = max(max_dt, float(np.abs(ts[m] - fr[m] / fps).max()))
    rep.add(scope, 'timestamps', not bad_fr and max_dt < 1e-4 and np.array_equal(ix, np.arange(len(ix)) + ix[0]),
            dict(episodes_bad_frame_index=bad_fr, max_abs_timestamp_minus_frame_over_fps_s=max_dt))

    # boundaries + padding
    E = ds.meta.episodes
    fr0, to0 = np.asarray(E['dataset_from_index']), np.asarray(E['dataset_to_index'])
    contiguous = bool(fr0[0] == 0 and np.array_equal(fr0[1:], to0[:-1]) and to0[-1] == ds.meta.total_frames)
    pad_ok, pad_detail = True, []
    for e in rng.choice(len(E), min(3, len(E)), replace=False):
        end = int(to0[e])
        for back in (1, 5):
            s = ds[end - back]
            want = np.r_[np.zeros(back, bool), np.ones(chunk - back, bool)]
            real = act[end - back:end]
            ok = (np.array_equal(s['action_is_pad'].numpy(), want) and int(s['episode_index']) == int(ep[end - 1])
                  and np.allclose(s['action'][:back].numpy(), real, atol=1e-6)
                  and np.allclose(s['action'][back:].numpy(), act[end - 1], atol=1e-6))
            pad_ok &= ok
            pad_detail.append(dict(episode=int(e), frames_before_end=back, ok=bool(ok)))
    rep.add(scope, 'episode_boundaries', contiguous and pad_ok,
            dict(contiguous=contiguous, chunk=chunk, padding_samples=pad_detail,
                 note='beyond the episode end LeRobot repeats the last action and sets action_is_pad (masked in L1)'))

    # images decoded
    s = ds[int(rng.integers(len(ds)))]
    shp = {k: list(s[k].shape) for k in img_keys}
    rng_px = {k: [round(float(s[k].min()), 3), round(float(s[k].max()), 3)] for k in img_keys}
    rep.add(scope, 'image_decode', all(v == [3, hw[0], hw[1]] for v in shp.values()), dict(shapes=shp, value_range=rng_px))

    # action units
    dq = np.abs(act[:, :6] - st[:, :6])
    g_a, g_s = act[:, 6], st[:, 12]
    # units: a joint target in rad tracks qpos (median gap << 0.01 rad); large gaps are listed, not failed: the servo
    # lags fast moves (I2 elbow ~1.6 rad/s) and the expert steps wrist_3 by ~pi/2 (C3), measured on the preview 18/11
    big = np.argwhere(dq > 0.3)
    rep.add(scope, 'action_units', np.median(dq) < 0.01 and g_a.min() >= GRIPPER_RANGE_M[0] - 1e-6 and g_a.max() <= GRIPPER_RANGE_M[1] + 1e-6,
            dict(joint_target_minus_qpos_rad=dict(p50=float(np.median(dq)), p99=float(np.quantile(dq, 0.99)), max=float(dq.max())),
                 frames_gap_over_0p3_rad=int(len(np.unique(big[:, 0]))), gap_frames_per_joint=np.bincount(big[:, 1], minlength=6).tolist(),
                 gap_examples=[dict(episode=int(ep[i]), frame=int(fr[i]), joint=int(j), qpos=round(float(st[i, j]), 3),
                                    target=round(float(act[i, j]), 3), qvel=round(float(st[i, 6 + j]), 3)) for i, j in big[:6]],
                 gripper_cmd_m=[float(g_a.min()), float(g_a.max())], gripper_state_m=[float(g_s.min()), float(g_s.max())],
                 units='6 joint position targets [rad] + gripper command [m] (dataset_schema.py)'))

    # FK vs recorded sim TCP
    fk = TcpFK(ur3e_tcp_chain())
    pick = rng.choice(len(st), min(n_fk, len(st)), replace=False)
    with torch.no_grad():
        p = fk(torch.as_tensor(st[pick, :6])).numpy()
    err = np.linalg.norm(p - d['observation.tcp_pose'][pick, :3], axis=1) * 1000
    perm = st[pick, :6][:, [1, 0, 2, 3, 4, 5]]
    with torch.no_grad():
        err_perm = np.linalg.norm(fk(torch.as_tensor(perm)).numpy() - d['observation.tcp_pose'][pick, :3], axis=1) * 1000
    rep.add(scope, 'joint_order_fk', err.max() < 1.0, dict(frames=len(pick), fk_tcp_error_mm=dict(
        mean=float(err.mean()), max=float(err.max())), swapped_q0_q1_error_mm_mean=float(err_perm.mean())))

    # forecast validity / joint order
    fc = d['observation.human_forecast'].reshape(-1, 10, 6, 3).astype(np.float64)
    val = d['observation.forecast_valid'].reshape(-1) > 0.5
    inval_zero = bool(np.all(fc[~val] == 0)) if (~val).any() else True
    hp = d['observation.human_pose'].reshape(-1, 6, 3)
    swap = [1, 0, 3, 2, 5, 4]
    e_id = np.linalg.norm(fc[val, 0] - hp[val], axis=-1).mean() if val.any() else float('nan')
    e_sw = np.linalg.norm(fc[val, 0][:, swap] - hp[val], axis=-1).mean() if val.any() else float('nan')
    rep.add(scope, 'forecast_validity', inval_zero and (not val.any() or e_id < e_sw), dict(
        valid_fraction=float(val.mean()), invalid_frames_all_zero=inval_zero,
        h0_vs_observed_joint_m=dict(identity=float(e_id), left_right_swapped=float(e_sw)),
        joints='l_shoulder, r_shoulder, l_elbow, r_elbow, l_wrist, r_wrist (sim_env.JOINTS)'))
    prov = d['observation.forecast_provided'].reshape(-1)
    mode = 'observed' if 'observed' in str(lr) else 'forecast'
    rep.add(scope, 'forecast_provided', (prov.max() == 0) if mode == 'observed' else True,
            dict(mode=mode, provided_fraction=float(prov.mean())))

    # stats of a built train set
    if (lr / 'meta/stats_source.json').exists():
        stats = json.loads((lr / 'meta/stats.json').read_text())
        dev = {}
        for k, x in (('observation.state', st), ('action', act)):
            sd = x.std(0)
            dev[k] = dict(mean=float(np.abs(np.asarray(stats[k]['mean']) - x.mean(0)).max()),
                          std=float(np.abs(np.asarray(stats[k]['std']) - sd).max()),
                          std_ok=bool(np.all(np.abs(np.asarray(stats[k]['std']) - sd) < np.maximum(0.01 * sd, 5e-3))))
        # pooled per-episode stats (float32): mean exact to ~1e-5, std within max(1 %, 5e-3) per dim (smoke set 18/11:
        # <= 3.2e-3 abs on near-constant dims such as wrist_2)
        rep.add(scope, 'stats_from_own_episodes', all(v['mean'] < 1e-3 and v['std_ok'] for v in dev.values()),
                dict(max_abs_dev_vs_direct=dev, source=json.loads((lr / 'meta/stats_source.json').read_text())))

    # metadata
    metas = [json.loads(l) for l in (lr / 'meta/hrc_episodes.jsonl').read_text().splitlines() if l.strip()]
    lengths = {int(i): int(t) - int(f) for i, f, t in zip(E['episode_index'], fr0, to0)}
    rows = [dict(episode_id=m['episode_id'], task=m['task_name'], variant=m['task_variant'], group=m['group'],
                 seed=int(m['seed']), split=m['split'], grade=m.get('grade', m.get('quality_grade')),
                 mode=m.get('expert_mode', m.get('mode')), frames=lengths[k], dataset_version=m.get('dataset_version'))
            for k, m in enumerate(metas)]
    rep.add(scope, 'metadata_rows', len(metas) == ds.meta.total_episodes, dict(rows=len(metas), episodes=ds.meta.total_episodes))
    return dict(rows=rows, features=sorted(feats), frames=int(ds.meta.total_frames))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True, help='output prefix (folder created)')
    p.add_argument('--chunk', type=int, default=40, help='ACT chunk size for the boundary check (03/QĐ-HL)')
    p.add_argument('--fk-frames', type=int, default=200)
    p.add_argument('--collection-config', type=Path, default=ROOT / 'configs/hrc/collection.yaml')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    coll = yaml.safe_load(a.collection_config.read_text())
    rng = np.random.default_rng(a.seed)
    rep = Report()
    roots = lerobot_roots(a.root)
    if not roots:
        raise SystemExit(f'no LeRobot root under {a.root}')
    per_root, all_rows = {}, []
    for lr in roots:
        r = check_root(lr, rep, coll, a.chunk, a.fk_frames, rng)
        rel = str(lr.relative_to(a.root)) if lr != a.root else lr.name
        per_root[rel] = dict(frames=r['frames'], episodes=len(r['rows']))
        all_rows += [dict(x, root=rel) for x in r['rows']]
        feats = r['features']

    # split leakage
    rng_split = {k: tuple(v) for k, v in coll['splits'].items()}
    wrong = [x for x in all_rows if not (rng_split[x['split']][0] <= x['seed'] <= rng_split[x['split']][1])]
    test = [x for x in all_rows if rng_split['test'][0] <= x['seed'] <= rng_split['test'][1] or x['split'] == 'test']
    seed_split = {}
    for x in all_rows:
        seed_split.setdefault((x['task'], x['seed']), set()).add(x['split'])
    both = [f'{t}/{s}' for (t, s), v in seed_split.items() if len(v) > 1]
    dup = [k for k, n in Counter((x['root'], x['episode_id']) for x in all_rows).items() if n > 1]
    rep.add('all', 'split_leakage', not (wrong or test or both or dup), dict(
        ranges=coll['splits'], seed_outside_its_split=[x['episode_id'] for x in wrong][:20],
        test_seed_episodes=[x['episode_id'] for x in test][:20], task_seed_in_two_splits=both[:20], duplicate_ids=dup[:20],
        splits_present=dict(Counter(x['split'] for x in all_rows))))

    # balance
    bal = {}
    for x in all_rows:
        b = bal.setdefault(x['root'], {'task': Counter(), 'task_frames': Counter(), 'variant': Counter(), 'group': Counter()})
        b['task'][x['task']] += 1
        b['task_frames'][x['task']] += x['frames']
        b['variant'][f'{x["task"]}/{x["variant"]}'] += 1
        b['group'][x['group']] += 1
    bal = {k: {kk: dict(sorted(vv.items())) for kk, vv in v.items()} for k, v in bal.items()}
    grades = Counter(x['grade'] for x in all_rows)
    rep.add('all', 'balance', True, dict(by_root=bal, grades=dict(grades),
                                         note='task-balanced sampling (03/TTr-HL #3) equalises the task weights in training'))
    for dataset in ('D_observed', 'D_forecast'):
        training_tasks = set(bal.get(f'{dataset}/train', {}).get('task', {}))
        if training_tasks:
            validation_tasks = set(bal.get(f'{dataset}/val', {}).get('task', {}))
            missing = sorted(training_tasks - validation_tasks)
            rep.add(dataset, 'validation_task_coverage', not missing, dict(missing_tasks=missing))

    # policy inputs
    models = {m: model_inputs(Path(f)) for m, f in MODELS.items()}
    bad = {m: [k for k in v['inputs'] if k in FORBIDDEN or k not in feats] for m, v in models.items()}
    rep.add('all', 'policy_inputs', not any(bad.values()), dict(
        models=models, not_consumed={k: v for k, v in FORBIDDEN.items() if k in feats},
        frame_features=feats, violations=bad,
        note='episode metadata (task_name, group, fault, events, FSM phases) lives in meta/hrc_episodes.jsonl / '
             'sidecar only; the LeRobot `task` string is never read by ACT; scene distinguishability of the tasks '
             'needs the user review of the camera clips (not checkable here)'))

    a.out.mkdir(parents=True, exist_ok=True)
    ok = all(c['ok'] for c in rep.checks)
    out = dict(root=str(a.root), ok=ok, roots=per_root, checks=rep.checks)
    (a.out / 'check_train_data.json').write_text(json.dumps(out, indent=1, default=str))
    lines = [f'# Kiểm dữ liệu train bước 2 — `{a.root}`', '', f'Kết quả chung: **{"ĐẠT" if ok else "KHÔNG ĐẠT"}**. '
             f'{len(roots)} LeRobot root, {sum(v["episodes"] for v in per_root.values())} episode.', '',
             '| Phạm vi | Mục | Đạt | Chi tiết |', '|---|---|---|---|']
    for c in rep.checks:
        det = json.dumps(c['detail'], default=str, ensure_ascii=False)
        lines.append(f'| {Path(c["scope"]).name if c["scope"] != "all" else "all"} | {c["check"]} | '
                     f'{"có" if c["ok"] else "**KHÔNG**"} | {det[:300]}{"…" if len(det) > 300 else ""} |')
    lines += ['', '## Input của từng model', '']
    for m, v in models.items():
        lines.append(f'- **{m}** (`{v["dataset"]}`): ' + ', '.join(f'`{k}`' for k in v['inputs']))
    lines += ['', 'Không model nào đọc: ' + ', '.join(f'`{k}` ({v})' for k, v in FORBIDDEN.items() if k in feats) + '.']
    (a.out / 'check_train_data.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(ok=ok, failed=[(c['scope'], c['check']) for c in rep.checks if not c['ok']], roots=per_root), indent=1))


if __name__ == '__main__':
    main()

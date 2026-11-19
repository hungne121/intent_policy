"""Closed-loop policy runner (12/QĐ-HL, 08/QĐ-KT; agent G, 18/11/2026).

Written in this repo; the episode loop borrows LeRobot's rollout idea (lerobot/scripts/lerobot_eval.py `rollout`,
commit 64b2317) without importing it: per episode policy.reset() -> env.reset(seed) -> per tick obs -> batch ->
preprocessor -> select_action -> postprocessor -> env.step until terminated / truncated. The lerobot package is used
only as the model library (HRIACTPolicy = LeRobot ACT subclass, from_pretrained, make_pre_post_processors).

The same episodic structure (start = env.reset / operator start, policy.reset(), end on done or time limit) is meant
to drive the real robot later through an env with the same reset / step API (no real-robot code here).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------- policy loading
def resolve_checkpoint(path) -> Path:
    """A `pretrained_model` folder, or a checkpoint folder (tools/train_policy.py layout) that contains one."""
    p = Path(path)
    if (p / 'config.json').exists() and (p / 'model.safetensors').exists():
        return p
    if (p / 'pretrained_model' / 'config.json').exists():
        return p / 'pretrained_model'
    raise FileNotFoundError(f'{p}: no pretrained_model (config.json + model.safetensors)')


def load_policy(path, n_action_steps: int | None = None, device: str | None = None):
    """(policy, preprocessor, postprocessor, meta). `n_action_steps` overrides the execution horizon (03/TTr-HL #2:
    1 and 10) without retraining: it is an inference-only field of ACT (queue length of select_action)."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies import make_pre_post_processors
    from intent_policy.policies.hri_act import HRIACTPolicy          # registers 'hri_act'
    path = resolve_checkpoint(path)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = PreTrainedConfig.from_pretrained(path)
    cfg.device, cfg.pretrained_path = device, str(path)
    if n_action_steps is not None:
        if not 1 <= int(n_action_steps) <= cfg.chunk_size:
            raise ValueError(f'n_action_steps {n_action_steps} outside 1..{cfg.chunk_size}')
        cfg.n_action_steps = int(n_action_steps)
    if cfg.temporal_ensemble_coeff is not None:
        raise ValueError('03/TTr-HL #2: no temporal ensembling')
    policy = HRIACTPolicy.from_pretrained(path, config=cfg).to(device).eval()
    if policy.forecast_encoder is not None and not bool(policy.forecast_encoder.stats_set):
        raise RuntimeError(f'{path}: forecast normalisation stats not set in the checkpoint')
    pre, post = make_pre_post_processors(cfg, pretrained_path=str(path),
                                         preprocessor_overrides={'device_processor': {'device': device}})
    meta = json.loads((path / 'run_meta.json').read_text()) if (path / 'run_meta.json').exists() else {}
    return policy, pre, post, dict(path=str(path), model=meta.get('model'), step=meta.get('step'),
                                   best_val_l1=meta.get('best_val_l1'), dataset=meta.get('dataset'),
                                   n_action_steps=cfg.n_action_steps, chunk_size=cfg.chunk_size, device=device,
                                   use_forecast=bool(getattr(cfg, 'use_forecast', False)))


# ---------------------------------------------------------------------- observation -> policy batch
def to_policy_batch(obs: dict, task: str) -> dict:
    """Env observation -> LeRobot policy keys with a batch dim (the mapping of LeRobot's
    envs/utils.preprocess_observation): pixels/<cam> uint8 HWC -> observation.images.<cam> float CHW in [0, 1];
    agent_pos -> observation.state; any other key -> observation.<key>."""
    batch = {}
    for cam, img in obs['pixels'].items():
        t = torch.from_numpy(np.ascontiguousarray(img))
        assert t.dtype == torch.uint8 and t.ndim == 3 and t.shape[-1] == 3, t.shape
        batch[f'observation.images.{cam}'] = t.permute(2, 0, 1).unsqueeze(0).float().div_(255)
    batch['observation.state'] = torch.from_numpy(np.asarray(obs['agent_pos'], np.float32)).unsqueeze(0)
    for k, v in obs.items():
        if k not in ('pixels', 'agent_pos'):
            batch[f'observation.{k}'] = torch.from_numpy(np.asarray(v, np.float32)).unsqueeze(0)
    batch['task'] = [task]
    return batch


# ---------------------------------------------------------------------- episode loop
@torch.no_grad()
def run_episode(env, policy, preprocessor, postprocessor, seed: int, task: str, variant: str = 'nominal',
                video_path: Path | None = None, fault_type: str | None = None, forecast_condition: str = 'true',
                trace_path: Path | None = None) -> dict:
    """`forecast_condition` (20/QĐ-HL 2-3): 'true' = the recorded contract; 'masked' = the forecast branch input is what the
    recorder writes for an invalid frame (observation.human_forecast = 0, observation.forecast_valid = 0), applied to the
    preprocessed batch so nothing else changes. With `trace_path` (and a forecast policy) every inference tick also runs
    the same observation through both conditions (shadow forwards, z = 0, no effect on the rollout) and an .npz holds
    raw action chunks (true / masked / executed), executed actions and the executed TCP / human wrist / palm."""
    if forecast_condition not in ('true', 'masked'):
        raise ValueError(forecast_condition)
    policy.reset()
    fe = getattr(policy, 'forecast_encoder', None)
    cap = {'chunk': None, 'on': True}
    hook = policy.model.register_forward_hook(lambda m, i, o: cap.__setitem__('chunk', o[0].detach()) if cap['on'] else None)
    fire = []                                   # (tick, raw chunk true, raw chunk masked, raw chunk executed)
    exec_actions = []
    obs, info = env.reset(seed=seed, options={'task': task, 'variant': variant, **({} if fault_type is None else {'fault_type': fault_type})})
    writer = None
    if video_path is not None:
        from intent_policy.benchmark.video import VideoWriter
        frame = env.render()
        Path(video_path).parent.mkdir(parents=True, exist_ok=True)
        writer = VideoWriter(Path(video_path), width=frame.shape[1], height=frame.shape[0])
        writer.add(frame)
    t0, infer_s, steps = time.time(), 0., 0
    try:
        for _ in range(env.max_episode_steps + 1):
            batch = preprocessor(to_policy_batch(obs, task))
            batch_true = batch
            if fe is not None and (forecast_condition == 'masked' or trace_path is not None):
                batch_mask = dict(batch)
                batch_mask[fe.key] = torch.zeros_like(batch[fe.key])
                batch_mask[fe.valid_key] = torch.zeros_like(batch[fe.valid_key])
                if forecast_condition == 'masked':
                    batch = batch_mask
            row = env.fault_trace[-1] if getattr(env, 'fault_trace', None) else None
            if row is not None and row['strength'] > 0 and getattr(policy, 'forecast_encoder', None) is not None:
                # 15/QĐ-HL 6: the faulted forecast goes through the policy's own forecast preprocessing (forecast_token.py)
                enc = policy.forecast_encoder
                xe, ve = enc.features(batch)
                rb = dict(batch); rb[enc.key] = torch.from_numpy(env.raw_forecast_flat).to(batch[enc.key].device)[None]
                xr, _ = enc.features(rb)
                row.update(enc_valid=float(ve[0]), enc_finite=bool(torch.isfinite(xe).all()),
                           enc_feature_delta_max=float((xe - xr).abs().max()))
            ti = time.time()
            cap['chunk'] = None
            with torch.inference_mode():
                action = policy.select_action(batch)
                if cap['chunk'] is not None and trace_path is not None and fe is not None:     # a new chunk was predicted this tick
                    cap['on'] = False
                    chunks = []
                    for b in (batch_true, batch_mask):
                        bb = dict(b)
                        bb['observation.images'] = [b[c] for c in policy.config.image_features]
                        chunks.append(postprocessor(policy.model(bb)[0]).float().cpu().numpy()[0])
                    cap['on'] = True
                    fire.append((steps, chunks[0], chunks[1], postprocessor(cap['chunk']).float().cpu().numpy()[0]))
            action = postprocessor(action)
            infer_s += time.time() - ti
            a = action.to('cpu').numpy()
            assert a.ndim == 2 and a.shape[0] == 1, a.shape
            if row is not None:
                row['robot_action'] = np.round(a[0], 4).tolist()
            exec_actions.append(a[0].copy())
            obs, _, terminated, truncated, info = env.step(a[0])
            steps += 1
            if writer is not None:
                writer.add(env.render())
            if terminated or truncated:
                break
        else:
            raise RuntimeError('env neither terminated nor truncated within its time limit')
    finally:
        hook.remove()
        if writer is not None:
            writer.close()
    rec = dict(info['episode'])
    rec['forecast_condition'] = forecast_condition
    if trace_path is not None and fe is None:                       # policy without a forecast branch: executed motion only
        tr = env.ep.trace
        Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(trace_path, action_exec=np.stack(exec_actions), tcp=np.array([r['tcp'] for r in tr]),
                            palm=np.array([r['palm'] for r in tr]), wrist=np.array([r['wrist'] for r in tr]),
                            robot_person_m=np.array([np.nan if r['robot_person_m'] is None else r['robot_person_m'] for r in tr]),
                            t=np.array([r['t'] for r in tr]), n_action_steps=policy.config.n_action_steps)
    if trace_path is not None and fe is not None and fire:
        tr = env.ep.trace
        ct, cm = np.stack([f[1] for f in fire]), np.stack([f[2] for f in fire])
        with torch.no_grad():
            tcp = lambda c: fe.fk(torch.as_tensor(c[..., :6].reshape(-1, 6), dtype=torch.float64).to(fe.pos_mean.device)).cpu().numpy().reshape(c.shape[0], c.shape[1], 3) * 1000.0
            d = np.linalg.norm(tcp(ct) - tcp(cm), axis=-1)               # (fires, 40) mm between the two conditions' chunks
        rec['intervention'] = dict(n_chunks=len(fire), chunk_tcp_mm_median=float(np.median(d.mean(1))), chunk_tcp_mm_max=float(d.max()),
                                   chunk_joint_abs_mean=float(np.abs(ct - cm)[..., :6].mean()),
                                   first10_tcp_mm_median=float(np.median(d[:, :10].mean(1))))
        Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(trace_path, fire_tick=np.array([f[0] for f in fire]), chunk_true=ct, chunk_masked=cm,
                            chunk_executed=np.stack([f[3] for f in fire]), action_exec=np.stack(exec_actions),
                            tcp=np.array([r['tcp'] for r in tr]), palm=np.array([r['palm'] for r in tr]),
                            wrist=np.array([r['wrist'] for r in tr]), robot_person_m=np.array([np.nan if r['robot_person_m'] is None else r['robot_person_m'] for r in tr]),
                            t=np.array([r['t'] for r in tr]), n_action_steps=policy.config.n_action_steps)
    if fault_type is not None:
        rec['fault_type'] = fault_type
    rec.update(eval_ep_s=round(time.time() - t0, 1), policy_ms_per_tick=round(1000 * infer_s / max(steps, 1), 2),
               video=None if video_path is None else str(video_path))
    return rec


def summarize(records: list[dict]) -> dict:
    """LeRobot eval_info-like aggregate (pc_success, eval_ep_s) + CT over successes + safety counts."""
    def agg(rs):
        ok = [r for r in rs if r['success']]
        ct = [r['ct_s'] for r in ok if r.get('ct_s') is not None]
        on = [r['robot_onset_s'] for r in rs if r.get('robot_onset_s') is not None]
        dmin = [r['min_robot_person_m'] for r in rs if r.get('min_robot_person_m') is not None]
        return dict(n_episodes=len(rs), n_success=len(ok), pc_success=100. * len(ok) / max(len(rs), 1),
                    avg_ct_s=float(np.mean(ct)) if ct else None, n_ct=len(ct),
                    grade_counts={g: sum(r['grade'] == g for r in rs) for g in 'ABC'},
                    episodes_with_contact_violation=sum(bool(r['violating_contact_classes']) for r in rs),
                    episodes_with_observable_violation=sum(bool(r.get('observable_violations')) for r in rs),
                    physics_invalid=sum(not r['physics_valid'] for r in rs),
                    failure_reasons={k: sum(r['failure_reason'] == k for r in rs)
                                     for k in sorted({str(r['failure_reason']) for r in rs if not r['success']})},
                    min_robot_person_m=float(min(dmin)) if dmin else None,
                    avg_robot_onset_s=float(np.mean(on)) if on else None,
                    eval_ep_s=float(np.mean([r['eval_ep_s'] for r in rs])) if rs else None)
    groups = {}
    for r in records:
        groups.setdefault(f"{r['task']}_{r['variant']}", []).append(r)
    return dict(per_task={k: agg(v) for k, v in sorted(groups.items())}, aggregated=agg(records))


def evaluate(policy, preprocessor, postprocessor, env, tasks, variants, seeds, out_dir: Path, tags: dict,
             n_video: int = 0, log=print, fault_type: str | None = None, forecast_condition: str = 'true',
             trace_dir: Path | None = None) -> dict:
    """Run tasks x variants x seeds; write episodes.jsonl (appended per episode) and eval_info.json."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    records, t0 = [], time.time()
    for task in tasks:
        for variant in variants:
            for i, seed in enumerate(seeds):
                video = out_dir / 'videos' / f'{task}_{variant}_{seed}.mp4' if i < n_video else None
                tp = None if trace_dir is None else Path(trace_dir) / f'{task}_{variant}_{seed}_{forecast_condition}.npz'
                rec = dict(tags, **run_episode(env, policy, preprocessor, postprocessor, seed, task, variant, video, fault_type,
                                               forecast_condition, tp))
                trace = rec.pop('fault_trace', None)          # 14/QĐ-HL P3 24: per-tick forecast before / after injection
                if trace is not None:
                    (out_dir / 'fault_traces').mkdir(exist_ok=True)
                    (out_dir / 'fault_traces' / f'{task}_{variant}_{fault_type}_{seed}.json').write_text(
                        json.dumps(dict(case=rec['fault'], trace=trace)))
                records.append(rec)
                with (out_dir / 'episodes.jsonl').open('a') as f:
                    f.write(json.dumps(rec) + '\n')
                log(f"{tags.get('model')} step {tags.get('step')} n={tags.get('n_action_steps')} {task}/{variant} "
                    f"seed {seed}: success={rec['success']} ({rec['failure_reason']}) dur={rec['duration_s']}s "
                    f"ct={rec['ct_s']} grade={rec['grade']} dmin={rec['min_robot_person_m']} "
                    f"onset={rec['robot_onset_s']} wall={rec['eval_ep_s']}s policy={rec['policy_ms_per_tick']}ms/tick")
    info = dict(tags, per_episode=records, **summarize(records))
    info['aggregated'].update(eval_s=round(time.time() - t0, 1))
    (out_dir / 'eval_info.json').write_text(json.dumps(info, indent=1))
    return info

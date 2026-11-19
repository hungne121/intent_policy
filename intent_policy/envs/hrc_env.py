"""HRC station as a gymnasium Env for LeRobot's own evaluation path (12/QĐ-HL, agent G, 18/11/2026).

Driven by our runner intent_policy/eval/runner.py (08/QĐ-KT: written in this repo, ideas borrowed from LeRobot's
rollout; lerobot is used only as the model library): reset(seed, options={task, variant}) -> obs, then per 20 Hz tick
obs -> to_policy_batch -> policy preprocessor -> select_action -> postprocessor -> step(action).

One episode = the collection rollout of tools/preview.run_episode with the expert's commands replaced by the policy's:
the same scenario build, `Episode` (mode 'forecast' so ManiCast runs every tick, as in collection both modes compute
and record it), the same human generator (+ N1 variation as tools/collect.py samples it), the same `Episode.step`
(env joint-limit / joint-speed clipping, physics validator) and the same scoring: `expert_class(task)(ep).success()`
(+ `sc.check`) on the final world state, grade by `episode_quality.classify` (66/QĐ-DL). The expert is instantiated
for scoring only; its run() is never called.

Observation = exactly the recorded features the policies were trained on (data/lerobot_recorder.add_frame):
  pixels/<cam>    -> observation.images.<cam>   (ep.env.render(cam, 256, 256), uint8 HWC; LeRobot -> float CHW / 255)
  agent_pos       -> observation.state           (qpos 6 + qvel 6 + gripper opening 1)
  human_forecast  -> observation.human_forecast  (10 x 6 x 3 ManiCast forecast, zeros when not valid)
  forecast_valid  -> observation.forecast_valid
(the runner maps pixels/<cam> -> observation.images.<cam>, agent_pos -> observation.state, other keys -> observation.<key>,
the same mapping as LeRobot's envs/utils.preprocess_observation.)

Evaluator-side additions a policy cannot provide (reported, not part of the policy; values in HRCEnvConfig):
  * phase label from the commanded gripper (close / open / policy) and, while closing, `robot_selection` = nearest
    graspable body: the physics validator reads `ep.phase` (grasp-drop rule) and `robot_selection` (intended finger
    contact) exactly as the expert's `close_on` / `open_gripper` set them;
  * `robot_grasp` event when the env attaches an object (the expert's close_on emits it; I3 check reads it);
  * NO human hand-shake substitutes (09/QĐ-KT P0): the scripted human reacts only to intent_policy/shared/events.py
    (current robot / object state, parameters configs/hrc/events.yaml), the same detector as in collection. The
    moments the person reacted are the `human_cue` events of the episode (reported as `human_cues`);
  * ground truth for SCORING ONLY (never read by the human): expert FSM-only fields read by success(): I1 ep.phase
    'done', I2 selected = the plate the person chose (the scenario's evaluator-only plan, as I2 `check` uses), I3
    phase 'done'. The world-state predicates are unchanged.
Termination: the scenario's human is done and the task predicate has held for `success_hold_ticks` (Episode.run lets
the scene settle 12 ticks after the robot ends); truncation at the scenario's MAX_TICKS (= collection time limit).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml
from intent_policy.shared.sim_env import ROOT, DT

CFG = yaml.safe_load((ROOT / 'configs/hrc/collection.yaml').read_text())
STATE_DIM, FORECAST_DIM = 13, 10 * 6 * 3
POLICY_TRIGGERS = ('hand_speed_above_still_bound',)       # n5.policy_trigger (15/QĐ-HL rules 1-3, amends 14/QĐ-HL 22)
_PREDICTOR = None


def predictor():
    """ManiCast loaded once per process (as tools/preview / tools/collect do)."""
    global _PREDICTOR
    if _PREDICTOR is None:
        from intent_policy.shared.manicast_adapter import load_predictor
        _PREDICTOR = load_predictor()
    return _PREDICTOR


def policy_observation(ep, cameras, image_hw):
    """The policy inputs of data/lerobot_recorder.add_frame, same sources and dtypes (tests/test_eval_policy.py)."""
    h, w = image_hw
    q, qd = ep.d.qpos[ep.env.qids], ep.d.qvel[ep.env.dids]
    grip = float(ep.d.qpos[ep.env.gid])
    mc = ep.manicast
    fc = mc.forecast if mc.valid else np.zeros((10, 6, 3))
    return {'pixels': {c: np.asarray(ep.env.render(c, w, h), np.uint8) for c in cameras},
            'agent_pos': np.r_[q, qd, grip].astype(np.float32),
            'human_forecast': np.asarray(fc, np.float32).reshape(-1),
            'forecast_valid': np.array([float(mc.valid)], np.float32)}


@dataclass
class HRCEnvConfig:
    """Evaluator settings of HRCEnv (our config; borrowed idea: LeRobot EnvConfig.gym_kwargs)."""
    group: str = 'N1'                          # collection group: N1 variation on, N6 / PREVIEW without
    allowed_splits: list = field(default_factory=lambda: ['val'])   # 12/QĐ-HL: never test seeds now
    success_hold_ticks: int = 12               # Episode.run: 12 settle ticks after the robot ends
    onset_speed_m_s: float = 0.01              # CartesianFollower arrival tol 0.5 mm per 0.05 s tick
    selection_radius_m: float = 0.06           # design value: graspable body nearest the TCP while closing
    fault_type: str | None = None              # N5 only (14/QĐ-HL P3): spatial_fault / temporal_fault / direction_fault; reset options override


class HRCEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': CFG['fps']}

    def __init__(self, task='I1', variant='nominal', config: HRCEnvConfig | None = None):
        c = config or HRCEnvConfig()
        self.cfg = c
        self.task, self.variant, self.group = task, variant, c.group
        self.cameras, self.image_hw = list(CFG['cameras']), tuple(CFG['image_hw'])
        self.allowed_splits = list(c.allowed_splits)
        self.hold_ticks, self.onset_speed = int(c.success_hold_ticks), float(c.onset_speed_m_s)
        self.tol = dict(select=c.selection_radius_m)
        h, w = self.image_hw
        self.observation_space = gym.spaces.Dict({
            'pixels': gym.spaces.Dict({c: gym.spaces.Box(0, 255, (h, w, 3), np.uint8) for c in self.cameras}),
            'agent_pos': gym.spaces.Box(-np.inf, np.inf, (STATE_DIM,), np.float32),
            'human_forecast': gym.spaces.Box(-np.inf, np.inf, (FORECAST_DIM,), np.float32),
            'forecast_valid': gym.spaces.Box(0, 1, (1,), np.float32)})
        self.action_space = gym.spaces.Box(-np.inf, np.inf, (7,), np.float32)
        self.fault_type = c.fault_type
        self.ep = None
        self._obs = None
        self.injector, self.fault_trace, self.fault_case, self.hand_motion_bound = None, [], None, None

    @property
    def max_episode_steps(self):
        return int(self.sc.MAX_TICKS)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.task, self.variant = options.get('task', self.task), options.get('variant', self.variant)
        self.fault_type = options.get('fault_type', self.cfg.fault_type)
        if seed is None:
            raise ValueError('HRCEnv needs an explicit seed (the scenario is drawn from it)')
        split = next((k for k, (lo, hi) in CFG['splits'].items() if lo <= int(seed) <= hi), None)
        if split not in self.allowed_splits:
            raise ValueError(f'seed {seed} is in split {split!r}; allowed {self.allowed_splits} (12/QĐ-HL)')
        self._close_episode()
        self._build(int(seed))
        return self._obs, {'is_success': False}

    def step(self, action):
        ep = self.ep
        action = np.asarray(action, float).reshape(-1)
        self._label_phase(action)
        self._count_clipping(action)
        ep.step(action)                                   # same path as the expert's commands
        self._after_step()
        now = self._success_now()
        if now:
            self.streak_start = ep.t if self.streak_start is None else self.streak_start
            self.streak += 1
        else:
            self.streak, self.streak_start = 0, None
        terminated = truncated = False
        if self.sc.human_done(ep) and self.streak >= self.hold_ticks:
            terminated = True
        elif ep.tick >= ep.max_ticks:
            truncated = True
        elif not self._advance_human():
            terminated = True                             # human_reach_failed (Episode.run breaks the same way)
        self._obs = policy_observation(ep, self.cameras, self.image_hw)
        info = {'is_success': False}
        if terminated or truncated:
            rec = self._finish(truncated)
            info.update(is_success=bool(rec['success']), episode=rec)
        return self._obs, 0.0, terminated, truncated, info

    def render(self):
        return None if self._obs is None else self._obs['pixels'][CFG['review_camera']]

    def close(self):
        self._close_episode()

    # ------------------------------------------------------------------ episode
    def _build(self, seed):
        from intent_policy.shared.episode import Episode
        from intent_policy.experts import expert_class
        from tools.collect import task_cfg
        from intent_policy.shared.variation import sample_variation
        from intent_policy.scenarios import scenario
        self.sc = sc = scenario(self.task)
        scene, positions, xml, info = sc.build(seed, self.variant) if hasattr(sc, 'VARIANTS') else sc.build(seed)
        use_variation = CFG['groups'][self.group].get('variation', True)
        variation = sample_variation(seed, task_cfg('n1', self.task)) if use_variation else None
        # 14/QĐ-HL P3 (21-22): N5 gets the injector of the collection (same FaultSpec / sample_fault for the same
        # task, fault_type, seed); N1 / N6 / PREVIEW keep injector=None. Fault type comes from the env config / CLI.
        self.injector, self.fault_trace, self.fault_case, self.hand_motion_bound = None, [], None, None
        if self.group == 'N5':
            from intent_policy.shared.forecast_fault import FaultInjector, sample_fault
            if self.fault_type not in CFG['groups']['N5']['fault_types']:
                raise ValueError(f"group N5 needs fault_type in {CFG['groups']['N5']['fault_types']}, got {self.fault_type!r}")
            n5 = task_cfg('n5', self.task)
            if n5.get('policy_trigger') not in POLICY_TRIGGERS:
                raise ValueError(f"configs/hrc/collection.yaml n5.policy_trigger must be one of {POLICY_TRIGGERS}")
            ts = n5['policy_trigger_speed']
            self.hand_motion_bound = float(yaml.safe_load((ROOT / ts['file']).read_text())[ts['section']][ts['key']])
            n5 = dict(n5, generator_version=n5['policy_generator_version'])   # 16/QĐ-HL: policy eval = v2
            self.injector = FaultInjector(sample_fault(self.fault_type, seed, n5), n5)
            self.fault_case = dict(task=self.task, variant=self.variant, fault_type=self.fault_type, seed=int(seed),
                                   group='N5', trigger=n5['policy_trigger'], generator_version=n5['generator_version'], hand_motion_bound_m_s=self.hand_motion_bound)
        elif self.fault_type is not None:
            raise ValueError(f'fault_type {self.fault_type!r} only with group N5 (got {self.group})')
        self.wall0 = time.time()
        ep = Episode(self.task, 'forecast', seed, scene, positions, xml, predictor(), out=None,
                     max_ticks=sc.MAX_TICKS, review_camera=CFG['review_camera'], review_hw=CFG['review_hw'],
                     recorder=None, variant=info.get('variant', 'nominal'), injector=self.injector, variation=variation)
        self.ep, self.info = ep, info
        self.expert = expert_class(self.task)(ep)         # scoring only, run() never called
        ep.event('task_active', **{k + '_evaluator_only': v for k, v in info.items() if k != 'variant'})
        ep.phase = 'policy'
        from intent_policy.shared.events import detector
        detector(ep).ep._metric_tracker.info = dict(info)      # scoring-only tracker starts before the first tick
        # 09/QĐ-KT: no hand-shake flags; the human reads intent_policy/shared/events.py (as in collection)
        human = sc.human(ep, info)
        if variation:
            from intent_policy.shared.variation import timed_human
            ep.variation_log['human'] = {}
            human = timed_human(human, variation, ep.variation_log['human'])
        self.human, self.last_frame = human, None
        self.streak, self.streak_start, self.finished = 0, None, False
        self.clip = {'joint_limit_ticks': 0, 'joint_speed_ticks': 0}
        self._attached_prev = None
        if not self._advance_human():
            raise RuntimeError(f'{self.task} seed {seed}: human failed before the first tick')
        self._obs = policy_observation(ep, self.cameras, self.image_hw)

    def _advance_human(self):
        """One human frame, as at the top of Episode.run's loop. False on a human reach failure."""
        from intent_policy.shared.episode import HumanReachError
        ep = self.ep
        try:
            self.last_frame = next(self.human)
        except HumanReachError as exc:
            ep.human_failed = True; ep.event('human_reach_timeout', detail=str(exc))
            return False
        except StopIteration:
            pass
        if self.last_frame is None:
            raise RuntimeError('human script yielded nothing')
        ep.apply_human(self.last_frame)
        self._mark_forecast_use()
        return True

    def _mark_forecast_use(self):
        """N5 trigger for a policy (15/QĐ-HL 1-3): onset of MEANINGFUL human motion, causal. Tick n is a motion tick when
        the OBSERVED hand speed (episode.hand_speed, from the observed history only) exceeds the measured still-hand
        bound events.yaml metric.hand_still_max_m_s (n5.policy_trigger_speed). The injector counts an onset after
        n5.rearm_s of non-motion ticks (a still period) and starts the fault n5.delay_s later, reading the flag at tick
        n+1 (episode.py:287 `used_last_tick`). No expert-read flag, no forecast-valid condition, no robot motion, no
        future / human plan. The flag is written into ep.forecast_use_ticks only as the carrier the injector reads."""
        ep = self.ep
        if self.injector is None:
            return
        mc = ep.manicast
        used = bool(ep.hand_speed > self.hand_motion_bound)
        if used:
            ep.forecast_use_ticks.add(ep.tick)
        viol = {v['t']: v['checks'] for v in self.injector.log['violations']}
        chk = viol.get(round(ep.t, 3)) if mc.fault_strength > 0 else None
        w = ep.wrist_index()
        self.raw_forecast_flat = None if mc.raw_forecast is None or not mc.valid else np.asarray(mc.raw_forecast, np.float32).reshape(-1)
        raw = None if mc.raw_forecast is None or not mc.valid else mc.raw_forecast[-1, w]
        eff = None if not mc.valid else mc.forecast[-1, w]
        self.fault_trace.append(dict(
            tick=ep.tick, t=round(ep.t, 3), forecast_valid=bool(mc.valid), used=used, human_phase=ep.human_phase,
            checks_pass=None if mc.fault_strength <= 0 else chk is None, checks_failed=chk, hand_speed=round(float(ep.hand_speed), 4),
            strength=round(float(mc.fault_strength), 4),
            raw_wrist_1s=None if raw is None else np.round(raw, 4).tolist(),
            eff_wrist_1s=None if eff is None else np.round(eff, 4).tolist(),
            wrist_dev_1s_m=None if raw is None or eff is None else round(float(np.linalg.norm(eff - raw)), 4),
            max_joint_dev_m=None if raw is None or not mc.valid else round(float(np.abs(mc.forecast - mc.raw_forecast).max()), 4)))

    def _close_episode(self):
        if self.ep is not None and not self.finished:
            try:
                self.ep.env.close()
            except Exception:
                pass
        self.ep = None

    # ------------------------------------------------------------------ evaluator-side labels / hand-shakes
    def _label_phase(self, action):
        ep = self.ep
        opening = float(ep.d.qpos[ep.env.gid])
        if action[6] > opening + 1e-3:
            ep.phase, ep.robot_selection = 'open', ep.robot_selection
        elif action[6] < opening - 1e-3:
            ep.phase = 'close'
            if ep.attached_name() is None:
                tcp = ep.tcp()
                d = {b: float(np.linalg.norm(ep.d.xpos[b] - tcp)) for b in ep.env.grasp_welds}
                b = min(d, key=d.get) if d else None
                ep.robot_selection = ep.m.body(b).name if b is not None and d[b] < self.tol['select'] else None
        else:
            ep.phase, ep.robot_selection = 'policy', None

    def _count_clipping(self, action):
        env = self.ep.env
        lim = env.arm_limits
        arm = np.clip(action[:6], lim[:, 0], lim[:, 1])
        if np.any(np.abs(arm - action[:6]) > 1e-9):
            self.clip['joint_limit_ticks'] += 1
        vmax = np.asarray(env.cfg['max_joint_speed'], float) * env.dt
        if np.any(np.abs(arm - self.ep.d.qpos[env.qids]) > vmax + 1e-12):
            self.clip['joint_speed_ticks'] += 1

    def _after_step(self):
        ep = self.ep
        att = ep.attached_name()
        if att is not None and att != self._attached_prev:
            ep.event('robot_grasp', object=att)
        self._attached_prev = att

    # ------------------------------------------------------------------ scoring
    def _expert_success(self):
        """expert.success() with its FSM-only fields filled by the evaluator (module docstring)."""
        ep, ex = self.ep, self.expert
        saved = ep.phase
        if self.task == 'I1':
            ep.phase = 'done'
        elif self.task == 'I2':
            ex.selected = self.info['plate']
        elif self.task == 'I3':
            ex.phase = 'done'
        try:
            ok, reason = ex.success()
        finally:
            ep.phase = saved
        if ok and hasattr(self.sc, 'check'):
            ok, reason = self.sc.check(ep, self.info)
        return bool(ok), reason

    def _success_now(self):
        return self._expert_success()[0]

    def _finish(self, truncated):
        from intent_policy.data.episode_quality import classify, observable
        ep = self.ep
        ok, reason = self._expert_success()
        if ep.human_failed:
            ok, reason = False, 'human_reach_failed'
        elif not ok and truncated:
            reason = 'time_limit'
        extra = {'reading': self.expert.reading, 'human_plan_evaluator_only': dict(self.info)}
        s = ep.finish(ok, reason, extra)
        from intent_policy.shared.events import metric_record
        metrics = metric_record(ep, bool(s['success']), round(self.streak_start or ep.t, 3), s)   # 10/QĐ-KT, scoring only
        s.pop('_arrays', None)
        self.finished = True
        ticks = round(s['duration_s'] * CFG['fps'])
        grade, grade_reasons = classify(s, len(ep.trace), ticks, CFG.get('quality', {}).get('review_allowed_contacts', False))
        tcp = np.array([r['tcp'] for r in ep.trace])
        speed = np.r_[0., np.linalg.norm(np.diff(tcp, axis=0), axis=1) / DT] if len(tcp) else np.zeros(0)
        moving = np.flatnonzero(speed > self.onset_speed)
        near = [r['robot_person_m'] for r in ep.trace if r['robot_person_m'] is not None]
        valid = [r['forecast_valid'] for r in ep.trace]
        rec = dict(task=self.task, variant=self.variant, group=self.group, seed=ep.seed,
                   success=bool(s['success']), failure_reason=s['failure_reason'],
                   duration_s=s['duration_s'], ct_s=None if not s['success'] else round(self.streak_start or ep.t, 3),
                   time_limit_s=round(ep.max_ticks * DT, 2), truncated=bool(truncated), ticks=ep.tick,
                   grade=grade, grade_reasons=grade_reasons, physics_valid=s['physics_valid'],
                   violating_contact_classes=sorted(k for k, v in s.get('contact_classes', {}).items() if v.get('violation')),
                   observable_violations=sorted(k for k, v in s.get('contact_classes', {}).items()
                                                if v.get('violation') and observable(v)),
                   contact_violations=sorted(s.get('contact_violations') or [])[:20],
                   robot_limits_ok=s.get('robot_limits_ok'), grasp_drops=(s.get('robot_limits') or {}).get('grasp_drops'),
                   min_robot_person_m=None if not near else round(float(min(near)), 4),
                   robot_onset_s=None if not len(moving) else round(float(moving[0] * DT), 3),
                   onset_speed_m_s=self.onset_speed,
                   first_robot_grasp_s=next((e['t'] for e in ep.events if e.get('event') == 'robot_grasp'), None),
                   forecast_valid_ratio=round(float(np.mean(valid)), 4) if valid else None,
                   env_clipping=self.clip,
                   human_cues=[e for e in ep.events if e.get('event') == 'human_cue'], metrics=metrics,
                   events=[e for e in ep.events if e.get('event') in ('robot_grasp', 'human_grasp', 'human_reach_timeout')],
                   wall_s=round(time.time() - self.wall0, 1))
        if self.injector is not None:                      # 14/QĐ-HL P3 24: case, schedule, checks; per-tick log -> runner
            rec['fault'] = dict(self.fault_case, **self.injector.summary())
            rec['fault_trace'] = self.fault_trace
        return json.loads(json.dumps(rec, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)))

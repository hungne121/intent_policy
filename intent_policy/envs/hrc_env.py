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
  * human hand-shake flags that the scripted human waits on and that only the expert sets: I1 `robot_prepared`,
    C2 `wiped` / 'present_cloth', C4 `delivered` (+ the human grasp of the handed part). Each is set from the
    observable world state at the moment the expert would set it;
  * expert FSM-only fields read by success(): I1 ep.phase 'done', I2 selected = the plate the person chose (the
    scenario's evaluator-only plan, as I2 `check` uses), I3 phase 'done'. The world-state predicates are unchanged.
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
from intent_policy.shared.sim_env import ROOT, DT, TRAVEL_Z, GRASP_DZ

CFG = yaml.safe_load((ROOT / 'configs/hrc/collection.yaml').read_text())
STATE_DIM, FORECAST_DIM = 13, 10 * 6 * 3
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
    i1_prepared_tol_m: float = 0.02            # design value: TCP above a remaining cup at travel height
    c4_handover_tol_m: float = 0.02            # design value: TCP at palm + part-on-palm + grasp offset
    c2_wipe_radius_m: float = 0.035            # c2_expert: wipe marks have a 35 mm effect radius
    c2_present_speed_m_s: float = 0.005        # scenarios/c2 presented(): controller speed < 0.005
    selection_radius_m: float = 0.06           # design value: graspable body nearest the TCP while closing


class HRCEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': CFG['fps']}

    def __init__(self, task='I1', variant='nominal', config: HRCEnvConfig | None = None):
        c = config or HRCEnvConfig()
        self.cfg = c
        self.task, self.variant, self.group = task, variant, c.group
        self.cameras, self.image_hw = list(CFG['cameras']), tuple(CFG['image_hw'])
        self.allowed_splits = list(c.allowed_splits)
        self.hold_ticks, self.onset_speed = int(c.success_hold_ticks), float(c.onset_speed_m_s)
        self.tol = dict(i1=c.i1_prepared_tol_m, c4=c.c4_handover_tol_m, c2_wipe=c.c2_wipe_radius_m,
                        c2_speed=c.c2_present_speed_m_s, select=c.selection_radius_m)
        h, w = self.image_hw
        self.observation_space = gym.spaces.Dict({
            'pixels': gym.spaces.Dict({c: gym.spaces.Box(0, 255, (h, w, 3), np.uint8) for c in self.cameras}),
            'agent_pos': gym.spaces.Box(-np.inf, np.inf, (STATE_DIM,), np.float32),
            'human_forecast': gym.spaces.Box(-np.inf, np.inf, (FORECAST_DIM,), np.float32),
            'forecast_valid': gym.spaces.Box(0, 1, (1,), np.float32)})
        self.action_space = gym.spaces.Box(-np.inf, np.inf, (7,), np.float32)
        self.ep = None
        self._obs = None

    @property
    def max_episode_steps(self):
        return int(self.sc.MAX_TICKS)

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.task, self.variant = options.get('task', self.task), options.get('variant', self.variant)
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
        self.wall0 = time.time()
        ep = Episode(self.task, 'forecast', seed, scene, positions, xml, predictor(), out=None,
                     max_ticks=sc.MAX_TICKS, review_camera=CFG['review_camera'], review_hw=CFG['review_hw'],
                     recorder=None, variant=info.get('variant', 'nominal'), injector=None, variation=variation)
        self.ep, self.info = ep, info
        self.expert = expert_class(self.task)(ep)         # scoring only, run() never called
        ep.event('task_active', **{k + '_evaluator_only': v for k, v in info.items() if k != 'variant'})
        ep.phase = 'policy'
        # hand-shake state the human scripts read (set by the experts' run() in collection)
        ep.robot_prepared, ep.coaster_down = False, [False] * 3          # I1
        ep.wiped, ep.delivered, ep.claimed = {}, None, None              # C2, C4, W2 / W3
        human = sc.human(ep, info)
        if variation:
            from intent_policy.shared.variation import timed_human
            ep.variation_log['human'] = {}
            human = timed_human(human, variation, ep.variation_log['human'])
        self.human, self.last_frame = human, None
        self.streak, self.streak_start, self.finished = 0, None, False
        self.clip = {'joint_limit_ticks': 0, 'joint_speed_ticks': 0}
        self.handshakes = []
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
        return True

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

    def _tcp_speed(self):
        tr = self.ep.trace
        return 0. if len(tr) < 2 else float(np.linalg.norm(np.subtract(tr[-1]['tcp'], tr[-2]['tcp']))) / DT

    def _after_step(self):
        ep = self.ep
        att = ep.attached_name()
        if att is not None and att != self._attached_prev:
            ep.event('robot_grasp', object=att)
        self._attached_prev = att
        getattr(self, f'_handshake_{self.task}', lambda: None)()

    def _handshake_I1(self):
        from intent_policy.scenarios.i1 import CUPS
        ep = self.ep
        tcp = ep.tcp()
        free = [c for c in CUPS if np.linalg.norm(ep.pos(c)[:2] - ep.initial[c][:2]) < .02]
        prepared = ep.attached_name() is None and abs(tcp[2] - TRAVEL_Z) < self.tol['i1'] and any(
            np.linalg.norm(tcp[:2] - ep.pos(c)[:2]) < self.tol['i1'] for c in free)
        if prepared and not ep.robot_prepared:
            self.handshakes.append({'t': round(ep.t, 2), 'flag': 'robot_prepared'})
        ep.robot_prepared = bool(prepared)

    def _handshake_C4(self):
        from intent_policy.experts.c4_expert import PART_ON_PALM
        ep = self.ep
        for i in range(3):
            part = f'part{i}'
            if ep.human_phase != f'wait_part_{i}' or ep.attached_name() != part or ep.delivered == i:
                continue
            palm = np.asarray(ep.pose['palm'], float)
            if np.linalg.norm(ep.tcp() - (palm + [0, 0, PART_ON_PALM + GRASP_DZ])) < self.tol['c4']:
                ep.human_grasp(part); ep.delivered = i
                self.handshakes.append({'t': round(ep.t, 2), 'flag': f'delivered_{i}'})

    def _handshake_C2(self):
        from intent_policy.shared.episode import dirt_geoms
        ep = self.ep
        for obj in ('cup', 'clock'):
            if obj in ep.wiped or obj not in ep.held_by_human:
                continue
            xy = ep.initial[obj][:2]
            if not any(ep.m.geom_rgba[g, 3] > 0 and np.linalg.norm(ep.m.geom_pos[g, :2] - xy) < self.tol['c2_wipe']
                       for g in dirt_geoms(ep.m)):
                ep.wiped[obj] = True
                self.handshakes.append({'t': round(ep.t, 2), 'flag': f'wiped_{obj}'})
        if (ep.attached_name() == 'cloth' and ep.human_phase == 'wait_cloth'
                and self._tcp_speed() < self.tol['c2_speed']):
            ep.phase = 'present_cloth'

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
                   env_clipping=self.clip, handshakes=self.handshakes,
                   events=[e for e in ep.events if e.get('event') in ('robot_grasp', 'human_grasp', 'human_reach_timeout')],
                   wall_s=round(time.time() - self.wall0, 1))
        return json.loads(json.dumps(rec, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)))

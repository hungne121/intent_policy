"""N1 nominal variation shared by every task (59/QĐ-DL §E N1). Adds, on top of what each scenario already draws
from its seed (object / target choice, order, HY-Motion sample, onset, jitter), four generic axes:

  * human speed       - the scripted human stream is re-timed by a factor s (s > 1 faster: frames advanced
                        more than once per tick; s < 1 slower: a frame is repeated). Side effects of the scripted
                        human (grasp / release) run in their original order. Range: measured HY-Motion spread.
  * start shift       - later start: the first frame is held longer; earlier start: up to |shift| of the leading
                        STATIONARY frames are dropped (never a moving frame).
  * hesitation        - with probability p, one pause of d s while the hand is moving, after `after_s`.
  * robot start state - the robot starts from home TCP + an offset (top-down IK), a valid start pose.

No change of mind and no forecast fault here (N1). All draws use their own seed stream, so the scenario's own
random stream is unchanged and observed / forecast rollouts of a seed get the same variation.
"""
import numpy as np

from intent_policy.shared.sim_env import DT


def sample_variation(seed, cfg):
    """Variation spec for one seed; `cfg` = collection.yaml n1 block (with per-task overrides applied)."""
    rng = np.random.default_rng([int(seed), 59, 1])
    spec = {'speed_scale': float(rng.uniform(*cfg['speed_scale'])),
            'start_shift_s': float(rng.uniform(*cfg['start_shift_s'])),
            'pause': None, 'robot_tcp_offset_m': [float(rng.uniform(-a, a)) for a in cfg['robot_tcp_offset_m']]}
    if rng.uniform() < float(cfg['pause']['probability']):
        spec['pause'] = {'after_s': float(rng.uniform(*cfg['pause']['after_s'])),
                         'duration_s': float(rng.uniform(*cfg['pause']['duration_s']))}
    return spec


NOMINAL = {'speed_scale': 1., 'start_shift_s': 0., 'pause': None, 'robot_tcp_offset_m': [0., 0., 0.]}


def _palms(frame):
    return np.r_[np.asarray(frame['palm'], float),
                 np.asarray(frame['idle_palm'], float) if frame.get('idle_palm') is not None else np.zeros(3)]


def timed_human(gen, spec, log):
    """Re-time a scripted human frame generator. `log` (dict) receives what was applied."""
    s = float(spec.get('speed_scale', 1.))
    shift = int(round(float(spec.get('start_shift_s', 0.)) / DT))
    pause = spec.get('pause')
    log.update({'dropped_start_frames': 0, 'held_start_frames': 0, 'pause_at_s': None, 'frames_skipped': 0,
                'frames_repeated': 0})
    try:
        frame = next(gen)
    except StopIteration:
        return
    first = _palms(frame)
    if shift < 0:                                   # earlier start: drop leading stationary frames
        for _ in range(-shift):
            try:
                nxt = next(gen)
            except StopIteration:
                break
            if np.linalg.norm(_palms(nxt) - first) > 1e-4:
                frame = nxt; break                  # it moves: keep it, stop dropping
            frame = nxt; log['dropped_start_frames'] += 1
    for _ in range(max(0, shift)):                  # later start
        log['held_start_frames'] += 1
        yield frame
    acc, tick, paused = 0., 0, pause is None
    prev = _palms(frame)
    yield frame
    while True:
        tick += 1
        if not paused and tick * DT >= pause['after_s'] and np.linalg.norm(_palms(frame) - prev) > .03 * DT:
            paused = True; log['pause_at_s'] = round(tick * DT, 3)
            for _ in range(max(1, int(round(pause['duration_s'] / DT)))):
                yield frame
        prev = _palms(frame)
        acc += s
        n = int(acc); acc -= n
        if n == 0:
            log['frames_repeated'] += 1
            yield frame; continue
        try:
            for _ in range(n):
                frame = next(gen)
        except StopIteration:
            return                                  # the episode keeps the last applied frame
        log['frames_skipped'] += n - 1
        yield frame


def apply_robot_start(ep, offset):
    """Move the robot to home TCP + offset before the first tick (top-down DLS IK, gripper open)."""
    import mujoco
    offset = np.asarray(offset, float)
    if not np.any(offset):
        return None
    env, m, d = ep.env, ep.m, ep.d
    home_tcp = d.site_xpos[env.tcp].copy()
    q = np.asarray(env.solve_ik(home_tcp + offset), float)
    lo, hi = env.arm_limits[:, 0], env.arm_limits[:, 1]
    q = np.clip(q, lo, hi)
    d.qpos[env.qids] = q; d.qvel[env.dids] = 0.
    d.ctrl[:6] = q
    mujoco.mj_forward(m, d)
    return {'home_tcp': home_tcp.round(4).tolist(), 'start_tcp': d.site_xpos[env.tcp].round(4).tolist(),
            'ik_error_m': float(np.linalg.norm(d.site_xpos[env.tcp] - home_tcp - offset))}

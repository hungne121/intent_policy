"""Predictive Minimum-Interruption Avoidance, shared by the Cowork experts W1-W3 (57/QĐ-DL).

Per control tick (ManiCast updates every tick):
  1. Human occupancy over time: both arms as capsules (upper arm, forearm, hand, held objects) at t = 0 (observed)
     and t = 0.1 ... 1.0 s (the 10 ManiCast steps). Beyond 1 s the person is held at the 1.0 s pose. In the
     `observed` mode, and while the observed hand is still (54/QĐ-DL), the observed pose is held instead.
  2. Robot sweep: every collision link of the arm and gripper (and the object it holds) as capsules, at the
     configuration the arm will have at each check time along a candidate plan (IK of the TCP on the path).
  3. The current plan is kept if it is safe. Otherwise candidates are generated: other speed / start time, other
     path (via a raised or robot-side point), another valid operation offered by the task expert, wait (smooth
     stop where the arm is), retreat to the expert's safe pose.
  4. Candidates closer than the clearance to the person at any check time, or with an IK miss, are dropped.
  5. The cheapest one wins: expected time to finish its operation, plus a weight on the idle (waiting) part of
     it, plus a cost for changing operation or path.
  6. Hysteresis: a new operation / path must be cheaper by `hysteresis_s`; resuming after a wait needs
     `resume_margin_m` of extra clearance.
  7. No safe candidate at all: keep safe (stand still or move to the clearest escape point, kept with hysteresis).
The task expert only lists the valid remaining operations (`operations()`) and a safe pose (`safe_pose()`);
the occupancy, the sweep check, the candidate set and the cost are the same for the three tasks.

Sources: ManiCast (Kedia, Dan, Bhardwaj, Choudhury, 2023) plans robot motion against the cost of the forecast
human trajectory; Mainprice and Berenson (2013) plan against the predicted swept volume of the human arm;
Unhelkar et al. (2018) adapt the robot's timing (speed / start) to predicted human motion instead of stopping;
Tonola et al. (2025) replan the path from the human state rather than only slowing on a fixed path;
ISO/TS 15066:2016 and Marvel and Norcross (2017): speed and separation monitoring, S_h = v_h (T_r + T_s).
"""
from dataclasses import dataclass, field

import mujoco
import numpy as np

from intent_policy.shared.sim_env import STATION, DT, BASE_XY, JOINTS
from intent_policy.shared.cartesian_controller import WORK_SPEED, WORK_ACCEL, VERTICAL_PHASES

CFG = STATION['predictive_avoidance']
CLEAR = float(CFG['clearance_m'])
RESUME = float(CFG['resume_margin_m'])
HORIZON = float(CFG['check_horizon_s'])
DELAYS = [float(v) for v in CFG['delays_s']]
SCALES = [float(v) for v in CFG['speed_scales']]
SWITCH_COST = float(CFG['switch_cost_s'])
REROUTE_COST = float(CFG['reroute_cost_s'])
WAIT_COST = float(CFG['wait_cost_s'])
IDLE_WEIGHT = float(CFG['idle_weight'])
HYST = float(CFG['hysteresis_s'])
DWELL = {k: float(v) for k, v in CFG['dwell_s'].items()}
IK_TOL = float(CFG['ik_tol_m'])
RETREAT_FROM = float(CFG['retreat_judged_from_s'])
SSM_REACTION = float(CFG['ssm_reaction_s'])
UP, BACK, MAX_Z = (float(CFG['reroute'][k]) for k in ('up_m', 'back_m', 'max_z_m'))
CHECK_T = np.round(np.r_[np.arange(0., 1.0001, .1), np.arange(1.25, HORIZON + 1e-6, .25)], 3)
STEP_K = np.minimum(np.round(CHECK_T / .1).astype(int), 10)       # forecast index used at each check time


# ---------------------------------------------------------------------- geometry
def seg_dist(a0, a1, b0, b1):
    """Distance between segments a0-a1 [..., 3] and b0-b1 [..., 3] (broadcast); clamped closest points."""
    d1, d2, r = a1 - a0, b1 - b0, a0 - b0
    a = np.sum(d1 * d1, -1); e = np.sum(d2 * d2, -1); f = np.sum(d2 * r, -1)
    c = np.sum(d1 * r, -1); b = np.sum(d1 * d2, -1)
    den = np.maximum(a * e - b * b, 1e-12)
    s = np.where(a > 1e-12, np.clip((b * f - c * e) / den, 0., 1.), 0.)
    t = np.where(e > 1e-12, (b * s + f) / np.maximum(e, 1e-12), 0.)
    t0 = np.clip(t, 0., 1.)
    s = np.where(t < 0., np.where(a > 1e-12, np.clip(-c / np.maximum(a, 1e-12), 0., 1.), 0.),
                 np.where(t > 1., np.where(a > 1e-12, np.clip((b - c) / np.maximum(a, 1e-12), 0., 1.), 0.), s))
    p = a0 + d1 * s[..., None]; q = b0 + d2 * t0[..., None]
    return np.linalg.norm(p - q, axis=-1)


_LOCAL = {}


def _local_capsule(m, g):
    """Capsule (end A, end B, radius) in the geom frame that contains the whole geom. Meshes: tightest capsule
    along the longest axis that covers every vertex; primitives: closed form (cylinder / box corners covered)."""
    key = (id(m), g)
    if key in _LOCAL:
        return _LOCAL[key]
    t, size = int(m.geom_type[g]), m.geom_size[g]
    z = np.zeros(3)
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        out = (z, z, float(size[0]))
    elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
        out = (np.array([0, 0, -size[1]]), np.array([0, 0, size[1]]), float(size[0]))
    elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        out = (np.array([0, 0, -size[1]]), np.array([0, 0, size[1]]), float(size[0]))
    elif t == mujoco.mjtGeom.mjGEOM_MESH:
        mid = int(m.geom_dataid[g]); a, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
        v = np.asarray(m.mesh_vert[a:a + n], float)
        c = (v.max(0) + v.min(0)) / 2; ax = int(np.argmax(v.max(0) - v.min(0)))
        e = np.eye(3)[ax]; zz = (v - c) @ e
        rho = np.linalg.norm((v - c) - np.outer(zz, e), axis=1)
        r = float(rho.max()); cap = np.sqrt(np.maximum(r * r - rho * rho, 0.))
        # a vertex past an end is covered when (z - end)^2 + rho^2 <= r^2: lower end <= z + cap, upper end >= z - cap
        a_end, b_end = float((zz + cap).min()), float((zz - cap).max())
        out = (c + e * min(a_end, b_end), c + e * max(a_end, b_end), r)
    else:                                                           # box / ellipsoid: longest axis, corners covered
        h = np.asarray(size[:3], float); ax = int(np.argmax(h)); others = [k for k in range(3) if k != ax]
        r = float(np.hypot(h[others[0]], h[others[1]])); e = np.eye(3)[ax] * h[ax]
        out = (-e, e, r)
    _LOCAL[key] = out
    return out


def capsule_from_geom(m, d, g):
    """World capsule (end A, end B, radius) containing a geom (checked against mj_geomDistance in tests)."""
    a, b, r = _local_capsule(m, g)
    R = d.geom_xmat[g].reshape(3, 3)
    return d.geom_xpos[g] + R @ a, d.geom_xpos[g] + R @ b, r


class HumanOccupancy:
    """Capsules the person's arms (and held objects) occupy at every check time: arrays A, B [T, n, 3], r [n]."""

    def __init__(self, ep):
        m, d = ep.m, ep.d
        kp = np.asarray(ep.kp, float)
        # forecast used only while the observed hand moves (54/QĐ-DL common condition, ForecastPlanner.active)
        f = ep.forecast_for_expert() if ep.planner.active() else None
        steps = [kp] + ([f[k] for k in range(10)] if f is not None else [kp] * 10)
        steps = np.stack(steps)                                     # [11, 6, 3]
        idx = {n: i for i, n in enumerate(JOINTS)}
        A, B, R = [], [], []
        for s in 'lr':
            S = s.upper()
            sh, el, wr = steps[:, idx[f'{s}_shoulder']], steps[:, idx[f'{s}_elbow']], steps[:, idx[f'{s}_wrist']]
            A += [sh, el]; B += [el, wr]
            R += [ep.validator.radius(f'human_v3_{S}_Shoulder_{S}_Elbow'), ep.validator.radius(f'human_v3_{S}_Elbow_{S}_Wrist')]
            # hand geometry (palm sphere, finger capsule) carried rigidly with the wrist over the horizon
            shift = wr - wr[0]
            for gname in (f'human_v3_{S}_Wrist_palm', f'human_v3_{S}_Wrist_fingers'):
                a, b, r = capsule_from_geom(m, d, m.geom(gname).id)
                A.append(a + shift); B.append(b + shift); R.append(r)
            held = [n for n in ep.held_by_human if (s == ep.human.active)]
            held += [n for n in getattr(ep, 'idle_hand_objects', {}) if s != ep.human.active]
            for n in held:
                for g in ep.validator.held_geoms(n):
                    a, b, r = capsule_from_geom(m, d, g)
                    A.append(a + shift); B.append(b + shift); R.append(r)
        self.A = np.stack(A, 1)[STEP_K]; self.B = np.stack(B, 1)[STEP_K]; self.r = np.asarray(R)   # [T, n, 3]
        self.forecast = f is not None


# ---------------------------------------------------------------------- operations offered by the expert
@dataclass
class Op:
    """One valid remaining operation: a list of steps {'kind': 'move', 'xyz', 'phase', 'grip'} |
    {'kind': 'close', 'obj'} | {'kind': 'open'}; `claim` is the object the person sees the robot claim."""
    key: str
    steps: list
    claim: str = None


def pick_op(ep, name, travel_z, grasp_dz):
    """Operation: take `name` from where it lies (approach above, descend, close, lift)."""
    p = ep.pos(name)
    return Op(f'pick:{name}', [
        {'kind': 'move', 'xyz': np.r_[p[:2], travel_z], 'phase': 'approach', 'grip': ep.grip_open},
        {'kind': 'move', 'xyz': np.r_[p[:2], p[2] + grasp_dz], 'phase': 'descend', 'grip': ep.grip_open},
        {'kind': 'close', 'obj': name},
        {'kind': 'move', 'xyz': np.r_[p[:2], travel_z], 'phase': 'lift', 'grip': ep.grip_closed}], claim=name)


def place_op(ep, key, xyz, travel_z, event, claim=None):
    """Operation: put the held object at `xyz` (move above, lower, open, retreat up)."""
    xyz = np.asarray(xyz, float)
    return Op(key, [
        {'kind': 'move', 'xyz': np.r_[xyz[:2], travel_z], 'phase': 'move', 'grip': ep.grip_closed},
        {'kind': 'move', 'xyz': xyz, 'phase': 'lower', 'grip': ep.grip_closed},
        {'kind': 'open', 'event': event},
        {'kind': 'move', 'xyz': np.r_[xyz[:2], travel_z], 'phase': 'retreat', 'grip': ep.grip_open}], claim=claim)


@dataclass
class Plan:
    kind: str                         # 'op', 'wait', 'retreat', 'stop'
    op: Op = None
    path: str = 'direct'
    delay: float = 0.
    scale: float = 1.
    via: list = field(default_factory=list)
    cost: float = np.inf
    clearance: float = np.inf

    def label(self, current_key):
        if self.kind != 'op':
            return self.kind
        if current_key is not None and self.op.key != current_key:
            return 'switch'
        if self.path != 'direct':
            return 'reroute'                          # other path (possibly also later / slower)
        if self.delay > 0:
            return 'wait'                             # hold now, start the same operation later
        if self.scale < 1.:
            return 'slow'                             # same path, lower speed
        return 'continue'


# ---------------------------------------------------------------------- planner
class PredictiveAvoider:
    def __init__(self, ep):
        self.ep = ep
        m = ep.m
        self.scratch = mujoco.MjData(m)
        self.robot_geoms = ep.validator.robot_collision
        self.cache = {}
        self.current = None           # Plan being executed
        self.step = 0                 # index into current.op.steps
        self.waiting = False
        self.decisions = {}           # label -> ticks
        self.stop_goal = None         # fixed natural stopping point while waiting
        self.last_label = None
        self.safe_choice = None       # (name, goal) of the keep-safe option in use (58/QĐ-DL hysteresis)

    # ------------------------------------------------------------ robot sweep
    def robot_capsules(self, xyz, seed):
        """(A, B, r, q, ik_error) of the robot links with the TCP at `xyz` (cached on a 1 cm grid)."""
        key = tuple(np.round(np.asarray(xyz) / .01).astype(int))
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        env, d = self.ep.env, self.scratch
        q = env.solve_ik(np.asarray(xyz, float), seed=seed, data=d)
        d.qpos[env.qids] = q
        mujoco.mj_kinematics(env.model, d)
        err = float(np.linalg.norm(np.asarray(xyz) - d.site_xpos[env.tcp]))
        caps = [capsule_from_geom(env.model, d, g) for g in self.robot_geoms]
        out = (np.array([c[0] for c in caps]), np.array([c[1] for c in caps]), np.array([c[2] for c in caps]), q, err)
        self.cache[key] = out
        return out

    def held_capsules(self, tcp_at):
        """Capsules of the object the robot holds, moved rigidly with the TCP."""
        ep = self.ep
        name = ep.attached_name()
        if name is None:
            return []
        shift = np.asarray(tcp_at) - ep.tcp()
        return [(a + shift, b + shift, r) for a, b, r in
                (capsule_from_geom(ep.m, ep.d, g) for g in ep.validator.held_geoms(name))]

    def timeline(self, plan):
        """TCP waypoints with arrival times [(t, xyz)] for a candidate, plus its total duration."""
        ep = self.ep
        p = ep.tcp(); v = ep.controller.follower.velocity.copy(); t = 0.
        pts = [(0., p.copy())]
        if plan.delay > 0 or plan.kind in ('wait', 'stop'):
            sp = float(np.linalg.norm(v))
            stop = p + v * sp / (2 * WORK_ACCEL); t_stop = sp / WORK_ACCEL
            p = stop; t += t_stop; pts.append((t, p.copy()))
            t = max(t, plan.delay) if plan.kind == 'op' else HORIZON
            pts.append((t, p.copy()))
        if plan.kind in ('wait', 'stop'):
            return pts, t
        steps = list(plan.op.steps[self.step if (self.current is not None and self.current.op is not None
                                                 and plan.op.key == self.current.op.key) else 0:]) if plan.kind == 'op' else \
            [{'kind': 'move', 'xyz': plan.op.steps[0]['xyz'], 'phase': 'retreat'}]
        moves = [{'kind': 'move', 'xyz': w, 'phase': 'via'} for w in plan.via] + steps
        for s in moves:
            if s['kind'] == 'move':
                q = np.asarray(s['xyz'], float)
                dist = float(np.linalg.norm(q - p))
                speed = (WORK_SPEED['vertical'] if s.get('phase') in VERTICAL_PHASES else WORK_SPEED['horizontal']) * plan.scale
                t += dist / max(speed, 1e-6) + .1          # + acceleration / settling allowance
                p = q; pts.append((t, p.copy()))
            else:
                t += DWELL[s['kind']]; pts.append((t, p.copy()))
        return pts, t

    @staticmethod
    def at(pts, t):
        for (t0, p0), (t1, p1) in zip(pts, pts[1:]):
            if t <= t1:
                u = 0. if t1 - t0 < 1e-9 else (t - t0) / (t1 - t0)
                return p0 + (p1 - p0) * u
        return pts[-1][1]

    def clearance(self, pts, occ, t_from=0., full=False):
        """Minimum distance robot (links + held object) - person over the check times from `t_from`; -inf on an
        IK miss (a retreat is judged from 0.3 s on: it may start inside the clearance it is leaving)."""
        seed = self.ep.d.qpos[self.ep.env.qids].copy()
        best = np.inf
        for i, t in enumerate(CHECK_T):
            if t < t_from - 1e-9:
                continue
            x = self.at(pts, t)
            A, B, r, q, err = self.robot_capsules(x, seed)
            if err > IK_TOL:
                return -np.inf
            seed = q
            held = self.held_capsules(x)
            if held:
                A = np.vstack([A, [h[0] for h in held]]); B = np.vstack([B, [h[1] for h in held]]); r = np.r_[r, [h[2] for h in held]]
            dist = seg_dist(A[:, None], B[:, None], occ.A[i][None], occ.B[i][None]) - r[:, None] - occ.r[None]
            best = min(best, float(dist.min()))
            if best < CLEAR - .05 and not full:  # clearly unsafe: no need to look further
                break
        return best

    # ------------------------------------------------------------ candidates
    def vias(self, goal):
        """Path variants to the next move goal: direct, over a raised point, through a point towards the base."""
        p = self.ep.tcp(); g = np.asarray(goal, float); mid = (p + g) / 2
        up = np.r_[mid[:2], min(max(p[2], g[2]) + UP, MAX_Z)]
        v = np.asarray(BASE_XY) - mid[:2]; v /= max(float(np.linalg.norm(v)), 1e-9)
        back = np.r_[mid[:2] + v * BACK, max(p[2], g[2])]
        return {'direct': [], 'up': [up], 'back': [back]}

    def first_goal(self, op, same):
        steps = op.steps[self.step:] if same else op.steps
        for s in steps:
            if s['kind'] == 'move':
                return s['xyz']
        return self.ep.tcp()

    def candidates(self, ops, safe_pose):
        cur_key = None if self.current is None or self.current.op is None else self.current.op.key
        out = []
        for op in ops:
            same = op.key == cur_key and self.current.kind == 'op'
            paths = self.vias(self.first_goal(op, same))
            if same and self.current.via:
                paths = {self.current.path: list(self.current.via), 'direct': []}
            for path, via in paths.items():
                for delay in DELAYS:
                    for scale in SCALES:
                        out.append(Plan('op', op, path, delay, scale, via))
        out.append(Plan('wait'))
        out.append(Plan('retreat', Op('retreat', [{'kind': 'move', 'xyz': np.asarray(safe_pose, float), 'phase': 'retreat'}])))
        return out

    def cost(self, plan, duration, cur_key):
        if plan.kind == 'wait':
            return WAIT_COST
        if plan.kind == 'retreat':
            return WAIT_COST + 1.
        c = duration + IDLE_WEIGHT * plan.delay            # rule 6: fastest finish, then little idle time
        if cur_key is not None and plan.op.key != cur_key:
            c += SWITCH_COST
        if plan.path != 'direct' and not (self.current is not None and plan.path == self.current.path):
            c += REROUTE_COST
        return c

    def decide(self, ops, safe_pose):
        occ = HumanOccupancy(self.ep)
        cur_key = None if self.current is None or self.current.op is None else self.current.op.key
        # 58/QĐ-DL speed and separation monitoring (ISO/TS 15066:2016; Marvel and Norcross 2017): the distance the
        # observed hand covers while the robot reacts and stops, S_h = v_h (T_r + T_s), on top of the clearance;
        # same in both modes (observed hand speed only)
        t_stop = float(np.linalg.norm(self.ep.controller.follower.velocity)) / WORK_ACCEL
        clear = CLEAR + float(self.ep.hand_speed) * (SSM_REACTION + t_stop)
        best, keep = None, None
        for plan in self.candidates(ops, safe_pose):
            pts, dur = self.timeline(plan)
            need = clear + (RESUME if (self.waiting and plan.kind == 'op' and plan.delay == 0.) else 0.)
            if plan.kind == 'wait':
                need = clear + (RESUME if (self.current is not None and self.current.kind == 'retreat') else 0.)
            clr = self.clearance(pts, occ, RETREAT_FROM if plan.kind == 'retreat' else 0.)
            if clr < need:
                continue
            plan.cost, plan.clearance = self.cost(plan, dur, cur_key), clr
            if best is None or plan.cost < best.cost:
                best = plan
            if (self.current is not None and plan.kind == self.current.kind == 'op' and plan.op.key == cur_key
                    and plan.path == self.current.path and (keep is None or plan.cost < keep.cost)):
                keep = plan                                 # best timing of the current operation and path
        if best is None:
            return self.keep_safe(occ, safe_pose), occ
        self.safe_choice = None
        if keep is not None and keep.cost <= best.cost + HYST:
            return keep, occ                                # current plan still safe and not clearly worse
        return best, occ

    def keep_safe(self, occ, safe_pose):
        """Rule 8: no candidate keeps the clearance (e.g. the person's hand comes to where the robot is). Keep as
        safe as possible: of standing still, the safe pose, straight up, towards the base, and away from the
        person's nearest predicted arm point (horizontally, also raised), take the one with the largest predicted
        clearance ('stop' if standing still, else 'evade'). 58/QĐ-DL: the choice is kept unless another option is
        `resume_margin_m` clearer (switching between options every tick left the arm nearly in place, W2)."""
        p = self.ep.tcp()
        v = np.asarray(BASE_XY) - p[:2]; v /= max(float(np.linalg.norm(v)), 1e-9)
        # nearest point of the person's predicted occupancy (any check time) to the TCP: escape away from it
        u = np.clip(np.sum((p - occ.A) * (occ.B - occ.A), -1) / np.maximum(np.sum((occ.B - occ.A) ** 2, -1), 1e-12), 0., 1.)
        near = occ.A + (occ.B - occ.A) * u[..., None]
        k = np.unravel_index(np.argmin(np.linalg.norm(near - p, axis=-1)), u.shape)
        a = p[:2] - near[k][:2]; a /= max(float(np.linalg.norm(a)), 1e-9)
        z_up = min(p[2] + UP, MAX_Z)
        goals = {'safe_pose': np.asarray(safe_pose, float), 'up': np.r_[p[:2], z_up],
                 'back': np.r_[p[:2] + v * BACK, p[2]], 'back_up': np.r_[p[:2] + v * BACK, z_up],
                 'away': np.r_[p[:2] + a * BACK, p[2]], 'away_up': np.r_[p[:2] + a * BACK, z_up]}
        prev = self.safe_choice
        if prev is not None and prev[0] != 'stop':
            goals['kept'] = prev[1]                                             # the escape point already chosen
        opts = []
        plan = Plan('stop'); pts, _ = self.timeline(plan); plan.clearance = self.clearance(pts, occ, RETREAT_FROM, full=True)
        opts.append(('stop', plan))
        for name, g in goals.items():
            plan = Plan('evade', Op(f'evade_{name}', [{'kind': 'move', 'xyz': g, 'phase': 'avoid_evade'}]), path=name)
            pts, _ = self.timeline(plan)
            plan.clearance = self.clearance(pts, occ, RETREAT_FROM, full=True)
            opts.append((name, plan))
        best = max(opts, key=lambda o: o[1].clearance)
        if prev is not None:
            kept = next((o for o in opts if o[0] == ('kept' if prev[0] != 'stop' else 'stop')), None)
            if kept is not None and kept[1].clearance >= best[1].clearance - RESUME:
                best = kept
        name, plan = best
        if name != 'kept':
            self.safe_choice = ('stop', None) if plan.kind == 'stop' else (name, plan.op.steps[0]['xyz'])
        return plan

    # ------------------------------------------------------------ driver
    def run(self, expert):
        """Shared W driver: replan every tick, execute one command (atomic gripper steps run to completion)."""
        ep = self.ep
        while True:
            ops = expert.operations()
            cur = self.current
            if cur is not None and cur.kind == 'op' and self.step > 0 and all(o.key != cur.op.key for o in ops):
                ops = [cur.op] + ops                        # finish the steps of an operation already under way
            if not ops:
                break
            plan, occ = self.decide(ops, expert.safe_pose())
            cur_key = None if self.current is None or self.current.op is None else self.current.op.key
            label = plan.label(cur_key) if plan.kind == 'op' else {'wait': 'hold'}.get(plan.kind, plan.kind)
            self.decisions[label] = self.decisions.get(label, 0) + 1
            if label != self.last_label:
                ep.event('avoid_decision', decision=label, op=None if plan.op is None else plan.op.key,
                         path=plan.path, delay_s=plan.delay, speed_scale=plan.scale,
                         clearance_m=None if not np.isfinite(plan.clearance) else round(plan.clearance, 4),
                         forecast=occ.forecast)
                self.last_label = label
            if plan.kind == 'op':
                if cur_key != plan.op.key:
                    self.step = 0
                    self.claim = plan.op.claim          # avoider state; the person reads events.py (09/QĐ-KT)
                    ep.event('robot_claims', op=plan.op.key, claim=plan.op.claim)
                    # leave a low pose vertically before heading for another operation
                    if self.current is not None and ep.tcp()[2] < plan.op.steps[0]['xyz'][2] - .02 and plan.op.steps[0]['kind'] == 'move':
                        plan.via = [np.r_[ep.tcp()[:2], plan.op.steps[0]['xyz'][2]]] + list(plan.via)
                self.current = plan
                self.waiting = plan.delay > 0
                if plan.delay == 0:
                    self.stop_goal = None
                if plan.delay > 0:
                    yield self.smooth_stop(f'wait_{plan.op.key}')
                    continue
                yield from self.execute(plan)
            elif plan.kind in ('retreat', 'evade'):
                self.waiting = True; self.stop_goal = None
                self.current = Plan(plan.kind, plan.op, path=plan.path)
                g = plan.op.steps[0]['xyz']
                yield ep.command_to(g, 'avoid_' + plan.kind, self.grip())
            else:                                           # wait (safe where it is) or stop (nothing is safe)
                self.waiting = True
                if self.current is not None and self.current.kind != 'op':
                    self.current = None
                yield self.smooth_stop('avoid_wait' if plan.kind == 'wait' else 'avoid_stop')
        self.claim = None

    def grip(self):
        return self.ep.grip_closed if self.ep.attached_name() is not None else self.ep.grip_open

    def smooth_stop(self, phase):
        """Come to rest at the natural stopping point (bounded deceleration, no hold() jerk)."""
        ep = self.ep
        if self.stop_goal is None:
            v = ep.controller.follower.velocity
            sp = float(np.linalg.norm(v))
            base = ep.controller.setpoint if ep.controller.setpoint is not None else ep.tcp()
            self.stop_goal = base + v * sp / (2 * WORK_ACCEL)
        return ep.command_to(self.stop_goal, phase, self.grip())

    def execute(self, plan):
        """One tick of the chosen plan (via points first), or a whole atomic gripper step."""
        ep = self.ep
        if plan.via:
            g = plan.via[0]
            if ep.reached(g, .015):
                plan.via.pop(0)
            else:
                yield ep.command_to(g, 'via', self.grip(), scale=plan.scale)
                return
        steps = plan.op.steps
        if self.step >= len(steps):
            self.current = None; return
        s = steps[self.step]
        if s['kind'] == 'move':
            tol = s.get('tol', .004)
            if ep.reached(s['xyz'], tol):
                ep.controller.follower.velocity *= 0.
                self.step += 1
                if self.step >= len(steps):
                    self.finish(plan)
                    return
                yield from self.execute(plan)
                return
            yield ep.command_to(s['xyz'], s['phase'], s.get('grip', self.grip()), scale=plan.scale)
        elif s['kind'] == 'close':
            ok = yield from ep.close_on(s['obj'])
            self.step += 1
            if not ok:
                ep.event('grasp_failed_op', op=plan.op.key); self.current = None
        elif s['kind'] == 'open':
            yield from ep.open_gripper()
            if s.get('event'):
                ev = dict(s['event']); ep.event(ev.pop('name'), **ev)
            self.step += 1
        if self.step >= len(steps):
            self.finish(plan)

    def finish(self, plan):
        self.ep.event('op_done', op=plan.op.key)
        self.current = None; self.step = 0; self.claim = None

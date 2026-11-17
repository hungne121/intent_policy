"""Physics validator shared by both expert modes (QĐ 53/QĐ-DL §3, §8).

Measures, never modifies commands: human contacts at every physics substep (depth, normal force), geometric
overlaps of kinematic pairs at every control tick, human self-intersection, robot joint position / velocity /
acceleration against the UR3e limits, actuator force saturation and grasp stability. The same thresholds apply
to observed and forecast rollouts. Moved from scripts/v2_preview_common.py, i3_self_check.py, contact_context.py.
"""
import mujoco
import numpy as np
import yaml

from intent_policy.sim.demo_contact_audit import DemoContactAudit, PairRule
from intent_policy.sim.human_self_check import HumanSelfCheck
from intent_policy.shared.sim_env import ROOT, STATION, TOL, DT, geom_sets

ROBOT_CFG = yaml.safe_load((ROOT / 'configs/robot.yaml').read_text())
CONTACTS = yaml.safe_load((ROOT / 'configs/hrc/contacts.yaml').read_text())   # 61/QĐ-DL classes + limits
CATEGORIES = ('allowed_contact', 'unintended_contact', 'human_self_collision', 'invalid_contact')
LIMIT_MARGIN = 1e-3          # rad, distance to a joint limit counted as a hit


# ------------------------------------------------------------------ allowed-contact rule
def overlap_tolerance(*, robot_body, other_body, other_geom, attached, selection,
                      phase, static_robot, human_pair, tolerances):
    """Contact allowance from the current interaction state only."""
    if human_pair:
        return tolerances['human']
    if other_geom == 'table':
        return None if static_robot else tolerances['robot_table']
    intended = other_body == attached or (
        other_body == selection and phase in {'close', 'open', 'return_open', 'hold_switch'})
    if 'finger' in robot_body and intended:
        return tolerances['finger_grasp']
    return tolerances['robot_other']


# ------------------------------------------------------------------ human self-intersection
V3_TRUNK = ('Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Head', 'L_Collar', 'R_Collar')
V3_ARM = {'l': ('L_Shoulder', 'L_Elbow', 'L_Wrist'), 'r': ('R_Shoulder', 'R_Elbow', 'R_Wrist')}


def _collision_geoms(model, bodies):
    return [g for g in range(model.ngeom)
            if model.body(model.geom_bodyid[g]).name in bodies and (model.geom_contype[g] or model.geom_conaffinity[g])]


class V3SelfCheck:
    """v3 person: each arm's collision geometry against the trunk capsules, and the two arms against each other."""
    def __init__(self, model):
        self.model = model
        arms = {s: _collision_geoms(model, {f'human_seg_{j}' for j in V3_ARM[s]}) for s in 'lr'}
        trunk = _collision_geoms(model, {f'human_seg_{j}' for j in V3_TRUNK})
        pairs = [PairRule(a, b, 0., 0.) for s in 'lr' for a in arms[s] for b in trunk]
        shoulder = {g for s in 'lr' for g in arms[s] if 'Shoulder' in model.geom(g).name}
        pairs = [p for p in pairs if p.geom_a not in shoulder]
        pairs += [PairRule(a, b, 0., 0.) for a in arms['l'] for b in arms['r']]
        self.audit = DemoContactAudit(model, pairs)

    def measure(self, data):
        self.audit.observe(self.model, data)
        out = {}
        for index, distance, _, _ in self.audit.step_violations:
            pair = self.audit.rules[index]
            names = (self.model.geom(pair.geom_a).name, self.model.geom(pair.geom_b).name)
            out[names] = max(out.get(names, 0.), -float(distance))
        return out


class V2SelfCheck(HumanSelfCheck):
    def __init__(self, model):
        super().__init__(model)
        arms = {s: [g for g in range(model.ngeom) if model.body(model.geom_bodyid[g]).name in
                    {f'human_seg_{s}_{p}' for p in ('upperarm', 'forearm', 'hand')}] for s in 'lr'}
        self.opposite = DemoContactAudit(model, [PairRule(a, b, 0., 0.) for a in arms['l'] for b in arms['r']])

    def measure(self, data):
        result = super().measure(data)
        self.opposite.observe(self.model, data)
        for index, distance, _, _ in self.opposite.step_violations:
            pair = self.opposite.rules[index]
            names = (self.model.geom(pair.geom_a).name, self.model.geom(pair.geom_b).name)
            result[names] = max(result.get(names, 0.), -float(distance))
        return result


def self_check(model):
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'human_seg_Pelvis') >= 0:
        return V3SelfCheck(model)
    return V2SelfCheck(model)


# ------------------------------------------------------------------ validator
class PhysicsValidator:
    def __init__(self, ep):
        self.ep = ep
        m = ep.m
        self._is_robot, self.robot_geoms, self.human_geoms, self.world_geoms = geom_sets(m, ep.positions)
        self.v3 = ep.human_model in ('v3', 'v4')
        if self.v3:
            # audit the collision geometry of the articulated person, not its visual hand meshes
            self.human_geoms = [g for g in self.human_geoms if m.geom_contype[g] or m.geom_conaffinity[g]]
        self.human_set = set(self.human_geoms)
        arm_words = ('Shoulder', 'Elbow', 'Wrist', 'upperarm', 'forearm', 'hand')
        self.human_arm = [g for g in self.human_geoms if any(w in m.body(m.geom_bodyid[g]).name for w in arm_words)
                          and (m.geom_contype[g] or m.geom_conaffinity[g])]
        self.robot_collision = [g for g in self.robot_geoms if m.geom_contype[g] or m.geom_conaffinity[g]]
        self.away = None
        pairs = [PairRule(a, b, 0., 0.) for a in self.human_geoms for b in self.robot_geoms + self.world_geoms]
        pairs += [PairRule(a, b, 0., 0.) for a in self.robot_geoms for b in self.world_geoms]
        self.audit = DemoContactAudit(m, pairs)
        from intent_policy.sim.human_physics import physics_config
        self.contact_depth_tolerance = float(physics_config()['max_contact_depth_m'])
        self.proximity_margin = float(STATION.get('proximity_margin_m', 0.))
        if self.v3 and self.proximity_margin > 0:
            for g in self.human_geoms:             # proximity contacts: detected within margin, no force
                m.geom_margin[g] = self.proximity_margin; m.geom_gap[g] = self.proximity_margin
        self.self_check = self_check(m)
        self.contact_measurements, self.violations, self.self_hits = {}, {}, {}
        self.visual_overlap_warnings, self.human_contacts = {}, {}
        self.closest_human_m = np.inf
        self.physics_contact_samples = 0
        limits = np.asarray(ep.env.arm_limits, float)
        self.joint_lo, self.joint_hi = limits[:, 0], limits[:, 1]
        self.joint_vmax = np.asarray(ROBOT_CFG['max_joint_speed'], float)
        self.robot = {'joint_limit_hits': 0, 'max_joint_speed_ratio': 0., 'max_joint_accel_rad_s2': 0.,
                      'force_saturation_ticks': 0, 'grasp_drops': 0, 'max_ik_error_m': 0.}
        self._prev_qvel = None
        self._attached_prev = None
        # 61/QĐ-DL: raw contact samples per control tick, classified at the end (grasp windows need the events)
        self.samples = {}                  # (tick, geom_a, geom_b) -> [depth, force]
        self.context = []                  # per tick: phases, attached, selection, held_by_human
        self.objects = set(ep.positions)
        self.robot_set = set(self.robot_geoms)
        gid = [i for i in range(m.nu) if 'grip' in (m.actuator(i).name or '').lower() or i == 6]
        self.gripper_force_max = float(np.abs(m.actuator_forcerange[gid[0]]).max()) if gid and m.actuator_forcelimited[gid[0]] else None
        ep.env.solver_observers.append(self._audit_physics_contacts)

    def _sample(self, ga, gb, depth, force=0.):
        if depth <= 0:
            return
        k = (self.ep.tick, ga, gb) if ga < gb else (self.ep.tick, gb, ga)
        r = self.samples.get(k)
        if r is None:
            self.samples[k] = [float(depth), float(force)]
        else:
            r[0] = max(r[0], float(depth)); r[1] = max(r[1], float(force))

    # ---------------------------------------------------------------- substep contacts
    def _audit_physics_contacts(self, model, data):
        self.physics_contact_samples += 1
        for i, c in enumerate(data.contact[:data.ncon]):
            a, b = int(c.geom1), int(c.geom2)
            static = model.body_weldid[model.geom_bodyid[a]] == 0 and model.body_weldid[model.geom_bodyid[b]] == 0
            if c.dist < 0 and not static and (a in self.robot_set or b in self.robot_set) \
                    and a not in self.human_set and b not in self.human_set:
                f = np.zeros(6); mujoco.mj_contactForce(model, data, i, f)
                self._sample(a, b, -float(c.dist), float(f[0]))
            if a not in self.human_set and b not in self.human_set:
                continue
            if b in self.human_set:
                a, b = b, a
            if c.dist >= 0:
                continue
            ba = model.body(model.geom_bodyid[a]).name
            other_geom = model.geom(b).name or str(b)
            bb = model.body(model.geom_bodyid[b]).name
            if bb == 'world':
                bb = other_geom
            key = f'{ba}|{bb}'
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, i, force)
            self._sample(a, b, -float(c.dist), float(force[0]))
            rec = self.contact_measurements.setdefault(key, {'samples': 0, 'max_depth_m': 0., 'max_normal_force_n': 0.,
                                                             'first_t_s': round(float(data.time), 4)})
            rec['samples'] += 1
            rec['max_depth_m'] = max(rec['max_depth_m'], -float(c.dist))
            rec['max_normal_force_n'] = max(rec['max_normal_force_n'], float(force[0]))
            if -c.dist > self.contact_depth_tolerance:
                self.violations[key] = max(self.violations.get(key, 0.), -float(c.dist))
            if b in self.robot_geoms:
                names = f'{model.geom(a).name}|{model.geom(b).name or b}'
                self.human_contacts[names] = min(self.human_contacts.get(names, 0.), float(c.dist))
                self.closest_human_m = min(self.closest_human_m, float(c.dist))

    # ---------------------------------------------------------------- robot-person distance (observed)
    def radius(self, geom_name):
        return float(self.ep.m.geom_size[self.ep.m.geom(geom_name).id][0])

    def held_geoms(self, name):
        m = self.ep.m; body = m.body(name).id
        return [g for g in range(m.ngeom) if m.geom_bodyid[g] == body and (m.geom_contype[g] or m.geom_conaffinity[g])]

    def robot_near_human(self, max_dist=.25, include_held=False):
        """(min signed distance, closest pair) between the robot's collision geometry and the person's arms
        (upper arm, forearm, hand) from the OBSERVED current pose, up to `max_dist`. Geometric distance
        (mj_geomDistance), not only solver contacts: the contact margin (3 cm) cannot see 0.16 m (fixed 16/11).
        `include_held` (57/QĐ-DL, expert safety rules): the object the robot holds counts as robot (C2: the held brush
        must keep the clearance too, not only the gripper); with 'both' objects the person holds also count as
        person. An object both hold (the handover, the C3 tray) counts as neither. The contact
        audit keeps the body-only measure."""
        m, d = self.ep.m, self.ep.d
        best = (np.inf, None)
        fromto = np.zeros(6)
        self.away = None                       # unit vector from the closest arm point to the closest robot point
        robot, human = self.robot_collision, self.human_arm
        if include_held:
            mine = self.ep.attached_name()
            theirs = set(self.ep.held_by_human)
            shared = {mine} & theirs
            if mine is not None and mine not in shared:
                robot = robot + self.held_geoms(mine)
            if include_held == 'both':
                for name in sorted(theirs - shared):
                    human = human + self.held_geoms(name)
        for a in robot:
            for b in human:
                dist = mujoco.mj_geomDistance(m, d, a, b, max_dist, fromto)
                if dist < best[0]:
                    best = (float(dist), (m.geom(a).name, m.geom(b).name))
                    v = fromto[:3] - fromto[3:]
                    n = float(np.linalg.norm(v))
                    self.away = v / n if n > 1e-9 else None
        return best if best[0] < max_dist else (np.inf, None)

    # ---------------------------------------------------------------- control-tick geometry
    def audit_geometry(self):
        ep = self.ep
        m, d = ep.m, ep.d
        if self.v3:
            dist, pair = self.robot_near_human()
            if pair is not None:
                self.closest_human_m = min(self.closest_human_m, dist)
                if dist < 0:
                    key = f'{pair[0]}|{pair[1]}'
                    self.human_contacts[key] = min(self.human_contacts.get(key, 0.), dist)
        self.audit.observe(m, d)
        attached = ep.attached_name()
        if len(self.context) <= ep.tick:
            self.context.append({'robot_phase': ep.phase, 'human_phase': ep.human_phase, 'attached': attached,
                                 'selection': ep.robot_selection, 'held_by_human': set(ep.held_by_human)})
        for idx, dist, _, _ in self.audit.step_violations:
            pair = self.audit.rules[idx]
            if not (m.body_weldid[m.geom_bodyid[pair.geom_a]] == 0 and m.geom(pair.geom_b).name == 'table'):
                self._sample(pair.geom_a, pair.geom_b, -float(dist))
            ba = m.body(m.geom_bodyid[pair.geom_a]).name; bb = m.body(m.geom_bodyid[pair.geom_b]).name
            tol = overlap_tolerance(robot_body=ba, other_body=bb, other_geom=m.geom(pair.geom_b).name,
                                    attached=attached, selection=ep.robot_selection, phase=ep.phase,
                                    static_robot=bool(m.body_weldid[m.geom_bodyid[pair.geom_a]] == 0),
                                    human_pair=ba.startswith('human_seg_'), tolerances=TOL)
            if ba.startswith('human_seg_'):
                tol = self.contact_depth_tolerance
            if ba.startswith('human_seg_') and bb in ep.held_by_human:
                continue                       # the object welded to the hand is an allowed contact
            if tol is not None and -dist > tol:
                key = f'{ba}|{bb}'
                self.violations[key] = max(self.violations.get(key, 0.), -float(dist))
                if not (m.geom_contype[pair.geom_b] or m.geom_conaffinity[pair.geom_b]):
                    vkey = f'{ba}|{m.geom(pair.geom_b).name or pair.geom_b}'
                    self.visual_overlap_warnings[vkey] = max(self.visual_overlap_warnings.get(vkey, 0.), -float(dist))
        for pair, depth in self.self_check.measure(d).items():
            self._sample(m.geom(pair[0]).id, m.geom(pair[1]).id, float(depth))
            if depth > self.contact_depth_tolerance:
                self.self_hits[str(pair)] = max(self.self_hits.get(str(pair), 0.), float(depth))

    def audit_robot(self):
        """Joint limits, joint speed / acceleration, actuator force saturation, grasp stability, IK error."""
        ep = self.ep
        q = ep.d.qpos[ep.env.qids]; qv = ep.d.qvel[ep.env.dids]
        r = self.robot
        if np.any(q <= self.joint_lo + LIMIT_MARGIN) or np.any(q >= self.joint_hi - LIMIT_MARGIN):
            r['joint_limit_hits'] += 1
        r['max_joint_speed_ratio'] = max(r['max_joint_speed_ratio'], float(np.max(np.abs(qv) / self.joint_vmax)))
        if self._prev_qvel is not None:
            r['max_joint_accel_rad_s2'] = max(r['max_joint_accel_rad_s2'], float(np.max(np.abs(qv - self._prev_qvel)) / DT))
        self._prev_qvel = qv.copy()
        frc = ep.d.actuator_force[:6]; rng = ep.m.actuator_forcerange[:6]
        limited = ep.m.actuator_forcelimited[:6].astype(bool)
        if np.any(limited & ((frc <= rng[:, 0] + 1e-6) | (frc >= rng[:, 1] - 1e-6))):
            r['force_saturation_ticks'] += 1
        attached = ep.attached_name()
        if self._attached_prev is not None and attached is None and ep.phase not in ('open', 'release', 'return_open'):
            r['grasp_drops'] += 1
        self._attached_prev = attached
        r['max_ik_error_m'] = max(r['max_ik_error_m'], float(ep.env.last_ik_error))

    # ---------------------------------------------------------------- 61/QĐ-DL classification
    def _windows(self):
        """Per object: list of [grasp_t - w, release_t + w] intervals of the HUMAN hand holding it."""
        w = float(CONTACTS['grasp_window_s']); out, open_ = {}, {}
        for e in self.ep.events:
            if e['event'] == 'human_grasp':
                open_[e['object']] = e['t']
            elif e['event'] == 'human_release' and e['object'] in open_:
                out.setdefault(e['object'], []).append((open_.pop(e['object']) - w, e['t'] + w))
        for o, t0 in open_.items():
            out.setdefault(o, []).append((t0 - w, np.inf))
        return out

    def _classify(self, ga, gb, ctx, t, windows):
        """(category, rule, pair label) of one contact sample."""
        m = self.ep.m
        def side(g):
            if g in self.human_set:
                return 'human'
            if g in self.robot_set:
                return 'robot'
            return 'object' if m.body(m.geom_bodyid[g]).name in self.objects else 'world'
        sa, sb = side(ga), side(gb)
        if (sb, sa) in {('human', 'robot'), ('human', 'object'), ('human', 'world'), ('robot', 'object'), ('robot', 'world')}:
            ga, gb, sa, sb = gb, ga, sb, sa
        na, nb = m.geom(ga).name or str(ga), m.geom(gb).name or str(gb)
        ba, bb = m.body(m.geom_bodyid[ga]).name, m.body(m.geom_bodyid[gb]).name
        label = f"{ba}|{bb if bb != 'world' else nb}"
        in_win = lambda o: any(a <= t <= b for a, b in windows.get(o, ()))
        if sa == 'human' and sb == 'human':
            return 'human_self_collision', 'arm_vs_body', f'{na}|{nb}'
        if sa == 'human' and sb == 'robot':
            return 'invalid_contact', 'robot_touches_person', label
        if sa == 'human' and sb == 'object':
            if bb == ctx['attached']:
                return ('allowed_contact', 'handover', label) if (bb in ctx['held_by_human'] or in_win(bb)) else \
                       ('invalid_contact', 'person_touches_robot_held_object', label)
            if bb in ctx['held_by_human'] or in_win(bb):
                return 'allowed_contact', 'human_grasp', label
            return 'unintended_contact', 'human_object', label
        if sa == 'human':
            return 'unintended_contact', 'human_world', label
        if sa == 'robot' and sb == 'object' and 'finger' in ba and (
                bb == ctx['attached'] or (bb == ctx['selection'] and ctx['robot_phase'] in CONTACTS['robot_grasp_phases'])):
            return 'allowed_contact', 'robot_grasp', label
        if sa == 'robot' and sb == 'object':
            return 'unintended_contact', 'robot_object', label
        return 'unintended_contact', 'robot_world', label

    def classified(self):
        windows = self._windows()
        rigid, selfmax = float(CONTACTS['rigid_penetration_m']), float(CONTACTS['human_self_penetration_m'])
        fmax = float(CONTACTS['human_force_n'])
        out = {}
        for (tick, ga, gb), (depth, force) in self.samples.items():
            ctx = self.context[min(tick, len(self.context) - 1)] if self.context else {
                'robot_phase': None, 'human_phase': None, 'attached': None, 'selection': None, 'held_by_human': set()}
            t = tick * DT
            cat, rule, label = self._classify(ga, gb, ctx, t, windows)
            key = f'{cat}|{rule}|{label}'
            r = out.setdefault(key, {'category': cat, 'rule': rule, 'pair': label, 'ticks': 0, 'max_depth_m': 0.,
                                     'max_force_n': 0., 't_max_s': None, 'robot_phase': None, 'human_phase': None,
                                     'first_t_s': round(t, 3)})
            r['ticks'] += 1
            r['max_force_n'] = max(r['max_force_n'], force)
            if depth > r['max_depth_m']:
                r.update(max_depth_m=depth, t_max_s=round(t, 3), robot_phase=ctx['robot_phase'], human_phase=ctx['human_phase'])
        for r in out.values():
            why = []
            if r['category'] == 'invalid_contact':
                why.append('never allowed')
            elif r['category'] == 'human_self_collision':
                if r['max_depth_m'] > selfmax: why.append(f"depth {r['max_depth_m']:.4f} > {selfmax}")
            else:
                if r['max_depth_m'] > rigid: why.append(f"depth {r['max_depth_m']:.4f} > {rigid}")
                if r['pair'].startswith('human_seg_') and r['max_force_n'] > fmax: why.append(f"force {r['max_force_n']:.0f} N > {fmax:.0f}")
                if r['rule'] == 'robot_grasp' and self.gripper_force_max and r['max_force_n'] > self.gripper_force_max:
                    why.append(f"force {r['max_force_n']:.0f} N > gripper {self.gripper_force_max:.0f}")
            r['violation'] = '; '.join(why) or None
            r['max_depth_m'] = round(r['max_depth_m'], 5); r['max_force_n'] = round(r['max_force_n'], 2)
        return out

    # ---------------------------------------------------------------- summary
    def summary(self):
        r = self.robot
        # gates: joint limits, joint speed vs configs/robot.yaml max_joint_speed, grasp drops. Joint acceleration and
        # actuator force saturation are MEASURED only: no sourced acceptance threshold has been decided yet.
        robot_ok = (r['joint_limit_hits'] == 0 and r['max_joint_speed_ratio'] <= 1.0 + 1e-6 and r['grasp_drops'] == 0)
        cls = self.classified()
        bad = {k: v for k, v in cls.items() if v['violation']}
        self.violations = {k: v['max_depth_m'] for k, v in bad.items() if v['category'] != 'human_self_collision'}
        self.self_hits = {k: v['max_depth_m'] for k, v in bad.items() if v['category'] == 'human_self_collision'}
        by_cat = {c: sum(1 for v in bad.values() if v['category'] == c) for c in CATEGORIES}
        return {'contact_classes': cls, 'contact_failures_by_category': by_cat,
                'contact_violations': self.violations, 'human_self_violations': self.self_hits,
                'contact_measurements': self.contact_measurements,
                'visual_overlap_warnings': self.visual_overlap_warnings,
                'physics_contact_samples': self.physics_contact_samples,
                'human_contact_depth_tolerance_m': self.contact_depth_tolerance,
                'robot_human_penetrations_m': {k: round(-v, 4) for k, v in self.human_contacts.items()},
                'robot_human_closest_m': None if not np.isfinite(self.closest_human_m) else round(float(self.closest_human_m), 4),
                'robot_limits': {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()},
                'robot_limits_ok': bool(robot_ok),
                'physics_valid': bool(robot_ok and not self.violations and not self.self_hits)}

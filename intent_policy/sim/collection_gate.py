"""Configured collection-only rejection gate. No action steering or test filtering.

Requires exhaustive pairs from a scene-specific reviewed specification. The
reviewed flag is a record of external geometry review, never inferred here.
This component cannot certify an unimplemented scene from a storyboard.

Pair coverage: an explicit list, or `all_scene_pairs` = every robot geom x every human segment geom, every human
segment geom x every object geom, every robot geom x every object geom and every object x object geom pair (table
contacts of resting objects are not audited). Permitted contact is phase/state specific, never a blanket object
exemption:
  - `allowed_contacts`: geom pair + human/robot stages (as before);
  - `contact_rules` kind `robot_manipulation`: gripper geoms (bodies named sus2f_*) x the object the robot is
    manipulating, i.e. the sim reports it grasped, or released less than `release_grace_s` ago (the opening fingers
    leave the rim), or the expert is in one of `stages` with it as its target;
  - `contact_rules` kind `support`: two objects when the upper one stands within the lower one's footprint (a
    cup on its coaster); side contact or a push is not support;
  - `contact_rules` kind `human_manipulation`: the human's hand segment geoms x the object the human holds, or the
    object the current hand motion is aimed at during one of `stages` (fingertips arriving on it to take it), or any of
    `objects` during `stages` (the can lowered into the open palm);
  - `contact_rules` kind `handover`: the human's hand x the gripper during the human `stages` of taking the object
    the robot still holds (both hands on it);
  - `contact_rules` kind `container`: two objects inside the same open box (decor open_box) may rest against each
    other. `container_exempt`: objects in an open box are not movement-checked (they settle as others drop in).
Penetration of permitted contact stays bounded by the rule's `max_penetration_m` (else the pair's
max_contact_penetration_m).
Movement: `protected_objects` (list or `all`) must not move while nobody manipulates them (held by the human,
grasped / manipulated by the robot); after a manipulation ends the reference pose is re-anchored once the object
has settled (speed below `settle_speed_mps` for `settle_s`).
"""
from itertools import combinations, product
import numpy as np
from intent_policy.sim.demo_contact_audit import DemoContactAudit, PairRule


def scene_pairs(sc, clearance_m, max_contact_penetration_m):
    m = sc.env.model
    human = [g for g in range(m.ngeom) if m.geom(g).name.startswith('human_seg_')]
    objects = [g for k in sc.on_table for g in sc.object_geom_ids[k]]
    pairs = set(product(sc.robot_geoms, human)) | set(product(human, objects)) | set(product(sc.robot_geoms, objects))
    for a, b in combinations(sc.on_table, 2):
        pairs.update(product(sc.object_geom_ids[a], sc.object_geom_ids[b]))
    decor = [g for g in range(m.ngeom) if m.geom(g).name.startswith('decor_')]   # static trays / boxes: the human and
    pairs |= set(product(human, decor)) | set(product(sc.robot_geoms, decor))    #   the robot must not enter them
    return [dict(geom_a=int(a), geom_b=int(b), clearance_m=clearance_m,
                 max_contact_penetration_m=max_contact_penetration_m) for a, b in sorted(pairs)]


class CollectionGate:
    def __init__(self, scenario, agent, config):
        self.sc, self.agent, self.config = scenario, agent, config
        if config.get('scene_coverage_reviewed') is not True:
            raise ValueError('Collection requires reviewed scene geometry coverage')
        self.permissions = config.get('allowed_contacts', [])
        for rule in self.permissions:
            if not rule.get('human_stages') and not rule.get('robot_stages'):
                raise ValueError('Contact permission must specify a phase; no blanket object exemptions')
        self.rules = config.get('contact_rules', [])
        for rule in self.rules:
            if rule.get('kind') not in ('robot_manipulation', 'support', 'human_manipulation', 'handover', 'container'):
                raise ValueError(f"Unknown contact rule {rule.get('kind')!r}")
            if rule['kind'] in ('robot_manipulation', 'human_manipulation', 'handover') and not rule.get('stages'):
                raise ValueError(f"{rule['kind']} needs the stages of the manipulation")
            rule['_stages'] = frozenset(rule.get('stages', ()))
        m = scenario.env.model
        pairs = config['pairs']
        if pairs == 'all_scene_pairs':
            pairs = scene_pairs(scenario, float(config['pair_clearance_m']), float(config['pair_max_penetration_m']))
        self.owner = {}
        if self.rules or config.get('protected_objects') == 'all':
            for k, ids in scenario.object_geom_ids.items():
                for g in ids:
                    self.owner[int(g)] = k
        self.gripper = {g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name.startswith('sus2f_')}
        self.human_hand = {g for g in range(m.ngeom) if m.body(m.geom_bodyid[g]).name in ('human_seg_r_hand', 'human_seg_l_hand')}
        self.audit = DemoContactAudit(m, [PairRule(**r) for r in pairs], self.allowed)
        protected = config['protected_objects']
        self.protected = list(scenario.on_table) if protected == 'all' else list(protected)
        self.reference = {k: (scenario.pos(k).copy(), scenario.quat(k).copy()) for k in self.protected}
        self.displacement_limit = float(config['protected_translation_m'])
        self.rotation_limit = float(config['protected_rotation_rad'])
        self.settle_speed = float(config.get('settle_speed_mps', 0.0))
        self.settle_s = float(config.get('settle_s', 0.0))
        self.calm_since = {}
        if not np.isfinite([self.displacement_limit, self.rotation_limit]).all() or min(
                self.displacement_limit, self.rotation_limit) < 0:
            raise ValueError('Invalid protected object movement tolerance')
        self.manipulated = set()           # objects whose reference waits for them to settle after a manipulation
        self.release_grace_s = float(config.get('release_grace_s', 0.0))
        self._was_grasped, self._released_at = {}, {}
        self._human_held, self._human_released_at = None, {}
        self.movement_violation = None
        self.observe(scenario.env.model, scenario.env.data)
        scenario.env.physics_observers.append(self.observe)
        scenario.env.pre_physics_observers.append(self.observe)

    def _name(self, g):
        return self.sc.env.model.geom(g).name if isinstance(g, (int, np.integer)) else g

    def _expert_stage(self):
        return getattr(getattr(self.agent, 'expert', None), 'stage', None)

    def _robot_manipulates(self, key, stages=None):
        sc = self.sc
        if getattr(sc, 'grasp_state', {}).get(key):
            return True
        t_rel = self._released_at.get(key)
        if t_rel is not None and float(sc.env.data.time) - t_rel <= self.release_grace_s:
            return True                          # the opening fingers leave the rim just after the release
        target = sc.robot_target() if hasattr(sc, 'robot_target') else None
        return stages is not None and key == target and self._expert_stage() in stages

    def _supports(self, a, b):
        """a and b are objects: the upper one stands within the lower one's footprint."""
        sc = self.sc
        pa, pb = sc.pos(a), sc.pos(b)
        upper, lower = (a, b) if pa[2] >= pb[2] else (b, a)
        return float(np.linalg.norm(sc.pos(upper)[:2] - sc.pos(lower)[:2])) <= float(sc.radius[lower])

    def allowed(self, a, b, t):
        if not self.permissions and isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
            oa, ob = self.owner.get(int(a)), self.owner.get(int(b))
            for r in self.rules:
                if self._rule_allows(r, int(a), int(b), oa, ob):
                    return float(r['max_penetration_m']) if 'max_penetration_m' in r else True
            return False
        for r in self.permissions:
            if {self._name(a), self._name(b)} != {r['geom_a'], r['geom_b']} and {a, b} != {r['geom_a'], r['geom_b']}:
                continue
            human_ok = not r.get('human_stages') or self.sc.human.stage in r['human_stages']
            robot_ok = not r.get('robot_stages') or self._expert_stage() in r['robot_stages']
            if human_ok and robot_ok:
                return True
        ga, gb = int(self.sc.env.model.geom(a).id), int(self.sc.env.model.geom(b).id)
        oa, ob = self.owner.get(ga), self.owner.get(gb)
        for r in self.rules:
            if self._rule_allows(r, ga, gb, oa, ob):
                return float(r['max_penetration_m']) if 'max_penetration_m' in r else True
        return False

    def _rule_allows(self, r, ga, gb, oa, ob):
        if r['kind'] == 'robot_manipulation':
            return any(g in self.gripper and o is not None and self._robot_manipulates(o, r['_stages'])
                       for g, o in ((ga, ob), (gb, oa)))
        if r['kind'] == 'human_manipulation':
            h = self.sc.human
            objs = r.get('objects')            # with `objects`: those objects in `stages`, whatever the motion aims at
            t = float(self.sc.env.data.time)
            just = lambda o: self._human_released_at.get(o) is not None and t - self._human_released_at[o] <= self.release_grace_s
            return any(g in self.human_hand and o is not None and (getattr(h, 'holding', None) == o or just(o) or (
                h.stage in r['_stages'] and (o in objs if objs else getattr(h, 'motion_key', None) == o)))
                for g, o in ((ga, ob), (gb, oa)))
        if r['kind'] == 'handover':            # the human's hand and the gripper both on the object being handed over
            return self.sc.human.stage in r['_stages'] and (
                (ga in self.human_hand and gb in self.gripper) or (gb in self.human_hand and ga in self.gripper))
        if r['kind'] == 'container' and oa is not None and ob is not None and oa != ob:
            return self._same_container(oa, ob)      # objects dropped into one open box may rest against each other
        if r['kind'] == 'support' and oa is not None and ob is not None and oa != ob:
            objs = r.get('objects')
            if objs and not any({oa, ob} == set(p) for p in objs):
                return False
            return self._supports(oa, ob)
        return False

    def _containers(self):
        return [(np.asarray(d['xy'], float), np.asarray(d['half'], float), self.sc.table_z + float(d['height']))
                for d in self.sc.scene.get('decor', []) or [] if d.get('type') == 'open_box']

    def _in_container(self, key):
        p = self.sc.pos(key)
        for i, (xy, half, top) in enumerate(self._containers()):
            if np.all(np.abs(p[:2] - xy) <= half) and p[2] < top + 0.04:    # also one leaning on another inside
                return i
        return None

    def _same_container(self, a, b):
        i = self._in_container(a)
        return i is not None and i == self._in_container(b)

    def _free(self, key):
        """Somebody is moving the object on purpose: the human holds it or the robot manipulates it."""
        if getattr(self.sc.human, 'holding', None) == key:
            return True
        stages = next((r['_stages'] for r in self.rules if r['kind'] == 'robot_manipulation'), None)
        return self._robot_manipulates(key, stages)

    def observe(self, model, data):
        for k, g in getattr(self.sc, 'grasp_state', {}).items():
            if g:
                self._was_grasped[k] = True
                self._released_at[k] = None
            elif self._was_grasped.get(k):
                self._was_grasped[k] = False
                self._released_at[k] = float(data.time)
        held = getattr(self.sc.human, 'holding', None)
        if self._human_held is not None and held != self._human_held:
            self._human_released_at[self._human_held] = float(data.time)   # the hand lets go: it may still touch it
        self._human_held = held
        had = self.audit.first_violation is not None
        self.audit.observe(model, data)
        if not had and self.audit.first_violation is not None:
            h = self.sc.human
            self.audit.first_violation.update(human_stage=getattr(h, 'stage', None), robot_stage=self._expert_stage(),
                human_holding=getattr(h, 'holding', None), human_motion_key=getattr(h, 'motion_key', None),
                hand_shape=getattr(h, 'hand_shape', None))
        if self.audit.first_violation is not None and self.config.get('stop_on_rejection', True):
            fail = getattr(self.sc, 'fail', None)
            if fail is not None:
                fail('collection_geometry_rejected')
        if self.movement_violation is not None:
            return
        for key in self.protected:
            if self.rules or self.config.get('protected_objects') == 'all':
                if self.config.get('container_exempt') and self._in_container(key) is not None:
                    self.manipulated.add(key)            # settling among the others in its box; re-anchored when out
                    continue
                if self._free(key):
                    self.manipulated.add(key)
                    continue
                if key in self.manipulated:
                    # re-anchor once the released object has been at rest for settle_s (it may still drop / turn flat)
                    if self.sc.speed(key) >= self.settle_speed:
                        self.calm_since.pop(key, None)
                        continue
                    t0 = self.calm_since.setdefault(key, float(data.time))
                    if float(data.time) - t0 < self.settle_s:
                        continue
                    self.manipulated.discard(key)
                    self.calm_since.pop(key, None)
                    self.reference[key] = (self.sc.pos(key).copy(), self.sc.quat(key).copy())
            p, q = self.reference[key]
            pos, quat = self.sc.pos(key), self.sc.quat(key)
            translation = float(np.linalg.norm(pos - p))
            cosine = abs(float(np.dot(quat, q))) / max(float(np.linalg.norm(quat) * np.linalg.norm(q)), 1e-12)
            rotation = float(2 * np.arccos(np.clip(cosine, 0, 1)))
            if translation > self.displacement_limit or rotation > self.rotation_limit:
                self.movement_violation = dict(kind='protected_object_moved', object=key, t=float(data.time),
                    translation_m=translation, rotation_rad=rotation)
                if self.config.get('stop_on_rejection', True) and hasattr(self.sc, 'fail'):
                    self.sc.fail('collection_protected_object_moved')
                return

    def finish(self, events):
        self.sc.env.physics_observers.remove(self.observe)
        self.sc.env.pre_physics_observers.remove(self.observe)
        geometry = self.audit.report()
        missing = []
        required = set(self.config['require_hymotion_labels'])
        seen = set()
        for e in events:
            if e['event_type'] != 'human_motion_start':
                continue
            p = e['payload']
            if p.get('label') not in required:
                continue
            seen.add(p['label'])
            source = p.get('motion_source') or {}
            if source.get('kind') != 'hymotion' or not source.get('clip_id'):
                missing.append(dict(t=e['timestamp'], label=p['label'], source=source))
        for label in sorted(required - seen):
            missing.append(dict(label=label, reason='required_motion_not_observed'))
        passed = geometry['passed_configured_pairs'] and self.movement_violation is None and not missing
        return dict(passed=passed, geometry=geometry, movement_violation=self.movement_violation,
                    missing_hymotion=missing, scope='reviewed_configured_geometry_and_motion_sources')

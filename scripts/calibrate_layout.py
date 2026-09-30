"""Table layout calibration in simulation (scence_construct.md §3.3), replacing teleop / human trials.

  measure  R_eff: grasp a 4 cm cube on a 5 cm polar grid around the robot base with the restricted controller
           (valid = 3/3 successful grasps with +-1 cm jitter) and check IK reach 15 cm above the table (hand
           zones). r_h: palm positions the mannequin reaches comfortably (waist lean <= --comfort-lean-deg) on
           the table and 15 cm above it. Writes <out>/reach.json and a top-down map <out>/reach_map.png.

  check    Verify configs/layout.yaml against scence_construct.md §3.2 / §3.3: every robot location within R_eff,
           robot tests 5/5 (grasp a cube at S / P and a cup at U_cup, reach H at +10 and +20 cm, each object
           type 5/5 at a slot), human comfort (reach H, the cubes in U and the cup at U_cup, point at every P / S
           with waist lean <= comfort_lean_deg), slot rows / stagger (angle between pointing rays from the eye and
           the shoulder), spacing and tape overlap, shared-zone depth, and camera visibility of the face and the
           hands in every pose. Writes image_coords (u, v) of every zone for the fixed cameras into layout.yaml,
           <out>/layout_check.md, a top-down photo with a ruler, an overview photo and the three camera views.

  ./run.sh -m scripts.calibrate_layout measure --out outputs/layout
  ./run.sh -m scripts.calibrate_layout check --out outputs/layout
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from intent_policy.scenarios.config import _deep_merge
from intent_policy.sim.base_env import ManipulationEnv, ROOT
from intent_policy.sim.human_body import HumanBody
import itertools
import re
import mujoco
from intent_policy.experts.scripted_expert import R_SAFE
from intent_policy.sim.restricted_action import RestrictedAction as A, RestrictedActionConfig, RestrictedActionMapper
from intent_policy.sim.scene_builder import build_scene_xml
from intent_policy.utils import load_yaml, resolve

CUBE = {'shape': 'box', 'size': [0.02, 0.02, 0.02], 'color': 'red'}
CUP = {'shape': 'cup', 'size': [0.036, 0.085], 'color': 'white', 'grasp_dz': 0.0475,   # catalog cup, grasped by the
       'grasp_offset': [-0.034, 0.0]}                                                   # wall on the -X side of the rim
GRASP_DZ, TRAVEL_DZ, TOL = 0.012, 0.13, 0.0051      # TOL: half a 1 cm set-point step
UR3E_REACH = 0.50                     # spec reach [m]
AXES = [(A.MOVE_FORWARD, A.MOVE_BACKWARD), (A.MOVE_LEFT, A.MOVE_RIGHT), (A.MOVE_UP, A.MOVE_DOWN)]


def scene_config(objects: dict | None = None) -> dict:
    scene = load_yaml('configs/scene/common.yaml')
    return _deep_merge(scene, {'objects': objects or {'cube': CUBE}})


class GraspTester:
    def __init__(self, scene: dict, key: str = 'cube'):
        self.env = ManipulationEnv(ROOT / 'configs/robot.yaml', model_xml=build_scene_xml(scene))
        m = self.env.model
        obj = scene['objects'][key]
        self.body = m.body(key).id
        self.qadr = int(m.joint(f'{key}_free').qposadr[0])
        self.env.register_graspable(self.body, m.equality(f'robot_grasp_{key}').id)
        self.half = float(obj['size'][-1]) if obj['shape'] != 'cup' else float(obj['size'][1]) / 2
        self.grasp_dz = float(obj.get('grasp_dz', GRASP_DZ))
        self.grasp_offset = np.asarray(obj.get('grasp_offset', [0.0, 0.0]), float)
        self.scene = scene
        cfg = RestrictedActionConfig.load()
        cfg.limits = {'x': [-0.4, 0.9], 'y': [-0.7, 0.7], 'z': [0.63, 1.2]}   # the layout decides the real box
        self.cfg = cfg
        self.top = float(scene['table_top_z'])

    def _goto(self, mapper, target, max_steps=400) -> bool:
        """Axis-wise moves; a move across the band |y| < R_SAFE first goes out to x >= R_SAFE (around the robot base,
        as the scripted expert does), then across, then inwards."""
        sp, tg = mapper.setpoint, np.asarray(target, float)
        if abs(tg[1] - sp[1]) > TOL and min(sp[1], tg[1]) < R_SAFE and max(sp[1], tg[1]) > -R_SAFE and sp[0] < R_SAFE:
            out = [R_SAFE, sp[1], max(sp[2], tg[2])]
            return self._line(mapper, out, max_steps) and self._line(mapper, [R_SAFE, tg[1], out[2]], max_steps) \
                and self._line(mapper, tg, max_steps)
        return self._line(mapper, tg, max_steps)

    def _line(self, mapper, target, max_steps=400) -> bool:
        for _ in range(max_steps):
            err = np.asarray(target) - mapper.setpoint
            over = np.abs(err) > TOL
            if not over.any():
                return True
            axis = int(np.argmax(np.abs(err) - TOL))
            cmd = mapper.map(AXES[axis][0 if err[axis] > 0 else 1])
            self.env.step(cmd.joint_target)
            if cmd.rejected:
                return False
        return False

    def _grip(self, mapper, action, ticks):
        for _ in range(ticks):
            self.env.step(mapper.map(action).joint_target)

    def grasp(self, xy) -> bool:
        env = self.env
        env.reset()
        env.data.qpos[self.qadr:self.qadr + 7] = [xy[0], xy[1], self.top + self.half + 0.001, 1, 0, 0, 0]
        env.settle(10)
        mapper = RestrictedActionMapper(env, self.cfg)
        grasp_z = self.top + self.half + self.grasp_dz
        gx, gy = np.asarray(xy, float) + self.grasp_offset
        if not self._goto(mapper, [gx, gy, self.top + TRAVEL_DZ]) or not self._goto(mapper, [gx, gy, grasp_z]):
            return False
        for _ in range(15):
            env.step(mapper.map(A.HOLD).joint_target)
        for _ in range(30):
            env.step(mapper.map(A.CLOSE_GRIPPER).joint_target)
            if env.is_grasped(self.body):
                break
        self._grip(mapper, A.HOLD, 3)
        # lift to travel height, and at least 8 cm (a cup grasped by the rim sits only 4 cm below travel height)
        self._goto(mapper, [mapper.setpoint[0], mapper.setpoint[1], max(self.top + TRAVEL_DZ, grasp_z + 0.08)])
        return bool(env.is_grasped(self.body) and env.data.qpos[self.qadr + 2] > self.top + self.half + 0.05)

    def reach(self, p) -> bool:
        """Drive the empty gripper to p with the restricted controller (hand zones)."""
        self.env.reset()
        mapper = RestrictedActionMapper(self.env, self.cfg)
        ok = self._goto(mapper, [p[0], p[1], max(p[2], mapper.setpoint[2])]) and self._goto(mapper, p)
        for _ in range(10):
            self.env.step(mapper.map(A.HOLD).joint_target)
        return bool(ok and np.linalg.norm(self.env.observe().ee_pos - np.asarray(p)) < 0.012)

    def ik_ok(self, p) -> bool:
        self.env.reset()
        self.env.solve_ik(np.asarray(p, float))
        return self.env.last_ik_error < self.cfg.max_ik_error


def table_rect(scene_xml_path) -> tuple[float, float, float, float]:
    import xml.etree.ElementTree as ET
    t = ET.parse(scene_xml_path).getroot().find(".//geom[@name='table']")
    pos, size = np.fromstring(t.get('pos'), sep=' '), np.fromstring(t.get('size'), sep=' ')
    return pos[0] - size[0], pos[0] + size[0], pos[1] - size[1], pos[1] + size[1]


def measure(args):
    scene = scene_config()
    x0, x1, y0, y1 = table_rect(ROOT / scene['base_model'])
    top = float(scene['table_top_z'])
    tester = GraspTester(scene)
    rng = np.random.default_rng(0)
    robot = []
    for r in np.arange(0.15, 0.56, 0.05):
        for ang in np.arange(-165, 166, 15):
            xy = np.array([r * np.cos(np.radians(ang)), r * np.sin(np.radians(ang))])
            if not (x0 + 0.03 < xy[0] < x1 - 0.03 and y0 + 0.03 < xy[1] < y1 - 0.03):
                continue
            ik_table = tester.ik_ok([*xy, top + 0.02 + GRASP_DZ])
            ik_hand = tester.ik_ok([*xy, top + 0.15])
            wins = sum(tester.grasp(xy + rng.uniform(-0.01, 0.01, 2)) for _ in range(3)) if ik_table else 0
            robot.append(dict(xy=xy.round(4).tolist(), r=round(float(r), 3), angle=int(ang), ik_table=ik_table,
                              ik_hand=ik_hand, grasps=wins, valid=bool(wins == 3 and ik_hand)))
            print(f"robot r={r:.2f} ang={ang:+4d} ik_table={ik_table} ik_hand={ik_hand} grasps={wins}/3", flush=True)

    human_cfg = {**scene['human'], 'floor_z': scene.get('floor_z', 0.0)}
    body = HumanBody(human_cfg)
    comfort = np.radians(args.comfort_lean_deg)
    human = []
    for x in np.arange(x0 + 0.025, x1, 0.05):
        for y in np.arange(y0 + 0.025, y1, 0.05):
            row = dict(xy=[round(float(x), 4), round(float(y), 4)])
            for name, dz in (('table', 0.03), ('hand', 0.15)):
                pose = body.pose([x, y, top + dz])
                row[f'{name}_comfort'] = bool(pose['reach_error'] < 1e-6 and pose['lean'] <= comfort + 1e-9)
                row[f'{name}_reach'] = bool(pose['reach_error'] < 1e-6)
            human.append(row)

    valid = [p for p in robot if p['valid']]
    r_eff_measured = max((p['r'] for p in valid), default=0.0)
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = dict(table=dict(x=[x0, x1], y=[y0, y1], top_z=top), ur3e_spec_reach=UR3E_REACH,
                  r_eff_spec_band=[0.7 * UR3E_REACH, 0.8 * UR3E_REACH], r_eff_measured_max_valid=r_eff_measured,
                  comfort_lean_deg=args.comfort_lean_deg, human=dict(stand_xy=scene['human']['stand_xy'],
                                                                      facing_deg=scene['human'].get('facing_deg', 180.0)),
                  robot_grid=robot, human_grid=human)
    (out / 'reach.json').write_text(json.dumps(report, indent=1))
    draw_map(report, out / 'reach_map.png')
    print(f"valid robot points {len(valid)}/{len(robot)}; max valid radius {r_eff_measured:.2f} m; "
          f"human comfortable table points {sum(h['table_comfort'] for h in human)}; wrote {out}")


def pointing_pose(body: HumanBody, target) -> np.ndarray:
    """Palm position for pointing at a table location (as the scripted human points)."""
    return body.pointing_palm(target, 0.50)


def project(model, data, cam: str, p, width: int, height: int) -> tuple[float, float, float]:
    """Pixel (u, v) and depth of world point p in camera `cam` (u right, v down)."""
    cid = model.camera(cam).id
    pos, R = data.cam_xpos[cid], data.cam_xmat[cid].reshape(3, 3)
    c = R.T @ (np.asarray(p, float) - pos)                        # camera frame: x right, y up, looks along -z
    f = height / 2 / np.tan(np.radians(model.cam_fovy[cid]) / 2)
    return width / 2 + f * c[0] / -c[2], height / 2 - f * c[1] / -c[2], -c[2]


def visible(model, data, cam: str, p, human_geoms: set) -> bool:
    """First geometry on the ray camera -> p is the human (or nothing before p): p is not occluded."""
    cid = model.camera(cam).id
    origin = data.cam_xpos[cid].copy()
    vec = np.asarray(p, float) - origin
    dist = np.linalg.norm(vec)
    gid = np.zeros(1, np.int32)
    hit = mujoco.mj_ray(model, data, origin, vec / dist, None, 1, -1, gid)
    return hit < 0 or hit >= dist - 0.03 or int(gid[0]) in human_geoms


def check(args):
    layout = load_yaml('configs/layout.yaml')
    scene = scene_config({'cube': CUBE, 'cup': CUP})
    top, r_eff, comfort = float(scene['table_top_z']), float(layout['r_eff']), np.radians(layout['comfort_lean_deg'])
    zones = layout['zones']
    xy = {k: np.asarray(z['xy'], float) for k, z in zones.items()}
    kinds = {k: z['kind'] for k, z in zones.items()}
    S = [k for k in zones if kinds[k] == 'slot']
    P = [k for k in zones if kinds[k] == 'place']
    Hz = [k for k in zones if kinds[k] == 'hand']
    shared = [k for k in zones if kinds[k] in ('place', 'hand', 'support')]
    U = [k for k in zones if kinds[k] == 'human_store']
    blocks = [np.asarray(b, float) for k in U for b in zones[k].get('blocks', [zones[k]['xy']])]
    taped = [k for k in zones if kinds[k] != 'slot' and zones[k].get('marker', True)]
    robot_locs = S + shared
    results, lines = {}, ['# Layout check (scence_construct.md §3.2 / §3.3)', '']

    def rule(name, ok, detail):
        results[name] = dict(ok=bool(ok), detail=detail)
        lines.append(f"- [{'x' if ok else ' '}] **{name}**: {detail}")

    radius = {k: float(np.linalg.norm(xy[k])) for k in robot_locs}
    rule('1 robot locations within R_eff', max(radius.values()) <= r_eff + 1e-9,
         f"max radius {max(radius.values()):.3f} m ({max(radius, key=radius.get)}) <= R_eff {r_eff} m")

    body = HumanBody({**scene['human'], 'floor_z': scene.get('floor_z', 0.0)})
    reach_lean = lambda p: np.degrees(body.pose(p)['lean']) + 1e3 * body.pose(p)['reach_error']
    point_lean = lambda k: np.degrees(body.pose(pointing_pose(body, [*xy[k], top]), [*xy[k], top])['lean'])
    lean = {}
    for k in Hz:
        lean[k] = max(reach_lean([*xy[k], top + h]) for h in (0.10, 0.20))
    for k in [k for k in shared if kinds[k] == 'support']:
        lean[k] = reach_lean([*xy[k], top + 0.12])               # the cube dropped into the cup standing there
    for i, b in enumerate(blocks):
        lean[f'U cube {i + 1}'] = reach_lean([*b, top + 0.03])
    for k in P + S:                                              # the human only points at P and S
        lean[f'point {k}'] = point_lean(k)
    worst = max(lean, key=lean.get)
    rule('2 human reaches H / U (cubes, cup) and points at every P / S comfortably', lean[worst] <= np.degrees(comfort) + 1e-6,
         f"max waist lean {lean[worst]:.1f} deg ({worst}) <= {layout['comfort_lean_deg']} deg; touching P (not needed: "
         f"the human only points at P) would need up to {max(reach_lean([*xy[k], top + 0.03]) for k in P):.0f} deg")
    # §3.2 zone table: S only the robot reaches (the human cannot take the object comfortably), the cubes in U only
    # the human; U_cup (support) is shared
    s_reach = {k: body.pose([*xy[k], top + 0.03]) for k in S}
    s_comfy = [k for k, p in s_reach.items() if p['reach_error'] < 1e-6 and p['lean'] <= comfort + 1e-9]
    u_robot = [i + 1 for i, b in enumerate(blocks) if np.linalg.norm(b) <= r_eff]
    rule('zones by reach: S robot only, cubes in U human only', not s_comfy and not u_robot,
         f"S within the human's comfortable reach: {s_comfy or 'none'} (min lean "
         f"{min(np.degrees(p['lean']) + 1e3 * p['reach_error'] for p in s_reach.values()):.0f} deg); "
         f"U cube radii {[round(float(np.linalg.norm(b)), 2) for b in blocks]} m > R_eff")

    eye = body.pose(body.palm_T[body.active])['face']
    shoulder = body.J[f'{body.active.upper()}_Shoulder']
    ang = lambda o, a, b: np.degrees(np.arccos(np.clip(np.dot(unit3(a - o), unit3(b - o)), -1, 1)))
    rays = min(min(ang(eye, np.r_[xy[a], top], np.r_[xy[b], top]), ang(shoulder, np.r_[xy[a], top], np.r_[xy[b], top]))
               for a, b in itertools.combinations(S, 2))
    rows = sorted({round(float(xy[k][1]), 3) for k in S})
    row_x = [sorted(float(xy[k][0]) for k in S if round(float(xy[k][1]), 3) == y) for y in rows]
    shift = min(abs(a - b) for a in row_x[0] for b in row_x[1]) if len(rows) == 2 else 0.0   # stagger between rows
    straight = len(rows) == 2 and all(len(r) == 3 for r in row_x)
    min_ray = float(layout.get('min_ray_angle_deg', 6.0))
    rule('3 slots in two straight rows, staggered (no shared line of sight / pointing ray)',
         straight and shift > 0.01 and rays >= min_ray,
         f"rows y = {rows} m, x = {[[round(x, 3) for x in r] for r in row_x]} (shift {shift:.3f} m); min angle between "
         f"rays from the eye / shoulder {rays:.1f} deg (>= {min_ray})")

    dist = lambda a, b: float(np.linalg.norm(xy[a] - xy[b]))
    half = lambda k: np.broadcast_to(zones[k].get('size', layout['marker_size']), 2) / 2
    overlap = [f'{a}/{b}' for a, b in itertools.combinations(taped, 2)
               if np.all(np.abs(xy[a] - xy[b]) < half(a) + half(b) - 1e-9)]
    ss = min(dist(a, b) for a, b in itertools.combinations(S, 2))
    sh = min(dist(a, b) for a, b in itertools.combinations(shared, 2))
    cross = min(dist(a, b) for a in S for b in shared)
    rule('4 spacing', ss >= 0.12 - 1e-9 and sh >= 0.10 - 1e-9 and cross >= 0.12 - 1e-9 and not overlap,
         f"slot-slot {ss:.3f} m (>= 0.12), shared-shared {sh:.3f} m (>= 0.10), slot-shared {cross:.3f} m (>= 0.12), "
         f"tape overlap: {overlap or 'none'}")
    ext = [(xy[k][1] - half(k)[1], xy[k][1] + half(k)[1]) for k in taped]
    depth = max(e[1] for e in ext) - min(e[0] for e in ext)
    rule('5 shared zone depth >= 0.15-0.20 m', depth >= 0.15, f"{depth:.2f} m (U to P); N_P = {len(P)}, N_H = {len(Hz)}")
    rule('6 zone size ~12 x 12 cm', abs(layout['marker_size'] - 0.12) < 1e-9,
         f"P / H tape squares {layout['marker_size']} m; U {[round(float(v), 2) for v in half(U[0]) * 2]} m (cubes + the cup)")

    # robot tests (5/5) ---------------------------------------------------------------------------------------
    rng = np.random.default_rng(1)
    cube, cup = GraspTester(scene, 'cube'), GraspTester(scene, 'cup')
    robot = {}
    for k in S + P:
        robot[k] = sum(cube.grasp(xy[k] + rng.uniform(-0.005, 0.005, 2)) for _ in range(5))
    for k in [k for k in shared if kinds[k] == 'support']:           # the robot stands the cup there (rim grasp)
        robot[k] = sum(cup.grasp(xy[k] + rng.uniform(-0.005, 0.005, 2)) for _ in range(5))
    for k in [k for k in shared if kinds[k] == 'hand']:
        robot[k] = sum(cube.reach([*(xy[k] + rng.uniform(-0.005, 0.005, 2)), top + h]) for h in (0.10, 0.20, 0.15, 0.10, 0.20))
    objects = {'cube 4 cm': sum(cube.grasp(xy[S[0]] + rng.uniform(-0.005, 0.005, 2)) for _ in range(5))}
    objects.update({f'cup at {k}': sum(cup.grasp(xy[k] + rng.uniform(-0.005, 0.005, 2)) for _ in range(5)) for k in S})
    bad = {k: v for k, v in robot.items() if v < 5}
    rule('robot test 5/5 at every location (§3.3 step 7)', not bad, f"{robot}" + (f" FAILED {bad}" if bad else ''))
    rule('gripper grasps every object type 5/5 (§3.1)', all(v == 5 for v in objects.values()), f"{objects}")

    # cameras: image coordinates and face / hand visibility --------------------------------------------------
    env = cube.env
    env.reset()
    body.bind(env.model)
    human_geoms = {g for g in range(env.model.ngeom) if env.model.geom(g).name.startswith('human_seg_')}
    W, H = 640, 480
    image_coords = {}
    for cam in ('high', 'top'):
        mujoco.mj_forward(env.model, env.data)
        image_coords[cam] = {k: [round(float(c), 1) for c in project(env.model, env.data, cam, [*xy[k], top + zones[k].get('height', 0.0)], W, H)[:2]]
                             for k in zones}
    outside = [f'{cam}:{k}' for cam, coords in image_coords.items() for k, (u, v) in coords.items()
               if not (0.02 * W < u < 0.98 * W and 0.02 * H < v < 0.98 * H)]
    rule('every zone inside the high and top images (§3.3 step 6)', not outside,
         f"image_coords of {len(zones)} zones written to layout.yaml" + (f"; OUTSIDE {outside}" if outside else ''))
    poses = {f'reach {k}': ([*xy[k], top + (0.15 if kinds[k] == 'hand' else 0.12)], None, 'relaxed')
             for k in Hz + [k for k in shared if kinds[k] == 'support']}
    poses.update({f'reach U cube {i + 1}': ([*b, top + 0.03], None, 'relaxed') for i, b in enumerate(blocks)})
    poses.update({f'point {k}': (pointing_pose(body, [*xy[k], top]), [*xy[k], top], 'point') for k in P + S})
    vis_fail = []
    for name, (palm, aim, shape) in poses.items():
        pose = body.apply(env.data, np.asarray(palm, float), aim, shape)
        mujoco.mj_forward(env.model, env.data)
        for part, p in (('face', pose['face']), ('hand', pose['palm'])):
            u, v, z = project(env.model, env.data, 'high', p, W, H)
            inside = z > 0 and 0.03 * W < u < 0.97 * W and 0.03 * H < v < 0.97 * H
            if not (inside and visible(env.model, env.data, 'high', p, human_geoms)):
                vis_fail.append(f'{name}: {part}')
    rule('7 overview camera sees the face and the hands in every pose', not vis_fail,
         f"{len(poses)} poses (reach H / U, point at P / S), robot at home" + (f"; FAILED {vis_fail}" if vis_fail else ''))

    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    text = (ROOT / 'configs/layout.yaml').read_text()
    block = 'image_coords:                       # (u, v) px in 640x480 renders, filled by `calibrate_layout check`\n' + ''.join(
        f"  {cam}: {{{', '.join(f'{k}: {uv}' for k, uv in coords.items())}}}\n" for cam, coords in image_coords.items())
    text = re.sub(r'image_coords:.*\Z', block, text, flags=re.S)
    (ROOT / 'configs/layout.yaml').write_text(text)
    photos(env, body, layout, top, image_coords, out)
    ok = all(r['ok'] for r in results.values())
    lines += ['', f"**Done criterion (§3.3): {'MET' if ok else 'NOT MET'}**", '',
              f"Photos: `{out / 'layout_top.png'}` (top-down with a 10 cm ruler), `{out / 'layout_overview.png'}`, "
              f"`{out / 'cameras.png'}` (high / top / wrist)."]
    (out / 'layout_check.md').write_text('\n'.join(lines) + '\n')
    (out / 'layout_check.json').write_text(json.dumps(dict(results=results, lean=lean, robot=robot, objects=objects), indent=1))
    print('\n'.join(lines))


def unit3(v):
    return np.asarray(v, float) / np.linalg.norm(v)


def photos(env, body, layout, top, image_coords, out: Path) -> None:
    """Top-down photo with a 10 cm ruler and labels, the overview camera photo and the three camera views side by
    side (human at rest)."""
    body.apply(env.data, np.asarray(load_yaml('configs/scene/common.yaml')['human']['hand_rest'], float))
    for j in range(env.model.njnt):                             # test objects out of the picture
        if env.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            env.data.qpos[env.model.jnt_qposadr[j]:env.model.jnt_qposadr[j] + 3] = [0.0, 0.0, -5.0 - j]
    mujoco.mj_forward(env.model, env.data)
    for cam, name in (('top', 'layout_top.png'), ('high', 'layout_overview.png')):
        img = Image.fromarray(env.render(cam, 640, 480))
        d = ImageDraw.Draw(img)
        for k, (u, v) in image_coords[cam].items():
            d.text((u + 7, v - 7), k, fill=(0, 0, 0) if cam == 'top' else (255, 255, 0))
        if cam == 'top':                                        # ruler along the robot-side table edge (x = 0)
            x0, x1 = layout['table']['y']
            pts = [project(env.model, env.data, 'top', [0.0, y, top], 640, 480)[:2] for y in np.arange(-0.3, 0.41, 0.1)]
            d.line([pts[0], pts[-1]], fill=(200, 0, 0), width=2)
            for i, (u, v) in enumerate(pts):
                d.line([(u, v - 5), (u, v + 5)], fill=(200, 0, 0), width=2)
                d.text((u - 8, v + 7), f'{-30 + 10 * i}', fill=(200, 0, 0))
            d.text((6, 6), 'ruler: y [cm] along x = 0; S slots (crosses), P blue, H orange, U grey (cup: left part)', fill=(0, 0, 0))
        img.save(out / name)
    sheet = Image.new('RGB', (3 * 640, 480))                    # the three policy cameras, robot at rest
    for i, cam in enumerate(('high', 'top', 'wrist')):
        sheet.paste(Image.fromarray(env.render(cam, 640, 480)), (640 * i, 0))
        ImageDraw.Draw(sheet).text((640 * i + 6, 6), cam, fill=(255, 255, 0))
    sheet.save(out / 'cameras.png')


def draw_map(rep: dict, path: Path, px_per_m: int = 500) -> Image.Image:
    (x0, x1), (y0, y1) = rep['table']['x'], rep['table']['y']
    W, H = int((y1 - y0) * px_per_m) + 80, int((x1 - x0) * px_per_m) + 80
    img = Image.new('RGB', (W, H), (245, 240, 228))
    d = ImageDraw.Draw(img)
    # top-down view seen from above with +X up the image and +Y to the left (robot at the bottom)
    to_px = lambda x, y: (40 + (y1 - y) * px_per_m, 40 + (x1 - x) * px_per_m)
    d.rectangle([*to_px(x1, y1), *to_px(x0, y0)], outline=(90, 80, 60), width=2)
    for h in rep['human_grid']:
        cx, cy = to_px(*h['xy'])
        if h['table_comfort']:
            d.rectangle([cx - 12, cy - 12, cx + 12, cy + 12], fill=(170, 200, 250))
    for p in rep['robot_grid']:
        cx, cy = to_px(*p['xy'])
        col = (30, 150, 60) if p['valid'] else (230, 140, 30) if p['grasps'] or p['ik_table'] else (200, 40, 40)
        d.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=col)
    bx, by = to_px(0.0, 0.0)
    for r in (0.35, 0.40, 0.50):
        d.ellipse([bx - r * px_per_m, by - r * px_per_m, bx + r * px_per_m, by + r * px_per_m], outline=(120, 120, 120))
    d.text((bx + 4, by + 4), 'robot base', fill=(0, 0, 0))
    hx, hy = to_px(*rep['human']['stand_xy'])
    d.ellipse([hx - 10, hy - 10, hx + 10, hy + 10], fill=(160, 110, 70))
    d.text((hx + 12, hy - 6), 'human', fill=(0, 0, 0))
    for m in range(0, int((x1 - x0) * 10) + 1):                 # 10 cm ruler ticks along the table edges
        tx, ty = to_px(x0 + m * 0.1, y1)
        d.line([tx - 6, ty, tx, ty], fill=(0, 0, 0))
    d.text((6, 6), 'green: robot grasp 3/3 + IK at +15 cm | orange: partial | red: no | blue: human comfortable '
                   '| rings: 0.35/0.40/0.50 m | ticks 10 cm', fill=(0, 0, 0))
    img.save(path)
    return img


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    m = sub.add_parser('measure')
    m.add_argument('--out', default='outputs/layout')
    m.add_argument('--comfort-lean-deg', type=float, default=30.0)
    c = sub.add_parser('check')
    c.add_argument('--out', default='outputs/layout')
    args = p.parse_args()
    {'measure': measure, 'check': check}[args.cmd](args)


if __name__ == '__main__':
    main()

"""Render the kinematic storyboard of the proposed T1-T5 v6 scenarios (configs/scenarios/v6_preview.yaml).

Reuses the scene builder, the mannequin (HumanBody), the robot IK and the video writer of the real pipeline; only
the hand-written timelines are new. No physics, no expert, no intent model: a preview for approving the scripts.
  python scripts/preview_scenarios_v6.py [--tasks T1 T3]   -> outputs/scenario_v6_preview/<date_time>/<task>.mp4
"""
import argparse
import copy
import datetime
import json
import shutil
import xml.etree.ElementTree as ET

import av
import mujoco
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from intent_policy.sim.base_env import ManipulationEnv, ROOT
from intent_policy.sim.human_body import HumanBody
from intent_policy.sim.scene_builder import build_scene_xml, _sub
from intent_policy.sim.scripted_human import min_jerk

SPECIAL_SHAPES = ('mesh', 'frame', 'rack', 'roller', 'carton', 'tray', 'drawer', 'donut', 'pcb', 'can', 'flask')


def load_scene(cfg: dict, task: dict) -> dict:
    base = yaml.safe_load((ROOT / cfg['scene_base']).read_text())
    scene = copy.deepcopy(base)
    scene['cameras'] = {**cfg['cameras'], **task.get('cameras', {})}
    scene['objects'] = {k: {**{kk: vv for kk, vv in o.items() if kk != 'pos'}, 'color': cfg['colors'][o['color']]}
                        for k, o in task['objects'].items() if o['shape'] not in SPECIAL_SHAPES}   # build_xml
    return scene


def add_mesh(root, parent, name: str, spec: dict) -> None:
    """A textured visual mesh (assets/objects/gso/<mesh>/, base on z = 0) as a geom of `parent`: `scale`, `yaw` [deg]."""
    asset, folder = root.find('asset'), ROOT / 'assets/objects/gso' / spec['mesh']
    s, yaw = float(spec.get('scale', 1.0)), float(spec.get('yaw', 0.0))
    _sub(asset, 'texture', name=f'{name}_tex', type='2d', file=str(folder / 'texture.png'))
    _sub(asset, 'material', name=f'{name}_mat', texture=f'{name}_tex', specular='0.3', shininess='0.3')
    _sub(asset, 'mesh', name=f'{name}_mesh', file=str(folder / 'model.obj'), scale=[s, s, s])
    _sub(parent, 'geom', name=f'{name}_geom', type='mesh', mesh=f'{name}_mesh', material=f'{name}_mat',
         euler=[0, 0, yaw], contype='0', conaffinity='0', **({'pos': spec['pos']} if parent.tag == 'worldbody' else {}))


def build_xml(cfg: dict, scene: dict, task: dict) -> str:
    root = ET.fromstring(build_scene_xml(scene))
    world = root.find('worldbody')
    for g in [g for g in world.findall('geom') if g.get('name', '').startswith('marker_')]:
        world.remove(g)                                        # v6 has no S/P/H/U tape
    for k, o in task['objects'].items():                       # mesh / frame objects: free bodies posed like the others
        if o['shape'] in SPECIAL_SHAPES:
            body = _sub(world, 'body', name=k, pos=o['pos'])
            _sub(body, 'freejoint', name=f'{k}_free')
            if o['shape'] == 'mesh':
                add_mesh(root, body, k, o)
                continue
            if o['shape'] == 'rack':
                # A lightweight dish rack; all parts follow the same rigid body in the storyboard.
                hx, hy = o.get('half', [0.085, 0.055])
                col = [0.55, 0.62, 0.66, 1]
                for x in np.linspace(-hx, hx, 6):
                    _sub(body, 'geom', type='capsule', fromto=[x, -hy, 0.008, x, hy, 0.008], size=[0.0025], rgba=col)
                for y in (-hy, hy):
                    _sub(body, 'geom', type='capsule', fromto=[-hx, y, 0.01, hx, y, 0.01], size=[0.003], rgba=col)
                    _sub(body, 'geom', type='capsule', fromto=[-hx, y, 0.055, hx, y, 0.055], size=[0.003], rgba=col)
                    for x in (-hx, hx):
                        _sub(body, 'geom', type='capsule', fromto=[x, y, 0.01, x, y, 0.055], size=[0.003], rgba=col)
                continue
            if o['shape'] == 'roller':
                _sub(body, 'geom', type='capsule', fromto=[-0.055, 0, 0.015, 0.015, 0, 0.015], size=[0.006], rgba=[0.12, 0.17, 0.25, 1])
                _sub(body, 'geom', type='cylinder', pos=[0.035, 0, 0.015], euler=[0, 90, 0], size=[0.018, 0.025], rgba=[0.93, 0.93, 0.87, 1])
                continue
            if o['shape'] in ('carton', 'tray', 'drawer'):
                hx, hy = o.get('half', [0.055, 0.045])
                h, w = float(o.get('height', 0.026)), float(o.get('wall', 0.002))
                col = cfg['colors'].get(o.get('color'), [0.86, 0.75, 0.58, 1])
                for size, pos in [([hx, hy, w], [0, 0, w]), ([hx, w, h / 2], [0, hy, h / 2]),
                                  ([hx, w, h / 2], [0, -hy, h / 2]), ([w, hy, h / 2], [hx, 0, h / 2]),
                                  ([w, hy, h / 2], [-hx, 0, h / 2])]:
                    _sub(body, 'geom', type='box', size=size, pos=pos, rgba=col)
                if o['shape'] == 'drawer':
                    _sub(body, 'geom', type='box', size=[hx * 0.40, 0.009, 0.006],
                         pos=[0, -hy - 0.014, h * 0.72], rgba=[0.28, 0.30, 0.33, 1])
                continue
            if o['shape'] == 'donut':
                for a in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                    _sub(body, 'geom', type='sphere', size=[0.0105], pos=[0.018 * np.cos(a), 0.018 * np.sin(a), 0.012], rgba=[0.66, 0.34, 0.12, 1])
                    _sub(body, 'geom', type='sphere', size=[0.007], pos=[0.018 * np.cos(a), 0.018 * np.sin(a), 0.019], rgba=[0.88, 0.51, 0.35, 1])
                continue
            if o['shape'] == 'pcb':
                _sub(body, 'geom', type='box', size=[0.055, 0.038, 0.002], rgba=[0.08, 0.37, 0.22, 1])
                for x, y, sx, sy in [(-0.025, 0, 0.012, 0.010), (0.015, 0.012, 0.010, 0.006), (0.025, -0.014, 0.014, 0.005)]:
                    _sub(body, 'geom', type='box', pos=[x, y, 0.004], size=[sx, sy, 0.002], rgba=[0.12, 0.13, 0.14, 1])
                for y in np.linspace(-0.027, 0.027, 7):
                    _sub(body, 'geom', type='box', pos=[0.045, y, 0.003], size=[0.006, 0.001, 0.001], rgba=[0.85, 0.70, 0.25, 1])
                continue
            if o['shape'] == 'can':
                col = cfg['colors'][o.get('color', 'blue')]
                _sub(body, 'geom', type='cylinder', size=[0.022, 0.035], rgba=col)
                for z in (-0.035, 0.035):
                    _sub(body, 'geom', type='cylinder', pos=[0, 0, z], size=[0.022, 0.001], rgba=[0.75, 0.77, 0.79, 1])
                continue
            if o['shape'] == 'flask':
                _sub(body, 'geom', type='cylinder', size=[0.027, 0.036], rgba=[0.48, 0.75, 0.80, 0.9])
                _sub(body, 'geom', type='cylinder', pos=[0, 0, 0.041], size=[0.014, 0.009], rgba=[0.58, 0.80, 0.82, 1])
                _sub(body, 'geom', type='cylinder', pos=[0, 0, 0.052], size=[0.015, 0.004], rgba=[0.18, 0.23, 0.30, 1])
                continue
            (hx, hy, hz), w = o['half'], 0.003                 # box + a coloured plate per side + a socket on top
            _sub(body, 'geom', name=f'{k}_box', type='box', size=[hx, hy, hz], rgba=cfg['colors'][o['color']],
                 contype='0', conaffinity='0')
            for i, (c, (nx, ny)) in enumerate(zip(o['faces'], [(0, 1), (-1, 0), (0, -1), (1, 0)])):
                size = [w, hy * 0.8, hz * 0.8] if nx else [hx * 0.8, w, hz * 0.8]
                _sub(body, 'geom', name=f'{k}_face{i}', type='box', size=size, pos=[nx * (hx + w), ny * (hy + w), 0],
                     rgba=cfg['colors'][c], contype='0', conaffinity='0')
                _sub(body, 'geom', name=f'{k}_socket{i}', type='cylinder', size=[0.011, 0.002],
                     pos=[nx * 0.7 * hx, ny * 0.7 * hy, hz + 0.002], rgba=[0.1, 0.1, 0.1, 1], contype='0', conaffinity='0')
    for d in task.get('decor', []):
        if d['type'] == 'mesh':
            add_mesh(root, world, d['name'], d)
            continue
        rgba = cfg['colors'][d['color']]
        if d['type'] == 'open_box':                            # bottom + 4 walls, open top
            (hx, hy), h, w, (x, y, z) = d['half'], d['height'], d['wall'], d['pos']
            parts = [([hx, hy, w / 2], [x, y, z + w / 2]),
                     ([hx, w / 2, h / 2], [x, y + hy - w / 2, z + h / 2]), ([hx, w / 2, h / 2], [x, y - hy + w / 2, z + h / 2]),
                     ([w / 2, hy, h / 2], [x + hx - w / 2, y, z + h / 2]), ([w / 2, hy, h / 2], [x - hx + w / 2, y, z + h / 2])]
            for i, (size, pos) in enumerate(parts):
                _sub(world, 'geom', name=f"{d['name']}_{i}", type='box', size=size, pos=pos, rgba=rgba,
                     contype='0', conaffinity='0')
        else:
            _sub(world, 'geom', name=d['name'], type=d['type'], size=d['size'], pos=d['pos'], rgba=rgba,
                 contype='0', conaffinity='0')
    return ET.tostring(root, encoding='unicode')


class Track:
    """Keyframe steps applied in time order; each step min-jerk interpolates from the value at its start."""

    def __init__(self, steps, init: dict):
        self.steps, self.i, self.cur, self.start, self.goal = sorted(steps, key=lambda s: s['t'][0]), 0, dict(init), None, None

    def at(self, t: float, resolve) -> dict:
        while self.i < len(self.steps) and self.steps[self.i]['t'][0] <= t + 1e-9:
            s = self.steps[self.i]
            if self.start is None:
                self.start = dict(self.cur)
                self.goal = {k: (resolve(v) if k in ('to', 'point_at') else v) for k, v in s.items() if k != 't'}
            t0, t1 = s['t']
            u = 1.0 if t >= t1 else min_jerk((t - t0) / (t1 - t0))
            for k, v in self.goal.items():
                if isinstance(v, (int, float, np.ndarray)) and k in self.start and self.start[k] is not None \
                        and not isinstance(self.start[k], str):
                    self.cur[k] = np.asarray(self.start[k], float) + u * (np.asarray(v, float) - np.asarray(self.start[k], float))
                else:
                    self.cur[k] = v
            self.cur['progress'], self.cur['shape_from'] = u, self.start.get('shape')
            if t < t1:
                break
            self.i, self.start = self.i + 1, None
        return self.cur


def render_task(cfg: dict, name: str, task: dict, out_path, font, *, preflight=False, review_frames=False) -> dict:
    scene = load_scene(cfg, task)
    env = ManipulationEnv(model_xml=build_xml(cfg, scene, task))
    m, d = env.model, env.data
    body = HumanBody({**scene['human'], 'floor_z': scene.get('floor_z', 0.0)})
    body.bind(m)
    qadr = {k: int(m.joint(f'{k}_free').qposadr[0]) for k in task['objects']}
    obj = {k: np.asarray(o['pos'], float) for k, o in task['objects'].items()}
    holder: dict[str, tuple[str, np.ndarray]] = {}
    yaw = {k: 0.0 for k in obj}                                # about z; `spin` steps turn an object (min-jerk)
    spins, spin_from = sorted(task.get('spin', []), key=lambda s: s['t'][0]), {}
    Rz = lambda a: np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])

    def resolve(v):
        if isinstance(v, dict):
            return obj[v['obj']] + np.asarray(v.get('d', [0, 0, 0]), float)
        return None if v is None else np.asarray(v, float)

    mujoco.mj_forward(m, d)
    tcp0 = d.site_xpos[env.tcp].copy()
    rest = np.asarray(scene['human']['hand_rest'], float)
    human = Track(task.get('human', []), dict(to=rest, shape='relaxed', palm_up=0.0, point_at=None))
    two_hands = bool(task.get('human_l'))                      # the other hand (two-handed actions), else hangs
    human_l = Track(task.get('human_l', []), dict(to=body.pose(rest)['idle_palm'], shape='relaxed', palm_up=0.0))
    robot = Track(task.get('robot', []), dict(to=tcp0, grip=0.085))
    events = sorted(task.get('events', []), key=lambda e: e['t'])
    vc = cfg['video']
    fps = min(cfg['fps'], 10) if preflight else cfg['fps']
    W, H, CH = vc['width'], vc['height'], vc['caption_h']
    cams = cfg['show']
    out, st, renderer = None, None, None
    if not preflight:
        out = av.open(str(out_path), 'w', options={'movflags': 'faststart'})
        st = out.add_stream('libx264', rate=fps)
        st.width, st.height, st.pix_fmt = W * len(cams), H + CH, 'yuv420p'
        st.options = {'crf': '22', 'preset': 'veryfast'}
        renderer = mujoco.Renderer(m, height=H, width=W)
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0
    stats = dict(ik_max_m=0.0, reach_max_m=0.0)
    review_probes = sorted(task.get('state_probes', []), key=lambda p: p['t'])
    probe_i = 0
    previous_tcp = None
    path_total, path_after_cue = 0.0, 0.0
    cue_t = task.get('cost_review', {}).get('cue_t')
    if review_probes:
        stats['state_probes'] = []
    decision = task.get('decision', {})
    decision_origin, pre_reveal_motion = None, 0.0
    snapshots = {int(round(float(t) * fps)) for t in task.get('review_times', [])}
    if review_frames:
        old_frames = out_path.parent / 'review_frames' / name
        if old_frames.exists():
            for old_frame in old_frames.glob('*.jpg'):
                old_frame.unlink()
    T, ei = float(task['duration']), 0
    forecast_log = None
    if task.get('forecast_overlay') and not preflight:
        forecast_log = np.load(ROOT / task['forecast_overlay']['path'])
        stats['forecast_overlay'] = task['forecast_overlay']['path']
    for f in range(int(round(T * fps)) + 1):
        t = f / fps
        while ei < len(events) and events[ei]['t'] <= t + 1e-9:
            e = events[ei]
            if 'grab' in e and e['by'] in obj:                 # attached to another object (in its frame)
                holder[e['grab']] = (e['by'], Rz(yaw[e['by']]).T @ (obj[e['grab']] - obj[e['by']]))
            elif 'grab' in e:
                ref = {'robot': d.site_xpos[env.tcp], 'human': hp['palm'], 'human_l': hp['idle_palm']}[e['by']]
                holder[e['grab']] = (e['by'], obj[e['grab']] - ref)
            else:
                holder.pop(e['drop'], None)
                obj[e['drop']] = np.asarray(e['at'], float)
            ei += 1
        h = human.at(t, resolve)
        hl = human_l.at(t, resolve)
        r = robot.at(t, resolve)
        q = env.solve_ik(r['to'])
        if env.last_ik_error > stats['ik_max_m']:
            stats['ik_max_t'] = t
            stats['ik_max_target'] = np.asarray(r['to']).tolist()
        stats['ik_max_m'] = max(stats['ik_max_m'], env.last_ik_error)
        d.qpos[env.qids] = q
        env.set_gripper_qpos(float(r['grip']))
        hp = body.apply(d, h['to'], h.get('point_at'), h['shape'], palm_up=float(h.get('palm_up') or 0.0),
                        aim_blend=float(h.get('progress', 1.0)), shape_from=h.get('shape_from'),
                        shape_progress=h.get('progress'), idle_palm=hl['to'] if two_hands else None,
                        idle_shape=hl['shape'], idle_palm_up=float(hl.get('palm_up') or 0.0))
        stats['reach_max_m'] = max(stats['reach_max_m'], hp['reach_error'], hp['idle_reach_error'])
        mujoco.mj_forward(m, d)
        if decision and t >= decision['action_start']:
            if decision_origin is None:
                decision_origin = d.site_xpos[env.tcp].copy()
            if t < decision['reveal_t']:
                pre_reveal_motion = max(pre_reveal_motion, float(np.linalg.norm(d.site_xpos[env.tcp] - decision_origin)))
        for k, (who, off) in holder.items():
            if who not in obj:
                obj[k] = {'robot': d.site_xpos[env.tcp], 'human': hp['palm'], 'human_l': hp['idle_palm']}[who] + off
        for i, s in enumerate(spins):
            if t >= s['t'][0]:
                a0 = spin_from.setdefault(i, yaw[s['obj']])
                u = min_jerk(min(1.0, (t - s['t'][0]) / (s['t'][1] - s['t'][0])))
                yaw[s['obj']] = a0 + u * (np.radians(s['yaw']) - a0)
        for k, (who, off) in holder.items():
            if who in obj:
                obj[k] = obj[who] + Rz(yaw[who]) @ off
        for k, p in obj.items():
            d.qpos[qadr[k]:qadr[k] + 3] = p
            d.qpos[qadr[k] + 3:qadr[k] + 7] = [np.cos(yaw[k] / 2), 0.0, 0.0, np.sin(yaw[k] / 2)]
        mujoco.mj_forward(m, d)
        tcp_now = d.site_xpos[env.tcp].copy()
        if previous_tcp is not None:
            ds = float(np.linalg.norm(tcp_now - previous_tcp))
            path_total += ds
            if cue_t is not None and t > cue_t:
                path_after_cue += ds
        previous_tcp = tcp_now
        while probe_i < len(review_probes) and (t >= review_probes[probe_i]['t'] - 1e-8
                or (f == int(round(T * fps)) and review_probes[probe_i]['t'] <= T + 1e-8)):
            probe = review_probes[probe_i]
            stats['state_probes'].append(dict(label=probe['label'], t=t,
                tcp=tcp_now.tolist(), gripper=float(r['grip']),
                held_by_robot=[k for k, (who, _) in holder.items() if who == 'robot'],
                objects={k: dict(pos=p.tolist(), holder=holder[k][0] if k in holder else None)
                         for k, p in obj.items()}))
            probe_i += 1
        if preflight:
            continue
        frame = Image.new('RGB', (W * len(cams), H + CH), (18, 18, 18))
        for i, c in enumerate(cams):
            renderer.update_scene(d, camera=c, scene_option=opt)
            if forecast_log is not None:
                fi = int(np.argmin(abs(forecast_log['time_s'] - t)))
                points = forecast_log['display_predicted'][fi]
                active = bool(forecast_log['intervention_active'][fi])
                color = np.array([1., .22, .12, .95] if active else [.10, .85, 1., .90], np.float32)
                for joint in task['forecast_overlay'].get('joints', [5]):
                    for k, pos in enumerate(points[:, joint]):
                        geom = renderer.scene.geoms[renderer.scene.ngeom]
                        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_SPHERE,
                                           np.array([.0045 + k * .00035] * 3), np.asarray(pos, np.float64),
                                           np.eye(3).ravel(), color)
                        renderer.scene.ngeom += 1
            frame.paste(Image.fromarray(renderer.render()), (i * W, 0))
        draw_overlay(ImageDraw.Draw(frame), task, t, T, W * len(cams), H, CH, font)
        if review_frames and f in snapshots:
            folder = out_path.parent / 'review_frames' / name
            folder.mkdir(parents=True, exist_ok=True)
            frame.save(folder / f'{t:06.2f}.jpg', quality=88)
        for p in st.encode(av.VideoFrame.from_ndarray(np.asarray(frame), format='rgb24')):
            out.mux(p)
    if out is not None:
        for p in st.encode():
            out.mux(p)
        out.close()
        renderer.close()
    if decision:
        stats['pre_reveal_motion_m'] = pre_reveal_motion
        stats['action_before_reveal'] = bool(decision['action_start'] < decision['reveal_t'] and pre_reveal_motion >= 0.02)
    if review_probes or cue_t is not None:
        stats['scripted_tcp_path_m'] = path_total
        stats['scripted_robot_grabs'] = sum(e.get('by') == 'robot' for e in events)
        if cue_t is not None:
            stats['scripted_tcp_path_after_cue_m'] = path_after_cue
            stats['scripted_robot_grabs_after_cue'] = sum(e.get('by') == 'robot' and e['t'] >= cue_t for e in events)
        stats['cost_scope'] = 'kinematic_storyboard_only_not_policy_or_optimal_recovery'
    env.close()
    return stats


def draw_overlay(dr, task, t, T, width, H, CH, font) -> None:
    dr.rectangle([0, 0, width, 22], fill=(0, 0, 0))
    dr.text((6, 3), f"{task['title']}    t = {t:4.1f} s", fill=(255, 255, 255), font=font)
    text = next((c[2] for c in task.get('captions', []) if c[0] <= t < c[1]), '')
    words, lines, line = text.split(), [], ''
    for w_ in words:                                           # wrap to the frame width
        if dr.textlength(line + ' ' + w_, font=font) > width - 16:
            lines.append(line)
            line = w_
        else:
            line = (line + ' ' + w_).strip()
    lines.append(line)
    for i, ln in enumerate(lines[:2]):
        dr.text((8, H + 6 + 20 * i), ln, fill=(255, 235, 140), font=font)
    y0, x0, x1 = H + 64, 10, width - 10                        # timeline bar with marks and the time cursor
    dr.line([x0, y0, x1, y0], fill=(120, 120, 120), width=2)
    for mk in task.get('marks', []):
        x = x0 + (x1 - x0) * mk['t'] / T
        col = (255, 150, 40) if mk['who'] == 'human' else (90, 160, 255)
        up = mk['who'] == 'human'
        dr.line([x, y0 - 8, x, y0 + 8], fill=col, width=3)
        dr.text((x + 3, y0 - 22 if up else y0 + 6), f"{mk['t']:.1f} {mk['label']}", fill=col,
                font=font.font_variant(size=12))
    xc = x0 + (x1 - x0) * min(t, T) / T
    dr.polygon([(xc - 6, y0 - 12), (xc + 6, y0 - 12), (xc, y0 - 3)], fill=(255, 255, 255))
    if task.get('annotation_legend'):
        dr.text((8, H + CH - 19), task['annotation_legend'], fill=(145, 215, 235),
                font=font.font_variant(size=12))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='configs/scenarios/v6_preview.yaml')
    ap.add_argument('--tasks', nargs='*')
    ap.add_argument('--output-dir', help='Explicit output folder for batch rendering/resume')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--preflight', action='store_true', help='Check IK/reach and scripted case timing without video')
    ap.add_argument('--review-frames', action='store_true', help='Save the specified review keyframes alongside clips')
    ap.add_argument('--log-name', default='run.log', help='Separate log names for disjoint batch workers')
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    out = ROOT / args.output_dir if args.output_dir else ROOT / cfg.get('out_dir', 'outputs/scenario_v6_preview') / datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / args.config, out / 'config.yaml')
    font = ImageFont.truetype(cfg['video']['font'], cfg['video']['font_size'])
    log = []
    for name in args.tasks or list(cfg['tasks']):
        stat_path = out / f'{name}.{"preflight" if args.preflight else "stats"}.json'
        clip_path = out / f'{name}.mp4'
        if args.resume and not args.preflight and clip_path.exists() and stat_path.exists() \
                and stat_path.stat().st_mtime_ns >= clip_path.stat().st_mtime_ns:
            print(f'{name}: already rendered', flush=True)
            continue
        stats = render_task(cfg, name, cfg['tasks'][name], out / f'{name}.mp4', font,
                            preflight=args.preflight, review_frames=args.review_frames)
        stat_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + '\n')
        line = f"{name}: {out / f'{name}.mp4'}  max IK error {stats['ik_max_m'] * 1000:.1f} mm, " \
               f"max human reach shortfall {stats['reach_max_m'] * 1000:.1f} mm"
        print(line, flush=True)
        log.append(line)
    with (out / ('preflight.log' if args.preflight else args.log_name)).open('a' if args.resume else 'w') as handle:
        handle.write('\n'.join(log) + '\n')


if __name__ == '__main__':
    main()

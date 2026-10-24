"""Render the kinematic storyboard of the proposed T1-T5 v6 scenarios (configs/scenarios/v6_preview.yaml).

Reuses the scene builder, the mannequin (HumanBody), the robot IK and the video writer of the real pipeline; only
the hand-written timelines are new. No physics, no expert, no intent model: a preview for approving the scripts.
  python scripts/preview_scenarios_v6.py [--tasks T1 T3]   -> outputs/scenario_v6_preview/<date_time>/<task>.mp4
"""
import argparse
import copy
import datetime
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


def load_scene(cfg: dict, task: dict) -> dict:
    base = yaml.safe_load((ROOT / cfg['scene_base']).read_text())
    scene = copy.deepcopy(base)
    scene['cameras'] = cfg['cameras']
    scene['objects'] = {k: {**{kk: vv for kk, vv in o.items() if kk != 'pos'}, 'color': cfg['colors'][o['color']]}
                        for k, o in task['objects'].items()}
    return scene


def build_xml(cfg: dict, scene: dict, task: dict) -> str:
    root = ET.fromstring(build_scene_xml(scene))
    world = root.find('worldbody')
    for g in [g for g in world.findall('geom') if g.get('name', '').startswith('marker_')]:
        world.remove(g)                                        # v6 has no S/P/H/U tape
    for d in task.get('decor', []):
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


def render_task(cfg: dict, name: str, task: dict, out_path, font) -> dict:
    scene = load_scene(cfg, task)
    env = ManipulationEnv(model_xml=build_xml(cfg, scene, task))
    m, d = env.model, env.data
    body = HumanBody({**scene['human'], 'floor_z': scene.get('floor_z', 0.0)})
    body.bind(m)
    qadr = {k: int(m.joint(f'{k}_free').qposadr[0]) for k in task['objects']}
    obj = {k: np.asarray(o['pos'], float) for k, o in task['objects'].items()}
    holder: dict[str, tuple[str, np.ndarray]] = {}

    def resolve(v):
        if isinstance(v, dict):
            return obj[v['obj']] + np.asarray(v.get('d', [0, 0, 0]), float)
        return None if v is None else np.asarray(v, float)

    mujoco.mj_forward(m, d)
    tcp0 = d.site_xpos[env.tcp].copy()
    human = Track(task.get('human', []), dict(to=np.asarray(scene['human']['hand_rest'], float), shape='relaxed',
                                              palm_up=0.0, point_at=None))
    robot = Track(task.get('robot', []), dict(to=tcp0, grip=0.085))
    events = sorted(task.get('events', []), key=lambda e: e['t'])
    vc, fps = cfg['video'], cfg['fps']
    W, H, CH = vc['width'], vc['height'], vc['caption_h']
    cams = cfg['show']
    out = av.open(str(out_path), 'w', options={'movflags': 'faststart'})
    st = out.add_stream('libx264', rate=fps)
    st.width, st.height, st.pix_fmt = W * len(cams), H + CH, 'yuv420p'
    st.options = {'crf': '22', 'preset': 'veryfast'}
    renderer = mujoco.Renderer(m, height=H, width=W)
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0
    stats = dict(ik_max_m=0.0, reach_max_m=0.0)
    T, ei = float(task['duration']), 0
    for f in range(int(round(T * fps)) + 1):
        t = f / fps
        while ei < len(events) and events[ei]['t'] <= t + 1e-9:
            e = events[ei]
            if 'grab' in e:
                ref = d.site_xpos[env.tcp] if e['by'] == 'robot' else hp['palm']
                holder[e['grab']] = (e['by'], obj[e['grab']] - ref)
            else:
                holder.pop(e['drop'], None)
                obj[e['drop']] = np.asarray(e['at'], float)
            ei += 1
        h = human.at(t, resolve)
        r = robot.at(t, resolve)
        q = env.solve_ik(r['to'])
        stats['ik_max_m'] = max(stats['ik_max_m'], env.last_ik_error)
        d.qpos[env.qids] = q
        env.set_gripper_qpos(float(r['grip']))
        hp = body.apply(d, h['to'], h.get('point_at'), h['shape'], palm_up=float(h.get('palm_up') or 0.0),
                        aim_blend=float(h.get('progress', 1.0)), shape_from=h.get('shape_from'),
                        shape_progress=h.get('progress'))
        stats['reach_max_m'] = max(stats['reach_max_m'], hp['reach_error'])
        mujoco.mj_forward(m, d)
        for k, (who, off) in holder.items():
            obj[k] = (d.site_xpos[env.tcp] if who == 'robot' else hp['palm']) + off
        for k, p in obj.items():
            d.qpos[qadr[k]:qadr[k] + 3], d.qpos[qadr[k] + 3:qadr[k] + 7] = p, [1, 0, 0, 0]
        mujoco.mj_forward(m, d)
        frame = Image.new('RGB', (W * len(cams), H + CH), (18, 18, 18))
        for i, c in enumerate(cams):
            renderer.update_scene(d, camera=c, scene_option=opt)
            frame.paste(Image.fromarray(renderer.render()), (i * W, 0))
        draw_overlay(ImageDraw.Draw(frame), task, t, T, W * len(cams), H, CH, font)
        for p in st.encode(av.VideoFrame.from_ndarray(np.asarray(frame), format='rgb24')):
            out.mux(p)
    for p in st.encode():
        out.mux(p)
    out.close()
    renderer.close()
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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', default='configs/scenarios/v6_preview.yaml')
    ap.add_argument('--tasks', nargs='*')
    args = ap.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text())
    out = ROOT / 'outputs/scenario_v6_preview' / datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / args.config, out / 'config.yaml')
    font = ImageFont.truetype(cfg['video']['font'], cfg['video']['font_size'])
    log = []
    for name in args.tasks or list(cfg['tasks']):
        stats = render_task(cfg, name, cfg['tasks'][name], out / f'{name}.mp4', font)
        line = f"{name}: {out / f'{name}.mp4'}  max IK error {stats['ik_max_m'] * 1000:.1f} mm, " \
               f"max human reach shortfall {stats['reach_max_m'] * 1000:.1f} mm"
        print(line, flush=True)
        log.append(line)
    (out / 'run.log').write_text('\n'.join(log) + '\n')


if __name__ == '__main__':
    main()

"""Episode videos (H.264 mp4, plays in any player), recorded while the episode runs as `lerobot-eval` does.

Frame: the `high` camera large (rendered at the panel size), the agent's other camera images in a side column, the
episode, time and TCP speed on top, the benchmark events logged so far at the bottom (cue onset, intention evident,
target commit, grasp, release, ...), and a SUCCESS / FAILURE banner held for 2 s at the end.
"""
from __future__ import annotations
from pathlib import Path

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SHOWN_EVENTS = ('human_cue_onset', 'human_intention_evident', 'human_intention_change', 'robot_target_commit',
                'object_grasp', 'object_release', 'disruption_start', 'disruption_end', 'human_robot_contact',
                'safety_distance_violation', 'task_success', 'task_failure')
MAIN_CAMERA = 'high'                       # shown large; the side column shows the other cameras
MAIN_W, MAIN_H, SIDE_W = 512, 384, 256
FPS = 20                                   # one decision per 0.05 s tick, as in the datasets
FONT = ImageFont.load_default(size=12)      # scalable (FreeType) default font


def event_line(e: dict) -> str:
    p = e.get('payload') or {}
    detail = ' '.join(f'{k}={v}' for k, v in p.items() if k in ('target', 'object', 'region', 'kind', 'new', 'reason'))
    return f"{e['timestamp']:5.2f}s {e['event_type']} {detail}".rstrip()


class VideoWriter:
    def __init__(self, path: Path, width: int = MAIN_W + SIDE_W, height: int = MAIN_H):
        self.out = av.open(str(path), 'w', options={'movflags': 'faststart'})
        self.stream = self.out.add_stream('libx264', rate=FPS)
        self.stream.width, self.stream.height, self.stream.pix_fmt = width, height, 'yuv420p'
        self.stream.options = {'crf': '23', 'preset': 'veryfast'}

    def add(self, frame: np.ndarray) -> None:
        for packet in self.stream.encode(av.VideoFrame.from_ndarray(frame, format='rgb24')):
            self.out.mux(packet)

    def close(self) -> None:
        for packet in self.stream.encode():
            self.out.mux(packet)
        self.out.close()


def compose(main: np.ndarray, side: list[np.ndarray], header: str, events: list[str], banner: str | None = None,
            side_names: list[str] | None = None) -> np.ndarray:
    """One video frame: `main` (any size, scaled to MAIN_W x MAIN_H) + up to two `side` images stacked on the right."""
    canvas = Image.new('RGB', (MAIN_W + SIDE_W, MAIN_H), (20, 20, 20))
    img = Image.fromarray(np.asarray(main))
    canvas.paste(img if img.size == (MAIN_W, MAIN_H) else img.resize((MAIN_W, MAIN_H)), (0, 0))
    y = 0
    for s in side[:MAIN_H // (SIDE_W * 3 // 4)]:
        canvas.paste(Image.fromarray(np.asarray(s)).resize((SIDE_W, SIDE_W * 3 // 4)), (MAIN_W, y))
        y += SIDE_W * 3 // 4
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, MAIN_W, 18], fill=(0, 0, 0))
    d.text((4, 2), header, fill=(255, 255, 255), font=FONT)
    shown = events[-6:]
    if shown:
        d.rectangle([0, MAIN_H - 16 * len(shown) - 4, MAIN_W, MAIN_H], fill=(0, 0, 0))
    for i, line in enumerate(shown):
        d.text((4, MAIN_H - 16 * (len(shown) - i) - 2), line, fill=(255, 230, 120), font=FONT)
    if side_names:
        d.rectangle([MAIN_W, 0, MAIN_W + SIDE_W, 16], fill=(0, 0, 0))
        d.text((MAIN_W + 4, 1), ' / '.join(side_names), fill=(200, 200, 200), font=FONT)
    if banner:
        ok = banner.startswith('SUCCESS')
        d.rectangle([0, MAIN_H // 2 - 20, MAIN_W, MAIN_H // 2 + 20], fill=(20, 110, 40) if ok else (140, 30, 30))
        d.text((12, MAIN_H // 2 - 7), banner, fill=(255, 255, 255), font=FONT)
    return np.asarray(canvas)


class EpisodeVideo:
    """`on_frame` callback for `runner.run_episode` that records one episode into a temporary file; `finish` then
    keeps it (at the given path, with the closing banner) or deletes it."""

    def __init__(self, folder: Path, title: str):
        folder.mkdir(parents=True, exist_ok=True)
        self.tmp = folder / f'.{title.replace(" ", "_")}.part.mp4'
        self.title, self.writer, self.last = title, VideoWriter(self.tmp), None

    def __call__(self, obs, decision, target, s) -> None:
        keys = sorted(k for k in obs if k.startswith('observation.images.') and k != f'observation.images.{MAIN_CAMERA}')
        main = s.env.render(MAIN_CAMERA, MAIN_W, MAIN_H)
        events = [event_line(e) for e in s.get_events() if e['event_type'] in SHOWN_EVENTS]
        self.last = (main, [obs[k] for k in keys], f'{self.title}  t={s.time:5.2f}s  tcp={s.tcp_speed:.2f} m/s',
                     [k.removeprefix('observation.images.') for k in keys])
        self.writer.add(compose(self.last[0], self.last[1], self.last[2], events, side_names=self.last[3]))

    def finish(self, rec: dict, path: Path | None) -> Path | None:
        """Keep the video at `path` (None: delete it)."""
        if path is not None and self.last is not None:
            banner = 'SUCCESS' if rec['success'] else f"FAILURE: {rec['failure']}"
            events = [event_line(e) for e in rec['events'] if e['event_type'] in SHOWN_EVENTS]
            final = compose(self.last[0], self.last[1], self.last[2], events, banner, side_names=self.last[3])
            for _ in range(FPS * 2):
                self.writer.add(final)
        self.writer.close()
        if path is None or self.last is None:
            self.tmp.unlink(missing_ok=True)
            return None
        self.tmp.replace(path)
        return path

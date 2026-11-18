"""LeRobot recorder shared by both modes (QĐ 53/QĐ-DL §6). One dataset per mode (hrc_observed / hrc_forecast), all
10 tasks inside, distinguished by the LeRobot task string and the episode metadata sidecar `meta/hrc_episodes.jsonl`.

Frames are added during the rollout (observation at t + the action applied at t) and either saved
(`save_episode`) or discarded (`clear_episode_buffer`) once the episode validator decides. Rejected episodes keep
their summary and trace under `rejected_episodes/`.
"""
from pathlib import Path
import json

import mujoco
import numpy as np

from intent_policy.data.dataset_schema import features, HISTORY, HORIZONS, N_JOINTS


class LeRobotRecorder:
    def __init__(self, root, repo_id, fps, cameras, image_hw, robot_type):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        self.root, self.cameras, self.image_hw = Path(root), list(cameras), tuple(image_hw)
        if (self.root / 'meta' / 'info.json').exists():
            self.ds = LeRobotDataset.resume(repo_id, root=self.root)
        else:
            self.ds = LeRobotDataset.create(repo_id, fps=fps, features=features(self.cameras, self.image_hw),
                                            root=self.root, robot_type=robot_type, use_videos=True)
        self.sidecar = self.root / 'meta' / 'hrc_episodes.jsonl'
        self.task_string = None
        self.frames = 0

    # -------------------------------------------------------------- per episode
    def start_episode(self, ep):
        self.episode = ep
        self.task_string = ep.task
        self.frames = 0

    def add_frame(self, ep, action):
        h, w = self.image_hw
        q, qd = ep.d.qpos[ep.env.qids], ep.d.qvel[ep.env.dids]
        grip = float(ep.d.qpos[ep.env.gid])
        quat = np.zeros(4); mujoco.mju_mat2Quat(quat, ep.d.site_xmat[ep.env.tcp])
        hist = list(ep.history)
        hist_valid = float(len(hist) >= HISTORY)
        hist = [hist[0]] * (HISTORY - len(hist)) + hist          # pad by repeating the oldest frame (mask = 0)
        mc = ep.manicast
        fc = mc.forecast if mc.valid else np.zeros((HORIZONS, N_JOINTS, 3))
        ref = ep.controller.reference
        frame = {f'observation.images.{c}': ep.env.render(c, w, h) for c in self.cameras}
        frame.update({
            'observation.state': np.r_[q, qd, grip].astype(np.float32),
            'action': np.asarray(action, np.float32),
            'observation.tcp_pose': np.r_[ep.tcp(), quat].astype(np.float32),
            'observation.tcp_reference': (ep.tcp() if ref is None else ref).astype(np.float32),
            'observation.tcp_reference_valid': np.array([ref is not None], np.float32),
            'observation.human_pose': np.asarray(ep.kp, np.float32).reshape(-1),
            'observation.human_history': np.asarray(hist, np.float32).reshape(-1),
            'observation.human_history_valid': np.array([hist_valid], np.float32),
            'observation.human_forecast': np.asarray(fc, np.float32).reshape(-1),
            'observation.forecast_valid': np.array([float(mc.valid)], np.float32),
            'observation.forecast_age': np.array([mc.age(ep.t) if mc.valid else 0.], np.float32),
            'observation.forecast_provided': np.array([float(ep.forecast_available() is not None)], np.float32),
            'task': self.task_string,
        })
        for key, value in frame.items():
            if key != 'task' and not np.isfinite(value).all():
                raise ValueError(f'non-finite {key} at tick {ep.tick}')
        self.ds.add_frame(frame)
        self.frames += 1

    def accept(self, metadata, arrays=None):
        index = self.ds.meta.total_episodes
        self.ds.save_episode()
        trace_dir = self.root / 'meta' / 'hrc_sidecar'
        trace_dir.mkdir(parents=True, exist_ok=True)
        (trace_dir / f'episode_{index:06d}_trace.json').write_text(json.dumps({
            'time_convention': 't = observation/action start; trace TCP/objects measured after physics at t + dt',
            'dt_s': 1. / self.ds.fps, 'frames': self.episode.trace}) + '\n')
        if arrays:                                   # 59/QĐ-DL: raw forecast, fault strength, natural error
            d = self.root / 'meta' / 'hrc_sidecar'; d.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(d / f'episode_{index:06d}.npz', **arrays)
        with self.sidecar.open('a') as f:
            f.write(json.dumps({'episode_index': index, **metadata}) + '\n')
        return index

    def reject(self):
        if self.ds.has_pending_frames():
            self.ds.clear_episode_buffer(delete_images=True)

    def finalize(self):
        self.ds.finalize()

    def recorded_ids(self):
        if not self.sidecar.exists():
            return set()
        return {json.loads(l)['episode_id'] for l in self.sidecar.read_text().splitlines() if l.strip()}

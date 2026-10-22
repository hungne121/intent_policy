"""The intent measuring device in the loop: the multi-gesture DeePoint (docs/requirements/INTENT_ACT_v3.2.md §4.2e)
running on the simulated sensor camera, turned into the belief the experts act on (`expert.trigger: device`).

  DeePointClient   the streaming DeePoint (scripts/deepoint_server.py, its own python environment) as a subprocess:
                   one 15 fps camera frame in, class probabilities + DeePoint direction + 2D keypoints out
  Camera           pinhole camera of a MuJoCo camera; DeePoint's camera-centred direction -> world direction
  DeviceBelief     per 20 Hz control frame (the device sees every 15 fps frame: frame k = control frame round(4k/3),
                   as the training data): the gesture class and, for the directed gestures, the belief over the task's
                   candidates  L(g) = p_c exp(-theta_g^2 / 2 sigma_c^2) + (1 - p_c),  b = L / sum L,  theta_g = angle
                   between the world direction and hand -> g (hand = wrist + 0.2 (wrist - elbow), DeePoint's origin;
                   sigma_c measured on the validation fold): objects, zones, p_stop, p_withdraw, onset_seen, snapshot
                   for the belief expert (experts/scripted_expert.py) and the intent contract (intent/source_v32.py).
Five models (`folds_file`, written by workstation/job1_device.sh): model k is trained without folds k (test) and
k + 1 (validation) and calibrated on fold k + 1; an episode is measured by the model of its own fold
(intent/folds.fold_of), so no episode is measured by a model that saw it.
Decision rules (`rule`, the user's definition, §0 and §4.2c-j):
  late    the gesture is complete - the device saw the hand stop at its pose (speed below STILL_FRACTION of the
          motion's peak, the DP / motor-control convention) - and a candidate's belief >= late_star (0.95): "late =
          the expert waits for the gesture to finish" (§0; 2026-10-14, before: belief >= 0.95 alone, reached mid-gesture
          only ~0.2 s after 0.7, so the two rules hardly differed). 'wait' / withdraw once their probability >= late_star
  early   a candidate's belief >= early_star (0.7), during the gesture; 'wait' / withdraw once their probability
          >= early_star
          2026-10-18 (the user's rule, docs/discussions/handover_som_va_intent_sai.md; slide bao_cao_2026-10-18):
          `approach_star` (0.85) = the lowest belief level after which the leading candidate changes in < 10 % of the
          gestures (scripts/device_stability.py, 241 unseen gestures): from it the robot moves towards the candidates
          (reversible moves only, the belief-weighted point, Javdani et al. 2018); `commit_star` (0.9) = the commit,
          after which the irreversible steps follow (descend, grasp). 'wait' / withdraw keep early_star and their
          calibrated holds. Unset: both equal early_star (the 0.7 rule).
The two experts differ only by these thresholds (neither moves before its first approach / commit). The device
runs before the episode (`preroll_frames` copies of the resting human), as a real camera already running would, so
its first output is at t = 0 instead of after its 15-frame window (0.95 s).
"""
from __future__ import annotations
import json
import struct
import subprocess
from pathlib import Path
import numpy as np
from intent_policy.intent.folds import fold_of
from intent_policy.intent.places import target_positions
from intent_policy.utils import resolve

R_WRIST, R_ELBOW = 10, 8             # COCO-17 (DeePoint's hand_id / elbow_id, right arm)


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def angle(a, b) -> float:
    return float(np.degrees(np.arccos(np.clip(np.dot(unit(a), unit(b)), -1.0, 1.0))))


class Camera:
    """Pinhole camera from the MuJoCo pose (columns of xmat: x right, y up, looking along -z) and fovy."""

    def __init__(self, pos, xmat, fy, W, H):
        self.pos, self.fy, self.W, self.H = np.asarray(pos, float), float(fy), int(W), int(H)
        self.R = np.asarray(xmat, float) @ np.diag([1.0, -1.0, -1.0])      # OpenCV axes (x right, y down, z fwd)

    @classmethod
    def from_env(cls, env, name: str, fovy_deg: float, W: int, H: int) -> 'Camera':
        cid = env.model.camera(name).id
        return cls(env.data.cam_xpos[cid], env.data.cam_xmat[cid].reshape(3, 3), H / 2 / np.tan(np.radians(fovy_deg) / 2), W, H)

    def ray(self, uv) -> np.ndarray:
        return unit(self.R @ np.array([(uv[0] - self.W / 2) / self.fy, (uv[1] - self.H / 2) / self.fy, 1.0]))

    def project(self, X) -> np.ndarray | None:
        c = self.R.T @ (np.asarray(X, float) - self.pos)
        return None if c[2] <= 1e-6 else np.array([self.fy * c[0] / c[2] + self.W / 2, self.fy * c[1] / c[2] + self.H / 2])

    def world_direction(self, d_dp, kp2d) -> np.ndarray | None:
        """DeePoint direction (x right, y depth, z up in its hand-centred frame, src/dataset.py
        hand2marker_direction) -> world unit vector; the frame is rebuilt from DeePoint's own 2D wrist / elbow."""
        w, e = kp2d[R_WRIST], kp2d[R_ELBOW]
        if w[2] <= 0 or e[2] <= 0 or d_dp is None or not np.all(np.isfinite(d_dp)):
            return None
        Ez = self.ray(w[:2] + 0.2 * (w[:2] - e[:2]))
        Ey = unit(np.cross(self.R[:, 0], Ez))
        if Ey @ self.R[:, 1] > 0:              # image y is down: Ey must point up
            Ey = -Ey
        Ex = np.cross(Ey, Ez)
        if Ex @ self.R[:, 0] < 0:
            Ex = -Ex
        x, depth, up = d_dp
        return unit(x * Ex + up * Ey + depth * Ez)


class DeePointClient:
    """scripts/deepoint_server.py as a subprocess holding the fold models (one per process, kept across episodes);
    `reset(model)` selects the model of the next episode."""
    _shared: dict[tuple, 'DeePointClient'] = {}

    def __init__(self, ckpts: tuple[Path, ...]):
        root = resolve('.')
        (root / 'outputs').mkdir(exist_ok=True)
        log = open(root / 'outputs' / 'deepoint_server.log', 'a')
        self.proc = subprocess.Popen([str(root / 'scripts' / 'deepoint.sh'), 'scripts/deepoint_server.py', '--ckpt',
                                      *map(str, ckpts)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, cwd=root)
        ready = json.loads(self.proc.stdout.readline() or '{}')
        if not ready.get('ready'):
            raise RuntimeError('the DeePoint server did not start (outputs/deepoint_server.log)')
        self.n_classes = int(ready['classes'])

    @classmethod
    def shared(cls, ckpts) -> 'DeePointClient':
        key = tuple(str(c) for c in ckpts)
        if key not in cls._shared or cls._shared[key].proc.poll() is not None:
            cls._shared[key] = cls(key)
        return cls._shared[key]

    def _send(self, header: dict, payload: bytes = b'') -> dict:
        h = json.dumps(header).encode()
        self.proc.stdin.write(struct.pack('<I', len(h)) + h + payload)
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError('the DeePoint server stopped (outputs/deepoint_server.log)')
        return json.loads(line)

    def reset(self, model: int = 0) -> None:
        self._send({'cmd': 'reset', 'model': int(model)})

    def step(self, rgb: np.ndarray) -> dict:
        rgb = np.ascontiguousarray(rgb, np.uint8)
        return self._send({'cmd': 'frame', 'h': rgb.shape[0], 'w': rgb.shape[1]}, rgb.tobytes())


def load_folds(path, classes: list[str]) -> tuple[dict[int, dict], dict]:
    """The fold models and their calibration (workstation/job1_device.sh -> device_folds.json):
    {fold: {checkpoint, sigma_deg, sigma_deg_by_frames, hold_frames: {late, early}, pause_timeout_s}}, and the file's
    assignment (`assign: single`: one model for every episode, `seen_seeds` refused; else the model of the fold)."""
    f = resolve(path)
    if not f.exists():
        raise FileNotFoundError(f'{f}: no device models yet (run workstation/job1_device.sh and download its outputs, '
                                f'workstation/README.md)')
    d = json.loads(f.read_text())
    if list(d['classes']) != list(classes):
        raise ValueError(f"{f}: classes {d['classes']} != device.classes {classes}")
    return {int(k): v for k, v in d['folds'].items()}, dict(assign=d.get('assign', 'fold'), model=d.get('model'),
                                                             seen_seeds=set(d.get('seen_seeds') or []))


class DeviceBelief:
    """The device's belief for a running scenario (see the module doc). Candidates, as the training labels:
      T1  point over the instructed kind's objects + the place zones P1 / P2 (one joint set)
      T2  point over the instructed kind's objects; palm_up over the hand zones H1 / H2
      T3  reach over the cubes in U -> their paired cups
      T4  none (the yield rule stays the constant-velocity hand, Schöller et al. 2020)
    Task-protocol prior (the context cue, like the instruction prior of FETCH-POMDP; observable only): the objects are
    candidates while the robot holds nothing, and again after the device has seen the 'wait' gesture (a change of
    mind, T5) until the next grasp; T1's place zones are candidates while the robot holds the object (the place is
    shown once the cube is lifted). Without it, the arm sweeping past the objects on its way to point at a zone gives
    an object a momentary belief >= 0.95 (device check 2026-10-12).
    Decision lock: a gesture's belief is frozen once the hand has stopped (speed below STILL_FRACTION of the
    motion's peak, the DP / motor-control convention used for the pose reached, §4.2c-d) and stays so for the rest of
    that gesture: until the device confirms a change-of-mind signal ('wait' or withdraw) or detects the start of a new
    gesture of the class (its per-frame decision turns from 'none' to the class: the action start of online action
    detection, Shou et al. ECCV 2018). While the hand moves the belief - and the expert's choice - may still change.
    Without it, the direction from a hand that has arrived at its target (the vector hand -> target ~ 0, T3 lifting
    the cube) flips the belief. Before 2026-10-12 the lock lasted across gestures, so a missed withdraw kept the old
    choice for the next reach (late T5 idx 484: withdraw p >= 0.95 for 6 frames, hold 7)."""

    STILL_FRACTION = 0.10

    def __init__(self, scenario, schema: dict, cfg: dict):
        self.sc, self.schema, self.cfg = scenario, schema, dict(cfg)
        self.classes = list(cfg['classes'])
        self.rule = cfg['rule']
        if self.rule not in ('early', 'late'):
            raise ValueError(f"device rule must be early | late, got {self.rule!r}")
        self.late_star = float(cfg['late_star'])
        self.star = self.late_star if self.rule == 'late' else float(cfg['early_star'])
        # target beliefs (module doc): the commit level and, early only, the approach level (None: no approach)
        self.commit_star = self.star if self.rule == 'late' else float(cfg.get('commit_star', self.star))
        self.approach_star = None if self.rule == 'late' else cfg.get('approach_star')
        if self.approach_star is not None:
            self.approach_star = float(self.approach_star)
            if not self.approach_star <= self.commit_star:
                raise ValueError(f'device approach_star {self.approach_star} must be <= commit_star {self.commit_star}')
        self.preroll = int(cfg.get('preroll_frames', 0))
        self.folds, self.assign = load_folds(cfg['folds_file'], self.classes)
        self.cam_name, self.W, self.H = cfg['camera'], int(cfg['width']), int(cfg['height'])
        self.client = DeePointClient.shared(tuple(resolve(self.folds[k]['checkpoint']) for k in sorted(self.folds)))
        if self.client.n_classes != len(self.classes):
            raise ValueError(f'checkpoints have {self.client.n_classes} classes, device.classes {len(self.classes)}')
        self.code = scenario.cfg.task_code
        self.places = target_positions(schema)
        self.fps = float(schema['fps'])                # step() runs once per control frame

    # -- episode
    def reset(self) -> None:
        sc, var = self.sc, self.sc.variation
        spec, desc = var['spec'], self.schema['instruction']['object_desc']
        if self.assign['assign'] == 'single':              # preliminary: one model, never on what it saw
            if int(sc.seed) in self.assign['seen_seeds']:
                raise ValueError(f'seed {sc.seed}: the single device model was trained / validated on this episode')
            self.fold = int(self.assign['model'])
        else:
            self.fold = fold_of(sc.seed)                   # the model that did not see this episode
        f = self.folds[self.fold]
        self.hold = {g: int(k) for g, k in f['hold_frames'].get(self.rule, {}).items()}
        self.sigma = {k: float(v) for k, v in f['sigma_deg'].items()}
        sb = f.get('sigma_deg_by_frames') or {}
        self.sigma_frames = np.asarray(sb.get('frames', []), int)
        self.sigma_table = {k: np.asarray(v, float) for k, v in sb.items() if k != 'frames'}
        self.pause_timeout_s = float(f.get('pause_timeout_s') or 0.0)   # none measured: no timeout
        self.client.reset(sorted(self.folds).index(self.fold))
        self.camera = None
        if self.preroll:                               # the sensor ran before the episode: the resting human
            self._ensure_camera()
            rest = sc.env.render(self.cam_name, self.W, self.H)
            for _ in range(self.preroll):
                self.client.step(rest)
        self.t = self.k15 = 0
        self.probs = self.dw = None
        self.top = 'none'
        self.onset_seen = False
        self.holding = False
        self.wait_seen = False                         # 'wait' detected since the last grasp
        self.runs = {g: 0 for g in ('palm_forward', 'withdraw')}   # consecutive device frames with the gesture flag
        self.class_run = {c: 0 for c in self.classes}  # consecutive device frames with the class as argmax
        self.last_point_t: float | None = None         # time of the last device frame that detected pointing
        self.first_top: str | None = None              # the device's class at its first output (pre-roll check)
        self.frozen: dict[tuple, dict[str, float]] = {}  # (class, candidates) -> belief locked when the hand stopped
        self.await_motion = False                      # a lock was cleared: a reach needs a new motion (step)
        self.reach_best: dict[str, float] = {}         # most confident reach belief of the current motion (_freeze)
        self.hand = None
        self.peak = 0.0                                # peak hand speed of the current motion [m/s]
        self.moving = False
        self.point_set: dict[str, np.ndarray] = {}         # candidate -> position, pointing
        self.zone_set: dict[str, np.ndarray] = {}          # palm_up (T2)
        self.reach_set: dict[str, np.ndarray] = {}         # reach (T3: cube spot -> its cup)
        self.reach_map: dict[str, str] = {}
        self.object_keys: list[str] = []
        self.zone_keys: list[str] = []
        if self.code in ('T1', 'T2'):
            kind = desc[spec['target']]
            self.object_keys = [k for k in spec['layout'] if desc.get(k) == kind]
            self.point_set = {k: np.asarray(sc.start_pos[k], float) for k in self.object_keys}
        if self.code == 'T1':
            self.zone_keys = [k for k, z in sc.zones.items() if z['kind'] == 'place']
            self.point_set.update({k: self.places[k] for k in self.zone_keys})
        if self.code == 'T2':
            self.zone_keys = [k for k, z in sc.zones.items() if z['kind'] == 'hand']
            self.zone_set = {k: self.places[k] for k in self.zone_keys}
        if self.code == 'T3':
            cup_of = var['pairs']
            for b in spec['u_blocks']:
                self.reach_map[b] = cup_of[b]
                self.reach_set[b] = np.asarray(sc.start_pos[b], float)
            self.object_keys = list(self.reach_map.values())
        self.t3_handover = self.code == 'T3' and bool(sc.cfg.human_behavior.get('handover'))
        if self.t3_handover:                           # T3 new: the open hand over H1 / H2, as T2
            self.zone_keys = [k for k, z in sc.zones.items() if z['kind'] == 'hand']
            self.zone_set = {k: self.places[k] for k in self.zone_keys}

    def step(self) -> None:
        """Advance one 20 Hz control frame; the device sees the 15 fps frames."""
        sc = self.sc
        if round(self.k15 * 4 / 3) == self.t:
            self._ensure_camera()
            out = self.client.step(sc.env.render(self.cam_name, self.W, self.H))
            self.k15 += 1
            if out['probs'] is not None:
                self.probs = np.asarray(out['probs'], float)
                prev, self.top = self.top, self.classes[int(np.argmax(self.probs))]
                if prev == 'none' and self.top != 'none':     # a new gesture starts: its decision is open
                    self.frozen = {k: v for k, v in self.frozen.items() if k[0] != self.top}
                if self.first_top is None:
                    self.first_top = self.top
                self.dw = self.camera.world_direction(out['direction'], np.asarray(out['kp'], float))
                self.onset_seen = self.onset_seen or self.top != 'none'
                for g in self.runs:
                    self.runs[g] = self.runs[g] + 1 if self._gesture_flag(g) else 0
                for c in self.class_run:
                    self.class_run[c] = self.class_run[c] + 1 if self.top == c else 0
                if self.top == 'point':
                    self.last_point_t = float(sc.time)
        holding = any(sc.grasp_state.values())
        if holding and not self.holding:
            self.wait_seen = False                     # a (new) object is in the gripper
        self.holding = holding
        if self.confirmed('palm_forward'):
            self.wait_seen = True
        if self.confirmed('palm_forward') or self.confirmed('withdraw'):
            if self.frozen:
                self.await_motion = True
            self.frozen.clear()                        # a change of mind is signalled: decisions open again
        seg = sc.human_pose['segments']
        wrist, elbow = np.asarray(seg['r_hand'][0], float), np.asarray(seg['r_forearm'][0], float)
        hand = wrist + 0.2 * (wrist - elbow)
        speed = 0.0 if self.hand is None else float(np.linalg.norm(hand - self.hand)) * self.fps   # per control frame
        self.hand = hand
        self.peak = max(self.peak, speed)
        if speed >= self.STILL_FRACTION * self.peak and speed > 0:
            if not self.moving:
                self.reach_best = {}                   # a new motion
            self.moving = True
            self.await_motion = False
            if self.code == 'T3' and self.top == 'reach':
                b = self._belief('reach', self.reach_set)
                if b and max(b.values()) > max(self.reach_best.values(), default=0.0):
                    self.reach_best = b
        elif self.moving:                              # the motion ended: lock the current gesture's decision
            self.moving, self.peak = False, 0.0
            self._freeze()
        self.t += 1

    def _ensure_camera(self) -> None:
        if self.camera is None:
            fovy = float(self.sc.scene['cameras'][self.cam_name]['fovy'])
            self.camera = Camera.from_env(self.sc.env, self.cam_name, fovy, self.W, self.H)

    def _sets(self) -> list[tuple[str, dict[str, np.ndarray]]]:
        """(class, candidates) of the episode's directed gestures, as the task protocol allows them now."""
        out = []
        if self.code in ('T1', 'T2'):
            out.append(('point', self._pointing_candidates()))
        if self.code == 'T2' or self.t3_handover:
            out.append(('palm_up', self.zone_set))
        if self.code == 'T3':
            out.append(('reach', self.reach_set))
        return [(c, cands) for c, cands in out if cands]

    def _freeze(self) -> None:
        for cls, cands in self._sets():
            key = (cls, tuple(sorted(cands)))
            if key not in self.frozen and self.top == cls:
                b = self._belief(cls, cands)
                if cls == 'reach' and self.reach_best and max(self.reach_best.values()) > max(b.values(), default=0.0):
                    # 2026-10-18: a reach's target shows in the motion (Pérez-D'Arpino & Shah 2015: the first part of
                    # the reach), while at its end the hand is at the cube and the direction hand -> cube is degenerate;
                    # locking the end value left the late rule at 0.5 / 0.5 for ever (T5 on T3, expert check)
                    b = dict(self.reach_best)
                if b:
                    self.frozen[key] = b

    # -- beliefs
    def _belief(self, cls: str, cands: dict[str, np.ndarray]) -> dict[str, float]:
        key = (cls, tuple(sorted(cands)))
        if key in self.frozen:
            return dict(self.frozen[key])
        if not cands or self.probs is None or self.dw is None or self.hand is None:
            return {}
        if cls == 'reach' and self.await_motion:
            # 2026-10-18: after a change-of-mind signal cleared the lock, a reach gives no target evidence until the
            # hand moves again: a hand still at (or holding) the reached object has a degenerate direction hand ->
            # target (class doc), and false withdraw detections while the human holds the new cube still above U
            # made the belief swing back to the old cube (T5 on T3, expert check outputs/trial_t235/2026-10-18)
            return {k: 1.0 / len(cands) for k in cands}
        c = self.classes.index(cls)
        p, sigma = float(self.probs[c]), self.width(cls)
        L = {k: p * np.exp(-0.5 * (angle(self.dw, pos - self.hand) / sigma) ** 2) + (1 - p) for k, pos in cands.items()}
        z = sum(L.values())
        return {k: v / z for k, v in L.items()}

    def width(self, cls: str) -> float:
        """Likelihood width of the class now: by the device frames since it was detected (sigma_deg_by_frames;
        before a detection the first bin), else the single per-class width."""
        table = self.sigma_table.get(cls)
        if table is None or not len(self.sigma_frames):
            return self.sigma[cls]
        tau = max(self.class_run.get(cls, 0) - 1, 0)
        return float(table[np.searchsorted(self.sigma_frames, tau, 'right') - 1])

    def _pointing_candidates(self) -> dict[str, np.ndarray]:
        """The pointing candidates the task protocol allows now (class doc)."""
        objects = not self.holding or self.wait_seen
        zones = self.code == 'T1' and self.holding
        return {k: p for k, p in self.point_set.items()
                if (k in self.object_keys and objects) or (k in self.zone_keys and zones)}

    def complete(self, kind: str) -> bool:
        """The directed gesture behind the `kind` ('object' / 'zone') candidates has finished: the hand is still at the
        gesture's pose while the device detects the gesture, or its decision was locked when the hand stopped (the
        late rule acts only then)."""
        if kind == 'object':
            key = ('reach', self.reach_set) if self.code == 'T3' else ('point', self._pointing_candidates())
        elif self.code == 'T2' or self.t3_handover:
            key = ('palm_up', self.zone_set)
        elif self.code == 'T1':
            key = ('point', self._pointing_candidates())
        else:
            return False
        cls, cands = key
        if not cands:
            return False
        held = not self.moving and self.top == cls and self.class_run.get(cls, 0) > 0   # still at the pose
        return held or (cls, tuple(sorted(cands))) in self.frozen

    def objects(self) -> dict[str, float]:
        if self.code == 'T3':
            return {self.reach_map[k]: v for k, v in self._belief('reach', self.reach_set).items()}
        b = self._belief('point', self._pointing_candidates())
        return {k: b[k] for k in self.object_keys if k in b}

    def zones(self) -> dict[str, float]:
        if self.code == 'T2' or self.t3_handover:
            return self._belief('palm_up', self.zone_set)
        if self.code == 'T1':
            b = self._belief('point', self._pointing_candidates())
            return {k: b[k] for k in self.zone_keys if k in b}
        return {}

    def requests(self) -> dict[str, float]:
        """T3: the belief over the cubes the human reaches for - the human's own goal. Which cup goes with a cube is
        task knowledge the policy learns from the demonstrations (doc 'Hướng triển khai' §2.2, 2026-10-18); the
        expert (a teacher) maps cube -> cup in `objects()`."""
        return self._belief('reach', self.reach_set) if self.code == 'T3' else {}

    def distribution(self) -> dict[str, float]:
        """One distribution over the episode's candidates (T1 the joint pointing set; T2 half objects, half zones;
        T3 the cubes, `requests`)."""
        if self.code == 'T1':
            return self._belief('point', self._pointing_candidates())
        if self.code == 'T3' and not self.t3_handover:
            return self.requests()
        parts = [d for d in ((self.requests() if self.code == 'T3' else self.objects()), self.zones()) if d]
        out: dict[str, float] = {}
        for d in parts:
            tot = sum(d.values()) or 1.0
            for k, v in d.items():
                out[k] = out.get(k, 0.0) + v / tot / len(parts)
        return out

    def _gesture_flag(self, cls: str) -> bool:
        """The rule's per-frame decision for a gesture: its probability >= the rule's threshold (0.95 / 0.7)."""
        if self.probs is None:
            return False
        return bool(self.probs[self.classes.index(cls)] >= self.star)

    def confirmed(self, cls: str) -> bool:
        """The gesture flag has lasted `hold_frames` device frames (calibrated false-trigger rate <= 5 %)."""
        return self.runs.get(cls, 0) >= self.hold.get(cls, 1)

    @property
    def p_stop(self) -> float:
        return 1.0 if self.confirmed('palm_forward') else 0.0

    @property
    def p_withdraw(self) -> float:
        return 1.0 if self.confirmed('withdraw') else 0.0

    @property
    def p_gesture(self) -> np.ndarray:
        return np.full(len(self.classes), np.nan) if self.probs is None else self.probs

    def thresholds(self) -> tuple[float, int, float]:
        """(b_star, n_hold, stop_star) of the belief expert: the rule's commit threshold (late 0.95, early commit_star),
        no hold on the belief; p_stop / p_withdraw are already 0 / 1 decisions (`confirmed`)."""
        return self.commit_star, 1, 0.5

    def snapshot(self) -> dict:
        r = lambda d: {k: round(v, 3) for k, v in d.items()}
        return dict(objects=r(self.objects()), zones=r(self.zones()), p_stop=round(self.p_stop, 3),
                    p_withdraw=round(self.p_withdraw, 3), gesture=self.top,
                    p_gesture=None if self.probs is None else np.round(self.probs, 3).tolist())

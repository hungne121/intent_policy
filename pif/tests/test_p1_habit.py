"""P1 HABIT helpers: gripper debouncing, plane fit and table frame, back-projection, tracking."""
import numpy as np

from pif.data.habit import debounce, gripper_events, runs
from pif.perception.habit_perception import backproject, fit_plane, table_frame, track


def test_runs_and_debounce():
    x = np.array([0, 0, 1, 0, 1, 1, 1, 1, 0, 0], bool)
    assert [(a, b) for a, b, _ in runs(x)] == [(0, 1), (2, 2), (3, 3), (4, 7), (8, 9)]
    d = debounce(x, 3)                                  # the 1-frame close / open flicker disappears
    assert d.astype(int).tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 0, 0]
    assert debounce(np.array([0, 0, 0, 1], bool), 3).astype(int).tolist() == [0, 0, 0, 1]   # last run kept


def test_gripper_events_threshold():
    ep = dict(robot0_grip=np.r_[np.zeros(5), 0.4 * np.ones(6), np.zeros(5)], robot1_grip=np.zeros(16),
              robot0_tcp=np.zeros((16, 3)), robot1_tcp=np.zeros((16, 3)))
    assert gripper_events(ep, threshold=0.5) == []      # the soft bread grasp never exceeds 0.5
    ev = gripper_events(ep, threshold=0.25, min_frames=3)
    assert [(e['arm'], e['frame'], e['kind']) for e in ev] == [('robot0', 5, 'close'), ('robot0', 11, 'open')]


def camera_looking_at_table(rng, n=3000):
    """Points of a table plane 0.5 m below a camera tilted 40 deg down (camera: x right, y down, z forward)."""
    tilt = np.radians(40)
    R = np.array([[1, 0, 0], [0, np.cos(tilt), -np.sin(tilt)], [0, np.sin(tilt), np.cos(tilt)]])   # world -> cam
    world = np.c_[rng.uniform(-0.5, 0.5, n), rng.uniform(0.2, 1.2, n), np.full(n, -0.5)]  # world z up, y forward
    to_cam = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])                                  # world axes in camera axes
    return (R @ to_cam @ world.T).T


def test_plane_fit_and_table_frame():
    rng = np.random.default_rng(0)
    pts = camera_looking_at_table(rng)
    pts = np.r_[pts + rng.normal(0, 0.002, pts.shape), rng.uniform(-1, 1, (300, 3)) + [0, 0, 2]]   # + clutter
    n, d, frac = fit_plane(pts, iters=300, inlier=0.01, rng=rng)
    assert frac > 0.85 and abs(d - 0.5) < 0.01          # camera 0.5 m above the table
    R, o = table_frame(n, d)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(R), 1.0)
    on_table = (pts[:3000] - o) @ R.T
    assert np.abs(on_table[:, 2]).max() < 0.02           # table points at z = 0
    cam = (np.zeros(3) - o) @ R.T
    assert np.allclose(cam[:2], 0, atol=1e-9) and cam[2] > 0   # origin under the camera, camera above
    ahead = (np.array([0, 0, 1.0]) - o) @ R.T            # a point on the optical axis lies along +x
    assert ahead[0] > 0 and abs(ahead[1]) < 1e-9


def test_backproject_roundtrip():
    f, cx, cy = 500.0, 320.0, 240.0
    p = np.array([0.1, -0.05, 0.8])
    u, v = f * p[0] / p[2] + cx, f * p[1] / p[2] + cy
    assert np.allclose(backproject(u, v, p[2], f, cx, cy), p)


def test_track_keeps_identity_and_label():
    dets = [[dict(label='box', score=0.9, xyz=np.array([0.0, 0, 0])), dict(label='box', score=0.8, xyz=np.array([0.3, 0, 0]))],
            [dict(label='box', score=0.9, xyz=np.array([0.31, 0, 0])), dict(label='box', score=0.8, xyz=np.array([0.01, 0, 0]))],
            [dict(label='bread', score=0.9, xyz=np.array([0.0, 0, 0]))]]
    tr = track(dets, gate=0.1)
    assert [t['label'] for t in tr] == ['box', 'box', 'bread']
    assert sorted(tr[0]['frames']) == [0, 1] and np.allclose(tr[0]['frames'][1]['xyz'], [0.01, 0, 0])


def test_movement_onset_from_baseline():
    from pif.data.onset import movements, smoothed_speed
    rng = np.random.default_rng(0)
    T, fps = 120, 10.0
    x = np.zeros((T, 3))
    x[40:60, 0] = np.linspace(0, 0.4, 20)               # a 2 s reach of 0.4 m, then hold
    x[60:] = x[59]
    x += rng.normal(0, 0.002, x.shape)                  # 2 mm keypoint noise
    s = smoothed_speed(x, fps)
    runs, thr = movements(s, slice(0, 30))
    assert len(runs) == 1
    onset, offset = runs[0]
    assert 37 <= onset <= 42 and 57 <= offset <= 62

"""scripts/record_real.py pure parts (no camera / MediaPipe needed)."""
import numpy as np

from scripts.record_real import KEYPOINTS, deproject, hands_to_keypoints, noise_stats


def test_deprojection_and_hand_mapping():
    assert np.allclose(deproject(320, 240, 1.0, 600, 600, 320, 240), [0, 0, 1])
    assert np.allclose(deproject(380, 240, 2.0, 600, 600, 320, 240), [0.2, 0, 2])
    depth = np.full((480, 640), 1000, np.uint16)
    lm = [(320.0, 240.0)] * 21
    kp, valid = hands_to_keypoints([dict(label='Right', landmarks=lm)], depth, (600, 600, 320, 240), 0.001, mirrored=False)
    assert valid[:3].all() and not valid[3:].any() and np.allclose(kp[KEYPOINTS.index('r_index_tip')], [0, 0, 1])
    _, v2 = hands_to_keypoints([dict(label='Right', landmarks=lm)], depth, (600, 600, 320, 240), 0.001, mirrored=True)
    assert v2[3:].all() and not v2[:3].any()


def test_noise_stats_recover_known_noise():
    rng = np.random.default_rng(0)
    T = 600
    base = np.array([[0.4, 0.0, 0.8], [0.45, 0.0, 0.8], [0.5, 0.0, 0.8]] * 2)
    kp = base[None] + rng.normal(0, 0.004, (T, 6, 3))
    valid = rng.random((T, 6)) > 0.05
    st = noise_stats(np.arange(T) / 30, kp, valid, np.full(T, 0.02), point_at=np.array([1.0, 0.0, 0.8]), still_speed=1.0)
    assert abs(st['dropout_prob'] - 0.05) < 0.02 and np.isclose(st['latency_s'], 0.02)
    assert 0.002 < st['jitter_m'] < 0.005 and st['angle_deg'] < 15

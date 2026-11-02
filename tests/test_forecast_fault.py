import numpy as np
from intent_policy.intent.forecast_fault import ForecastFault, corrupt_forecast


def test_fault_changes_only_selected_arm_and_preserves_bones():
    rng = np.random.default_rng(4)
    now = rng.normal(size=(5, 6, 3)).astype('float32')
    xi = rng.normal(scale=.05, size=(5, 8, 6, 3)).astype('float32')
    original, original_now = xi.copy(), now.copy()
    tt, h = np.arange(5), np.arange(1, 9) / 8
    fault = ForecastFault(1, 3, 90, 'right', .2)
    out = corrupt_forecast(xi, now, tt, h, fault)
    np.testing.assert_array_equal(xi, original)
    np.testing.assert_array_equal(now, original_now)
    np.testing.assert_array_equal(out[[0, 1, 3, 4]], xi[[0, 1, 3, 4]])
    np.testing.assert_allclose(out[:, :, [0, 1, 2, 4]], xi[:, :, [0, 1, 2, 4]], atol=1e-7)
    a, b = xi + now[:, None], out + now[:, None]
    for p, q in [(1, 3), (3, 5)]:
        np.testing.assert_allclose(np.linalg.norm(a[:, :, p] - a[:, :, q], axis=-1),
                                   np.linalg.norm(b[:, :, p] - b[:, :, q], axis=-1), atol=5e-7)
    assert np.linalg.norm(out[2, -1, 5] - xi[2, -1, 5]) > .1
    online = np.concatenate([corrupt_forecast(xi[i:i+1], now[i:i+1], tt[i:i+1], h, fault) for i in range(5)])
    np.testing.assert_array_equal(online, out)

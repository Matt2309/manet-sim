"""Block 4 tests: per-link Kalman and neighbour table.

Each test checks a known fact (theory or closed formula); statistical tolerances at 4 sigma.
"""

import math

import numpy as np
import pytest
from scipy.linalg import solve_discrete_are

from src.kalman import filter_all, filter_link, measurement_noise, predict, predict_links, process_noise, transition


# ---------------------------------------------------------------------------
# 1. Noiseless line
# ---------------------------------------------------------------------------


def test_noiseless_line_slope_converges(cfg):
    """With ``z = a + b t`` and regular `Δt` the estimated slope converges to `b`."""
    kc = cfg["kalman"]
    a, b, dt = -70.0, -0.8, 0.5
    times = np.arange(200) * dt
    z = a + b * times
    r, s, P, nis, reinit = filter_link(times, z, 1.0, kc["sigma_a"], kc["initial_slope_std"], kc["reinit_gap"])
    assert s[-1] == pytest.approx(b, abs=1e-3)
    assert r[-1] == pytest.approx(a + b * times[-1], abs=1e-3)
    assert reinit[0] and not reinit[1:].any()
    assert np.isnan(nis[0])
    # irregular Δt
    rng = np.random.default_rng(0)
    times = np.cumsum(rng.uniform(0.3, 0.7, 300))
    r, s, *_ = filter_link(times, a + b * times, 1.0, kc["sigma_a"], kc["initial_slope_std"], kc["reinit_gap"])
    assert s[-1] == pytest.approx(b, abs=1e-3)


# ---------------------------------------------------------------------------
# 2. Steady-state gain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sigma_a", [0.25, 1.0])
def test_steady_state_covariance_matches_dare(sigma_a):
    """With constant `Δt` the steady-state prior covariance solves the discrete Riccati equation."""
    dt, R = 0.5, 11.0
    rng = np.random.default_rng(1)
    times = np.arange(400) * dt
    z = rng.normal(-70.0, math.sqrt(R), len(times))
    r, s, P, *_ = filter_link(times, z, R, sigma_a, 3.0, 10.0)
    F = transition(dt)
    H = np.array([[1.0, 0.0]])
    Q = process_noise(dt, sigma_a)
    expected = solve_discrete_are(F.T, H.T, Q, np.array([[R]]))
    prior = F @ P[-1] @ F.T + Q
    np.testing.assert_allclose(prior, expected, rtol=1e-7)
    # steady-state gain
    gain = expected[:, 0] / (expected[0, 0] + R)
    p_post = (np.eye(2) - np.outer(gain, [1.0, 0.0])) @ expected
    np.testing.assert_allclose(P[-1], p_post, rtol=1e-7)


# ---------------------------------------------------------------------------
# 3. Additivity of Q and growth during gaps
# ---------------------------------------------------------------------------


def test_process_noise_additive():
    """Two predictions of Δt/2 are equivalent to one of Δt (same P, same x)."""
    rng = np.random.default_rng(2)
    a = rng.normal(size=(2, 2))
    P = a @ a.T + np.eye(2)
    x = rng.normal(size=2)
    for dt in (0.3, 1.7, 6.0):
        x1, P1 = predict(x, P, dt, 0.8)
        xh, Ph = predict(x, P, dt / 2.0, 0.8)
        x2, P2 = predict(xh, Ph, dt / 2.0, 0.8)
        np.testing.assert_allclose(P2, P1, rtol=1e-12)
        np.testing.assert_allclose(x2, x1, rtol=1e-12)
    # Q(dt) = ∫ Φ(u) G Gᵀ Φ(u)ᵀ σ² du by quadrature
    dt, sigma_a = 1.3, 0.8
    u = np.linspace(0.0, dt, 200001)
    g = np.stack([dt - u, np.ones_like(u)], axis=-1)
    outer = g[:, :, None] * g[:, None, :]
    q_num = sigma_a**2 * np.trapezoid(outer, u, axis=0)
    np.testing.assert_allclose(process_noise(dt, sigma_a), q_num, rtol=1e-8)


def test_variance_grows_strictly_during_gaps(cfg):
    kc = cfg["kalman"]
    n_nodes, n_beacons = 2, 60
    times = np.full((n_nodes, n_beacons), np.nan)
    times[0] = np.arange(n_beacons) * 0.5
    received = np.zeros((n_nodes, n_beacons, n_nodes), dtype=bool)
    received[0, :, 1] = True
    received[0, 20:40, 1] = False  # 10 s gap
    rssi = np.full((n_nodes, n_beacons, n_nodes), np.nan)
    rssi[0, :, 1] = -70.0 + np.random.default_rng(3).normal(0.0, 3.0, n_beacons)
    kres = filter_all(times, received, rssi, 10.0, kc["sigma_a"], kc)
    t = np.arange(0.0, 29.9, 0.1)
    est = predict_links(kres, t)
    p00 = est["p00"][:, 0, 1]
    in_gap = (t > 9.5) & (t < 20.0)
    assert np.all(np.diff(p00[in_gap]) > 0.0)
    # prediction on the grid = explicit prediction from the last update
    k = 19
    k_idx = int(np.searchsorted(t, 12.0))
    _, p_exp = predict(
        np.array([kres.r[0, k, 1], kres.s[0, k, 1]]), kres.P[0, k, 1], t[k_idx] - times[0, k], kc["sigma_a"]
    )
    assert est["p00"][k_idx, 0, 1] == pytest.approx(p_exp[0, 0], rel=1e-12)
    assert np.isnan(est["r"][:, 0, 0]).all() and np.isnan(est["r"][:, 1, 0]).all()


def test_reinit_after_long_silence(cfg):
    kc = cfg["kalman"]
    times = np.concatenate([np.arange(20) * 0.5, 10.0 + kc["reinit_gap"] + 1.0 + np.arange(20) * 0.5])
    z = np.full(len(times), -70.0)
    z[20:] = -90.0
    r, s, P, nis, reinit = filter_link(times, z, 8.0, kc["sigma_a"], kc["initial_slope_std"], kc["reinit_gap"])
    assert reinit[0] and reinit[20] and reinit.sum() == 2
    assert r[20] == -90.0 and s[20] == 0.0
    np.testing.assert_allclose(P[20], np.diag([8.0, kc["initial_slope_std"] ** 2]))
    assert np.isnan(nis[20])


# ---------------------------------------------------------------------------
# 4. Consistency
# ---------------------------------------------------------------------------


def test_consistency_constant_level_white_noise():
    """Constant level and white noise (σ_a = 0): mean NIS 1 and error variance equal to P."""
    R, level, n_real, n_steps = 9.0, -72.0, 1500, 60
    rng = np.random.default_rng(4)
    times = np.arange(n_steps) * 0.5
    nis_all, err_r, err_s = [], [], []
    for _ in range(n_real):
        z = level + rng.normal(0.0, math.sqrt(R), n_steps)
        r, s, P, nis, _ = filter_link(times, z, R, 0.0, 3.0, 10.0)
        nis_all.append(nis[1:])
        err_r.append(r[-1] - level)
        err_s.append(s[-1])
    nis_all = np.concatenate(nis_all)
    n = len(nis_all)
    assert nis_all.mean() == pytest.approx(1.0, abs=4.0 * math.sqrt(2.0 / n))
    tol = 4.0 * math.sqrt(2.0 / n_real)
    assert np.var(err_r) / P[-1, 0, 0] == pytest.approx(1.0, abs=tol)
    assert np.var(err_s) / P[-1, 1, 1] == pytest.approx(1.0, abs=tol)


def test_consistency_random_acceleration_irregular_dt():
    """With truth drawn from the model itself (σ_a > 0, irregular Δt) mean NIS 1 and error covariance equal to P."""
    R, sigma_a, slope_std = 6.0, 0.7, 3.0
    n_real, n_steps = 1500, 50
    rng = np.random.default_rng(5)
    dts = rng.uniform(0.3, 0.7, n_steps - 1)
    dts[rng.random(n_steps - 1) < 0.2] += rng.uniform(1.0, 3.0)  # gaps (< reinit_gap)
    times = np.concatenate([[0.0], np.cumsum(dts)])
    chol = [np.linalg.cholesky(process_noise(d, sigma_a)) for d in dts]
    nis_all, err = [], []
    for _ in range(n_real):
        x = np.array([0.0, rng.normal(0.0, slope_std)])
        xs = [x]
        for d, c in zip(dts, chol):
            x = transition(d) @ x + c @ rng.standard_normal(2)
            xs.append(x)
        xs = np.array(xs)
        z = xs[:, 0] + rng.normal(0.0, math.sqrt(R), n_steps)
        r, s, P, nis, _ = filter_link(times, z, R, sigma_a, slope_std, 10.0)
        nis_all.append(nis[1:])
        err.append([r[-1] - xs[-1, 0], s[-1] - xs[-1, 1]])
    nis_all = np.concatenate(nis_all)
    err = np.array(err)
    n = len(nis_all)
    assert nis_all.mean() == pytest.approx(1.0, abs=4.0 * math.sqrt(2.0 / n))
    tol = 4.0 * math.sqrt(2.0 / n_real)
    assert np.var(err[:, 0]) / P[-1, 0, 0] == pytest.approx(1.0, abs=tol)
    assert np.var(err[:, 1]) / P[-1, 1, 1] == pytest.approx(1.0, abs=tol)
    assert np.mean(err[:, 0]) == pytest.approx(0.0, abs=4.0 * math.sqrt(P[-1, 0, 0] / n_real))


def test_measurement_noise_auto_and_explicit(cfg):
    q = cfg["channel"]["measurement"]["quantization_step"]
    assert measurement_noise(cfg, 10.0) == pytest.approx(10.0 + q**2 / 12.0)
    cfg["kalman"]["measurement_noise"] = 7.5
    assert measurement_noise(cfg, 10.0) == 7.5
    cfg["kalman"]["measurement_noise"] = "other"
    with pytest.raises(ValueError):
        measurement_noise(cfg, 10.0)


# ---------------------------------------------------------------------------
# 5. 5-byte table
# ---------------------------------------------------------------------------


def test_table_roundtrip_half_step_and_saturation(cfg):
    from src.detector import decode_table, encode_table

    tc = cfg["neighbour_table"]
    rng = np.random.default_rng(6)
    n = 5000
    r = rng.uniform(-127.9, 126.9, n)
    s = rng.uniform(-12.7, 12.7, n)
    std = rng.uniform(0.0, 25.5, n)
    age = rng.uniform(0.0, 25.5, n)
    codes = encode_table(r, s, std, age, tc)
    back = decode_table(codes, tc)
    assert np.all(np.abs(back["r"] - r) <= tc["rssi_step"] / 2 + 1e-9)
    assert np.all(np.abs(back["s"] - s) <= tc["slope_step"] / 2 + 1e-9)
    assert np.all(np.abs(back["std"] - std) <= tc["std_step"] / 2 + 1e-9)
    assert np.all(np.abs(back["age"] - age) <= tc["age_step"] / 2 + 1e-9)
    # 4 data bytes + 1 id byte
    assert [codes[k].dtype for k in ("r", "s", "std", "age")] == [np.int8, np.int8, np.uint8, np.uint8]
    assert sum(codes[k].dtype.itemsize for k in codes) + 1 == 5

    # saturation outside the range
    sat = decode_table(encode_table(np.array([-300.0, 300.0]), np.array([-50.0, 50.0]),
                                    np.array([-1.0, 99.0]), np.array([-3.0, 99.0]), tc), tc)
    assert sat["r"].tolist() == [-128.0 * tc["rssi_step"], 127.0 * tc["rssi_step"]]
    assert sat["s"].tolist() == pytest.approx([-12.7, 12.7])
    assert sat["std"].tolist() == pytest.approx([0.0, 25.5])
    assert sat["age"].tolist() == pytest.approx([0.0, 25.5])

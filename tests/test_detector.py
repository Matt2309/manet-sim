"""Block 4 tests: forwarding, fusion, detector, metrics and cache.

Each test checks a known fact (theory, closed formula or hand calculation); statistical tolerances at 4 sigma.
"""

import copy
import math

import numpy as np
import pytest

from src.common.paths import DEFAULT_CONFIG as CONFIG_PATH
from tests.helpers import manual_channel, static_mobility
from src.refinement import block4_config
from src.detector import (
    build_inputs,
    combined_alarm,
    count_episodes,
    fuse,
    fusion_indices,
    latched_alarm,
    lost_alarm,
    lost_system_alarm,
    neighbour_knowledge,
    pre_alarm,
    reception_features,
    run_duration,
    system_alarm,
)
from src.kalman import filter_all, predict_links
from src.metrics import (
    aggregate,
    nearest_rank,
    poisson_interval,
    summarize_none,
    summarize_separated,
    threshold_grid,
    wilson_interval,
    working_point,
)
from src.mobility import load_config
from src.packets import beacon_schedule, forward_link_payload, simulate_packets
from src.simulator import (
    cache_key,
    draw_separations,
    load_or_simulate,
    run_seeds,
    scenario_config,
    simulate_run,
)


@pytest.fixture(scope="session")
def base_config():
    """Config with the Block 4 values set explicitly (`history.block4`): these tests were written for those defaults."""
    return block4_config(load_config(CONFIG_PATH))


# ---------------------------------------------------------------------------
# Frozen copy of the Block 3 forwarding (commit f4333e3), for test 6
# ---------------------------------------------------------------------------


def _old_neighbour_tables(times, received, rssi):
    n_nodes, n_beacons = times.shape
    table_rssi = np.full((n_nodes, n_beacons, n_nodes), np.nan)
    table_origin = np.full((n_nodes, n_beacons, n_nodes), np.nan)
    for j in range(n_nodes):
        valid_k = np.flatnonzero(~np.isnan(times[j]))
        for i in range(n_nodes):
            if i == j:
                continue
            got = np.flatnonzero(received[i, :, j])
            if len(got) == 0:
                continue
            pos = np.searchsorted(times[i, got], times[j, valid_k], side="left") - 1
            ok = pos >= 0
            src = got[pos[ok]]
            table_rssi[j, valid_k[ok], i] = rssi[i, src, j]
            table_origin[j, valid_k[ok], i] = times[i, src]
    return table_rssi, table_origin


def _old_knowledge(t, times, received, rssi):
    n_steps = len(t)
    n_nodes = times.shape[0]
    table_rssi, table_origin = _old_neighbour_tables(times, received, rssi)
    k_rssi = np.full((n_steps, n_nodes, n_nodes, n_nodes), np.nan, dtype=np.float32)
    k_age = np.full_like(k_rssi, np.nan)
    for m in range(n_nodes):
        for j in range(n_nodes):
            if j == m:
                continue
            got = np.flatnonzero(received[j, :, m])
            if len(got) == 0:
                continue
            pos = np.searchsorted(times[j, got], t, side="right") - 1
            ok = pos >= 0
            k = got[pos[ok]]
            origin = table_origin[j, k, :]
            k_rssi[ok, m, :, j] = table_rssi[j, k, :]
            k_age[ok, m, :, j] = t[ok, None] - origin
        for i in range(n_nodes):
            if i == m:
                continue
            got = np.flatnonzero(received[i, :, m])
            if len(got) == 0:
                continue
            pos = np.searchsorted(times[i, got], t, side="right") - 1
            ok = pos >= 0
            k = got[pos[ok]]
            k_rssi[ok, m, i, m] = rssi[i, k, m]
            k_age[ok, m, i, m] = t[ok] - times[i, k]
    idx = np.arange(n_nodes)
    k_rssi[:, :, idx, idx] = np.nan
    k_age[:, :, idx, idx] = np.nan
    return k_rssi, k_age


def _random_beacons(cfg, n_nodes=5, t_end=200.0, loss=0.3, seed=0):
    rng = np.random.default_rng(seed)
    dt = cfg["simulation"]["dt"]
    times, steps = beacon_schedule(n_nodes, t_end, dt, cfg["packets"], rng)
    received = rng.random((n_nodes, times.shape[1], n_nodes)) > loss
    received &= (~np.isnan(times))[:, :, None] & ~np.eye(n_nodes, dtype=bool)[:, None, :]
    rssi = np.where(received, -60.0 + rng.normal(0.0, 6.0, received.shape).round(), np.nan)
    t = np.arange(int(t_end / dt) + 1) * dt
    return t, times, received, rssi


# ---------------------------------------------------------------------------
# 6. Forwarding refactor
# ---------------------------------------------------------------------------


def test_forwarding_refactor_matches_block3(cfg):
    t, times, received, rssi = _random_beacons(cfg)
    old_rssi, old_age = _old_knowledge(t, times, received, rssi)
    fields, age, relay = forward_link_payload(t, times, received, {"rssi": rssi})
    np.testing.assert_array_equal(fields["rssi"], old_rssi)  # exact equality, NaNs included
    np.testing.assert_array_equal(age, old_age)
    assert np.isfinite(old_rssi).any() and np.isnan(old_rssi).any()

    # a second field travels along with the first, without interfering
    fields2, age2, _ = forward_link_payload(t, times, received, {"a": rssi, "b": 2.0 * rssi})
    np.testing.assert_array_equal(fields2["a"], old_rssi)
    np.testing.assert_allclose(fields2["b"], 2.0 * fields2["a"])
    np.testing.assert_array_equal(age2, old_age)

    # forwarding time: missing on the direct link, on the others it precedes t and follows the origin
    n = times.shape[0]
    direct = np.zeros((n, n, n), dtype=bool)  # [m, i, j]
    for m in range(n):
        direct[m, :, m] = True
    sel = np.isfinite(relay)
    assert not sel[:, direct].any()
    s_idx, m_idx, i_idx, j_idx = np.nonzero(sel)
    origin = t[s_idx] - age[s_idx, m_idx, i_idx, j_idx].astype(float)
    assert np.all(relay[sel] <= t[s_idx] + 1e-9) and np.all(relay[sel] >= origin - 1e-4)


def test_forwarding_matches_simulate_packets(cfg):
    n, dt, n_steps = 5, cfg["simulation"]["dt"], 3000
    rng = np.random.default_rng(1)
    rssi_true = -96.0 + rng.normal(0.0, 3.0, size=(n_steps, n, n))
    ch = manual_channel(rssi_true, dt)
    res = simulate_packets(static_mobility(n_steps, n, dt), ch, cfg)
    fields, age, _ = forward_link_payload(ch.t, res.beacon_times, res.received, {"rssi": res.rssi})
    np.testing.assert_array_equal(fields["rssi"], res.knowledge_rssi)
    np.testing.assert_array_equal(age, res.knowledge_age)


# ---------------------------------------------------------------------------
# 7. Fusion
# ---------------------------------------------------------------------------


def test_fuse_equal_variances_is_simple_mean():
    values = np.array([[-70.0, -80.0, -75.0]])
    slopes = np.array([[-1.0, 0.0, -2.0]])
    level, slope, var, n = fuse(values, slopes, np.full((1, 3), 4.0), np.zeros((1, 3)), 2.0)
    assert level[0] == pytest.approx(-75.0)
    assert slope[0] == pytest.approx(-1.0)
    assert var[0] == pytest.approx(4.0 / 3.0)
    assert n[0] == 3


def test_fuse_inverse_variance_weights_and_huge_variance_ignored():
    values = np.array([-70.0, -80.0])
    slopes = np.array([-1.0, -3.0])
    level, slope, var, _ = fuse(values, slopes, np.array([1.0, 4.0]), np.zeros(2), 2.0)
    # weights 1 and 1/4: (-70 - 20) / 1.25 = -72; slope same weights
    assert level == pytest.approx((-70.0 - 80.0 / 4.0) / 1.25)
    assert slope == pytest.approx((-1.0 - 3.0 / 4.0) / 1.25)
    assert var == pytest.approx(0.8)
    level, slope, var, _ = fuse(values, slopes, np.array([1.0, 1e15]), np.zeros(2), 2.0)
    assert level == pytest.approx(-70.0, abs=1e-9) and slope == pytest.approx(-1.0, abs=1e-9)


def test_fuse_max_age_excludes_old_entries():
    values = np.array([-70.0, -90.0, -50.0])
    ages = np.array([0.5, 2.0, 2.0 + 1e-6])
    level, _, _, n = fuse(values, np.zeros(3), np.ones(3), ages, 2.0)
    assert n == 2 and level == pytest.approx(-80.0)  # age == max_age still valid
    level, slope, var, n = fuse(values, np.zeros(3), np.ones(3), np.full(3, 5.0), 2.0)
    assert n == 0 and np.isnan(level) and np.isnan(slope) and np.isnan(var)
    # NaN in an entry: excluded
    level, _, _, n = fuse(np.array([-70.0, np.nan]), np.zeros(2), np.ones(2), np.zeros(2), 2.0)
    assert n == 1 and level == -70.0


def _steady_network(cfg, own_level, other_level, n_beacons=120, stop_own_after=None):
    """3 nodes with constant per-link levels; the target is node 1, the observer is node 0."""
    n = 3
    period = cfg["packets"]["beacon_period"]
    times = np.tile(np.arange(n_beacons) * period, (n, 1)) + np.arange(n)[:, None] * 0.01
    received = np.ones((n, n_beacons, n), dtype=bool) & ~np.eye(n, dtype=bool)[:, None, :]
    level = np.full((n, n), -60.0)
    level[1, 0] = own_level  # link 1 → 0 (own estimate of observer 0)
    level[1, 2] = other_level  # link 1 → 2 (arrives with the table of node 2)
    level[0, 1] = level[2, 1] = level[0, 2] = level[2, 0] = -60.0
    rssi = np.where(received, level[:, None, :], np.nan)
    if stop_own_after is not None:
        received[1, int(stop_own_after / period) :, 0] = False
        rssi = np.where(received, level[:, None, :], np.nan)
    t = np.arange(0.0, n_beacons * period, 0.1)
    return t, times, received, rssi


def test_fusion_weights_and_pairwise_uses_only_own_estimate(cfg):
    t, times, received, rssi = _steady_network(cfg, own_level=-70.0, other_level=-80.0)
    R = 4.0
    kres = filter_all(times, received, rssi, R, cfg["kalman"]["sigma_a"], cfg["kalman"])
    out = fusion_indices(t, kres, cfg)
    m, i, k = 0, 1, 2
    idx = int(np.searchsorted(t, 40.0))

    own = predict_links(kres, t)
    know = neighbour_knowledge(t, kres, cfg["neighbour_table"])
    sigma_a = kres.sigma_a
    age_f = float(know["age"][idx, m, i, k])
    var_own = float(own["p00"][idx, i, m])
    var_f = float(know["std"][idx, m, i, k]) ** 2 + sigma_a**2 * age_f**3 / 3.0
    r_own = float(own["r"][idx, i, m])
    r_f = float(know["r"][idx, m, i, k]) + float(know["s"][idx, m, i, k]) * age_f
    expected = (r_own / var_own + r_f / var_f) / (1.0 / var_own + 1.0 / var_f)
    level_f, slope_f, var_fused, n_f = out["fused"]
    assert n_f[idx, m, i] == 2
    assert level_f[idx, m, i] == pytest.approx(expected, rel=1e-9)
    assert var_fused[idx, m, i] == pytest.approx(1.0 / (1.0 / var_own + 1.0 / var_f), rel=1e-9)
    assert -80.0 < level_f[idx, m, i] < -70.0  # a mean, not one of the two entries

    level_p, _, _, n_p = out["pairwise"]
    assert n_p[idx, m, i] == 1
    assert level_p[idx, m, i] == pytest.approx(r_own, rel=1e-12)
    assert level_p[idx, m, i] == pytest.approx(-70.0, abs=1e-6)
    # the fused level of link 1→2 is not used by pairwise even though the value is different
    assert abs(level_f[idx, m, i] - level_p[idx, m, i]) > 1e-3
    # diagonal: observer = target, no index
    assert np.isnan(level_f[:, 1, 1]).all() and np.isnan(level_p[:, 0, 0]).all()


def test_fusion_max_age_applies_to_own_estimate_by_option(cfg):
    t, times, received, rssi = _steady_network(cfg, -70.0, -80.0, stop_own_after=30.0)
    kres = filter_all(times, received, rssi, 4.0, cfg["kalman"]["sigma_a"], cfg["kalman"])
    idx_late = int(np.searchsorted(t, 50.0))  # the direct link 1→0 has been silent for 20 s: own estimate too old
    out = fusion_indices(t, kres, cfg)
    assert np.isnan(out["pairwise"][0][idx_late, 0, 1])  # only the own estimate, excluded
    assert out["fused"][3][idx_late, 0, 1] == 1  # the entry forwarded by node 2 remains
    assert out["fused"][0][idx_late, 0, 1] == pytest.approx(-80.0, abs=0.2)
    cfg2 = copy.deepcopy(cfg)
    cfg2["fusion"]["max_age_own"] = False
    out2 = fusion_indices(t, kres, cfg2)
    assert out2["pairwise"][0][idx_late, 0, 1] == pytest.approx(-70.0, abs=1e-6)  # own estimate always used
    assert out2["fused"][3][idx_late, 0, 1] == 2
    # with the variance grown, the fusion shifts toward the forwarded entry
    assert out2["fused"][0][idx_late, 0, 1] < -75.0


# ---------------------------------------------------------------------------
# 8. Detector on synthetic series
# ---------------------------------------------------------------------------

DT = 0.1


def _time(duration=80.0):
    return np.arange(0.0, duration, DT)


def _first_true(state, t):
    return float(t[np.argmax(state)]) if state.any() else None


def _first_false_after(state, t, start):
    k = int(round(start / DT))
    off = np.flatnonzero(~state[k:])
    return float(t[k + off[0]]) if len(off) else None


def test_run_duration_and_latch():
    t = _time(10.0)
    cond = np.zeros(len(t), dtype=bool)
    cond[20:35] = True
    cond[50:52] = True
    d = run_duration(cond, t)
    assert d[20] == 0.0 and d[34] == pytest.approx(1.4) and d[35] == 0.0 and d[51] == pytest.approx(0.1)
    on = np.zeros(10, dtype=bool)
    off = np.zeros(10, dtype=bool)
    on[[1, 6]] = True
    off[[4, 8]] = True
    state = latched_alarm(on, off)
    assert state.tolist() == [False, True, True, True, False, False, True, True, False, False]
    both = latched_alarm(np.array([False, True]), np.array([False, True]))
    assert both.tolist() == [False, False]  # at the same index turning off wins


def test_pre_alarm_on_after_min_duration_and_off_after_release_time(cfg):
    d = cfg["detector"]
    t = _time()
    level = np.full(len(t), -60.0)
    level[(t >= 20.0) & (t < 30.0)] = -90.0
    slope = np.zeros(len(t))
    rx_ok = np.ones(len(t), dtype=bool)
    state = pre_alarm(level, slope, rx_ok, t, d, mode="level")
    assert _first_true(state, t) == pytest.approx(20.0 + d["min_duration"])
    # back since 30 s: it turns off after release_time of return, not before
    assert _first_false_after(state, t, 21.0) == pytest.approx(30.0 + d["release_time"])
    assert state[(t >= 21.0) & (t < 34.95)].all()
    assert not state[t < 20.95].any()


def test_pre_alarm_shorter_than_min_duration_does_not_fire(cfg):
    d = cfg["detector"]
    t = _time(30.0)
    level = np.full(len(t), -60.0)
    level[(t >= 10.0) & (t < 10.0 + d["min_duration"] - 0.15)] = -95.0
    assert not pre_alarm(level, np.zeros(len(t)), np.ones(len(t), dtype=bool), t, d, mode="level").any()


def test_pre_alarm_release_needs_good_reception(cfg):
    d = cfg["detector"]
    t = _time()
    level = np.full(len(t), -60.0)
    level[(t >= 20.0) & (t < 30.0)] = -90.0
    rx_ok = np.ones(len(t), dtype=bool)
    rx_ok[(t >= 25.0) & (t < 42.0)] = False  # index back but reception poor
    state = pre_alarm(level, np.zeros(len(t)), rx_ok, t, d, mode="level")
    assert state[(t >= 21.0) & (t < 42.0)].all()  # stays on: reception not good
    assert _first_false_after(state, t, 21.0) == pytest.approx(42.0)  # as soon as reception is good again


def test_pre_alarm_hysteresis_and_margin(cfg):
    d = cfg["detector"]
    t = _time()
    level = np.full(len(t), -60.0)
    level[(t >= 20.0) & (t < 30.0)] = -90.0
    # brief return (3 s < release_time) then down again: no turning off
    level[(t >= 33.0) & (t < 45.0)] = -90.0
    rx_ok = np.ones(len(t), dtype=bool)
    state = pre_alarm(level, np.zeros(len(t)), rx_ok, t, d, mode="level")
    assert state[(t >= 21.0) & (t < 50.0)].all()
    # return above the threshold but within the margin (L between threshold and threshold+margin): not enough
    level2 = np.full(len(t), -60.0)
    level2[(t >= 20.0) & (t < 30.0)] = -90.0
    level2[t >= 30.0] = d["level_threshold"] + d["release_level_margin"] / 2.0
    state2 = pre_alarm(level2, np.zeros(len(t)), rx_ok, t, d, mode="level")
    assert state2[t >= 21.0].all()
    # NaN (no valid entry) while on: stays on
    level3 = level.copy()
    level3[(t >= 33.0) & (t < 60.0)] = np.nan
    assert pre_alarm(level3, np.zeros(len(t)), rx_ok, t, d, mode="level")[(t >= 21.0) & (t < 60.0)].all()


def test_pre_alarm_modes(cfg):
    d = cfg["detector"]
    t = _time(60.0)
    rx_ok = np.ones(len(t), dtype=bool)
    level = np.full(len(t), -60.0)
    slope = np.zeros(len(t))
    level[(t >= 10.0) & (t < 20.0)] = -90.0  # low level, zero slope
    slope[(t >= 30.0) & (t < 40.0)] = -2.0  # negative slope, high level
    level[(t >= 45.0) & (t < 55.0)] = -90.0  # both
    slope[(t >= 45.0) & (t < 55.0)] = -2.0
    on = {m: pre_alarm(level, slope, rx_ok, t, d, mode=m) for m in ("level", "slope", "both")}
    assert on["level"][int(12 / DT)] and not on["level"][int(33 / DT)] and on["level"][int(47 / DT)]
    assert not on["slope"][int(12 / DT)] and on["slope"][int(33 / DT)] and on["slope"][int(47 / DT)]
    assert not on["both"][int(12 / DT)] and not on["both"][int(33 / DT)] and on["both"][int(47 / DT)]
    with pytest.raises(ValueError):
        pre_alarm(level, slope, rx_ok, t, d, mode="other")


def _beacons(rx_windows, period=0.5, t_end=120.0):
    """Beacons of node 0 received by node 1 only in the `rx_windows` intervals."""
    times = np.full((2, int(t_end / period)), np.nan)
    times[0] = np.arange(times.shape[1]) * period
    received = np.zeros((2, times.shape[1], 2), dtype=bool)
    for lo, hi in rx_windows:
        received[0, :, 1] |= (times[0] >= lo) & (times[0] < hi)
    return times, received


def test_lost_alarm_and_isolated_beacon_does_not_clear_it(cfg):
    d = cfg["detector"]
    t = _time(120.0)
    # full reception until 20 s; silence until 60; an isolated beacon at 52 s; then full reception
    times, received = _beacons([(0.0, 20.0), (52.0, 52.4), (60.0, 120.0)])
    silence, rx_ok = reception_features(t, times, received, cfg)
    s = silence[:, 1, 0]  # observer 1, target 0
    state = lost_alarm(silence, rx_ok, t, d)[:, 1, 0]
    last_before = 19.5
    assert _first_true(state, t) == pytest.approx(last_before + d["lost_silence"], abs=2 * DT)
    assert not state[t < last_before + d["lost_silence"] - 0.2].any()
    # the isolated beacon resets the silence but does not turn off the alarm
    assert s[int(52.3 / DT)] < 1.0
    assert state[(t >= 50.0) & (t < 60.0)].all()
    # with full reception it turns off when there are >= 8 beacons out of 10 expected in 5 s: 60 + 3.5 s
    needed = d["release_reception"] * d["release_time"] / cfg["packets"]["beacon_period"]
    assert needed == pytest.approx(8.0)
    assert _first_false_after(state, t, 55.0) == pytest.approx(63.5, abs=DT)
    # the diagonal (observer = target) never produces alarms
    valid = ~np.eye(2, dtype=bool)
    assert not lost_alarm(silence, rx_ok, t, d, valid)[:, 0, 0].any()


def test_system_alarm_and_episode_count(cfg):
    d = cfg["detector"]
    t = _time(100.0)
    # three observers: the system alarm requires at least min_observers
    obs = np.zeros((len(t), 3), dtype=bool)
    obs[(t >= 20.0) & (t < 30.0), 0] = True
    obs[(t >= 25.0) & (t < 35.0), 1] = True
    two = system_alarm(obs, 2)
    assert two[(t >= 25.0) & (t < 30.0)].all() and not two[t < 25.0].any() and not two[t >= 30.0].any()
    assert system_alarm(obs, 1)[(t >= 20.0) & (t < 35.0)].all()
    # episodes = rising edges after the warmup
    sysa = np.zeros((len(t), 2), dtype=bool)
    sysa[(t >= 5.0) & (t < 15.0), 0] = True  # on across the warmup (10 s): not an episode
    sysa[(t >= 20.0) & (t < 25.0), 0] = True  # episode
    sysa[(t >= 40.0) & (t < 40.5), 0] = True  # episode
    sysa[(t >= 60.0) & (t < 61.0), 1] = True  # episode
    sysa[(t >= 70.0) & (t < 71.0), 1] = True  # episode
    sysa[t >= 90.0, 1] = True  # episode until the end
    assert count_episodes(sysa, t, d["warmup"]).tolist() == [2, 3]
    assert count_episodes(sysa, t, 0.0).tolist() == [3, 3]  # without warmup the first one counts


def _full_reception(n_nodes, t_end, period=0.5):
    """Staggered periodic beacons (node k at k * period / n_nodes), all received by everyone (diagonal excluded)."""
    times = np.array([np.arange(k * period / n_nodes, t_end, period) for k in range(n_nodes)])
    received = np.ones((n_nodes, times.shape[1], n_nodes), bool)
    received[np.arange(n_nodes), :, np.arange(n_nodes)] = False
    return times, received


def _lost_observers(cfg, times, received, t):
    silence, rx_ok = reception_features(t, times, received, cfg)
    return lost_alarm(silence, rx_ok, t, cfg["detector"], ~np.eye(times.shape[0], dtype=bool))


def test_combined_alarm_with_one_observer_is_the_old_formula(cfg):
    """`lost_min_observers` 1 (and None) give bit for bit `system_alarm(pre | lost, min_observers)`; `lost_system_alarm` the old "lost" alone."""
    rng = np.random.default_rng(0)
    pre = rng.random((400, 5, 5)) < 0.05
    lost = rng.random((400, 5, 5)) < 0.03
    for mo in (1, 2):
        old = system_alarm(pre | lost, mo)
        np.testing.assert_array_equal(combined_alarm(pre, lost, mo, None), old)
        if mo == 1:
            np.testing.assert_array_equal(combined_alarm(pre, lost, mo, 1), old)
    d = {**cfg["detector"], "min_observers": 1, "lost_min_observers": 1}
    np.testing.assert_array_equal(lost_system_alarm(lost, d), system_alarm(lost, 1))
    d["lost_min_observers"] = None
    np.testing.assert_array_equal(lost_system_alarm(lost, {**d, "min_observers": 2}), system_alarm(lost, 2))


def test_deaf_observer_alone_cannot_raise_the_system_alarm_with_two_confirmations(cfg):
    """Observer 0 hears nobody for 15 s: its four "lost" alarms reach the system with `lost_min_observers` 1, not with 2."""
    n, t_end = 5, 120.0
    t = _time(t_end)
    times, received = _full_reception(n, t_end)
    received[:, :, 0] &= ~((times >= 40.0) & (times < 55.0))
    lost = _lost_observers(cfg, times, received, t)
    pre = np.zeros_like(lost)
    one = combined_alarm(pre, lost, 1, 1)
    two = combined_alarm(pre, lost, 1, 2)
    assert one[:, 1:].any(axis=0).all()
    assert not two.any()


def test_node_that_everyone_stops_hearing_raises_the_alarm_with_two_confirmations(cfg):
    """Node 3 stops transmitting at 40 s: the four other observers declare it lost, so the alarm fires with 2 confirmations (and only on node 3)."""
    n, t_end = 5, 120.0
    t = _time(t_end)
    times, received = _full_reception(n, t_end)
    received[3, times[3] >= 40.0, :] = False
    lost = _lost_observers(cfg, times, received, t)
    two = combined_alarm(np.zeros_like(lost), lost, 1, 2)
    assert _first_true(two[:, 3], t) == pytest.approx(40.0 + cfg["detector"]["lost_silence"], abs=1.5)
    assert two[t > 52.0, 3].all()
    assert not two[:, [0, 1, 2, 4]].any()


def test_detection_chain_on_a_fading_link(cfg):
    """Deterministic run: node 4 moves away with an RSSI ramp; the others stay at -60 dBm."""
    n, dt, n_steps = 5, cfg["simulation"]["dt"], 6000  # 600 s
    cfg["packets"]["reception"]["background_loss"] = 0.0
    steps = np.arange(n_steps)[:, None, None]
    t_grid = steps * dt
    rssi = np.full((n_steps, n, n), -60.0)
    ramp = -60.0 - 1.0 * np.clip(t_grid - 300.0, 0.0, None)  # -1 dB/s from 300 s
    rssi[:, 4, :] = ramp[:, 0, :]
    rssi[:, :, 4] = ramp[:, 0, :]
    mob, ch = static_mobility(n_steps, n, dt), manual_channel(rssi, dt)
    res = simulate_packets(mob, ch, cfg)
    run = {"t": ch.t, "beacon_times": res.beacon_times, "received": res.received, "rssi": res.rssi}
    inputs, kres = build_inputs(run, cfg, cfg["kalman"]["sigma_a"], 1.0)
    d = cfg["detector"]
    for scope in ("fused", "pairwise"):
        pre = pre_alarm(inputs.level[scope], inputs.slope[scope], inputs.rx_ok, inputs.t, d, mode="level",
                        valid=inputs.valid)
        system = system_alarm(pre, 1)
        crossing = 300.0 + (-60.0 - d["level_threshold"])  # RSSI = threshold
        first = _first_true(system[:, 4], inputs.t)
        assert crossing + d["min_duration"] - 0.5 <= first <= crossing + d["min_duration"] + 4.0
        # the remaining nodes are never flagged by those in the group
        assert not system_alarm(pre[:, :4, :4], 1).any()
        if scope == "pairwise":  # without fusion the isolated node sees only its own links, which get worse
            assert system[:, :4].any()
        # in fused the neighbour tables dominate as long as they arrive, then they age beyond max_age
    # the node stays in alarm; "lost" when reception ceases (RSSI below sensitivity)
    lost = system_alarm(lost_alarm(inputs.silence, inputs.rx_ok, inputs.t, d, inputs.valid), 1)
    sens = cfg["packets"]["reception"]["sensitivity_dbm"]
    last_rx = 300.0 + (-60.0 - sens)
    assert _first_true(lost[:, 4], inputs.t) > last_rx + d["lost_silence"] - 3.0
    assert _first_true(lost[:, 4], inputs.t) < last_rx + d["lost_silence"] + 8.0
    assert not system_alarm(lost_alarm(inputs.silence, inputs.rx_ok, inputs.t, d, inputs.valid)[:, :4, :4], 1).any()
    # fused slope close to -1 dB/s on the ramp
    s_mid = inputs.slope["fused"][int(330 / dt) : int(335 / dt), :4, 4]
    assert np.nanmean(s_mid) == pytest.approx(-1.0, abs=0.15)


# ---------------------------------------------------------------------------
# 9. Metrics on synthetic series
# ---------------------------------------------------------------------------


def test_summarize_separated_by_hand():
    t = np.arange(0.0, 200.0, DT)
    idx_start = int(round(100.0 / DT))
    dist = np.where(t >= 100.0, 2.0 * (t - 100.0), 0.0)  # 2 m/s: exceeds 30 m at 115.1 s
    limit = 30.0
    out = summarize_separated(t >= 110.3, t, idx_start, dist, limit)
    assert out["delay"] == pytest.approx(10.3, abs=1e-9)
    assert out["distance"] == pytest.approx(20.6) and not out["fortuitous"]
    assert out["success"]  # on at 20.6 m
    # on when the distance is already beyond 30 m: detected, but not in time
    out = summarize_separated(t >= 120.0, t, idx_start, dist, limit)
    assert out["delay"] == pytest.approx(20.0) and out["distance"] == pytest.approx(40.0) and not out["success"]
    # at the limit: at 115.0 s the distance is 30.0 (does not exceed): in time; at 115.1 s no
    assert summarize_separated(t >= 115.0, t, idx_start, dist, limit)["success"]
    assert not summarize_separated(t >= 115.1, t, idx_start, dist, limit)["success"]
    # already active at the start instant: delay 0, fortuitous detection (and successful)
    out = summarize_separated(t >= 90.0, t, idx_start, dist, limit)
    assert out["delay"] == 0.0 and out["fortuitous"] and out["success"]
    assert out["distance"] == pytest.approx(dist[idx_start])
    # an alarm ended before the start does not count; never on: failure
    out = summarize_separated((t >= 50.0) & (t < 99.0), t, idx_start, dist, limit)
    assert out["delay"] == math.inf and np.isnan(out["distance"]) and not out["success"]
    # the distance never exceeds the limit: every detection is in time
    assert summarize_separated(t >= 190.0, t, idx_start, np.zeros(len(t)), limit)["success"]


def test_wilson_interval_known_values():
    assert wilson_interval(0, 20, 0.95) == pytest.approx((0.0, 3.8416 / 23.8416), abs=1e-4)  # 0.1611
    lo, hi = wilson_interval(20, 20, 0.95)
    assert lo == pytest.approx(0.8389, abs=1e-4) and hi == pytest.approx(1.0)
    lo, hi = wilson_interval(10, 20, 0.95)
    assert (lo, hi) == pytest.approx((0.2993, 0.7007), abs=1e-4) and lo + hi == pytest.approx(1.0)
    # more trials, narrower interval; higher level, wider interval
    assert np.subtract(*wilson_interval(50, 100, 0.95)[::-1]) < np.subtract(*wilson_interval(10, 20, 0.95)[::-1])
    assert np.subtract(*wilson_interval(10, 20, 0.99)[::-1]) > np.subtract(*wilson_interval(10, 20, 0.95)[::-1])


def test_poisson_interval_known_values():
    assert poisson_interval(0, 1.0, 0.95) == pytest.approx((0.0, 3.689), abs=1e-3)
    assert poisson_interval(1, 1.0, 0.95) == pytest.approx((0.02532, 5.572), abs=1e-3)
    assert poisson_interval(10, 1.0, 0.95) == pytest.approx((4.795, 18.39), abs=1e-2)
    # per hour: the same limits divided by the observation hours
    assert poisson_interval(10, 2.5, 0.95) == pytest.approx((4.795 / 2.5, 18.39 / 2.5), abs=1e-2)


def test_summarize_none_false_alarms_by_hand():
    dt, warmup = DT, 10.0
    t = np.arange(18100) * dt  # 1810 s: after the warmup 1800 s = 0.5 h
    system = np.zeros((len(t), 2), dtype=bool)
    system[(t >= 100.0) & (t < 110.0), 0] = True
    system[(t >= 500.0) & (t < 501.0), 0] = True
    system[(t >= 900.0) & (t < 903.0), 1] = True
    system[(t >= 5.0) & (t < 12.0), 1] = True  # across the warmup: does not count
    out = summarize_none(system, t, warmup)
    assert out["episodes"] == 3
    assert out["hours"] == pytest.approx(0.5)
    assert out["active"] == round((10.0 + 1.0 + 3.0 + 2.0) / dt)  # the 12-10 s after the warmup is inside
    assert out["samples"] == 18000 * 2
    # false alarms per hour = 3 / 0.5 h
    assert out["episodes"] / out["hours"] == pytest.approx(6.0)


def test_nearest_rank_and_threshold_grid(cfg):
    assert nearest_rank([10.0, 20.0, 40.0, math.inf], 0.5) == 20.0
    assert nearest_rank([10.0, 20.0, 40.0, math.inf], 0.9) == math.inf
    assert nearest_rank([3.0], 0.9) == 3.0 and math.isnan(nearest_rank([], 0.5))
    lv = threshold_grid(cfg["metrics"]["level_threshold_sweep"])
    sl = threshold_grid(cfg["metrics"]["slope_threshold_sweep"])
    assert len(lv) == 15 and lv[0] == -95.0 and lv[-1] == -60.0
    assert len(sl) == 16 and sl[0] == -4.0 and sl[-1] == -0.25


def _hand_rows(cfg):
    """Rows of `aggregate` for three thresholds, with hand-built results."""
    cfg["metrics"]["success_window"] = 30.0
    key = ("fused", "level", "both")
    thr = np.array([-90.0, -80.0, -70.0])

    def none_run(episodes, hours, active):
        return {key: {"threshold": thr, "episodes": np.array(episodes), "hours": np.full(3, hours),
                      "active": np.array(active), "samples": np.full(3, 1000)}}

    none = [none_run([0, 1, 4], 0.5, [0, 10, 100]), none_run([0, 0, 2], 0.5, [0, 0, 50])]
    delays = {  # (runs, thresholds)
        "slowdown": np.array([[math.inf, 10.0, 5.0], [math.inf, 40.0, 8.0], [math.inf, 20.0, 9.0], [50.0, 25.0, 1.0]]),
        "stop": np.array([[math.inf, 5.0, 2.0]] * 4),
    }
    success = {  # successful by distance, independent of the delay in seconds
        "slowdown": np.array([[False, True, True], [False, True, True], [False, True, True], [False, True, True]]),
        "stop": np.array([[False, True, True]] * 4),
    }

    def sep_run(row, scenario):
        return {key: {"threshold": thr, "delay": delays[scenario][row],
                      "distance": np.where(np.isfinite(delays[scenario][row]), 10.0 * delays[scenario][row], np.nan),
                      "fortuitous": np.array([False, False, row == 3 and scenario == "slowdown"]),
                      "success": success[scenario][row]}}

    sep = {s: [sep_run(r, s) for r in range(4)] for s in ("slowdown", "stop")}
    rows = aggregate(none, sep, cfg)
    for r in rows:
        r["sigma_a"] = 0.5
    return rows, thr


def test_aggregate_by_hand(cfg):
    rows, thr = _hand_rows(cfg)
    pick = {(r["scenario"], r["threshold"]): r for r in rows}
    # P_d = fraction of runs successful by distance; the informative column uses the 30 s
    assert [pick[("slowdown", x)]["p_detect"] for x in thr] == [0.0, 1.0, 1.0]
    assert [pick[("slowdown", x)]["p_detect_window"] for x in thr] == pytest.approx([0.0, 3 / 4, 1.0])
    assert pick[("slowdown", -80.0)]["delay_median"] == 20.0 and pick[("slowdown", -80.0)]["delay_p90"] == 40.0
    assert pick[("slowdown", -80.0)]["distance_median"] == 200.0
    assert pick[("slowdown", -70.0)]["fortuitous"] == 1
    assert pick[("stop", -80.0)]["p_detect"] == 1.0 and pick[("slowdown", -90.0)]["delay_median"] == math.inf
    # false alarms per hour = total episodes / total hours = (0, 1, 6) / 1 h; time in alarm (0, 0.5%, 7.5%)
    assert [pick[("slowdown", x)]["fa_per_hour"] for x in thr] == pytest.approx([0.0, 1.0, 6.0])
    assert [pick[("slowdown", x)]["active_fraction"] for x in thr] == pytest.approx([0.0, 0.005, 0.075])
    # intervals: Wilson over 4 runs, Poisson over 1 hour
    r = pick[("slowdown", -80.0)]
    assert (r["p_lo"], r["p_hi"]) == pytest.approx(wilson_interval(4, 4, 0.95))
    assert (r["fa_lo"], r["fa_hi"]) == pytest.approx(poisson_interval(1, 1.0, 0.95))
    assert pick[("slowdown", -90.0)]["fa_lo"] == 0.0 and pick[("slowdown", -90.0)]["fa_hi"] == pytest.approx(3.689, abs=1e-3)
    # feasible: FA <= 1 and time in alarm <= 1%
    assert [pick[("slowdown", x)]["feasible"] for x in thr] == [1, 1, 0]


def test_working_point_uses_only_feasible_points(cfg):
    rows, thr = _hand_rows(cfg)
    # -70 has maximum P_d but 6 false alarms/hour: not chosen; -80 (FA = 1.0 allowed) beats -90
    wp = working_point(rows, "fused", "level", "both")
    assert wp["threshold"] == -80.0 and wp["fa_per_hour"] == pytest.approx(1.0) and wp["p_detect"] == 1.0
    # cap on time in alarm: -80 has 0.5% time in alarm
    cfg2 = copy.deepcopy(cfg)
    cfg2["metrics"]["max_alarm_time_fraction"] = 0.004
    rows2, _ = _hand_rows(cfg2)
    assert working_point(rows2, "fused", "level", "both")["threshold"] == -90.0
    # cap on false alarms per hour
    cfg3 = copy.deepcopy(cfg)
    cfg3["metrics"]["target_false_alarms_per_hour"] = 0.5
    rows3, _ = _hand_rows(cfg3)
    assert working_point(rows3, "fused", "level", "both")["threshold"] == -90.0
    # no feasible point, or combination absent: None
    cfg4 = copy.deepcopy(cfg)
    cfg4["metrics"]["target_false_alarms_per_hour"] = -1.0
    rows4, _ = _hand_rows(cfg4)
    assert working_point(rows4, "fused", "level", "both") is None
    assert working_point(rows, "pairwise", "level", "both") is None


def _random_obs(rng, t, m=3, depth=15.0):
    """Slow random level (filtered Gaussian walk) for m observers."""
    base = -70.0 + np.cumsum(rng.normal(0.0, 0.25, (len(t), m)), axis=0)
    base -= np.linspace(0.0, depth, len(t))[:, None]
    return base


def test_widening_threshold_never_decreases_detection_or_active_time(cfg):
    d = cfg["detector"]
    t = np.arange(0.0, 300.0, DT)
    thresholds = np.arange(-95.0, -55.0, 2.5)
    for seed in range(6):
        rng = np.random.default_rng(seed)
        level = _random_obs(rng, t)
        slope = np.gradient(level, DT, axis=0)
        rx_ok = rng.random(level.shape) > 0.2
        rx_ok = np.where(rng.random() > 0.5, rx_ok, np.ones_like(rx_ok))
        prev_obs = prev_sys = None
        delays, fractions, successes = [], [], []
        for thr in thresholds:
            obs = pre_alarm(level, slope, rx_ok, t, d, mode="both", level_threshold=thr, slope_threshold=0.0)
            sysa = system_alarm(obs, 1)
            if prev_obs is not None:
                assert np.all(obs | ~prev_obs)  # set in alarm: monotonic (superset)
                assert np.all(sysa | ~prev_sys)
            prev_obs, prev_sys = obs, sysa
            fractions.append(sysa.mean())
            out = summarize_separated(sysa, t, 1000, np.linspace(0.0, 300.0, len(t)), 30.0)
            delays.append(out["delay"])
            successes.append(out["success"])
        assert np.all(np.diff(fractions) >= -1e-12)
        assert all(b <= a for a, b in zip(delays, delays[1:]))  # the delay never grows (inf included)
        assert np.all(np.diff(np.array(successes, dtype=int)) >= 0)  # success in metres is not lost
    # slope mode: same monotonicity on the slope threshold
    for seed in range(3):
        rng = np.random.default_rng(10 + seed)
        level = _random_obs(rng, t)
        slope = np.gradient(level, DT, axis=0)
        rx_ok = np.ones(level.shape, dtype=bool)
        prev = None
        for thr in np.arange(-4.0, 0.0, 0.5):
            obs = pre_alarm(level, slope, rx_ok, t, d, mode="slope", slope_threshold=thr)
            if prev is not None:
                assert np.all(obs | ~prev)
            prev = obs


def test_episodes_can_drop_when_threshold_widens_but_not_without_merging(cfg):
    d = cfg["detector"]
    t = np.arange(0.0, 100.0, DT)
    rx_ok = np.ones(len(t), dtype=bool)
    slope = np.zeros(len(t))
    # two deep dips (-90) separated by a stretch at -80
    level = np.full(len(t), -60.0)
    level[(t >= 20.0) & (t < 25.0)] = -90.0
    level[(t >= 25.0) & (t < 40.0)] = -80.0
    level[(t >= 40.0) & (t < 45.0)] = -90.0
    ep = {}
    for thr in (-92.0, -85.0, -75.0):
        sysa = system_alarm(pre_alarm(level, slope, rx_ok, t, d, mode="level", level_threshold=thr)[:, None], 1)
        ep[thr] = int(count_episodes(sysa, t, d["warmup"]))
    assert ep == {-92.0: 0, -85.0: 2, -75.0: 1}  # two episodes merge into one: episodes can decrease
    # without merges (a single dip) the count never decreases
    level = np.full(len(t), -60.0)
    level[(t >= 20.0) & (t < 25.0)] = -90.0
    counts = []
    for thr in np.arange(-95.0, -62.0, 2.5):  # below the base level: no alarm already on at the warmup
        sysa = system_alarm(pre_alarm(level, slope, rx_ok, t, d, mode="level", level_threshold=thr)[:, None], 1)
        counts.append(int(count_episodes(sysa, t, d["warmup"])))
    assert np.all(np.diff(counts) >= 0) and counts[0] == 0 and counts[-1] == 1


# ---------------------------------------------------------------------------
# 10. Reproducibility, seeds and cache
# ---------------------------------------------------------------------------


def test_seeds_and_separation_draws(cfg):
    seeds = run_seeds(cfg)
    assert len(seeds) == cfg["metrics"]["n_runs"] and len(set(seeds.tolist())) == len(seeds)
    np.testing.assert_array_equal(seeds, run_seeds(copy.deepcopy(cfg)))
    nodes, starts = draw_separations(cfg)
    rng = np.random.default_rng(np.random.SeedSequence(cfg["simulation"]["seed"], spawn_key=(6,)))
    np.testing.assert_array_equal(nodes, rng.integers(0, cfg["group"]["n_nodes"], size=len(seeds)))
    np.testing.assert_array_equal(starts, rng.uniform(*cfg["metrics"]["separation_start_range"], size=len(seeds)))
    assert nodes.min() >= 0 and nodes.max() < cfg["group"]["n_nodes"]
    lo, hi = cfg["metrics"]["separation_start_range"]
    assert starts.min() >= lo and starts.max() <= hi
    assert len(np.unique(nodes)) > 1 and starts.std() > 100.0
    cfg2 = copy.deepcopy(cfg)
    cfg2["simulation"]["seed"] += 1
    assert not np.array_equal(draw_separations(cfg2)[1], starts)

    slow = scenario_config(cfg, "slowdown", seeds[3], nodes[3], starts[3])
    stop = scenario_config(cfg, "stop", seeds[3], nodes[3], starts[3])
    none = scenario_config(cfg, "none", seeds[3], nodes[3], starts[3])
    for c in (slow, stop, none):
        assert c["simulation"]["seed"] == seeds[3]
    for c in (slow, stop):
        assert c["separation"]["enabled"] and c["separation"]["node_id"] == nodes[3]
        assert c["separation"]["start_time"] == starts[3]
    assert not none["separation"]["enabled"]
    assert slow["separation"]["target_speed"] == cfg["separation"]["target_speed"]
    assert stop["separation"]["target_speed"] == cfg["metrics"]["stop_speed"] == 0.0
    with pytest.raises(ValueError):
        scenario_config(cfg, "other", 1, 0, 0.0)


def test_cache_key_depends_on_blocks_1_to_3_only(cfg):
    key = cache_key(cfg)
    other = copy.deepcopy(cfg)
    other["kalman"]["sigma_a"] = 2.0
    other["detector"]["mode"] = "level"
    other["fusion"]["max_age"] = 9.0
    other["metrics"]["n_runs"] = 3
    assert cache_key(other) == key  # the Kalman and the detector do not change Blocks 1-3
    for section, name, value in (("channel", "tx_power", 5.0), ("packets", "jitter", 0.1), ("group", "n_nodes", 4)):
        other = copy.deepcopy(cfg)
        other[section][name] = value
        assert cache_key(other) != key
    other = copy.deepcopy(cfg)
    other["simulation"]["seed"] += 1
    assert cache_key(other) != key and cache_key(other).startswith(str(other["simulation"]["seed"]))


def test_reproducibility_cache_and_blocks_1_to_3_unchanged(cfg, tmp_path):
    cfg["simulation"]["duration"] = 90.0
    cfg["separation"]["enabled"] = False
    run = load_or_simulate(cfg, tmp_path)
    assert len(list(tmp_path.glob("*.npz"))) == 1
    again = load_or_simulate(cfg, tmp_path)
    fresh = simulate_run(cfg)
    for key in ("t", "distances", "beacon_times", "received", "rssi"):
        np.testing.assert_array_equal(again[key], fresh[key])
        np.testing.assert_array_equal(run[key], fresh[key])
    assert again["fading_var"] == fresh["fading_var"] > 0.0
    assert len(list(tmp_path.glob("*.npz"))) == 1

    snapshot = copy.deepcopy(again)
    R = 10.0
    in1, k1 = build_inputs(again, cfg, 0.5, R)
    in2, k2 = build_inputs(again, cfg, 0.5, R)
    for scope in ("fused", "pairwise"):
        np.testing.assert_array_equal(in1.level[scope], in2.level[scope])
        np.testing.assert_array_equal(in1.slope[scope], in2.slope[scope])
    np.testing.assert_array_equal(k1.r, k2.r)
    np.testing.assert_array_equal(k1.nis, k2.nis)
    for key in ("t", "distances", "beacon_times", "received", "rssi"):  # Blocks 1-3 unchanged
        np.testing.assert_array_equal(again[key], snapshot[key])
    # another sigma_a changes only the Kalman: same beacons, same cache
    in3, k3 = build_inputs(again, cfg, 2.0, R)
    assert not np.array_equal(k1.r, k3.r)
    assert cache_key(cfg) == cache_key(copy.deepcopy(cfg))
    # with 5 nodes: the target index has no NaN on the diagonal of the tables
    assert np.isnan(in1.level["fused"][:, 2, 2]).all()

"""Test del Blocco 4: inoltro, fusione, rilevatore, metriche e cache.

Ogni test verifica un fatto noto (teoria, formula chiusa o calcolo a mano); tolleranze statistiche a 4 sigma.
"""

import copy
import math
from pathlib import Path

import numpy as np
import pytest

from src.channel import ChannelResult
from src.detector import (
    build_inputs,
    count_episodes,
    fuse,
    fusion_indices,
    latched_alarm,
    lost_alarm,
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
from src.mobility import MobilityResult, load_config
from src.packets import beacon_schedule, forward_link_payload, simulate_packets
from src.simulator import (
    cache_key,
    draw_separations,
    load_or_simulate,
    run_seeds,
    scenario_config,
    simulate_run,
)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


@pytest.fixture(scope="session")
def base_config():
    return load_config(CONFIG_PATH)


@pytest.fixture
def cfg(base_config):
    return copy.deepcopy(base_config)


# ---------------------------------------------------------------------------
# Copia congelata dell'inoltro del Blocco 3 (commit f4333e3), per il test 6
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
# 6. Rifattorizzazione dell'inoltro
# ---------------------------------------------------------------------------


def test_forwarding_refactor_matches_block3(cfg):
    t, times, received, rssi = _random_beacons(cfg)
    old_rssi, old_age = _old_knowledge(t, times, received, rssi)
    fields, age, relay = forward_link_payload(t, times, received, {"rssi": rssi})
    np.testing.assert_array_equal(fields["rssi"], old_rssi)  # uguaglianza esatta, NaN compresi
    np.testing.assert_array_equal(age, old_age)
    assert np.isfinite(old_rssi).any() and np.isnan(old_rssi).any()

    # un secondo campo viaggia insieme al primo, senza interferire
    fields2, age2, _ = forward_link_payload(t, times, received, {"a": rssi, "b": 2.0 * rssi})
    np.testing.assert_array_equal(fields2["a"], old_rssi)
    np.testing.assert_allclose(fields2["b"], 2.0 * fields2["a"])
    np.testing.assert_array_equal(age2, old_age)

    # istante di inoltro: sul link diretto manca, sugli altri precede t e segue l'origine
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
    ch = _channel(rssi_true, dt)
    res = simulate_packets(_mobility(n_steps, n, dt), ch, cfg)
    fields, age, _ = forward_link_payload(ch.t, res.beacon_times, res.received, {"rssi": res.rssi})
    np.testing.assert_array_equal(fields["rssi"], res.knowledge_rssi)
    np.testing.assert_array_equal(age, res.knowledge_age)


def _mobility(n_steps, n_nodes, dt):
    positions = np.zeros((n_steps, n_nodes, 2))
    positions[:, :, 0] = np.arange(n_nodes)[None, :]
    diff = positions[:, :, None, :] - positions[:, None, :, :]
    return MobilityResult(
        t=np.arange(n_steps) * dt,
        positions=positions,
        s_nodes=np.zeros((n_steps, n_nodes)),
        s_centroid=np.zeros(n_steps),
        distances=np.linalg.norm(diff, axis=-1),
        headings=np.tile([1.0, 0.0], (n_steps, n_nodes, 1)),
        lateral_offsets=np.zeros((n_steps, n_nodes)),
        metadata={},
    )


def _channel(rssi, dt):
    rssi = np.array(rssi, dtype=float)
    n_steps, n, _ = rssi.shape
    idx = np.arange(n)
    rssi[:, idx, idx] = np.nan
    zeros = np.zeros_like(rssi)
    return ChannelResult(
        t=np.arange(n_steps) * dt,
        rssi_true=rssi,
        rssi_measured=np.round(rssi),
        rssi_path_loss=rssi.copy(),
        shadowing=zeros,
        body_own=zeros,
        body_others=zeros,
        fading=zeros,
        obstacles=zeros,
        tx_offset=np.zeros(n),
        rx_offset=np.zeros(n),
        shadow_field=None,
        metadata={},
    )


# ---------------------------------------------------------------------------
# 7. Fusione
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
    # pesi 1 e 1/4: (-70 - 20) / 1.25 = -72; pendenza stessi pesi
    assert level == pytest.approx((-70.0 - 80.0 / 4.0) / 1.25)
    assert slope == pytest.approx((-1.0 - 3.0 / 4.0) / 1.25)
    assert var == pytest.approx(0.8)
    level, slope, var, _ = fuse(values, slopes, np.array([1.0, 1e15]), np.zeros(2), 2.0)
    assert level == pytest.approx(-70.0, abs=1e-9) and slope == pytest.approx(-1.0, abs=1e-9)


def test_fuse_max_age_excludes_old_entries():
    values = np.array([-70.0, -90.0, -50.0])
    ages = np.array([0.5, 2.0, 2.0 + 1e-6])
    level, _, _, n = fuse(values, np.zeros(3), np.ones(3), ages, 2.0)
    assert n == 2 and level == pytest.approx(-80.0)  # età == max_age ancora valida
    level, slope, var, n = fuse(values, np.zeros(3), np.ones(3), np.full(3, 5.0), 2.0)
    assert n == 0 and np.isnan(level) and np.isnan(slope) and np.isnan(var)
    # NaN in una voce: esclusa
    level, _, _, n = fuse(np.array([-70.0, np.nan]), np.zeros(2), np.ones(2), np.zeros(2), 2.0)
    assert n == 1 and level == -70.0


def _steady_network(cfg, own_level, other_level, n_beacons=120, stop_own_after=None):
    """3 nodi con livelli costanti per link; il bersaglio è il nodo 1, l'osservatore il nodo 0."""
    n = 3
    period = cfg["packets"]["beacon_period"]
    times = np.tile(np.arange(n_beacons) * period, (n, 1)) + np.arange(n)[:, None] * 0.01
    received = np.ones((n, n_beacons, n), dtype=bool) & ~np.eye(n, dtype=bool)[:, None, :]
    level = np.full((n, n), -60.0)
    level[1, 0] = own_level  # link 1 → 0 (propria stima dell'osservatore 0)
    level[1, 2] = other_level  # link 1 → 2 (arriva con la tabella del nodo 2)
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
    assert -80.0 < level_f[idx, m, i] < -70.0  # una media, non una delle due voci

    level_p, _, _, n_p = out["pairwise"]
    assert n_p[idx, m, i] == 1
    assert level_p[idx, m, i] == pytest.approx(r_own, rel=1e-12)
    assert level_p[idx, m, i] == pytest.approx(-70.0, abs=1e-6)
    # il livello fuso del link 1→2 non è usato dal pairwise anche se il valore è diverso
    assert abs(level_f[idx, m, i] - level_p[idx, m, i]) > 1e-3
    # diagonale: osservatore = bersaglio, nessun indice
    assert np.isnan(level_f[:, 1, 1]).all() and np.isnan(level_p[:, 0, 0]).all()


def test_fusion_max_age_applies_to_own_estimate_by_option(cfg):
    t, times, received, rssi = _steady_network(cfg, -70.0, -80.0, stop_own_after=30.0)
    kres = filter_all(times, received, rssi, 4.0, cfg["kalman"]["sigma_a"], cfg["kalman"])
    idx_late = int(np.searchsorted(t, 50.0))  # il link diretto 1→0 tace da 20 s: stima propria troppo vecchia
    out = fusion_indices(t, kres, cfg)
    assert np.isnan(out["pairwise"][0][idx_late, 0, 1])  # solo la stima propria, esclusa
    assert out["fused"][3][idx_late, 0, 1] == 1  # resta la voce inoltrata dal nodo 2
    assert out["fused"][0][idx_late, 0, 1] == pytest.approx(-80.0, abs=0.2)
    cfg2 = copy.deepcopy(cfg)
    cfg2["fusion"]["max_age_own"] = False
    out2 = fusion_indices(t, kres, cfg2)
    assert out2["pairwise"][0][idx_late, 0, 1] == pytest.approx(-70.0, abs=1e-6)  # stima propria sempre usata
    assert out2["fused"][3][idx_late, 0, 1] == 2
    # con varianza cresciuta, la fusione si sposta verso la voce inoltrata
    assert out2["fused"][0][idx_late, 0, 1] < -75.0


# ---------------------------------------------------------------------------
# 8. Rilevatore su serie sintetiche
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
    assert both.tolist() == [False, False]  # a pari indice vince lo spegnimento


def test_pre_alarm_on_after_min_duration_and_off_after_release_time(cfg):
    d = cfg["detector"]
    t = _time()
    level = np.full(len(t), -60.0)
    level[(t >= 20.0) & (t < 30.0)] = -90.0
    slope = np.zeros(len(t))
    rx_ok = np.ones(len(t), dtype=bool)
    state = pre_alarm(level, slope, rx_ok, t, d, mode="level")
    assert _first_true(state, t) == pytest.approx(20.0 + d["min_duration"])
    # rientrato da 30 s: si spegne dopo release_time di rientro, non prima
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
    rx_ok[(t >= 25.0) & (t < 42.0)] = False  # indice rientrato ma ricezione scarsa
    state = pre_alarm(level, np.zeros(len(t)), rx_ok, t, d, mode="level")
    assert state[(t >= 21.0) & (t < 42.0)].all()  # resta acceso: ricezione non buona
    assert _first_false_after(state, t, 21.0) == pytest.approx(42.0)  # appena la ricezione torna buona


def test_pre_alarm_hysteresis_and_margin(cfg):
    d = cfg["detector"]
    t = _time()
    level = np.full(len(t), -60.0)
    level[(t >= 20.0) & (t < 30.0)] = -90.0
    # rientro breve (3 s < release_time) poi di nuovo giù: nessuno spegnimento
    level[(t >= 33.0) & (t < 45.0)] = -90.0
    rx_ok = np.ones(len(t), dtype=bool)
    state = pre_alarm(level, np.zeros(len(t)), rx_ok, t, d, mode="level")
    assert state[(t >= 21.0) & (t < 50.0)].all()
    # rientro sopra la soglia ma dentro il margine (L fra soglia e soglia+margine): non basta
    level2 = np.full(len(t), -60.0)
    level2[(t >= 20.0) & (t < 30.0)] = -90.0
    level2[t >= 30.0] = d["level_threshold"] + d["release_level_margin"] / 2.0
    state2 = pre_alarm(level2, np.zeros(len(t)), rx_ok, t, d, mode="level")
    assert state2[t >= 21.0].all()
    # NaN (nessuna voce valida) mentre acceso: resta acceso
    level3 = level.copy()
    level3[(t >= 33.0) & (t < 60.0)] = np.nan
    assert pre_alarm(level3, np.zeros(len(t)), rx_ok, t, d, mode="level")[(t >= 21.0) & (t < 60.0)].all()


def test_pre_alarm_modes(cfg):
    d = cfg["detector"]
    t = _time(60.0)
    rx_ok = np.ones(len(t), dtype=bool)
    level = np.full(len(t), -60.0)
    slope = np.zeros(len(t))
    level[(t >= 10.0) & (t < 20.0)] = -90.0  # livello basso, pendenza nulla
    slope[(t >= 30.0) & (t < 40.0)] = -2.0  # pendenza negativa, livello alto
    level[(t >= 45.0) & (t < 55.0)] = -90.0  # entrambe
    slope[(t >= 45.0) & (t < 55.0)] = -2.0
    on = {m: pre_alarm(level, slope, rx_ok, t, d, mode=m) for m in ("level", "slope", "both")}
    assert on["level"][int(12 / DT)] and not on["level"][int(33 / DT)] and on["level"][int(47 / DT)]
    assert not on["slope"][int(12 / DT)] and on["slope"][int(33 / DT)] and on["slope"][int(47 / DT)]
    assert not on["both"][int(12 / DT)] and not on["both"][int(33 / DT)] and on["both"][int(47 / DT)]
    with pytest.raises(ValueError):
        pre_alarm(level, slope, rx_ok, t, d, mode="altro")


def _beacons(rx_windows, period=0.5, t_end=120.0):
    """Beacon del nodo 0 ricevuti dal nodo 1 solo negli intervalli `rx_windows`."""
    times = np.full((2, int(t_end / period)), np.nan)
    times[0] = np.arange(times.shape[1]) * period
    received = np.zeros((2, times.shape[1], 2), dtype=bool)
    for lo, hi in rx_windows:
        received[0, :, 1] |= (times[0] >= lo) & (times[0] < hi)
    return times, received


def test_lost_alarm_and_isolated_beacon_does_not_clear_it(cfg):
    d = cfg["detector"]
    t = _time(120.0)
    # ricezione piena fino a 20 s; silenzio fino a 60; un beacon isolato a 52 s; poi ricezione piena
    times, received = _beacons([(0.0, 20.0), (52.0, 52.4), (60.0, 120.0)])
    silence, rx_ok = reception_features(t, times, received, cfg)
    s = silence[:, 1, 0]  # osservatore 1, bersaglio 0
    state = lost_alarm(silence, rx_ok, t, d)[:, 1, 0]
    last_before = 19.5
    assert _first_true(state, t) == pytest.approx(last_before + d["lost_silence"], abs=2 * DT)
    assert not state[t < last_before + d["lost_silence"] - 0.2].any()
    # il beacon isolato azzera il silenzio ma non spegne l'allarme
    assert s[int(52.3 / DT)] < 1.0
    assert state[(t >= 50.0) & (t < 60.0)].all()
    # con ricezione piena si spegne quando in 5 s ci sono >= 8 beacon su 10 attesi: 60 + 3,5 s
    needed = d["release_reception"] * d["release_time"] / cfg["packets"]["beacon_period"]
    assert needed == pytest.approx(8.0)
    assert _first_false_after(state, t, 55.0) == pytest.approx(63.5, abs=DT)
    # la diagonale (osservatore = bersaglio) non produce mai allarmi
    valid = ~np.eye(2, dtype=bool)
    assert not lost_alarm(silence, rx_ok, t, d, valid)[:, 0, 0].any()


def test_system_alarm_and_episode_count(cfg):
    d = cfg["detector"]
    t = _time(100.0)
    # tre osservatori: l'allarme di sistema richiede almeno min_observers
    obs = np.zeros((len(t), 3), dtype=bool)
    obs[(t >= 20.0) & (t < 30.0), 0] = True
    obs[(t >= 25.0) & (t < 35.0), 1] = True
    two = system_alarm(obs, 2)
    assert two[(t >= 25.0) & (t < 30.0)].all() and not two[t < 25.0].any() and not two[t >= 30.0].any()
    assert system_alarm(obs, 1)[(t >= 20.0) & (t < 35.0)].all()
    # episodi = fronti di salita dopo il warmup
    sysa = np.zeros((len(t), 2), dtype=bool)
    sysa[(t >= 5.0) & (t < 15.0), 0] = True  # acceso a cavallo del warmup (10 s): non è un episodio
    sysa[(t >= 20.0) & (t < 25.0), 0] = True  # episodio
    sysa[(t >= 40.0) & (t < 40.5), 0] = True  # episodio
    sysa[(t >= 60.0) & (t < 61.0), 1] = True  # episodio
    sysa[(t >= 70.0) & (t < 71.0), 1] = True  # episodio
    sysa[t >= 90.0, 1] = True  # episodio fino alla fine
    assert count_episodes(sysa, t, d["warmup"]).tolist() == [2, 3]
    assert count_episodes(sysa, t, 0.0).tolist() == [3, 3]  # senza warmup il primo conta


def test_detection_chain_on_a_fading_link(cfg):
    """Corsa deterministica: il nodo 4 si allontana con una rampa di RSSI; gli altri restano a -60 dBm."""
    n, dt, n_steps = 5, cfg["simulation"]["dt"], 6000  # 600 s
    cfg["packets"]["reception"]["background_loss"] = 0.0
    steps = np.arange(n_steps)[:, None, None]
    t_grid = steps * dt
    rssi = np.full((n_steps, n, n), -60.0)
    ramp = -60.0 - 1.0 * np.clip(t_grid - 300.0, 0.0, None)  # -1 dB/s da 300 s
    rssi[:, 4, :] = ramp[:, 0, :]
    rssi[:, :, 4] = ramp[:, 0, :]
    mob, ch = _mobility(n_steps, n, dt), _channel(rssi, dt)
    res = simulate_packets(mob, ch, cfg)
    run = {"t": ch.t, "beacon_times": res.beacon_times, "received": res.received, "rssi": res.rssi}
    inputs, kres = build_inputs(run, cfg, cfg["kalman"]["sigma_a"], 1.0)
    d = cfg["detector"]
    for scope in ("fused", "pairwise"):
        pre = pre_alarm(inputs.level[scope], inputs.slope[scope], inputs.rx_ok, inputs.t, d, mode="level",
                        valid=inputs.valid)
        system = system_alarm(pre, 1)
        crossing = 300.0 + (-60.0 - d["level_threshold"])  # RSSI = soglia
        first = _first_true(system[:, 4], inputs.t)
        assert crossing + d["min_duration"] - 0.5 <= first <= crossing + d["min_duration"] + 4.0
        # i nodi rimasti non sono mai segnalati da chi è nel gruppo
        assert not system_alarm(pre[:, :4, :4], 1).any()
        if scope == "pairwise":  # senza fusione il nodo isolato vede solo i propri link, che peggiorano
            assert system[:, :4].any()
        # nel fused le tabelle dei vicini dominano finché arrivano, poi invecchiano oltre max_age
    # il nodo resta in allarme; "perso" quando la ricezione cessa (RSSI sotto la sensibilità)
    lost = system_alarm(lost_alarm(inputs.silence, inputs.rx_ok, inputs.t, d, inputs.valid), 1)
    sens = cfg["packets"]["reception"]["sensitivity_dbm"]
    last_rx = 300.0 + (-60.0 - sens)
    assert _first_true(lost[:, 4], inputs.t) > last_rx + d["lost_silence"] - 3.0
    assert _first_true(lost[:, 4], inputs.t) < last_rx + d["lost_silence"] + 8.0
    assert not system_alarm(lost_alarm(inputs.silence, inputs.rx_ok, inputs.t, d, inputs.valid)[:, :4, :4], 1).any()
    # pendenza fusa vicina a -1 dB/s sulla rampa
    s_mid = inputs.slope["fused"][int(330 / dt) : int(335 / dt), :4, 4]
    assert np.nanmean(s_mid) == pytest.approx(-1.0, abs=0.15)


# ---------------------------------------------------------------------------
# 9. Metriche su serie sintetiche
# ---------------------------------------------------------------------------


def test_summarize_separated_by_hand():
    t = np.arange(0.0, 200.0, DT)
    idx_start = int(round(100.0 / DT))
    dist = np.where(t >= 100.0, 2.0 * (t - 100.0), 0.0)  # 2 m/s: supera 30 m a 115,1 s
    limit = 30.0
    out = summarize_separated(t >= 110.3, t, idx_start, dist, limit)
    assert out["delay"] == pytest.approx(10.3, abs=1e-9)
    assert out["distance"] == pytest.approx(20.6) and not out["fortuitous"]
    assert out["success"]  # acceso a 20,6 m
    # acceso quando la distanza è già oltre i 30 m: rilevato, ma non in tempo
    out = summarize_separated(t >= 120.0, t, idx_start, dist, limit)
    assert out["delay"] == pytest.approx(20.0) and out["distance"] == pytest.approx(40.0) and not out["success"]
    # al limite: a 115,0 s la distanza vale 30,0 (non supera): in tempo; a 115,1 s no
    assert summarize_separated(t >= 115.0, t, idx_start, dist, limit)["success"]
    assert not summarize_separated(t >= 115.1, t, idx_start, dist, limit)["success"]
    # già attivo all'istante d'inizio: ritardo 0, rilevamento fortuito (e riuscito)
    out = summarize_separated(t >= 90.0, t, idx_start, dist, limit)
    assert out["delay"] == 0.0 and out["fortuitous"] and out["success"]
    assert out["distance"] == pytest.approx(dist[idx_start])
    # un allarme finito prima dell'inizio non conta; mai acceso: fallimento
    out = summarize_separated((t >= 50.0) & (t < 99.0), t, idx_start, dist, limit)
    assert out["delay"] == math.inf and np.isnan(out["distance"]) and not out["success"]
    # la distanza non supera mai il limite: ogni rilevamento è in tempo
    assert summarize_separated(t >= 190.0, t, idx_start, np.zeros(len(t)), limit)["success"]


def test_wilson_interval_known_values():
    assert wilson_interval(0, 20, 0.95) == pytest.approx((0.0, 3.8416 / 23.8416), abs=1e-4)  # 0,1611
    lo, hi = wilson_interval(20, 20, 0.95)
    assert lo == pytest.approx(0.8389, abs=1e-4) and hi == pytest.approx(1.0)
    lo, hi = wilson_interval(10, 20, 0.95)
    assert (lo, hi) == pytest.approx((0.2993, 0.7007), abs=1e-4) and lo + hi == pytest.approx(1.0)
    # più prove, intervallo più stretto; livello più alto, intervallo più largo
    assert np.subtract(*wilson_interval(50, 100, 0.95)[::-1]) < np.subtract(*wilson_interval(10, 20, 0.95)[::-1])
    assert np.subtract(*wilson_interval(10, 20, 0.99)[::-1]) > np.subtract(*wilson_interval(10, 20, 0.95)[::-1])


def test_poisson_interval_known_values():
    assert poisson_interval(0, 1.0, 0.95) == pytest.approx((0.0, 3.689), abs=1e-3)
    assert poisson_interval(1, 1.0, 0.95) == pytest.approx((0.02532, 5.572), abs=1e-3)
    assert poisson_interval(10, 1.0, 0.95) == pytest.approx((4.795, 18.39), abs=1e-2)
    # per ora: gli stessi limiti divisi per le ore di osservazione
    assert poisson_interval(10, 2.5, 0.95) == pytest.approx((4.795 / 2.5, 18.39 / 2.5), abs=1e-2)


def test_summarize_none_false_alarms_by_hand():
    dt, warmup = DT, 10.0
    t = np.arange(18100) * dt  # 1810 s: dopo il warmup 1800 s = 0,5 h
    system = np.zeros((len(t), 2), dtype=bool)
    system[(t >= 100.0) & (t < 110.0), 0] = True
    system[(t >= 500.0) & (t < 501.0), 0] = True
    system[(t >= 900.0) & (t < 903.0), 1] = True
    system[(t >= 5.0) & (t < 12.0), 1] = True  # a cavallo del warmup: non conta
    out = summarize_none(system, t, warmup)
    assert out["episodes"] == 3
    assert out["hours"] == pytest.approx(0.5)
    assert out["active"] == round((10.0 + 1.0 + 3.0 + 2.0) / dt)  # il 12-10 s dopo il warmup è dentro
    assert out["samples"] == 18000 * 2
    # falsi allarmi per ora = 3 / 0,5 h
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
    """Righe di `aggregate` per tre soglie, con risultati costruiti a mano."""
    cfg["metrics"]["success_window"] = 30.0
    key = ("fused", "level", "both")
    thr = np.array([-90.0, -80.0, -70.0])

    def none_run(episodes, hours, active):
        return {key: {"threshold": thr, "episodes": np.array(episodes), "hours": np.full(3, hours),
                      "active": np.array(active), "samples": np.full(3, 1000)}}

    none = [none_run([0, 1, 4], 0.5, [0, 10, 100]), none_run([0, 0, 2], 0.5, [0, 0, 50])]
    delays = {  # (corse, soglie)
        "slowdown": np.array([[math.inf, 10.0, 5.0], [math.inf, 40.0, 8.0], [math.inf, 20.0, 9.0], [50.0, 25.0, 1.0]]),
        "stop": np.array([[math.inf, 5.0, 2.0]] * 4),
    }
    success = {  # riuscito per distanza, indipendente dal ritardo in secondi
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
    # P_d = frazione di corse riuscite per distanza; la colonna informativa usa i 30 s
    assert [pick[("slowdown", x)]["p_detect"] for x in thr] == [0.0, 1.0, 1.0]
    assert [pick[("slowdown", x)]["p_detect_window"] for x in thr] == pytest.approx([0.0, 3 / 4, 1.0])
    assert pick[("slowdown", -80.0)]["delay_median"] == 20.0 and pick[("slowdown", -80.0)]["delay_p90"] == 40.0
    assert pick[("slowdown", -80.0)]["distance_median"] == 200.0
    assert pick[("slowdown", -70.0)]["fortuitous"] == 1
    assert pick[("stop", -80.0)]["p_detect"] == 1.0 and pick[("slowdown", -90.0)]["delay_median"] == math.inf
    # falsi allarmi per ora = episodi totali / ore totali = (0, 1, 6) / 1 h; tempo in allarme (0, 0,5%, 7,5%)
    assert [pick[("slowdown", x)]["fa_per_hour"] for x in thr] == pytest.approx([0.0, 1.0, 6.0])
    assert [pick[("slowdown", x)]["active_fraction"] for x in thr] == pytest.approx([0.0, 0.005, 0.075])
    # intervalli: Wilson su 4 corse, Poisson su 1 ora
    r = pick[("slowdown", -80.0)]
    assert (r["p_lo"], r["p_hi"]) == pytest.approx(wilson_interval(4, 4, 0.95))
    assert (r["fa_lo"], r["fa_hi"]) == pytest.approx(poisson_interval(1, 1.0, 0.95))
    assert pick[("slowdown", -90.0)]["fa_lo"] == 0.0 and pick[("slowdown", -90.0)]["fa_hi"] == pytest.approx(3.689, abs=1e-3)
    # ammissibile: FA <= 1 e tempo in allarme <= 1%
    assert [pick[("slowdown", x)]["feasible"] for x in thr] == [1, 1, 0]


def test_working_point_uses_only_feasible_points(cfg):
    rows, thr = _hand_rows(cfg)
    # -70 ha P_d massima ma 6 falsi allarmi/ora: non si sceglie; -80 (FA = 1,0 ammessa) batte -90
    wp = working_point(rows, "fused", "level", "both")
    assert wp["threshold"] == -80.0 and wp["fa_per_hour"] == pytest.approx(1.0) and wp["p_detect"] == 1.0
    # tetto sul tempo in allarme: -80 ha lo 0,5% di tempo in allarme
    cfg2 = copy.deepcopy(cfg)
    cfg2["metrics"]["max_alarm_time_fraction"] = 0.004
    rows2, _ = _hand_rows(cfg2)
    assert working_point(rows2, "fused", "level", "both")["threshold"] == -90.0
    # tetto sui falsi allarmi per ora
    cfg3 = copy.deepcopy(cfg)
    cfg3["metrics"]["target_false_alarms_per_hour"] = 0.5
    rows3, _ = _hand_rows(cfg3)
    assert working_point(rows3, "fused", "level", "both")["threshold"] == -90.0
    # nessun punto ammissibile, oppure combinazione assente: None
    cfg4 = copy.deepcopy(cfg)
    cfg4["metrics"]["target_false_alarms_per_hour"] = -1.0
    rows4, _ = _hand_rows(cfg4)
    assert working_point(rows4, "fused", "level", "both") is None
    assert working_point(rows, "pairwise", "level", "both") is None


def _random_obs(rng, t, m=3, depth=15.0):
    """Livello casuale lento (passeggiata gaussiana filtrata) per m osservatori."""
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
                assert np.all(obs | ~prev_obs)  # insieme in allarme: monotono (sovrainsieme)
                assert np.all(sysa | ~prev_sys)
            prev_obs, prev_sys = obs, sysa
            fractions.append(sysa.mean())
            out = summarize_separated(sysa, t, 1000, np.linspace(0.0, 300.0, len(t)), 30.0)
            delays.append(out["delay"])
            successes.append(out["success"])
        assert np.all(np.diff(fractions) >= -1e-12)
        assert all(b <= a for a, b in zip(delays, delays[1:]))  # il ritardo non cresce mai (inf compreso)
        assert np.all(np.diff(np.array(successes, dtype=int)) >= 0)  # il successo in metri non si perde
    # modo slope: stessa monotonia sulla soglia di pendenza
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
    # due buche profonde (-90) separate da un tratto a -80
    level = np.full(len(t), -60.0)
    level[(t >= 20.0) & (t < 25.0)] = -90.0
    level[(t >= 25.0) & (t < 40.0)] = -80.0
    level[(t >= 40.0) & (t < 45.0)] = -90.0
    ep = {}
    for thr in (-92.0, -85.0, -75.0):
        sysa = system_alarm(pre_alarm(level, slope, rx_ok, t, d, mode="level", level_threshold=thr)[:, None], 1)
        ep[thr] = int(count_episodes(sysa, t, d["warmup"]))
    assert ep == {-92.0: 0, -85.0: 2, -75.0: 1}  # due episodi si fondono in uno: gli episodi possono calare
    # senza fusioni (una sola buca) il conteggio non cala mai
    level = np.full(len(t), -60.0)
    level[(t >= 20.0) & (t < 25.0)] = -90.0
    counts = []
    for thr in np.arange(-95.0, -62.0, 2.5):  # sotto il livello base: nessun allarme acceso già al warmup
        sysa = system_alarm(pre_alarm(level, slope, rx_ok, t, d, mode="level", level_threshold=thr)[:, None], 1)
        counts.append(int(count_episodes(sysa, t, d["warmup"])))
    assert np.all(np.diff(counts) >= 0) and counts[0] == 0 and counts[-1] == 1


# ---------------------------------------------------------------------------
# 10. Riproducibilità, semi e cache
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
        scenario_config(cfg, "altro", 1, 0, 0.0)


def test_cache_key_depends_on_blocks_1_to_3_only(cfg):
    key = cache_key(cfg)
    other = copy.deepcopy(cfg)
    other["kalman"]["sigma_a"] = 2.0
    other["detector"]["mode"] = "level"
    other["fusion"]["max_age"] = 9.0
    other["metrics"]["n_runs"] = 3
    assert cache_key(other) == key  # il Kalman e il rilevatore non cambiano i Blocchi 1-3
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
    for key in ("t", "distances", "beacon_times", "received", "rssi"):  # Blocchi 1-3 invariati
        np.testing.assert_array_equal(again[key], snapshot[key])
    # un altro sigma_a cambia solo il Kalman: stessi beacon, stessa cache
    in3, k3 = build_inputs(again, cfg, 2.0, R)
    assert not np.array_equal(k1.r, k3.r)
    assert cache_key(cfg) == cache_key(copy.deepcopy(cfg))
    # con 5 nodi: l'indice del bersaglio non ha NaN sulla diagonale delle tabelle
    assert np.isnan(in1.level["fused"][:, 2, 2]).all()

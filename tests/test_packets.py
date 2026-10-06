"""Test del Blocco 3 (pacchetti).

Ogni test verifica un fatto noto indipendentemente (teoria o formula
chiusa), non solo l'assenza di eccezioni. Si usano canali costruiti a mano
(RSSI costante, nodi fermi) o corse brevi. Le tolleranze statistiche sono di
4 deviazioni standard binomiali.
"""

import copy
import math
from pathlib import Path

import numpy as np
import pytest

from src.channel import ChannelResult, simulate_channel
from src.mobility import MobilityResult, load_config, simulate_mobility
from src.packets import (
    beacon_schedule,
    logistic_parameters,
    reception_probability,
    simulate_packets,
)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


# ---------------------------------------------------------------------------
# Fixture e utilità
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def base_config():
    return load_config(CONFIG_PATH)


@pytest.fixture
def cfg(base_config):
    return copy.deepcopy(base_config)


def _mobility(n_steps: int, n_nodes: int, dt: float) -> MobilityResult:
    """`MobilityResult` con nodi fermi."""
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


def _channel(rssi: np.ndarray, dt: float) -> ChannelResult:
    """`ChannelResult` a mano da un RSSI vero (T, N, N); il misurato è l'intero più vicino."""
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


def _constant(cfg: dict, level: float, n_steps: int):
    """Mobilità e canale con RSSI costante `level` su tutti i link."""
    n = cfg["group"]["n_nodes"]
    dt = cfg["simulation"]["dt"]
    return _mobility(n_steps, n, dt), _channel(np.full((n_steps, n, n), level), dt)


def _run(cfg: dict, level: float, n_steps: int):
    mob, ch = _constant(cfg, level, n_steps)
    return simulate_packets(mob, ch, cfg)


def _logistic_independent(x, sensitivity, success, width):
    """Logistica scritta in altro modo: passa per 0,1 e 0,9 a ±width/2 dal centro."""
    centre = sensitivity - (width / 2.0) * math.log(success / (1.0 - success)) / math.log(9.0)
    return 1.0 / (1.0 + 9.0 ** (-2.0 * (np.asarray(x) - centre) / width))


def _binomial_tol(p: float, n: int) -> float:
    return 4.0 * math.sqrt(p * (1.0 - p) / n)


def _off(arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Elementi (i, k, j) con i != j e beacon valido."""
    n = arr.shape[0]
    mask = (~np.eye(n, dtype=bool))[:, None, :] & valid[:, :, None]
    return arr[mask]


# ---------------------------------------------------------------------------
# 1. Istanti dei beacon
# ---------------------------------------------------------------------------


def test_beacon_schedule(cfg):
    pk = cfg["packets"]
    period, jitter, dt = pk["beacon_period"], pk["jitter"], cfg["simulation"]["dt"]
    t_end = 3000.0
    rng = np.random.default_rng(1)
    times, steps = beacon_schedule(5, t_end, dt, pk, rng)

    for i in range(5):
        tt = times[i][~np.isnan(times[i])]
        d = np.diff(tt)
        assert d.min() >= period - jitter - 1e-9 and d.max() <= period + jitter + 1e-9
        assert (len(tt) - 1) / (tt[-1] - tt[0]) == pytest.approx(1.0 / period, rel=0.01)
        assert 0.0 <= tt[0] < period
        assert tt[-1] <= t_end
    # nessun beacon perso in coda: gli intervalli estratti coprono t_end, quindi
    # l'ultima colonna è sempre oltre la durata (NaN) per ogni nodo
    assert np.all(np.isnan(times[:, -1]))

    phases = times[:, 0]
    assert len(np.unique(np.round(phases, 6))) == 5

    valid = ~np.isnan(times)
    assert np.array_equal(steps[valid], np.rint(times[valid] / dt).astype(int))
    assert np.all(np.abs(times[valid] - steps[valid] * dt) <= dt / 2 + 1e-9)
    assert np.all(steps[~valid] == -1)


# ---------------------------------------------------------------------------
# 2. Curva di ricezione
# ---------------------------------------------------------------------------


def test_reception_curve(cfg):
    rc = cfg["packets"]["reception"]
    s, succ, w = rc["sensitivity_dbm"], rc["sensitivity_success"], rc["transition_width"]
    assert reception_probability(np.array([s]), rc)[0] == pytest.approx(succ, abs=1e-12)

    xs = np.linspace(s - 15, s + 15, 300001)
    p = reception_probability(xs, rc)
    assert np.all(np.diff(p) >= 0)
    x10 = xs[np.searchsorted(p, 0.1)]
    x90 = xs[np.searchsorted(p, 0.9)]
    assert x90 - x10 == pytest.approx(w, abs=1e-3)
    assert p[0] < 1e-6 and p[-1] > 1.0 - 1e-6
    np.testing.assert_allclose(p, _logistic_independent(xs, s, succ, w), atol=1e-12)

    centre, scale = logistic_parameters(s, succ, w)
    assert reception_probability(np.array([centre]), rc)[0] == pytest.approx(0.5)
    assert centre < s

    rc_thr = dict(rc, model="threshold")
    assert reception_probability(np.array([s + 0.01, s + 20.0]), rc_thr).tolist() == [1.0, 1.0]
    assert reception_probability(np.array([s - 0.01, s - 20.0]), rc_thr).tolist() == [0.0, 0.0]

    with pytest.raises(ValueError):
        reception_probability(xs, dict(rc, model="altro"))


# ---------------------------------------------------------------------------
# 3. Frequenza empirica
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("offset", [0.0, 1.0, -2.0])
def test_empirical_reception_rate(cfg, offset):
    rc = cfg["packets"]["reception"]
    res = _run(cfg, rc["sensitivity_dbm"] + offset, 20000)
    valid = ~np.isnan(res.beacon_times)
    rec = _off(res.received, valid)
    expected = float(_logistic_independent(rc["sensitivity_dbm"] + offset, rc["sensitivity_dbm"],
                                           rc["sensitivity_success"], rc["transition_width"]))
    expected *= 1.0 - rc["background_loss"]
    assert rec.mean() == pytest.approx(expected, abs=_binomial_tol(expected, len(rec)))
    assert _off(res.p_success, valid) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 4. Perdita di fondo
# ---------------------------------------------------------------------------


def test_background_loss_rate(cfg):
    res = _run(cfg, -40.0, 20000)
    valid = ~np.isnan(res.beacon_times)
    lost = ~_off(res.received, valid)
    q = cfg["packets"]["reception"]["background_loss"]
    assert lost.mean() == pytest.approx(q, abs=_binomial_tol(q, len(lost)))
    assert np.all(_off(res.passed_signal, valid))


def test_background_losses_independent(cfg):
    q = 0.3
    cfg["packets"]["reception"]["background_loss"] = q
    res = _run(cfg, -40.0, 40000)

    # nel tempo: lunghezza delle serie di perdite consecutive geometrica
    lengths = []
    for i in range(5):
        for j in range(5):
            if i == j:
                continue
            ok = res.received[i, : np.sum(~np.isnan(res.beacon_times[i])), j]
            lost = np.concatenate([[False], ~ok, [False]]).astype(int)
            edges = np.flatnonzero(np.diff(lost))
            lengths.extend((edges[1::2] - edges[0::2]).tolist())
    lengths = np.array(lengths)
    n_runs = len(lengths)
    for length in (1, 2, 3):
        expected = q ** (length - 1) * (1.0 - q)
        frac = np.mean(lengths == length)
        assert frac == pytest.approx(expected, abs=_binomial_tol(expected, n_runs))
    assert lengths.mean() == pytest.approx(1.0 / (1.0 - q), rel=0.05)

    # fra ricevitori: stesso beacon di i, ricevitori j1 e j2
    n_valid = int(np.sum(~np.isnan(res.beacon_times[0])))
    a = (~res.received[0, :n_valid, 1]).astype(float)
    b = (~res.received[0, :n_valid, 2]).astype(float)
    corr = np.corrcoef(a, b)[0, 1]
    assert abs(corr) < 4.0 / math.sqrt(n_valid)


# ---------------------------------------------------------------------------
# 5. RSSI riportato
# ---------------------------------------------------------------------------


def test_reported_rssi(cfg):
    n, dt, n_steps = 5, cfg["simulation"]["dt"], 3000
    rng = np.random.default_rng(3)
    rssi = -98.4 + rng.normal(0.0, 3.0, size=(n_steps, n, n))
    ch = _channel(rssi, dt)
    res = simulate_packets(_mobility(n_steps, n, dt), ch, cfg)
    assert res.received.any() and not res.received.all()  # il test vede sia ricevuti sia persi
    for i in range(n):
        for k in range(res.beacon_steps.shape[1]):
            s = res.beacon_steps[i, k]
            if s < 0:
                continue
            for j in range(n):
                if i == j:
                    assert not res.received[i, k, j] and np.isnan(res.rssi[i, k, j])
                elif res.received[i, k, j]:
                    assert res.rssi[i, k, j] == ch.rssi_measured[s, i, j]
                else:
                    assert np.isnan(res.rssi[i, k, j])


# ---------------------------------------------------------------------------
# 6. Conoscenza, caso deterministico
# ---------------------------------------------------------------------------


def _deterministic(cfg):
    cfg["packets"]["reception"]["background_loss"] = 0.0
    n, dt, n_steps = 5, cfg["simulation"]["dt"], 3000
    # RSSI alto, variabile nel tempo e diverso per ogni link: così i valori
    # inoltrati si possono ricondurre al beacon d'origine
    steps = np.arange(n_steps)[:, None, None]
    link = np.arange(n)[None, :, None] * 5 + np.arange(n)[None, None, :]
    rssi = -60.0 + (steps * 7 + link * 3) % 23
    ch = _channel(rssi, dt)
    return ch, simulate_packets(_mobility(n_steps, n, dt), ch, cfg)


def test_knowledge_deterministic(cfg):
    period, jitter, dt = cfg["packets"]["beacon_period"], cfg["packets"]["jitter"], cfg["simulation"]["dt"]
    ch, res = _deterministic(cfg)
    assert res.received[~np.isnan(res.beacon_times)[:, :, None] & ~np.eye(5, dtype=bool)[:, None, :]].all()
    warm = int(3 * period / dt)
    n = 5

    for m in range(n):
        for i in range(n):
            if i == m:
                continue
            age = res.knowledge_age[warm:, m, i, m]
            assert np.all(np.isfinite(age))
            assert age.min() >= 0.0 and age.max() < period + jitter
            assert age.max() > period - jitter - dt  # dente di sega che sale fino a circa un periodo
            assert age.min() < dt
            # dente di sega: l'età cresce di dt a ogni passo, salvo i ritorni a zero
            d = np.diff(age)
            assert np.all((np.abs(d - dt) < 1e-4) | (d < 0))

    # link inoltrati: età < 2 (period + jitter), valori del beacon d'origine
    for m in range(n):
        for i in range(n):
            for j in range(n):
                if j == m or i == j:
                    continue
                age = res.knowledge_age[warm:, m, i, j]
                assert np.all(np.isfinite(age))
                assert age.min() >= 0.0 and age.max() < 2.0 * (period + jitter)
                # origine: beacon di i con istante t - età
                for s in range(warm, res.knowledge_age.shape[0], 97):
                    origin = ch.t[s] - float(res.knowledge_age[s, m, i, j])
                    k = int(np.nanargmin(np.abs(res.beacon_times[i] - origin)))
                    assert res.beacon_times[i, k] == pytest.approx(origin, abs=2e-3)
                    assert res.knowledge_rssi[s, m, i, j] == pytest.approx(res.rssi[i, k, j])


# ---------------------------------------------------------------------------
# 7. Conoscenza, nodo isolato
# ---------------------------------------------------------------------------


def test_knowledge_isolated_node(cfg):
    n, dt, n_steps = 5, cfg["simulation"]["dt"], 3000
    cfg["packets"]["reception"]["background_loss"] = 0.0
    rssi = np.full((n_steps, n, n), -50.0)
    rssi[:, 4, :] = -150.0
    rssi[:, :, 4] = -150.0
    res = simulate_packets(_mobility(n_steps, n, dt), _channel(rssi, dt), cfg)

    assert not res.received[4].any() and not res.received[:, :, 4].any()
    assert np.all(np.isnan(res.knowledge_rssi[:, :, 4, :]))
    assert np.all(np.isnan(res.knowledge_rssi[:, :, :, 4]))
    assert np.all(np.isnan(res.knowledge_age[:, :, 4, :]))
    assert np.all(np.isnan(res.knowledge_age[:, :, :, 4]))
    # gli altri si conoscono a vicenda
    assert np.isfinite(res.knowledge_rssi[-1, 0, 1, 2])
    assert np.isfinite(res.knowledge_rssi[-1, 0, 1, 0])


# ---------------------------------------------------------------------------
# 8. Riproducibilità e indipendenza
# ---------------------------------------------------------------------------


def test_reproducibility_and_independence(cfg):
    mob, ch = _constant(cfg, cfg["packets"]["reception"]["sensitivity_dbm"], 5000)
    before = (ch.rssi_true.copy(), ch.rssi_measured.copy())
    r1 = simulate_packets(mob, ch, cfg)
    r2 = simulate_packets(mob, ch, cfg)
    for name in ("beacon_times", "received", "rssi", "knowledge_rssi", "knowledge_age", "passed_signal"):
        np.testing.assert_array_equal(getattr(r1, name), getattr(r2, name))
    np.testing.assert_array_equal(ch.rssi_true, before[0])
    np.testing.assert_array_equal(ch.rssi_measured, before[1])

    cfg2 = copy.deepcopy(cfg)
    cfg2["simulation"]["seed"] += 1
    r3 = simulate_packets(mob, ch, cfg2)
    assert not np.array_equal(r1.received, r3.received)

    cfg3 = copy.deepcopy(cfg)
    cfg3["packets"]["reception"]["background_loss"] = 0.4
    r4 = simulate_packets(mob, ch, cfg3)
    np.testing.assert_array_equal(r1.passed_signal, r4.passed_signal)
    np.testing.assert_array_equal(r1.beacon_times, r4.beacon_times)
    assert not np.array_equal(r1.passed_background, r4.passed_background)

    # cambiare i parametri dei pacchetti non cambia il canale e viceversa
    cfg4 = copy.deepcopy(cfg)
    cfg4["simulation"]["duration"] = 30.0
    cfg4["separation"]["enabled"] = False
    mob_s = simulate_mobility(cfg4)
    c1 = simulate_channel(mob_s, cfg4)
    cfg5 = copy.deepcopy(cfg4)
    cfg5["packets"]["jitter"] = 0.1
    cfg5["packets"]["reception"]["background_loss"] = 0.5
    c2 = simulate_channel(mob_s, cfg5)
    np.testing.assert_array_equal(c1.rssi_true, c2.rssi_true)
    snapshot = c1.rssi_true.copy()
    simulate_packets(mob_s, c1, cfg4)
    np.testing.assert_array_equal(c1.rssi_true, snapshot)


# ---------------------------------------------------------------------------
# 9. Distacco
# ---------------------------------------------------------------------------


def test_separation_stops_reception(cfg):
    cfg["separation"]["enabled"] = True
    cfg["simulation"]["duration"] = 800.0
    mob = simulate_mobility(cfg)
    ch = simulate_channel(mob, cfg)
    node = cfg["separation"]["node_id"]
    t_end = mob.t[-1]

    res = simulate_packets(mob, ch, cfg)
    others = [k for k in range(5) if k != node]

    def fraction(r, t_lo, t_hi):
        sel = (r.beacon_times[node] >= t_lo) & (r.beacon_times[node] < t_hi)
        return r.received[node][sel][:, others].mean()

    assert fraction(res, 100.0, 250.0) > 0.9  # nel gruppo il nodo viene sentito
    assert fraction(res, t_end - 60.0, t_end + 1.0) == 0.0

    cfg_lo = copy.deepcopy(cfg)
    cfg_lo["packets"]["reception"]["sensitivity_dbm"] = -92.0
    res_lo = simulate_packets(mob, ch, cfg_lo)
    assert np.all(res.received[res_lo.received])  # ricevuti con -92 ⊆ ricevuti con -98,4
    assert res.received.sum() > res_lo.received.sum()

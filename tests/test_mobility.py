"""Test del Blocco 1 (mobilità).

Ogni test verifica un fatto noto indipendentemente (sulla traccia GPX o
sulla teoria del processo di Ornstein-Uhlenbeck), non solo l'assenza di
eccezioni.
"""

import copy
from pathlib import Path

import numpy as np
import pytest

from src.mobility import (
    load_config,
    load_track,
    ou_process,
    range_statistic_of_normals,
    simulate_mobility,
)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def base_config():
    return load_config(CONFIG_PATH)


@pytest.fixture(scope="session")
def track(base_config):
    return load_track(base_config)


@pytest.fixture
def short_config(base_config):
    """Configurazione a durata ridotta, senza separazione, per i test che
    devono eseguire l'intera simulazione senza rallentare la suite.
    """
    cfg = copy.deepcopy(base_config)
    cfg["simulation"]["duration"] = 300.0
    cfg["separation"]["enabled"] = False
    return cfg


@pytest.fixture
def separation_config(base_config):
    cfg = copy.deepcopy(base_config)
    cfg["simulation"]["duration"] = 400.0
    cfg["separation"]["enabled"] = True
    return cfg


@pytest.fixture
def dispersion_config(base_config):
    """Durata più lunga di `short_config`: serve a stimare in modo stabile
    una statistica di coda (p95) sull'estensione del gruppo.
    """
    cfg = copy.deepcopy(base_config)
    cfg["simulation"]["duration"] = 1200.0
    cfg["separation"]["enabled"] = False
    return cfg


# ---------------------------------------------------------------------------
# 1-4: caricamento e pulizia della traccia
# ---------------------------------------------------------------------------


def test_track_length(track):
    assert track.length == pytest.approx(8534.0, abs=50.0)


def test_track_raw_point_count(track):
    assert track.n_points_raw == 3032


def test_glitch_repair(track, base_config):
    assert track.n_glitches_corrected == 1
    residual_steps = np.hypot(np.diff(track.x), np.diff(track.y))
    assert np.max(residual_steps) <= base_config["track"]["glitch_max_step"]


def test_median_speed(track):
    assert np.median(track.speed_raw) == pytest.approx(2.79, abs=0.15)


# ---------------------------------------------------------------------------
# 5-6: interrogazione del percorso
# ---------------------------------------------------------------------------


def test_query_interpolation(track):
    p0 = track.query(0.0)
    assert p0.position[0] == pytest.approx(track.x[0])
    assert p0.position[1] == pytest.approx(track.y[0])

    idx = len(track.s) // 2
    p_known = track.query(track.s[idx])
    assert p_known.position[0] == pytest.approx(track.x[idx], abs=1e-6)
    assert p_known.position[1] == pytest.approx(track.y[idx], abs=1e-6)

    # un punto a metà segmento deve stare esattamente sulla polilinea
    s_mid = (track.s[idx] + track.s[idx + 1]) / 2.0
    p_mid = track.query(s_mid)
    seg = np.array([track.x[idx + 1] - track.x[idx], track.y[idx + 1] - track.y[idx]])
    to_point = p_mid.position - np.array([track.x[idx], track.y[idx]])
    cross = seg[0] * to_point[1] - seg[1] * to_point[0]
    dist_from_line = abs(cross) / (np.hypot(*seg) + 1e-12)
    assert dist_from_line < 1e-6


def test_orthonormal_frame(track):
    s_samples = np.linspace(0.0, track.length, 200)
    sample = track.query(s_samples)
    t_norm = np.linalg.norm(sample.tangent, axis=-1)
    n_norm = np.linalg.norm(sample.normal, axis=-1)
    dot = np.sum(sample.tangent * sample.normal, axis=-1)
    assert np.allclose(t_norm, 1.0, atol=1e-9)
    assert np.allclose(n_norm, 1.0, atol=1e-9)
    assert np.allclose(dot, 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# 7-8: processo di Ornstein-Uhlenbeck
# ---------------------------------------------------------------------------


def test_ou_stationary_std():
    rng = np.random.default_rng(123)
    sigma = 1.5
    x = ou_process(n_steps=200_000, dt=0.1, tau=8.0, sigma=sigma, rng=rng, size=1)[:, 0]
    assert np.std(x) == pytest.approx(sigma, rel=0.05)


def test_ou_autocorrelation():
    rng = np.random.default_rng(321)
    dt, tau = 0.1, 8.0
    x = ou_process(n_steps=200_000, dt=dt, tau=tau, sigma=1.0, rng=rng, size=1)[:, 0]
    lag = int(round(tau / dt))
    x_centered = x - x.mean()
    autocorr = np.mean(x_centered[:-lag] * x_centered[lag:]) / np.var(x)
    assert autocorr == pytest.approx(1.0 / np.e, abs=0.05)


# ---------------------------------------------------------------------------
# statistica del range di N gaussiane standard
# ---------------------------------------------------------------------------


def test_range_statistic_mean():
    assert range_statistic_of_normals(5, "mean") == pytest.approx(2.3259, abs=0.001)


def test_range_statistic_p95():
    assert range_statistic_of_normals(5, "p95") == pytest.approx(3.858, abs=0.01)


# ---------------------------------------------------------------------------
# 9-10: simulazione completa
# ---------------------------------------------------------------------------


def test_group_dispersion_p95(dispersion_config):
    """Con spread_statistic: p95 (default), è il 95° percentile
    dell'estensione testa-coda, misurato DOPO il vincolo di distanza
    minima, ad avvicinarsi a longitudinal_spread — non la media.
    """
    assert dispersion_config["group"]["spread_statistic"] == "p95"
    result = simulate_mobility(dispersion_config)
    spread = np.max(result.s_nodes, axis=1) - np.min(result.s_nodes, axis=1)
    p95_spread = np.percentile(spread, 95)
    target = dispersion_config["group"]["longitudinal_spread"]
    assert p95_spread == pytest.approx(target, rel=0.20)


def test_group_dispersion_mean(dispersion_config):
    cfg = copy.deepcopy(dispersion_config)
    cfg["group"]["spread_statistic"] = "mean"
    result = simulate_mobility(cfg)
    spread = np.max(result.s_nodes, axis=1) - np.min(result.s_nodes, axis=1)
    mean_spread = np.mean(spread)
    target = cfg["group"]["longitudinal_spread"]
    assert mean_spread == pytest.approx(target, rel=0.20)


def test_start_offset_too_small_raises(short_config):
    cfg = copy.deepcopy(short_config)
    cfg["track"]["start_offset"] = 0.1
    with pytest.raises(ValueError, match="start_offset"):
        simulate_mobility(cfg)


def test_spread_too_small_raises(short_config):
    cfg = copy.deepcopy(short_config)
    # (n_nodes - 1) * min_node_gap * spread_margin_factor con i valori di
    # default è ben oltre 1 m: sicuramente troppo piccolo.
    cfg["group"]["longitudinal_spread"] = 1.0
    with pytest.raises(ValueError, match="longitudinal_spread"):
        simulate_mobility(cfg)


def test_min_node_gap_enforced(dispersion_config):
    """Senza separazione, il vincolo di distanza minima deve valere in
    ogni istante per ogni coppia di nodi, in coordinate stradali (il
    contratto esatto di `_enforce_min_distance`). La distanza euclidea
    vera può scendere leggermente sotto `min_node_gap` solo nei tratti di
    curvatura marcata (la corda è più corta dell'arco): sulla traccia di
    riferimento questo riguarda una frazione trascurabile dei campioni,
    concentrata in corrispondenza delle curve più strette (non una sola:
    la traccia ne ha più di una abbastanza stretta da produrre l'effetto).
    """
    result = simulate_mobility(dispersion_config)
    n_nodes = dispersion_config["group"]["n_nodes"]

    diff = result.distances + np.eye(n_nodes)[None, :, :] * 1e9
    min_dist_per_step = diff.min(axis=(1, 2))

    # Soglia calibrata sui dati osservati: la frazione di passi con una
    # coppia sotto 1 m è tipicamente sotto lo 0,1% (misurata: ~0,05%).
    frac_below_1m = np.mean(min_dist_per_step < 1.0)
    assert frac_below_1m < 0.02, (
        f"{frac_below_1m:.4%} dei passi con distanza minima sotto 1 m: "
        "troppi per essere spiegati dalla sola curvatura della traccia."
    )
    assert np.all(min_dist_per_step > 0.0)


def test_min_node_gap_enforced_with_separation(separation_config):
    """Il vincolo di distanza minima si applica SEMPRE a tutti i nodi, a
    ogni istante — anche dopo l'inizio della separazione (Fix 1: prima di
    questo fix, escludere il nodo separato produceva un buco nella
    repulsione proprio nei primi istanti dopo `start_time`, quando è
    ancora fisicamente dentro al gruppo: la distanza scendeva a ~0,3 m).
    Qui si verifica direttamente che i primi secondi dopo `start_time` —
    la finestra in cui il bug si manifestava — non mostrino alcuna
    distanza anomala rispetto al resto della simulazione.
    """
    result = simulate_mobility(separation_config)
    n_nodes = separation_config["group"]["n_nodes"]
    node_id = separation_config["separation"]["node_id"]
    dt = separation_config["simulation"]["dt"]
    other_nodes = [i for i in range(n_nodes) if i != node_id]

    idx_start = int(np.searchsorted(result.t, separation_config["separation"]["start_time"]))
    window_steps = max(int(round(5.0 / dt)), 1)
    window = result.distances[idx_start : idx_start + window_steps, node_id][:, other_nodes]

    min_node_gap = separation_config["group"]["min_node_gap"]
    tolerance = separation_config["group"]["min_gap_tolerance"]
    assert window.min() >= min_node_gap - tolerance, (
        f"distanza minima nodo separato-gruppo nei primi 5 s dopo start_time = "
        f"{window.min():.3f} m, sotto min_node_gap={min_node_gap} m: il vincolo "
        "non sta agendo sulla traiettoria reale del nodo separato."
    )

    # riprova indiretta: quella finestra non deve essere un outlier rispetto
    # ai minimi osservati su finestre di pari durata in tutto il resto della
    # simulazione (un buco di repulsione locale produrrebbe un outlier).
    diff = result.distances + np.eye(n_nodes)[None, :, :] * 1e9
    n_steps = len(result.t)
    other_windows = [
        diff[k : k + window_steps].min()
        for k in range(0, n_steps - window_steps, window_steps)
        if k != idx_start
    ]
    p10_other = np.percentile(other_windows, 10)
    assert window.min() >= p10_other - tolerance, (
        "la finestra subito dopo l'inizio della separazione ha una distanza "
        "minima anomala rispetto al resto della simulazione."
    )


def test_min_gap_softness_reduces_pileup(dispersion_config):
    """La repulsione morbida (Fix 2) deve eliminare l'atomo di probabilità
    a `min_node_gap` che produce il clamp rigido: prima del fix, il 13,6%
    delle coppie cadeva entro 1 cm da `min_node_gap` (misurato). Con
    `min_gap_softness = 0.15` la stessa frazione scende sotto il 3%.
    """
    result = simulate_mobility(dispersion_config)
    n_nodes = dispersion_config["group"]["n_nodes"]
    min_node_gap = dispersion_config["group"]["min_node_gap"]
    assert dispersion_config["group"]["min_gap_softness"] > 0.0

    iu, ju = np.triu_indices(n_nodes, k=1)
    pair_dist = result.distances[:, iu, ju].reshape(-1)

    frac_at_gap = np.mean(np.abs(pair_dist - min_node_gap) < 0.01)
    assert frac_at_gap < 0.03, (
        f"{frac_at_gap:.2%} delle coppie entro 1 cm da min_node_gap: la "
        "repulsione morbida non sta smussando il picco come atteso."
    )


def test_separation_monotonic_and_initial_range(separation_config):
    result = simulate_mobility(separation_config)
    sep_cfg = separation_config["separation"]
    node_id = sep_cfg["node_id"]
    start_time = sep_cfg["start_time"]
    ramp_end = start_time + sep_cfg["ramp_duration"]
    dt = separation_config["simulation"]["dt"]

    idx_start = int(np.searchsorted(result.t, start_time))
    other_nodes = [i for i in range(separation_config["group"]["n_nodes"]) if i != node_id]

    # a start_time il nodo è ancora nel range normale del gruppo
    normal_spread = np.max(result.s_nodes[idx_start, other_nodes]) - np.min(
        result.s_nodes[idx_start, other_nodes]
    )
    gap_at_start = abs(result.s_centroid[idx_start] - result.s_nodes[idx_start, node_id])
    assert gap_at_start <= normal_spread + separation_config["group"]["longitudinal_spread"]

    # dopo la rampa, il divario in ascissa curvilinea cresce in modo
    # monotono, campionando ogni 10 s (assorbe i micro-cali di velocità)
    idx_ramp_end = int(np.searchsorted(result.t, ramp_end))
    step_10s = max(int(round(10.0 / dt)), 1)
    sample_idx = np.arange(idx_ramp_end, len(result.t), step_10s)
    gap = result.s_centroid[sample_idx] - result.s_nodes[sample_idx, node_id]
    assert np.all(np.diff(gap) > 0)

    # la distanza euclidea media dagli altri nodi cresce nel primo minuto
    # dopo la fine della rampa
    idx_1min_later = min(idx_ramp_end + int(round(60.0 / dt)), len(result.t) - 1)
    dist_at_ramp_end = result.distances[idx_ramp_end, node_id, other_nodes]
    dist_1min_later = result.distances[idx_1min_later, node_id, other_nodes]
    assert np.mean(dist_1min_later) > np.mean(dist_at_ramp_end)


# ---------------------------------------------------------------------------
# 11-12: riproducibilità e forme
# ---------------------------------------------------------------------------


def test_reproducibility(short_config):
    r1 = simulate_mobility(short_config)
    r2 = simulate_mobility(short_config)
    assert np.array_equal(r1.positions, r2.positions)

    cfg_other_seed = copy.deepcopy(short_config)
    cfg_other_seed["simulation"]["seed"] = short_config["simulation"]["seed"] + 1
    r3 = simulate_mobility(cfg_other_seed)
    assert not np.array_equal(r1.positions, r3.positions)


def test_output_shapes(short_config):
    result = simulate_mobility(short_config)
    n_steps = len(result.t)
    n_nodes = short_config["group"]["n_nodes"]

    assert result.positions.shape == (n_steps, n_nodes, 2)
    assert result.s_nodes.shape == (n_steps, n_nodes)
    assert result.s_centroid.shape == (n_steps,)
    assert result.distances.shape == (n_steps, n_nodes, n_nodes)
    assert np.allclose(np.diagonal(result.distances, axis1=1, axis2=2), 0.0)
    assert np.allclose(result.distances, np.transpose(result.distances, (0, 2, 1)))

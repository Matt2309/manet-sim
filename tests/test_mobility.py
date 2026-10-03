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
    critically_damped_process,
    simulate_group_offsets,
    relative_speed,
    lateral_offset_speed,
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


def _turning_bound(track, window: float) -> float:
    """Massima rotazione totale |Δθ| dei vertici della polilinea in una
    finestra di ascissa curvilinea di larghezza `window`.
    """
    dx, dy = np.diff(track.x), np.diff(track.y)
    theta = np.unwrap(np.arctan2(dy, dx))
    vertex_s = track.s[1:-1]
    turning = np.abs(np.diff(theta))
    cumulative = np.concatenate([[0.0], np.cumsum(turning)])
    start = np.searchsorted(vertex_s, vertex_s, side="left")
    end = np.searchsorted(vertex_s, vertex_s + window, side="right")
    return float(np.max(cumulative[end] - cumulative[start]))


def test_heading_continuity(track, base_config):
    """La variazione dell'angolo fra due campioni di `s` distanti ds = 0,1 m
    non supera ds·T/W, con W = `heading_smoothing` e T la massima rotazione
    totale dei vertici della polilinea in una finestra di larghezza W.
    Derivazione: l'angolo lisciato è una media mobile di larghezza W, la
    cui derivata (θ(s+W/2) − θ(s−W/2))/W è limitata da T/W; l'interpolazione
    lineare fra i punti medi dei segmenti ne è una media, quindi mantiene il
    limite. Il riferimento a tratti di prima salta invece fino a 162° in
    un solo vertice e lo viola.
    """
    window = base_config["track"]["heading_smoothing"]
    ds = 0.1
    bound = ds * _turning_bound(track, window) / window

    s = np.arange(0.0, track.length, ds)
    sample = track.query(s)
    theta = np.unwrap(np.arctan2(sample.tangent[:, 1], sample.tangent[:, 0]))
    assert np.max(np.abs(np.diff(theta))) <= bound + 1e-9

    dx, dy = np.diff(track.x), np.diff(track.y)
    idx = np.clip(np.searchsorted(track.s, s, side="right") - 1, 0, len(dx) - 1)
    old_theta = np.unwrap(np.arctan2(dy[idx], dx[idx]))
    assert np.max(np.abs(np.diff(old_theta))) > bound


def test_frame_matches_segment_on_straight(base_config, tmp_path):
    """Lontano dall'angolo (più di W/2 più un passo), il riferimento
    lisciato coincide con la direzione del segmento (±1°). Percorso a L:
    rettilineo verso est, angolo di 90°, rettilineo verso nord.
    """
    lat0, lon0, step_m, n_leg = 45.0, 9.0, 2.8, 100
    dlon = np.degrees(step_m / (6371000.0 * np.cos(np.radians(lat0))))
    dlat = np.degrees(step_m / 6371000.0)
    points = [(lat0, lon0 + k * dlon) for k in range(n_leg + 1)]
    points += [(lat0 + k * dlat, lon0 + n_leg * dlon) for k in range(1, n_leg + 1)]
    pts = "\n".join(f'<trkpt lat="{la:.9f}" lon="{lo:.9f}"><ele>100.0</ele></trkpt>' for la, lo in points)
    gpx_path = tmp_path / "corner.gpx"
    gpx_path.write_text(
        '<?xml version="1.0"?><gpx version="1.1" creator="test" '
        f'xmlns="http://www.topografix.com/GPX/1/1"><trk><trkseg>{pts}</trkseg></trk></gpx>'
    )
    cfg = copy.deepcopy(base_config)
    cfg["track"]["gpx_file"] = str(gpx_path)
    corner_track = load_track(cfg)

    corner_s = n_leg * step_m
    guard = cfg["track"]["heading_smoothing"] / 2.0 + step_m
    s = np.linspace(0.0, corner_track.length, 2000)
    before, after = s < corner_s - guard, s > corner_s + guard
    tangent = corner_track.query(s).tangent
    angle = np.degrees(np.arctan2(tangent[:, 1], tangent[:, 0]))
    assert before.any() and after.any()
    assert np.all(np.abs(angle[before] - 0.0) < 1.0)
    assert np.all(np.abs(angle[after] - 90.0) < 1.0)
    # e in prossimità dell'angolo il riferimento ruota con continuità
    assert np.all(np.diff(angle[~before & ~after]) >= -1e-9)


# ---------------------------------------------------------------------------
# 7-8: processo del secondo ordine a smorzamento critico
# ---------------------------------------------------------------------------


def test_cd_stationary_variances():
    """Varianza stazionaria della posizione = sigma^2 e della velocità =
    sigma^2/tau^2, su più realizzazioni indipendenti per ridurre l'errore
    campionario (tau = 8 s: una sola traccia ha pochi tempi di correlazione).
    """
    rng = np.random.default_rng(123)
    sigma, tau, dt = 1.5, 8.0, 0.1
    x, v = critically_damped_process(n_steps=40_000, dt=dt, tau=tau, sigma=sigma, rng=rng, size=20)
    assert np.var(x) == pytest.approx(sigma**2, rel=0.05)
    assert np.var(v) == pytest.approx(sigma**2 / tau**2, rel=0.05)


@pytest.mark.parametrize("lag_in_taus", [0.5, 1.0, 2.0])
def test_cd_autocorrelation(lag_in_taus):
    """Autocorrelazione della posizione (1 + |t|/tau)·exp(-|t|/tau)."""
    rng = np.random.default_rng(321)
    dt, tau = 0.1, 8.0
    x, _ = critically_damped_process(n_steps=40_000, dt=dt, tau=tau, sigma=1.0, rng=rng, size=20)
    lag = int(round(lag_in_taus * tau / dt))
    xc = x - x.mean(axis=0)
    autocorr = np.mean(xc[:-lag] * xc[lag:]) / np.var(x)
    assert autocorr == pytest.approx((1.0 + lag_in_taus) * np.exp(-lag_in_taus), abs=0.05)


def test_cd_position_is_differentiable():
    """La posizione è derivabile: la deviazione standard della differenza
    finita a passo dt coincide con quella della velocità, sigma/tau (±5%).
    Con l'OU del primo ordine valeva invece sigma·sqrt(1 - e^(-2dt/tau))/dt,
    circa 12 volte più grande a dt = 0,1 s.
    """
    rng = np.random.default_rng(7)
    dt, tau, sigma = 0.1, 8.0, 2.0
    x, v = critically_damped_process(n_steps=40_000, dt=dt, tau=tau, sigma=sigma, rng=rng, size=20)
    finite_diff = np.diff(x, axis=0) / dt
    assert np.std(finite_diff) == pytest.approx(sigma / tau, rel=0.05)
    assert np.std(v) == pytest.approx(sigma / tau, rel=0.05)


def _free_group_offsets(tau_long: float, tau_lat: float, sigma_long: float, sigma_lat: float):
    """Scostamenti senza repulsione, su più gruppi indipendenti, per verificare
    la parte lineare della dinamica del gruppo con costanti di tempo diverse.
    """
    return simulate_group_offsets(
        n_steps=20_000,
        dt=0.1,
        tau_long=tau_long,
        tau_lat=tau_lat,
        sigma_long=sigma_long,
        sigma_lat=sigma_lat,
        rng=np.random.default_rng(11),
        n_nodes=8,
        repulsion=None,
        size=40,
    )


def test_group_offsets_variances_per_direction():
    """Con tau diverse per le due direzioni: var(posizione) = sigma^2 e
    var(velocità) = sigma^2/tau^2 (±5%), separatamente per ciascuna direzione.
    """
    tau_long, tau_lat, sigma_long, sigma_lat = 8.0, 2.5, 2.0, 0.5
    off = _free_group_offsets(tau_long, tau_lat, sigma_long, sigma_lat)
    assert np.var(off.long) == pytest.approx(sigma_long**2, rel=0.05)
    assert np.var(off.lat) == pytest.approx(sigma_lat**2, rel=0.05)
    assert np.var(off.vel_long) == pytest.approx(sigma_long**2 / tau_long**2, rel=0.05)
    assert np.var(off.vel_lat) == pytest.approx(sigma_lat**2 / tau_lat**2, rel=0.05)


@pytest.mark.parametrize("lag_in_taus", [0.5, 1.0, 2.0])
def test_group_offsets_autocorrelation_per_direction(lag_in_taus):
    """Autocorrelazione (1 + |t|/tau)·exp(-|t|/tau), con la tau di ciascuna
    direzione, ai ritardi tau/2, tau e 2·tau (±0,05).
    """
    dt = 0.1
    tau_long, tau_lat = 8.0, 2.5
    off = _free_group_offsets(tau_long, tau_lat, 1.0, 1.0)
    for series, tau in ((off.long, tau_long), (off.lat, tau_lat)):
        lag = int(round(lag_in_taus * tau / dt))
        xc = series - series.mean(axis=0)
        autocorr = np.mean(xc[:-lag] * xc[lag:]) / np.var(series)
        assert autocorr == pytest.approx((1.0 + lag_in_taus) * np.exp(-lag_in_taus), abs=0.05)


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


def test_relative_speed_realistic(separation_config):
    """Prima della separazione, la velocità di ogni nodo relativa al
    baricentro, misurata a passo dt, ha 99° percentile sotto
    `max_relative_speed_p99`. Con l'OU del primo ordine era ~15 m/s.
    """
    result = simulate_mobility(separation_config)
    idx_start = int(np.searchsorted(result.t, separation_config["separation"]["start_time"]))
    v_rel = relative_speed(result)[: idx_start - 1]
    limit = separation_config["group"]["max_relative_speed_p99"]
    limit_max = separation_config["group"]["max_relative_speed"]
    for node in range(v_rel.shape[1]):
        p99 = np.percentile(v_rel[:, node], 99)
        assert p99 < limit, f"nodo {node}: p99 della velocità relativa = {p99:.2f} m/s, oltre {limit} m/s"
        assert v_rel[:, node].max() < limit_max, (
            f"nodo {node}: massimo della velocità relativa = {v_rel[:, node].max():.2f} m/s, oltre {limit_max} m/s"
        )


def test_lateral_2d_speed_realistic(separation_config):
    """Prima della separazione, la velocità 2D dello scostamento laterale di
    ogni nodo, a passo dt, ha p99 sotto `max_relative_speed_p99` e massimo
    sotto `max_relative_speed`. Con tangente e normale a tratti il
    massimo arrivava a ~11 m/s, per i salti ai vertici in curva.
    """
    result = simulate_mobility(separation_config)
    track = load_track(separation_config)
    idx_start = int(np.searchsorted(result.t, separation_config["separation"]["start_time"]))
    v = lateral_offset_speed(result, track)[: idx_start - 1]
    limit = separation_config["group"]["max_relative_speed_p99"]
    limit_max = separation_config["group"]["max_relative_speed"]
    for node in range(v.shape[1]):
        p99 = np.percentile(v[:, node], 99)
        assert p99 < limit, f"nodo {node}: p99 = {p99:.2f} m/s, oltre {limit} m/s"
        assert v[:, node].max() < limit_max, f"nodo {node}: massimo = {v[:, node].max():.2f} m/s, oltre {limit_max} m/s"


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


def _pair_distances(result, n_nodes: int) -> np.ndarray:
    iu, ju = np.triu_indices(n_nodes, k=1)
    return result.distances[:, iu, ju].reshape(-1)


def test_repulsion_keeps_distance(dispersion_config):
    """Senza separazione, la repulsione fra i corridori tiene le distanze:
    1° percentile della distanza fra coppie almeno 1,1 m e al massimo lo
    0,05% dei campioni sotto 1 m. La distanza è quella vera nel piano, quindi
    include l'effetto della curvatura della traccia.
    """
    result = simulate_mobility(dispersion_config)
    pair_dist = _pair_distances(result, dispersion_config["group"]["n_nodes"])
    assert np.percentile(pair_dist, 1) >= 1.1
    assert np.mean(pair_dist < 1.0) <= 0.0005
    assert np.all(pair_dist > 0.0)


def test_repulsion_acts_on_separated_node(separation_config):
    """Il nodo separato esercita e subisce la repulsione come gli altri: nei
    primi secondi dopo `start_time`, quando è ancora dentro il gruppo, la
    distanza dagli altri nodi non scende sotto 1 m e non è un outlier rispetto
    alle finestre di pari durata del resto della simulazione (un buco di
    repulsione produrrebbe una distanza anomala).
    """
    result = simulate_mobility(separation_config)
    n_nodes = separation_config["group"]["n_nodes"]
    node_id = separation_config["separation"]["node_id"]
    dt = separation_config["simulation"]["dt"]
    other_nodes = [i for i in range(n_nodes) if i != node_id]

    idx_start = int(np.searchsorted(result.t, separation_config["separation"]["start_time"]))
    window_steps = max(int(round(5.0 / dt)), 1)
    window = result.distances[idx_start : idx_start + window_steps, node_id][:, other_nodes]
    assert window.min() >= 1.0, (
        f"distanza minima nodo separato-gruppo nei primi 5 s dopo start_time = "
        f"{window.min():.3f} m: la repulsione non agisce sul nodo separato."
    )

    diff = result.distances + np.eye(n_nodes)[None, :, :] * 1e9
    n_steps = len(result.t)
    other_windows = [
        diff[k : k + window_steps].min()
        for k in range(0, n_steps - window_steps, window_steps)
        if k != idx_start
    ]
    assert window.min() >= np.percentile(other_windows, 10) - 0.1


def test_no_pileup_at_min_node_gap(dispersion_config):
    """La forza è continua e senza gradini, quindi la distribuzione delle
    distanze non ha accumulo a `min_node_gap` (quello che produceva il
    vincolo rigido, ed era la ragione del vecchio margine casuale): la
    densità nell'intorno di ±1 cm non supera 1,5 volte quella delle fasce
    vicine, e meno del 3% delle coppie ci cade dentro.
    """
    result = simulate_mobility(dispersion_config)
    min_node_gap = dispersion_config["group"]["min_node_gap"]
    pair_dist = _pair_distances(result, dispersion_config["group"]["n_nodes"])

    near = np.abs(pair_dist - min_node_gap) < 0.01
    sides = (np.abs(pair_dist - min_node_gap) >= 0.01) & (np.abs(pair_dist - min_node_gap) < 0.11)
    density_near = near.mean() / 0.02
    density_sides = sides.mean() / 0.20
    assert near.mean() < 0.03
    assert density_near < 1.5 * density_sides, (
        f"densità a min_node_gap {density_near:.3f} contro {density_sides:.3f} nelle fasce vicine: accumulo."
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


# ---------------------------------------------------------------------------
# direzione di marcia (headings)
# ---------------------------------------------------------------------------


def test_headings_are_unit_vectors(short_config):
    result = simulate_mobility(short_config)
    n_nodes = short_config["group"]["n_nodes"]
    assert result.headings.shape == (len(result.t), n_nodes, 2)
    assert np.allclose(np.linalg.norm(result.headings, axis=-1), 1.0, atol=1e-9)


def test_headings_on_straight_track(short_config, tmp_path):
    """Su un tracciato rettilineo (latitudine costante, quindi y = 0 nella
    proiezione) la direzione di marcia è esattamente +x per ogni nodo e
    istante, e la direzione dello spostamento netto di ogni nodo coincide
    con essa. Sul tracciato reale l'uguaglianza vale solo in modo
    approssimato, perché lo spostamento contiene anche la deriva laterale.
    """
    lat = 45.0
    n_points = 600
    step_m = 2.8
    dlon = np.degrees(step_m / (6371000.0 * np.cos(np.radians(lat))))
    pts = "\n".join(
        f'<trkpt lat="{lat}" lon="{9.0 + k * dlon:.9f}"><ele>100.0</ele></trkpt>'
        for k in range(n_points)
    )
    gpx = (
        '<?xml version="1.0"?><gpx version="1.1" creator="test" '
        'xmlns="http://www.topografix.com/GPX/1/1"><trk><trkseg>'
        f"{pts}</trkseg></trk></gpx>"
    )
    gpx_path = tmp_path / "straight.gpx"
    gpx_path.write_text(gpx)

    cfg = copy.deepcopy(short_config)
    cfg["track"]["gpx_file"] = str(gpx_path)
    cfg["simulation"]["duration"] = 300.0
    result = simulate_mobility(cfg)

    assert np.allclose(result.headings[..., 0], 1.0, atol=1e-9)
    assert np.allclose(result.headings[..., 1], 0.0, atol=1e-9)

    net = result.positions[-1] - result.positions[0]
    net_dir = net / np.linalg.norm(net, axis=-1, keepdims=True)
    assert np.allclose(net_dir, result.headings[0], atol=1e-2)

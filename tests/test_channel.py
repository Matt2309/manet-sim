"""Test del Blocco 2 (canale).

Ogni test verifica un fatto noto indipendentemente (teoria o formula
chiusa), non solo l'assenza di eccezioni. Dove possibile si usano posizioni
costruite a mano o configurazioni ridotte, per tenere veloce la suite.
"""

import copy
import math
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from src.channel import (
    generate_shadow_field,
    independent_shadowing,
    link_shadowing,
    link_variance_theory,
    other_bodies_loss,
    own_body_loss,
    reference_path_loss,
    rician_fading_db,
    shadow_field_sigma_p,
    simulate_channel,
)
from src.mobility import MobilityResult, load_config, simulate_mobility

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
C_LIGHT = 299792458.0


# ---------------------------------------------------------------------------
# Fixture e utilità
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def base_config():
    return load_config(CONFIG_PATH)


@pytest.fixture
def cfg(base_config):
    return copy.deepcopy(base_config)


@pytest.fixture(scope="session")
def short_mobility(base_config):
    """Simulazione di mobilità breve (60 s), senza separazione."""
    c = copy.deepcopy(base_config)
    c["simulation"]["duration"] = 60.0
    c["separation"]["enabled"] = False
    return simulate_mobility(c)


def _quiet(cfg: dict) -> dict:
    """Disattiva tutte le componenti casuali: resta la sola attenuazione."""
    ch = cfg["channel"]
    ch["shadowing"]["enabled"] = False
    ch["body"]["own"]["enabled"] = False
    ch["body"]["others"]["enabled"] = False
    ch["fading"]["enabled"] = False
    ch["device_offset"]["enabled"] = False
    return cfg


def _mobility(positions: np.ndarray, headings=None, dt: float = 0.1) -> MobilityResult:
    """`MobilityResult` costruito a mano da posizioni (T, N, 2)."""
    positions = np.asarray(positions, dtype=float)
    n_steps, n_nodes, _ = positions.shape
    if headings is None:
        headings = np.tile([1.0, 0.0], (n_steps, n_nodes, 1))
    diff = positions[:, :, None, :] - positions[:, None, :, :]
    return MobilityResult(
        t=np.arange(n_steps) * dt,
        positions=positions,
        s_nodes=np.zeros((n_steps, n_nodes)),
        s_centroid=np.zeros(n_steps),
        distances=np.linalg.norm(diff, axis=-1),
        headings=np.asarray(headings, dtype=float),
        lateral_offsets=np.zeros((n_steps, n_nodes)),
        metadata={},
    )


def _offdiag(arr: np.ndarray) -> np.ndarray:
    n = arr.shape[1]
    return arr[:, ~np.eye(n, dtype=bool)]


def _exp_cov_link_matrix(a1, b1, a2, b2, delta: float, n: int = 500) -> float:
    """Integrale doppio numerico (punto medio) di exp(-r/delta) fra due
    segmenti, normalizzato come l'integrale dello shadowing:
    (1/sqrt(d1 d2)) * int int exp(-|x1(u) - x2(v)|/delta) du dv.
    """
    d1 = np.linalg.norm(b1 - a1)
    d2 = np.linalg.norm(b2 - a2)
    u = (np.arange(n) + 0.5) / n
    p1 = a1[None, :] + u[:, None] * (b1 - a1)[None, :]
    p2 = a2[None, :] + u[:, None] * (b2 - a2)[None, :]
    r = np.linalg.norm(p1[:, None, :] - p2[None, :, :], axis=-1)
    return d1 * d2 / math.sqrt(d1 * d2) * float(np.mean(np.exp(-r / delta)))


# ---------------------------------------------------------------------------
# 1. Attenuazione con la distanza
# ---------------------------------------------------------------------------


def test_path_loss_matches_formula(cfg):
    ch = cfg["channel"]
    pl0 = reference_path_loss(ch["frequency"], ch["path_loss"]["reference_distance"])
    assert pl0 == pytest.approx(40.2, abs=0.1)

    n = ch["path_loss"]["exponent"]
    d0 = ch["path_loss"]["reference_distance"]
    distances = np.array([0.3, 0.5, 1.0, 2.57, 7.08, 100.0, 400.0])
    positions = np.zeros((len(distances), 2, 2))
    positions[:, 1, 0] = distances
    result = simulate_channel(_mobility(positions), _quiet(cfg))

    # formula scritta indipendentemente (c fisica, nessun valore del codice)
    pl0_ref = 20.0 * np.log10(4.0 * np.pi * d0 * 2.437e9 / C_LIGHT)
    expected = 10.0 - (pl0_ref + 10.0 * n * np.log10(np.maximum(distances, d0) / d0))
    assert np.allclose(result.rssi_true[:, 0, 1], expected, atol=1e-9)
    assert np.allclose(result.rssi_path_loss[:, 0, 1], expected, atol=1e-9)

    # sotto d0 resta costante
    assert result.rssi_true[0, 0, 1] == pytest.approx(result.rssi_true[2, 0, 1], abs=1e-12)
    assert result.rssi_true[1, 0, 1] == pytest.approx(result.rssi_true[2, 0, 1], abs=1e-12)

    # pendenza in log10(d) = -10 n
    above = distances >= d0
    slope = np.polyfit(np.log10(distances[above]), result.rssi_true[above, 0, 1], 1)[0]
    assert slope == pytest.approx(-10.0 * n, abs=1e-9)


# ---------------------------------------------------------------------------
# 2. Mappa: varianza e correlazione
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def big_fields():
    """Tre realizzazioni di una mappa 600x600 m a passo 0,5 m."""
    sigma, delta = 4.0, 7.0
    sigma_p = shadow_field_sigma_p(sigma, delta)
    rng = np.random.default_rng(2024)
    fields = [
        generate_shadow_field((0.0, 600.0), (0.0, 600.0), sigma_p, delta, 0.5, 70.0, rng).values.astype(float)
        for _ in range(3)
    ]
    return fields, sigma_p, delta, 0.5


def test_field_is_float32_and_zero_mean(big_fields):
    fields, _, _, _ = big_fields
    sf = generate_shadow_field((0.0, 50.0), (0.0, 50.0), 1.0, 7.0, 1.0, 70.0, np.random.default_rng(0))
    assert sf.values.dtype == np.float32
    assert abs(np.mean(np.concatenate([f.ravel() for f in fields]))) < 0.1 * fields[0].std()


def test_field_variance(big_fields):
    fields, sigma_p, _, _ = big_fields
    var = np.mean([f.var() for f in fields])
    assert var == pytest.approx(sigma_p**2, rel=0.10)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("lag_in_deltas", [0.5, 1.0, 2.0])
def test_field_autocorrelation_is_exponential(big_fields, axis, lag_in_deltas):
    fields, _, delta, res = big_fields
    shift = int(round(lag_in_deltas * delta / res))
    r = shift * res
    corr = []
    for f in fields:
        f0 = f - f.mean()
        a = np.take(f0, range(0, f0.shape[axis] - shift), axis=axis)
        b = np.take(f0, range(shift, f0.shape[axis]), axis=axis)
        corr.append(np.mean(a * b) / f0.var())
    assert np.mean(corr) == pytest.approx(math.exp(-r / delta), abs=0.07)


# ---------------------------------------------------------------------------
# 3. Varianza del link in funzione della lunghezza
# ---------------------------------------------------------------------------


def test_link_std_vs_length(cfg):
    sh = cfg["channel"]["shadowing"]
    sigma, delta = sh["sigma"], sh["correlation_distance"]
    sigma_p = shadow_field_sigma_p(sigma, delta)
    rng = np.random.default_rng(7)
    n_real, n_seg = 20, 200
    for length in (2.0, 7.0, 30.0, 150.0):
        samples = []
        for _ in range(n_real):
            sf = generate_shadow_field(
                (0.0, 500.0), (0.0, 500.0), sigma_p, delta, sh["grid_resolution"], sh["fft_padding"], rng
            )
            a = rng.uniform(175.0, 325.0, size=(n_seg, 2))
            ang = rng.uniform(0.0, 2.0 * np.pi, size=n_seg)
            b = a + length * np.stack([np.cos(ang), np.sin(ang)], axis=1)
            samples.append(link_shadowing(sf, a, b, sh))
        std = np.std(np.concatenate(samples))
        theory = math.sqrt(link_variance_theory(length, sigma, delta))
        assert std == pytest.approx(theory, rel=0.12), f"lunghezza {length} m"


# ---------------------------------------------------------------------------
# 4. Correlazione fra link vicini (il test che motiva il modello)
# ---------------------------------------------------------------------------

_A = np.array([100.0, 150.0])
_B1 = np.array([180.0, 150.0])
_B2 = np.array([180.0, 153.0])  # a 3 m da B1


def test_neighbouring_links_correlation_field(cfg):
    sh = cfg["channel"]["shadowing"]
    sigma, delta = sh["sigma"], sh["correlation_distance"]
    sigma_p = shadow_field_sigma_p(sigma, delta)
    rng = np.random.default_rng(11)

    a = np.stack([_A, _A])
    b = np.stack([_B1, _B2])
    n_real = 500
    s = np.empty((n_real, 2))
    for k in range(n_real):
        sf = generate_shadow_field(
            (60.0, 220.0), (100.0, 200.0), sigma_p, delta, sh["grid_resolution"], sh["fft_padding"], rng
        )
        s[k] = link_shadowing(sf, a, b, sh)
    empirical = np.corrcoef(s[:, 0], s[:, 1])[0, 1]

    cov12 = _exp_cov_link_matrix(_A, _B1, _A, _B2, delta)
    var1 = _exp_cov_link_matrix(_A, _B1, _A, _B1, delta)
    var2 = _exp_cov_link_matrix(_A, _B2, _A, _B2, delta)
    theory = cov12 / math.sqrt(var1 * var2)

    assert theory > 0.5  # i link condividono quasi tutto il percorso
    assert empirical == pytest.approx(theory, abs=0.1)


def test_neighbouring_links_correlation_independent(cfg):
    sh = cfg["channel"]["shadowing"]
    positions = np.array([[_A, _B1, _B2]])  # (1, 3, 2): link 0-1 e 0-2 condividono il nodo 0
    rng = np.random.default_rng(12)
    s = np.array(
        [independent_shadowing(positions, sh["sigma"], sh["correlation_distance"], rng)[0, 0, 1:] for _ in range(2000)]
    )
    assert np.corrcoef(s[:, 0], s[:, 1])[0, 1] == pytest.approx(0.0, abs=0.1)


# ---------------------------------------------------------------------------
# 5. Ritorno nello stesso punto
# ---------------------------------------------------------------------------


def test_field_same_positions_same_shadowing(cfg):
    rng = np.random.default_rng(5)
    p0 = rng.uniform(0.0, 300.0, size=(4, 2))
    p1 = rng.uniform(0.0, 300.0, size=(4, 2))
    positions = np.stack([p0, p1, p0])
    ch = cfg["channel"]
    assert ch["shadowing"]["mode"] == "field"
    result = simulate_channel(_mobility(positions), cfg)
    assert np.array_equal(result.shadowing[0], result.shadowing[2], equal_nan=True)
    assert not np.array_equal(result.shadowing[0], result.shadowing[1], equal_nan=True)


# ---------------------------------------------------------------------------
# 6. Modalità indipendente
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def parallel_run(base_config):
    """5 nodi che avanzano in parallelo di 0,05 m a passo: ogni coppia
    percorre 0,1 m per passo (somma dei due estremi)."""
    sh = base_config["channel"]["shadowing"]
    n_steps, n_nodes, step = 100_000, 5, 0.05
    k = np.arange(n_steps)
    positions = np.zeros((n_steps, n_nodes, 2))
    positions[:, :, 0] = (step * k)[:, None]
    positions[:, :, 1] = 10.0 * np.arange(n_nodes)[None, :]
    rng = np.random.default_rng(99)
    s = independent_shadowing(positions, sh["sigma"], sh["correlation_distance"], rng)
    return s, 2.0 * step, sh


def test_independent_stationary_std(parallel_run):
    s, _, sh = parallel_run
    iu, ju = np.triu_indices(s.shape[1], k=1)
    assert np.std(s[:, iu, ju]) == pytest.approx(sh["sigma"], rel=0.10)


@pytest.mark.parametrize("lag_in_deltas", [0.5, 1.0, 2.0])
def test_independent_autocorrelation_in_distance(parallel_run, lag_in_deltas):
    s, per_step, sh = parallel_run
    delta = sh["correlation_distance"]
    lag = int(round(lag_in_deltas * delta / per_step))
    iu, ju = np.triu_indices(s.shape[1], k=1)
    x = s[:, iu, ju]
    x = x - x.mean(axis=0)
    corr = np.mean(x[:-lag] * x[lag:]) / np.var(x)
    assert corr == pytest.approx(math.exp(-lag * per_step / delta), abs=0.05)


def test_independent_static_nodes_constant_shadowing(cfg):
    positions = np.tile(np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 20.0]]), (100, 1, 1))
    sh = cfg["channel"]["shadowing"]
    s = independent_shadowing(positions, sh["sigma"], sh["correlation_distance"], np.random.default_rng(3))
    assert np.ptp(s, axis=0).max() == 0.0
    assert np.std(s[0, [0, 0, 1], [1, 2, 2]]) > 0.0  # coppie diverse, valori diversi


# ---------------------------------------------------------------------------
# 7. Reciprocità
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["field", "independent"])
def test_reciprocity(cfg, short_mobility, mode):
    cfg["channel"]["shadowing"]["mode"] = mode
    result = simulate_channel(short_mobility, cfg)
    for name in ("shadowing", "body_own", "body_others"):
        arr = getattr(result, name)
        np.testing.assert_array_equal(arr, np.transpose(arr, (0, 2, 1)), err_msg=name)
    assert np.nanmax(np.abs(result.shadowing)) > 0.0
    assert np.nanmax(result.body_own) > 0.0

    f = result.fading
    assert not np.allclose(f, np.transpose(f, (0, 2, 1)), equal_nan=True)
    upper = f[:, 0, 1]
    lower = f[:, 1, 0]
    assert abs(np.corrcoef(upper, lower)[0, 1]) < 0.15


# ---------------------------------------------------------------------------
# 8. Torso proprio
# ---------------------------------------------------------------------------


def _torso_case(cfg, other_pos, other_heading):
    """Nodo 0 in (0,0) che corre verso est; l'altro nodo in `other_pos`, con
    una direzione scelta perché la sua perdita sia nulla: il totale è la
    sola perdita dell'estremo 0."""
    positions = np.array([[[0.0, 0.0], other_pos]])
    headings = np.array([[[1.0, 0.0], other_heading]])
    return own_body_loss(positions, headings, cfg["channel"]["body"]["own"])[0, 0, 1]


def test_own_body_lobe(cfg):
    own = cfg["channel"]["body"]["own"]
    max_loss, n = own["max_loss"], own["lobe_exponent"]
    assert own["mount_side"] == "right"

    # altro a nord (a sinistra di chi corre verso est): torso massimo
    assert _torso_case(cfg, [0.0, 10.0], [1.0, 0.0]) == pytest.approx(max_loss, abs=1e-9)
    # altro a sud (a destra): nessuna perdita
    assert _torso_case(cfg, [0.0, -10.0], [-1.0, 0.0]) == pytest.approx(0.0, abs=1e-9)
    # altro davanti: metà lobo
    assert _torso_case(cfg, [10.0, 0.0], [0.0, -1.0]) == pytest.approx(max_loss * 0.5**n, abs=1e-9)


def test_own_body_mount_side_left_mirrors(cfg):
    cfg["channel"]["body"]["own"]["mount_side"] = "left"
    max_loss = cfg["channel"]["body"]["own"]["max_loss"]
    # con la scheda a sinistra il torso blocca verso destra (sud)
    # (la direzione dell'altro nodo è scelta perché la sua perdita sia nulla)
    assert _torso_case(cfg, [0.0, -10.0], [1.0, 0.0]) == pytest.approx(max_loss, abs=1e-9)
    assert _torso_case(cfg, [0.0, 10.0], [-1.0, 0.0]) == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# 9. Altri corridori
# ---------------------------------------------------------------------------


def _others(cfg, positions):
    return other_bodies_loss(np.array([positions], dtype=float), cfg["channel"]["body"]["others"])[0]


def test_other_runner_in_the_middle(cfg):
    o = cfg["channel"]["body"]["others"]
    loss = _others(cfg, [[0, 0], [10, 0], [5, 0]])[0, 1]
    expected = o["loss"] / (1.0 + math.exp(-o["radius"] / o["transition_width"]))
    assert loss == pytest.approx(expected, abs=1e-9)
    assert loss == pytest.approx(o["loss"], rel=0.02)


def test_other_runner_far_from_segment(cfg):
    o = cfg["channel"]["body"]["others"]
    loss = _others(cfg, [[0, 0], [10, 0], [5, 2 * o["radius"]]])[0, 1]
    assert loss < 0.02 * o["loss"]


def test_other_runner_beyond_endpoint(cfg):
    assert _others(cfg, [[0, 0], [10, 0], [12, 0]])[0, 1] == 0.0
    assert _others(cfg, [[0, 0], [10, 0], [-2, 0]])[0, 1] == 0.0


def test_runners_in_line_hit_the_cap(cfg):
    """Il tetto (12 dB) vale per più corridori in fila: con 8 dB ciascuno già due
    lo raggiungono, e anche tre danno esattamente il tetto; un corridore solo no.
    """
    o = cfg["channel"]["body"]["others"]
    assert 2 * o["loss"] > o["max_total_loss"]  # il tetto è davvero attivo
    three = _others(cfg, [[0, 0], [10, 0], [3, 0], [5, 0], [7, 0]])[0, 1]
    assert three == pytest.approx(o["max_total_loss"], abs=1e-9)
    two = _others(cfg, [[0, 0], [10, 0], [3, 0], [7, 0]])[0, 1]
    assert two == pytest.approx(o["max_total_loss"], abs=1e-9)
    # un solo corridore in mezzo sta sotto il tetto
    one = _others(cfg, [[0, 0], [10, 0], [5, 0]])[0, 1]
    assert one == pytest.approx(o["loss"] / (1.0 + math.exp(-o["radius"] / o["transition_width"])), abs=1e-9)
    assert one < o["max_total_loss"]


# ---------------------------------------------------------------------------
# 10. Variazioni rapide
# ---------------------------------------------------------------------------


def test_rician_fading_matches_scipy():
    k_db = 6.0
    k = 10.0 ** (k_db / 10.0)
    f = rician_fading_db((400_000,), k_db, np.random.default_rng(1))

    assert np.mean(10.0 ** (f / 10.0)) == pytest.approx(1.0, rel=0.02)

    dist = stats.rice(b=math.sqrt(2.0 * k), scale=math.sqrt(1.0 / (2.0 * (k + 1.0))))
    mean_db = dist.expect(lambda r: 20.0 * np.log10(r))
    second = dist.expect(lambda r: (20.0 * np.log10(r)) ** 2)
    std_db = math.sqrt(second - mean_db**2)
    assert np.mean(f) == pytest.approx(mean_db, abs=0.03)
    assert np.std(f) == pytest.approx(std_db, rel=0.02)


# ---------------------------------------------------------------------------
# 11. Scarti delle schede
# ---------------------------------------------------------------------------


def test_device_offsets_break_reciprocity(cfg, short_mobility):
    cfg["channel"]["fading"]["enabled"] = False
    result = simulate_channel(short_mobility, cfg)
    tx, rx = result.tx_offset, result.rx_offset
    assert np.std(tx) > 0.0
    n = len(tx)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = result.rssi_true[:, i, j] - result.rssi_true[:, j, i]
            assert np.allclose(diff, (tx[i] + rx[j]) - (tx[j] + rx[i]), atol=1e-9)


# ---------------------------------------------------------------------------
# 12. Lettura
# ---------------------------------------------------------------------------


def test_measurement_saturation_and_quantization(cfg, short_mobility):
    cfg["channel"]["tx_power"] = 30.0  # forza la saturazione dei link corti
    m = cfg["channel"]["measurement"]
    result = simulate_channel(short_mobility, cfg)
    meas = _offdiag(result.rssi_measured)
    assert np.max(meas) <= m["saturation_dbm"]
    assert np.any(meas == m["saturation_dbm"])
    ratio = meas / m["quantization_step"]
    assert np.allclose(ratio, np.round(ratio), atol=1e-9)
    assert result.metadata["saturated_fraction"] > 0.0

    # quantizzazione più grossolana: multipli di 3 dB
    cfg["channel"]["measurement"]["quantization_step"] = 3.0
    coarse = _offdiag(simulate_channel(short_mobility, cfg).rssi_measured) / 3.0
    assert np.allclose(coarse, np.round(coarse), atol=1e-9)


# ---------------------------------------------------------------------------
# 13. Riproducibilità e indipendenza dei generatori
# ---------------------------------------------------------------------------


def test_reproducibility(cfg, short_mobility):
    r1 = simulate_channel(short_mobility, cfg)
    r2 = simulate_channel(short_mobility, cfg)
    np.testing.assert_array_equal(r1.rssi_true, r2.rssi_true)

    cfg2 = copy.deepcopy(cfg)
    cfg2["simulation"]["seed"] += 1
    r3 = simulate_channel(short_mobility, cfg2)
    assert not np.array_equal(r1.rssi_true, r3.rssi_true, equal_nan=True)


def test_generators_are_independent(cfg, short_mobility):
    full = simulate_channel(short_mobility, cfg)

    no_fading = copy.deepcopy(cfg)
    no_fading["channel"]["fading"]["enabled"] = False
    r = simulate_channel(short_mobility, no_fading)
    np.testing.assert_array_equal(r.shadowing, full.shadowing)
    np.testing.assert_array_equal(r.tx_offset, full.tx_offset)
    np.testing.assert_array_equal(r.rx_offset, full.rx_offset)
    assert np.array_equal(r.shadow_field.values, full.shadow_field.values)

    no_shadow = copy.deepcopy(cfg)
    no_shadow["channel"]["shadowing"]["enabled"] = False
    r = simulate_channel(short_mobility, no_shadow)
    np.testing.assert_array_equal(r.fading, full.fading)
    np.testing.assert_array_equal(r.tx_offset, full.tx_offset)

    no_offsets = copy.deepcopy(cfg)
    no_offsets["channel"]["device_offset"]["enabled"] = False
    r = simulate_channel(short_mobility, no_offsets)
    np.testing.assert_array_equal(r.fading, full.fading)
    np.testing.assert_array_equal(r.shadowing, full.shadowing)


# ---------------------------------------------------------------------------
# 14. Forme e componenti disattivate
# ---------------------------------------------------------------------------


def test_output_shapes_and_nan_diagonal(cfg, short_mobility):
    result = simulate_channel(short_mobility, cfg)
    n_steps = len(short_mobility.t)
    n = cfg["group"]["n_nodes"]
    idx = np.arange(n)
    for name in (
        "rssi_true",
        "rssi_measured",
        "rssi_path_loss",
        "shadowing",
        "body_own",
        "body_others",
        "fading",
        "obstacles",
    ):
        arr = getattr(result, name)
        assert arr.shape == (n_steps, n, n), name
        assert np.all(np.isnan(arr[:, idx, idx])), name
        assert np.all(np.isfinite(_offdiag(arr))), name
    assert result.tx_offset.shape == (n,)
    assert result.rx_offset.shape == (n,)
    assert np.array_equal(result.t, short_mobility.t)


def test_disabled_components_are_zero(cfg, short_mobility):
    result = simulate_channel(short_mobility, _quiet(cfg))
    for name in ("shadowing", "body_own", "body_others", "fading", "obstacles"):
        assert np.all(_offdiag(getattr(result, name)) == 0.0), name
    assert np.all(result.tx_offset == 0.0) and np.all(result.rx_offset == 0.0)
    np.testing.assert_allclose(_offdiag(result.rssi_true), _offdiag(result.rssi_path_loss))


def test_obstacles_not_implemented(cfg, short_mobility):
    cfg["channel"]["obstacles"]["enabled"] = True
    with pytest.raises(NotImplementedError):
        simulate_channel(short_mobility, cfg)

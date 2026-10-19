"""Test del Blocco 5: regole di fusione, livello relativo, spegnimento lento, robustezza, monotonia, cache.

Ogni test verifica un fatto noto (calcolo a mano, formula chiusa o confronto indipendente).
"""

import copy
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from src.mobility import load_config

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "default.yaml"
DATA = Path(__file__).resolve().parent / "data"


@pytest.fixture(scope="session")
def base_config():
    return load_config(CONFIG_PATH)


@pytest.fixture
def cfg(base_config):
    return copy.deepcopy(base_config)


def _load_reference_module():
    spec = importlib.util.spec_from_file_location("make_block4_reference", DATA / "make_block4_reference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 1. Default = Blocco 4
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", ["none", "slowdown", "stop"])
def test_default_reproduces_block4(scenario):
    """Con la config di default, Kalman, indici fusi, allarmi e metriche coincidono ESATTAMENTE con il Blocco 4.

    Il riferimento `tests/data/block4_reference.npz` è stato generato con il codice del commit 16c6180, prima del Blocco 5
    (`tests/data/make_block4_reference.py`): una corsa breve per scenario, su tutte le modalità, gli ambiti e le soglie.
    """
    module = _load_reference_module()
    config = module.reference_config()
    assert config["kalman"]["robust"] == "none" and config["fusion"]["rule"] == "weighted_mean"
    assert config["detector"]["reference"] == "absolute" and config["kalman"]["r_inflation"] == 1.0
    fresh = module.reference_outputs(scenario, config)
    with np.load(DATA / "block4_reference.npz") as ref:
        names = [k.split("/", 1)[1] for k in ref.files if k.startswith(scenario + "/")]
        assert set(names) == set(fresh)
        for name in names:
            np.testing.assert_array_equal(fresh[name], ref[f"{scenario}/{name}"], err_msg=name)


# ---------------------------------------------------------------------------
# 2. Regole di fusione
# ---------------------------------------------------------------------------


def test_fusion_rules_by_hand():
    """`median` e `max` su voci sintetiche, con 4 voci e con voci escluse per età; peso `weighted_mean` invariato."""
    from src.detector import fuse
    from src.fusion import combine

    values = np.array([-70.0, -80.0, -60.0, -90.0])
    slopes = np.array([-0.1, -0.2, -0.3, -0.4])
    variances = np.array([1.0, 1.0, 4.0, 1.0])
    ages = np.zeros(4)
    # 4 voci: la mediana è la media delle due centrali (-80, -70); il massimo è -60 (con la sua pendenza)
    level, slope, _, n = fuse(values, slopes, variances, ages, 2.0, rule="median")
    assert (level, slope, n) == (-75.0, pytest.approx(-0.15), 4)
    level, slope, var, n = fuse(values, slopes, variances, ages, 2.0, rule="max")
    assert (level, slope, var, n) == (-60.0, -0.3, 4.0, 4)
    # la voce da -60 è troppo vecchia: restano -70, -80, -90
    ages = np.array([0.0, 0.0, 3.0, 0.0])
    assert fuse(values, slopes, variances, ages, 2.0, rule="median")[0] == -80.0
    assert fuse(values, slopes, variances, ages, 2.0, rule="max")[0] == -70.0
    # media pesata 1/varianza: (-70 - 80 - 90) / 3 con pesi uguali
    assert fuse(values, slopes, variances, ages, 2.0, rule="weighted_mean")[0] == pytest.approx(-80.0)
    # nessuna voce valida: NaN
    none = fuse(values, slopes, variances, np.full(4, 9.0), 2.0, rule="median")
    assert np.isnan(none[0]) and none[3] == 0
    # più bersagli insieme (asse delle voci per ultimo)
    stacked = combine(np.stack([values, values[::-1] - 10.0]), np.zeros((2, 4)), np.ones((2, 4)), np.ones((2, 4), bool), "median")
    np.testing.assert_allclose(stacked[0], [-75.0, -85.0])
    with pytest.raises(ValueError):
        combine(values, slopes, variances, np.ones(4, bool), "altra")


def _synthetic_inputs(values, new_every=5, dt=0.1, n_obs=2, n_tgt=2):
    """Ingressi sintetici del livello relativo: voci `values` (T, K) del link del bersaglio 1 visto dall'osservatore 0."""
    from src.detector import DetectionInputs

    n_steps, n_links = values.shape
    shape = (n_steps, n_obs, n_tgt, n_links)
    value = np.full(shape, np.nan)
    value[:, 0, 1, :] = values
    valid = np.zeros(shape, dtype=bool)
    valid[:, 0, 1, :] = True
    new = np.zeros(shape, dtype=bool)
    new[::new_every, 0, 1, :] = True
    entries = {"value": value, "var": np.ones(shape), "valid": valid, "new": new & valid}
    return DetectionInputs(
        t=np.arange(n_steps) * dt, level={}, slope={}, n_entries={}, silence=np.zeros((n_steps, n_obs, n_tgt)),
        rx_ok=np.ones((n_steps, n_obs, n_tgt), dtype=bool), valid=~np.eye(n_obs, n_tgt, dtype=bool), entries=entries,
    )  # fmt: skip


def test_relative_max_picks_smallest_drop_and_median_by_hand(cfg):
    """Con il livello relativo `max` (miglior osservatore sull'indice -calo) sceglie il calo più piccolo; `median` quello centrale."""
    from src.relative import relative_trace

    cfg["detector"]["reference"] = "relative"
    n_steps = 800
    levels = np.array([-60.0, -60.0, -60.0, -60.0])
    series = np.tile(levels, (n_steps, 1))
    series[600:] = levels + np.array([-5.0, -12.0, -3.0, -9.0])  # cali dei 4 link, tutti entro freeze_drop=6 tranne due
    inputs = _synthetic_inputs(series)
    for rule, expected in (("max", 3.0), ("median", 7.0)):  # mediana di 5, 12, 3, 9 = (5 + 9) / 2
        cfg["fusion"]["rule"] = rule
        trace = relative_trace(inputs, cfg, "fused", 20.0, 0, 1)
        # appena dopo il salto la base non è ancora cambiata: i link a -12 e -9 non sono ancora congelati nella base, ma il calo è il valore del salto
        assert trace["drop"][700] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 3. Linea di base
# ---------------------------------------------------------------------------


def _single_link(cfg, series, threshold=20.0, scope="fused"):
    from src.relative import relative_trace

    cfg["detector"]["reference"] = "relative"
    return relative_trace(_synthetic_inputs(series[:, None]), cfg, scope, threshold, 0, 1)


def test_baseline_converges_to_median_of_accepted_samples(cfg):
    """Serie costante più rumore: la base è la mediana dei campioni della finestra (calcolata qui a parte) e vale la mediana vera."""
    rng = np.random.default_rng(3)
    n_steps = 1500
    series = -60.0 + rng.uniform(-2.0, 2.0, n_steps)
    trace = _single_link(cfg, series)
    t = np.arange(n_steps) * 0.1
    window = cfg["detector"]["baseline_window"]
    events = np.arange(0, n_steps, 5)
    last = events[-1]
    in_window = events[t[events] >= t[last] - window - 1e-9]
    assert trace["base"][-1, 0] == pytest.approx(np.median(series[in_window]))
    assert abs(trace["base"][-1, 0] - (-60.0)) < 0.5
    first = int(np.flatnonzero(np.isfinite(trace["base"][:, 0]))[0])
    assert t[first] == pytest.approx((cfg["detector"]["baseline_min_samples"] - 1) * 0.5)  # 20 campioni a 2 Hz
    assert not trace["frozen"].any()  # il rumore di ±2 dB non supera freeze_drop


def test_baseline_freezes_on_large_step_and_resumes_from_frozen_value(cfg):
    """Gradino di -15 dB: la base si congela entro `min_duration`, resta ferma, e al ritorno riparte dal valore congelato."""
    n_steps = 2000  # 200 s
    series = np.full(n_steps, -60.0)
    series[600:1000] = -75.0  # t = 60..100 s
    series[1000:] = -63.0  # ritorno, 3 dB sotto la base congelata (entro freeze_drop)
    trace = _single_link(cfg, series, threshold=20.0)  # soglia alta: il pre-allarme non interviene, conta solo il congelamento
    t = np.arange(n_steps) * 0.1
    step_event = 600  # primo evento (multiplo di 5) dopo il gradino
    assert trace["frozen"][step_event, 0] and not trace["frozen"][step_event - 5, 0]
    assert t[step_event] - t[600] <= cfg["detector"]["min_duration"]
    assert np.all(trace["base"][step_event:1095, 0] == -60.0)  # ferma durante il gradino e per i primi 19 campioni del ritorno
    assert trace["base"][1095, 0] == -63.0  # 20 campioni accettati (t = 100.0 ... 109.5): riparte
    assert not trace["state"].any()
    # con soglia 10 dB il pre-allarme scatta dopo min_duration di calo > 10 dB e, finché è attivo, la base resta ferma
    # anche se il segnale è rientrato (si spegne dopo release_time, poi servono altri 20 campioni accettati)
    alarmed = _single_link(cfg, series, threshold=10.0)
    first_on = int(np.flatnonzero(alarmed["state"])[0])
    assert t[first_on] - t[600] == pytest.approx(cfg["detector"]["min_duration"], abs=0.11)
    off = 1000 + int(np.flatnonzero(~alarmed["state"][1000:])[0])
    assert t[off] - t[1000] == pytest.approx(cfg["detector"]["release_time"], abs=0.11)
    assert alarmed["base"][1095, 0] == -60.0 and alarmed["base"][off, 0] == -60.0
    assert alarmed["base"][off + 85, 0] == -60.0 and alarmed["base"][off + 105, 0] == -63.0  # ~20 campioni (9,5 s) dopo lo spegnimento


def test_baseline_follows_small_step_in_about_half_a_window(cfg):
    """Gradino di -3 dB (sotto freeze_drop): la mediana della finestra lo raggiunge in circa mezza finestra."""
    n_steps = 1800
    series = np.full(n_steps, -60.0)
    series[900:] = -63.0
    trace = _single_link(cfg, series)
    t = np.arange(n_steps) * 0.1
    crossed = int(np.flatnonzero(trace["base"][:, 0] <= -61.5)[0])
    delay = t[crossed] - t[900]
    half = cfg["detector"]["baseline_window"] / 2.0
    assert abs(delay - half) <= 1.0, delay
    assert trace["base"][-1, 0] == -63.0
    assert not trace["frozen"].any()


# ---------------------------------------------------------------------------
# 4. Invarianza agli scarti fissi
# ---------------------------------------------------------------------------


def test_relative_is_invariant_to_constant_offset_on_a_link(cfg):
    """+7 dB costanti su un link dall'inizio: calo fuso e allarmi in `relative` non cambiano (dopo l'avvio)."""
    rng = np.random.default_rng(11)
    n_steps = 1600
    series = np.tile(-60.0 + rng.uniform(-1.5, 1.5, (n_steps, 1)), (1, 3)) + rng.uniform(-1.0, 1.0, (n_steps, 3))
    series[800:] -= 15.0  # distacco vero: calo di 15 dB su tutti i link
    shifted = series.copy()
    shifted[:, 1] += 7.0
    from src.relative import relative_alarms, relative_trace

    cfg["detector"]["reference"] = "relative"
    a = relative_trace(_synthetic_inputs(series), cfg, "fused", 10.0, 0, 1)
    b = relative_trace(_synthetic_inputs(shifted), cfg, "fused", 10.0, 0, 1)
    start = int(np.flatnonzero(np.isfinite(a["drop"]))[0])
    np.testing.assert_allclose(b["drop"][start:], a["drop"][start:], atol=1e-9)
    np.testing.assert_array_equal(a["state"], b["state"])
    assert a["state"][-1] and not a["state"][700]
    # il livello assoluto invece cambia: il link a +7 dB sposta l'indice fuso
    sa = relative_alarms(_synthetic_inputs(series), cfg, "fused", np.array([10.0]))
    assert sa[0, :, 0, 1].any()


# ---------------------------------------------------------------------------
# 5. Spegnimento lento
# ---------------------------------------------------------------------------


def test_slow_release_survives_20s_return_but_not_35s(cfg):
    """Con `release_time` = 30 un rientro di 20 s non spegne l'allarme, uno di 35 s sì; conteggio degli spegnimenti a torto."""
    from src.detector import pre_alarm
    from src.metrics import summarize_separated_ext

    dt = 0.1
    cfg["detector"]["release_time"] = 30.0
    t = np.arange(0, 200.0, dt)

    def alarm_with_return(return_seconds):
        level = np.full(len(t), -60.0)
        level[(t >= 20.0) & (t < 60.0)] = -90.0  # distacco
        level[(t >= 60.0) & (t < 60.0 + return_seconds)] = -60.0  # il gruppo ripassa
        level[t >= 60.0 + return_seconds] = -90.0
        obs = pre_alarm(level, level, np.ones(len(t), bool), t, cfg["detector"], "level")
        return obs.reshape(len(t))

    short = alarm_with_return(20.0)
    assert short[t >= 21.0].all()  # mai spento
    long = alarm_with_return(35.0)
    off = np.flatnonzero(~long & (t > 21.0))
    assert t[off[0]] == pytest.approx(60.0 + 30.0, abs=dt)  # spento dopo release_time di rientro
    cfg["detector"]["release_time"] = 5.0  # il Blocco 4: anche il rientro breve spegne
    assert (~alarm_with_return(20.0) & (t > 21.0)).any()

    # conteggio a mano: allarme acceso 4-7, spento 8-10, acceso 11-13, spento 14-19; distanza 10 + 3 t
    system = np.zeros(20, dtype=bool)
    system[4:8] = True
    system[11:14] = True
    tt = np.arange(20.0)
    dist = 10.0 + 3.0 * tt
    out = summarize_separated_ext(system, tt, 2, dist, 30.0, 30.0)
    assert out["n_wrong_off"] == 2 and out["wrong_off_dist_sum"] == dist[8] + dist[14] and out["wrong_off_dist_min"] == dist[8]
    assert out["alarm_on_end"] is False and out["repass"] is False
    # ripassaggio: la distanza sale a 45 m e poi scende a 20 m mentre l'allarme è spento
    dist2 = np.array([10, 12, 14, 16, 18, 35, 45, 40, 25, 20, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15.0])
    out = summarize_separated_ext(system, tt, 2, dist2, 30.0, 30.0)
    assert out["repass"] and out["repass_off"]
    never = summarize_separated_ext(np.zeros(20, bool), tt, 2, dist, 30.0, 30.0)
    assert never["n_wrong_off"] == 0 and not never["repass"] and never["delay"] == math.inf


# ---------------------------------------------------------------------------
# 6. Robustezza del Kalman
# ---------------------------------------------------------------------------


def _filter(z, robust="none", gate=9.0, R=9.0, sigma_a=0.25):
    from src.kalman import filter_link

    times = np.arange(len(z)) * 0.5
    return filter_link(times, z, R, sigma_a, 3.0, 10.0, gate if robust == "nis_gate" else math.inf)


def test_nis_gate_reduces_isolated_jump_and_matches_below_gate():
    """Salto isolato di 12 dB: con il gate la stima si sposta meno; formula a mano; sotto il gate l'aggiornamento è identico alla base."""
    z = np.full(80, -60.0)
    z[40] = -48.0
    plain = _filter(z)[0]
    gated = _filter(z, "nis_gate")[0]
    assert abs(gated[40] + 60.0) < abs(plain[40] + 60.0)
    assert abs(gated[41] + 60.0) <= abs(plain[41] + 60.0)
    # a mano sul primo aggiornamento: P = diag(R, 3²), Δt = 0,5, innovazione 15 dB (NIS sopra il gate)
    R, sa, dt = 9.0, 0.25, 0.5
    p00 = R + dt**2 * 9.0 + sa**2 * dt**3 / 3.0
    nis = 15.0**2 / (p00 + R)
    assert nis > 9.0
    noise = R * nis / 9.0
    expected = -60.0 + p00 / (p00 + noise) * 15.0
    z2 = np.array([-60.0, -45.0])
    assert _filter(z2, "nis_gate")[0][1] == pytest.approx(expected)
    # sotto il gate: array identici (NIS = 1 / (p00 + R) < 9)
    small = np.array([-60.0, -59.0, -60.5, -60.0, -61.0])
    for a, b in zip(_filter(small), _filter(small, "nis_gate")):
        np.testing.assert_array_equal(a, b)


def test_nis_gate_reaches_persistent_step_in_finite_time():
    """Gradino persistente di -12 dB: la stima con il gate lo raggiunge (entro 1 dB) in un tempo finito, calcolato qui."""
    z = np.full(240, -60.0)
    z[60:] = -72.0
    times = np.arange(len(z)) * 0.5
    reach = {}
    for robust in ("none", "nis_gate"):
        r = _filter(z, robust)[0]
        idx = np.flatnonzero(np.abs(r[60:] + 72.0) <= 1.0)
        assert len(idx), robust
        reach[robust] = times[60 + idx[0]] - times[60]
    # tempi misurati (R = 9 dB², sigma_a = 0,25 dB/s², beacon a 2 Hz): vedi il messaggio in caso di fallimento
    assert reach["nis_gate"] < 30.0, reach
    assert reach["nis_gate"] >= reach["none"], reach


def test_inflate_option_scales_measurement_noise(cfg):
    """`inflate` equivale a filtrare con R·r_inflation; `none` ignora `r_inflation`."""
    from src.kalman import filter_all

    rng = np.random.default_rng(5)
    times = np.tile(np.arange(60) * 0.5, (2, 1))
    received = np.ones((2, 60, 2), dtype=bool)
    received[:, :, 0] = received[:, :, 1] = True
    rssi = -60.0 + rng.normal(0.0, 2.0, (2, 60, 2))
    kcfg = dict(cfg["kalman"])
    base = filter_all(times, received, rssi, 9.0, 0.25, kcfg)
    inflated = filter_all(times, received, rssi, 9.0, 0.25, {**kcfg, "robust": "inflate", "r_inflation": 4.0})
    direct = filter_all(times, received, rssi, 36.0, 0.25, kcfg)
    np.testing.assert_array_equal(inflated.r, direct.r)
    ignored = filter_all(times, received, rssi, 9.0, 0.25, {**kcfg, "r_inflation": 4.0})
    np.testing.assert_array_equal(ignored.r, base.r)
    with pytest.raises(ValueError):
        filter_all(times, received, rssi, 9.0, 0.25, {**kcfg, "robust": "altro"})


# ---------------------------------------------------------------------------
# 7. Monotonia
# ---------------------------------------------------------------------------


def _active_and_first(state_obs, t):
    system = state_obs.any(axis=-1) if state_obs.ndim > 1 else state_obs
    first = np.flatnonzero(system)
    return int(system.sum()), (t[first[0]] if len(first) else math.inf)


def test_widening_threshold_is_monotone_for_all_fusion_rules(cfg):
    """Con ogni regola, allentare la soglia di livello non diminuisce il tempo in allarme né ritarda il primo allarme (P_d, FA/h)."""
    from src.detector import pre_alarm
    from src.fusion import RULES, combine

    rng = np.random.default_rng(8)
    n_steps, n_links = 3000, 4
    t = np.arange(n_steps) * 0.1
    walk = -65.0 + np.cumsum(rng.normal(0.0, 0.25, n_steps))
    entries = walk[:, None] + rng.normal(0.0, 3.0, (n_steps, n_links))
    grid = np.arange(-95.0, -59.9, 2.5)
    for rule in RULES:
        fused = combine(entries, np.zeros_like(entries), np.ones_like(entries), np.ones(entries.shape, bool), rule)[0]
        prev = (-1, math.inf)
        for thr in grid:
            state = pre_alarm(fused[:, None], fused[:, None], np.ones((n_steps, 1), bool), t, cfg["detector"], "level", level_threshold=thr)
            active, first = _active_and_first(state, t)
            assert active >= prev[0] and first <= prev[1], (rule, thr)
            prev = (active, first)


def test_relative_threshold_sweep_monotonicity_is_measured(cfg):
    """Livello relativo: si misura la monotonia in soglia (il congelamento dipende dall'allarme, quindi non è garantita in teoria).

    Sul caso sintetico sotto, abbassare la soglia del calo non diminuisce il tempo in allarme: se fallisse, andrebbe riportato nei risultati.
    """
    from src.relative import relative_alarms

    cfg["detector"]["reference"] = "relative"
    rng = np.random.default_rng(2)
    n_steps = 4000
    walk = -60.0 - np.cumsum(np.abs(rng.normal(0.0, 0.02, n_steps)))  # calo lento
    series = walk[:, None] + rng.normal(0.0, 2.0, (n_steps, 3))
    states = relative_alarms(_synthetic_inputs(series), cfg, "fused", np.arange(6.0, 30.0, 2.0))
    active = states[:, :, 0, 1].sum(axis=1)
    assert np.all(np.diff(active) <= 0)  # soglia più alta -> meno allarme
    first = [np.flatnonzero(s[:, 0, 1])[:1] for s in states]
    firsts = [f[0] if len(f) else n_steps for f in first]
    assert np.all(np.diff(firsts) >= 0)


# ---------------------------------------------------------------------------
# 8. Riproducibilità, cache e voci
# ---------------------------------------------------------------------------


def _short_config(cfg, duration=100.0):
    cfg["simulation"]["duration"] = duration
    cfg["metrics"]["level_threshold_sweep"] = [-90.0, -65.0, 5.0]
    cfg["metrics"]["drop_threshold_sweep"] = [10.0, 30.0, 10.0]
    cfg["detector"]["warmup"] = 5.0
    return cfg


def test_process_run_reproducible_and_variants_independent(cfg, tmp_path):
    """Stesso seme, stesso risultato; i risultati di una variante non dipendono dalle altre valutate insieme."""
    from src.refinement import make_variant, process_run
    from src.simulator import scenario_config

    cfg = _short_config(cfg)
    run_info = {"scenario": "stop", "index": 0, "config": scenario_config(cfg, "stop", 7, 1, 40.0)}
    variants = [
        make_variant("base", "base", {"kalman.sigma_a": 0.25}, ("fused", "pairwise"), (1, 2), corr=False),
        make_variant("B:relative", "B", {"kalman.sigma_a": 0.25, "detector.reference": "relative"}),
        make_variant("D:gate", "D", {"kalman.sigma_a": 0.25, "kalman.robust": "nis_gate"}),
    ]
    first = process_run((run_info, cfg, tmp_path, variants, 9.0))
    second = process_run((run_info, cfg, tmp_path, variants, 9.0))
    for key, res in first["results"].items():
        for mo, stacked in res.items():
            for field, arr in stacked.items():
                np.testing.assert_array_equal(arr, second["results"][key][mo][field], err_msg=str((key, mo, field)))
    alone = process_run((run_info, cfg, tmp_path, variants[1:2], 9.0))
    for mo, stacked in alone["results"][("B:relative", "fused")].items():
        for field, arr in stacked.items():
            np.testing.assert_array_equal(arr, first["results"][("B:relative", "fused")][mo][field])


def test_independent_shadowing_does_not_overwrite_field_cache(cfg, tmp_path):
    """Le corse `independent` hanno una chiave di cache diversa e non sovrascrivono quelle `field`."""
    from src.simulator import cache_key, cache_path, load_or_simulate

    cfg = _short_config(cfg, duration=60.0)
    cfg["separation"]["enabled"] = False
    field = copy.deepcopy(cfg)
    independent = copy.deepcopy(cfg)
    independent["channel"]["shadowing"]["mode"] = "independent"
    assert cache_key(field) != cache_key(independent)
    run_field = load_or_simulate(field, tmp_path)
    stamp = cache_path(field, tmp_path).stat().st_mtime_ns
    run_ind = load_or_simulate(independent, tmp_path)
    assert len(list(tmp_path.glob("*.npz"))) == 2
    assert cache_path(field, tmp_path).stat().st_mtime_ns == stamp
    np.testing.assert_array_equal(load_or_simulate(field, tmp_path)["rssi"], run_field["rssi"])
    assert not np.array_equal(run_field["rssi"], run_ind["rssi"])  # lo shadowing è diverso


def test_new_value_flags_match_received_beacons(cfg, tmp_path):
    """Per il link proprio, i valori nuovi coincidono con i beacon ricevuti da `m` (conteggio indipendente dai Blocchi 1-3)."""
    from src.detector import fusion_entries
    from src.kalman import filter_all
    from src.simulator import load_or_simulate

    cfg = _short_config(cfg, duration=80.0)
    cfg["separation"]["enabled"] = False
    run = load_or_simulate(cfg, tmp_path)
    kres = filter_all(run["beacon_times"], run["received"], run["rssi"], 9.0, 0.25, cfg["kalman"])
    entries = fusion_entries(run["t"], kres, cfg, 0.25)
    t = run["t"]
    for m, i in ((0, 1), (2, 4), (3, 0)):
        rx = run["beacon_times"][i][run["received"][i, :, m]]
        expected = int(((rx >= t[0]) & (rx <= t[-1])).sum())
        assert int(entries["new"][:, m, i, m].sum()) == expected
    # le voci inoltrate dai compagni arrivano alla cadenza dei loro beacon
    k = 2
    rx_k = run["beacon_times"][k][run["received"][k, :, 0]]
    assert abs(int(entries["new"][:, 0, 1, k].sum()) - int(((rx_k >= t[0]) & (rx_k <= t[-1])).sum())) <= 2

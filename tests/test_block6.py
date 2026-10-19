"""Test del Blocco 6: config del Blocco 4 e pacchetti intatti, insiemi separati, soglie congelate, Gilbert-Elliott, `not_deaf`, verdetto, bootstrap, cache.

Ogni test verifica un fatto noto (riferimento generato con il codice precedente, formula chiusa, calcolo a mano o confronto indipendente).
"""

import copy
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import expon

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


def _load(name):
    spec = importlib.util.spec_from_file_location(name, DATA / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# 1. Blocchi 3, 4 e 5 intatti
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", ["none", "slowdown", "stop"])
def test_block4_config_still_reproduces_block4(scenario):
    """Con la config del Blocco 4 fissata esplicitamente (release_time 5 s compreso) allarmi e metriche coincidono con il riferimento del Blocco 4."""
    module = _load("make_block4_reference")
    config = module.reference_config()
    assert config["detector"]["release_time"] == 5.0 and config["kalman"]["sigma_a"] == 0.5 and config["detector"]["lost_rule"] == "any"
    fresh = module.reference_outputs(scenario, config)
    with np.load(DATA / "block4_reference.npz") as ref:
        names = [k.split("/", 1)[1] for k in ref.files if k.startswith(scenario + "/")]
        assert set(names) == set(fresh)
        for name in names:
            np.testing.assert_array_equal(fresh[name], ref[f"{scenario}/{name}"], err_msg=name)


@pytest.mark.parametrize("scenario", ["none", "slowdown", "stop"])
def test_independent_model_reproduces_block3(scenario):
    """Con ``background_model: independent`` pacchetti e conoscenza coincidono bit per bit con il riferimento generato prima dell'opzione (commit 0b8c38d)."""
    module = _load("make_block3_reference")
    config = module.reference_config()
    assert config["packets"]["reception"]["background_model"] == "independent"
    fresh = module.reference_outputs(scenario, config)
    with np.load(DATA / "block3_reference.npz") as ref:
        names = [k.split("/", 1)[1] for k in ref.files if k.startswith(scenario + "/")]
        assert set(names) == set(fresh)
        for name in names:
            np.testing.assert_array_equal(fresh[name], ref[f"{scenario}/{name}"], err_msg=name)


def test_lost_rule_any_alarms_identical_to_default_call(cfg):
    """`lost_alarm` con ``lost_rule: any`` (anche passando i dati di sordità) dà gli stessi allarmi della chiamata di prima."""
    from src.detector import lost_alarm

    t = np.arange(0, 100.0, 0.1)
    rng = np.random.default_rng(1)
    silence = rng.uniform(0, 20, (len(t), 3, 3))
    rx_ok = rng.random((len(t), 3, 3)) > 0.5
    deaf = rng.random((len(t), 3)) > 0.5
    d = cfg["detector"]
    d["lost_rule"] = "any"
    a = lost_alarm(silence, rx_ok, t, d)
    b = lost_alarm(silence, rx_ok, t, d, deaf_window=deaf, deaf_episode=np.ones(silence.shape, bool))
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# 2. Insiemi separati
# ---------------------------------------------------------------------------


def test_tuning_and_verification_sets_are_disjoint_and_paired(cfg):
    """Nessun seme in comune fra taratura e verifica; slowdown e stop di verifica hanno stessi semi, nodi e istanti (nell'intervallo di config)."""
    from src.final import run_sets

    tuning, verification = run_sets(cfg)
    assert len(tuning) == 3 * cfg["metrics"]["n_runs"] and len(verification) == 3 * cfg["final"]["n_runs"]
    seeds_t, seeds_v = {r["seed"] for r in tuning}, {r["seed"] for r in verification}
    assert not seeds_t & seeds_v
    assert len({r["seed"] for r in verification if r["scenario"] == "none"}) == cfg["final"]["n_runs"]  # semi distinti
    by = lambda s: [r for r in verification if r["scenario"] == s]  # noqa: E731
    for a, b in zip(by("slowdown"), by("stop")):
        assert (a["seed"], a["node"], a["start"]) == (b["seed"], b["node"], b["start"])
    lo, hi = cfg["metrics"]["separation_start_range"]
    assert all(lo <= r["start"] <= hi and 0 <= r["node"] < cfg["group"]["n_nodes"] for r in verification)
    # le chiavi sono quelle dichiarate: i semi di verifica riproducibili da SeedSequence(seed, spawn_key=(8,))
    expect = np.random.SeedSequence(cfg["simulation"]["seed"], spawn_key=(8,)).generate_state(cfg["final"]["n_runs"]).astype(np.int64)
    assert [r["seed"] for r in by("none")] == expect.tolist()
    # la taratura è quella dei Blocchi 4-5 (chiave 7), invariata
    expect_t = np.random.SeedSequence(cfg["simulation"]["seed"], spawn_key=(7,)).generate_state(cfg["metrics"]["n_runs"]).astype(np.int64)
    assert [r["seed"] for r in tuning if r["scenario"] == "none"] == expect_t.tolist()


# ---------------------------------------------------------------------------
# 3. Soglie congelate
# ---------------------------------------------------------------------------


def _synthetic_outs(n_runs, best_threshold_index, seed=0):
    """Esiti sintetici (corse none e con distacco) per una configurazione ``media``: P_d massima alla soglia `best_threshold_index`, nessun falso allarme."""
    grid = np.array([-80.0, -75.0, -70.0])
    rng = np.random.default_rng(seed)
    outs = []
    for scenario in ("none", "slowdown", "stop"):
        for i in range(n_runs):
            if scenario == "none":
                res = {
                    "threshold": grid, "episodes": np.zeros(3, int), "hours": np.full(3, 1.0), "active": np.zeros(3, int), "samples": np.full(3, 1000),
                    "ep_level": np.zeros(3, int), "ep_lost": np.zeros(3, int),
                }  # fmt: skip
            else:
                success = np.zeros(3, bool)
                success[best_threshold_index] = rng.random() < 0.9
                res = {
                    "threshold": grid, "success": success, "delay": np.where(success, 10.0, math.inf), "distance": np.where(success, 12.0, math.nan),
                    "fortuitous": np.zeros(3, bool), "n_wrong_off": np.zeros(3, int), "wrong_off_dist_sum": np.zeros(3), "wrong_off_dist_min": np.full(3, np.nan),
                    "repass": np.zeros(3, bool), "repass_off": np.zeros(3, bool), "alarm_on_end": np.zeros(3, bool),
                }  # fmt: skip
            outs.append({"scenario": scenario, "index": i, "results": {("media", "any"): res}})
    return outs


def test_evaluation_uses_frozen_threshold_not_the_best_on_verification(cfg):
    """`evaluate_at` usa la soglia che gli si dà (quella del file): sulla verifica il massimo è a -70, ma si valuta a -75; cambiare le corse non sposta la soglia."""
    from src.final import evaluate_at, load_frozen, write_frozen

    outs = _synthetic_outs(20, best_threshold_index=2)  # sulla verifica la soglia migliore sarebbe -70
    e75 = evaluate_at(outs, "media", "any", -75.0, cfg)
    e70 = evaluate_at(outs, "media", "any", -70.0, cfg)
    assert e75["summary"]["sep"]["stop"]["p_detect"] == 0.0 and e70["summary"]["sep"]["stop"]["p_detect"] > 0.5
    with pytest.raises(ValueError):
        evaluate_at(outs, "media", "any", -73.0, cfg)  # una soglia fuori griglia non si arrotonda in silenzio
    # il file: scritto una volta, riletto; con altre corse il valore letto non cambia
    import tempfile

    path = Path(tempfile.mkdtemp()) / "frozen.yaml"
    write_frozen(path, {"media": {"threshold": -75.0, "scope": "fused", "fusion_rule": "weighted_mean", "tuning": {}}}, cfg, 10.8)
    doc = load_frozen(path)
    assert doc["thresholds"]["media"]["threshold"] == -75.0 and doc["R"] == 10.8 and "commit" in doc and "date" in doc
    other = evaluate_at(_synthetic_outs(35, best_threshold_index=2, seed=9), "media", "any", doc["thresholds"]["media"]["threshold"], cfg)
    assert other["summary"]["sep"]["stop"]["p_detect"] == 0.0
    assert load_frozen(path)["thresholds"]["media"]["threshold"] == -75.0


def test_only_freeze_thresholds_chooses_a_threshold():
    """Nel codice di valutazione e degli script di verifica non c'è altra scelta di soglia: `working_point` si usa solo in `freeze_thresholds`, `retuned_point` (informativo)."""
    import re

    text = (ROOT / "src" / "final.py").read_text()
    users = [m.start() for m in re.finditer(r"working_point\(", text)]
    names = []
    for pos in users:
        names.append(re.findall(r"def (\w+)\(", text[:pos])[-1])
    assert sorted(names) == ["freeze_thresholds", "retuned_point"]
    script = (ROOT / "scripts" / "run_block6_experiment.py").read_text()
    assert "working_point" not in script


# ---------------------------------------------------------------------------
# 4. Gilbert-Elliott
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("loss, burst", [(0.02, 1.0), (0.05, 3.0)])
def test_gilbert_elliott_fraction_duration_and_exponential_bursts(loss, burst):
    """Su una catena lunga: frazione disturbata = L, durata media = T_b (entro 5%), raffiche oltre 10 s ≈ exp(-10/T_b) (esponenziali)."""
    from src.packets import burst_lengths, gilbert_elliott_states

    dt = 0.1
    n_steps = 1_500_000  # 150000 s
    state = gilbert_elliott_states(n_steps, (6,), dt, {"mean_loss": loss, "mean_burst": burst}, np.random.default_rng(4))
    assert state.shape == (n_steps, 6)
    assert abs(state.mean() / loss - 1.0) < 0.05
    lengths = burst_lengths(state, dt)
    assert len(lengths) > 500
    # la discretizzazione a passo dt dà una durata geometrica con media dt / (1 - exp(-dt/T_b)) ≈ T_b + dt/2: si confronta con quella
    expected_mean = dt / (1.0 - math.exp(-dt / burst))
    assert abs(lengths.mean() / expected_mean - 1.0) < 0.05
    assert abs(lengths.mean() / burst - 1.0) < 0.07  # e con T_b, nei limiti della discretizzazione
    tail = (lengths > 10.0).mean()
    assert abs(tail - math.exp(-10.0 / burst)) < 4 * math.sqrt(math.exp(-10.0 / burst) * (1 - math.exp(-10.0 / burst)) / len(lengths)) + 0.01
    # esponenziale: Kolmogorov-Smirnov contro la durata continua (tolleranza della discretizzazione)
    from scipy.stats import kstest

    assert kstest(lengths - dt / 2.0, expon(scale=burst).cdf).statistic < 0.06


def test_gilbert_elliott_start_is_stationary_and_chain_independent():
    """Lo stato iniziale è estratto dalla stazionaria (frazione L su molte catene); catene diverse non sono identiche."""
    from src.packets import gilbert_elliott_states

    state = gilbert_elliott_states(2, (20000,), 0.1, {"mean_loss": 0.3, "mean_burst": 2.0}, np.random.default_rng(0))
    assert abs(state[0].mean() - 0.3) < 0.02
    assert not np.array_equal(state[:, 0], state[:, 1])


def _simulate_with_strong_signal(cfg, scope, seed=3, duration=400.0):
    """`simulate_packets` con un canale finto a segnale fortissimo (la prova del segnale passa sempre): si vedono solo le perdite di fondo."""
    from types import SimpleNamespace

    from src.packets import simulate_packets

    cfg = copy.deepcopy(cfg)
    cfg["simulation"]["seed"] = seed
    pk = cfg["packets"]["reception"]
    pk["background_model"] = "bursty"
    pk["bursty"].update(mean_loss=0.1, mean_burst=5.0, loss_bad=1.0, burst_scope=scope)
    dt = cfg["simulation"]["dt"]
    t = np.arange(0.0, duration, dt)
    n = 4
    rssi = np.full((len(t), n, n), -40.0)
    ch = SimpleNamespace(t=t, rssi_true=rssi, rssi_measured=rssi.copy())
    return simulate_packets(None, ch, cfg), cfg


def test_bursty_receiver_scope_loses_all_beacons_to_that_receiver_together(cfg):
    """Con ``burst_scope: receiver`` e segnale forte, un beacon è perso esattamente quando il RICEVITORE è nello stato disturbato (qualunque il trasmettitore)."""
    pk, cfg2 = _simulate_with_strong_signal(cfg, "receiver")
    state = pk.background_state  # (T, N)
    assert state.ndim == 2
    n = state.shape[1]
    checked = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ok = pk.beacon_steps[i] >= 0
            bad = state[pk.beacon_steps[i][ok], j]
            np.testing.assert_array_equal(pk.received[i][ok, j], ~bad)  # perso <=> disturbato, senza eccezioni (loss_bad = 1)
            checked += int(bad.sum())
    assert checked > 100  # il test non è vuoto
    # nello stesso istante i beacon di trasmettitori diversi verso j si perdono insieme: la frazione di tempo disturbato di j è la stessa per tutti
    assert 0.05 < state.mean() < 0.2


def test_bursty_link_scope_is_independent_between_links(cfg):
    """Con ``burst_scope: link`` lo stato è per coppia (T, N, N) e due link verso lo stesso ricevitore non sono perfettamente correlati."""
    pk, _ = _simulate_with_strong_signal(cfg, "link")
    state = pk.background_state
    assert state.ndim == 3 and not np.array_equal(state[:, 0, 2], state[:, 1, 2])


def test_bursty_does_not_touch_streams_4_and_5(cfg):
    """Con ``bursty`` gli istanti dei beacon (chiave 4) e le uniformi del segnale (chiave 5) sono gli stessi che con ``independent``."""
    from types import SimpleNamespace

    from src.packets import simulate_packets

    cfg = copy.deepcopy(cfg)
    t = np.arange(0.0, 200.0, cfg["simulation"]["dt"])
    rssi = np.full((len(t), 4, 4), -95.0)  # nella zona di transizione: il segnale decide in parte
    ch = SimpleNamespace(t=t, rssi_true=rssi, rssi_measured=rssi.copy())
    a = simulate_packets(None, ch, cfg)
    cfg["packets"]["reception"]["background_model"] = "bursty"
    b = simulate_packets(None, ch, cfg)
    np.testing.assert_array_equal(a.beacon_times, b.beacon_times)
    np.testing.assert_array_equal(a.passed_signal, b.passed_signal)


# ---------------------------------------------------------------------------
# 5. not_deaf
# ---------------------------------------------------------------------------


def _beacons(n_nodes, t_end, period=0.5):
    """Beacon periodici sfasati (nodo k a k * period / n_nodes)."""
    times = np.array([np.arange(k * period / n_nodes, t_end, period) for k in range(n_nodes)])
    return times


def _lost_system_for_observer(cfg, received, times, t, rule, m=0):
    from src.detector import deafness, lost_alarm, reception_features

    silence, rx_ok = reception_features(t, times, received, cfg)
    deaf_window, deaf_episode = deafness(t, times, received, silence, cfg)
    d = {**cfg["detector"], "lost_rule": rule}
    valid = ~np.eye(times.shape[0], dtype=bool)
    return lost_alarm(silence, rx_ok, t, d, valid, deaf_window=deaf_window, deaf_episode=deaf_episode)[:, m, :]


@pytest.mark.parametrize("rule", ["not_deaf", "not_deaf_episode"])
def test_not_deaf_deaf_observer_does_not_alarm_but_blind_to_one_target_does(cfg, rule):
    """Osservatore 0 che non sente nessuno per 15 s: niente "perso" con `not_deaf` (sì con `any`); osservatore che sente gli altri ma non `i`: "perso" (come `any`)."""
    n, t_end = 5, 120.0
    t = np.arange(0.0, t_end, 0.1)
    times = _beacons(n, t_end)
    # (a) 0 sordo per 15 s (da 40 a 55 s): non riceve nessun beacon in quell'intervallo
    received = np.ones((n, times.shape[1], n), bool)
    received[np.arange(n), :, np.arange(n)] = False
    deaf = (times >= 40.0) & (times < 55.0)
    received[:, :, 0] &= ~deaf
    any_ = _lost_system_for_observer(cfg, received, times, t, "any")
    nd = _lost_system_for_observer(cfg, received, times, t, rule)
    assert any_[:, 1:].any()  # senza la regola l'osservatore sordo dichiara persi i vicini
    window = (t > 20.0) & (t < 54.0)  # il periodo sordo è 40-55 s
    if rule == "not_deaf_episode":
        assert not nd[window].any()  # decisione per episodio: nessun "perso" da parte dell'osservatore sordo
    else:
        # regola letterale (finestra mobile di `lost_silence`): i beacon dei vicini sono sfasati, quindi ogni vicino supera 10 s di silenzio mentre `m`
        # ha ancora sentito qualcuno nella finestra (i vicini con ultimo beacon più recente) e il "perso" a scatto resta acceso. Solo l'ultimo vicino
        # sentito (finestra ormai vuota) è soppresso: su 4 vicini ne restano in allarme 3. Con `any` sono 4.
        assert nd[window][:, 1:].any(axis=0).sum() == 3 and any_[window][:, 1:].any(axis=0).sum() == 4
        # al rientro (55-60 s) la regola letterale perde di nuovo: `m` sente un vicino prima degli altri e i silenzi residui (> 10 s) scattano
        assert nd[(t > 55.0) & (t < 60.0)][:, 1:].any(axis=0).sum() >= 3
    # (b) 0 sente tutti ma non 3, da 40 s in poi: il "perso" su 3 scatta come con `any`
    received = np.ones((n, times.shape[1], n), bool)
    received[np.arange(n), :, np.arange(n)] = False
    received[3, times[3] >= 40.0, 0] = False
    any_ = _lost_system_for_observer(cfg, received, times, t, "any")
    nd = _lost_system_for_observer(cfg, received, times, t, rule)
    np.testing.assert_array_equal(nd[:, 3], any_[:, 3])
    assert nd[t > 51.0, 3].all() and not nd[:, 1].any() and not nd[:, 2].any()


def test_not_deaf_equals_any_on_a_run_without_bursts(cfg, tmp_path):
    """Corsa ``none`` breve senza raffiche e senza silenzi lunghi: gli allarmi "perso" e di sistema di `any` e `not_deaf` coincidono."""
    from src.detector import build_inputs, lost_alarm
    from src.kalman import filter_all
    from src.simulator import load_or_simulate

    cfg["simulation"]["duration"] = 150.0
    cfg["separation"]["enabled"] = False
    run = load_or_simulate(cfg, tmp_path)
    inputs, _ = build_inputs(run, cfg, cfg["kalman"]["sigma_a"], 9.0)
    for rule in ("not_deaf", "not_deaf_episode"):
        a = lost_alarm(inputs.silence, inputs.rx_ok, inputs.t, {**cfg["detector"], "lost_rule": "any"}, inputs.valid, **inputs.lost_kwargs())
        b = lost_alarm(inputs.silence, inputs.rx_ok, inputs.t, {**cfg["detector"], "lost_rule": rule}, inputs.valid, **inputs.lost_kwargs())
        np.testing.assert_array_equal(a, b)
    assert not inputs.deaf_window[inputs.t > cfg["detector"]["warmup"]].any()  # nessun ricevitore sordo in questa corsa


# ---------------------------------------------------------------------------
# 6. Verdetto
# ---------------------------------------------------------------------------


def _success(gain, loss, n=20):
    """Coppie di esiti: `gain` corse solo di `migliore`, `loss` solo di `media`, il resto uguale."""
    media = np.zeros(n, bool)
    migliore = np.zeros(n, bool)
    migliore[:gain] = True
    media[gain : gain + loss] = True
    media[n - 5 :] = migliore[n - 5 :] = True  # corse concordanti
    return media, migliore


def test_verdict_rule_on_synthetic_outcomes():
    """+4/-1 con falsi allarmi uguali: migliore; +4/-1 con falsi allarmi più alti: media; +2/-0: media (netto sotto 3); McNemar a mano."""
    from src.final import decide_verdict, mcnemar_exact

    media, migliore = _success(4, 1)
    v = decide_verdict(media, migliore, 0.8, 0.8, 3)
    assert (v["gained"], v["lost"], v["net"], v["adopted"]) == (4, 1, 3, "migliore")
    assert decide_verdict(media, migliore, 0.8, 0.81, 3)["adopted"] == "media"
    assert decide_verdict(media, migliore, 0.8, 0.5, 3)["adopted"] == "migliore"
    media, migliore = _success(2, 0)
    assert decide_verdict(media, migliore, 0.8, 0.8, 3)["adopted"] == "media"
    # McNemar esatto: 4 contro 1 -> n = 5, P(X <= 1) = 6/32, bilaterale 12/32 = 0,375; nessuna discordanza -> 1; 5 contro 0 -> 2/32
    assert mcnemar_exact(4, 1) == pytest.approx(0.375)
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(5, 0) == pytest.approx(2 / 32)
    assert mcnemar_exact(3, 3) == 1.0


# ---------------------------------------------------------------------------
# 7. Bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_similar_to_poisson_for_independent_runs_and_wider_for_clustered_events():
    """Corse indipendenti con eventi di Poisson: bootstrap e Garwood simili; stessi eventi totali raggruppati in poche corse: il bootstrap è più largo."""
    from src.final import bootstrap_rate_interval
    from src.metrics import poisson_interval

    rng = np.random.default_rng(5)
    n_runs, hours = 100, np.full(100, 0.5)
    counts = rng.poisson(0.8 * 0.5, n_runs)
    lo, hi = bootstrap_rate_interval(counts, hours, 2000, 0.95, np.random.default_rng(1))
    plo, phi = poisson_interval(int(counts.sum()), float(hours.sum()), 0.95)
    assert abs((hi - lo) / (phi - plo) - 1.0) < 0.25
    # stesso numero totale di eventi, ma tutti in 5 corse su 100 (a raffiche): Poisson non lo vede, il bootstrap sì
    clustered = np.zeros(n_runs, int)
    clustered[:5] = counts.sum() // 5
    lo_c, hi_c = bootstrap_rate_interval(clustered, hours, 2000, 0.95, np.random.default_rng(1))
    assert (hi_c - lo_c) > 1.5 * (hi - lo)
    plo_c, phi_c = poisson_interval(int(clustered.sum()), float(hours.sum()), 0.95)
    assert (hi_c - lo_c) > 1.5 * (phi_c - plo_c)
    # stesso generatore, stesso risultato
    assert bootstrap_rate_interval(counts, hours, 2000, 0.95, np.random.default_rng(1)) == (lo, hi)


# ---------------------------------------------------------------------------
# 8. Riproducibilità e cache
# ---------------------------------------------------------------------------


def _short(cfg, duration=120.0):
    cfg["simulation"]["duration"] = duration
    cfg["metrics"]["level_threshold_sweep"] = [-90.0, -65.0, 5.0]
    cfg["detector"]["warmup"] = 5.0
    return cfg


def test_process_run_final_reproducible_and_sensitivity_point_does_not_touch_base_cache(cfg, tmp_path):
    """Stesso seme, stesso risultato; un punto di sensibilità non scrive nella cache delle corse (né modifica la base); base dal canale = base dalla cache."""
    import os

    from src.final import process_run_final
    from src.simulator import cache_key, cache_path, load_or_simulate, scenario_config

    cfg = _short(cfg)
    info = {"scenario": "stop", "index": 0, "config": scenario_config(cfg, "stop", 7, 1, 40.0)}
    spec_cache = {"mode": "cache", "point": {}, "configs": ["media", "migliore", "coppia"], "rules": ["any"]}
    first = process_run_final((info, cfg, tmp_path, 9.0, spec_cache))
    second = process_run_final((info, cfg, tmp_path, 9.0, spec_cache))
    for key, res in first["results"].items():
        for field, arr in res.items():
            np.testing.assert_array_equal(arr, second["results"][key][field], err_msg=str((key, field)))
    base_file = cache_path(info["config"], tmp_path)
    stamp = base_file.stat().st_mtime_ns
    files_before = {p.name for p in tmp_path.glob("*.npz")}
    point = cfg["final"]["sensitivity_points"]["bursty_5_3"]
    spec_point = {"mode": "channel", "point": point, "configs": ["media"], "rules": ["any", "not_deaf"]}
    p1 = process_run_final((info, cfg, tmp_path, 9.0, spec_point))
    p2 = process_run_final((info, cfg, tmp_path, 9.0, spec_point))
    for key, res in p1["results"].items():
        for field, arr in res.items():
            np.testing.assert_array_equal(arr, p2["results"][key][field])
    assert {p.name for p in tmp_path.glob("*.npz")} == files_before  # nessun nuovo file nella cache delle corse
    assert base_file.stat().st_mtime_ns == stamp and os.path.exists(tmp_path / "channel")
    # la chiave di cache con bursty è diversa da quella con independent, e con independent è quella di prima (opzioni del Blocco 6 non hashate)
    bursty = copy.deepcopy(info["config"])
    bursty["packets"]["reception"]["background_model"] = "bursty"
    assert cache_key(bursty) != cache_key(info["config"])
    explicit = copy.deepcopy(info["config"])
    explicit["packets"]["reception"]["bursty"]["mean_loss"] = 0.5  # con independent non conta
    assert cache_key(explicit) == cache_key(info["config"])
    # base dal canale = base dalla cache, bit per bit
    chan = process_run_final((info, cfg, tmp_path, 9.0, {**spec_cache, "mode": "channel"}))
    for key, res in first["results"].items():
        for field, arr in res.items():
            np.testing.assert_array_equal(arr, chan["results"][key][field], err_msg=str((key, field)))
    run = load_or_simulate(info["config"], tmp_path)
    assert run["received"].shape[0] == cfg["group"]["n_nodes"]

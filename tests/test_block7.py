"""Test del Blocco 7: routing verso il capogruppo (salti, AHP/entropia, punteggio, tempo alla rottura, unicast, pacchetti, carry, duplicati, `direct`, riproducibilità).

Ogni test verifica un fatto noto (calcolo a mano, formula chiusa, confronto con un calcolo indipendente). Le topologie sono sintetiche: la rete è descritta da una funzione
``up(i, j, t)`` (il collegamento ``i → j`` esiste a quell'istante) e da qui si costruiscono beacon, stime dei link e probabilità di ricezione con le stesse funzioni dell'esperimento.
"""

import copy
import math
from pathlib import Path

import numpy as np
import pytest

from src.mobility import load_config
from src.packets import beacon_schedule
from src.routing import (
    ISOLATED,
    KIND_NORMAL,
    KIND_SOS,
    STRATEGIES,
    UniformStream,
    ahp_weights,
    airtime,
    announce_hops,
    announced_grid,
    combine_weights,
    dqrem_scores,
    entropy_weights,
    link_state,
    position_class,
    record_bytes,
    records_per_packet,
    select_packet,
    simulate_strategy,
    time_to_break,
    unicast_attempts,
)

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "default.yaml"


@pytest.fixture(scope="session")
def base_config():
    return load_config(CONFIG_PATH)


@pytest.fixture
def cfg(base_config):
    return copy.deepcopy(base_config)


def synthetic_rd(cfg, n, duration, leader, up, events=(), seed=3, p_override=None):
    """Corsa sintetica: beacon veri (`beacon_schedule`), ricevuti se ``up(i, j, t)``; stima -60 dBm sui link che esistono, -120 sugli altri; pendenza 0; ``p`` = 1 dove il link esiste."""
    rcfg = cfg["routing"]
    dt = cfg["simulation"]["dt"]
    t = np.arange(0.0, duration, dt)
    times, steps = beacon_schedule(n, float(t[-1]), dt, cfg["packets"], np.random.default_rng(seed))
    k = times.shape[1]
    ok = steps >= 0
    received = np.zeros((n, k, n), dtype=bool)
    p = np.zeros((len(t), n, n), dtype=np.float32)
    est_r = np.full((len(t), n, n), -120.0)
    level = np.full((n, k, n), -120.0)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            received[i, :, j] = up(i, j, np.where(ok[i], times[i], 0.0)) & ok[i]
            exists = up(i, j, t)
            p[:, i, j] = exists
            if exists.any():
                est_r[:, i, j] = -60.0
                level[i, :, j] = -60.0
    ls = link_state(t, times, received, est_r, np.zeros_like(est_r), level, np.zeros_like(level), rcfg, cfg["packets"]["beacon_period"])
    ann = announce_hops(times, steps, received, est_r, leader, rcfg)
    rd = {
        "dt": dt, "t": t, "times": times, "steps": steps, "received": received, "ann": ann, "ann_grid": announced_grid(t, times, ann), "p": p,
        "leader": leader, "events": list(events), "seed": 11, "sep_node": -1, **ls,
    }  # fmt: skip
    if p_override is not None:
        p_override(rd)
    return rd


def chain(i, j, tt):
    return np.full(np.shape(tt), abs(i - j) == 1)


# ---------------------------------------------------------------------------
# 1. Nessuna regressione
# ---------------------------------------------------------------------------


def test_default_config_is_the_block6_detector_and_block7_does_not_change_cache_keys(cfg):
    """La config di default resta quella del rilevatore finale a 5 nodi; la sezione ``routing`` non entra nelle chiavi di cache dei Blocchi 1-3."""
    from src.simulator import cache_key, channel_cache_key

    assert cfg["group"]["n_nodes"] == 5 and cfg["group"]["longitudinal_spread"] == 10.0
    assert cfg["detector"]["level_threshold"] == -75.0 and cfg["detector"]["mode"] == "level"
    without = copy.deepcopy(cfg)
    del without["routing"]
    assert cache_key(cfg) == cache_key(without) and channel_cache_key(cfg) == channel_cache_key(without)
    from src.routing_experiment import block7_config

    long = block7_config(cfg)
    assert long["group"]["n_nodes"] == 10 and cache_key(long) != cache_key(cfg)  # il gruppo lungo ha una cache propria (mai quella da 5 nodi)


def test_block6_results_identical_with_or_without_routing_section(cfg, tmp_path):
    """`process_run_final` dà lo stesso risultato con e senza la sezione ``routing``: il Blocco 7 non tocca i Blocchi 1-6."""
    from src.final import process_run_final
    from src.simulator import scenario_config

    cfg["simulation"]["duration"] = 120.0
    cfg["metrics"]["level_threshold_sweep"] = [-90.0, -65.0, 5.0]
    cfg["detector"]["warmup"] = 5.0
    without = copy.deepcopy(cfg)
    del without["routing"]
    spec = {"mode": "cache", "point": {}, "configs": ["media"], "rules": ["any"]}
    info = {"scenario": "stop", "index": 0, "config": scenario_config(cfg, "stop", 7, 1, 40.0)}
    info_no = {"scenario": "stop", "index": 0, "config": scenario_config(without, "stop", 7, 1, 40.0)}
    a = process_run_final((info, cfg, tmp_path / "a", 9.0, spec))
    b = process_run_final((info_no, without, tmp_path / "b", 9.0, spec))
    for key, res in a["results"].items():
        for field, arr in res.items():
            np.testing.assert_array_equal(arr, b["results"][key][field], err_msg=str((key, field)))


def test_real_run_reproducible_and_writes_nothing_to_run_cache(cfg):
    """Una corsa reale corta (gruppo lungo): stesso seme, stesso risultato; la cache delle corse (``results/cache``) non cambia."""
    from src.routing_experiment import block7_config, process_run_routing
    from src.simulator import scenario_config

    cfg["group"]["calibration"].update({"groups": 10, "duration": 60.0})
    c7 = block7_config(cfg)
    c7["simulation"]["duration"] = 150.0
    run = {"scenario": "stop", "index": 0, "leader": 2, "sos": 0.3, "config": scenario_config(c7, "stop", 5, 6, 40.0)}
    cache = ROOT / "results" / "cache"
    before = sorted(p.name for p in cache.glob("*")) if cache.exists() else []
    a = process_run_routing((run, ["direct", "dqrem"], [], False))
    b = process_run_routing((run, ["direct", "dqrem"], [], False))
    after = sorted(p.name for p in cache.glob("*")) if cache.exists() else []
    assert before == after
    for name in ("direct", "dqrem"):
        sa, sb = a["strategies"][name], b["strategies"][name]
        for key in ("breaks", "bridge_changes", "ttl_drops", "cycles"):
            assert sa[key] == sb[key]
        np.testing.assert_array_equal(sa["classes"]["group"]["delays"], sb["classes"]["group"]["delays"])
        assert sa["classes"]["group"]["created"] > 0


# ---------------------------------------------------------------------------
# 2. Salti
# ---------------------------------------------------------------------------


def test_hops_on_a_chain_and_isolation_when_a_link_is_cut(cfg):
    """Catena 0-1-2-3: salti 0, 1, 2, 3. Tolto il collegamento 1-2, i nodi 2 e 3 diventano isolati entro ``max_hops`` periodi di beacon (conteggio all'infinito limitato)."""
    rcfg = cfg["routing"]
    rd = synthetic_rd(cfg, 4, 80.0, 0, chain)
    times, ann = rd["times"], rd["ann"]
    last = [ann[i, np.flatnonzero(~np.isnan(times[i]))[-1]] for i in range(4)]
    assert last == [0, 1, 2, 3]

    cut = 30.0
    cut_chain = lambda i, j, tt: chain(i, j, tt) & ~(({i, j} == {1, 2}) & (tt >= cut))  # noqa: E731
    rd = synthetic_rd(cfg, 4, 80.0, 0, cut_chain)
    times, ann = rd["times"], rd["ann"]
    assert [ann[i, np.flatnonzero(~np.isnan(times[i]))[-1]] for i in range(4)] == [0, 1, ISOLATED, ISOLATED]
    period, jitter = cfg["packets"]["beacon_period"], cfg["packets"]["jitter"]
    bound = cut + rcfg["neighbour_max_age"] + (rcfg["max_hops"] + 2) * (period + jitter)
    for node in (2, 3):
        ks = np.flatnonzero(~np.isnan(times[node]))
        ok = ks[(ann[node, ks] == ISOLATED) & (times[node, ks] > cut)]
        assert times[node, ok[0]] <= bound, f"nodo {node}: isolato solo a {times[node, ok[0]]:.1f} s (limite {bound:.1f} s)"
    # nel frattempo il valore sbagliato non supera mai max_hops
    finite = ann[2:, :][(ann[2:, :] != ISOLATED)]
    assert finite.max() <= rcfg["max_hops"]


def test_no_ttl_drops_on_a_still_topology(cfg):
    """Su una topologia ferma nessun pacchetto viene scartato per TTL e tutti i record arrivano."""
    rd = synthetic_rd(cfg, 4, 60.0, 0, chain)
    out = simulate_strategy(rd, "best_rssi", cfg["routing"])
    assert out["ttl_drops"] == 0
    assert set(np.unique(out["log"]["origin"])) == {1, 2, 3}
    hops = {o: out["log"]["hops"][out["log"]["origin"] == o] for o in (1, 2, 3)}
    assert all(np.all(h == o) for o, h in hops.items())  # il nodo a `o` salti dal capogruppo ci mette esattamente `o` salti


# ---------------------------------------------------------------------------
# 3. AHP, entropia
# ---------------------------------------------------------------------------


def test_ahp_weights_equal_independent_eigenvector_and_consistent(cfg):
    """Pesi AHP = autovettore principale calcolato a parte (iterazione di potenza), CR < 0,1, valori attesi del brief."""
    rcfg = cfg["routing"]
    a = np.array(rcfg["ahp_matrix"], dtype=float)
    w, cr = ahp_weights(a, rcfg["ahp_random_index"])
    v = np.ones(5)
    for _ in range(500):
        v = a @ v
        v /= v.sum()
    np.testing.assert_allclose(w, v, atol=1e-9)
    np.testing.assert_allclose(w, [0.364, 0.364, 0.131, 0.070, 0.070], atol=2e-3)
    assert w.sum() == pytest.approx(1.0) and cr < 0.1 and cr == pytest.approx(0.001, abs=1e-3)


def test_entropy_weight_zero_for_constant_criterion_and_known_case():
    """Un criterio costante ha peso 0; con colonne note il peso va a chi varia di più (calcolo a mano: 3 righe, colonna A concentrata, colonna B uniforme)."""
    benefit = np.array([True, True, True])
    x = np.array([[1.0, 5.0, 3.0], [2.0, 5.0, 9.0], [3.0, 5.0, 4.0], [4.0, 5.0, 8.0]])
    w = entropy_weights(x, benefit)
    assert w[1] == 0.0 and w.sum() == pytest.approx(1.0) and np.all(w >= 0)
    # a mano per una colonna: v = (x - min)/(max - min) = [0, 1/3, 2/3, 1] -> p = [0, 1/6, 2/6, 3/6], E = -Σ p ln p / ln 4
    p = np.array([1 / 6, 2 / 6, 3 / 6])
    e = -(p * np.log(p)).sum() / math.log(4)
    only = entropy_weights(x[:, [0, 1]], benefit[:2])
    assert only[0] == pytest.approx(1.0) and 1 - e > 0  # una sola colonna che varia: tutto il peso
    two = entropy_weights(np.column_stack([x[:, 0], [0.0, 0.0, 0.0, 9.0]]), benefit[:2])
    assert two[1] > two[0]  # la colonna concentrata su un solo valore ha entropia più bassa, quindi peso maggiore
    assert np.all(entropy_weights(np.ones((5, 3)), benefit) == 0)
    np.testing.assert_allclose(combine_weights(np.array([0.6, 0.4]), np.array([0.0, 0.0]), 0.5), [0.6, 0.4])


# ---------------------------------------------------------------------------
# 4. Punteggio
# ---------------------------------------------------------------------------


def test_score_dominant_candidate_wins_and_directions(cfg):
    """Un candidato migliore in tutti i criteri vince sempre; con uno solo viene scelto; benefici alti e costi bassi alzano il punteggio."""
    benefit = np.array([True, True, False, True, False])
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(2, 6))
        x = np.column_stack([rng.uniform(-90, -50, n), rng.uniform(0, 60, n), rng.integers(1, 6, n), rng.uniform(0, 1, n), rng.integers(0, 16, n)]).astype(float)
        best = int(rng.integers(0, n))
        x[best] = [x[:, 0].max() + 1, x[:, 1].max() + 1, x[:, 2].min() - 1, x[:, 3].max() + 0.01, x[:, 4].min() - 1]
        w = rng.dirichlet(np.ones(5)) + 1e-3
        assert int(np.argmax(dqrem_scores(x, w / w.sum(), benefit))) == best
    assert dqrem_scores(np.array([[-70.0, 10.0, 2.0, 0.9, 3.0]]), np.full(5, 0.2), benefit)[0] == pytest.approx(1.0)
    w = np.array([1.0, 0, 0, 0, 0])
    x = np.array([[-80.0, 0, 1, 1, 0], [-60.0, 0, 1, 1, 0]])
    assert dqrem_scores(x, w, benefit)[1] > dqrem_scores(x, w, benefit)[0]  # beneficio: segnale più alto è meglio
    w = np.array([0, 0, 0, 0, 1.0])
    x = np.array([[-70.0, 0, 1, 1, 2.0], [-70.0, 0, 1, 1, 9.0]])
    assert dqrem_scores(x, w, benefit)[0] > dqrem_scores(x, w, benefit)[1]  # costo: coda più corta è meglio


# ---------------------------------------------------------------------------
# 5. Tempo alla rottura
# ---------------------------------------------------------------------------


def test_time_to_break(cfg):
    """Pendenza ≥ 0 → cap; −80 dBm con −0,5 dB/s e soglia −90 → (−80 − (−90)) / 0,5 = 20 s; limitato al cap e mai negativo."""
    rcfg = cfg["routing"]
    thr, cap = rcfg["candidate_threshold"], rcfg["ttb_cap"]
    assert time_to_break(-70.0, 0.0, thr, cap) == cap and time_to_break(-70.0, 0.4, thr, cap) == cap
    assert time_to_break(-80.0, -0.5, thr, cap) == pytest.approx(20.0)
    assert time_to_break(-60.0, -0.1, thr, cap) == cap  # 300 s limitato a 60
    assert time_to_break(-95.0, -0.5, thr, cap) == 0.0  # già sotto la soglia


# ---------------------------------------------------------------------------
# 6. Unicast
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p_fwd,p_ret,retries", [(0.7, 0.8, 3), (0.5, 0.5, 1), (0.9, 0.6, 0)])
def test_unicast_delivery_matches_closed_form(p_fwd, p_ret, retries):
    """Con probabilità ``p = andata × ritorno`` per tentativo, la consegna Monte Carlo vale ``1 − (1 − p)^(r + 1)`` entro 4 errori standard."""
    stream = UniformStream(np.random.default_rng(5))
    n = 40000
    ok = sum(unicast_attempts(p_fwd, p_ret, retries, stream)[0] for _ in range(n))
    p = p_fwd * p_ret
    expect = 1 - (1 - p) ** (retries + 1)
    assert abs(ok / n - expect) < 4 * math.sqrt(expect * (1 - expect) / n)


# ---------------------------------------------------------------------------
# 7. Aggregazione
# ---------------------------------------------------------------------------


def test_packet_sizes_and_priority(cfg):
    """Mai oltre ``max_packet_bytes`` (4 record con 250 B e 10 nodi, 10 con 1470 B); un SOS parte prima dei record normali; tempo d'aria di 250 B ≈ 2,5 ms con la conferma."""
    rcfg = cfg["routing"]
    assert record_bytes(10) == 53
    assert records_per_packet(rcfg, 10) == 4 and records_per_packet(rcfg, 10, 1470) == 10
    assert rcfg["packet_header_bytes"] + records_per_packet(rcfg, 10) * 53 <= rcfg["max_packet_bytes"]
    assert airtime(250, rcfg) + airtime(rcfg["airtime"]["ack_bytes"], rcfg) == pytest.approx(2.5e-3, abs=0.05e-3)
    queue = [[float(k), 1, k, KIND_NORMAL, 8, 0, False, 0.0] for k in range(6)]
    queue.append([7.0, 1, 2_000_000, KIND_SOS, 8, 0, False, 7.0])
    pkt = select_packet(queue, 4, 1)
    assert pkt[0][3] == KIND_SOS and [r[2] for r in pkt[1:]] == [0, 1, 2] and len(pkt) == 4
    assert [r[2] for r in select_packet(queue[:6], 4, 0)] == [0, 1, 2, 3]  # senza prioritari: i più vecchi
    # nella simulazione: nessun pacchetto oltre il limite, e un record solo parte dopo ``max_wait``
    rd = synthetic_rd(cfg, 3, 60.0, 0, chain)
    out = simulate_strategy(rd, "best_rssi", rcfg)
    assert out["max_packet_bytes_sent"] <= rcfg["max_packet_bytes"]
    delay = out["log"]["t_arrive"] - out["log"]["t_create"]
    one_hop = delay[out["log"]["hops"] == 1]
    assert one_hop.max() == pytest.approx(rcfg["max_wait"], abs=1e-6)  # un record solo aspetta ``max_wait`` e poi parte
    assert delay.max() <= 2 * rcfg["max_wait"] + 1e-6  # a ogni salto al più ``max_wait`` di attesa


def test_sos_leaves_immediately(cfg):
    """Un SOS non aspetta ``max_wait``: arriva al capogruppo nello stesso passo in cui nasce (collegamento diretto, p = 1)."""
    rcfg = cfg["routing"]
    step = 355  # 35,5 s: lontano dall'istante dei record (interi)
    rd = synthetic_rd(cfg, 3, 60.0, 0, chain, events=[(step, 1, KIND_SOS)])
    out = simulate_strategy(rd, "best_rssi", rcfg)
    sel = out["log"]["kind"] == KIND_SOS
    assert sel.sum() == 1 and (out["log"]["t_arrive"][sel] - out["log"]["t_create"][sel])[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 8. Carry
# ---------------------------------------------------------------------------


def test_isolated_node_fills_buffer_degrades_keeps_sos_and_delivers_in_time_order(cfg):
    """Nodo 2 isolato fino a 100 s: riempie il buffer (30), poi crea record a 0,2 Hz sovrascrivendo i più vecchi non-SOS e tiene l'SOS; al ritorno consegna in ordine di tempo."""
    cfg["routing"]["buffer_records"] = 30
    rcfg = cfg["routing"]
    back = 100.0
    topo = lambda i, j, tt: chain(i, j, tt) & ~(({i, j} == {1, 2}) & (tt < back))  # noqa: E731
    rd = synthetic_rd(cfg, 3, 140.0, 0, topo, events=[(100, 2, KIND_SOS)])  # SOS a 10 s, durante l'isolamento
    out = simulate_strategy(rd, "best_rssi", rcfg)
    created = out["created"]
    mine = created[(created[:, 0] == 2) & (created[:, 3] == KIND_NORMAL)]
    during = mine[mine[:, 2] < back]
    # 29 record al secondo (con l'SOS il buffer da 30 è pieno), poi uno ogni 5 s da 30 a 95 s: 14 record
    assert len(during) == 29 + 14
    late = during[during[:, 2] > 40.0]
    assert np.all(np.diff(late[:, 2]) >= 5.0 - 1e-9)  # a buffer pieno: 0,2 Hz
    assert out["max_occupancy"][2] == 30 and out["lost_buffer"][2] > 0
    log = out["log"]
    sos = log["kind"] == KIND_SOS
    assert sos.sum() == 1 and log["t_arrive"][sos][0] >= back  # l'SOS sopravvive e parte al ritorno
    arrive = (log["origin"] == 2) & (log["kind"] == KIND_NORMAL) & (log["t_arrive"] >= back)
    assert arrive.sum() >= 20
    tc = log["t_create"][arrive]
    assert np.all(np.diff(tc) > 0)  # ordine di tempo
    buffered = arrive & (log["t_create"] < back)
    assert buffered.sum() >= 20 and np.all(log["carry"][buffered])  # i record del buffer sono arrivati via carry
    assert not log["carry"][(log["origin"] == 2) & (log["t_create"] > back + 5.0)].any()  # quelli nati dopo il ritorno no


# ---------------------------------------------------------------------------
# 9. Duplicati
# ---------------------------------------------------------------------------


def test_duplicate_records_counted_once(cfg):
    """Con conferma sempre persa (ritorno 0) il capogruppo riceve ogni record più volte, ma ogni (nodo, progressivo) compare una volta sola nel registro."""

    def kill_ack(rd):
        rd["p"][:, 0, 1] = 0.0

    rd = synthetic_rd(cfg, 2, 40.0, 0, chain, p_override=kill_ack)
    out = simulate_strategy(rd, "best_rssi", cfg["routing"])
    keys = list(zip(out["log"]["origin"].tolist(), out["log"]["seq"].tolist()))
    cap = records_per_packet(cfg["routing"], 2)
    assert out["duplicates"] > 0 and len(keys) == len(set(keys)) == cap  # senza conferma il mittente rimanda sempre i `cap` record più vecchi
    assert out["fail_both"] > 0  # il mittente le considera fallite


# ---------------------------------------------------------------------------
# 10. direct
# ---------------------------------------------------------------------------


def test_direct_delivers_only_on_the_direct_link(cfg):
    """Catena 0-1-2-3: `direct` consegna solo i record del nodo 1 (vicino del capogruppo); `best_rssi` anche quelli di 2 e 3 (via ponte)."""
    rd = synthetic_rd(cfg, 4, 60.0, 0, chain)
    d = simulate_strategy(rd, "direct", cfg["routing"])
    assert set(np.unique(d["log"]["origin"])) == {1}
    assert np.all(d["log"]["hops"] == 1) and d["bridge_changes"] == 0
    b = simulate_strategy(rd, "best_rssi", cfg["routing"])
    assert set(np.unique(b["log"]["origin"])) == {1, 2, 3}
    assert d["max_occupancy"][3] > 0 and max(d["queue_lengths"][2:]) > 30  # senza ponte i record dei nodi lontani restano in coda (carry)


# ---------------------------------------------------------------------------
# 11. Riproducibilità
# ---------------------------------------------------------------------------


def test_reproducible_same_seed_same_result_and_position_class(cfg):
    """Stesso seme, stesso risultato per tutte le strategie, con ricezioni probabilistiche; un seme diverso cambia gli unicast."""

    def lossy(rd):
        rd["p"][:] = np.where(rd["p"] > 0, 0.6, 0.0)

    rd = synthetic_rd(cfg, 4, 80.0, 0, chain, p_override=lossy)
    for name in STRATEGIES:
        a, b = simulate_strategy(rd, name, cfg["routing"]), simulate_strategy(rd, name, cfg["routing"])
        for key in a["log"]:
            np.testing.assert_array_equal(a["log"][key], b["log"][key])
        assert a["breaks"] == b["breaks"] and a["bridge_changes"] == b["bridge_changes"]
    other = dict(rd, seed=12)
    assert not np.array_equal(simulate_strategy(rd, "dqrem", cfg["routing"])["log"]["t_arrive"], simulate_strategy(other, "dqrem", cfg["routing"])["log"]["t_arrive"])
    # posizione del capogruppo: terzi del rango (10 nodi: coda 0-2, centro 3-6, testa 7-9)
    offsets = np.arange(10, dtype=float)[::-1]
    assert [position_class(offsets, 9)[1], position_class(offsets, 6)[1], position_class(offsets, 0)[1]] == [0, 1, 2]


# ---------------------------------------------------------------------------
# 12. Profili di canale, connettività, portata
# ---------------------------------------------------------------------------


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        out.update(_flatten(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v})
    return out


def test_nominal_profile_is_the_previous_block7_config_and_heltec_changes_three_keys(cfg):
    """`nominal` = solo ``scenario_overrides`` (la config del primo punto di lavoro, quindi risultati identici); `heltec_run` cambia soltanto potenza, sensibilità e soglia dei candidati."""
    from src.refinement import apply_overrides
    from src.routing_experiment import block7_config

    previous = apply_overrides(cfg, cfg["routing"]["scenario_overrides"])
    nominal = block7_config(cfg, "nominal")
    assert nominal == previous
    heltec = block7_config(cfg, "heltec_run")
    a, b = _flatten(nominal), _flatten(heltec)
    changed = {k for k in a if a[k] != b[k]}
    assert changed == {"channel.tx_power", "packets.reception.sensitivity_dbm", "routing.candidate_threshold"}
    assert (b["packets.reception.sensitivity_dbm"], b["routing.candidate_threshold"]) == (-96.0, -88.0)
    assert b["routing.candidate_threshold"] - b["packets.reception.sensitivity_dbm"] == pytest.approx(8.0)
    assert block7_config(cfg)["channel"]["tx_power"] == heltec["channel"]["tx_power"]  # default = routing.channel_profile
    with pytest.raises(KeyError):
        block7_config(cfg, "inesistente")


def test_heltec_profile_median_rssi_in_compact_group_of_five(cfg):
    """Con la potenza equivalente del profilo ``heltec_run`` la mediana dell'RSSI nel gruppo da 5 compatti (scenario none, canale base) è −78 ± 1 dBm; a 10 dBm è −51."""
    from src.refinement import apply_overrides
    from src.routing_experiment import group_median_rssi

    channel_only = {k: v for k, v in cfg["routing"]["channel_profiles"]["heltec_run"].items() if k.startswith("channel.")}
    assert group_median_rssi(apply_overrides(cfg, channel_only)) == pytest.approx(-78.0, abs=1.0)
    assert group_median_rssi(cfg) == pytest.approx(-51.0, abs=1.0)


def test_connectivity_on_a_cut_chain(cfg):
    """Catena 0-1-2-3 con il capogruppo 0; da metà il collegamento 1-2 cade. Prima tutti connessi e gruppo intero; dopo, i nodi 2 e 3 senza strada e gruppo spezzato in 2 parti (campioni a 1 s)."""
    from src.routing import connectivity_stats

    t = np.arange(0.0, 100.0, 0.1)
    rssi = np.full((len(t), 4, 4), -120.0)
    for i in range(3):
        rssi[:, i, i + 1] = rssi[:, i + 1, i] = -70.0
    rssi[t >= 50.0, 1, 2] = rssi[t >= 50.0, 2, 1] = -120.0
    c = connectivity_stats(t, rssi, 0, -1, float("nan"), cfg["routing"], 10)
    np.testing.assert_allclose(c["in"] / c["n"], [1.0, 1.0, 0.5, 0.5])
    assert c["split"] == 50 and c["split_n"] == 100
    # un solo senso sopra soglia non basta (serve anche la conferma)
    rssi[:, 0, 1] = -120.0
    c = connectivity_stats(t, rssi, 0, -1, float("nan"), cfg["routing"], 10)
    assert c["in"][1] == 0


def test_connectivity_excludes_separated_node_from_the_group(cfg):
    """Dopo il distacco il nodo separato esce dal gruppo: non rompe la connettività del gruppo e viene contato a parte sul grafo completo."""
    from src.routing import connectivity_stats

    t = np.arange(0.0, 100.0, 0.1)
    rssi = np.full((len(t), 3, 3), -70.0)
    rssi[t >= 60.0, 2, :] = rssi[t >= 60.0, :, 2] = -120.0  # il nodo 2 si stacca a 60 s
    c = connectivity_stats(t, rssi, 0, 2, 60.0, cfg["routing"], 10)
    assert c["split"] == 0 and c["n"][2] == 60 and c["in"][2] == 60
    assert c["sep_n"] == 40 and c["sep_in"] == 0


def test_range_inversion_matches_path_loss(cfg):
    """La portata solo-distanza inverte `path_loss_db`: a quella distanza il livello vale il valore richiesto."""
    from src.channel import path_loss_db
    from src.routing_experiment import block7_config, distance_range

    c = block7_config(cfg, "heltec_run")
    for level in (-88.0, -96.0):
        d = distance_range(level, c)
        ch = c["channel"]
        assert ch["tx_power"] + ch["antenna_gain_tx"] + ch["antenna_gain_rx"] - float(path_loss_db(np.array(d), ch)) == pytest.approx(level, abs=1e-6)

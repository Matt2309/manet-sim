"""Blocco 7 — Esperimento del routing verso il capogruppo: corse, riassunti, aggregazione, verdetto.

Gruppo lungo (sostituzioni di ``routing.scenario_overrides``), scenari ``none`` / ``slowdown`` / ``stop`` con ``routing.n_runs`` corse ciascuno. Semi delle corse e nodo/istante del
distacco dai generatori ``routing.seed_keys.runs`` / ``separation`` (comuni ai tre scenari: confronto a coppie), capogruppo dal generatore ``leader`` (mai il nodo che si stacca, stesso nei tre
scenari), frazione di tempo del SOS dal generatore ``sos``. Il rilevatore NON si valuta qui (vedi `src/routing.py`). Ogni corsa è indipendente (`process_run_routing`): il risultato non dipende
dal numero di processi. Nessuna cache su disco: ogni processo simula la propria corsa.

Parametri: sezione ``routing`` della config.
"""

from __future__ import annotations

import copy
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from src.final import bootstrap_rng
from src.metrics import min_distance_to_group, nearest_rank
from src.refinement import apply_overrides
from src.routing import prepare_run, simulate_strategy
from src.simulator import draw_separations, experiment_runs

POSITIONS = ("coda", "centro", "testa")


# ---------------------------------------------------------------------------
# Insiemi di corse
# ---------------------------------------------------------------------------


def block7_config(config: dict, profile: str | None = None) -> dict:
    """Copia della config con le sostituzioni del Blocco 7: gruppo lungo (``routing.scenario_overrides``) e profilo di canale.

    `profile` è ``nominal`` o ``heltec_run`` (default ``routing.channel_profile``); le sostituzioni del profilo stanno in ``routing.channel_profiles``.
    ``nominal`` non cambia nulla rispetto al primo punto di lavoro del Blocco 7.
    """
    rcfg = config["routing"]
    profile = rcfg["channel_profile"] if profile is None else profile
    if profile not in rcfg["channel_profiles"]:
        raise KeyError(f"routing.channel_profile sconosciuto: {profile!r} (disponibili: {', '.join(rcfg['channel_profiles'])})")
    cfg = apply_overrides(config, rcfg["scenario_overrides"])
    return apply_overrides(cfg, rcfg["channel_profiles"][profile] or {})


def group_median_rssi(config: dict, duration: float = 300.0) -> float:
    """Mediana dell'RSSI misurato fra i nodi del gruppo da 5 compatti, scenario ``none``, canale base del Blocco 2.

    Serve a tarare ``channel.tx_power`` del profilo ``heltec_run``: la potenza trovata è una potenza EQUIVALENTE (EIRP comprensivo delle perdite di antenna,
    montaggio sul braccio e corpi), NON un valore impostabile sull'ESP32. Stessa statistica di ``scripts/plot_channel.py`` (mediana −51 dBm a 10 dBm).
    `config`: config base con le sostituzioni di canale da provare (qui NON si applica il gruppo lungo). Output: dBm.
    """
    from src.channel import simulate_channel
    from src.mobility import simulate_mobility

    cfg = copy.deepcopy(config)
    cfg["separation"]["enabled"] = False
    cfg["simulation"]["duration"] = duration
    res = simulate_channel(simulate_mobility(cfg), cfg)
    n = res.rssi_measured.shape[1]
    return float(np.median(res.rssi_measured[:, ~np.eye(n, dtype=bool)]))


def draw_leaders(config: dict, sep_nodes: np.ndarray) -> np.ndarray:
    """Capogruppo di ogni corsa: uniforme fra i nodi diversi da quello che si stacca (generatore ``SeedSequence(simulation.seed, spawn_key=(seed_keys.leader,))``).

    Input: `config` (del Blocco 7), `sep_nodes` (n_runs,) nodi estratti per il distacco. Procedimento: indice uniforme fra ``N − 1`` posizioni, saltando il nodo del distacco. Output (n_runs,) int.
    """
    rng = np.random.default_rng(np.random.SeedSequence(config["simulation"]["seed"], spawn_key=(config["routing"]["seed_keys"]["leader"],)))
    u = rng.integers(0, config["group"]["n_nodes"] - 1, size=len(sep_nodes))
    return (u + (u >= sep_nodes)).astype(int)


def draw_sos_fractions(config: dict, n_runs: int) -> np.ndarray:
    """Istante del SOS come frazione di ``sos_window`` dopo il distacco: uniformi (0, 1) dal generatore ``spawn_key=(seed_keys.sos,)``, una per corsa (uguale nei tre scenari)."""
    rng = np.random.default_rng(np.random.SeedSequence(config["simulation"]["seed"], spawn_key=(config["routing"]["seed_keys"]["sos"],)))
    return rng.random(n_runs)


def routing_runs(config: dict) -> list:
    """Corse del Blocco 7 (`experiment_runs` con chiavi 12/13 e ``routing.n_runs``), ciascuna con ``leader`` e ``sos`` (frazione). `config` è quella del Blocco 7 (`block7_config`)."""
    rcfg = config["routing"]
    keys = {"runs": rcfg["seed_keys"]["runs"], "separation": rcfg["seed_keys"]["separation"]}
    runs = experiment_runs(config, keys, rcfg["n_runs"])
    nodes, _ = draw_separations(config, keys["separation"], rcfg["n_runs"])
    leaders, sos = draw_leaders(config, nodes), draw_sos_fractions(config, rcfg["n_runs"])
    for r in runs:
        r["leader"], r["sos"] = int(leaders[r["index"]]), float(sos[r["index"]])
    return runs


# ---------------------------------------------------------------------------
# Riassunto di una strategia su una corsa
# ---------------------------------------------------------------------------


def _keys(origin: np.ndarray, seq: np.ndarray) -> np.ndarray:
    return np.asarray(origin).astype(np.int64) * 10_000_000 + np.asarray(seq).astype(np.int64)


def summarize_strategy(out: dict, rd: dict, rcfg: dict, scenario: str) -> dict:
    """Riduce l'uscita di `simulate_strategy` a poche cifre per corsa.

    Input: uscita della simulazione, corsa preparata `rd`, `rcfg`, scenario.
    Procedimento: record normali creati fra ``warmup`` e ``fine − end_margin``, divisi in classe ``group`` e ``sep`` (record del nodo separato dopo il distacco); per ciascuna: creati,
    consegnati, ritardi, istogramma dei salti, record arrivati via carry (con ritardi), consegna per fascia di distanza dal capogruppo. Età della vista del capogruppo (da quanto non riceve un
    record NUOVO di un nodo: ``t − max t_creazione`` dei record arrivati) campionata ogni secondo da ``warmup``: istogramma (60 contenitori da 1 s, l'ultimo raccoglie il resto) e conteggi oltre le
    soglie ``view_age_thresholds``. Separato: ultimo record fresco (età < ``fresh_age``), con tempo dal distacco e distanza dal capogruppo e dal gruppo. Allarme oracolo e SOS: consegnati e ritardo.
    Carico: tempo d'aria e tentativi per nodo (al secondo), occupazione massima del buffer, record persi, salti mediani annunciati.
    Output: dizionario di scalari e piccoli array.
    """
    t, L, sep, sep_start, dt = rd["t"], rd["leader"], rd["sep_node"], rd["sep_start"], rd["dt"]
    n_nodes = rd["ann_grid"].shape[1]
    with_sep = scenario != "none"
    duration = float(t[-1])
    lo, hi = rcfg["warmup"], duration - rcfg["end_margin"]
    created, log = out["created"], out["log"]
    lkey = _keys(log["origin"], log["seq"])
    sorter = np.argsort(lkey)

    def lookup(ck):
        pos = np.minimum(np.searchsorted(lkey[sorter], ck), max(len(lkey) - 1, 0))
        if len(lkey) == 0:
            return np.zeros(len(ck), dtype=bool), np.zeros(len(ck), dtype=int)
        idx = sorter[pos]
        return lkey[idx] == ck, idx

    normal = (created[:, 3] == 0) & (created[:, 2] >= lo) & (created[:, 2] <= hi)
    cn = created[normal]
    hit, idx = lookup(_keys(cn[:, 0], cn[:, 1]))
    delay = np.where(hit, log["t_arrive"][idx] - cn[:, 2], np.nan) if len(cn) else np.zeros(0)
    hops = np.where(hit, log["hops"][idx], 0).astype(int) if len(cn) else np.zeros(0, dtype=int)
    carry = np.where(hit, log["carry"][idx], False).astype(bool) if len(cn) else np.zeros(0, dtype=bool)
    dist = rd["distances"][cn[:, 4].astype(int), cn[:, 0].astype(int), L].astype(float) if len(cn) else np.zeros(0)
    is_sep = with_sep & (cn[:, 0] == sep) & (cn[:, 2] >= sep_start) if len(cn) else np.zeros(0, dtype=bool)
    edges = np.asarray(rcfg["distance_bins"], dtype=float)

    def class_stats(mask):
        got = mask & hit
        return {
            "created": int(mask.sum()), "delivered": int(got.sum()), "delays": delay[got].astype(np.float32),
            "hops": np.bincount(hops[got], minlength=rcfg["max_hops"] + 1)[: rcfg["max_hops"] + 1],
            "carried": int((got & carry).sum()), "carry_delays": delay[got & carry].astype(np.float32),
            "dist_created": np.histogram(dist[mask], bins=edges)[0], "dist_delivered": np.histogram(dist[got], bins=edges)[0],
        }  # fmt: skip

    classes = {"group": class_stats(~is_sep), "sep": class_stats(is_sep)}

    # età della vista del capogruppo
    thresholds = rcfg["view_age_thresholds"]
    grid = np.arange(lo, duration, 1.0)
    age_edges = np.arange(0.0, 61.0, 1.0)
    view = {c: {"hist": np.zeros(60, dtype=np.int64), "n": 0, "over": np.zeros(len(thresholds), dtype=np.int64)} for c in ("group", "sep")}
    for node in range(n_nodes):
        if node == L:
            continue
        sel = (log["origin"] == node) & (log["kind"] == 0)
        ta, tc = log["t_arrive"][sel], log["t_create"][sel]
        order = np.argsort(ta, kind="stable")
        ta, tc = ta[order], tc[order]
        if len(ta):
            pos = np.searchsorted(ta, grid, side="right") - 1
            last = np.where(pos >= 0, np.maximum.accumulate(tc)[np.maximum(pos, 0)], 0.0)
        else:
            last = np.zeros(len(grid))
        age = grid - last
        sep_time = (with_sep and node == sep) & (grid >= sep_start)
        for name, mask in (("group", ~sep_time), ("sep", sep_time)):
            a = age[mask]
            view[name]["hist"] += np.histogram(np.minimum(a, 60.0), bins=age_edges)[0]
            view[name]["n"] += len(a)
            view[name]["over"] += np.array([(a > th).sum() for th in thresholds])

    # nodo separato: ultimo record fresco, allarme oracolo e SOS
    sep_out = {"fresh_time": np.nan, "fresh_dist_leader": np.nan, "fresh_dist_group": np.nan}
    if with_sep:
        sel = (log["origin"] == sep) & (log["kind"] == 0) & (log["t_create"] >= sep_start) & (log["t_arrive"] - log["t_create"] < rcfg["fresh_age"])
        if sel.any():
            ta = float(log["t_arrive"][sel].max())
            step = min(int(round(ta / dt)), len(t) - 1)
            sep_out = {
                "fresh_time": ta - sep_start,
                "fresh_dist_leader": float(rd["distances"][step, sep, L]),
                "fresh_dist_group": float(min_distance_to_group(rd["distances"][step : step + 1], sep)[0]),
            }

    def message(kind):
        rows = created[created[:, 3] == kind]
        if len(rows) == 0:
            return {"created": False, "delivered": False, "delay": np.nan}
        ok, i = lookup(_keys(rows[:, 0], rows[:, 1]))
        return {"created": True, "delivered": bool(ok[0]), "delay": float(log["t_arrive"][i[0]] - rows[0, 2]) if ok[0] else np.nan}

    hops_med = np.array([np.median(rd["ann_grid"][:, n][rd["ann_grid"][:, n] < 255]) if (rd["ann_grid"][:, n] < 255).any() else np.nan for n in range(n_nodes)])
    return {
        "classes": classes, "view": view, "sep": sep_out, "oracle": message(1), "sos": message(2),
        "breaks": out["breaks"], "break_episodes": out["break_episodes"], "fail_both": out["fail_both"], "bridge_changes": out["bridge_changes"],
        "ttl_drops": out["ttl_drops"], "duplicates": out["duplicates"], "cycles": out["cycles"], "capacity": out["capacity"], "max_packet_bytes_sent": out["max_packet_bytes_sent"],
        "airtime_per_s": out["tx_airtime"] / duration, "packets_per_s": out["attempts"] / duration, "max_occupancy": out["max_occupancy"],
        "lost_buffer": out["lost_buffer"], "hops_median": hops_med, "duration": duration,
    }  # fmt: skip


# ---------------------------------------------------------------------------
# Lavoro per corsa
# ---------------------------------------------------------------------------


def process_run_routing(args: tuple) -> dict:
    """Una corsa del Blocco 7: la prepara una volta (`prepare_run`) e valuta tutte le strategie sulla stessa corsa (stesse stime, stessi beacon, stesso generatore degli unicast).

    Input: `args` = (corsa di `routing_runs`, strategie, dimensioni di pacchetto del confronto ESP-NOW v2 (per `dqrem`), `trace`).
    Output: ``scenario``, ``index``, ``leader``, ``pos_class``, ``rank``, ``extent``, ``wrong`` (salti sbagliati), ``strategies`` (nome → `summarize_strategy`), ``bytes`` (dimensione → riassunto di `dqrem`),
    ``elapsed``, ``trace`` (solo se richiesto: decisioni del separato, record del separato, distanza dal gruppo).
    """
    run, strategies, byte_sweep, trace = args
    t0 = time.perf_counter()
    cfg = run["config"]
    rcfg = cfg["routing"]
    rd = prepare_run(cfg, run["leader"], run["sos"])
    t_prepare = time.perf_counter() - t0
    res = {
        "scenario": run["scenario"], "index": run["index"], "leader": run["leader"], "pos_class": rd["pos_class"], "rank": rd["rank"], "extent": rd["extent"],
        "wrong": rd["wrong"], "R": rd["R"], "node_rank": rd["node_rank"], "connectivity": rd["connectivity"], "range_hist": rd["range_hist"], "strategies": {}, "bytes": {}, "times": {}, "weights": {}, "trace": None,
    }  # fmt: skip
    traced = None
    for name in strategies:
        t1 = time.perf_counter()
        out = simulate_strategy(rd, name, rcfg, trace=trace and name == "dqrem")
        res["strategies"][name] = summarize_strategy(out, rd, rcfg, run["scenario"])
        res["times"][name] = time.perf_counter() - t1
        if "ahp_weights" in out:
            res["weights"][name] = (out["ahp_weights"], out["cr"])
        if trace and name == "dqrem":
            traced = out
    for size in byte_sweep:
        out = simulate_strategy(rd, "dqrem", rcfg, max_packet_bytes=size)
        res["bytes"][size] = summarize_strategy(out, rd, rcfg, run["scenario"])
    if trace and traced is not None and rd["sep_node"] >= 0:
        sep, L = rd["sep_node"], rd["leader"]
        log = traced["log"]
        sel = (log["origin"] == sep) & (log["kind"] == 0)
        stride = int(round(1.0 / rd["dt"]))
        res["trace"] = {
            "t": rd["t"][::stride], "sep_start": rd["sep_start"], "sep_node": sep, "leader": L,
            "dist_group": min_distance_to_group(rd["distances"][::stride], sep), "dist_leader": rd["distances"][::stride, sep, L].astype(float),
            "decisions": np.asarray(traced["decisions"], dtype=float).reshape(-1, 4),
            "rec_create": log["t_create"][sel], "rec_arrive": log["t_arrive"][sel], "rec_hops": log["hops"][sel], "rec_carry": log["carry"][sel],
            "announced": rd["ann_grid"][::stride, sep], "breaks": np.asarray(traced["break_times"]),
        }  # fmt: skip
    res["elapsed"] = time.perf_counter() - t0
    res["t_prepare"] = t_prepare
    return res


def run_all(runs: list, strategies: list, byte_sweep: list, jobs: int, reference: tuple, done: dict | None = None) -> list:
    """`process_run_routing` su tutte le corse (in parallelo se ``jobs > 1``; risultato indipendente da `jobs`). `reference` = (scenario, indice) della corsa con la traccia; `done` = risultati già pronti per (scenario, indice)."""
    done = done or {}
    todo = [r for r in runs if (r["scenario"], r["index"]) not in done]
    tasks = [(r, strategies, byte_sweep, (r["scenario"], r["index"]) == reference) for r in todo]
    if jobs <= 1:
        outs = [process_run_routing(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            outs = list(pool.map(process_run_routing, tasks, chunksize=1))
    allout = list(done.values()) + outs
    return sorted(allout, key=lambda o: (o["scenario"], o["index"]))


# ---------------------------------------------------------------------------
# Aggregazione
# ---------------------------------------------------------------------------


def select(outs: list, scenario: str | None = None, pos: int | None = None) -> list:
    """Corse di `outs` di uno scenario (None = tutti) e con il capogruppo in una posizione (0 coda, 1 centro, 2 testa; None = tutte)."""
    return [o for o in outs if (scenario is None or o["scenario"] == scenario) and (pos is None or o["pos_class"] == pos)]


def pooled(outs: list, strategy: str, cls: str) -> dict:
    """Somma sulle corse delle statistiche di una classe (``group`` o ``sep``): creati, consegnati, ritardi concatenati, salti, carry, consegna per fascia di distanza."""
    parts = [o["strategies"][strategy]["classes"][cls] for o in outs]
    if not parts:
        return {"created": 0, "delivered": 0, "delays": np.zeros(0), "hops": 0, "carried": 0, "carry_delays": np.zeros(0), "dist_created": 0, "dist_delivered": 0}
    return {
        "created": sum(p["created"] for p in parts), "delivered": sum(p["delivered"] for p in parts),
        "delays": np.concatenate([p["delays"] for p in parts]), "hops": sum(p["hops"] for p in parts), "carried": sum(p["carried"] for p in parts),
        "carry_delays": np.concatenate([p["carry_delays"] for p in parts]), "dist_created": sum(p["dist_created"] for p in parts),
        "dist_delivered": sum(p["dist_delivered"] for p in parts),
    }  # fmt: skip


def delivery_stats(p: dict) -> tuple:
    """(frazione consegnata, ritardo mediano s, ritardo p95 s) di una classe già sommata; NaN se non ci sono record."""
    frac = p["delivered"] / p["created"] if p["created"] else float("nan")
    return frac, nearest_rank(p["delays"], 0.5), nearest_rank(p["delays"], 0.95)


def view_over(outs: list, strategy: str, cls: str) -> tuple:
    """Frazione di tempo con età della vista oltre ogni soglia (``view_age_thresholds``) e campioni, sommando le corse."""
    n = sum(o["strategies"][strategy]["view"][cls]["n"] for o in outs)
    over = sum(o["strategies"][strategy]["view"][cls]["over"] for o in outs) if outs else 0
    return (over / n if n else np.full(2, np.nan)), n


def per_run_values(outs: list, strategy: str) -> dict:
    """Valori per corsa di una strategia (per il confronto a coppie): ``breaks`` (tentativi falliti dopo tutte le ritrasmissioni al primo candidato), ``delivery`` (frazione consegnata, nodi nel gruppo),
    ``age5`` (frazione di tempo con età della vista > prima soglia, nodi nel gruppo)."""
    breaks = np.array([o["strategies"][strategy]["breaks"] for o in outs], dtype=float)
    delivery = np.array([o["strategies"][strategy]["classes"]["group"]["delivered"] / max(o["strategies"][strategy]["classes"]["group"]["created"], 1) for o in outs])
    age5 = np.array([o["strategies"][strategy]["view"]["group"]["over"][0] / max(o["strategies"][strategy]["view"]["group"]["n"], 1) for o in outs])
    return {"breaks": breaks, "delivery": delivery, "age5": age5}


def paired_bootstrap(a: dict, b: dict, config: dict) -> dict:
    """Confronto a coppie `a` (es. ``dqrem``) contro `b` (es. ``best_rssi``) con bootstrap per corsa (generatore ``final.seed_keys.bootstrap`` = chiave 11, ``final.bootstrap_samples`` ricampionamenti).

    Input: valori per corsa (`per_run_values`) delle due strategie sulle stesse corse. Procedimento: si ricampionano le CORSE con reinserimento, lo stesso ricampionamento per entrambe.
    Output: ``break_reduction`` = ``1 − Σ rotture(a) / Σ rotture(b)`` (valore e intervallo percentile al livello ``metrics.confidence_level``), ``delivery_diff`` e ``age5_diff`` = medie delle differenze (a − b).
    """
    rng = bootstrap_rng(config)
    n = len(a["breaks"])
    idx = rng.integers(0, n, size=(config["final"]["bootstrap_samples"], n))
    alpha = 1.0 - config["metrics"]["confidence_level"]

    def interval(values):
        return tuple(float(x) for x in np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0]))

    def reduction(ia):
        sb = b["breaks"][ia].sum(axis=-1)
        return np.where(sb > 0, 1.0 - a["breaks"][ia].sum(axis=-1) / np.where(sb > 0, sb, 1.0), np.nan)

    sb = b["breaks"].sum()
    point = float(1.0 - a["breaks"].sum() / sb) if sb > 0 else float("nan")
    red = reduction(idx)
    out = {"break_reduction": (point, interval(red[np.isfinite(red)]) if sb > 0 else (np.nan, np.nan)), "n_runs": n}
    for name in ("delivery", "age5"):
        diff = a[name] - b[name]
        out[name + "_diff"] = (float(diff.mean()), interval(diff[idx].mean(axis=1)))
    return out


def predictive_verdict(outs: list, config: dict) -> dict:
    """La parte predittiva ripaga? ``dqrem`` contro ``best_rssi`` e contro ``dqrem_no_slope`` (`paired_bootstrap`) su tutte le corse e per scenario.

    Regola fissata prima dei risultati: DQREM "ripaga" rispetto a ``best_rssi`` se riduce le rotture di almeno ``min_break_reduction`` e NON peggiora la consegna, cioè l'estremo superiore
    dell'intervallo della differenza di consegna è ≥ 0 (la differenza non è significativamente negativa).
    Output: ``{"all": {...}, scenario: {...}}``, ciascuno con ``vs_best_rssi``, ``vs_no_slope`` e ``pays`` (bool; per tutte le corse e per scenario).
    """
    need = config["routing"]["min_break_reduction"]
    out = {}
    for key, subset in (("all", outs), *((s, select(outs, s)) for s in ("none", "slowdown", "stop"))):
        if not subset:
            continue
        dq = per_run_values(subset, "dqrem")
        res = {name: paired_bootstrap(dq, per_run_values(subset, other), config) for name, other in (("vs_best_rssi", "best_rssi"), ("vs_no_slope", "dqrem_no_slope"))}
        v = res["vs_best_rssi"]
        reduction = v["break_reduction"][0]
        res["pays"] = bool(np.isfinite(reduction) and reduction >= need and v["delivery_diff"][1][1] >= 0.0)
        out[key] = res
    return out


# ---------------------------------------------------------------------------
# Stampa
# ---------------------------------------------------------------------------


def _f(x, spec):
    return "   -" if x is None or (isinstance(x, float) and np.isnan(x)) else format(x, spec)


def calibration_text(outs: list, config: dict) -> str:
    """Calibrazione del gruppo lungo: sigma, estensione mediana/p95 osservata (longitudinale e laterale, media sulle corse ``none``)."""
    none = select(outs, "none") or outs
    e = [o["extent"] for o in none]
    mean = lambda k: float(np.mean([x[k] for x in e]))  # noqa: E731
    g = config["group"]
    return (
        f"Gruppo lungo: {g['n_nodes']} nodi, estensione longitudinale richiesta p95 {g['longitudinal_spread']:g} m, laterale {g['lateral_spread']:g} m\n"
        f"  sigma_long = {mean('sigma_long'):.3f} m, sigma_lat = {mean('sigma_lat'):.3f} m\n"
        f"  estensione osservata (media sulle {len(none)} corse): longitudinale mediana {mean('long_median'):.1f} m, p95 {mean('long_p95'):.1f} m;"
        f" laterale mediana {mean('lat_median'):.1f} m, p95 {mean('lat_p95'):.1f} m"
    )


def delivery_table(outs: list, config: dict, pos: int | None = None) -> str:
    """Tabella per strategia e scenario: consegna e ritardo (nodi nel gruppo e separato), età della vista, rotture, cambi di ponte, scarti TTL (per corsa)."""
    thr = config["routing"]["view_age_thresholds"]
    lines = [
        f"{'strategia':<15} {'scen.':<9} {'gr.cons':>7} {'med s':>6} {'p95 s':>6} {'sep.cons':>8} {'med s':>6} {'p95 s':>6}"
        f" {'>' + format(thr[0], 'g') + 's gr':>7} {'>' + format(thr[1], 'g') + 's gr':>7} {'>' + format(thr[0], 'g') + 's sep':>8} {'rotture/c':>9} {'ep./c':>6} {'ponti/c':>7} {'TTL/c':>5}"
    ]
    for scen in ("none", "slowdown", "stop"):
        sub = select(outs, scen, pos)
        if not sub:
            continue
        for name in config["routing"]["strategies"]:
            g, s = delivery_stats(pooled(sub, name, "group")), delivery_stats(pooled(sub, name, "sep"))
            og, _ = view_over(sub, name, "group")
            osep, _ = view_over(sub, name, "sep")
            mean = lambda k: float(np.mean([o["strategies"][name][k] for o in sub]))  # noqa: E731
            lines.append(
                f"{name:<15} {scen:<9} {_f(g[0], '7.3f'):>7} {_f(g[1], '6.1f'):>6} {_f(g[2], '6.1f'):>6} {_f(s[0], '8.3f'):>8} {_f(s[1], '6.1f'):>6} {_f(s[2], '6.1f'):>6}"
                f" {_f(float(og[0]), '7.3f'):>7} {_f(float(og[1]), '7.3f'):>7} {_f(float(osep[0]), '8.3f'):>8} {mean('breaks'):>9.1f} {mean('break_episodes'):>6.1f}"
                f" {mean('bridge_changes'):>7.1f} {mean('ttl_drops'):>5.1f}"
            )
    return "\n".join(lines)


def separated_table(outs: list, config: dict) -> str:
    """Nodo separato: ultimo record fresco (tempo dal distacco e distanza), carry (record e ritardo), allarme oracolo e SOS (consegnati e ritardo), per strategia e scenario."""
    lines = [
        f"{'strategia':<15} {'scen.':<9} {'fresco t s':>10} {'dist.cap m':>10} {'dist.gr m':>9} {'carry rec':>9} {'carry rit.med s':>15} {'oracolo':>9} {'rit.s':>6} {'SOS':>9} {'rit.s':>6}"
    ]
    for scen in ("slowdown", "stop"):
        sub = select(outs, scen)
        for name in config["routing"]["strategies"]:
            rows = [o["strategies"][name] for o in sub]
            med = lambda key: float(np.nanmedian([r["sep"][key] for r in rows])) if rows else float("nan")  # noqa: E731
            sep = pooled(sub, name, "sep")
            orc, sos = [r["oracle"] for r in rows], [r["sos"] for r in rows]
            frac = lambda m: sum(x["delivered"] for x in m) / max(sum(x["created"] for x in m), 1)  # noqa: E731
            dly = lambda m: float(np.nanmedian([x["delay"] for x in m])) if any(x["delivered"] for x in m) else float("nan")  # noqa: E731
            lines.append(
                f"{name:<15} {scen:<9} {_f(med('fresh_time'), '10.1f'):>10} {_f(med('fresh_dist_leader'), '10.1f'):>10} {_f(med('fresh_dist_group'), '9.1f'):>9}"
                f" {sep['carried']:>9d} {_f(nearest_rank(sep['carry_delays'], 0.5), '15.1f'):>15} {frac(orc):>9.2f} {_f(dly(orc), '6.1f'):>6} {frac(sos):>9.2f} {_f(dly(sos), '6.1f'):>6}"
            )
    return "\n".join(lines)


def load_table(outs: list, config: dict) -> str:
    """Carico: tempo d'aria (frazione del tempo) e pacchetti/s per nodo, massimo fra i nodi e media, per strategia; occupazione massima del buffer e record persi per buffer pieno."""
    lines = [f"{'strategia':<15} {'aria max %':>10} {'aria media %':>12} {'pkt/s max':>9} {'pkt/s media':>11} {'buffer max':>10} {'persi/corsa':>11} {'duplicati/c':>11}"]
    for name in config["routing"]["strategies"]:
        air = np.array([o["strategies"][name]["airtime_per_s"] for o in outs])
        pkt = np.array([o["strategies"][name]["packets_per_s"] for o in outs])
        occ = np.array([o["strategies"][name]["max_occupancy"] for o in outs])
        lost = np.array([o["strategies"][name]["lost_buffer"].sum() for o in outs])
        dup = np.mean([o["strategies"][name]["duplicates"] for o in outs])
        lines.append(f"{name:<15} {100 * air.max():>10.2f} {100 * air.mean():>12.2f} {pkt.max():>9.2f} {pkt.mean():>11.2f} {occ.max():>10d} {lost.mean():>11.1f} {dup:>11.1f}")
    return "\n".join(lines)


def hops_table(outs: list, config: dict) -> str:
    """Distribuzione dei salti dei record consegnati (nodi nel gruppo), per strategia, in percentuale."""
    k = config["routing"]["max_hops"] + 1
    lines = [f"{'strategia':<15} " + " ".join(f"{h:>6d}" for h in range(k)) + "  salti medi"]
    for name in config["routing"]["strategies"]:
        h = pooled(outs, name, "group")["hops"]
        h = np.asarray(h, dtype=float)
        tot = h.sum()
        lines.append(f"{name:<15} " + " ".join(f"{100 * x / tot:>5.1f}%" for x in h) + f"  {float((h * np.arange(k)).sum() / tot):>9.2f}")
    return "\n".join(lines)


def wrong_hops_text(outs: list) -> str:
    """Salti annunciati sbagliati: nodi che annunciano un valore finito senza una strada vera (episodi di campioni a 1 s): durata mediana/p95/massima e valore massimo annunciato."""
    d = np.concatenate([o["wrong"]["durations"] for o in outs])
    peak = np.concatenate([o["wrong"]["max_announced"] for o in outs])
    frac = float(np.mean([o["wrong"]["wrong_fraction"] for o in outs]))
    if len(d) == 0:
        return "Salti sbagliati: nessun episodio."
    return (
        f"Salti sbagliati dopo aver perso la strada: {len(d)} episodi in {len(outs)} corse ({frac * 100:.2f}% del tempo-nodo); durata mediana {np.median(d):.0f} s, p95 {nearest_rank(d, 0.95):.0f} s,"
        f" massima {d.max():.0f} s; valore massimo annunciato mediano {np.median(peak):.0f} (max {peak.max()})"
    )


def verdict_text(verdict: dict, config: dict) -> str:
    """Testo del verdetto: per ciascun insieme di corse, riduzione delle rotture, differenza di consegna ed età con intervalli, e se la parte predittiva ripaga."""
    need = config["routing"]["min_break_reduction"]
    lines = [f"Regola: DQREM ripaga se riduce le rotture di almeno {100 * need:.0f}% rispetto a best_rssi E l'intervallo della differenza di consegna non sta tutto sotto lo zero."]
    ci = lambda x: f"{x[0]:+.3f} [{x[1][0]:+.3f}, {x[1][1]:+.3f}]"  # noqa: E731
    for key, res in verdict.items():
        for name, label in (("vs_best_rssi", "dqrem contro best_rssi"), ("vs_no_slope", "dqrem contro dqrem_no_slope")):
            v = res[name]
            red = v["break_reduction"]
            lines.append(
                f"  [{key:<8}] {label:<30} rotture {100 * red[0]:+6.1f}% [{100 * red[1][0]:+6.1f}, {100 * red[1][1]:+6.1f}]  consegna {ci(v['delivery_diff'])}  vista>5s {ci(v['age5_diff'])}  (n={v['n_runs']})"
            )
        lines.append(f"  [{key:<8}] la parte predittiva RIPAGA rispetto a best_rssi: {'SI' if res['pays'] else 'NO'}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Portata, connettività, carry del gruppo, confronto fra profili
# ---------------------------------------------------------------------------


def distance_range(level_dbm: float, config: dict) -> float:
    """Distanza (m) a cui il solo path loss (``channel.tx_power`` + guadagni − ``PL0 + 10 n log10(d/d0)``) dà `level_dbm`; inversione analitica di `path_loss_db`."""
    from src.channel import reference_path_loss

    ch = config["channel"]
    pl = ch["tx_power"] + ch["antenna_gain_tx"] + ch["antenna_gain_rx"] - level_dbm
    pl0 = reference_path_loss(ch["frequency"], ch["path_loss"]["reference_distance"])
    return float(ch["path_loss"]["reference_distance"] * 10.0 ** ((pl - pl0) / (10.0 * ch["path_loss"]["exponent"])))


def median_curve(outs: list) -> tuple:
    """Mediana del segnale (path loss e corpi, senza shadowing, fading e scarti delle schede) per distanza, dall'istogramma sommato sulle corse. Output: (centri di distanza, mediane dBm, campioni)."""
    from src.routing import RANGE_RSSI_EDGES, RANGE_STEP

    hist = sum(o["range_hist"].astype(np.int64) for o in outs)
    n = hist.sum(axis=1)
    cum = np.cumsum(hist, axis=1)
    centers = RANGE_RSSI_EDGES[:-1] + 0.5
    med = np.full(len(hist), np.nan)
    for i in np.flatnonzero(n > 0):
        med[i] = centers[int(np.searchsorted(cum[i], 0.5 * n[i]))]
    return (np.arange(len(hist)) + 0.5) * RANGE_STEP, med, n


def range_text(outs: list, config: dict) -> str:
    """Portata stimata: distanza a cui l'RSSI mediano scende a −88 e −96 dBm, solo con la distanza (analitica) e con i corpi (curva delle mediane, corse di tutti gli scenari)."""
    dist, med, n = median_curve(outs)
    ok = (n >= 200) & np.isfinite(med)
    lines = [f"Portata stimata (potenza equivalente {config['channel']['tx_power']:g} dBm; mediane senza shadowing e fading):"]
    for level in (-88.0, -96.0):
        below = np.flatnonzero(ok & (med <= level))
        with_body = f"{dist[below[0]]:.1f} m" if len(below) else f"oltre {dist[ok][-1]:.0f} m"
        lines.append(f"  RSSI mediano {level:g} dBm: solo distanza {distance_range(level, config):.1f} m, con i corpi {with_body}")
    return "\n".join(lines)


def connectivity_by_rank(outs: list, n_nodes: int) -> tuple:
    """Frazione di tempo con una strada verso il capogruppo per rango longitudinale del nodo (0 = coda), escluso il capogruppo. Output: (frazioni (N,), campioni (N,))."""
    num, den = np.zeros(n_nodes), np.zeros(n_nodes)
    for o in outs:
        c, rank = o["connectivity"], o["node_rank"]
        for node in range(n_nodes):
            if node != o["leader"]:
                num[rank[node]] += c["in"][node]
                den[rank[node]] += c["n"][node]
    return np.where(den > 0, num / np.maximum(den, 1), np.nan), den


def connectivity_summary(outs: list, n_nodes: int) -> dict:
    """Connettività dell'insieme di corse: frazione media (nodi nel gruppo, capogruppo escluso), frazione di tempo con il gruppo spezzato, frazione del separato dopo il distacco."""
    frac, den = connectivity_by_rank(outs, n_nodes)
    num = sum(float(f * d) for f, d in zip(np.nan_to_num(frac), den))
    split = sum(o["connectivity"]["split"] for o in outs) / max(sum(o["connectivity"]["split_n"] for o in outs), 1)
    sep_n = sum(o["connectivity"]["sep_n"] for o in outs)
    sep = sum(o["connectivity"]["sep_in"] for o in outs) / sep_n if sep_n else float("nan")
    return {"nodes": num / max(den.sum(), 1), "split": split, "sep": sep}


def connectivity_table(outs: list, config: dict) -> str:
    """Connettività verso il capogruppo per scenario: per rango del nodo (0 coda … N−1 testa), media, tempo con il gruppo spezzato, separato dopo il distacco."""
    n_nodes = config["group"]["n_nodes"]
    lines = [f"{'scen.':<9} " + " ".join(f"{'r' + str(r):>6}" for r in range(n_nodes)) + f" {'media':>7} {'spezzato':>9} {'separato':>9}"]
    for scen in ("none", "slowdown", "stop"):
        sub = select(outs, scen)
        if not sub:
            continue
        frac, _ = connectivity_by_rank(sub, n_nodes)
        s = connectivity_summary(sub, n_nodes)
        lines.append(f"{scen:<9} " + " ".join(f"{_f(float(x), '6.3f'):>6}" for x in frac) + f" {s['nodes']:>7.3f} {s['split']:>9.3f} {_f(s['sep'], '9.3f'):>9}")
    return "\n".join(lines)


def group_carry_text(outs: list, config: dict) -> str:
    """Telemetria dei nodi nel gruppo che passa per carry (record fermi in coda perché nessun candidato) anche senza distacco: frazione dei consegnati e ritardo mediano/p95, per strategia."""
    lines = [f"{'strategia':<15} {'carry %':>8} {'carry rit.med s':>15} {'carry rit.p95 s':>15}"]
    for name in config["routing"]["strategies"]:
        p = pooled(outs, name, "group")
        lines.append(f"{name:<15} {100 * p['carried'] / max(p['delivered'], 1):>8.2f} {_f(nearest_rank(p['carry_delays'], 0.5), '15.1f'):>15} {_f(nearest_rank(p['carry_delays'], 0.95), '15.1f'):>15}")
    return "\n".join(lines)


def hops_mean(outs: list, name: str) -> float:
    h = np.asarray(pooled(outs, name, "group")["hops"], dtype=float)
    return float((h * np.arange(len(h))).sum() / max(h.sum(), 1))


def profile_comparison_text(by_profile: dict, configs: dict) -> str:
    """Confronto fianco a fianco fra profili di canale (`by_profile`: nome → corse; `configs`: nome → config) per ``best_rssi`` e ``dqrem``, su tutte le corse: consegna e ritardo del gruppo, età della vista,
    salti, carry, consegna del separato, allarme oracolo e SOS, rotture, riduzione delle rotture (e verdetto), connettività, tempo spezzato, tempo d'aria."""
    names = list(by_profile)
    cols = [(p, s) for p in names for s in ("best_rssi", "dqrem")]
    rows = []

    def add(label, fn):
        rows.append((label, [fn(by_profile[p], configs[p], s) for p, s in cols]))

    def delivery(o, c, s):
        return delivery_stats(pooled(o, s, "group"))

    def message(kind):
        def fn(o, c, s):
            m = [x["strategies"][s][kind] for x in select(o, "slowdown") + select(o, "stop")]
            made = sum(x["created"] for x in m)
            d = [x["delay"] for x in m if x["delivered"]]
            return (sum(x["delivered"] for x in m) / made if made else float("nan")), (float(np.median(d)) if d else float("nan"))
        return fn

    add("consegna gruppo", lambda o, c, s: f"{delivery(o, c, s)[0]:.4f}")
    add("ritardo mediano / p95 (s)", lambda o, c, s: f"{_f(delivery(o, c, s)[1], '.1f')} / {_f(delivery(o, c, s)[2], '.1f')}")
    add("vista > 5 s (gruppo)", lambda o, c, s: f"{float(view_over(o, s, 'group')[0][0]):.4f}")
    add("vista > 10 s (gruppo)", lambda o, c, s: f"{float(view_over(o, s, 'group')[0][1]):.4f}")
    add("salti medi", lambda o, c, s: f"{hops_mean(o, s):.2f}")
    add("carry nel gruppo (%)", lambda o, c, s: f"{100 * pooled(o, s, 'group')['carried'] / max(pooled(o, s, 'group')['delivered'], 1):.2f}")
    add("consegna separato", lambda o, c, s: _f(delivery_stats(pooled(o, s, "sep"))[0], ".3f"))
    add("oracolo: consegna / ritardo s", lambda o, c, s: f"{_f(message('oracle')(o, c, s)[0], '.2f')} / {_f(message('oracle')(o, c, s)[1], '.1f')}")
    add("SOS: consegna / ritardo s", lambda o, c, s: f"{_f(message('sos')(o, c, s)[0], '.2f')} / {_f(message('sos')(o, c, s)[1], '.1f')}")
    add("rotture per corsa", lambda o, c, s: f"{np.mean([x['strategies'][s]['breaks'] for x in o]):.1f}")
    add("connettività verso il capogruppo", lambda o, c, s: f"{connectivity_summary(o, c['group']['n_nodes'])['nodes']:.3f}")
    add("tempo con il gruppo spezzato", lambda o, c, s: f"{connectivity_summary(o, c['group']['n_nodes'])['split']:.3f}")
    add("tempo d'aria max (%)", lambda o, c, s: f"{100 * max(x['strategies'][s]['airtime_per_s'].max() for x in o):.2f}")
    width = max(len(r[0]) for r in rows)
    head = f"{'':<{width}} " + " ".join(f"{p + ' ' + s:>26}" for p, s in cols)
    lines = [head] + [f"{label:<{width}} " + " ".join(f"{v:>26}" for v in vals) for label, vals in rows]
    for p in names:
        v = predictive_verdict(by_profile[p], configs[p])["all"]
        red = v["vs_best_rssi"]["break_reduction"]
        lines.append(f"[{p}] dqrem contro best_rssi: rotture {100 * red[0]:+.1f}% [{100 * red[1][0]:+.1f}, {100 * red[1][1]:+.1f}], ripaga: {'SI' if v['pays'] else 'NO'}")
    return "\n".join(lines)

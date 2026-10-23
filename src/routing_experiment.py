"""Block 7 — Routing-to-leader experiment: runs, summaries, aggregation, verdict.

Long group (overrides of ``routing.scenario_overrides``), scenarios ``none`` / ``slowdown`` / ``stop`` with ``routing.n_runs`` runs each. Run seeds and detachment node/instant
from the generators ``routing.seed_keys.runs`` / ``separation`` (shared by the three scenarios: paired comparison), group leader from the ``leader`` generator (never the node that detaches, same across the three
scenarios), SOS time fraction from the ``sos`` generator. The detector is NOT evaluated here (see `src/routing.py`). Each run is independent (`process_run_routing`): the result does not depend
on the number of processes. No disk cache: each process simulates its own run.

Parameters: ``routing`` section of the config.
"""

from __future__ import annotations

import copy
import time

import numpy as np

from src.channel import distance_from_path_loss
from src.common.enums import Scenario, Strategy
from src.common.parallel import parallel_map
from src.common.stats import percentile_interval, resample_indices
from src.common.rng import bootstrap_rng, spawn_rng
from src.metrics import SEPARATED, min_distance_to_group, nearest_rank
from src.refinement import apply_overrides
from src.routing import KIND_ALARM, KIND_SOS, prepare_run, second_stride, simulate_strategy
from src.simulator import draw_separations, experiment_runs

POSITIONS = ("coda", "centro", "testa")
SCENARIOS = tuple(s.value for s in Scenario)
VIEW_AGE_BINS = 60


# ---------------------------------------------------------------------------
# Sets of runs
# ---------------------------------------------------------------------------


def block7_config(config: dict, profile: str | None = None) -> dict:
    """Copy of the config with the Block 7 overrides: long group (``routing.scenario_overrides``) and channel profile.

    `profile` is ``nominal`` or ``heltec_run`` (default ``routing.channel_profile``); the profile overrides live in ``routing.channel_profiles``.
    ``nominal`` changes nothing with respect to the first Block 7 operating point. Both profiles are built on the channel of
    ``history.channel_v1`` (before the Heltec re-calibration of the defaults): the profile overrides never add up to the new channel.
    """
    rcfg = config["routing"]
    profile = rcfg["channel_profile"] if profile is None else profile
    if profile not in rcfg["channel_profiles"]:
        raise KeyError(f"unknown routing.channel_profile: {profile!r} (available: {', '.join(rcfg['channel_profiles'])})")
    cfg = apply_overrides(config, {**rcfg["scenario_overrides"], **config["history"]["channel_v1"]})
    return apply_overrides(cfg, rcfg["channel_profiles"][profile] or {})


def group_median_rssi(config: dict, duration: float = 300.0) -> float:
    """Median of the RSSI measured among the nodes of the compact 5-node group, scenario ``none``, base channel of Block 2.

    Used to calibrate ``channel.tx_power`` of the ``heltec_run`` profile: the power found is an EQUIVALENT power (EIRP including antenna losses,
    arm mounting and bodies), NOT a value that can be set on the ESP32. Same statistic as ``scripts/plot_channel.py`` (median −51 dBm at 10 dBm).
    `config`: base config with the channel overrides to try (the long group is NOT applied here). Output: dBm.
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
    """Group leader of each run: uniform among the nodes other than the one that detaches (generator ``SeedSequence(simulation.seed, spawn_key=(seed_keys.leader,))``).

    Input: `config` (Block 7), `sep_nodes` (n_runs,) nodes drawn for the detachment. Procedure: uniform index among ``N − 1`` positions, skipping the detachment node. Output (n_runs,) int.
    """
    rng = spawn_rng(config["simulation"]["seed"], config["routing"]["seed_keys"]["leader"])
    u = rng.integers(0, config["group"]["n_nodes"] - 1, size=len(sep_nodes))
    return (u + (u >= sep_nodes)).astype(int)


def draw_sos_fractions(config: dict, n_runs: int) -> np.ndarray:
    """SOS instant as a fraction of ``sos_window`` after the detachment: uniforms (0, 1) from the generator ``spawn_key=(seed_keys.sos,)``, one per run (same across the three scenarios)."""
    rng = spawn_rng(config["simulation"]["seed"], config["routing"]["seed_keys"]["sos"])
    return rng.random(n_runs)


def routing_runs(config: dict) -> list:
    """Block 7 runs (`experiment_runs` with keys 12/13 and ``routing.n_runs``), each with ``leader`` and ``sos`` (fraction). `config` is the Block 7 one (`block7_config`)."""
    rcfg = config["routing"]
    keys = {"runs": rcfg["seed_keys"]["runs"], "separation": rcfg["seed_keys"]["separation"]}
    runs = experiment_runs(config, keys, rcfg["n_runs"])
    nodes, _ = draw_separations(config, keys["separation"], rcfg["n_runs"])
    leaders, sos = draw_leaders(config, nodes), draw_sos_fractions(config, rcfg["n_runs"])
    for r in runs:
        r["leader"], r["sos"] = int(leaders[r["index"]]), float(sos[r["index"]])
    return runs


# ---------------------------------------------------------------------------
# Summary of a strategy on a run
# ---------------------------------------------------------------------------


def _keys(origin: np.ndarray, seq: np.ndarray) -> np.ndarray:
    return np.asarray(origin).astype(np.int64) * 10_000_000 + np.asarray(seq).astype(np.int64)


def _log_lookup(log: dict):
    lkey = _keys(log["origin"], log["seq"])
    sorter = np.argsort(lkey)

    def lookup(ck):
        pos = np.minimum(np.searchsorted(lkey[sorter], ck), max(len(lkey) - 1, 0))
        if len(lkey) == 0:
            return np.zeros(len(ck), dtype=bool), np.zeros(len(ck), dtype=int)
        idx = sorter[pos]
        return lkey[idx] == ck, idx

    return lookup


def _class_summaries(created: np.ndarray, log: dict, lookup, rd: dict, rcfg: dict, with_sep: bool, lo: float, hi: float) -> dict:
    L, sep, sep_start = rd["leader"], rd["sep_node"], rd["sep_start"]
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

    return {"group": class_stats(~is_sep), "sep": class_stats(is_sep)}


def _view_age(log: dict, rd: dict, rcfg: dict, with_sep: bool, lo: float, duration: float) -> dict:
    L, sep, sep_start = rd["leader"], rd["sep_node"], rd["sep_start"]
    n_nodes = rd["ann_grid"].shape[1]
    thresholds = rcfg["view_age_thresholds"]
    grid = np.arange(lo, duration, 1.0)
    age_edges = np.arange(0.0, VIEW_AGE_BINS + 1.0, 1.0)
    view = {c: {"hist": np.zeros(VIEW_AGE_BINS, dtype=np.int64), "n": 0, "over": np.zeros(len(thresholds), dtype=np.int64)} for c in ("group", "sep")}
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
            view[name]["hist"] += np.histogram(np.minimum(a, float(VIEW_AGE_BINS)), bins=age_edges)[0]
            view[name]["n"] += len(a)
            view[name]["over"] += np.array([(a > th).sum() for th in thresholds])
    return view


def _fresh_record(log: dict, rd: dict, rcfg: dict, with_sep: bool) -> dict:
    t, L, sep, sep_start, dt = rd["t"], rd["leader"], rd["sep_node"], rd["sep_start"], rd["dt"]
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
    return sep_out


def _message_outcome(created: np.ndarray, log: dict, lookup, kind: int) -> dict:
    rows = created[created[:, 3] == kind]
    if len(rows) == 0:
        return {"created": False, "delivered": False, "delay": np.nan}
    ok, i = lookup(_keys(rows[:, 0], rows[:, 1]))
    return {"created": True, "delivered": bool(ok[0]), "delay": float(log["t_arrive"][i[0]] - rows[0, 2]) if ok[0] else np.nan}


def _hops_median(ann_grid: np.ndarray) -> np.ndarray:
    n_nodes = ann_grid.shape[1]
    return np.array([np.median(ann_grid[:, n][ann_grid[:, n] < 255]) if (ann_grid[:, n] < 255).any() else np.nan for n in range(n_nodes)])


def summarize_strategy(out: dict, rd: dict, rcfg: dict, scenario: str) -> dict:
    """Reduces the output of `simulate_strategy` to a few figures per run.

    Input: simulation output, prepared run `rd`, `rcfg`, scenario.
    Procedure: normal records created between ``warmup`` and ``end − end_margin``, split into classes ``group`` and ``sep`` (records of the separated node after the detachment); for each: created,
    delivered, delays, hop histogram, records that arrived via carry (with delays), delivery by distance band from the leader. Age of the leader's view (how long since it received a
    NEW record from a node: ``t − max t_create`` of the arrived records) sampled every second from ``warmup``: histogram (60 bins of 1 s, the last collects the rest) and counts beyond the
    thresholds ``view_age_thresholds``. Separated: last fresh record (age < ``fresh_age``), with time since the detachment and distance from the leader and from the group. Oracle alarm and SOS: delivered and delay.
    Load: airtime and attempts per node (per second), maximum buffer occupancy, lost records, median announced hops.
    Output: dict of scalars and small arrays.
    """
    t = rd["t"]
    with_sep = scenario != Scenario.NONE
    duration = float(t[-1])
    lo, hi = rcfg["warmup"], duration - rcfg["end_margin"]
    created, log = out["created"], out["log"]
    lookup = _log_lookup(log)
    classes = _class_summaries(created, log, lookup, rd, rcfg, with_sep, lo, hi)
    view = _view_age(log, rd, rcfg, with_sep, lo, duration)
    sep_out = _fresh_record(log, rd, rcfg, with_sep)
    hops_med = _hops_median(rd["ann_grid"])
    return {
        "classes": classes, "view": view, "sep": sep_out, "oracle": _message_outcome(created, log, lookup, KIND_ALARM), "sos": _message_outcome(created, log, lookup, KIND_SOS),
        "breaks": out["breaks"], "break_episodes": out["break_episodes"], "fail_both": out["fail_both"], "bridge_changes": out["bridge_changes"],
        "ttl_drops": out["ttl_drops"], "duplicates": out["duplicates"], "cycles": out["cycles"], "capacity": out["capacity"], "max_packet_bytes_sent": out["max_packet_bytes_sent"],
        "airtime_per_s": out["tx_airtime"] / duration, "packets_per_s": out["attempts"] / duration, "max_occupancy": out["max_occupancy"],
        "lost_buffer": out["lost_buffer"], "hops_median": hops_med, "duration": duration,
    }  # fmt: skip


# ---------------------------------------------------------------------------
# Work per run
# ---------------------------------------------------------------------------


def process_run_routing(args: tuple) -> dict:
    """One Block 7 run: prepares it once (`prepare_run`) and evaluates all strategies on the same run (same estimates, same beacons, same unicast generator).

    Input: `args` = (run from `routing_runs`, strategies, packet sizes of the ESP-NOW v2 comparison (for `dqrem`), `trace`).
    Output: ``scenario``, ``index``, ``leader``, ``pos_class``, ``rank``, ``extent``, ``wrong`` (wrong hops), ``strategies`` (name → `summarize_strategy`), ``bytes`` (size → summary of `dqrem`),
    ``elapsed``, ``trace`` (only if requested: decisions of the separated node, records of the separated node, distance from the group).
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
        out = simulate_strategy(rd, name, rcfg, trace=trace and name == Strategy.DQREM)
        res["strategies"][name] = summarize_strategy(out, rd, rcfg, run["scenario"])
        res["times"][name] = time.perf_counter() - t1
        if "ahp_weights" in out:
            res["weights"][name] = (out["ahp_weights"], out["cr"])
        if trace and name == Strategy.DQREM:
            traced = out
    for size in byte_sweep:
        out = simulate_strategy(rd, Strategy.DQREM, rcfg, max_packet_bytes=size)
        res["bytes"][size] = summarize_strategy(out, rd, rcfg, run["scenario"])
    if trace and traced is not None and rd["sep_node"] >= 0:
        sep, L = rd["sep_node"], rd["leader"]
        log = traced["log"]
        sel = (log["origin"] == sep) & (log["kind"] == 0)
        stride = second_stride(rd["dt"])
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
    """`process_run_routing` on all runs (in parallel if ``jobs > 1``; result independent of `jobs`). `reference` = (scenario, index) of the run with the trace; `done` = results already available by (scenario, index)."""
    done = done or {}
    todo = [r for r in runs if (r["scenario"], r["index"]) not in done]
    tasks = [(r, strategies, byte_sweep, (r["scenario"], r["index"]) == reference) for r in todo]
    outs = parallel_map(process_run_routing, tasks, jobs)
    allout = list(done.values()) + outs
    return sorted(allout, key=lambda o: (o["scenario"], o["index"]))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def select(outs: list, scenario: str | None = None, pos: int | None = None) -> list:
    """Runs of `outs` for a scenario (None = all) and with the leader in a position (0 tail, 1 centre, 2 head; None = all)."""
    return [o for o in outs if (scenario is None or o["scenario"] == scenario) and (pos is None or o["pos_class"] == pos)]


def pooled(outs: list, strategy: str, cls: str) -> dict:
    """Sum over the runs of the statistics of a class (``group`` or ``sep``): created, delivered, concatenated delays, hops, carry, delivery by distance band."""
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
    """(delivered fraction, median delay s, p95 delay s) of an already summed class; NaN if there are no records."""
    frac = p["delivered"] / p["created"] if p["created"] else float("nan")
    return frac, nearest_rank(p["delays"], 0.5), nearest_rank(p["delays"], 0.95)


def view_over(outs: list, strategy: str, cls: str) -> tuple:
    """Fraction of time with view age beyond each threshold (``view_age_thresholds``) and samples, summed over the runs."""
    n = sum(o["strategies"][strategy]["view"][cls]["n"] for o in outs)
    over = sum(o["strategies"][strategy]["view"][cls]["over"] for o in outs) if outs else 0
    return (over / n if n else np.full(2, np.nan)), n


def per_run_values(outs: list, strategy: str) -> dict:
    """Per-run values of a strategy (for the paired comparison): ``breaks`` (attempts failed after all retransmissions to the first candidate), ``delivery`` (delivered fraction, nodes in the group),
    ``age5`` (fraction of time with view age > first threshold, nodes in the group)."""
    breaks = np.array([o["strategies"][strategy]["breaks"] for o in outs], dtype=float)
    delivery = np.array([o["strategies"][strategy]["classes"]["group"]["delivered"] / max(o["strategies"][strategy]["classes"]["group"]["created"], 1) for o in outs])
    age5 = np.array([o["strategies"][strategy]["view"]["group"]["over"][0] / max(o["strategies"][strategy]["view"]["group"]["n"], 1) for o in outs])
    return {"breaks": breaks, "delivery": delivery, "age5": age5}


def paired_bootstrap(a: dict, b: dict, config: dict) -> dict:
    """Paired comparison of `a` (e.g. ``dqrem``) against `b` (e.g. ``best_rssi``) with per-run bootstrap (generator ``final.seed_keys.bootstrap`` = key 11, ``final.bootstrap_samples`` resamples).

    Input: per-run values (`per_run_values`) of the two strategies on the same runs. Procedure: the RUNS are resampled with replacement, the same resampling for both.
    Output: ``break_reduction`` = ``1 − Σ breaks(a) / Σ breaks(b)`` (value and percentile interval at level ``metrics.confidence_level``), ``delivery_diff`` and ``age5_diff`` = means of the differences (a − b).
    """
    rng = bootstrap_rng(config)
    n = len(a["breaks"])
    idx = resample_indices(rng, n, config["final"]["bootstrap_samples"])
    level = config["metrics"]["confidence_level"]

    def interval(values):
        return percentile_interval(values, level)

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
    """Does the predictive part pay off? ``dqrem`` against ``best_rssi`` and against ``dqrem_no_slope`` (`paired_bootstrap`) over all runs and per scenario.

    Rule fixed before the results: DQREM "pays off" with respect to ``best_rssi`` if it reduces breaks by at least ``min_break_reduction`` and does NOT worsen delivery, i.e. the upper end
    of the interval of the delivery difference is ≥ 0 (the difference is not significantly negative).
    Output: ``{"all": {...}, scenario: {...}}``, each with ``vs_best_rssi``, ``vs_no_slope`` and ``pays`` (bool; for all runs and per scenario).
    """
    need = config["routing"]["min_break_reduction"]
    out = {}
    for key, subset in (("all", outs), *((s, select(outs, s)) for s in SCENARIOS)):
        if not subset:
            continue
        dq = per_run_values(subset, Strategy.DQREM.value)
        res = {name: paired_bootstrap(dq, per_run_values(subset, other), config) for name, other in (("vs_best_rssi", Strategy.BEST_RSSI.value), ("vs_no_slope", Strategy.DQREM_NO_SLOPE.value))}
        v = res["vs_best_rssi"]
        reduction = v["break_reduction"][0]
        res["pays"] = bool(np.isfinite(reduction) and reduction >= need and v["delivery_diff"][1][1] >= 0.0)
        out[key] = res
    return out


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def _f(x, spec):
    return "   -" if x is None or (isinstance(x, float) and np.isnan(x)) else format(x, spec)


def calibration_text(outs: list, config: dict) -> str:
    """Calibration of the long group: sigma, observed median/p95 extent (longitudinal and lateral, mean over the ``none`` runs)."""
    none = select(outs, "none") or outs
    e = [o["extent"] for o in none]
    mean = lambda k: float(np.mean([x[k] for x in e]))  # noqa: E731
    g = config["group"]
    return (
        f"Long group: {g['n_nodes']} nodes, requested longitudinal extent p95 {g['longitudinal_spread']:g} m, lateral {g['lateral_spread']:g} m\n"
        f"  sigma_long = {mean('sigma_long'):.3f} m, sigma_lat = {mean('sigma_lat'):.3f} m\n"
        f"  observed extent (mean over the {len(none)} runs): longitudinal median {mean('long_median'):.1f} m, p95 {mean('long_p95'):.1f} m;"
        f" lateral median {mean('lat_median'):.1f} m, p95 {mean('lat_p95'):.1f} m"
    )


def delivery_table(outs: list, config: dict, pos: int | None = None) -> str:
    """Table per strategy and scenario: delivery and delay (nodes in the group and separated), view age, breaks, bridge changes, TTL drops (per run)."""
    thr = config["routing"]["view_age_thresholds"]
    lines = [
        f"{'strategy':<15} {'scen.':<9} {'gr.deliv':>7} {'med s':>6} {'p95 s':>6} {'sep.deliv':>8} {'med s':>6} {'p95 s':>6}"
        f" {'>' + format(thr[0], 'g') + 's gr':>7} {'>' + format(thr[1], 'g') + 's gr':>7} {'>' + format(thr[0], 'g') + 's sep':>8} {'breaks/r':>9} {'ep./r':>6} {'bridges/r':>7} {'TTL/r':>5}"
    ]
    for scen in SCENARIOS:
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
    """Separated node: last fresh record (time since the detachment and distance), carry (records and delay), oracle alarm and SOS (delivered and delay), per strategy and scenario."""
    lines = [
        f"{'strategy':<15} {'scen.':<9} {'fresh t s':>10} {'dist.ld m':>10} {'dist.gr m':>9} {'carry rec':>9} {'carry med.dly s':>15} {'oracle':>9} {'dly s':>6} {'SOS':>9} {'dly s':>6}"
    ]
    for scen in SEPARATED:
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
    """Load: airtime (fraction of time) and packets/s per node, maximum over the nodes and mean, per strategy; maximum buffer occupancy and records lost to a full buffer."""
    lines = [f"{'strategy':<15} {'air max %':>10} {'air mean %':>12} {'pkt/s max':>9} {'pkt/s mean':>11} {'buffer max':>10} {'lost/run':>11} {'dupl./r':>11}"]
    for name in config["routing"]["strategies"]:
        air = np.array([o["strategies"][name]["airtime_per_s"] for o in outs])
        pkt = np.array([o["strategies"][name]["packets_per_s"] for o in outs])
        occ = np.array([o["strategies"][name]["max_occupancy"] for o in outs])
        lost = np.array([o["strategies"][name]["lost_buffer"].sum() for o in outs])
        dup = np.mean([o["strategies"][name]["duplicates"] for o in outs])
        lines.append(f"{name:<15} {100 * air.max():>10.2f} {100 * air.mean():>12.2f} {pkt.max():>9.2f} {pkt.mean():>11.2f} {occ.max():>10d} {lost.mean():>11.1f} {dup:>11.1f}")
    return "\n".join(lines)


def hops_table(outs: list, config: dict) -> str:
    """Distribution of the hops of the delivered records (nodes in the group), per strategy, in percent."""
    k = config["routing"]["max_hops"] + 1
    lines = [f"{'strategia':<15} " + " ".join(f"{h:>6d}" for h in range(k)) + "  mean hops"]
    for name in config["routing"]["strategies"]:
        h = pooled(outs, name, "group")["hops"]
        h = np.asarray(h, dtype=float)
        tot = h.sum()
        lines.append(f"{name:<15} " + " ".join(f"{100 * x / tot:>5.1f}%" for x in h) + f"  {float((h * np.arange(k)).sum() / tot):>9.2f}")
    return "\n".join(lines)


def wrong_hops_text(outs: list) -> str:
    """Wrong announced hops: nodes announcing a finite value without a real path (episodes of 1 s samples): median/p95/maximum duration and maximum announced value."""
    d = np.concatenate([o["wrong"]["durations"] for o in outs])
    peak = np.concatenate([o["wrong"]["max_announced"] for o in outs])
    frac = float(np.mean([o["wrong"]["wrong_fraction"] for o in outs]))
    if len(d) == 0:
        return "Wrong hops: no episodes."
    return (
        f"Wrong hops after losing the path: {len(d)} episodes in {len(outs)} runs ({frac * 100:.2f}% of node-time); median duration {np.median(d):.0f} s, p95 {nearest_rank(d, 0.95):.0f} s,"
        f" maximum {d.max():.0f} s; median peak announced value {np.median(peak):.0f} (max {peak.max()})"
    )


def verdict_text(verdict: dict, config: dict) -> str:
    """Verdict text: for each set of runs, break reduction, delivery and age difference with intervals, and whether the predictive part pays off."""
    need = config["routing"]["min_break_reduction"]
    lines = [f"Rule: DQREM pays off if it reduces breaks by at least {100 * need:.0f}% with respect to best_rssi AND the interval of the delivery difference is not entirely below zero."]
    ci = lambda x: f"{x[0]:+.3f} [{x[1][0]:+.3f}, {x[1][1]:+.3f}]"  # noqa: E731
    for key, res in verdict.items():
        for name, label in (("vs_best_rssi", "dqrem vs best_rssi"), ("vs_no_slope", "dqrem vs dqrem_no_slope")):
            v = res[name]
            red = v["break_reduction"]
            lines.append(
                f"  [{key:<8}] {label:<30} breaks {100 * red[0]:+6.1f}% [{100 * red[1][0]:+6.1f}, {100 * red[1][1]:+6.1f}]  delivery {ci(v['delivery_diff'])}  view>5s {ci(v['age5_diff'])}  (n={v['n_runs']})"
            )
        lines.append(f"  [{key:<8}] the predictive part PAYS OFF with respect to best_rssi: {'YES' if res['pays'] else 'NO'}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Range, connectivity, group carry, comparison between profiles
# ---------------------------------------------------------------------------


def distance_range(level_dbm: float, config: dict) -> float:
    """Distance (m) at which path loss alone (``channel.tx_power`` + gains − ``PL0 + 10 n log10(d/d0)``) gives `level_dbm`; analytical inversion of `path_loss_db`."""
    return distance_from_path_loss(level_dbm, config["channel"])


def median_curve(outs: list) -> tuple:
    """Median of the signal (path loss and bodies, without shadowing, fading and board offsets) per distance, from the histogram summed over the runs. Output: (distance centres, medians dBm, samples)."""
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
    """Estimated range: distance at which the median RSSI drops to −88 and −96 dBm, with distance only (analytical) and with the bodies (median curve, runs of all scenarios)."""
    dist, med, n = median_curve(outs)
    ok = (n >= 200) & np.isfinite(med)
    lines = [f"Estimated range (equivalent power {config['channel']['tx_power']:g} dBm; medians without shadowing and fading):"]
    for level in (-88.0, -96.0):
        below = np.flatnonzero(ok & (med <= level))
        with_body = f"{dist[below[0]]:.1f} m" if len(below) else f"beyond {dist[ok][-1]:.0f} m"
        lines.append(f"  median RSSI {level:g} dBm: distance only {distance_range(level, config):.1f} m, with bodies {with_body}")
    return "\n".join(lines)


def connectivity_by_rank(outs: list, n_nodes: int) -> tuple:
    """Fraction of time with a path to the leader per longitudinal rank of the node (0 = tail), leader excluded. Output: (fractions (N,), samples (N,))."""
    num, den = np.zeros(n_nodes), np.zeros(n_nodes)
    for o in outs:
        c, rank = o["connectivity"], o["node_rank"]
        for node in range(n_nodes):
            if node != o["leader"]:
                num[rank[node]] += c["in"][node]
                den[rank[node]] += c["n"][node]
    return np.where(den > 0, num / np.maximum(den, 1), np.nan), den


def connectivity_summary(outs: list, n_nodes: int) -> dict:
    """Connectivity of the set of runs: mean fraction (nodes in the group, leader excluded), fraction of time with the group split, fraction of the separated node after the detachment."""
    frac, den = connectivity_by_rank(outs, n_nodes)
    num = sum(float(f * d) for f, d in zip(np.nan_to_num(frac), den))
    split = sum(o["connectivity"]["split"] for o in outs) / max(sum(o["connectivity"]["split_n"] for o in outs), 1)
    sep_n = sum(o["connectivity"]["sep_n"] for o in outs)
    sep = sum(o["connectivity"]["sep_in"] for o in outs) / sep_n if sep_n else float("nan")
    return {"nodes": num / max(den.sum(), 1), "split": split, "sep": sep}


def connectivity_table(outs: list, config: dict) -> str:
    """Connectivity to the leader per scenario: by node rank (0 tail … N−1 head), mean, time with the group split, separated node after the detachment."""
    n_nodes = config["group"]["n_nodes"]
    lines = [f"{'scen.':<9} " + " ".join(f"{'r' + str(r):>6}" for r in range(n_nodes)) + f" {'mean':>7} {'split':>9} {'separated':>9}"]
    for scen in SCENARIOS:
        sub = select(outs, scen)
        if not sub:
            continue
        frac, _ = connectivity_by_rank(sub, n_nodes)
        s = connectivity_summary(sub, n_nodes)
        lines.append(f"{scen:<9} " + " ".join(f"{_f(float(x), '6.3f'):>6}" for x in frac) + f" {s['nodes']:>7.3f} {s['split']:>9.3f} {_f(s['sep'], '9.3f'):>9}")
    return "\n".join(lines)


def group_carry_text(outs: list, config: dict) -> str:
    """Telemetry of the nodes in the group that goes through carry (records held in the queue because there is no candidate) even without detachment: fraction of the delivered and median/p95 delay, per strategy."""
    lines = [f"{'strategy':<15} {'carry %':>8} {'carry med.dly s':>15} {'carry p95 dly s':>15}"]
    for name in config["routing"]["strategies"]:
        p = pooled(outs, name, "group")
        lines.append(f"{name:<15} {100 * p['carried'] / max(p['delivered'], 1):>8.2f} {_f(nearest_rank(p['carry_delays'], 0.5), '15.1f'):>15} {_f(nearest_rank(p['carry_delays'], 0.95), '15.1f'):>15}")
    return "\n".join(lines)


def hops_mean(outs: list, name: str) -> float:
    h = np.asarray(pooled(outs, name, "group")["hops"], dtype=float)
    return float((h * np.arange(len(h))).sum() / max(h.sum(), 1))


def profile_comparison_text(by_profile: dict, configs: dict) -> str:
    """Side-by-side comparison of channel profiles (`by_profile`: name → runs; `configs`: name → config) for ``best_rssi`` and ``dqrem``, over all runs: group delivery and delay, view age,
    hops, carry, separated node delivery, oracle alarm and SOS, breaks, break reduction (and verdict), connectivity, split time, airtime."""
    names = list(by_profile)
    cols = [(p, s) for p in names for s in (Strategy.BEST_RSSI.value, Strategy.DQREM.value)]
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

    add("group delivery", lambda o, c, s: f"{delivery(o, c, s)[0]:.4f}")
    add("median / p95 delay (s)", lambda o, c, s: f"{_f(delivery(o, c, s)[1], '.1f')} / {_f(delivery(o, c, s)[2], '.1f')}")
    add("view > 5 s (group)", lambda o, c, s: f"{float(view_over(o, s, 'group')[0][0]):.4f}")
    add("view > 10 s (group)", lambda o, c, s: f"{float(view_over(o, s, 'group')[0][1]):.4f}")
    add("mean hops", lambda o, c, s: f"{hops_mean(o, s):.2f}")
    add("carry in group (%)", lambda o, c, s: f"{100 * pooled(o, s, 'group')['carried'] / max(pooled(o, s, 'group')['delivered'], 1):.2f}")
    add("separated delivery", lambda o, c, s: _f(delivery_stats(pooled(o, s, "sep"))[0], ".3f"))
    add("oracle: delivery / delay s", lambda o, c, s: f"{_f(message('oracle')(o, c, s)[0], '.2f')} / {_f(message('oracle')(o, c, s)[1], '.1f')}")
    add("SOS: delivery / delay s", lambda o, c, s: f"{_f(message('sos')(o, c, s)[0], '.2f')} / {_f(message('sos')(o, c, s)[1], '.1f')}")
    add("breaks per run", lambda o, c, s: f"{np.mean([x['strategies'][s]['breaks'] for x in o]):.1f}")
    add("connectivity to the leader", lambda o, c, s: f"{connectivity_summary(o, c['group']['n_nodes'])['nodes']:.3f}")
    add("time with the group split", lambda o, c, s: f"{connectivity_summary(o, c['group']['n_nodes'])['split']:.3f}")
    add("max airtime (%)", lambda o, c, s: f"{100 * max(x['strategies'][s]['airtime_per_s'].max() for x in o):.2f}")
    width = max(len(r[0]) for r in rows)
    head = f"{'':<{width}} " + " ".join(f"{p + ' ' + s:>26}" for p, s in cols)
    lines = [head] + [f"{label:<{width}} " + " ".join(f"{v:>26}" for v in vals) for label, vals in rows]
    for p in names:
        v = predictive_verdict(by_profile[p], configs[p])["all"]
        red = v["vs_best_rssi"]["break_reduction"]
        lines.append(f"[{p}] dqrem vs best_rssi: breaks {100 * red[0]:+.1f}% [{100 * red[1][0]:+.1f}, {100 * red[1][1]:+.1f}], pays off: {'YES' if v['pays'] else 'NO'}")
    return "\n".join(lines)

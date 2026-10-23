"""Block 5 experiment: detector variants A-F, compared with the base on the same runs.

A VARIANT is a set of config overrides (dotted keys, e.g. ``fusion.rule``). For each
run (cached, `src/simulator.py`) all the variants of a phase are evaluated: Kalman → fusion →
detector over the whole threshold grid, ``both`` variant (pre-alarm + lost). Then, for each
variant: sweep → working point (Block 4 rule: maximum P_d with feasible false alarms/h and time in
alarm) → PAIRED comparison with the base (runs gained and lost).

Selection rule (`is_better`): a variant beats the current best if it gains at least
``refinement.min_net_gain_runs`` net runs (slowdown + stop); if equivalent in P_d, if it reduces by
as many the runs with wrong turn-offs. Otherwise the simplest is kept (the order of the lists).

Parameters: ``refinement``, ``metrics``, ``detector``, ``fusion``, ``kalman`` sections of the config.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from src.common.arrays import nearest_index, run_bounds
from src.common.config import apply_overrides  # noqa: F401
from src.common.parallel import parallel_map
from src.detector import build_inputs
from src.kalman import filter_all
from src.common.enums import Scenario
from src.metrics import (
    aggregate,
    SEPARATED,
    evaluate_config,
    gain_loss_counts,
    run_summarizer,
    split_by_scenario,
    observer_correlation,
    threshold_grid,
    working_point,
)
from src.relative import relative_trace
from src.simulator import load_or_simulate

PER_RUN_FIELDS = (
    "success", "delay", "distance", "fortuitous", "n_wrong_off", "wrong_off_dist_sum", "wrong_off_dist_min",
    "repass", "repass_off", "alarm_on_end",
)  # fmt: skip


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------


def block4_config(config: dict) -> dict:
    """Copy of the config with the Block 4 values (`history.channel_v1` then `history.block4`): the defaults are those of the final detector
    and of the Heltec channel, the Block 4-5 scripts restore the old ones to reproduce the numbers of that time."""
    return apply_overrides(config, {**config["history"]["channel_v1"], **config["history"]["block4"]})


def make_variant(name: str, group: str, overrides: dict, scopes=("fused",), min_observers=(1,), corr=False) -> dict:
    return {
        "name": name,
        "group": group,
        "overrides": dict(overrides),
        "scopes": list(scopes),
        "min_observers": list(min_observers),
        "corr": corr,
    }


def mode_label(overrides: dict, config: dict) -> str:
    """``level`` (absolute) or ``drop`` (relative): it is the ``mode`` field of the `aggregate` rows."""
    ref = overrides.get("detector.reference", config["detector"]["reference"])
    return "drop" if ref == "relative" else "level"


def phase1_variants(config: dict) -> list:
    """Variants A-D (one change at a time, all relative to the base). The base is ``base`` (``weighted_mean`` rule, absolute, 5 s, no robustness)."""
    base = config["refinement"]["base"]
    sigma = {"kalman.sigma_a": base["sigma_a"]}
    mos = config["detector"]["min_observers_sweep"]
    rules = config["fusion"]["rule_sweep"]
    variants = [make_variant("base", "base", sigma, ("fused", "pairwise"), mos, corr=True)]
    for rule in rules:
        if rule != "weighted_mean":
            variants.append(make_variant(f"A:{rule}", "A", {**sigma, "fusion.rule": rule}, ("fused", "pairwise"), mos, corr=True))
    variants.append(make_variant("B:relative", "B", {**sigma, "detector.reference": "relative"}, ("fused", "pairwise"), mos, corr=True))
    for rt in config["detector"]["release_time_sweep"]:
        if rt != config["detector"]["release_time"]:
            variants.append(make_variant(f"C:release={rt:g}", "C", {**sigma, "detector.release_time": rt}))
    robust = [(f"inflate={r:g}", {"kalman.robust": "inflate", "kalman.r_inflation": r}) for r in config["kalman"]["r_inflation_sweep"]]
    robust.append(("gate", {"kalman.robust": "nis_gate"}))
    for label, extra in robust:
        variants.append(make_variant(f"D:{label}", "D", {**sigma, **extra}))
        variants.append(make_variant(f"D:{label}+rel", "D", {**sigma, **extra, "detector.reference": "relative"}))
    return variants


# ---------------------------------------------------------------------------
# Per-run work
# ---------------------------------------------------------------------------


def _kalman_key(vcfg: dict) -> tuple:
    k = vcfg["kalman"]
    return (k["sigma_a"], k["robust"], k["r_inflation"], k["nis_gate"])


def process_run(args: tuple) -> dict:
    """Evaluates all the `variants` on a run; `args` = (run, config, cache directory, variants, ``R``).

    Returns ``scenario``, ``index``, ``results`` (``(name, scope)`` → `evaluate_config`), ``elapsed`` (s per variant) and, for
    the ``none`` runs, ``nis`` (Kalman key → finite NIS, float32).
    """
    run_info, config, cache_dir, variants, R = args
    run = load_or_simulate(run_info["config"], cache_dir)
    is_none = run_info["scenario"] == Scenario.NONE
    out = {"scenario": run_info["scenario"], "index": run_info["index"], "results": {}, "elapsed": {}, "nis": {}}
    warmup = config["detector"]["warmup"]
    summarize, node = run_summarizer(run, config, is_none, extended=True)
    kcache = {}
    for v in variants:
        t0 = time.perf_counter()
        vcfg = apply_overrides(config, v["overrides"])
        key = _kalman_key(vcfg)
        if key not in kcache:
            kcache[key] = filter_all(run["beacon_times"], run["received"], run["rssi"], R, key[0], vcfg["kalman"])
            if is_none:
                nis = kcache[key].nis
                out["nis"][key] = nis[np.isfinite(nis)].astype(np.float32)
        inputs, _ = build_inputs(run, vcfg, key[0], R, kcache[key])
        if not is_none:
            inputs = inputs.target_slice(node)
        for scope in v["scopes"]:
            out["results"][(v["name"], scope)] = evaluate_config(
                inputs, vcfg, scope, summarize, v["min_observers"], warmup if (is_none and v["corr"]) else None
            )
        out["elapsed"][v["name"]] = time.perf_counter() - t0
    return out


def run_phase(config: dict, runs: list, cache_dir: Path, variants: list, R: float, jobs: int) -> list:
    """`process_run` on all the runs (in parallel if ``jobs > 1``; the result does not depend on `jobs`)."""
    tasks = [(r, config, cache_dir, variants, R) for r in runs]
    return parallel_map(process_run, tasks, jobs)


# ---------------------------------------------------------------------------
# Aggregation and working point
# ---------------------------------------------------------------------------


def summarize_variant(outs: list, variant: dict, scope: str, mo: int, config: dict) -> dict:
    """Sweep and working point of a variant (`aggregate` + `working_point`), with the per-run data at the working point.

    Returns ``rows`` (all thresholds), ``wp`` (None if no point is feasible), ``point`` (scenario → row), ``per_run``
    (scenario → field → array over runs), ``corr`` (correlation between observers at the working point), ``elapsed`` (total s).
    """
    name = variant["name"]
    mode = mode_label(variant["overrides"], config)
    key = (scope, mode, "both")
    none, sep = split_by_scenario(outs)
    pick = lambda o: {key: o["results"][(name, scope)][mo]}  # noqa: E731
    rows = aggregate([pick(o) for o in none], {s: [pick(o) for o in v] for s, v in sep.items()}, config)
    wp = working_point(rows, scope, mode, "both")
    out = {
        "name": name, "group": variant["group"], "scope": scope, "min_observers": mo, "mode": mode, "rows": rows, "wp": wp,
        "point": {}, "per_run": {}, "corr": None, "elapsed": sum(o["elapsed"][name] for o in outs),
    }  # fmt: skip
    if wp is None:
        return out
    grid = rows and sorted({r["threshold"] for r in rows})
    n = nearest_index(grid, wp["threshold"])
    for scenario, runs in sep.items():
        out["point"][scenario] = next(r for r in rows if r["scenario"] == scenario and r["threshold"] == wp["threshold"])
        out["per_run"][scenario] = {
            f: np.array([o["results"][(name, scope)][mo][f][n] for o in runs]) for f in PER_RUN_FIELDS
        }
    stats = [o["results"][(name, scope)].get("corr") for o in none]
    if stats and stats[0] is not None:
        pooled = {k: sum(s[k] for s in stats) for k in stats[0]}
        out["corr"] = observer_correlation(pooled, n)
    out["none_point"] = {
        k: out["point"]["slowdown"][k] for k in ("fa_per_hour", "fa_lo", "fa_hi", "active_fraction")
    }
    return out


def gain_loss(variant: dict, base: dict) -> dict:
    """Runs gained and lost with respect to `base`, per scenario and in total; empty if either has no working point."""
    if variant["wp"] is None or base["wp"] is None:
        return {}
    return gain_loss_counts({s: variant["per_run"][s]["success"] for s in SEPARATED}, {s: base["per_run"][s]["success"] for s in SEPARATED})


def wrong_off_runs(summary: dict) -> int:
    """Runs (slowdown + stop) with at least one wrong turn-off."""
    return int(sum((summary["per_run"][s]["n_wrong_off"] > 0).sum() for s in SEPARATED))


def is_better(cand: dict, best: dict, min_net: int) -> bool:
    """Does `cand` beat `best`? Paired P_d: ≥ `min_net` net runs; if equivalent, ≥ `min_net` fewer runs with wrong turn-offs."""
    if cand["wp"] is None:
        return False
    if best["wp"] is None:
        return True
    gained, lost = gain_loss(cand, best)["total"]
    if gained - lost >= min_net:
        return True
    if gained - lost <= -min_net:
        return False
    return wrong_off_runs(best) - wrong_off_runs(cand) >= min_net


def choose(candidates: list, base: dict, min_net: int) -> dict:
    """Scans the candidates in order of simplicity: each must beat the current best (starting from `base`) beyond the noise."""
    best = base
    for cand in candidates:
        if is_better(cand, best, min_net):
            best = cand
    return best


# ---------------------------------------------------------------------------
# Combination (F)
# ---------------------------------------------------------------------------


def components(winners: dict) -> dict:
    """Winning components: name → config overrides (and `min_observers` in ``"_mo"``). Only those different from the base."""
    return {k: v for k, v in winners.items() if v is not None}


def combination_variant(name: str, parts: dict, config: dict, scopes, corr=False) -> dict:
    """Variant that merges the components `parts` (A, B, C, D). Robustness D uses the overrides of its scope (absolute/relative)."""
    overrides = {"kalman.sigma_a": config["refinement"]["base"]["sigma_a"]}
    mo = 1
    relative = "B" in parts
    for key, part in parts.items():
        if key == "D":
            overrides.update(part["relative" if relative else "absolute"] or {})
        elif key == "A":
            overrides.update({k: v for k, v in part.items() if k != "_mo"})
            mo = part.get("_mo", 1)
        else:
            overrides.update(part)
    return make_variant(name, "F", overrides, scopes, (mo,), corr)


# ---------------------------------------------------------------------------
# Baseline diagnostics (none runs)
# ---------------------------------------------------------------------------

EDGE_TOLERANCE = 1e-9


def _max_run(flag: np.ndarray, dt: float) -> float:
    """Maximum duration (s) of a consecutive sequence of True."""
    starts, stops = run_bounds(flag)
    return int((stops - starts).max(initial=0)) * dt


def baseline_diagnostics(args: tuple) -> dict:
    """Relative-level diagnostics on a ``none`` run at one threshold: fused-drop histogram, freezes.

    `args` = (run, config, cache directory, variant, threshold, ``R``, scope). For each (observer, target) pair after the warmup:
    ``hist`` (fused drop), ``frozen_time``/``time`` (instants with at least one frozen base / total), ``max_freeze`` (s, the longest
    freeze of a single link), ``state_time`` (instants in pre-alarm).
    """
    run_info, config, cache_dir, variant, threshold, R, scope = args
    run = load_or_simulate(run_info["config"], cache_dir)
    vcfg = apply_overrides(config, variant["overrides"])
    inputs, _ = build_inputs(run, vcfg, vcfg["kalman"]["sigma_a"], R)
    t = run["t"]
    first = int(np.searchsorted(t, vcfg["detector"]["warmup"]))
    dt = float(t[1] - t[0])
    lo, hi, step = config["refinement"]["drop_histogram"]
    edges = np.arange(lo, hi + step / 2, step)
    hist = np.zeros(len(edges) - 1)
    out = {"hist": hist, "edges": edges, "frozen_time": 0, "time": 0, "max_freeze": 0.0, "state_time": 0, "over_hi": 0, "pairs": []}
    n_nodes = run["received"].shape[0]
    for i in range(n_nodes):
        for m in range(n_nodes):
            if i == m:
                continue
            tr = relative_trace(inputs, vcfg, scope, threshold, m, i)
            drop = tr["drop"][first:]
            finite = drop[np.isfinite(drop)]
            hist += np.histogram(np.clip(finite, edges[0], edges[-1] - EDGE_TOLERANCE), bins=edges)[0]
            out["over_hi"] += int((finite >= edges[-1]).sum())
            frozen = tr["frozen"][first:]
            any_frozen = frozen.any(axis=1)
            out["frozen_time"] += int(any_frozen.sum())
            out["time"] += len(any_frozen)
            out["state_time"] += int(tr["state"][first:].sum())
            longest = max(_max_run(frozen[:, k], dt) for k in range(frozen.shape[1]))
            out["max_freeze"] = max(out["max_freeze"], longest)
            out["pairs"].append((m, i, longest, float(any_frozen.mean())))
    return out


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def best_threshold_index(summary: dict) -> int | None:
    if summary["wp"] is None:
        return None
    grid = sorted({r["threshold"] for r in summary["rows"]})
    return nearest_index(grid, summary["wp"]["threshold"])


def threshold_axis(config: dict, mode: str) -> np.ndarray:
    return threshold_grid(config["metrics"]["drop_threshold_sweep" if mode == "drop" else "level_threshold_sweep"])

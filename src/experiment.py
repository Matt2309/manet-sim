"""Block 4 experiment: per-run work and sweep.

For each run (scenario, index) and each ``σ_a``: Kalman → tables →
fusion → detector over the whole threshold grid, with Blocks 1-3 read
from the cache (`src/simulator.py`). Runs are independent: they execute in
parallel, and the result does not depend on the number of processes.

Parameters: ``metrics`` and ``detector`` sections of the config.
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import numpy as np

from src.common.parallel import parallel_map
from src.detector import build_inputs
from src.common.enums import Scenario
from src.metrics import (
    MODES,
    SCOPES,
    aggregate,
    best_over_sigma,
    evaluate_run,
    run_summarizer,
    split_by_scenario,
)
from src.simulator import cache_path, load_or_simulate



def nis_bin_edges(config: dict) -> np.ndarray:
    """Edges of the NIS histogram from ``metrics.nis_histogram`` = [start, stop, step] (for the plot only)."""
    start, stop, step = config["metrics"]["nis_histogram"]
    return np.linspace(start, stop, int(round((stop - start) / step)) + 1)


def pooled_fading_variance(runs: list, cache_dir: Path) -> float:
    """Fading variance (dB²) in the group: mean over the ``none`` runs of the off-diagonal variances."""
    values = []
    for run in runs:
        if run["scenario"] == Scenario.NONE:
            with np.load(cache_path(run["config"], cache_dir)) as f:
                values.append(float(f["fading_var"]))
    return float(np.mean(values))


def process_run(args: tuple) -> dict:
    """Block 4 on one run for one ``σ_a``; `args` = (run, config, cache directory, ``σ_a``, ``R``).

    ``none``: `evaluate_run` on all targets and NIS (sum, count, histogram, ``nis_over``);
    with separation: cropped to the separated node. Returns ``scenario``, ``index``, ``results``, ``elapsed``.
    """
    run_info, config, cache_dir, sigma_a, R = args
    t0 = time.perf_counter()
    run = load_or_simulate(run_info["config"], cache_dir)
    inputs, kres = build_inputs(run, config, sigma_a, R)
    out = {"scenario": run_info["scenario"], "index": run_info["index"]}
    is_none = run_info["scenario"] == Scenario.NONE
    summarize, node = run_summarizer(run, config, is_none)
    if is_none:
        out["results"] = evaluate_run(inputs, config, summarize)
        nis = kres.nis[np.isfinite(kres.nis)]
        out["nis_sum"] = float(nis.sum())
        out["nis_count"] = int(len(nis))
        edges = nis_bin_edges(config)
        out["nis_hist"] = np.histogram(nis, bins=edges)[0]
        out["nis_over"] = int((nis >= edges[-1]).sum())
    else:
        out["results"] = evaluate_run(inputs.target_slice(node), config, summarize)
    out["elapsed"] = time.perf_counter() - t0
    return out


def run_sweep(config: dict, runs: list, cache_dir: Path, sigma_values: list, R: float, jobs: int) -> tuple:
    """Sweep over ``σ_a``, mode, scope and threshold (`process_run` on all runs, then `aggregate`).

    Returns (rows with ``sigma_a``, details per ``σ_a``: per-run results, NIS, timings).
    """
    rows, details = [], {}
    for sigma_a in sigma_values:
        tasks = [(r, config, cache_dir, sigma_a, R) for r in runs]
        outs = parallel_map(process_run, tasks, jobs)
        none, separated = split_by_scenario(outs, [s for s in config["metrics"]["scenarios"] if s != Scenario.NONE])
        sigma_rows = aggregate(
            [o["results"] for o in none], {s: [o["results"] for o in v] for s, v in separated.items()}, config
        )
        for row in sigma_rows:
            row["sigma_a"] = sigma_a
        rows.extend(sigma_rows)
        details[sigma_a] = {
            "none": none,
            "separated": separated,
            "nis_mean": sum(o["nis_sum"] for o in none) / sum(o["nis_count"] for o in none),
            "nis_hist": sum(o["nis_hist"] for o in none),
            "nis_over": sum(o["nis_over"] for o in none),
            "nis_count": sum(o["nis_count"] for o in none),
            "elapsed": sum(o["elapsed"] for o in outs),
        }
    return rows, details


# ---------------------------------------------------------------------------
# Saving, loading and reporting
# ---------------------------------------------------------------------------

ROW_FIELDS = (
    "sigma_a", "scope", "mode", "variant", "threshold", "scenario", "p_detect", "p_lo", "p_hi",
    "p_detect_window", "n_runs", "fortuitous", "delay_median", "delay_p90", "distance_median",
    "distance_p90", "fa_per_hour", "fa_lo", "fa_hi", "active_fraction", "feasible",
)  # fmt: skip
_TEXT_FIELDS = ("scope", "mode", "variant", "scenario")
_INT_FIELDS = ("n_runs", "fortuitous", "feasible")


def save_results(out_dir: Path, runs: list, rows: list, details: dict, R: float, timings: dict, config: dict) -> None:
    """Writes ``runs.csv``, ``sweep.csv`` and ``details.npz`` (per-run arrays (runs × thresholds), NIS, ``R``, timings)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "runs.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario", "index", "seed", "node", "start"])
        for r in runs:
            writer.writerow([r["scenario"], r["index"], r["seed"], r["node"], r["start"]])
    with open(out_dir / "sweep.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows({k: row[k] for k in ROW_FIELDS} for row in rows)
    arrays = {"R": np.array(R), "sigma_values": np.array(list(details)), "nis_edges": nis_bin_edges(config)}
    for name, value in timings.items():
        arrays[f"time_{name}"] = np.array(value)
    for sigma_a, d in details.items():
        arrays[f"nis_mean|{sigma_a}"] = np.array(d["nis_mean"])
        arrays[f"nis_hist|{sigma_a}"] = d["nis_hist"]
        arrays[f"nis_over|{sigma_a}"] = np.array(d["nis_over"])
        arrays[f"nis_count|{sigma_a}"] = np.array(d["nis_count"])
        for scenario, outs in d["separated"].items():
            for key in outs[0]["results"]:
                prefix = f"{sigma_a}|{scenario}|{'|'.join(key)}"
                arrays[prefix + "|threshold"] = outs[0]["results"][key]["threshold"]
                for field in ("delay", "distance", "fortuitous", "success"):
                    arrays[prefix + "|" + field] = np.stack([o["results"][key][field] for o in outs])
    np.savez(out_dir / "details.npz", **arrays)


def load_results(out_dir: Path) -> tuple:
    """Reads ``sweep.csv`` and ``details.npz``: (typed rows, dict of arrays)."""
    out_dir = Path(out_dir)
    rows = []
    with open(out_dir / "sweep.csv", newline="") as f:
        for raw in csv.DictReader(f):
            row = {}
            for key, value in raw.items():
                if key in _TEXT_FIELDS:
                    row[key] = value
                elif key in _INT_FIELDS:
                    row[key] = int(value)
                else:
                    row[key] = float(value)
            rows.append(row)
    with np.load(out_dir / "details.npz") as f:
        details = {k: f[k] for k in f.files}
    return rows, details


def _fmt(value: float, spec: str) -> str:
    return "  -" if value is None or (isinstance(value, float) and math.isnan(value)) else format(value, spec)


def print_summary(rows: list, details: dict, config: dict) -> None:
    """Prints ``R``, mean NIS per ``σ_a`` and, per variant, mode, scope and scenario, the working point of the best ``σ_a`` (`best_over_sigma`)."""
    mcfg = config["metrics"]
    level = mcfg["confidence_level"]
    print(f"R (measurement noise) = {float(details['R']):.3f} dB²")
    print("Mean NIS in the group without separation (expected 1 if the filter is consistent):")
    for sigma_a in details["sigma_values"]:
        print(f"  sigma_a = {sigma_a:g} dB/s²: mean NIS = {float(details[f'nis_mean|{sigma_a}']):.3f}")
    scenarios = [s for s in mcfg["scenarios"] if s != Scenario.NONE]
    titles = {"both": "pre-alarm + lost", "pre": "pre-alarm only", "lost": "'lost' alarm only"}
    print(
        f"\nSuccess = alarm before the distance from the group exceeds {mcfg['success_distance']:g} m;"
        f" feasible = false alarms/hour <= {mcfg['target_false_alarms_per_hour']:g} and time in alarm"
        f" <= {100 * mcfg['max_alarm_time_fraction']:g}%; intervals at {100 * level:g}%"
        f" (Wilson for P_d, Poisson for FA/h)."
    )
    for variant in ("both", "pre", "lost"):
        print(f"\nWorking point ({titles[variant]}), best sigma_a:")
        print(
            f"{'mode':>8} {'scope':>8} {'scenario':>9} {'sigma_a':>7} {'thresh':>7} {'P_d':>5} {'Wilson':>13}"
            f" {'<=' + format(mcfg['success_window'], 'g') + 's':>6} {'fort.':>5}"
            f" {'del.med s':>9} {'del.p90 s':>9} {'dist.med m':>10} {'dist.p90 m':>10}"
            f" {'FA/h':>5} {'Poisson':>13} {'in alarm':>10}"
        )
        combos = [("-", "-")] if variant == "lost" else [(m, s) for m in MODES for s in SCOPES]
        for mode, scope in combos:
            best = best_over_sigma(rows, scope, mode, variant)
            for scenario in scenarios:
                if best is None:
                    print(f"{mode:>8} {scope:>8} {scenario:>9}   no feasible point")
                    continue
                row = next(
                    r for r in rows
                    if (r["scope"], r["mode"], r["variant"], r["scenario"], r["sigma_a"])
                    == (scope, mode, variant, scenario, best["sigma_a"])
                    and (r["threshold"] == best["threshold"] or (math.isnan(r["threshold"]) and math.isnan(best["threshold"])))
                )  # fmt: skip
                print(
                    f"{mode:>8} {scope:>8} {scenario:>9} {row['sigma_a']:>7g} {_fmt(row['threshold'], '7.2f'):>7}"
                    f" {row['p_detect']:>5.2f} {'[' + format(row['p_lo'], '.2f') + '-' + format(row['p_hi'], '.2f') + ']':>13}"
                    f" {row['p_detect_window']:>6.2f} {row['fortuitous']:>2d}/{row['n_runs']:<2d}"
                    f" {_fmt(row['delay_median'], '9.1f'):>9} {_fmt(row['delay_p90'], '9.1f'):>9}"
                    f" {_fmt(row['distance_median'], '10.1f'):>10} {_fmt(row['distance_p90'], '10.1f'):>10}"
                    f" {row['fa_per_hour']:>5.2f} {'[' + format(row['fa_lo'], '.2f') + '-' + format(row['fa_hi'], '.2f') + ']':>13}"
                    f" {100 * row['active_fraction']:>9.2f}%"
                )

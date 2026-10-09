"""Esperimento del Blocco 4: lavoro per corsa e sweep.

Per ogni corsa (scenario, indice) e ogni ``σ_a``: Kalman → tabelle →
fusione → rilevatore su tutta la griglia di soglie, con i Blocchi 1-3 letti
dalla cache (`src/simulator.py`). Le corse sono indipendenti: si eseguono in
parallelo, e il risultato non dipende dal numero di processi.

Parametri: sezioni ``metrics`` e ``detector`` della config.
"""

from __future__ import annotations

import csv
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from src.detector import build_inputs
from src.metrics import (
    MODES,
    SCOPES,
    aggregate,
    best_over_sigma,
    evaluate_run,
    min_distance_to_group,
    summarize_none,
    summarize_separated,
)
from src.simulator import cache_path, load_or_simulate

NIS_BIN_EDGES = np.linspace(0.0, 12.0, 121)  # solo per l'istogramma della NIS


def pooled_fading_variance(runs: list, cache_dir: Path) -> float:
    """Varianza del fading (dB²) nel gruppo: media sulle corse ``none`` delle varianze fuori diagonale."""
    values = []
    for run in runs:
        if run["scenario"] == "none":
            with np.load(cache_path(run["config"], cache_dir)) as f:
                values.append(float(f["fading_var"]))
    return float(np.mean(values))


def process_run(args: tuple) -> dict:
    """Blocco 4 su una corsa per un ``σ_a``; `args` = (corsa, config, cartella di cache, ``σ_a``, ``R``).

    ``none``: `evaluate_run` su tutti i bersagli e NIS (somma, conteggio, istogramma, ``nis_over``);
    con distacco: ritagliata sul nodo separato. Restituisce ``scenario``, ``index``, ``results``, ``elapsed``.
    """
    run_info, config, cache_dir, sigma_a, R = args
    t0 = time.perf_counter()
    run = load_or_simulate(run_info["config"], cache_dir)
    inputs, kres = build_inputs(run, config, sigma_a, R)
    out = {"scenario": run_info["scenario"], "index": run_info["index"]}
    t = run["t"]
    if run_info["scenario"] == "none":
        warmup = config["detector"]["warmup"]
        out["results"] = evaluate_run(inputs, config, lambda s: summarize_none(s, t, warmup))
        nis = kres.nis[np.isfinite(kres.nis)]
        out["nis_sum"] = float(nis.sum())
        out["nis_count"] = int(len(nis))
        out["nis_hist"] = np.histogram(nis, bins=NIS_BIN_EDGES)[0]
        out["nis_over"] = int((nis >= NIS_BIN_EDGES[-1]).sum())
    else:
        node = run["sep_node"]
        idx_start = int(np.searchsorted(t, run["sep_start"]))
        distance = min_distance_to_group(run["distances"], node)
        sliced = inputs.target_slice(node)
        success_distance = config["metrics"]["success_distance"]
        out["results"] = evaluate_run(sliced, config, lambda s: summarize_separated(s, t, idx_start, distance, success_distance))
    out["elapsed"] = time.perf_counter() - t0
    return out


def run_sweep(config: dict, runs: list, cache_dir: Path, sigma_values: list, R: float, jobs: int) -> tuple:
    """Sweep su ``σ_a``, modalità, ambito e soglia (`process_run` su tutte le corse, poi `aggregate`).

    Restituisce (righe con ``sigma_a``, dettagli per ``σ_a``: risultati per corsa, NIS, tempi).
    """
    rows, details = [], {}
    for sigma_a in sigma_values:
        tasks = [(r, config, cache_dir, sigma_a, R) for r in runs]
        if jobs <= 1:
            outs = [process_run(task) for task in tasks]
        else:
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                outs = list(pool.map(process_run, tasks))
        none = sorted((o for o in outs if o["scenario"] == "none"), key=lambda o: o["index"])
        separated = {
            s: sorted((o for o in outs if o["scenario"] == s), key=lambda o: o["index"])
            for s in config["metrics"]["scenarios"]
            if s != "none"
        }
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
# Salvataggio, lettura e resoconto
# ---------------------------------------------------------------------------

ROW_FIELDS = (
    "sigma_a", "scope", "mode", "variant", "threshold", "scenario", "p_detect", "p_lo", "p_hi",
    "p_detect_window", "n_runs", "fortuitous", "delay_median", "delay_p90", "distance_median",
    "distance_p90", "fa_per_hour", "fa_lo", "fa_hi", "active_fraction", "feasible",
)  # fmt: skip
_TEXT_FIELDS = ("scope", "mode", "variant", "scenario")
_INT_FIELDS = ("n_runs", "fortuitous", "feasible")


def save_results(out_dir: Path, runs: list, rows: list, details: dict, R: float, timings: dict) -> None:
    """Scrive ``runs.csv``, ``sweep.csv`` e ``details.npz`` (array per corsa (corse × soglie), NIS, ``R``, tempi)."""
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
    arrays = {"R": np.array(R), "sigma_values": np.array(list(details)), "nis_edges": NIS_BIN_EDGES}
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
    """Legge ``sweep.csv`` e ``details.npz``: (righe tipizzate, dizionario di array)."""
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
    """Stampa ``R``, NIS media per ``σ_a`` e, per variante, modalità, ambito e scenario, il punto di lavoro del miglior ``σ_a`` (`best_over_sigma`)."""
    mcfg = config["metrics"]
    level = mcfg["confidence_level"]
    print(f"R (rumore di misura) = {float(details['R']):.3f} dB²")
    print("NIS media nel gruppo senza distacco (attesa 1 se il filtro è consistente):")
    for sigma_a in details["sigma_values"]:
        print(f"  sigma_a = {sigma_a:g} dB/s²: NIS media = {float(details[f'nis_mean|{sigma_a}']):.3f}")
    scenarios = [s for s in mcfg["scenarios"] if s != "none"]
    titles = {"both": "pre-allarme + perso", "pre": "solo pre-allarme", "lost": "solo allarme 'perso'"}
    print(
        f"\nSuccesso = allarme prima che la distanza dal gruppo superi {mcfg['success_distance']:g} m;"
        f" ammissibile = falsi allarmi/ora <= {mcfg['target_false_alarms_per_hour']:g} e tempo in allarme"
        f" <= {100 * mcfg['max_alarm_time_fraction']:g}%; intervalli al {100 * level:g}%"
        f" (Wilson per P_d, Poisson per FA/h)."
    )
    for variant in ("both", "pre", "lost"):
        print(f"\nPunto di lavoro ({titles[variant]}), miglior sigma_a:")
        print(
            f"{'modalita':>8} {'ambito':>8} {'scenario':>9} {'sigma_a':>7} {'soglia':>7} {'P_d':>5} {'Wilson':>13}"
            f" {'<=' + format(mcfg['success_window'], 'g') + 's':>6} {'fort.':>5}"
            f" {'rit.med s':>9} {'rit.p90 s':>9} {'dist.med m':>10} {'dist.p90 m':>10}"
            f" {'FA/h':>5} {'Poisson':>13} {'in allarme':>10}"
        )
        combos = [("-", "-")] if variant == "lost" else [(m, s) for m in MODES for s in SCOPES]
        for mode, scope in combos:
            best = best_over_sigma(rows, scope, mode, variant)
            for scenario in scenarios:
                if best is None:
                    print(f"{mode:>8} {scope:>8} {scenario:>9}   nessun punto ammissibile")
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

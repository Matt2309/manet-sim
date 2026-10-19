#!/usr/bin/env python3
"""Rigenera tutte le figure dei Blocchi 1-6 e ``results/final_summary.md`` con un solo comando.

Usa la cache in ``results/cache/`` se c'è e salta gli esperimenti i cui risultati esistono già (``results/detection``, ``results/block5/results.pkl``,
``results/block6/results.pkl``). Con ``--clean`` riparte da zero: cancella la cache e i risultati degli esperimenti, ma NON
``results/block6/frozen_thresholds.yaml`` (la taratura deve riconfermare le soglie congelate). Stampa la durata di ogni fase e il totale.
"""

import argparse
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.final import _git, load_frozen, write_final_summary  # noqa: E402
from src.mobility import load_config  # noqa: E402
from src.refinement import load_pickle  # noqa: E402

LIBRARIES = ("numpy", "scipy", "matplotlib", "PyYAML", "numba", "llvmlite", "pytest")


def run(step: str, args: list, timings: dict) -> None:
    print(f"\n=== {step} ===", flush=True)
    t0 = time.perf_counter()
    subprocess.run([sys.executable, *args], check=True, cwd=_REPO_ROOT)
    timings[step] = time.perf_counter() - t0
    print(f"--- {step}: {timings[step]:.1f} s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", action="store_true", help="riparte senza cache né risultati degli esperimenti")
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="processi paralleli")
    args = parser.parse_args()
    results = _REPO_ROOT / "results"
    jobs = str(args.jobs)
    t_start = time.perf_counter()
    timings: dict = {}

    if args.clean:
        for path in (results / "cache", results / "detection", results / "block5" / "results.pkl", results / "block6" / "results.pkl"):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        print("Cache e risultati degli esperimenti cancellati (frozen_thresholds.yaml conservato)")

    for name in ("plot_mobility", "plot_channel", "plot_packets"):
        run(f"Blocchi 1-3: {name}", [f"scripts/{name}.py"], timings)
    if not (results / "detection" / "sweep.csv").exists():
        run("Blocco 4: esperimento", ["scripts/run_detection_experiment.py", "--jobs", jobs], timings)
    run("Blocco 4: figure", ["scripts/plot_detector.py"], timings)
    if not (results / "block5" / "results.pkl").exists():
        run("Blocco 5: esperimento", ["scripts/run_block5_experiment.py", "--jobs", jobs, "--force"], timings)
    run("Blocco 5: figure", ["scripts/plot_block5.py"], timings)
    if not (results / "block6" / "results.pkl").exists():
        run("Blocco 6: esperimento", ["scripts/run_block6_experiment.py", "--jobs", jobs], timings)
    run("Blocco 6: figure", ["scripts/plot_block6.py"], timings)

    print("\n=== Riepilogo finale ===", flush=True)
    config = load_config(_REPO_ROOT / "config" / "default.yaml")
    res = load_pickle(results / "block6" / "results.pkl")
    frozen = load_frozen(results / "block6" / "frozen_thresholds.yaml")
    final = frozen["thresholds"][res["final_name"]]
    mismatches = []
    if abs(config["detector"]["level_threshold"] - final["threshold"]) > 1e-9:
        mismatches.append(f"detector.level_threshold = {config['detector']['level_threshold']} ma la soglia congelata è {final['threshold']}")
    if config["fusion"]["rule"] != final["fusion_rule"]:
        mismatches.append(f"fusion.rule = {config['fusion']['rule']} ma il verdetto è {final['fusion_rule']}")
    if config["detector"]["scope"] != final["scope"]:
        mismatches.append(f"detector.scope = {config['detector']['scope']} ma la finale è {final['scope']}")
    for m in mismatches:
        print(f"ATTENZIONE: config/default.yaml non descrive il rilevatore finale: {m}")
    versions = {
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "libraries": {lib: importlib.metadata.version(lib) for lib in LIBRARIES},
    }
    summary = results / "final_summary.md"
    write_final_summary(res, summary, versions)
    print(f"Scritto {summary}")
    total = time.perf_counter() - t_start
    print(f"\nDurata totale: {total / 60:.1f} min ({total:.0f} s)")
    for step, value in timings.items():
        print(f"  {step:<34} {value:>8.1f} s")


if __name__ == "__main__":
    main()

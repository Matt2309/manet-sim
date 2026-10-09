#!/usr/bin/env python3
"""Esperimento del Blocco 4: rilevamento del distacco.

Per ogni scenario (none, slowdown, stop) ``metrics.n_runs`` corse (Blocchi 1-3
in cache in ``results/cache/``); per ogni ``sigma_a`` di
``kalman.sigma_a_sweep``, modalità, ambito e soglia calcola P_d e falsi
allarmi/ora. Salva in ``results/detection/`` e stampa il riepilogo.
Config: config/default.yaml se non indicata.
"""

import argparse
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.detector import count_episodes, build_inputs, lost_alarm, pre_alarm, system_alarm  # noqa: E402
from src.experiment import (  # noqa: E402
    pooled_fading_variance,
    print_summary,
    run_sweep,
    save_results,
    load_results,
)
from src.kalman import measurement_noise  # noqa: E402
from src.mobility import load_config  # noqa: E402
from src.simulator import ensure_all_cached, experiment_runs, load_or_simulate  # noqa: E402


def time_single_run(config: dict, run_info: dict, cache_dir: Path, R: float) -> float:
    """Tempo (s) del solo Blocco 4 (Kalman, tabelle, fusione, rilevatore) su una corsa, con ``sigma_a`` di default."""
    run = load_or_simulate(run_info["config"], cache_dir)
    t0 = time.perf_counter()
    inputs, _ = build_inputs(run, config, config["kalman"]["sigma_a"], R)
    d = config["detector"]
    scope = d["scope"]
    pre = pre_alarm(inputs.level[scope], inputs.slope[scope], inputs.rx_ok, inputs.t, d, valid=inputs.valid)
    lost = lost_alarm(inputs.silence, inputs.rx_ok, inputs.t, d, inputs.valid)
    count_episodes(system_alarm(pre | lost, d["min_observers"]), inputs.t, d["warmup"])
    return time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default=str(_REPO_ROOT / "config" / "default.yaml"))
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="processi paralleli")
    parser.add_argument("--cache-dir", default=str(_REPO_ROOT / "results" / "cache"))
    parser.add_argument("--out-dir", default=str(_REPO_ROOT / "results" / "detection"))
    args = parser.parse_args()

    config = load_config(args.config)
    cache_dir, out_dir = Path(args.cache_dir), Path(args.out_dir)
    runs = experiment_runs(config)
    print(f"{len(runs)} corse ({config['metrics']['n_runs']} per scenario: {config['metrics']['scenarios']})")

    t_start = time.perf_counter()
    t0 = time.perf_counter()
    ensure_all_cached(runs, cache_dir, args.jobs)
    t_cache = time.perf_counter() - t0
    print(f"Blocchi 1-3 (con cache): {t_cache:.1f} s")

    fading_var = pooled_fading_variance(runs, cache_dir)
    R = measurement_noise(config, fading_var)
    print(f"varianza del fading nel gruppo = {fading_var:.3f} dB², R = {R:.3f} dB²")

    t_one = time_single_run(config, next(r for r in runs if r["scenario"] == "none"), cache_dir, R)
    print(f"Blocco 4 su una corsa, un sigma_a: {t_one:.2f} s")

    t0 = time.perf_counter()
    rows, details = run_sweep(config, runs, cache_dir, config["kalman"]["sigma_a_sweep"], R, args.jobs)
    t_sweep = time.perf_counter() - t0
    timings = {
        "block4_one_run": t_one,
        "blocks123_cache": t_cache,
        "sweep": t_sweep,
        "experiment": time.perf_counter() - t_start,
    }
    save_results(out_dir, runs, rows, details, R, timings)
    print(f"Risultati salvati in {out_dir}\n")

    saved_rows, saved_details = load_results(out_dir)
    print_summary(saved_rows, saved_details, config)
    print(
        f"\nTempi: Blocco 4 su una corsa (un sigma_a) {t_one:.2f} s; sweep {t_sweep:.1f} s;"
        f" esperimento intero {timings['experiment']:.1f} s ({args.jobs} processi)"
    )


if __name__ == "__main__":
    main()

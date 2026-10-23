#!/usr/bin/env python3
"""Block 4 experiment: separation detection.

For each scenario (none, slowdown, stop) ``metrics.n_runs`` runs (Blocks 1-3
cached in ``results/cache/``); for each ``sigma_a`` in
``kalman.sigma_a_sweep``, mode, scope and threshold computes P_d and false
alarms/hour. Saves to ``results/block4/`` and prints the summary.
Config: config/default.yaml if not specified.
"""

import time
from pathlib import Path

from _cli import load_block4_config, make_parser  # noqa: E402
from src.common.paths import block_dir  # noqa: E402
from src.detector import build_inputs, combined_alarm, count_episodes, lost_alarm, pre_alarm  # noqa: E402
from src.experiment import load_results, pooled_fading_variance, print_summary, run_sweep, save_results  # noqa: E402
from src.kalman import measurement_noise  # noqa: E402
from src.simulator import ensure_all_cached, experiment_runs, load_or_simulate  # noqa: E402


def time_single_run(config: dict, run_info: dict, cache_dir: Path, R: float) -> float:
    """Time (s) of Block 4 alone (Kalman, tables, fusion, detector) on one run, with the default ``sigma_a``."""
    run = load_or_simulate(run_info["config"], cache_dir)
    t0 = time.perf_counter()
    inputs, _ = build_inputs(run, config, config["kalman"]["sigma_a"], R)
    d = config["detector"]
    scope = d["scope"]
    pre = pre_alarm(inputs.level[scope], inputs.slope[scope], inputs.rx_ok, inputs.t, d, valid=inputs.valid)
    lost = lost_alarm(inputs.silence, inputs.rx_ok, inputs.t, d, inputs.valid, **inputs.lost_kwargs())
    count_episodes(combined_alarm(pre, lost, d["min_observers"], d.get("lost_min_observers")), inputs.t, d["warmup"])
    return time.perf_counter() - t0


def main() -> None:
    parser = make_parser(__doc__, jobs=True, cache_dir=True, out_dir=block_dir(4))
    args = parser.parse_args()

    config = load_block4_config(args.config)  # Block 6: the defaults changed, here those of Block 4 apply
    cache_dir, out_dir = Path(args.cache_dir), Path(args.out_dir)
    runs = experiment_runs(config)
    print(f"{len(runs)} runs ({config['metrics']['n_runs']} per scenario: {config['metrics']['scenarios']})")

    t_start = time.perf_counter()
    t0 = time.perf_counter()
    ensure_all_cached(runs, cache_dir, args.jobs)
    t_cache = time.perf_counter() - t0
    print(f"Blocks 1-3 (with cache): {t_cache:.1f} s")

    fading_var = pooled_fading_variance(runs, cache_dir)
    R = measurement_noise(config, fading_var)
    print(f"fading variance in the group = {fading_var:.3f} dB², R = {R:.3f} dB²")

    t_one = time_single_run(config, next(r for r in runs if r["scenario"] == "none"), cache_dir, R)
    print(f"Block 4 on one run, one sigma_a: {t_one:.2f} s")

    t0 = time.perf_counter()
    rows, details = run_sweep(config, runs, cache_dir, config["kalman"]["sigma_a_sweep"], R, args.jobs)
    t_sweep = time.perf_counter() - t0
    timings = {
        "block4_one_run": t_one,
        "blocks123_cache": t_cache,
        "sweep": t_sweep,
        "experiment": time.perf_counter() - t_start,
    }
    save_results(out_dir, runs, rows, details, R, timings, config)
    print(f"Results saved to {out_dir}\n")

    saved_rows, saved_details = load_results(out_dir)
    print_summary(saved_rows, saved_details, config)
    print(
        f"\nTimings: Block 4 on one run (one sigma_a) {t_one:.2f} s; sweep {t_sweep:.1f} s;"
        f" whole experiment {timings['experiment']:.1f} s ({args.jobs} processes)"
    )


if __name__ == "__main__":
    main()

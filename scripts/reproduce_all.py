#!/usr/bin/env python3
"""Regenerate all the figures of Blocks 1-7 and ``results/final_summary.md`` with a single command.

Uses the cache in ``results/cache/`` if present and skips the experiments whose results already exist (``results/block4/sweep.csv``, ``results/block5/results.pkl``,
``results/block6/results.pkl``). With ``--clean`` it starts from scratch: deletes the cache and the experiment results, but NOT
``results/block6/frozen_thresholds.yaml`` (tuning must reconfirm the frozen thresholds). Prints the duration of each phase and the total.
"""

import importlib.metadata
import platform
import shutil
import subprocess
import sys
import time

from _cli import make_parser  # noqa: E402
from src.common.io import load_pickle  # noqa: E402
from src.common.paths import block_dir, DEFAULT_CONFIG, REPO_ROOT, RESULTS_DIR  # noqa: E402
from src.common.git import git_or_unknown
from src.final import load_frozen, write_final_summary  # noqa: E402
from src.mobility import load_config  # noqa: E402


LIBRARIES = ("numpy", "scipy", "matplotlib", "PyYAML", "numba", "llvmlite", "pytest")

BLOCK4_DATA = ("runs.csv", "sweep.csv", "details.npz")


def run(step: str, args: list, timings: dict) -> None:
    print(f"\n=== {step} ===", flush=True)
    t0 = time.perf_counter()
    subprocess.run([sys.executable, *args], check=True, cwd=REPO_ROOT)
    timings[step] = time.perf_counter() - t0
    print(f"--- {step}: {timings[step]:.1f} s", flush=True)


def main() -> None:
    parser = make_parser(__doc__, config=False, jobs=True)
    parser.add_argument("--clean", action="store_true", help="starts without cache or experiment results")
    args = parser.parse_args()
    results = RESULTS_DIR
    jobs = str(args.jobs)
    profile = load_config(DEFAULT_CONFIG)["routing"]["channel_profile"]
    t_start = time.perf_counter()
    timings: dict = {}

    if args.clean:
        for path in (results / "cache", *(block_dir(4) / name for name in BLOCK4_DATA), block_dir(5) / "results.pkl", block_dir(6) / "results.pkl", block_dir(7, profile) / "results.pkl"):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        print("Cache and experiment results deleted (frozen_thresholds.yaml kept)")

    for name in ("plot_mobility", "plot_channel", "plot_packets"):
        run(f"Blocks 1-3: {name}", [f"scripts/{name}.py"], timings)
    if not (block_dir(4) / "sweep.csv").exists():
        run("Block 4: experiment", ["scripts/run_detection_experiment.py", "--jobs", jobs], timings)
    run("Block 4: figures", ["scripts/plot_detector.py"], timings)
    if not (block_dir(5) / "results.pkl").exists():
        run("Block 5: experiment", ["scripts/run_block5_experiment.py", "--jobs", jobs, "--force"], timings)
    run("Block 5: figures", ["scripts/plot_block5.py"], timings)
    if not (block_dir(6) / "results.pkl").exists():
        run("Block 6: experiment", ["scripts/run_block6_experiment.py", "--jobs", jobs], timings)
    run("Block 6: figures", ["scripts/plot_block6.py"], timings)
    if not (block_dir(7, profile) / "results.pkl").exists():
        run("Block 7: experiment", ["scripts/run_block7_experiment.py", "--jobs", jobs], timings)
    run("Block 7: figures", ["scripts/plot_block7.py"], timings)

    print("\n=== Final summary ===", flush=True)
    config = load_config(DEFAULT_CONFIG)
    res = load_pickle(block_dir(6) / "results.pkl")
    frozen = load_frozen(block_dir(6) / "frozen_thresholds.yaml")
    final = frozen["thresholds"][res["final_name"]]
    mismatches = []
    if abs(config["detector"]["level_threshold"] - final["threshold"]) > 1e-9:
        mismatches.append(f"detector.level_threshold = {config['detector']['level_threshold']} but the frozen threshold is {final['threshold']}")
    if config["fusion"]["rule"] != final["fusion_rule"]:
        mismatches.append(f"fusion.rule = {config['fusion']['rule']} but the verdict is {final['fusion_rule']}")
    if config["detector"]["scope"] != final["scope"]:
        mismatches.append(f"detector.scope = {config['detector']['scope']} but the final one is {final['scope']}")
    for m in mismatches:
        print(f"WARNING: config/default.yaml does not describe the final detector: {m}")
    versions = {
        "commit": git_or_unknown("rev-parse", "HEAD"),
        "dirty": bool(git_or_unknown("status", "--porcelain")),
        "python": platform.python_version(),
        "libraries": {lib: importlib.metadata.version(lib) for lib in LIBRARIES},
    }
    summary = results / "final_summary.md"
    write_final_summary(res, summary, versions)
    print(f"Written {summary}")
    total = time.perf_counter() - t_start
    print(f"\nTotal duration: {total / 60:.1f} min ({total:.0f} s)")
    for step, value in timings.items():
        print(f"  {step:<34} {value:>8.1f} s")


if __name__ == "__main__":
    main()

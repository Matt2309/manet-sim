#!/usr/bin/env python3
"""Block 6 experiment: frozen thresholds, evaluation on new runs, media/migliore verdict, sensitivity.

1. tuning (150 runs of Blocks 4-5, keys 6-7): threshold sweep for `media`, `migliore`, `coppia`, choice of the working point, writing of
   ``results/block6/frozen_thresholds.yaml``;
2. verification (``final.n_runs`` runs per scenario, keys 8-9): evaluation at the thresholds READ from the file;
3. verdict media versus migliore (``results/block6/verdict.yaml``);
4. sensitivity (first ``final.sensitivity_runs`` verification runs; only Block 3 changes) of the final configuration.
If ``frozen_thresholds.yaml`` exists it is not rewritten: tuning must give the same thresholds again (otherwise error; ``--refreeze`` to rewrite it).
Config: config/default.yaml if not given.
"""

import datetime
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from _cli import make_parser  # noqa: E402
from src.common.git import git_or_unknown  # noqa: E402
from src.common.io import load_pickle, save_pickle  # noqa: E402
from src.common.paths import block_dir  # noqa: E402
from src.experiment import pooled_fading_variance  # noqa: E402
from src.final import (
    decide_verdict,
    evaluate_at,
    evaluate_sensitivity,
    evaluation_table,
    freeze_thresholds,
    load_frozen,
    LOST,
    process_run_final,
    run_final_phase,
    run_sets,
    sensitivity_cases,
    sensitivity_table,
    sweep_rows,
    total_success,
    verdict_text,
    write_frozen,
)
from src.kalman import measurement_noise  # noqa: E402
from src.mobility import load_config  # noqa: E402
from src.simulator import ensure_all_cached  # noqa: E402


def main() -> None:
    parser = make_parser(__doc__, jobs=True, cache_dir=True, out_dir=block_dir(6), force_help="proceeds even if the estimate exceeds ``final.time_budget_minutes``")
    parser.add_argument("--refreeze", action="store_true", help="rewrites frozen_thresholds.yaml from the tuning")
    args = parser.parse_args()

    config = load_config(args.config)
    cache_dir, out_dir, jobs = Path(args.cache_dir), Path(args.out_dir), args.jobs
    out_dir.mkdir(parents=True, exist_ok=True)
    fcfg = config["final"]
    timings = {}
    t_start = time.perf_counter()

    tuning, verification = run_sets(config)
    seeds_t, seeds_v = {r["seed"] for r in tuning}, {r["seed"] for r in verification}
    assert not seeds_t & seeds_v, "seeds in common between tuning and verification"
    n_sens = fcfg["sensitivity_runs"]
    print(
        f"Tuning: {len(tuning)} runs ({config['metrics']['n_runs']} per scenario); verification: {len(verification)} runs"
        f" ({fcfg['n_runs']} per scenario); sensitivity: first {n_sens} per scenario; {jobs} processes"
    )

    # ---- Cache ---------------------------------------------------------------
    t0 = time.perf_counter()
    ensure_all_cached(tuning, cache_dir, jobs)
    ensure_all_cached(verification, cache_dir, jobs, with_channel=lambda r: r["index"] < n_sens)
    timings["cache"] = time.perf_counter() - t0
    R = measurement_noise(config, pooled_fading_variance(tuning, cache_dir))
    print(f"R = {R:.3f} dB² (from tuning); cache: {timings['cache']:.1f} s")

    # ---- Time estimate -------------------------------------------------------
    spec_main = {"mode": "cache", "point": {}, "configs": list(fcfg["configs"]), "rules": ["any"]}
    cases = sensitivity_cases(config)
    pess = next(p for n, p, _ in cases if n == "pessimista")
    spec_probe = {"mode": "channel", "point": pess, "configs": ["media"], "rules": ["any", "not_deaf", "not_deaf_episode"]}
    probe = [next(r for r in verification if r["scenario"] == s) for s in ("none", "slowdown")]
    process_run_final((probe[0], config, cache_dir, R, {**spec_main, "configs": ["media"]}))  # numba compilation
    t_main, t_sens = [], []
    for r in probe:
        t0 = time.perf_counter()
        process_run_final((r, config, cache_dir, R, spec_main))
        t_main.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        process_run_final((r, config, cache_dir, R, spec_probe))
        t_sens.append(time.perf_counter() - t0)
    n_tun, n_ver = config["metrics"]["n_runs"], fcfg["n_runs"]
    est_main = ((t_main[0] * (n_tun + n_ver)) + 2 * t_main[1] * (n_tun + n_ver)) / jobs / 60.0
    est_sens = len(cases) * (t_sens[0] * n_sens + 2 * t_sens[1] * n_sens) / jobs / 60.0
    print(
        f"One none run: {t_main[0]:.1f} s (3 configurations), with separation {t_main[1]:.1f} s; sensitivity: {t_sens[0]:.1f} s / {t_sens[1]:.1f} s."
        f" Estimate: tuning + verification {est_main:.1f} min, sensitivity {est_sens:.1f} min, total about {est_main + est_sens:.1f} min (cache excluded)"
    )
    if est_main + est_sens > fcfg["time_budget_minutes"] and not args.force:
        print(f"The estimate exceeds {fcfg['time_budget_minutes']} min: stopping (--force to proceed).")
        sys.exit(2)

    # ---- 1. Tuning and frozen thresholds ---------------------------------------
    t0 = time.perf_counter()
    tuning_outs = run_final_phase(config, tuning, cache_dir, R, spec_main, jobs)
    timings["tuning"] = time.perf_counter() - t0
    chosen = freeze_thresholds(tuning_outs, config)
    frozen_path = out_dir / "frozen_thresholds.yaml"
    if frozen_path.exists() and not args.refreeze:
        frozen = load_frozen(frozen_path)
        for name, f in chosen.items():
            if abs(frozen["thresholds"][name]["threshold"] - f["threshold"]) > 1e-9:
                raise SystemExit(
                    f"Tuning gives threshold {f['threshold']} for {name} but the frozen file has {frozen['thresholds'][name]['threshold']}:"
                    f" the file is not rewritten (use --refreeze if the change is intended)."
                )
        print(f"Frozen thresholds read from {frozen_path} (tuning reconfirms them)")
    else:
        write_frozen(frozen_path, chosen, config, R)
        frozen = load_frozen(frozen_path)
        print(f"Frozen thresholds written to {frozen_path}")
    print(f"\n=== Frozen thresholds (tuning: {frozen['date']}, commit {frozen['commit']}{' + changes' if frozen['dirty'] else ''}, release_time {frozen['release_time']:g} s) ===")
    print(f"{'config':<9} {'scope':>9} {'rule':>13} {'thr. dBm':>10} {'P_d slowdown':>13} {'P_d stop':>9} {'FA/h':>6} {'in alarm':>10}")
    for name, f in frozen["thresholds"].items():
        t = f["tuning"]
        print(
            f"{name:<9} {f['scope']:>9} {f['fusion_rule']:>13} {f['threshold']:>10.1f} {t['p_detect_slowdown']:>13.2f} {t['p_detect_stop']:>9.2f}"
            f" {t['fa_per_hour']:>6.2f} {100 * t['active_fraction']:>9.2f}%"
        )
    block5 = block_dir(5) / "results.pkl"
    if block5.exists():
        s5 = load_pickle(block5)["summaries"].get(("C:release=30", "fused", 1))
        m = frozen["thresholds"]["media"]
        if s5 is not None and s5["wp"] is not None:
            print(
                f"Check against Block 5 (C:release=30): threshold {s5['wp']['threshold']:.1f} dBm, P_d {s5['point']['slowdown']['p_detect']:.2f}/"
                f"{s5['point']['stop']['p_detect']:.2f}, FA/h {s5['wp']['fa_per_hour']:.2f} -> "
                + ("matches" if abs(s5["wp"]["threshold"] - m["threshold"]) < 1e-9 else "DIFFERENT")
            )

    # ---- 2. Verification -------------------------------------------------------------
    t0 = time.perf_counter()
    verif_outs = run_final_phase(config, verification, cache_dir, R, spec_main, jobs)
    timings["verification"] = time.perf_counter() - t0
    names = [*fcfg["configs"], LOST]
    thr = {n: (frozen["thresholds"][n]["threshold"] if n != LOST else math.nan) for n in names}
    verification_eval = {n: evaluate_at(verif_outs, n, "any", thr[n], config) for n in names}
    tuning_eval = {n: evaluate_at(tuning_outs, n, "any", thr[n], config) for n in names}
    sweeps = {
        "tuning": {n: sweep_rows(tuning_outs, n, "any", config) for n in fcfg["configs"]},
        "verification": {n: sweep_rows(verif_outs, n, "any", config) for n in fcfg["configs"]},
    }

    # ---- 3. Verdict -------------------------------------------------------------
    v = decide_verdict(
        total_success(verification_eval["media"]["at"]),
        total_success(verification_eval["migliore"]["at"]),
        verification_eval["media"]["summary"]["none"]["fa_per_hour"],
        verification_eval["migliore"]["summary"]["none"]["fa_per_hour"],
        fcfg["min_net_gain"],
    )
    final_name = v["adopted"]
    with open(out_dir / "verdict.yaml", "w") as f:
        yaml.safe_dump(
            {"date": datetime.datetime.now().isoformat(timespec="seconds"), "commit": git_or_unknown("rev-parse", "HEAD"), **v,
             "final": final_name, "final_threshold": frozen["thresholds"][final_name]["threshold"]},
            f, sort_keys=False, allow_unicode=True,
        )  # fmt: skip
    results = {
        "config": config, "R": R, "frozen": frozen, "tuning": tuning_eval, "verification": verification_eval, "verdict": v,
        "final_name": final_name, "sweeps": sweeps, "diag_tuning": None,
    }  # fmt: skip
    print("\n=== Evaluation on the verification runs (frozen thresholds; optimism = verification minus tuning) ===")
    results["sensitivity"] = {}
    print(evaluation_table(results))
    print("\n=== Verdict media versus migliore ===")
    print(verdict_text(v))
    print(f"Final configuration: {final_name} (threshold {frozen['thresholds'][final_name]['threshold']:g} dBm)")

    # ---- 4. Sensitivity -----------------------------------------------------------
    t0 = time.perf_counter()
    subset = [r for r in verification if r["index"] < n_sens]
    case_outs = {}
    for name, point, rules in cases:
        spec = {"mode": "cache" if name == "base" else "channel", "point": point, "configs": [final_name], "rules": rules}
        case_outs[name] = run_final_phase(config, subset, cache_dir, R, spec, jobs)
        print(f"  sensitivity {name}: {time.perf_counter() - t0:.0f} s since the start of the phase")
    timings["sensitivity"] = time.perf_counter() - t0
    results["sensitivity"] = evaluate_sensitivity(case_outs, final_name, frozen["thresholds"][final_name]["threshold"], config)
    # the base rebuilt from the channel must match the cache (bit for bit)
    check_runs = [r for r in subset if r["scenario"] == "none"][:1]
    a = process_run_final((check_runs[0], config, cache_dir, R, {"mode": "cache", "point": {}, "configs": [final_name], "rules": ["any"]}))
    b = process_run_final((check_runs[0], config, cache_dir, R, {"mode": "channel", "point": {}, "configs": [final_name], "rules": ["any"]}))
    same = all(np.array_equal(a["results"][(final_name, "any")][k], b["results"][(final_name, "any")][k]) for k in a["results"][(final_name, "any")])
    print("\n=== Sensitivity (first %d runs per verification scenario, same seeds at all points) ===" % n_sens)
    print(sensitivity_table(results))
    print(f"Check: base rebuilt from the channel = base from the cache (none run): {'identical' if same else 'DIFFERENT'}")

    timings["total"] = time.perf_counter() - t_start
    results["timings"] = timings
    print("\n=== Execution times ===")
    for k, val in timings.items():
        print(f"  {k:<14} {val:>8.1f} s")
    print(f"  total {timings['total'] / 60:.1f} min with {jobs} processes")
    save_pickle(out_dir / "results.pkl", results)
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()

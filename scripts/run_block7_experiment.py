#!/usr/bin/env python3
"""Block 7 experiment: routing to the group leader (bridge, DQREM, carry) on a long group.

Group of 10 nodes with p95 extent of 35 m (overrides of ``routing.scenario_overrides``), scenarios none / slowdown / stop, ``routing.n_runs`` runs each, leader drawn per run.
Compares `direct`, `best_rssi`, `dqrem` and `dqrem_no_slope` on the SAME runs; alarms and SOS come from the ground truth (the detector is not evaluated here). Writes ``results/block7/results.pkl``.
Channel profile (``--profile``, default ``routing.channel_profile``): ``heltec_run`` (levels of the October 13 Heltec run, the thesis operating point, output in ``results/block7/heltec_run``)
or ``nominal`` (10 dBm, first operating point, output in ``results/block7``). If the time estimate exceeds ``routing.time_budget_minutes`` it stops (``--force`` to continue). Config: config/default.yaml if not given.
"""

import sys
import time
from pathlib import Path

import numpy as np

from _cli import make_parser  # noqa: E402
from src.common.io import load_pickle, save_pickle  # noqa: E402
from src.common.paths import block_dir  # noqa: E402
from src.mobility import load_config  # noqa: E402
from src.routing import airtime, record_bytes, records_per_packet  # noqa: E402
from src.routing_experiment import (
    block7_config,
    calibration_text,
    connectivity_table,
    delivery_table,
    group_carry_text,
    hops_table,
    load_table,
    POSITIONS,
    predictive_verdict,
    process_run_routing,
    profile_comparison_text,
    range_text,
    routing_runs,
    run_all,
    select,
    separated_table,
    verdict_text,
    wrong_hops_text,
)


def main() -> None:
    parser = make_parser(__doc__, jobs=True, force_help="continue even if the estimate exceeds ``routing.time_budget_minutes``")
    parser.add_argument("--profile", default=None, help="nominal | heltec_run (default: routing.channel_profile)")
    parser.add_argument("--out-dir", default=None, help="default: results/block7 for nominal, results/block7/<profile> for the others")
    args = parser.parse_args()

    base = load_config(args.config)
    profile = args.profile or base["routing"]["channel_profile"]
    config = block7_config(base, profile)
    rcfg = config["routing"]
    default_out = block_dir(7, profile)
    out_dir, jobs = Path(args.out_dir) if args.out_dir else default_out, args.jobs
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    n_nodes = config["group"]["n_nodes"]
    strategies = list(rcfg["strategies"])
    sweep = list(rcfg["packet_bytes_sweep"])
    ref = (rcfg["reference_run"]["scenario"], rcfg["reference_run"]["index"])

    print(f"Channel profile: {profile}; equivalent power {config['channel']['tx_power']:g} dBm, sensitivity {config['packets']['reception']['sensitivity_dbm']:g} dBm, candidate threshold {rcfg['candidate_threshold']:g} dBm\n")

    # ---- Packets ---------------------------------------------------------------
    print(f"Record: {record_bytes(n_nodes)} bytes (8 + 5·{n_nodes - 1}); header {rcfg['packet_header_bytes']} bytes")
    for size in (rcfg["max_packet_bytes"], *sweep):
        cap = records_per_packet(rcfg, n_nodes, size)
        full = rcfg["packet_header_bytes"] + cap * record_bytes(n_nodes)
        print(f"  limit {size} bytes: {cap} records per packet ({full} bytes), airtime {1e3 * airtime(full, rcfg):.2f} ms + ack {1e3 * airtime(rcfg['airtime']['ack_bytes'], rcfg):.2f} ms")

    runs = routing_runs(config)
    print(f"\n{len(runs)} runs ({rcfg['n_runs']} per scenario: {', '.join(config['metrics']['scenarios'])}); strategies: {', '.join(strategies)}; {jobs} processes")
    print("Leader per run: " + ", ".join(f"{r['leader']}" for r in runs[: rcfg['n_runs']][:12]) + " ... (same across the three scenarios)")

    # ---- Time estimate on two probe runs (reused) ---------------------------------
    probes = [next(r for r in runs if r["scenario"] == "none"), next(r for r in runs if (r["scenario"], r["index"]) == ref)]
    done, t_probe = {}, []
    for r in probes:
        t0 = time.perf_counter()
        done[(r["scenario"], r["index"])] = process_run_routing((r, strategies, sweep, (r["scenario"], r["index"]) == ref))
        t_probe.append(time.perf_counter() - t0)
        print(f"  probe run {r['scenario']} {r['index']}: {t_probe[-1]:.1f} s (preparation {done[(r['scenario'], r['index'])]['t_prepare']:.1f} s)")
    per_run = done[ref]['elapsed']  # the second probe run already has the calibration in memory
    estimate = len(runs) * per_run / jobs / 60.0
    print(f"Estimate: {estimate:.1f} min (excluding the group calibration, about 1 min per process, the first time)")
    if estimate > rcfg["time_budget_minutes"] and not args.force:
        print(f"The estimate exceeds {rcfg['time_budget_minutes']} min: stopping (--force to continue).")
        sys.exit(2)

    # ---- Experiment --------------------------------------------------------------------
    t0 = time.perf_counter()
    outs = run_all(runs, strategies, sweep, jobs, ref, done)
    t_run = time.perf_counter() - t0
    print(f"Runs completed in {t_run:.0f} s")

    # ---- Report -----------------------------------------------------------------------
    print("\n=== Calibration ===")
    print(calibration_text(outs, config))
    print("\n=== Range ===")
    print(range_text(outs, config))
    print("\n=== AHP weights ===")
    for name, (w, cr) in outs[0]["weights"].items():
        print(f"  {name:<15} weights {np.array2string(w, precision=3)}  CR = {cr:.4f} (< 0.1: consistent)")
    print("\n=== Delivery of records to the leader, per strategy and scenario (nodes in the group | separated node) ===")
    print(delivery_table(outs, config))
    for pos, label in enumerate(POSITIONS):
        sub = select(outs, None, pos)
        print(f"\n--- leader in {label} ({len(sub)} runs out of {len(outs)}; thirds of the MEAN longitudinal position of the run) ---")
        print(delivery_table(sub, config))
    print("\n=== Connectivity to the leader (fraction of time with a path, also over multiple hops, per node rank: 0 tail ... head) ===")
    print(connectivity_table(outs, config))
    print("\n=== Telemetry of the nodes in the group that arrived via carry ===")
    print(group_carry_text(outs, config))
    print("\n=== Distribution of the hops of the delivered records (nodes in the group) ===")
    print(hops_table(outs, config))
    print("\n=== " + wrong_hops_text(outs))
    print("\n=== Separated node, oracle alarm and SOS ===")
    print(separated_table(outs, config))
    print("\n=== Load ===")
    print(load_table(outs, config))
    print("\n=== ESP-NOW v2 (1470 bytes), dqrem ===")
    for size in sweep:
        sub = [o["bytes"][size] for o in outs]
        d250 = np.mean([o["strategies"]["dqrem"]["airtime_per_s"].max() for o in outs])
        d1470 = np.mean([s["airtime_per_s"].max() for s in sub])
        deliv = sum(s["classes"]["group"]["delivered"] for s in sub) / sum(s["classes"]["group"]["created"] for s in sub)
        print(f"  {size} bytes: {sub[0]['capacity']} records per packet; group delivery {deliv:.4f}; packets/s max {np.mean([s['packets_per_s'].max() for s in sub]):.2f}"
              f" (250 bytes: {np.mean([o['strategies']['dqrem']['packets_per_s'].max() for o in outs]):.2f}); max airtime {100 * d1470:.2f}% (250 bytes: {100 * d250:.2f}%)")
    print("\n=== Does the predictive part pay off? ===")
    verdict = predictive_verdict(outs, config)
    print(verdict_text(verdict, config))

    nominal_pkl = block_dir(7) / "results.pkl"
    if profile != "nominal" and nominal_pkl.exists():
        ref_res = load_pickle(nominal_pkl)
        if "connectivity" in ref_res["outs"][0]:
            print("\n=== Side-by-side comparison nominal / " + profile + " (all runs) ===")
            print(profile_comparison_text({"nominal": ref_res["outs"], profile: outs}, {"nominal": ref_res["config"], profile: config}))
        else:
            print("\n(comparison with nominal not available: results/block7/results.pkl lacks the connectivity metrics; regenerate it with --profile nominal)")

    total = time.perf_counter() - t_start
    print(f"\n=== Timings === runs {t_run:.0f} s, total {total:.0f} s ({total / 60:.1f} min) with {jobs} processes")
    save_pickle(out_dir / "results.pkl", {"config": config, "profile": profile, "outs": outs, "verdict": verdict, "timings": {"runs": t_run, "total": total}})
    print(f"Results saved in {out_dir}")


if __name__ == "__main__":
    main()

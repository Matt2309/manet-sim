#!/usr/bin/env python3
"""Esperimento del Blocco 7: routing verso il capogruppo (ponte, DQREM, carry) su un gruppo lungo.

Gruppo di 10 nodi con estensione p95 di 35 m (sostituzioni di ``routing.scenario_overrides``), scenari none / slowdown / stop, ``routing.n_runs`` corse ciascuno, capogruppo estratto per corsa.
Confronta `direct`, `best_rssi`, `dqrem` e `dqrem_no_slope` sulle STESSE corse; allarmi e SOS vengono dalla ground truth (il rilevatore non si valuta qui). Scrive ``results/block7/results.pkl``.
Profilo di canale (``--profile``, default ``routing.channel_profile``): ``heltec_run`` (livelli della corsa Heltec del 13 ottobre, il punto di lavoro della tesi, uscita in ``results/block7/heltec_run``)
o ``nominal`` (10 dBm, primo punto di lavoro, uscita in ``results/block7``). Se la stima dei tempi supera ``routing.time_budget_minutes`` si ferma (``--force`` per proseguire). Config: config/default.yaml se non indicata.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.mobility import load_config  # noqa: E402
from src.refinement import load_pickle, save_pickle  # noqa: E402
from src.routing import records_per_packet, airtime, record_bytes  # noqa: E402
from src.routing_experiment import (  # noqa: E402
    POSITIONS,
    block7_config,
    calibration_text,
    connectivity_table,
    delivery_table,
    group_carry_text,
    hops_table,
    load_table,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default=str(_REPO_ROOT / "config" / "default.yaml"))
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="processi paralleli")
    parser.add_argument("--profile", default=None, help="nominal | heltec_run (default: routing.channel_profile)")
    parser.add_argument("--out-dir", default=None, help="default: results/block7 per nominal, results/block7/<profilo> per gli altri")
    parser.add_argument("--force", action="store_true", help="prosegue anche se la stima supera ``routing.time_budget_minutes``")
    args = parser.parse_args()

    base = load_config(args.config)
    profile = args.profile or base["routing"]["channel_profile"]
    config = block7_config(base, profile)
    rcfg = config["routing"]
    default_out = _REPO_ROOT / "results" / "block7" / ("" if profile == "nominal" else profile)
    out_dir, jobs = Path(args.out_dir) if args.out_dir else default_out, args.jobs
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    n_nodes = config["group"]["n_nodes"]
    strategies = list(rcfg["strategies"])
    sweep = list(rcfg["packet_bytes_sweep"])
    ref = (rcfg["reference_run"]["scenario"], rcfg["reference_run"]["index"])

    print(f"Profilo di canale: {profile}; potenza equivalente {config['channel']['tx_power']:g} dBm, sensibilità {config['packets']['reception']['sensitivity_dbm']:g} dBm, soglia dei candidati {rcfg['candidate_threshold']:g} dBm\n")

    # ---- Pacchetti ---------------------------------------------------------------
    print(f"Record: {record_bytes(n_nodes)} byte (8 + 5·{n_nodes - 1}); intestazione {rcfg['packet_header_bytes']} byte")
    for size in (rcfg["max_packet_bytes"], *sweep):
        cap = records_per_packet(rcfg, n_nodes, size)
        full = rcfg["packet_header_bytes"] + cap * record_bytes(n_nodes)
        print(f"  limite {size} byte: {cap} record per pacchetto ({full} byte), tempo d'aria {1e3 * airtime(full, rcfg):.2f} ms + conferma {1e3 * airtime(rcfg['airtime']['ack_bytes'], rcfg):.2f} ms")

    runs = routing_runs(config)
    print(f"\n{len(runs)} corse ({rcfg['n_runs']} per scenario: {', '.join(config['metrics']['scenarios'])}); strategie: {', '.join(strategies)}; {jobs} processi")
    print("Capogruppo per corsa: " + ", ".join(f"{r['leader']}" for r in runs[: rcfg['n_runs']][:12]) + " ... (stesso nei tre scenari)")

    # ---- Stima dei tempi su due corse di prova (riusate) ---------------------------------
    probes = [next(r for r in runs if r["scenario"] == "none"), next(r for r in runs if (r["scenario"], r["index"]) == ref)]
    done, t_probe = {}, []
    for r in probes:
        t0 = time.perf_counter()
        done[(r["scenario"], r["index"])] = process_run_routing((r, strategies, sweep, (r["scenario"], r["index"]) == ref))
        t_probe.append(time.perf_counter() - t0)
        print(f"  corsa di prova {r['scenario']} {r['index']}: {t_probe[-1]:.1f} s (preparazione {done[(r['scenario'], r['index'])]['t_prepare']:.1f} s)")
    per_run = done[ref]['elapsed']  # la seconda corsa di prova ha già la calibrazione in memoria
    estimate = len(runs) * per_run / jobs / 60.0
    print(f"Stima: {estimate:.1f} min (esclusa la calibrazione del gruppo, circa 1 min per processo, la prima volta)")
    if estimate > rcfg["time_budget_minutes"] and not args.force:
        print(f"La stima supera {rcfg['time_budget_minutes']} min: mi fermo (--force per proseguire).")
        sys.exit(2)

    # ---- Esperimento --------------------------------------------------------------------
    t0 = time.perf_counter()
    outs = run_all(runs, strategies, sweep, jobs, ref, done)
    t_run = time.perf_counter() - t0
    print(f"Corse completate in {t_run:.0f} s")

    # ---- Resoconto -----------------------------------------------------------------------
    print("\n=== Calibrazione ===")
    print(calibration_text(outs, config))
    print("\n=== Portata ===")
    print(range_text(outs, config))
    print("\n=== Pesi AHP ===")
    for name, (w, cr) in outs[0]["weights"].items():
        print(f"  {name:<15} pesi {np.array2string(w, precision=3)}  CR = {cr:.4f} (< 0,1: consistente)")
    print("\n=== Consegna dei record al capogruppo, per strategia e scenario (nodi nel gruppo | nodo separato) ===")
    print(delivery_table(outs, config))
    for pos, label in enumerate(POSITIONS):
        sub = select(outs, None, pos)
        print(f"\n--- capogruppo in {label} ({len(sub)} corse su {len(outs)}; terzi della posizione longitudinale MEDIA della corsa) ---")
        print(delivery_table(sub, config))
    print("\n=== Connettività verso il capogruppo (frazione di tempo con una strada, anche a più salti, per rango del nodo: 0 coda ... testa) ===")
    print(connectivity_table(outs, config))
    print("\n=== Telemetria dei nodi nel gruppo arrivata via carry ===")
    print(group_carry_text(outs, config))
    print("\n=== Distribuzione dei salti dei record consegnati (nodi nel gruppo) ===")
    print(hops_table(outs, config))
    print("\n=== " + wrong_hops_text(outs))
    print("\n=== Nodo separato, allarme oracolo e SOS ===")
    print(separated_table(outs, config))
    print("\n=== Carico ===")
    print(load_table(outs, config))
    print("\n=== ESP-NOW v2 (1470 byte), dqrem ===")
    for size in sweep:
        sub = [o["bytes"][size] for o in outs]
        d250 = np.mean([o["strategies"]["dqrem"]["airtime_per_s"].max() for o in outs])
        d1470 = np.mean([s["airtime_per_s"].max() for s in sub])
        deliv = sum(s["classes"]["group"]["delivered"] for s in sub) / sum(s["classes"]["group"]["created"] for s in sub)
        print(f"  {size} byte: {sub[0]['capacity']} record per pacchetto; consegna gruppo {deliv:.4f}; pacchetti/s max {np.mean([s['packets_per_s'].max() for s in sub]):.2f}"
              f" (250 byte: {np.mean([o['strategies']['dqrem']['packets_per_s'].max() for o in outs]):.2f}); tempo d'aria max {100 * d1470:.2f}% (250 byte: {100 * d250:.2f}%)")
    print("\n=== La parte predittiva ripaga? ===")
    verdict = predictive_verdict(outs, config)
    print(verdict_text(verdict, config))

    nominal_pkl = _REPO_ROOT / "results" / "block7" / "results.pkl"
    if profile != "nominal" and nominal_pkl.exists():
        ref_res = load_pickle(nominal_pkl)
        if "connectivity" in ref_res["outs"][0]:
            print("\n=== Confronto fianco a fianco nominal / " + profile + " (tutte le corse) ===")
            print(profile_comparison_text({"nominal": ref_res["outs"], profile: outs}, {"nominal": ref_res["config"], profile: config}))
        else:
            print("\n(confronto con nominal non disponibile: results/block7/results.pkl non ha le metriche di connettività; rigenerarlo con --profile nominal)")

    total = time.perf_counter() - t_start
    print(f"\n=== Tempi === corse {t_run:.0f} s, totale {total:.0f} s ({total / 60:.1f} min) con {jobs} processi")
    save_pickle(out_dir / "results.pkl", {"config": config, "profile": profile, "outs": outs, "verdict": verdict, "timings": {"runs": t_run, "total": total}})
    print(f"Risultati salvati in {out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Esperimento del Blocco 6: soglie congelate, valutazione su corse nuove, verdetto media/migliore, sensibilità.

1. taratura (150 corse dei Blocchi 4-5, chiavi 6-7): sweep della soglia per `media`, `migliore`, `coppia`, scelta del punto di lavoro, scrittura di
   ``results/block6/frozen_thresholds.yaml``;
2. verifica (``final.n_runs`` corse per scenario, chiavi 8-9): valutazione alle soglie LETTE dal file;
3. verdetto media contro migliore (``results/block6/verdict.yaml``);
4. sensibilità (prime ``final.sensitivity_runs`` corse di verifica; cambia solo il Blocco 3) della configurazione finale.
Se ``frozen_thresholds.yaml`` esiste non viene riscritto: la taratura deve ridare le stesse soglie (altrimenti errore; ``--refreeze`` per riscriverlo).
Config: config/default.yaml se non indicata.
"""

import argparse
import datetime
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.experiment import pooled_fading_variance  # noqa: E402
from src.final import (  # noqa: E402
    LOST,
    decide_verdict,
    evaluate_at,
    evaluate_sensitivity,
    evaluation_table,
    freeze_thresholds,
    _git,
    load_frozen,
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
from src.refinement import SEPARATED, load_pickle, save_pickle  # noqa: E402
from src.simulator import ensure_all_cached  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default=str(_REPO_ROOT / "config" / "default.yaml"))
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="processi paralleli")
    parser.add_argument("--cache-dir", default=str(_REPO_ROOT / "results" / "cache"))
    parser.add_argument("--out-dir", default=str(_REPO_ROOT / "results" / "block6"))
    parser.add_argument("--force", action="store_true", help="prosegue anche se la stima supera ``final.time_budget_minutes``")
    parser.add_argument("--refreeze", action="store_true", help="riscrive frozen_thresholds.yaml dalla taratura")
    args = parser.parse_args()

    config = load_config(args.config)
    cache_dir, out_dir, jobs = Path(args.cache_dir), Path(args.out_dir), args.jobs
    out_dir.mkdir(parents=True, exist_ok=True)
    fcfg = config["final"]
    timings = {}
    t_start = time.perf_counter()

    tuning, verification = run_sets(config)
    seeds_t, seeds_v = {r["seed"] for r in tuning}, {r["seed"] for r in verification}
    assert not seeds_t & seeds_v, "semi in comune fra taratura e verifica"
    n_sens = fcfg["sensitivity_runs"]
    print(
        f"Taratura: {len(tuning)} corse ({config['metrics']['n_runs']} per scenario); verifica: {len(verification)} corse"
        f" ({fcfg['n_runs']} per scenario); sensibilità: prime {n_sens} per scenario; {jobs} processi"
    )

    # ---- Cache ---------------------------------------------------------------
    t0 = time.perf_counter()
    ensure_all_cached(tuning, cache_dir, jobs)
    ensure_all_cached(verification, cache_dir, jobs, with_channel=lambda r: r["index"] < n_sens)
    timings["cache"] = time.perf_counter() - t0
    R = measurement_noise(config, pooled_fading_variance(tuning, cache_dir))
    print(f"R = {R:.3f} dB² (dalla taratura); cache: {timings['cache']:.1f} s")

    # ---- Stima dei tempi -------------------------------------------------------
    spec_main = {"mode": "cache", "point": {}, "configs": list(fcfg["configs"]), "rules": ["any"]}
    cases = sensitivity_cases(config)
    pess = next(p for n, p, _ in cases if n == "pessimista")
    spec_probe = {"mode": "channel", "point": pess, "configs": ["media"], "rules": ["any", "not_deaf", "not_deaf_episode"]}
    probe = [next(r for r in verification if r["scenario"] == s) for s in ("none", "slowdown")]
    process_run_final((probe[0], config, cache_dir, R, {**spec_main, "configs": ["media"]}))  # compilazione numba
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
        f"Una corsa none: {t_main[0]:.1f} s (3 configurazioni), con distacco {t_main[1]:.1f} s; sensibilità: {t_sens[0]:.1f} s / {t_sens[1]:.1f} s."
        f" Stima: taratura + verifica {est_main:.1f} min, sensibilità {est_sens:.1f} min, totale circa {est_main + est_sens:.1f} min (cache esclusa)"
    )
    if est_main + est_sens > fcfg["time_budget_minutes"] and not args.force:
        print(f"La stima supera {fcfg['time_budget_minutes']} min: mi fermo (--force per proseguire).")
        sys.exit(2)

    # ---- 1. Taratura e soglie congelate ---------------------------------------
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
                    f"La taratura dà per {name} la soglia {f['threshold']} ma il file congelato ha {frozen['thresholds'][name]['threshold']}:"
                    f" il file non viene riscritto (usa --refreeze se la modifica è voluta)."
                )
        print(f"Soglie congelate lette da {frozen_path} (la taratura le riconferma)")
    else:
        write_frozen(frozen_path, chosen, config, R)
        frozen = load_frozen(frozen_path)
        print(f"Soglie congelate scritte in {frozen_path}")
    print(f"\n=== Soglie congelate (taratura: {frozen['date']}, commit {frozen['commit']}{' + modifiche' if frozen['dirty'] else ''}, release_time {frozen['release_time']:g} s) ===")
    print(f"{'config':<9} {'ambito':>9} {'regola':>13} {'soglia dBm':>10} {'P_d slowdown':>13} {'P_d stop':>9} {'FA/h':>6} {'in allarme':>10}")
    for name, f in frozen["thresholds"].items():
        t = f["tuning"]
        print(
            f"{name:<9} {f['scope']:>9} {f['fusion_rule']:>13} {f['threshold']:>10.1f} {t['p_detect_slowdown']:>13.2f} {t['p_detect_stop']:>9.2f}"
            f" {t['fa_per_hour']:>6.2f} {100 * t['active_fraction']:>9.2f}%"
        )
    block5 = _REPO_ROOT / "results" / "block5" / "results.pkl"
    if block5.exists():
        s5 = load_pickle(block5)["summaries"].get(("C:release=30", "fused", 1))
        m = frozen["thresholds"]["media"]
        if s5 is not None and s5["wp"] is not None:
            print(
                f"Controllo con il Blocco 5 (C:release=30): soglia {s5['wp']['threshold']:.1f} dBm, P_d {s5['point']['slowdown']['p_detect']:.2f}/"
                f"{s5['point']['stop']['p_detect']:.2f}, FA/h {s5['wp']['fa_per_hour']:.2f} -> "
                + ("coincide" if abs(s5["wp"]["threshold"] - m["threshold"]) < 1e-9 else "DIVERSO")
            )

    # ---- 2. Verifica -------------------------------------------------------------
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

    # ---- 3. Verdetto -------------------------------------------------------------
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
            {"date": datetime.datetime.now().isoformat(timespec="seconds"), "commit": _git("rev-parse", "HEAD"), **v,
             "final": final_name, "final_threshold": frozen["thresholds"][final_name]["threshold"]},
            f, sort_keys=False, allow_unicode=True,
        )  # fmt: skip
    results = {
        "config": config, "R": R, "frozen": frozen, "tuning": tuning_eval, "verification": verification_eval, "verdict": v,
        "final_name": final_name, "sweeps": sweeps, "diag_tuning": None,
    }  # fmt: skip
    print("\n=== Valutazione sulle corse di verifica (soglie congelate; ottimismo = verifica meno taratura) ===")
    results["sensitivity"] = {}
    print(evaluation_table(results))
    print("\n=== Verdetto media contro migliore ===")
    print(verdict_text(v))
    print(f"Configurazione finale: {final_name} (soglia {frozen['thresholds'][final_name]['threshold']:g} dBm)")

    # ---- 4. Sensibilità -----------------------------------------------------------
    t0 = time.perf_counter()
    subset = [r for r in verification if r["index"] < n_sens]
    case_outs = {}
    for name, point, rules in cases:
        spec = {"mode": "cache" if name == "base" else "channel", "point": point, "configs": [final_name], "rules": rules}
        case_outs[name] = run_final_phase(config, subset, cache_dir, R, spec, jobs)
        print(f"  sensibilità {name}: {time.perf_counter() - t0:.0f} s dall'inizio della fase")
    timings["sensitivity"] = time.perf_counter() - t0
    results["sensitivity"] = evaluate_sensitivity(case_outs, final_name, frozen["thresholds"][final_name]["threshold"], config)
    # la base ricostruita dal canale deve coincidere con la cache (bit per bit)
    check_runs = [r for r in subset if r["scenario"] == "none"][:1]
    a = process_run_final((check_runs[0], config, cache_dir, R, {"mode": "cache", "point": {}, "configs": [final_name], "rules": ["any"]}))
    b = process_run_final((check_runs[0], config, cache_dir, R, {"mode": "channel", "point": {}, "configs": [final_name], "rules": ["any"]}))
    same = all(np.array_equal(a["results"][(final_name, "any")][k], b["results"][(final_name, "any")][k]) for k in a["results"][(final_name, "any")])
    print("\n=== Sensibilità (prime %d corse per scenario di verifica, stessi semi in tutti i punti) ===" % n_sens)
    print(sensitivity_table(results))
    print(f"Controllo: base ricostruita dal canale = base dalla cache (corsa none): {'identica' if same else 'DIVERSA'}")

    timings["total"] = time.perf_counter() - t_start
    results["timings"] = timings
    print("\n=== Tempi di esecuzione ===")
    for k, val in timings.items():
        print(f"  {k:<14} {val:>8.1f} s")
    print(f"  totale {timings['total'] / 60:.1f} min con {jobs} processi")
    save_pickle(out_dir / "results.pkl", results)
    print(f"\nRisultati salvati in {out_dir}")


if __name__ == "__main__":
    main()

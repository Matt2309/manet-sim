#!/usr/bin/env python3
"""Esperimento del Blocco 5: rilevatore rifinito (varianti A-F).

Corse in cache (`results/cache/`, 50 per scenario, stessi semi, nodi e istanti del Blocco 4). Ogni variante
(regola di fusione, livello relativo, spegnimento lento, robustezza del Kalman) è confrontata con la base
a coppie, sulle stesse corse; poi si combinano le vincenti e si ripete con lo shadowing indipendente.
Salva in ``results/block5/`` e stampa i resoconti. Config: config/default.yaml se non indicata.
"""

import argparse
import itertools
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.common.parallel import parallel_map  # noqa: E402
from src.common.paths import block_dir  # noqa: E402
from src.experiment import pooled_fading_variance  # noqa: E402
from src.kalman import measurement_noise  # noqa: E402
from src.mobility import load_config  # noqa: E402
from src.refinement import block4_config  # noqa: E402
from src.refinement import (  # noqa: E402
    SEPARATED,
    _kalman_key,
    apply_overrides,
    baseline_diagnostics,
    choose,
    combination_variant,
    gain_loss,
    phase1_variants,
    process_run,
    run_phase,
    save_pickle,
    summarize_variant,
    wrong_off_runs,
)
from src.simulator import cache_path, ensure_all_cached, experiment_runs  # noqa: E402


# ---------------------------------------------------------------------------
# Stampa
# ---------------------------------------------------------------------------

HEADER = (
    f"{'variante':<26} {'soglia':>6} {'P_d slowdown [Wilson]':>22} {'P_d stop [Wilson]':>22} {'dist.med/p90 slow':>18}"
    f" {'dist.med/p90 stop':>18} {'FA/h [Poisson]':>18} {'%all.':>6} {'guad/perse sl':>13} {'st':>7} {'tot':>7}"
    f" {'torti: corse/n (dist.)':>24}"
)


def _pd(r):
    return f"{r['p_detect']:.2f} [{r['p_lo']:.2f}-{r['p_hi']:.2f}]"


def _label(s):
    return f"{s['name']} mo{s['min_observers']} {s['scope'][:4]}"


def print_row(s, base):
    if s["wp"] is None:
        print(f"{_label(s):<26}  nessun punto ammissibile")
        return
    sl, st = s["point"]["slowdown"], s["point"]["stop"]
    gl = gain_loss(s, base)
    g = lambda k: f"+{gl[k][0]}/-{gl[k][1]}"  # noqa: E731
    wrong = sum(int(s["per_run"][c]["n_wrong_off"].sum()) for c in SEPARATED)
    dist = sum(float(s["per_run"][c]["wrong_off_dist_sum"].sum()) for c in SEPARATED)
    wd = f"{dist / wrong:.0f} m" if wrong else "-"
    print(
        f"{_label(s):<26} {s['wp']['threshold']:>6.1f} {_pd(sl):>22} {_pd(st):>22}"
        f" {sl['distance_median']:>8.1f}/{sl['distance_p90']:<9.1f} {st['distance_median']:>8.1f}/{st['distance_p90']:<9.1f}"
        f" {sl['fa_per_hour']:>5.2f} [{sl['fa_lo']:.2f}-{sl['fa_hi']:.2f}] {100 * sl['active_fraction']:>6.2f}"
        f" {g('slowdown'):>13} {g('stop'):>7} {g('total'):>7} {wrong_off_runs(s):>9}/{wrong_off_total(s):<3d} ({wd})"
    )


def wrong_off_total(s):
    return int(sum(s["per_run"][c]["n_wrong_off"].sum() for c in SEPARATED))


def print_table(title, items, base):
    print(f"\n=== {title} ===")
    print(HEADER)
    for s in items:
        print_row(s, base)


def print_reference_runs(items, config):
    print("\n=== Corse di riferimento (scenario, indice): esito e distanza al rilevamento ===")
    refs = [(sc, i) for sc, idxs in config["metrics"]["reference_runs"].items() for i in idxs]
    print(f"{'variante':<26} " + " ".join(f"{sc + ' ' + str(i):>26}" for sc, i in refs))
    for s in items:
        if s["wp"] is None:
            continue
        cells = []
        for sc, i in refs:
            pr = s["per_run"][sc]
            d = pr["distance"][i]
            cells.append(f"{'ok  ' if pr['success'][i] else 'MISS'} dist. {d:6.1f} m rit. {pr['delay'][i]:6.1f} s")
        print(f"{_label(s):<26} " + " ".join(f"{c:>26}" for c in cells))


def print_pair_details(items, base):
    """Indici delle corse guadagnate e perse rispetto alla base (per scenario)."""
    print("\n=== Corse guadagnate (+) e perse (-) rispetto alla base, per indice di corsa ===")
    for s in items:
        if s["wp"] is None or s is base:
            continue
        parts = []
        for c in SEPARATED:
            v, b = s["per_run"][c]["success"].astype(bool), base["per_run"][c]["success"].astype(bool)
            gained, lost = np.flatnonzero(v & ~b).tolist(), np.flatnonzero(~v & b).tolist()
            if gained or lost:
                parts.append(f"{c}: +{gained} -{lost}")
        print(f"{_label(s):<26} " + ("; ".join(parts) if parts else "nessuna differenza"))


def print_release(items, base):
    print("\n=== C: spegnimenti a torto e ripassaggi (corse con distacco) ===")
    print(f"{'variante':<26} {'sc.':>8} {'corse con spegn.':>16} {'n totale':>9} {'dist. media/min (m)':>20} {'ripassaggi':>10} {'allarme spento':>14} {'acceso a fine':>13}")
    for s in items:
        if s["wp"] is None:
            continue
        for c in SEPARATED:
            pr = s["per_run"][c]
            n = int(pr["n_wrong_off"].sum())
            mean = pr["wrong_off_dist_sum"].sum() / n if n else float("nan")
            dmin = np.nanmin(pr["wrong_off_dist_min"]) if n else float("nan")
            rp = pr["repass"].astype(bool)
            print(
                f"{_label(s):<26} {c:>8} {int((pr['n_wrong_off'] > 0).sum()):>16d} {n:>9d} {mean:>12.1f}/{dmin:<6.1f} {int(rp.sum()):>10d}"
                f" {int((rp & pr['repass_off'].astype(bool)).sum()):>14d} {int(pr['alarm_on_end'].astype(bool).sum()):>13d}"
            )


def nis_report(outs, config, variants):
    """NIS media e percentile nel gruppo per ogni Kalman distinto."""
    q = config["refinement"]["nis_quantile"]
    names = {}
    for v in variants:
        vcfg = apply_overrides(config, v["overrides"])
        k = _kalman_key(vcfg)
        label = "nessuna" if vcfg["kalman"]["robust"] == "none" else (
            f"inflate x{vcfg['kalman']['r_inflation']:g}" if vcfg["kalman"]["robust"] == "inflate" else f"nis_gate {vcfg['kalman']['nis_gate']:g}"
        )
        names[k] = label
    stats, hists = {}, {}
    edges = np.arange(*[config["metrics"]["nis_histogram"][0], config["metrics"]["nis_histogram"][1] + 1e-9, config["metrics"]["nis_histogram"][2]])
    print("\n=== D: NIS nel gruppo senza distacco (attesa: media 1) ===")
    print(f"{'robustezza':<16} {'NIS media':>10} {'p' + format(100 * q, 'g'):>8} {'campioni':>10} {'oltre ' + format(edges[-1], 'g'):>10}")
    for k, label in names.items():
        nis = np.concatenate([o["nis"][k] for o in outs if o["scenario"] == "none"]).astype(np.float64)
        stats[label] = {"mean": float(nis.mean()), "q": float(np.quantile(nis, q)), "n": len(nis)}
        hists[label] = np.histogram(nis, bins=edges)[0]
        print(f"{label:<16} {nis.mean():>10.3f} {np.quantile(nis, q):>8.2f} {len(nis):>10d} {int((nis >= edges[-1]).sum()):>10d}")
    return {"stats": stats, "hist": hists, "edges": edges}


def print_corr(summaries):
    print("\n=== A: correlazione fra gli allarmi degli osservatori sullo stesso bersaglio (corse none, al punto di lavoro, min_observers = 1) ===")
    print(f"{'variante':<22} {'ambito':>9} {'phi medio':>10} {'P(>=2 oss. | >=1)':>18} {'terne':>6}")
    for (name, scope, mo), s in summaries.items():
        if mo != 1 or s["corr"] is None:
            continue
        c = s["corr"]
        print(f"{name:<22} {scope:>9} {c['phi']:>10.3f} {c['p_two_given_one']:>18.3f} {c['n_triples']:>6d}")


# ---------------------------------------------------------------------------
# Scelta e combinazione
# ---------------------------------------------------------------------------


def strip(variant):
    """Sostituzioni di una variante senza il ``sigma_a`` della base."""
    return {k: v for k, v in variant["overrides"].items() if k != "kalman.sigma_a"}


def decide(summaries, variants_by_name, config):
    """Applica la regola di scelta a A, B, C, D; restituisce (componenti vincenti, resoconto)."""
    min_net = config["refinement"]["min_net_gain_runs"]
    mos = config["detector"]["min_observers_sweep"]
    rules = config["fusion"]["rule_sweep"]
    S = lambda name, mo=1: summaries[(name, "fused", mo)]  # noqa: E731
    base = S("base")
    notes = []

    cand_a = [S("base" if r == "weighted_mean" else f"A:{r}", mo) for mo in mos for r in rules if (r, mo) != ("weighted_mean", 1)]
    best_a = choose(cand_a, base, min_net)
    best_b = choose([S("B:relative")], base, min_net)
    base_rt = config["detector"]["release_time"]
    cand_c = [S(f"C:release={rt:g}") for rt in config["detector"]["release_time_sweep"] if rt != base_rt]
    best_c = choose(cand_c, base, min_net)
    labels = [f"inflate={r:g}" for r in config["kalman"]["r_inflation_sweep"]] + ["gate"]
    best_d_abs = choose([S(f"D:{lb}") for lb in labels], base, min_net)
    best_d_rel = choose([S(f"D:{lb}+rel") for lb in labels], S("B:relative"), min_net)

    parts = {}
    if best_a is not base:
        rule = "weighted_mean" if best_a["name"] == "base" else best_a["name"].split(":", 1)[1]
        parts["A"] = {"fusion.rule": rule, "_mo": best_a["min_observers"]}
    if best_b is not base:
        parts["B"] = {"detector.reference": "relative"}
    if best_c is not base:
        parts["C"] = strip(variants_by_name[best_c["name"]])
    d = {
        "absolute": None if best_d_abs is base else strip(variants_by_name[best_d_abs["name"]]),
        "relative": None if best_d_rel is S("B:relative") else {k: v for k, v in strip(variants_by_name[best_d_rel["name"]]).items() if k != "detector.reference"},
    }
    if d["absolute"] or d["relative"]:
        parts["D"] = d
    d_abs = d["absolute"]
    if d_abs:
        d_abs = {k: v for k, v in d_abs.items() if k != "detector.reference"}
        d["absolute"] = d_abs
    notes.append(f"A -> {_label(best_a)}; B -> {best_b['name']}; C -> {best_c['name']}; D assoluto -> {best_d_abs['name']}; D relativo -> {best_d_rel['name']}")
    singles = {"A": best_a, "B": best_b, "C": best_c, "D": best_d_abs if best_d_abs is not base else best_d_rel}
    return parts, notes, singles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default=str(_REPO_ROOT / "config" / "default.yaml"))
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="processi paralleli")
    parser.add_argument("--cache-dir", default=str(_REPO_ROOT / "results" / "cache"))
    parser.add_argument("--out-dir", default=str(block_dir(5)))
    parser.add_argument("--force", action="store_true", help="prosegue anche se la stima supera ``refinement.time_budget_minutes``")
    args = parser.parse_args()

    config = block4_config(load_config(args.config))  # Blocco 6: i default sono cambiati, qui valgono quelli del Blocco 4
    cache_dir, out_dir = Path(args.cache_dir), Path(args.out_dir)
    rcfg = config["refinement"]
    min_net = rcfg["min_net_gain_runs"]
    jobs = args.jobs
    t_start = time.perf_counter()
    timings = {}

    runs = experiment_runs(config)
    print(f"{len(runs)} corse ({config['metrics']['n_runs']} per scenario: {config['metrics']['scenarios']}), {jobs} processi")
    t0 = time.perf_counter()
    ensure_all_cached(runs, cache_dir, jobs)
    timings["cache_field"] = time.perf_counter() - t0
    R = measurement_noise(config, pooled_fading_variance(runs, cache_dir))
    print(f"R = {R:.3f} dB² (come nel Blocco 4); cache: {timings['cache_field']:.1f} s")

    variants1 = phase1_variants(config)
    variants_by_name = {v["name"]: v for v in variants1}

    # stima del tempo, su una corsa none e una slowdown (scalda anche numba)
    probe = [next(r for r in runs if r["scenario"] == s) for s in ("none", "slowdown")]
    process_run((probe[0], config, cache_dir, variants1[:1], R))  # compilazione
    t_probe = []
    for r in probe:
        t0 = time.perf_counter()
        process_run((r, config, cache_dir, variants1, R))
        t_probe.append(time.perf_counter() - t0)
    n_none = config["metrics"]["n_runs"]
    est_phase1 = (t_probe[0] * n_none + t_probe[1] * n_none * 2) / jobs / 60.0
    te = rcfg["time_estimate"]
    est_total = est_phase1 * te["factor"] + te["extra_minutes"]  # fase F ed E + simulazione delle corse independent (circa 3 min con 8 processi)
    print(
        f"Una corsa none con {len(variants1)} varianti: {t_probe[0]:.1f} s; con distacco: {t_probe[1]:.1f} s."
        f" Stima: fase A-D {est_phase1:.1f} min, totale circa {est_total:.1f} min"
    )
    if est_total > rcfg["time_budget_minutes"] and not args.force:
        print(f"La stima supera {rcfg['time_budget_minutes']} min: mi fermo (--force per proseguire).")
        sys.exit(2)

    # ---- Fase 1: A-D -------------------------------------------------------
    t0 = time.perf_counter()
    outs1 = run_phase(config, runs, cache_dir, variants1, R, jobs)
    timings["phase1"] = time.perf_counter() - t0
    summaries = {}
    for v in variants1:
        for scope in v["scopes"]:
            for mo in v["min_observers"]:
                summaries[(v["name"], scope, mo)] = summarize_variant(outs1, v, scope, mo, config)
    base = summaries[("base", "fused", 1)]
    print(f"Fase A-D: {timings['phase1']:.1f} s")

    def group(g, scope="fused"):
        return [s for (n, sc, mo), s in summaries.items() if s["group"] == g and sc == scope]

    print(f"\nBase: modalità level, ambito fused, sigma_a {rcfg['base']['sigma_a']}, regola weighted_mean, soglia {base['wp']['threshold']:.1f} dBm")
    print(f"Regola di scelta: >= {min_net} corse nette (slowdown + stop), o a P_d equivalente >= {min_net} corse in meno con spegnimenti a torto.")
    print_table("Base e A: regole di fusione x min_observers (ambito fused)", [s for s in group("base") + group("A")], base)
    print_table("Base e A: ambito pairwise (riferimento)", [s for s in group("base", "pairwise") + group("A", "pairwise")], base)
    print_corr(summaries)
    print_table("B: livello relativo", group("B") + group("B", "pairwise"), base)
    print_table("C: spegnimento lento (release_time)", [base] + group("C"), base)
    print_release([base] + group("C"), base)
    print_table("D: robustezza del Kalman (assoluto e relativo)", [base] + [s for s in group("D") if not s["name"].endswith("+rel")] + [summaries[("B:relative", "fused", 1)]] + [s for s in group("D") if s["name"].endswith("+rel")], base)
    nis = nis_report(outs1, config, variants1)

    parts, notes, singles = decide(summaries, variants_by_name, config)
    print("\n=== Scelte (regola del metodo) ===")
    for n in notes:
        print(n)
    print(f"Componenti vincenti: {list(parts) or 'nessuno: resta la base'}")

    # ---- Fase 2: F ---------------------------------------------------------
    final_name = "base"
    variants2, final_variant = [], None
    if parts:
        keys = list(parts)
        final_variant = combination_variant("final", parts, config, ("fused", "pairwise"))
        variants2.append(final_variant)
        for size in range(2, len(keys)):
            for subset in itertools.combinations(keys, size):
                v = combination_variant("F:" + "+".join(subset), {k: parts[k] for k in subset}, config, ("fused",))
                variants2.append(v)
        final_name = "final"
        print(f"\nFase F: {len(variants2)} varianti combinate ({', '.join(v['name'] for v in variants2)})")
        t0 = time.perf_counter()
        outs2 = run_phase(config, runs, cache_dir, variants2, R, jobs)
        timings["phase2"] = time.perf_counter() - t0
        for v in variants2:
            for scope in v["scopes"]:
                for mo in v["min_observers"]:
                    summaries[(v["name"], scope, mo)] = summarize_variant(outs2, v, scope, mo, config)
        variants_by_name.update({v["name"]: v for v in variants2})
        print(f"Fase F: {timings['phase2']:.1f} s")
    final_mo = final_variant["min_observers"][0] if final_variant else 1
    final = summaries[(final_name, "fused", final_mo)]

    if parts:
        print_table("F: combinazioni", [summaries[(v["name"], "fused", v["min_observers"][0])] for v in variants2 if v["name"] != "final"] + [final], base)
        print("\nCombinazioni contro le singole componenti (corse nette della combinazione rispetto a ciascuna componente da sola):")
        for v in variants2:
            keys_v = [k for k in parts if (v["name"] == "final" or k in v["name"][2:].split("+"))]
            combo = summaries[(v["name"], "fused", v["min_observers"][0])]
            nets = {}
            for k in keys_v:
                single = singles[k]
                g = gain_loss(combo, single)
                nets[k] = (g["total"][0] - g["total"][1]) if g else None
            worse = all(n is not None and n < 0 for n in nets.values()) and len(nets) > 1
            print(f"  {v['name']:<12} nette vs {nets}{'  <-- PEGGIORA rispetto a ciascuna componente' if worse else ''}")

    print("\n=== F: configurazione finale ===")
    if parts:
        final_cfg = apply_overrides(config, final_variant["overrides"])
        for k, v in sorted(final_variant["overrides"].items()):
            print(f"  {k} = {v}")
        print(f"  detector.min_observers = {final_mo}; detector.mode = level; ambito fused")
        _ = final_cfg
    else:
        print("  nessuna variante batte la base oltre il rumore: configurazione finale = base")
    print_table("Finale contro la base", [base, final], base)
    print_reference_runs([base, final] + [s for k, s in summaries.items() if s["group"] in ("A", "B", "C", "D", "F") and k[1] == "fused"], config)

    print_pair_details([s for k, s in summaries.items() if k[1] == "fused" and s["group"] in ("A", "B", "C", "D", "F")], base)

    # ---- Diagnostica della linea di base (corse none) ------------------------
    diag = {}
    for name in ("B:relative", "final"):
        s = summaries.get((name, "fused", 1)) or summaries.get((name, "fused", final_mo))
        v = variants_by_name.get(name)
        if s is None or v is None or s["wp"] is None or mode_of(v, config) != "drop":
            continue
        tasks = [(r, config, cache_dir, v, s["wp"]["threshold"], R, "fused") for r in runs if r["scenario"] == "none"]
        t0 = time.perf_counter()
        res = parallel_map(baseline_diagnostics, tasks, jobs)
        timings[f"diag_{name}"] = time.perf_counter() - t0
        hist = sum(r["hist"] for r in res)
        edges = res[0]["edges"]
        total = hist.sum() + sum(r["over_hi"] for r in res)
        cum = np.cumsum(hist) / max(total, 1)
        pct = lambda q: edges[1:][int(np.searchsorted(cum, q))]  # noqa: E731
        frozen = sum(r["frozen_time"] for r in res) / sum(r["time"] for r in res)
        longest = max(res, key=lambda r: r["max_freeze"])
        worst_pair = max((p for r in res for p in r["pairs"]), key=lambda p: p[2])
        diag[name] = {"hist": hist, "edges": edges, "frozen_fraction": frozen, "max_freeze": longest["max_freeze"], "per_run": res}
        print(f"\n=== B: diagnostica della linea di base, {name}, corse none, soglia {s['wp']['threshold']:.0f} dB ===")
        print(f"  calo fuso (dB): mediana {pct(0.5):.0f}, p90 {pct(0.9):.0f}, p99 {pct(0.99):.0f}, p99.9 {pct(0.999):.0f}, massimo bin {edges[1:][np.flatnonzero(hist)[-1]]:.0f}")
        print(f"  tempo con almeno una base congelata: {100 * frozen:.2f}% (coppie osservatore-bersaglio, dopo il warmup)")
        print(f"  congelamento più lungo di un link: {longest['max_freeze']:.1f} s; coppia peggiore (oss {worst_pair[0]}, bers {worst_pair[1]}): {worst_pair[2]:.1f} s")
        print(f"  tempo in pre-allarme: {100 * sum(r['state_time'] for r in res) / sum(r['time'] for r in res):.3f}%")

    # ---- Fase 3: E, shadowing indipendente -----------------------------------
    shadow = {}
    base_v = variants_by_name["base"]
    shadow_variants = [make_both(base_v)] + ([final_variant] if final_variant else [])
    shadow["field"] = {
        "base": {sc: summaries[("base", sc, 1)] for sc in ("fused", "pairwise")},
        "final": {sc: summaries[(final_name, sc, final_mo)] for sc in ("fused", "pairwise")},
    }
    cfg_ind = apply_overrides(config, {"channel.shadowing.mode": "independent"})
    runs_ind = experiment_runs(cfg_ind)
    t0 = time.perf_counter()
    ensure_all_cached(runs_ind, cache_dir, jobs)
    timings["cache_independent"] = time.perf_counter() - t0
    assert all(cache_path(a["config"], cache_dir) != cache_path(b["config"], cache_dir) for a, b in zip(runs, runs_ind))
    t0 = time.perf_counter()
    outs3 = run_phase(cfg_ind, runs_ind, cache_dir, shadow_variants, R, jobs)
    timings["phase3"] = time.perf_counter() - t0
    shadow["independent"] = {"base": {}, "final": {}}
    for v in shadow_variants:
        tag = "base" if v["name"] == "base" else "final"
        for sc in ("fused", "pairwise"):
            shadow["independent"][tag][sc] = summarize_variant(outs3, v, sc, v["min_observers"][0], cfg_ind)
    if not parts:
        shadow["independent"]["final"] = shadow["independent"]["base"]
    print("\n=== E: shadowing condiviso (field) e indipendente (independent) ===")
    print(f"{'shadowing':<12} {'config':<6} {'ambito':<9} {'soglia':>6} {'P_d slowdown':>13} {'P_d stop':>9} {'P_d medio':>10} {'FA/h':>6}   guadagno fusione (fused - pairwise, P_d medio)")
    for mode in ("field", "independent"):
        for tag in ("base", "final"):
            gains = {}
            for sc in ("fused", "pairwise"):
                s = shadow[mode][tag][sc]
                if s["wp"] is None:
                    print(f"{mode:<12} {tag:<6} {sc:<9} nessun punto ammissibile")
                    continue
                pd_mean = (s["point"]["slowdown"]["p_detect"] + s["point"]["stop"]["p_detect"]) / 2
                gains[sc] = pd_mean
                extra = f"{gains['fused'] - gains['pairwise']:+.3f}" if sc == "pairwise" and len(gains) == 2 else ""
                print(
                    f"{mode:<12} {tag:<6} {sc:<9} {s['wp']['threshold']:>6.1f} {s['point']['slowdown']['p_detect']:>13.2f} {s['point']['stop']['p_detect']:>9.2f}"
                    f" {pd_mean:>10.3f} {s['wp']['fa_per_hour']:>6.2f}   {extra}"
                )

    # ---- Tempi e salvataggio ---------------------------------------------------
    timings["total"] = time.perf_counter() - t_start
    print("\n=== Tempi di esecuzione ===")
    for (name, scope, mo), s in summaries.items():
        if scope == "fused" and mo == min(summaries_mos(summaries, name)):
            print(f"  {name:<22} {s['elapsed']:>8.1f} s (somma sulle corse, un processo)")
    for k, v in timings.items():
        print(f"  {k:<22} {v:>8.1f} s")
    print(f"  totale {timings['total'] / 60:.1f} min con {jobs} processi")

    save_pickle(
        out_dir / "results.pkl",
        {
            "config": config, "summaries": summaries, "variants": variants_by_name, "parts": parts, "singles_names": {k: v["name"] for k, v in singles.items()},
            "final_name": final_name, "final_mo": final_mo, "nis": nis, "diag": diag, "shadow": shadow, "timings": timings, "R": R,
        },
    )  # fmt: skip
    print(f"\nRisultati salvati in {out_dir}")


def mode_of(variant, config):
    return "drop" if variant["overrides"].get("detector.reference", config["detector"]["reference"]) == "relative" else "level"


def make_both(variant):
    v = dict(variant)
    v["scopes"] = ["fused", "pairwise"]
    v["min_observers"] = [1]
    v["corr"] = False
    return v


def summaries_mos(summaries, name):
    return [mo for (n, sc, mo) in summaries if n == name and sc == "fused"]


if __name__ == "__main__":
    main()

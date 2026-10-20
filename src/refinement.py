"""Esperimento del Blocco 5: varianti A-F del rilevatore, confrontate con la base sulle stesse corse.

Una VARIANTE è un insieme di sostituzioni della config (chiavi puntate, es. ``fusion.rule``). Per ogni
corsa (in cache, `src/simulator.py`) si valutano tutte le varianti di una fase: Kalman → fusione →
rilevatore su tutta la griglia di soglie, variante ``both`` (pre-allarme + perso). Poi, per ogni
variante: sweep → punto di lavoro (regola del Blocco 4: P_d massima con falsi allarmi/h e tempo in
allarme ammissibili) → confronto A COPPIE con la base (corse guadagnate e perse).

Regola di scelta (`is_better`): una variante batte la migliore corrente se guadagna almeno
``refinement.min_net_gain_runs`` corse nette (slowdown + stop); se è equivalente in P_d, se riduce di
altrettante le corse con spegnimenti a torto. Altrimenti si tiene la più semplice (l'ordine delle liste).

Parametri: sezioni ``refinement``, ``metrics``, ``detector``, ``fusion``, ``kalman`` della config.
"""

from __future__ import annotations

import itertools
import math
import time
from pathlib import Path

import numpy as np

from src.common.config import apply_overrides  # noqa: F401
from src.common.io import load_pickle, save_pickle  # noqa: F401
from src.common.parallel import parallel_map
from src.detector import build_inputs
from src.kalman import filter_all
from src.metrics import (
    aggregate,
    evaluate_config,
    min_distance_to_group,
    observer_correlation,
    paired_counts,
    summarize_none,
    summarize_separated_ext,
    threshold_grid,
    working_point,
)
from src.relative import relative_trace
from src.simulator import experiment_runs, load_or_simulate

SEPARATED = ("slowdown", "stop")
PER_RUN_FIELDS = (
    "success", "delay", "distance", "fortuitous", "n_wrong_off", "wrong_off_dist_sum", "wrong_off_dist_min",
    "repass", "repass_off", "alarm_on_end",
)  # fmt: skip


# ---------------------------------------------------------------------------
# Varianti
# ---------------------------------------------------------------------------


def block4_config(config: dict) -> dict:
    """Copia della config con i valori del Blocco 4 (`history.block4`): i default sono quelli del rilevatore finale (Blocco 6),
    gli script dei Blocchi 4-5 li rimettono per rifare i numeri di allora."""
    return apply_overrides(config, config["history"]["block4"])


def make_variant(name: str, group: str, overrides: dict, scopes=("fused",), min_observers=(1,), corr=False) -> dict:
    return {
        "name": name,
        "group": group,
        "overrides": dict(overrides),
        "scopes": list(scopes),
        "min_observers": list(min_observers),
        "corr": corr,
    }


def mode_label(overrides: dict, config: dict) -> str:
    """``level`` (assoluto) o ``drop`` (relativo): è il campo ``mode`` delle righe di `aggregate`."""
    ref = overrides.get("detector.reference", config["detector"]["reference"])
    return "drop" if ref == "relative" else "level"


def phase1_variants(config: dict) -> list:
    """Varianti A-D (una modifica alla volta, tutte rispetto alla base). La base è ``base`` (regola ``weighted_mean``, assoluto, 5 s, nessuna robustezza)."""
    base = config["refinement"]["base"]
    sigma = {"kalman.sigma_a": base["sigma_a"]}
    mos = config["detector"]["min_observers_sweep"]
    rules = config["fusion"]["rule_sweep"]
    variants = [make_variant("base", "base", sigma, ("fused", "pairwise"), mos, corr=True)]
    for rule in rules:
        if rule != "weighted_mean":
            variants.append(make_variant(f"A:{rule}", "A", {**sigma, "fusion.rule": rule}, ("fused", "pairwise"), mos, corr=True))
    variants.append(make_variant("B:relative", "B", {**sigma, "detector.reference": "relative"}, ("fused", "pairwise"), mos, corr=True))
    for rt in config["detector"]["release_time_sweep"]:
        if rt != config["detector"]["release_time"]:
            variants.append(make_variant(f"C:release={rt:g}", "C", {**sigma, "detector.release_time": rt}))
    robust = [(f"inflate={r:g}", {"kalman.robust": "inflate", "kalman.r_inflation": r}) for r in config["kalman"]["r_inflation_sweep"]]
    robust.append(("gate", {"kalman.robust": "nis_gate"}))
    for label, extra in robust:
        variants.append(make_variant(f"D:{label}", "D", {**sigma, **extra}))
        variants.append(make_variant(f"D:{label}+rel", "D", {**sigma, **extra, "detector.reference": "relative"}))
    return variants


# ---------------------------------------------------------------------------
# Lavoro per corsa
# ---------------------------------------------------------------------------


def _kalman_key(vcfg: dict) -> tuple:
    k = vcfg["kalman"]
    return (k["sigma_a"], k["robust"], k["r_inflation"], k["nis_gate"])


def process_run(args: tuple) -> dict:
    """Valuta tutte le `variants` su una corsa; `args` = (corsa, config, cartella di cache, varianti, ``R``).

    Restituisce ``scenario``, ``index``, ``results`` (``(nome, ambito)`` → `evaluate_config`), ``elapsed`` (s per variante) e, per
    le corse ``none``, ``nis`` (chiave del Kalman → NIS finite, float32).
    """
    run_info, config, cache_dir, variants, R = args
    run = load_or_simulate(run_info["config"], cache_dir)
    t = run["t"]
    is_none = run_info["scenario"] == "none"
    out = {"scenario": run_info["scenario"], "index": run_info["index"], "results": {}, "elapsed": {}, "nis": {}}
    warmup = config["detector"]["warmup"]
    if not is_none:
        node = run["sep_node"]
        idx_start = int(np.searchsorted(t, run["sep_start"]))
        distance = min_distance_to_group(run["distances"], node)
        success_distance = config["metrics"]["success_distance"]
        repass = config["refinement"]["repass_distance"]
        summarize = lambda s: summarize_separated_ext(s, t, idx_start, distance, success_distance, repass)  # noqa: E731
    else:
        summarize = lambda s: summarize_none(s, t, warmup)  # noqa: E731
    kcache = {}
    for v in variants:
        t0 = time.perf_counter()
        vcfg = apply_overrides(config, v["overrides"])
        key = _kalman_key(vcfg)
        if key not in kcache:
            kcache[key] = filter_all(run["beacon_times"], run["received"], run["rssi"], R, key[0], vcfg["kalman"])
            if is_none:
                nis = kcache[key].nis
                out["nis"][key] = nis[np.isfinite(nis)].astype(np.float32)
        inputs, _ = build_inputs(run, vcfg, key[0], R, kcache[key])
        if not is_none:
            inputs = inputs.target_slice(node)
        for scope in v["scopes"]:
            out["results"][(v["name"], scope)] = evaluate_config(
                inputs, vcfg, scope, summarize, v["min_observers"], warmup if (is_none and v["corr"]) else None
            )
        out["elapsed"][v["name"]] = time.perf_counter() - t0
    return out


def run_phase(config: dict, runs: list, cache_dir: Path, variants: list, R: float, jobs: int) -> list:
    """`process_run` su tutte le corse (in parallelo se ``jobs > 1``; il risultato non dipende da `jobs`)."""
    tasks = [(r, config, cache_dir, variants, R) for r in runs]
    return parallel_map(process_run, tasks, jobs)


# ---------------------------------------------------------------------------
# Aggregazione e punto di lavoro
# ---------------------------------------------------------------------------


def summarize_variant(outs: list, variant: dict, scope: str, mo: int, config: dict) -> dict:
    """Sweep e punto di lavoro di una variante (`aggregate` + `working_point`), con i dati per corsa al punto di lavoro.

    Restituisce ``rows`` (tutte le soglie), ``wp`` (None se nessun punto è ammissibile), ``point`` (scenario → riga), ``per_run``
    (scenario → campo → array sulle corse), ``corr`` (correlazione fra osservatori al punto di lavoro), ``elapsed`` (s totali).
    """
    name = variant["name"]
    mode = mode_label(variant["overrides"], config)
    key = (scope, mode, "both")
    none = sorted((o for o in outs if o["scenario"] == "none"), key=lambda o: o["index"])
    sep = {s: sorted((o for o in outs if o["scenario"] == s), key=lambda o: o["index"]) for s in SEPARATED}
    pick = lambda o: {key: o["results"][(name, scope)][mo]}  # noqa: E731
    rows = aggregate([pick(o) for o in none], {s: [pick(o) for o in v] for s, v in sep.items()}, config)
    wp = working_point(rows, scope, mode, "both")
    out = {
        "name": name, "group": variant["group"], "scope": scope, "min_observers": mo, "mode": mode, "rows": rows, "wp": wp,
        "point": {}, "per_run": {}, "corr": None, "elapsed": sum(o["elapsed"][name] for o in outs),
    }  # fmt: skip
    if wp is None:
        return out
    grid = rows and sorted({r["threshold"] for r in rows})
    n = int(np.argmin(np.abs(np.asarray(grid) - wp["threshold"])))
    for scenario, runs in sep.items():
        out["point"][scenario] = next(r for r in rows if r["scenario"] == scenario and r["threshold"] == wp["threshold"])
        out["per_run"][scenario] = {
            f: np.array([o["results"][(name, scope)][mo][f][n] for o in runs]) for f in PER_RUN_FIELDS
        }
    stats = [o["results"][(name, scope)].get("corr") for o in none]
    if stats and stats[0] is not None:
        pooled = {k: sum(s[k] for s in stats) for k in stats[0]}
        out["corr"] = observer_correlation(pooled, n)
    out["none_point"] = {
        k: out["point"]["slowdown"][k] for k in ("fa_per_hour", "fa_lo", "fa_hi", "active_fraction")
    }
    return out


def gain_loss(variant: dict, base: dict) -> dict:
    """Corse guadagnate e perse rispetto a `base`, per scenario e totale; None se uno dei due non ha punto di lavoro."""
    if variant["wp"] is None or base["wp"] is None:
        return {}
    out = {s: paired_counts(variant["per_run"][s]["success"], base["per_run"][s]["success"]) for s in SEPARATED}
    out["total"] = tuple(sum(out[s][k] for s in SEPARATED) for k in (0, 1))
    return out


def wrong_off_runs(summary: dict) -> int:
    """Corse (slowdown + stop) con almeno uno spegnimento a torto."""
    return int(sum((summary["per_run"][s]["n_wrong_off"] > 0).sum() for s in SEPARATED))


def is_better(cand: dict, best: dict, min_net: int) -> bool:
    """`cand` batte `best`? P_d a coppie: ≥ `min_net` corse nette; se equivalente, ≥ `min_net` corse con spegnimenti a torto in meno."""
    if cand["wp"] is None:
        return False
    if best["wp"] is None:
        return True
    gained, lost = gain_loss(cand, best)["total"]
    if gained - lost >= min_net:
        return True
    if gained - lost <= -min_net:
        return False
    return wrong_off_runs(best) - wrong_off_runs(cand) >= min_net


def choose(candidates: list, base: dict, min_net: int) -> dict:
    """Scorre i candidati in ordine di semplicità: ognuno deve battere il migliore corrente (partendo da `base`) oltre il rumore."""
    best = base
    for cand in candidates:
        if is_better(cand, best, min_net):
            best = cand
    return best


# ---------------------------------------------------------------------------
# Combinazione (F)
# ---------------------------------------------------------------------------


def components(winners: dict) -> dict:
    """Componenti vincenti: nome → sostituzioni di config (e `min_observers` in ``"_mo"``). Solo quelle diverse dalla base."""
    return {k: v for k, v in winners.items() if v is not None}


def combination_variant(name: str, parts: dict, config: dict, scopes, corr=False) -> dict:
    """Variante che unisce i componenti `parts` (A, B, C, D). La robustezza D usa le sostituzioni del suo ambito (assoluto/relativo)."""
    overrides = {"kalman.sigma_a": config["refinement"]["base"]["sigma_a"]}
    mo = 1
    relative = "B" in parts
    for key, part in parts.items():
        if key == "D":
            overrides.update(part["relative" if relative else "absolute"] or {})
        elif key == "A":
            overrides.update({k: v for k, v in part.items() if k != "_mo"})
            mo = part.get("_mo", 1)
        else:
            overrides.update(part)
    return make_variant(name, "F", overrides, scopes, (mo,), corr)


# ---------------------------------------------------------------------------
# Diagnostica della linea di base (corse none)
# ---------------------------------------------------------------------------


def _max_run(flag: np.ndarray, dt: float) -> float:
    """Durata massima (s) di una sequenza consecutiva di True."""
    best = cur = 0
    for x in flag:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return best * dt


def baseline_diagnostics(args: tuple) -> dict:
    """Diagnostica del livello relativo su una corsa ``none`` a una soglia: istogramma del calo fuso, congelamenti.

    `args` = (corsa, config, cartella di cache, variante, soglia, ``R``, ambito). Per ogni coppia (osservatore, bersaglio) dopo il warmup:
    ``hist`` (calo fuso), ``frozen_time``/``time`` (istanti con almeno una base congelata / totali), ``max_freeze`` (s, il congelamento
    più lungo di un singolo link), ``state_time`` (istanti in pre-allarme).
    """
    run_info, config, cache_dir, variant, threshold, R, scope = args
    run = load_or_simulate(run_info["config"], cache_dir)
    vcfg = apply_overrides(config, variant["overrides"])
    inputs, _ = build_inputs(run, vcfg, vcfg["kalman"]["sigma_a"], R)
    t = run["t"]
    first = int(np.searchsorted(t, vcfg["detector"]["warmup"]))
    dt = float(t[1] - t[0])
    lo, hi, step = config["refinement"]["drop_histogram"]
    edges = np.arange(lo, hi + step / 2, step)
    hist = np.zeros(len(edges) - 1)
    out = {"hist": hist, "edges": edges, "frozen_time": 0, "time": 0, "max_freeze": 0.0, "state_time": 0, "over_hi": 0, "pairs": []}
    n_nodes = run["received"].shape[0]
    for i in range(n_nodes):
        for m in range(n_nodes):
            if i == m:
                continue
            tr = relative_trace(inputs, vcfg, scope, threshold, m, i)
            drop = tr["drop"][first:]
            finite = drop[np.isfinite(drop)]
            hist += np.histogram(np.clip(finite, edges[0], edges[-1] - 1e-9), bins=edges)[0]
            out["over_hi"] += int((finite >= edges[-1]).sum())
            frozen = tr["frozen"][first:]
            any_frozen = frozen.any(axis=1)
            out["frozen_time"] += int(any_frozen.sum())
            out["time"] += len(any_frozen)
            out["state_time"] += int(tr["state"][first:].sum())
            longest = max(_max_run(frozen[:, k], dt) for k in range(frozen.shape[1]))
            out["max_freeze"] = max(out["max_freeze"], longest)
            out["pairs"].append((m, i, longest, float(any_frozen.mean())))
    return out


# ---------------------------------------------------------------------------
# Salvataggio
# ---------------------------------------------------------------------------


def best_threshold_index(summary: dict) -> int | None:
    if summary["wp"] is None:
        return None
    grid = sorted({r["threshold"] for r in summary["rows"]})
    return int(np.argmin(np.abs(np.asarray(grid) - summary["wp"]["threshold"])))


def threshold_axis(config: dict, mode: str) -> np.ndarray:
    return threshold_grid(config["metrics"]["drop_threshold_sweep" if mode == "drop" else "level_threshold_sweep"])

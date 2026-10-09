"""Blocco 4 — Metriche del rilevatore (versione di base).

- **Tempo di rilevamento**: primo istante dall'inizio del distacco
  (``t[idx_start]``) con l'allarme di sistema attivo sul nodo separato;
  ritardo = differenza (0 = **rilevamento fortuito**, contato a parte).
  Distanza = minima dal resto del gruppo (ground truth) in quell'istante.
- **Rilevamento riuscito**: l'allarme precede il primo istante in cui la
  distanza minima supera ``success_distance``. ``P_d`` = frazione di corse
  riuscite; ``p_detect_window`` (ritardo ``<= success_window``) è solo
  informativa.
- **Falsi allarmi** (corse ``none``, tutti i bersagli): episodi per ora
  (fronti di salita dopo il ``warmup`` / ore dopo il ``warmup``) e frazione
  del tempo in allarme.
- **Punto ammissibile** (σ_a, ambito, modalità, soglia): falsi allarmi/ora
  ``<= target_false_alarms_per_hour`` E frazione in allarme
  ``<= max_alarm_time_fraction``. Il punto di lavoro è fra gli ammissibili.
- **Intervalli di confidenza**: Wilson per ``P_d``, Poisson esatto (Garwood)
  per gli episodi/ora (ottimistico: i bersagli di una corsa non sono
  indipendenti).
- Varianti: pre-allarme (``pre``), solo "nodo perso" (``lost``), entrambi
  (``both``: osservatore in allarme se lo è uno dei due).

Monotonia rispetto alla soglia: allargandola, gli accendimenti sono un
sovrainsieme e gli spegnimenti un sottoinsieme, quindi l'insieme degli
istanti in allarme cresce sempre (osservatori, allarme di sistema, frazione
in allarme, ritardo non crescente, ``P_d``). Gli **episodi** possono invece
CALARE (due episodi si fondono, o un allarme già acceso prima del ``warmup``
non è un fronte): monotoni solo senza fusioni.

Parametri: sezioni ``metrics`` e ``detector`` della config.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import chi2, norm

from src.detector import (
    DetectionInputs,
    count_episodes,
    lost_alarm,
    pre_alarm,
    system_alarm,
)

VARIANTS = ("pre", "lost", "both")
MODES = ("level", "slope", "both")
SCOPES = ("fused", "pairwise")
SEPARATED = ("slowdown", "stop")


# ---------------------------------------------------------------------------
# Soglie e statistiche elementari
# ---------------------------------------------------------------------------


def threshold_grid(spec: list) -> np.ndarray:
    """Griglia crescente da ``[inizio, fine, passo]`` (estremi inclusi)."""
    start, stop, step = spec
    n = int(round((stop - start) / step)) + 1
    return start + step * np.arange(n)


def nearest_rank(values: np.ndarray, q: float) -> float:
    """Percentile a rango più vicino (elemento ``ceil(q·n)``-esimo ordinato, `q` in [0, 1]); NaN se vuoto. Ammette ``inf``."""
    values = np.sort(np.asarray(values, dtype=float))
    if len(values) == 0:
        return float("nan")
    rank = max(int(math.ceil(q * len(values) - 1e-12)), 1)
    return float(values[rank - 1])


# ---------------------------------------------------------------------------
# Riassunti di una corsa
# ---------------------------------------------------------------------------


def summarize_none(system: np.ndarray, t: np.ndarray, warmup: float) -> dict:
    """Falsi allarmi di una corsa senza distacco (allarme di sistema (T, I)).

    Restituisce ``episodes``, ``active`` (campioni in allarme), ``samples`` (bersagli × istanti), ``hours`` dopo il ``warmup``.
    """
    first = int(np.searchsorted(t, warmup))
    dt = float(t[1] - t[0])
    after = system[first:]
    return {
        "episodes": int(count_episodes(system, t, warmup).sum()),
        "active": int(after.sum()),
        "samples": int(after.size),
        "hours": after.shape[0] * dt / 3600.0,
    }


def summarize_separated(
    system: np.ndarray, t: np.ndarray, idx_start: int, distance: np.ndarray, success_distance: float
) -> dict:
    """Rilevamento in una corsa con distacco (allarme (T,) sul nodo separato, `distance` (T,) in m).

    Con `k` primo campione in allarme ``>= idx_start`` e `c` primo con ``distance > success_distance``:
    riuscito se ``k < c``. Restituisce ``delay`` (s, ``inf`` se mai), ``distance`` (m, NaN se mai),
    ``fortuitous``, ``success``.
    """
    exceeded = np.flatnonzero(distance[idx_start:] > success_distance)
    crossing = idx_start + int(exceeded[0]) if len(exceeded) else len(t)
    active = np.flatnonzero(system[idx_start:])
    if len(active) == 0:
        return {"delay": math.inf, "distance": math.nan, "fortuitous": False, "success": False}
    k = idx_start + int(active[0])
    return {
        "delay": float(t[k] - t[idx_start]),
        "distance": float(distance[k]),
        "fortuitous": k == idx_start,
        "success": bool(k < crossing),
    }


def min_distance_to_group(distances: np.ndarray, node: int) -> np.ndarray:
    """Distanza minima del nodo dal resto del gruppo, a ogni istante (ground truth)."""
    others = [j for j in range(distances.shape[1]) if j != node]
    return distances[:, node, others].min(axis=1).astype(float)


def wilson_interval(successes: int, n: int, level: float) -> tuple:
    """Intervallo di Wilson (inferiore, superiore) in [0, 1] per `successes` su `n` prove, al livello `level`."""
    z = float(norm.ppf(0.5 + level / 2.0))
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(centre - half, 0.0), min(centre + half, 1.0)


def poisson_interval(count: int, hours: float, level: float) -> tuple:
    """Intervallo esatto (Garwood) per una frequenza di Poisson: (inferiore, superiore) in eventi/ora."""
    alpha = 1.0 - level
    lower = 0.0 if count == 0 else float(chi2.ppf(alpha / 2.0, 2 * count)) / 2.0
    upper = float(chi2.ppf(1.0 - alpha / 2.0, 2 * count + 2)) / 2.0
    return lower / hours, upper / hours


# ---------------------------------------------------------------------------
# Valutazione di una corsa su tutta la griglia di parametri
# ---------------------------------------------------------------------------


def _grid_for(mode: str, metrics_cfg: dict) -> np.ndarray:
    key = "slope_threshold_sweep" if mode == "slope" else "level_threshold_sweep"
    return threshold_grid(metrics_cfg[key])


def evaluate_run(inputs: DetectionInputs, config: dict, summarize) -> dict:
    """Valuta una corsa per ogni (ambito, modalità, variante, soglia) applicando `summarize(system) -> dict`.

    `inputs` è intero (corse ``none``) o ritagliato sul bersaglio. L'allarme "perso" non dipende da
    soglia, modalità né ambito (chiave ``("-", "-", "lost")``). ``level``/``both`` variano la soglia di
    livello (pendenza fissa), ``slope`` quella di pendenza. Restituisce ``chiave -> {"threshold",
    chiavi di summarize}`` con un elemento per soglia.
    """
    dcfg = config["detector"]
    mcfg = config["metrics"]
    t = inputs.t
    lost_obs = lost_alarm(inputs.silence, inputs.rx_ok, t, dcfg, inputs.valid)
    results = {("-", "-", "lost"): _stack([math.nan], [summarize(system_alarm(lost_obs, dcfg["min_observers"]))])}
    for scope in SCOPES:
        for mode in MODES:
            grid = _grid_for(mode, mcfg)
            rows = {"pre": [], "both": []}
            for thr in grid:
                kwargs = {"slope_threshold": thr} if mode == "slope" else {"level_threshold": thr}
                pre_obs = pre_alarm(
                    inputs.level[scope], inputs.slope[scope], inputs.rx_ok, t, dcfg, mode, valid=inputs.valid, **kwargs
                )
                rows["pre"].append(summarize(system_alarm(pre_obs, dcfg["min_observers"])))
                rows["both"].append(summarize(system_alarm(pre_obs | lost_obs, dcfg["min_observers"])))
            for variant, summaries in rows.items():
                results[(scope, mode, variant)] = _stack(grid, summaries)
    return results


def _stack(thresholds, summaries: list) -> dict:
    out = {"threshold": np.asarray(thresholds, dtype=float)}
    for key in summaries[0]:
        out[key] = np.array([s[key] for s in summaries])
    return out


# ---------------------------------------------------------------------------
# Aggregazione sulle corse
# ---------------------------------------------------------------------------


def aggregate(none_runs: list, separated_runs: dict, config: dict) -> list:
    """Righe di metriche per ogni (ambito, modalità, variante, soglia, scenario).

    Ingressi: `evaluate_run` delle corse ``none`` (lista) e con distacco (scenario → lista). Falsi
    allarmi/ora = episodi totali / ore totali; mediana e p90 con rango più vicino (mancati = ``inf``).
    Colonne: scope, mode, variant, threshold, scenario, p_detect, p_lo, p_hi, p_detect_window, n_runs,
    fortuitous, delay_*/distance_* (median, p90), fa_per_hour, fa_lo, fa_hi, active_fraction, feasible.
    """
    mcfg = config["metrics"]
    window, level = mcfg["success_window"], mcfg["confidence_level"]
    rows = []
    keys = none_runs[0].keys()
    for key in keys:
        thresholds = none_runs[0][key]["threshold"]
        episodes = sum(r[key]["episodes"] for r in none_runs)
        hours = sum(r[key]["hours"] for r in none_runs)
        active = sum(r[key]["active"] for r in none_runs)
        samples = sum(r[key]["samples"] for r in none_runs)
        fa = episodes / hours
        fraction = active / samples
        fa_bounds = [poisson_interval(int(k), float(h), level) for k, h in zip(episodes, hours)]
        feasible = (fa <= mcfg["target_false_alarms_per_hour"]) & (fraction <= mcfg["max_alarm_time_fraction"])
        for scenario, runs in separated_runs.items():
            delay = np.stack([r[key]["delay"] for r in runs])  # (runs, thr)
            dist = np.stack([r[key]["distance"] for r in runs])
            fort = np.stack([r[key]["fortuitous"] for r in runs])
            success = np.stack([r[key]["success"] for r in runs])
            n_runs = len(runs)
            for n, thr in enumerate(thresholds):
                d, x = delay[:, n], dist[:, n]
                k_ok = int(success[:, n].sum())
                p_lo, p_hi = wilson_interval(k_ok, n_runs, level)
                rows.append(
                    {
                        "scope": key[0],
                        "mode": key[1],
                        "variant": key[2],
                        "threshold": float(thr),
                        "scenario": scenario,
                        "p_detect": k_ok / n_runs,
                        "p_lo": p_lo,
                        "p_hi": p_hi,
                        "p_detect_window": float(np.mean(d <= window)),
                        "n_runs": n_runs,
                        "fortuitous": int(fort[:, n].sum()),
                        "delay_median": nearest_rank(d, 0.5),
                        "delay_p90": nearest_rank(d, 0.9),
                        "distance_median": nearest_rank(np.where(np.isnan(x), math.inf, x), 0.5),
                        "distance_p90": nearest_rank(np.where(np.isnan(x), math.inf, x), 0.9),
                        "fa_per_hour": float(fa[n]),
                        "fa_lo": fa_bounds[n][0],
                        "fa_hi": fa_bounds[n][1],
                        "active_fraction": float(fraction[n]),
                        "feasible": int(feasible[n]),
                    }
                )
    return rows


def working_point(rows: list, scope: str, mode: str, variant: str) -> dict | None:
    """Soglia ammissibile con ``P_d`` (media sugli scenari) più alta, da righe di `aggregate` di un solo ``sigma_a``.

    A parità: meno falsi allarmi/ora, poi soglia più bassa. ``{"threshold", "p_detect", "fa_per_hour"}``
    o None se nessuna è ammissibile.
    """
    by_threshold: dict = {}
    for row in rows:
        if (row["scope"], row["mode"], row["variant"]) != (scope, mode, variant) or not row["feasible"]:
            continue
        thr = None if math.isnan(row["threshold"]) else row["threshold"]  # "perso": nessuna soglia
        entry = by_threshold.setdefault(thr, {"p": [], "fa": row["fa_per_hour"]})
        entry["p"].append(row["p_detect"])
    candidates = [
        (-float(np.mean(v["p"])), v["fa"], 0.0 if thr is None else thr, thr) for thr, v in by_threshold.items()
    ]
    if not candidates:
        return None
    neg_p, fa, _, thr = min(candidates, key=lambda c: c[:3])
    return {"threshold": math.nan if thr is None else thr, "p_detect": -neg_p, "fa_per_hour": fa}


def best_over_sigma(rows: list, scope: str, mode: str, variant: str) -> dict | None:
    """Miglior ``σ_a`` (campo ``sigma_a`` delle righe): `working_point` con ``P_d`` più alta, poi meno falsi allarmi, poi ``σ_a`` minore.

    Punto di lavoro con in più ``sigma_a``, o None.
    """
    best = None
    for sigma_a in sorted({row["sigma_a"] for row in rows}):
        sel = [row for row in rows if row["sigma_a"] == sigma_a]
        point = working_point(sel, scope, mode, variant)
        if point is None:
            continue
        point = dict(point, sigma_a=sigma_a)
        if best is None or (-point["p_detect"], point["fa_per_hour"]) < (-best["p_detect"], best["fa_per_hour"]):
            best = point
    return best

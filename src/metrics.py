"""Block 4 — Detector metrics (base version).

- **Detection time**: first instant since the start of the separation
  (``t[idx_start]``) with the system alarm active on the separated node;
  delay = difference (0 = **fortuitous detection**, counted separately).
  Distance = minimum from the rest of the group (ground truth) at that instant.
- **Successful detection**: the alarm precedes the first instant at which the
  minimum distance exceeds ``success_distance``. ``P_d`` = fraction of
  successful runs; ``p_detect_window`` (delay ``<= success_window``) is
  informative only.
- **False alarms** (``none`` runs, all targets): episodes per hour
  (rising edges after ``warmup`` / hours after ``warmup``) and fraction
  of time in alarm.
- **Feasible point** (σ_a, scope, mode, threshold): false alarms/hour
  ``<= target_false_alarms_per_hour`` AND fraction in alarm
  ``<= max_alarm_time_fraction``. The working point is among the feasible ones.
- **Confidence intervals**: Wilson for ``P_d``, exact Poisson (Garwood)
  for episodes/hour (optimistic: the targets of a run are not
  independent).
- Variants: pre-alarm (``pre``), "lost node" only (``lost``), both
  (``both``: observer in alarm if either of the two is).

Monotonicity with respect to the threshold: widening it, the turn-ons are a
superset and the turn-offs a subset, so the set of
instants in alarm always grows (observers, system alarm, fraction
in alarm, non-increasing delay, ``P_d``). The **episodes** can instead
DECREASE (two episodes merge, or an alarm already on before ``warmup``
is not an edge): monotonic only without merges.

Parameters: ``metrics`` and ``detector`` sections of the config.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import chi2, norm

from src.common.enums import DetectorMode, Reference, Scenario
from src.detector import (
    DetectionInputs,
    count_episodes,
    lost_alarm,
    pre_alarm,
    system_alarm,
)
from src.relative import relative_alarms

VARIANTS = ("pre", "lost", "both")
MODES = tuple(m.value for m in DetectorMode)
SCOPES = ("fused", "pairwise")
RANK_TOLERANCE = 1e-12
SEPARATED = (Scenario.SLOWDOWN.value, Scenario.STOP.value)


# ---------------------------------------------------------------------------
# Thresholds and elementary statistics
# ---------------------------------------------------------------------------


def threshold_grid(spec: list) -> np.ndarray:
    """Increasing grid from ``[start, stop, step]`` (endpoints included)."""
    start, stop, step = spec
    n = int(round((stop - start) / step)) + 1
    return start + step * np.arange(n)


def nearest_rank(values: np.ndarray, q: float) -> float:
    """Nearest-rank percentile (the ``ceil(q·n)``-th sorted element, `q` in [0, 1]); NaN if empty. Allows ``inf``."""
    values = np.sort(np.asarray(values, dtype=float))
    if len(values) == 0:
        return float("nan")
    rank = max(int(math.ceil(q * len(values) - RANK_TOLERANCE)), 1)
    return float(values[rank - 1])


# ---------------------------------------------------------------------------
# Summaries of a run
# ---------------------------------------------------------------------------


def summarize_none(system: np.ndarray, t: np.ndarray, warmup: float) -> dict:
    """False alarms of a run without separation (system alarm (T, I)).

    Returns ``episodes``, ``active`` (samples in alarm), ``samples`` (targets × instants), ``hours`` after ``warmup``.
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
    """Detection in a run with separation (alarm (T,) on the separated node, `distance` (T,) in m).

    With `k` the first sample in alarm ``>= idx_start`` and `c` the first with ``distance > success_distance``:
    successful if ``k < c``. Returns ``delay`` (s, ``inf`` if never), ``distance`` (m, NaN if never),
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
    """Minimum distance of the node from the rest of the group, at each instant (ground truth)."""
    others = [j for j in range(distances.shape[1]) if j != node]
    return distances[:, node, others].min(axis=1).astype(float)


def separation_target(run: dict) -> tuple:
    node = run["sep_node"]
    return node, int(np.searchsorted(run["t"], run["sep_start"])), min_distance_to_group(run["distances"], node)


def run_summarizer(run: dict, config: dict, is_none: bool, extended: bool = False) -> tuple:
    t = run["t"]
    if is_none:
        warmup = config["detector"]["warmup"]
        return (lambda s: summarize_none(s, t, warmup)), None
    node, idx_start, distance = separation_target(run)
    success_distance = config["metrics"]["success_distance"]
    if extended:
        repass = config["refinement"]["repass_distance"]
        return (lambda s: summarize_separated_ext(s, t, idx_start, distance, success_distance, repass)), node
    return (lambda s: summarize_separated(s, t, idx_start, distance, success_distance)), node


def split_by_scenario(outs: list, scenarios=SEPARATED) -> tuple:
    none = sorted((o for o in outs if o["scenario"] == Scenario.NONE), key=lambda o: o["index"])
    separated = {s: sorted((o for o in outs if o["scenario"] == s), key=lambda o: o["index"]) for s in scenarios}
    return none, separated


def wilson_interval(successes: int, n: int, level: float) -> tuple:
    """Wilson interval (lower, upper) in [0, 1] for `successes` out of `n` trials, at level `level`."""
    z = float(norm.ppf(0.5 + level / 2.0))
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return max(centre - half, 0.0), min(centre + half, 1.0)


def poisson_interval(count: int, hours: float, level: float) -> tuple:
    """Exact (Garwood) interval for a Poisson rate: (lower, upper) in events/hour."""
    alpha = 1.0 - level
    lower = 0.0 if count == 0 else float(chi2.ppf(alpha / 2.0, 2 * count)) / 2.0
    upper = float(chi2.ppf(1.0 - alpha / 2.0, 2 * count + 2)) / 2.0
    return lower / hours, upper / hours


# ---------------------------------------------------------------------------
# Evaluation of a run over the whole parameter grid
# ---------------------------------------------------------------------------


def _grid_for(mode: str, metrics_cfg: dict) -> np.ndarray:
    key = "slope_threshold_sweep" if mode == DetectorMode.SLOPE else "level_threshold_sweep"
    return threshold_grid(metrics_cfg[key])


def evaluate_run(inputs: DetectionInputs, config: dict, summarize) -> dict:
    """Evaluates a run for each (scope, mode, variant, threshold) by applying `summarize(system) -> dict`.

    `inputs` is whole (``none`` runs) or cropped to the target. The "lost" alarm does not depend on
    threshold, mode or scope (key ``("-", "-", "lost")``). ``level``/``both`` vary the level
    threshold (fixed slope), ``slope`` the slope one. Returns ``key -> {"threshold",
    summarize keys}`` with one element per threshold.
    """
    dcfg = config["detector"]
    mcfg = config["metrics"]
    t = inputs.t
    lost_obs = lost_alarm(inputs.silence, inputs.rx_ok, t, dcfg, inputs.valid, **inputs.lost_kwargs())
    results = {("-", "-", "lost"): stack_summaries([math.nan], [summarize(system_alarm(lost_obs, dcfg["min_observers"]))])}
    for scope in SCOPES:
        for mode in MODES:
            grid = _grid_for(mode, mcfg)
            rows = {"pre": [], "both": []}
            for thr in grid:
                kwargs = {"slope_threshold": thr} if mode == DetectorMode.SLOPE else {"level_threshold": thr}
                pre_obs = pre_alarm(
                    inputs.level[scope], inputs.slope[scope], inputs.rx_ok, t, dcfg, mode, valid=inputs.valid, **kwargs
                )
                rows["pre"].append(summarize(system_alarm(pre_obs, dcfg["min_observers"])))
                rows["both"].append(summarize(system_alarm(pre_obs | lost_obs, dcfg["min_observers"])))
            for variant, summaries in rows.items():
                results[(scope, mode, variant)] = stack_summaries(grid, summaries)
    return results


def stack_summaries(thresholds, summaries: list) -> dict:
    out = {"threshold": np.asarray(thresholds, dtype=float)}
    for key in summaries[0]:
        out[key] = np.array([s[key] for s in summaries])
    return out


# ---------------------------------------------------------------------------
# Aggregation over runs
# ---------------------------------------------------------------------------


def aggregate(none_runs: list, separated_runs: dict, config: dict) -> list:
    """Metric rows for each (scope, mode, variant, threshold, scenario).

    Inputs: `evaluate_run` of the ``none`` runs (list) and of the separation runs (scenario → list). False
    alarms/hour = total episodes / total hours; median and p90 with nearest rank (missed = ``inf``).
    Columns: scope, mode, variant, threshold, scenario, p_detect, p_lo, p_hi, p_detect_window, n_runs,
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
    """Feasible threshold with the highest ``P_d`` (mean over scenarios), from `aggregate` rows of a single ``sigma_a``.

    On ties: fewer false alarms/hour, then lower threshold. ``{"threshold", "p_detect", "fa_per_hour"}``
    or None if none is feasible.
    """
    by_threshold: dict = {}
    for row in rows:
        if (row["scope"], row["mode"], row["variant"]) != (scope, mode, variant) or not row["feasible"]:
            continue
        thr = None if math.isnan(row["threshold"]) else row["threshold"]  # "lost": no threshold
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
    """Best ``σ_a`` (``sigma_a`` field of the rows): `working_point` with the highest ``P_d``, then fewer false alarms, then lower ``σ_a``.

    Working point with ``sigma_a`` added, or None.
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


# ---------------------------------------------------------------------------
# Block 5: additional metrics and evaluation of a configuration
# ---------------------------------------------------------------------------


def summarize_separated_ext(
    system: np.ndarray, t: np.ndarray, idx_start: int, distance: np.ndarray, success_distance: float, repass_distance: float
) -> dict:
    """`summarize_separated` plus wrong turn-offs and re-passes (Block 5, section C).

    Input: system alarm (T,) on the separated node, `distance` (T,) in m from the group (ground truth), `idx_start` first sample of the separation.
    Procedure: in ``slowdown`` and ``stop`` the separated node never returns, so every alarm turn-off AFTER detection
    (first sample in alarm `k`) is wrong. Re-pass: after `k` the distance exceeds `repass_distance` and then drops below it (the group
    passes by the stopped node again).
    Output: the keys of `summarize_separated` plus ``n_wrong_off`` (wrong turn-offs), ``wrong_off_dist_sum`` and ``wrong_off_dist_min``
    (m, distance from the group at the turn-offs; NaN if none), ``repass`` (bool), ``repass_off`` (the alarm turned off during the
    re-pass), ``alarm_on_end`` (on at the end of the run).
    """
    out = summarize_separated(system, t, idx_start, distance, success_distance)
    out.update(n_wrong_off=0, wrong_off_dist_sum=0.0, wrong_off_dist_min=math.nan, repass=False, repass_off=False)
    out["alarm_on_end"] = bool(system[-1]) if math.isfinite(out["delay"]) else False
    if not math.isfinite(out["delay"]):
        return out
    k = idx_start + int(np.flatnonzero(system[idx_start:])[0])
    falls = k + 1 + np.flatnonzero(system[k:-1] & ~system[k + 1 :])
    out["n_wrong_off"] = int(len(falls))
    if len(falls):
        out["wrong_off_dist_sum"] = float(distance[falls].sum())
        out["wrong_off_dist_min"] = float(distance[falls].min())
    far = np.flatnonzero(distance[k:] > repass_distance)
    if len(far):
        c = k + int(far[0])
        back = np.flatnonzero(distance[c:] < repass_distance)
        if len(back):
            first = c + int(back[0])
            leave = np.flatnonzero(distance[first:] >= repass_distance)
            end = first + int(leave[0]) if len(leave) else len(t)
            out["repass"] = True
            out["repass_off"] = bool((~system[first:end]).any())
    return out


def observer_states(inputs: DetectionInputs, config: dict, scope: str) -> tuple:
    """Pre-alarm of each observer for each threshold: (thresholds, states) with states ``(thresholds, T, M[, I])``.

    ``detector.reference``: ``absolute`` = `pre_alarm` in ``level`` mode on the ``level_threshold_sweep`` grid (dBm);
    ``relative`` = `relative_alarms` on the ``drop_threshold_sweep`` grid (dB).
    """
    dcfg, mcfg = config["detector"], config["metrics"]
    if dcfg["reference"] == Reference.RELATIVE:
        grid = threshold_grid(mcfg["drop_threshold_sweep"])
        return grid, relative_alarms(inputs, config, scope, grid)
    if dcfg["reference"] != Reference.ABSOLUTE:
        raise ValueError(f"detector.reference must be 'absolute' or 'relative', not {dcfg['reference']!r}")
    grid = threshold_grid(mcfg["level_threshold_sweep"])
    states = np.stack(
        [
            pre_alarm(inputs.level[scope], inputs.slope[scope], inputs.rx_ok, inputs.t, dcfg, "level", level_threshold=thr, valid=inputs.valid)
            for thr in grid
        ]
    )
    return grid, states


def observer_stats(observer: np.ndarray, t: np.ndarray, warmup: float) -> dict:
    """Sums for the correlation between the alarms of observers on the same target (``none`` runs).

    Input: observer alarms ``(thresholds, T, M, I)``. After `warmup`: ``n`` (instants), ``n1[s, m, i]`` (instants in alarm),
    ``n11[s, m, m', i]`` (in alarm together), ``hist[s, c]`` (target-instants with `c` observers in alarm). Summable across runs.
    """
    first = int(np.searchsorted(t, warmup))
    a = observer[:, first:].astype(np.float32)
    n_thr, n_steps, n_obs, n_tgt = a.shape
    n11 = np.stack([np.einsum("tmi,tni->mni", a[s], a[s]) for s in range(n_thr)])
    counts = a.sum(axis=2).astype(int)  # (thresholds, T, I)
    hist = np.stack([np.bincount(counts[s].ravel(), minlength=n_obs + 1) for s in range(n_thr)])
    return {"n": np.array(n_steps), "n1": a.sum(axis=1).astype(np.float64), "n11": n11.astype(np.float64), "hist": hist}


def observer_correlation(stats: dict, index: int) -> dict:
    """Mean correlation (phi) between the alarms of pairs of observers on the same target, at threshold `index`.

    `stats`: sum of `observer_stats` over runs. Phi of each triple (observer, observer', target) with both different from the
    target and non-zero variance; mean over triples (``phi``, ``n_triples``). Also ``p_two_given_one`` = P(≥2 observers | ≥1).
    """
    n = float(stats["n"])
    n1, n11 = stats["n1"][index], stats["n11"][index]
    n_obs, n_tgt = n1.shape
    phis = []
    for i in range(n_tgt):
        for m1 in range(n_obs):
            for m2 in range(m1 + 1, n_obs):
                if i in (m1, m2):
                    continue
                var1 = n1[m1, i] * (n - n1[m1, i])
                var2 = n1[m2, i] * (n - n1[m2, i])
                if var1 > 0 and var2 > 0:
                    phis.append((n * n11[m1, m2, i] - n1[m1, i] * n1[m2, i]) / math.sqrt(var1 * var2))
    hist = stats["hist"][index]
    at_least_one = float(hist[1:].sum())
    return {
        "phi": float(np.mean(phis)) if phis else math.nan,
        "n_triples": len(phis),
        "p_two_given_one": float(hist[2:].sum() / at_least_one) if at_least_one > 0 else math.nan,
    }


def evaluate_config(
    inputs: DetectionInputs, config: dict, scope: str, summarize, min_observers: list, corr_warmup: float | None = None
) -> dict:
    """``both`` variant (pre-alarm + lost) of a configuration, over the whole threshold grid, for each `min_observers`.

    Input: `inputs` (whole or cropped to a target), `scope`, `summarize(system) -> dict`, list of `min_observers`.
    Procedure: `observer_states` (absolute or relative according to ``detector.reference``), "lost" alarm (does not depend on the threshold),
    `system_alarm` and `summarize` at each threshold. With `corr_warmup` adds ``"corr"`` = `observer_stats` of the observers.
    Output: ``{min_observers: {"threshold", summarize keys}}``.
    """
    dcfg = config["detector"]
    grid, states = observer_states(inputs, config, scope)
    lost_obs = lost_alarm(inputs.silence, inputs.rx_ok, inputs.t, dcfg, inputs.valid, **inputs.lost_kwargs())
    out = {}
    for mo in min_observers:
        out[mo] = stack_summaries(grid, [summarize(system_alarm(states[n] | lost_obs, mo)) for n in range(len(grid))])
    if corr_warmup is not None:
        out["corr"] = observer_stats(states | lost_obs, inputs.t, corr_warmup)
    return out


def paired_counts(success: np.ndarray, base_success: np.ndarray) -> tuple:
    """Paired comparison with the base: (runs gained, runs lost) = successful only in the variant, successful only in the base."""
    success = np.asarray(success, dtype=bool)
    base_success = np.asarray(base_success, dtype=bool)
    return int((success & ~base_success).sum()), int((~success & base_success).sum())


def gain_loss_counts(success: dict, base_success: dict) -> dict:
    out = {s: paired_counts(success[s], base_success[s]) for s in SEPARATED}
    out["total"] = tuple(sum(out[s][k] for s in SEPARATED) for k in (0, 1))
    return out

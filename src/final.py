"""Block 6 — Final numbers, frozen thresholds, verdict and sensitivity.

Two sets of runs, never mixed (`run_sets`):

- **tuning**: the 150 runs of Blocks 4-5 (keys 6-7); here, and only here, the threshold is chosen (`freeze_thresholds`) and written to
  ``frozen_thresholds.yaml``;
- **verification**: ``final.n_runs`` runs per scenario with keys 8-9, never seen; thresholds are READ from the file (`evaluate_at` chooses nothing).

A *configuration* is ``media`` (``fused`` + ``weighted_mean``), ``migliore`` (``fused`` + ``max``) or ``coppia`` (``pairwise``); all with absolute
level, σ_a and release fixed in ``final.common``. Per-run result key: ``(name, lost_rule)``, with ``name`` also ``"perso"``
(the "lost node" alarm alone, no threshold).

Parameters: ``final``, ``detector``, ``metrics``, ``packets`` sections of the config.
"""

from __future__ import annotations

import datetime
import math
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import binom

from src.common.arrays import nearest_index, rising_edges, run_bounds, warmup_start
from src.common.enums import Scenario
from src.common.git import git_or_unknown
from src.common.parallel import parallel_map
from src.common.stats import percentile_interval, resample_indices
from src.common.rng import bootstrap_rng
from src.detector import build_inputs, lost_alarm, system_alarm
from src.kalman import filter_all
from src.metrics import (
    SEPARATED,
    stack_summaries,
    aggregate,
    gain_loss_counts,
    run_summarizer,
    split_by_scenario,
    nearest_rank,
    observer_states,
    paired_counts,
    poisson_interval,
    summarize_none,
    wilson_interval,
    working_point,
)
from src.packets import burst_lengths
from src.refinement import PER_RUN_FIELDS, apply_overrides
from src.simulator import TUNING_KEYS, experiment_runs, load_or_simulate, run_from_channel

NONE_FIELDS = ("episodes", "hours", "active", "samples", "ep_level", "ep_lost")
LOST = "perso"  # name of the "configuration" holding the "lost node" alarm alone


# ---------------------------------------------------------------------------
# Run sets and configurations
# ---------------------------------------------------------------------------


def run_sets(config: dict) -> tuple:
    """Tuning and verification sets.

    Output: (tuning, verification), lists of runs as in `experiment_runs`: tuning = ``metrics.n_runs`` runs per scenario with keys 7 (seeds) and 6
    (node and instant); verification = ``final.n_runs`` runs with ``final.seed_keys.runs`` and ``final.seed_keys.separation``.
    """
    keys = config["final"]["seed_keys"]
    tuning = experiment_runs(config)
    verification = experiment_runs(config, {"runs": keys["runs"], "separation": keys["separation"]}, config["final"]["n_runs"])
    return tuning, verification


GRID_TOLERANCE = 1e-9


def config_overrides(config: dict, name: str, point: dict | None = None) -> dict:
    """Config overrides of configuration `name` (``final.common`` + ``final.config_defs[name]``) and of a sensitivity point."""
    fcfg = config["final"]
    return {**fcfg["common"], **fcfg["config_defs"][name], **(point or {})}


def result_key(config: dict, name: str) -> tuple:
    """Key of the `aggregate` rows for configuration `name`: ``(scope, "level", "both")``, or that of "perso" alone."""
    if name == LOST:
        return ("-", "-", "lost")
    return (config["final"]["config_defs"][name]["detector.scope"], "level", "both")


# ---------------------------------------------------------------------------
# Per-run work
# ---------------------------------------------------------------------------


def episodes_by_type(pre_system: np.ndarray, system: np.ndarray, t: np.ndarray, warmup: float) -> tuple:
    """False alarms (rising edges of the system alarm after `warmup`, as in `count_episodes`) split by type.

    Input: `pre_system` and `system` (T, I) bool (level pre-alarm; pre-alarm or "lost"), `t`.
    Procedure: a rising edge is of type "level" if the system pre-alarm is on at the instant it fires, otherwise "lost".
    Output: (level episodes, lost episodes); the sum is ``count_episodes(system).sum()``.
    """
    first = warmup_start(t, warmup)
    edges, level = rising_edges(system)[first:], pre_system[1:][first:]
    return int((edges & level).sum()), int((edges & ~level).sum())


def _summ_none(pre_system, system, t, warmup):
    out = summarize_none(system, t, warmup)
    out["ep_level"], out["ep_lost"] = episodes_by_type(pre_system, system, t, warmup)
    return out


def _diagnostics(run: dict, inputs, config: dict) -> dict:
    """``none`` run: beacons received in the group, (observed) bursts and deafness. ``longest`` = (duration s, receiver, start time s)."""
    t = run["t"]
    dt = float(t[1] - t[0])
    n_nodes = run["received"].shape[0]
    trials = int((~np.isnan(run["beacon_times"])).sum()) * (n_nodes - 1)
    out = {"received": int(run["received"].sum()), "trials": trials, "duration": float(t[-1] - t[0]), "n_long": 0, "n_bursts": 0, "longest": (0.0, -1, math.nan)}
    state = run.get("background_state")
    if state is not None and state.ndim == 2:
        lengths = burst_lengths(state, dt)
        out["n_bursts"] = int(len(lengths))
        out["n_long"] = int((lengths > config["final"]["burst_long"]).sum())
        best = (0.0, -1, math.nan)
        for node in range(state.shape[1]):
            for a, b in zip(*run_bounds(state[:, node])):
                if (b - a) * dt > best[0]:
                    best = ((b - a) * dt, node, float(t[a]))
        out["longest"] = best
    dw = inputs.deaf_window
    out["deaf_episodes"] = int((dw[1:] & ~dw[:-1]).sum())
    return out


def process_run_final(args: tuple) -> dict:
    """Evaluate the configurations of `spec` on one run; `args` = (run, config, cache directory, ``R``, spec).

    Input: `spec` = ``{"mode": "cache" | "channel", "point": {config overrides}, "configs": [names], "rules": [lost rules]}``.
    ``cache``: Blocks 1-3 from the run cache; ``channel``: Blocks 1-2 from the channel cache and Block 3 redone with the point overrides
    (without writing to the run cache).
    Procedure: Kalman once (σ_a of ``final.common``); one `build_inputs` per fusion rule; for each configuration `observer_states`
    over the whole threshold grid; for each ``lost_rule`` the "lost" alarm and the system alarm (pre | lost) at each threshold, with `summarize_none`
    (plus episodes per type) or `summarize_separated_ext`; "lost" alone once per rule.
    Output: ``scenario``, ``index``, ``results`` (``(name, rule)`` → `stack_summaries` per threshold), ``diag`` (``none`` only), ``elapsed``.
    """
    run_info, config, cache_dir, R, spec = args
    t0 = time.perf_counter()
    point = spec.get("point") or {}
    rcfg = apply_overrides(run_info["config"], point) if point else run_info["config"]
    run = run_from_channel(rcfg, cache_dir) if spec["mode"] == "channel" else load_or_simulate(rcfg, cache_dir)
    t = run["t"]
    is_none = run_info["scenario"] == Scenario.NONE
    warmup = config["detector"]["warmup"]
    common = apply_overrides(config, {**config["final"]["common"], **point})
    sigma_a = common["kalman"]["sigma_a"]
    kres = filter_all(run["beacon_times"], run["received"], run["rssi"], R, sigma_a, common["kalman"])
    summarize_system, node = run_summarizer(run, config, is_none, extended=True)
    if is_none:
        summarize = lambda pre, system: _summ_none(pre, system, t, warmup)  # noqa: E731
    else:
        summarize = lambda pre, system: summarize_system(system)  # noqa: E731
    out = {"scenario": run_info["scenario"], "index": run_info["index"], "results": {}, "diag": None}
    inputs_by_rule = {}
    first_inputs = None
    for name in spec["configs"]:
        vcfg = apply_overrides(config, config_overrides(config, name, point))
        fusion_rule = vcfg["fusion"]["rule"]
        if fusion_rule not in inputs_by_rule:
            inputs, _ = build_inputs(run, vcfg, sigma_a, R, kres)
            inputs_by_rule[fusion_rule] = inputs if is_none else inputs.target_slice(node)
        inputs = inputs_by_rule[fusion_rule]
        first_inputs = first_inputs or inputs
        grid, states = observer_states(inputs, vcfg, vcfg["detector"]["scope"])
        mo = vcfg["detector"]["min_observers"]
        pre_systems = [system_alarm(states[n], mo) for n in range(len(grid))]
        for rule in spec["rules"]:
            dcfg = {**vcfg["detector"], "lost_rule": rule}
            lost_obs = lost_alarm(inputs.silence, inputs.rx_ok, t, dcfg, inputs.valid, **inputs.lost_kwargs())
            out["results"][(name, rule)] = stack_summaries(
                grid, [summarize(pre_systems[n], system_alarm(states[n] | lost_obs, mo)) for n in range(len(grid))]
            )
    for rule in spec["rules"]:
        dcfg = {**common["detector"], "lost_rule": rule}
        lost_sys = system_alarm(
            lost_alarm(first_inputs.silence, first_inputs.rx_ok, t, dcfg, first_inputs.valid, **first_inputs.lost_kwargs()), dcfg["min_observers"]
        )
        out["results"][(LOST, rule)] = stack_summaries([math.nan], [summarize(np.zeros_like(lost_sys), lost_sys)])
    if is_none:
        out["diag"] = _diagnostics(run, first_inputs, config)
    out["elapsed"] = time.perf_counter() - t0
    return out


def run_final_phase(config: dict, runs: list, cache_dir: Path, R: float, spec: dict, jobs: int) -> list:
    """`process_run_final` on all runs (in parallel if ``jobs > 1``; the result does not depend on `jobs`)."""
    tasks = [(r, config, cache_dir, R, spec) for r in runs]
    return parallel_map(process_run_final, tasks, jobs)


# ---------------------------------------------------------------------------
# Aggregation and frozen thresholds
# ---------------------------------------------------------------------------


def split_outs(outs: list) -> tuple:
    """(``none`` runs sorted by index, scenario → separation runs sorted)."""
    return split_by_scenario(outs)


def sweep_rows(outs: list, name: str, rule: str, config: dict) -> list:
    """`aggregate` rows (one per scenario and threshold) for configuration `name` and lost rule `rule`."""
    key = result_key(config, name)
    none, sep = split_outs(outs)
    pick = lambda o: {key: o["results"][(name, rule)]}  # noqa: E731
    return aggregate([pick(o) for o in none], {s: [pick(o) for o in v] for s, v in sep.items()}, config)


def freeze_thresholds(tuning_outs: list, config: dict) -> dict:
    """Choose the thresholds from the TUNING runs: Block 4 rule (`working_point`): maximum mean P_d with admissible false alarms/h and time in alarm.

    Output: ``{name: {"threshold", "scope", "fusion_rule", "tuning": {p_detect_slowdown, p_detect_stop, fa_per_hour, active_fraction}}}`` for the
    configurations of ``final.configs``. The only place in the code where a threshold that is then frozen is chosen.
    """
    frozen = {}
    for name in config["final"]["configs"]:
        rows = sweep_rows(tuning_outs, name, "any", config)
        scope = config["final"]["config_defs"][name]["detector.scope"]
        wp = working_point(rows, scope, "level", "both")
        if wp is None:
            raise RuntimeError(f"configuration {name}: no admissible point on the tuning set")
        at = {s: next(r for r in rows if r["scenario"] == s and r["threshold"] == wp["threshold"]) for s in SEPARATED}
        frozen[name] = {
            "threshold": float(wp["threshold"]),
            "scope": scope,
            "fusion_rule": config["final"]["config_defs"][name]["fusion.rule"],
            "tuning": {
                "p_detect_slowdown": float(at["slowdown"]["p_detect"]),
                "p_detect_stop": float(at["stop"]["p_detect"]),
                "fa_per_hour": float(at["slowdown"]["fa_per_hour"]),
                "active_fraction": float(at["slowdown"]["active_fraction"]),
            },
        }
    return frozen


def write_frozen(path: Path, frozen: dict, config: dict, R: float) -> None:
    """Write `frozen` as YAML with date, commit hash (and ``dirty`` if there are uncommitted changes), ``R`` and the tuning conditions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "commit": git_or_unknown("rev-parse", "HEAD"),
        "dirty": bool(git_or_unknown("status", "--porcelain")),
        "tuning_set": {"runs_per_scenario": config["metrics"]["n_runs"], "seed_keys": dict(TUNING_KEYS)},
        "R": float(R),
        "release_time": float(config["final"]["common"]["detector.release_time"]),
        "thresholds": frozen,
    }
    with open(path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)


def load_frozen(path: Path) -> dict:
    """Read ``frozen_thresholds.yaml``: return the document (thresholds are in ``["thresholds"][name]["threshold"]``)."""
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Evaluation at a given threshold
# ---------------------------------------------------------------------------


def threshold_index(grid: np.ndarray, threshold: float) -> int:
    """Index of threshold `threshold` in the grid (must match); NaN = "lost" alone (index 0)."""
    if math.isnan(threshold):
        return 0
    n = nearest_index(grid, threshold)
    if abs(grid[n] - threshold) > GRID_TOLERANCE:
        raise ValueError(f"threshold {threshold} is not in the sweep grid")
    return n


def at_threshold(outs: list, name: str, rule: str, threshold: float) -> dict:
    """Per-run data at threshold `threshold`: ``{"none": {field: array (runs,)}, "sep": {scenario: {field: array}}}``."""
    none, sep = split_outs(outs)
    n = threshold_index(next(iter(none))["results"][(name, rule)]["threshold"], threshold)
    return {
        "none": {f: np.array([o["results"][(name, rule)][f][n] for o in none]) for f in NONE_FIELDS},
        "sep": {s: {f: np.array([o["results"][(name, rule)][f][n] for o in v]) for f in PER_RUN_FIELDS} for s, v in sep.items()},
    }


def bootstrap_rate_interval(counts: np.ndarray, hours: np.ndarray, n_samples: int, level: float, rng: np.random.Generator) -> tuple:
    """Bootstrap (percentile) interval for the event rate per hour with RUNS as the independent unit.

    Input: `counts` and `hours` per run, number of resamples, level, generator.
    Procedure: runs are resampled with replacement; rate = sum of events / sum of hours; percentiles ``(1-level)/2`` and ``1-(1-level)/2``.
    """
    counts, hours = np.asarray(counts, dtype=float), np.asarray(hours, dtype=float)
    idx = resample_indices(rng, len(counts), n_samples)
    rates = counts[idx].sum(axis=1) / hours[idx].sum(axis=1)
    return percentile_interval(rates, level)


def summarize_at(at: dict, config: dict) -> dict:
    """Metrics of a configuration at one threshold (no threshold choice here).

    Output: ``sep[scenario]``: P_d within ``success_distance`` and within ``success_window`` with Wilson, delay and distance (median, p90; missed = inf),
    wrong shutdowns (runs, total), re-passes (total, with alarm off); ``none``: false alarms/h with Poisson and per-run bootstrap, by type,
    fraction of time in alarm.
    """
    mcfg, level = config["metrics"], config["metrics"]["confidence_level"]
    out = {"sep": {}, "none": {}}
    for s, d in at["sep"].items():
        n = len(d["success"])
        k = int(d["success"].sum())
        kw = int((d["delay"] <= mcfg["success_window"]).sum())
        dist = np.where(np.isnan(d["distance"]), math.inf, d["distance"])
        out["sep"][s] = {
            "n_runs": n,
            "p_detect": k / n,
            "p_interval": wilson_interval(k, n, level),
            "p_window": kw / n,
            "p_window_interval": wilson_interval(kw, n, level),
            "delay_median": nearest_rank(d["delay"], 0.5),
            "delay_p90": nearest_rank(d["delay"], 0.9),
            "distance_median": nearest_rank(dist, 0.5),
            "distance_p90": nearest_rank(dist, 0.9),
            "wrong_off_runs": int((d["n_wrong_off"] > 0).sum()),
            "wrong_off_total": int(d["n_wrong_off"].sum()),
            "repass": int(d["repass"].sum()),
            "repass_off": int((d["repass"].astype(bool) & d["repass_off"].astype(bool)).sum()),
        }
    nn = at["none"]
    hours = nn["hours"].sum()
    episodes = int(nn["episodes"].sum())
    out["none"] = {
        "n_runs": len(nn["episodes"]),
        "episodes": episodes,
        "hours": float(hours),
        "fa_per_hour": episodes / hours,
        "poisson": poisson_interval(episodes, float(hours), level),
        "bootstrap": bootstrap_rate_interval(nn["episodes"], nn["hours"], mcfg_bootstrap(config), level, bootstrap_rng(config)),
        "fa_level": float(nn["ep_level"].sum() / hours),
        "fa_lost": float(nn["ep_lost"].sum() / hours),
        "active_fraction": float(nn["active"].sum() / nn["samples"].sum()),
        "episodes_per_run": nn["episodes"].astype(int),
    }
    return out


def mcfg_bootstrap(config: dict) -> int:
    return int(config["final"]["bootstrap_samples"])


def evaluate_at(outs: list, name: str, rule: str, threshold: float, config: dict) -> dict:
    """`at_threshold` + `summarize_at`: evaluation at the GIVEN threshold (read from the frozen file). Returns ``summary`` and ``at`` (per-run data)."""
    at = at_threshold(outs, name, rule, threshold)
    return {"summary": summarize_at(at, config), "at": at}


def total_success(at: dict) -> np.ndarray:
    """Outcomes (slowdown then stop) concatenated: bool (separation runs,)."""
    return np.concatenate([at["sep"][s]["success"].astype(bool) for s in SEPARATED])


# ---------------------------------------------------------------------------
# Verdict: media versus migliore
# ---------------------------------------------------------------------------


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided p-value of the exact McNemar test on discordant runs (binomial with p = 1/2): ``min(1, 2 P(X <= min(b, c)))``."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    return float(min(1.0, 2.0 * binom.cdf(min(only_a, only_b), n, 0.5)))


def decide_verdict(success_mean: np.ndarray, success_best: np.ndarray, fa_mean: float, fa_best: float, min_net: int) -> dict:
    """Verdict fixed before seeing the data.

    Input: per-run outcomes (slowdown + stop together, same run in the same position) of `media` and `migliore` within 30 m; false alarms/h
    (point estimate) of each on the verification ``none`` runs; `min_net` (``final.min_net_gain``).
    Procedure: ``net`` = runs detected by `migliore` and not by `media`, minus those detected by `media` and not by `migliore`.
    Output: ``migliore`` is adopted only if ``net >= min_net`` AND ``fa_best <= fa_mean``; otherwise ``media``. Also reports the McNemar p-value (informative only).
    """
    gained, lost = paired_counts(success_best, success_mean)
    net = gained - lost
    adopt = "migliore" if (net >= min_net and fa_best <= fa_mean) else "media"
    return {
        "adopted": adopt,
        "net": int(net),
        "gained": int(gained),
        "lost": int(lost),
        "min_net_gain": int(min_net),
        "fa_media": float(fa_mean),
        "fa_migliore": float(fa_best),
        "mcnemar_p": mcnemar_exact(gained, lost),
    }


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------


def sensitivity_cases(config: dict) -> list:
    """Sensitivity points: list of ``(name, overrides, lost rules)``. ``base`` (no overrides) and the points of ``final.sensitivity_points``;
    ``any`` everywhere, plus ``not_deaf`` and ``not_deaf_episode`` at the points of ``final.not_deaf_points``."""
    fcfg = config["final"]
    cases = [("base", {})] + list(fcfg["sensitivity_points"].items())
    return [
        (name, point, ["any", "not_deaf", "not_deaf_episode"] if name in fcfg["not_deaf_points"] else ["any"]) for name, point in cases
    ]


def expected_long_bursts(config: dict, point: dict, duration: float) -> float | None:
    """Bursts longer than ``final.burst_long`` expected per run (all receivers): ``N · duration · L / T_b · exp(-burst_long / T_b)``; None if not ``bursty``."""
    cfg = apply_overrides(config, point)
    rc = cfg["packets"]["reception"]
    if rc["background_model"] != "bursty":
        return None
    b = rc["bursty"]
    n_scope = cfg["group"]["n_nodes"] if b["burst_scope"] == "receiver" else cfg["group"]["n_nodes"] ** 2
    return n_scope * duration * b["mean_loss"] / b["mean_burst"] * math.exp(-config["final"]["burst_long"] / b["mean_burst"])


def retuned_point(outs: list, name: str, rule: str, config: dict) -> dict | None:
    """INFORMATIVE ONLY (section D): threshold that would be chosen by retuning on the sensitivity point, with the usual rule. Never written to the frozen file."""
    rows = sweep_rows(outs, name, rule, config)
    scope = config["final"]["config_defs"][name]["detector.scope"]
    wp = working_point(rows, scope, "level", "both")
    return None if wp is None else {"threshold": float(wp["threshold"])}


def gain_loss_vs(at: dict, base_at: dict) -> dict:
    """Runs gained and lost (within 30 m) relative to the base, per scenario and total."""
    return gain_loss_counts({s: at["sep"][s]["success"] for s in SEPARATED}, {s: base_at["sep"][s]["success"] for s in SEPARATED})


def evaluate_sensitivity(case_outs: dict, final_name: str, frozen_threshold: float, config: dict) -> dict:
    """Evaluate each sensitivity point at the frozen threshold and at the retuned one.

    Input: `case_outs` = ``{point: runs}`` (the first ``final.sensitivity_runs`` runs per scenario, the same for all points), final configuration,
    frozen threshold.
    Output: ``{(point, rule): {"frozen": {summary, at}, "retuned": {threshold, summary, at} | None, "gain_loss": {...}, "diag": {...}}}``;
    ``gain_loss`` is relative to the base (rule ``any``) at the frozen threshold.
    """
    base_at = at_threshold(case_outs["base"], final_name, "any", frozen_threshold)
    out = {}
    for name, point, rules in sensitivity_cases(config):
        outs = case_outs[name]
        none = split_outs(outs)[0]
        diags = [o["diag"] for o in none]
        duration = float(np.mean([d["duration"] for d in diags]))
        longest = max((d["longest"] + (i,) for i, d in enumerate(diags)), key=lambda x: x[0])
        diag = {
            "received": int(sum(d["received"] for d in diags)),
            "trials": int(sum(d["trials"] for d in diags)),
            "n_long": int(sum(d["n_long"] for d in diags)),
            "n_bursts": int(sum(d["n_bursts"] for d in diags)),
            "deaf_episodes": int(sum(d["deaf_episodes"] for d in diags)),
            "expected_long": None,
            "longest": longest,  # (duration, receiver, instant, index of the none run)
        }
        exp = expected_long_bursts(config, point, duration)
        if exp is not None:
            diag["expected_long"] = exp * len(none)
        for rule in rules:
            frozen = evaluate_at(outs, final_name, rule, frozen_threshold, config)
            retuned = None
            wp = retuned_point(outs, final_name, rule, config)
            if wp is not None:
                retuned = {**wp, **evaluate_at(outs, final_name, rule, wp["threshold"], config)}
            lost_only = evaluate_at(outs, LOST, rule, math.nan, config)
            out[(name, rule)] = {
                "frozen": frozen,
                "retuned": retuned,
                "lost_only": lost_only,
                "gain_loss": gain_loss_vs(frozen["at"], base_at),
                "diag": diag,
            }
    return out


# ---------------------------------------------------------------------------
# Formatting (console and final_summary.md)
# ---------------------------------------------------------------------------


def _pd(p: float, interval: tuple) -> str:
    return f"{p:.2f} [{interval[0]:.2f}-{interval[1]:.2f}]"


def _ci(value: float, interval: tuple) -> str:
    return f"{value:.2f} [{interval[0]:.2f}-{interval[1]:.2f}]"


def evaluation_table(results: dict) -> str:
    """Section C table: per configuration and for "lost" alone, P_d (30 m and 30 s), delay, distance, false alarms, optimism."""
    verif, tuning, frozen = results["verification"], results["tuning"], results["frozen"]["thresholds"]
    lines = []
    head = (
        f"{'config':<9} {'sc.':<8} {'threshold':>6} {'P_d 30 m [Wilson]':>19} {'P_d 30 s [Wilson]':>19} {'del.med/p90 s':>14} {'dist.med/p90 m':>15}"
        f" {'tun.':>5} {'diff.':>6}"
    )
    lines.append(head)
    for name in [*results["config"]["final"]["configs"], LOST]:
        s = verif[name]["summary"]
        thr = frozen[name]["threshold"] if name in frozen else math.nan
        for sc in SEPARATED:
            d = s["sep"][sc]
            tun = tuning[name]["summary"]["sep"][sc]["p_detect"]
            lines.append(
                f"{name:<9} {sc:<8} {thr:>6.1f} {_pd(d['p_detect'], d['p_interval']):>19} {_pd(d['p_window'], d['p_window_interval']):>19}"
                f" {d['delay_median']:>6.1f}/{d['delay_p90']:<7.1f} {d['distance_median']:>7.1f}/{d['distance_p90']:<7.1f} {tun:>5.2f} {d['p_detect'] - tun:>+6.2f}"
            )
    lines.append("")
    lines.append(
        f"{'config':<9} {'FA/h Poisson':>19} {'FA/h bootstrap':>19} {'level/h':>9} {'lost/h':>8} {'in alarm':>10} {'FA/h tuning':>13} {'diff.':>6}"
        f" {'wrong (runs/n)':>16} {'re-passes (off)':>20}"
    )
    for name in [*results["config"]["final"]["configs"], LOST]:
        s, st = verif[name]["summary"], tuning[name]["summary"]
        n = s["none"]
        wrong = sum(s["sep"][sc]["wrong_off_runs"] for sc in SEPARATED)
        total = sum(s["sep"][sc]["wrong_off_total"] for sc in SEPARATED)
        rep = sum(s["sep"][sc]["repass"] for sc in SEPARATED)
        repo = sum(s["sep"][sc]["repass_off"] for sc in SEPARATED)
        lines.append(
            f"{name:<9} {_ci(n['fa_per_hour'], n['poisson']):>19} {_ci(n['fa_per_hour'], n['bootstrap']):>19} {n['fa_level']:>9.2f} {n['fa_lost']:>8.2f}"
            f" {100 * n['active_fraction']:>9.2f}% {st['none']['fa_per_hour']:>13.2f} {n['fa_per_hour'] - st['none']['fa_per_hour']:>+6.2f}"
            f" {wrong:>8d}/{total:<7d} {rep:>11d} ({repo})"
        )
    target = results["config"]["metrics"]["target_false_alarms_per_hour"]
    over = [n for n in results["config"]["final"]["configs"] if verif[n]["summary"]["none"]["fa_per_hour"] > target]
    lines.append(
        f"False alarm target: {target:g}/h. "
        + (f"EXCEEDED on verification by: {', '.join(over)} (thresholds not corrected, as required)." if over else "Met by all configurations on verification.")
    )
    return "\n".join(lines)


def verdict_text(v: dict) -> str:
    return (
        f"net = {v['net']:+d} (migliore gains {v['gained']}, loses {v['lost']}; threshold {v['min_net_gain']}); FA/h media {v['fa_media']:.2f}, migliore"
        f" {v['fa_migliore']:.2f}; exact McNemar p = {v['mcnemar_p']:.3f} (informative) -> final = {v['adopted']}"
    )


def sensitivity_table(results: dict) -> str:
    """Section D table: for each point and rule, frozen and retuned threshold."""
    sens, config = results["sensitivity"], results["config"]
    lines = [
        f"{'point':<11} {'rule':<16} {'threshold':>8} {'P_d slowdown':>17} {'P_d stop':>17} {'g/p sl':>7} {'g/p st':>7} {'dist.med/p90 sl':>16} {'dist.med/p90 st':>16}"
        f" {'FA/h [bootstrap]':>19} {'level':>7} {'lost':>6} {'in al.':>7}"
    ]
    for (name, rule), r in sens.items():
        for tag, block in (("frozen", r["frozen"]), ("retuned", r["retuned"])):
            label = f"{name:<11} {rule:<16}"
            if block is None:
                lines.append(f"{label} {tag:>9}  no admissible point")
                continue
            s = block["summary"]
            sl, st, nn = s["sep"]["slowdown"], s["sep"]["stop"], s["none"]
            thr = block.get("threshold", results["frozen"]["thresholds"][results["final_name"]]["threshold"])
            gl = r["gain_loss"] if tag == "frozen" else gain_loss_vs(block["at"], sens[("base", "any")]["frozen"]["at"])
            lines.append(
                f"{label} {tag:>9} {thr:>5.1f} {_pd(sl['p_detect'], sl['p_interval']):>17} {_pd(st['p_detect'], st['p_interval']):>17}"
                f" {'+%d/-%d' % gl['slowdown']:>7} {'+%d/-%d' % gl['stop']:>7}"
                f" {sl['distance_median']:>7.1f}/{sl['distance_p90']:<7.1f} {st['distance_median']:>7.1f}/{st['distance_p90']:<7.1f}"
                f" {_ci(nn['fa_per_hour'], nn['bootstrap']):>19} {nn['fa_level']:>7.2f} {nn['fa_lost']:>6.2f} {100 * nn['active_fraction']:>6.2f}%"
            )
    lines.append("")
    lines.append(f"{'point':<11} {'beacons received':>16} {'bursts > ' + format(config['final']['burst_long'], 'g') + ' s observed':>24} {'expected':>8} {'deafness episodes':>19}")
    for name, _, _ in sensitivity_cases(config):
        d = sens[(name, "any")]["diag"]
        exp = "-" if d["expected_long"] is None else f"{d['expected_long']:.1f}"
        obs = "-" if d["expected_long"] is None else str(d["n_long"])
        lines.append(f"{name:<11} {100 * d['received'] / d['trials']:>15.2f}% {obs:>24} {exp:>8} {d['deaf_episodes']:>19d}")
    return "\n".join(lines)


def write_final_summary(results: dict, path: Path, versions: dict) -> None:
    """Write ``final_summary.md`` from the results (no hand-written numbers): final config, thresholds, evaluation, verdict, sensitivity, versions."""
    config, frozen = results["config"], results["frozen"]
    final = results["final_name"]
    dcfg = config["detector"]
    doc = [
        "# Final summary (generated by `scripts/reproduce_all.py`)",
        "",
        f"Commit: `{versions['commit']}`{' (with uncommitted changes)' if versions['dirty'] else ''}  ",
        f"Python {versions['python']}; " + ", ".join(f"{k} {v}" for k, v in versions["libraries"].items()),
        "",
        "## Final configuration",
        "",
        f"- configuration: **{final}** (scope `{frozen['thresholds'][final]['scope']}`, fusion rule `{frozen['thresholds'][final]['fusion_rule']}`)",
        f"- frozen level threshold: **{frozen['thresholds'][final]['threshold']:g} dBm** (tuning: {frozen['date']}, commit `{frozen['commit']}`)",
        f"- release: {dcfg['release_time']:g} s; `sigma_a` {config['kalman']['sigma_a']:g}; `min_observers` {dcfg['min_observers']};"
        f" `lost_rule` {dcfg['lost_rule']}; shadowing `{config['channel']['shadowing']['mode']}`",
        "",
        "## Frozen thresholds (on the tuning set)",
        "",
        "| configuration | threshold (dBm) | P_d slowdown | P_d stop | FA/h | time in alarm |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, f in frozen["thresholds"].items():
        t = f["tuning"]
        doc.append(f"| {name} | {f['threshold']:g} | {t['p_detect_slowdown']:.2f} | {t['p_detect_stop']:.2f} | {t['fa_per_hour']:.2f} | {100 * t['active_fraction']:.2f}% |")
    doc += ["", "## Evaluation on the verification runs", "", "```", evaluation_table(results), "```", "", "## Verdict: media versus migliore", "", verdict_text(results["verdict"]), ""]
    doc += ["## Sensitivity", "", "```", sensitivity_table(results), "```", ""]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(doc))

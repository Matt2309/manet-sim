#!/usr/bin/env python3
"""Ispezione visiva del Blocco 5 (rilevatore rifinito).

Legge ``results/block5/results.pkl`` (da ``run_block5_experiment.py``) e rifà alcune corse dalla cache (corse 26 e 30 di ``stop``,
la corsa con lo spegnimento a torto più lungo a release_time = 5 s, la corsa ``none`` con il congelamento più lungo). Salva sette figure in ``results/block5/``.
Config: config/default.yaml se non indicata.
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2

from src.detector import build_inputs, lost_alarm, pre_alarm, system_alarm
from src.metrics import min_distance_to_group
from src.mobility import load_config
from src.refinement import SEPARATED, apply_overrides, load_pickle
from src.relative import relative_trace
from src.simulator import experiment_runs, load_or_simulate

BLUE, ORANGE, TEAL, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GREY = "#9a9892"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e3e2dd"
BEFORE_S = 60.0  # s prima del distacco nelle figure di dettaglio
AFTER_S = 700.0  # s dopo il distacco nelle figure di dettaglio
ZERO_FA_PLOT = 0.03  # un'ascissa log non rappresenta zero: i falsi allarmi nulli si disegnano qui (solo grafica)
LABELS = {"slowdown": "slowdown", "stop": "stop"}


def style_axes(ax):
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.title.set_color(INK)


def curve(summary, scenario):
    """(falsi allarmi/h, P_d) lungo la soglia, per una variante e uno scenario."""
    rows = sorted((r for r in summary["rows"] if r["scenario"] == scenario), key=lambda r: r["threshold"])
    fa = np.array([max(r["fa_per_hour"], ZERO_FA_PLOT) for r in rows])
    return fa, np.array([r["p_detect"] for r in rows])


def tradeoff_axes(ax, config, scenario):
    ax.set_xscale("log")
    ax.axvline(config["metrics"]["target_false_alarms_per_hour"], color=INK, linestyle=":", linewidth=1.0)
    ax.set_xlabel("falsi allarmi per ora (corse none)")
    ax.set_ylabel(f"P_d entro {config['metrics']['success_distance']:g} m")
    ax.set_title(LABELS[scenario], fontsize=11)
    ax.set_ylim(-0.02, 1.03)
    style_axes(ax)


def plot_curves(ax, summaries, scenario, items, config):
    for summary, color, style, label in items:
        fa, pd = curve(summary, scenario)
        ax.plot(fa, pd, style, color=color, linewidth=1.6, label=label)
        if summary["wp"] is not None:
            p = summary["point"][scenario]
            ax.plot([max(p["fa_per_hour"], ZERO_FA_PLOT)], [p["p_detect"]], "o", color=color, ms=5)
    tradeoff_axes(ax, config, scenario)


# ---------------------------------------------------------------------------
# Figura 1
# ---------------------------------------------------------------------------


def fig_fusion_rules(res, path):
    S, config = res["summaries"], res["config"]
    colors = {"weighted_mean": GREY, "median": BLUE, "max": ORANGE}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, scenario in zip(axes, SEPARATED):
        items = []
        for rule, color in colors.items():
            name = "base" if rule == "weighted_mean" else f"A:{rule}"
            for mo, style in ((1, "-"), (2, "--")):
                items.append((S[(name, "fused", mo)], color, style, f"{rule}, min_observers = {mo}"))
        plot_curves(ax, S, scenario, items, config)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Regole di fusione (grigio = base: media pesata); linea verticale = target falsi allarmi/h; punto = punto di lavoro", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Corse di dettaglio
# ---------------------------------------------------------------------------


def sep_run(config, cache_dir, scenario, index):
    info = next(r for r in experiment_runs(config) if r["scenario"] == scenario and r["index"] == index)
    return info, load_or_simulate(info["config"], cache_dir)


def stop_run(config, cache_dir, index):
    return sep_run(config, cache_dir, "stop", index)


def window(t, start, end=None):
    hi = min(t[-1], start + AFTER_S) if end is None else end
    return (t >= start - BEFORE_S) & (t <= hi)


def fused_series(res, config, cache_dir, index):
    """Per la corsa `index` di ``stop``: indice fuso assoluto e calo fuso relativo (per osservatore) e distanza dal gruppo."""
    info, run = stop_run(config, cache_dir, index)
    node = run["sep_node"]
    R = res["R"]
    base_cfg = apply_overrides(config, res["variants"]["base"]["overrides"])
    rel_cfg = apply_overrides(config, res["variants"]["B:relative"]["overrides"])
    inputs_abs, _ = build_inputs(run, base_cfg, base_cfg["kalman"]["sigma_a"], R)
    inputs_rel, _ = build_inputs(run, rel_cfg, rel_cfg["kalman"]["sigma_a"], R)
    thr = res["summaries"][("B:relative", "fused", 1)]["wp"]["threshold"]
    sliced = inputs_rel.target_slice(node)
    drops = np.stack([relative_trace(sliced, rel_cfg, "fused", thr, m, 0)["drop"] for m in range(run["received"].shape[0]) if m != node], axis=1)
    level = inputs_abs.level["fused"][:, [m for m in range(run["received"].shape[0]) if m != node], node]
    return info, run, level, drops, min_distance_to_group(run["distances"], node), thr


def repass_interval(t, distance, start, edge):
    """Primo ripassaggio dopo il distacco: (inizio, fine) in s, o None."""
    k = int(np.searchsorted(t, start))
    far = np.flatnonzero(distance[k:] > edge)
    if not len(far):
        return None
    c = k + int(far[0])
    back = np.flatnonzero(distance[c:] < edge)
    if not len(back):
        return None
    first = c + int(back[0])
    leave = np.flatnonzero(distance[first:] >= edge)
    return t[first], t[first + int(leave[0])] if len(leave) else t[-1]


# ---------------------------------------------------------------------------
# Figura 2
# ---------------------------------------------------------------------------


def fig_relative_vs_absolute(res, cache_dir, path):
    S, config = res["summaries"], res["config"]
    edge = config["metrics"]["success_distance"]
    refs = [(sc, i) for sc, idxs in config["metrics"]["reference_runs"].items() for i in idxs]
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(3, max(2, len(refs)), height_ratios=[1.3, 1, 1])
    for col, scenario in enumerate(SEPARATED):
        ax = fig.add_subplot(gs[0, col])
        plot_curves(ax, S, scenario, [(S[("base", "fused", 1)], GREY, "-", "assoluto (base)"), (S[("B:relative", "fused", 1)], BLUE, "-", "relativo")], config)
        if col == 0:
            ax.legend(frameon=False, fontsize=9, loc="lower right")
    for col, (scenario, index) in enumerate(refs):
        info, run, level, drops, distance, thr = fused_series(res, config, cache_dir, index)
        t, start = run["t"], run["sep_start"]
        base_thr = S[("base", "fused", 1)]["wp"]["threshold"]
        span = repass_interval(t, distance, start, edge)
        sel = window(t, start, None if span is None else span[1] + 60.0)
        for row, (series, ylabel, color, thr_line) in enumerate(((level, "indice fuso assoluto (dBm)", ORANGE, base_thr), (drops, "calo fuso (dB)", BLUE, thr))):
            ax = fig.add_subplot(gs[1 + row, col])
            for m in range(series.shape[1]):
                ax.plot(t[sel], series[sel, m], color=color, linewidth=0.9, alpha=0.55)
            ax.axhline(thr_line, color=color, linestyle="--", linewidth=1.0)
            ax.axvline(start, color=INK, linestyle=":", linewidth=1.0)
            ax.set_ylabel(ylabel)
            style_axes(ax)
            ax2 = ax.twinx()
            ax2.plot(t[sel], distance[sel], color=INK_2, linewidth=1.2)
            ax2.axhline(edge, color=INK_2, linestyle=":", linewidth=0.8)
            ax2.set_ylabel("distanza dal gruppo (m)", color=INK_2)
            ax2.tick_params(colors=INK_2, labelsize=9)
            ax2.spines["top"].set_visible(False)
            if row == 0:
                ax.set_title(f"corsa {scenario} {index}: distacco a {start:.0f} s", fontsize=10)
            else:
                ax.set_xlabel("tempo (s)")
    fig.suptitle("Livello relativo contro assoluto: curve (sopra) e corse di riferimento (sotto, un tratto per osservatore; linea tratteggiata = soglia del punto di lavoro)", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figura 3
# ---------------------------------------------------------------------------


def shade(ax, t, flag, color):
    edges = np.diff(np.concatenate([[0], flag.astype(int), [0]]))
    first = True
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        ax.axvspan(t[a], t[min(b, len(t) - 1)], color=color, alpha=0.18, linewidth=0, label="base congelata" if first else None)
        first = False


def fig_baseline_example(res, cache_dir, path):
    config = res["config"]
    S = res["summaries"]
    rel_cfg = apply_overrides(config, res["variants"]["B:relative"]["overrides"])
    thr = S[("B:relative", "fused", 1)]["wp"]["threshold"]
    R = res["R"]
    index = config["metrics"]["reference_runs"]["stop"][0]
    info, run = stop_run(config, cache_dir, index)
    node, start, t = run["sep_node"], run["sep_start"], run["t"]
    inputs, _ = build_inputs(run, rel_cfg, rel_cfg["kalman"]["sigma_a"], R)
    sliced = inputs.target_slice(node)
    m = (node + 1) % run["received"].shape[0]
    tr = relative_trace(sliced, rel_cfg, "fused", thr, m, 0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    sel = window(t, start)
    ax.plot(t[sel], np.where(sliced.entries["valid"][sel, m, m], sliced.entries["value"][sel, m, m], np.nan), color=GREY, linewidth=1.0, label=f"stima del link {node} → {m}")
    ax.plot(t[sel], tr["base"][sel, m], color=BLUE, linewidth=1.8, label="linea di base")
    shade(ax, t[sel], tr["frozen"][sel, m], ORANGE)
    ax.axvline(start, color=INK, linestyle=":", linewidth=1.0)
    ax.set_title(f"corsa stop {index}: link dal nodo separato", fontsize=11)
    ax.set_xlabel("tempo (s)")
    ax.set_ylabel("RSSI (dBm)")
    ax.legend(frameon=False, fontsize=9)
    style_axes(ax)

    diag = res["diag"].get("B:relative")
    ax = axes[1]
    if diag:
        none_runs = [r for r in experiment_runs(config) if r["scenario"] == "none"]
        worst = int(np.argmax([r["max_freeze"] for r in diag["per_run"]]))
        pm, pi, longest, _ = max(diag["per_run"][worst]["pairs"], key=lambda p: p[2])
        nrun = load_or_simulate(none_runs[worst]["config"], cache_dir)
        ninputs, _ = build_inputs(nrun, rel_cfg, rel_cfg["kalman"]["sigma_a"], R)
        ntr = relative_trace(ninputs, rel_cfg, "fused", thr, pm, pi)
        k = int(np.argmax([_longest(ntr["frozen"][:, kk]) for kk in range(ntr["frozen"].shape[1])]))
        nt = nrun["t"]
        ax.plot(nt, np.where(ninputs.entries["valid"][:, pm, pi, k], ninputs.entries["value"][:, pm, pi, k], np.nan), color=GREY, linewidth=0.8, label=f"stima del link {pi} → {k}")
        ax.plot(nt, ntr["base"][:, k], color=BLUE, linewidth=1.6, label="linea di base")
        shade(ax, nt, ntr["frozen"][:, k], ORANGE)
        ax.set_title(f"corsa none {none_runs[worst]['index']}: congelamento più lungo ({longest:.0f} s), osservatore {pm}", fontsize=11)
        ax.set_xlabel("tempo (s)")
        ax.set_ylabel("RSSI (dBm)")
        ax.legend(frameon=False, fontsize=9)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def _longest(flag):
    best = cur = 0
    for x in flag:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return best


# ---------------------------------------------------------------------------
# Figura 4
# ---------------------------------------------------------------------------


def system_alarm_for(res, run, name):
    """Allarme di sistema (T,) sul nodo separato per la variante `name`, alla soglia del suo punto di lavoro."""
    vcfg = apply_overrides(res["config"], res["variants"][name]["overrides"])
    inputs, _ = build_inputs(run, vcfg, vcfg["kalman"]["sigma_a"], res["R"])
    sliced = inputs.target_slice(run["sep_node"])
    thr = res["summaries"][(name, "fused", 1)]["wp"]["threshold"]
    t = run["t"]
    pre = pre_alarm(sliced.level["fused"], sliced.slope["fused"], sliced.rx_ok, t, vcfg["detector"], "level", level_threshold=thr, valid=sliced.valid)
    lost = lost_alarm(sliced.silence, sliced.rx_ok, t, vcfg["detector"], sliced.valid)
    return vcfg, system_alarm(pre | lost, 1)


def longest_wrong_off(res, cache_dir):
    """Fra le corse con spegnimenti a torto della variante base: quella con lo spegnimento più lungo.

    Restituisce (scenario, indice, t_spegnimento, t_riaccensione, durata) in s; la riaccensione è la fine della corsa se l'allarme non riparte.
    """
    config = res["config"]
    per_run = res["summaries"][("base", "fused", 1)]["per_run"]
    best = None
    for scenario in SEPARATED:
        indices = sorted(r["index"] for r in experiment_runs(config) if r["scenario"] == scenario)
        for pos, index in enumerate(indices):
            n_wrong = int(per_run[scenario]["n_wrong_off"][pos])
            if n_wrong == 0:
                continue
            _, run = sep_run(config, cache_dir, scenario, index)
            _, alarm = system_alarm_for(res, run, "base")
            t = run["t"]
            k = int(np.searchsorted(t, run["sep_start"]))
            k += int(np.flatnonzero(alarm[k:])[0])
            falls = k + 1 + np.flatnonzero(alarm[k:-1] & ~alarm[k + 1 :])
            assert len(falls) == n_wrong, (scenario, index, len(falls), n_wrong)
            for f in falls:
                on = np.flatnonzero(alarm[f:])
                t_on = t[f + int(on[0])] if len(on) else t[-1]
                print(f"  spegnimento a torto: {scenario} {index}: {t[f]:.0f} -> {t_on:.0f} s ({t_on - t[f]:.0f} s)")
                if best is None or t_on - t[f] > best[4]:
                    best = (scenario, index, float(t[f]), float(t_on), float(t_on - t[f]))
    return best


def fig_release_turnaround(res, cache_dir, path):
    S, config = res["summaries"], res["config"]
    edge = config["metrics"]["success_distance"]
    scenario, index, t_off, t_on, duration = longest_wrong_off(res, cache_dir)
    info, run = sep_run(config, cache_dir, scenario, index)
    node, start, t = run["sep_node"], run["sep_start"], run["t"]
    distance = min_distance_to_group(run["distances"], node)
    span = repass_interval(t, distance, start, edge)
    hi = start + AFTER_S if span is None else span[1] + 60.0
    hi = max(hi, t_on + 60.0)
    sel = (t >= start - BEFORE_S) & (t <= hi)
    names = ["base"] + [n for n in res["variants"] if n.startswith("C:")]
    fig, axes = plt.subplots(len(names) + 1, 1, figsize=(11, 1.5 * (len(names) + 1) + 2), sharex=True, gridspec_kw={"height_ratios": [2] + [1] * len(names)})
    axes[0].plot(t[sel], distance[sel], color=INK_2, linewidth=1.4)
    axes[0].axhline(edge, color=INK_2, linestyle=":", linewidth=0.8)
    axes[0].set_ylabel("distanza dal gruppo (m)")
    axes[0].set_title(
        f"corsa {scenario} {index}: spegnimento a torto più lungo con release_time = 5 s ({duration:.0f} s, da {t_off:.0f} s)"
        + ("" if span is None else f"; il gruppo ripassa a {span[0]:.0f}-{span[1]:.0f} s"),
        fontsize=11,
    )
    if span is not None:
        for ax in axes:
            ax.axvspan(span[0], span[1], color=YELLOW, alpha=0.15, linewidth=0)
    for ax, name in zip(axes[1:], names):
        vcfg, alarm = system_alarm_for(res, run, name)
        ax.fill_between(t[sel], 0, alarm[sel].astype(float), step="mid", color=ORANGE, alpha=0.7, linewidth=0)
        ax.set_ylim(0, 1.1)
        ax.set_yticks([])
        ax.set_ylabel(f"{vcfg['detector']['release_time']:g} s", rotation=0, ha="right", va="center")
        ax.axvline(start, color=INK, linestyle=":", linewidth=1.0)
    for ax in axes:
        style_axes(ax)
    axes[0].axvline(start, color=INK, linestyle=":", linewidth=1.0)
    for ax in axes:
        ax.axvline(t_off, color=ORANGE, linestyle="--", linewidth=1.2)
    axes[-1].set_xlabel("tempo (s)")
    fig.suptitle("Allarme di sistema per ogni release_time (soglia del punto di lavoro di ciascuno)", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figura 5
# ---------------------------------------------------------------------------


def fig_robust(res, path):
    S, config, nis = res["summaries"], res["config"], res["nis"]
    labels = list(nis["hist"])
    edges = nis["edges"]
    centres = 0.5 * (edges[1:] + edges[:-1])
    width = edges[1] - edges[0]
    colors = [GREY, BLUE, TEAL, ORANGE]
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 4)
    for col, label in enumerate(labels):
        ax = fig.add_subplot(gs[0, col])
        h = nis["hist"][label].astype(float)
        ax.bar(centres, h / (h.sum() * width), width=width, color=colors[col % 4], alpha=0.8, linewidth=0)
        x = np.linspace(max(edges[0], 0.02), edges[-1], 400)
        ax.plot(x, chi2.pdf(x, 1), color=INK, linewidth=1.2, label="chi quadro, 1 gdl")
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 10)
        ax.set_title(f"{label}: media {nis['stats'][label]['mean']:.2f}, p99 {nis['stats'][label]['q']:.1f}", fontsize=10)
        ax.set_xlabel("NIS")
        style_axes(ax)
        if col == 0:
            ax.legend(frameon=False, fontsize=8)
            ax.set_ylabel("densità")
    sets = {
        "assoluto": [("base", GREY, "nessuna"), ("D:inflate=2", BLUE, "inflate 2"), ("D:inflate=4", TEAL, "inflate 4"), ("D:gate", ORANGE, "nis_gate")],
        "relativo": [("B:relative", GREY, "nessuna"), ("D:inflate=2+rel", BLUE, "inflate 2"), ("D:inflate=4+rel", TEAL, "inflate 4"), ("D:gate+rel", ORANGE, "nis_gate")],
    }
    panels = [(r, s) for r in sets for s in SEPARATED]
    for col, (ref, scenario) in enumerate(panels):
        ax = fig.add_subplot(gs[1, col])
        plot_curves(ax, S, scenario, [(S[(n, "fused", 1)], c, "-", lb) for n, c, lb in sets[ref]], config)
        ax.set_title(f"{ref}, {scenario}", fontsize=10)
        if col == 0:
            ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figura 6
# ---------------------------------------------------------------------------


def fig_shadowing_mode(res, path):
    shadow, config = res["shadow"], res["config"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    series = [("base", "fused", BLUE, ""), ("base", "pairwise", BLUE, "//"), ("final", "fused", ORANGE, ""), ("final", "pairwise", ORANGE, "//")]
    modes = ["field", "independent"]
    width = 0.2
    for ax, scenario in zip(axes, SEPARATED):
        for j, (tag, scope, color, hatch) in enumerate(series):
            vals, errs = [], []
            for mode in modes:
                s = shadow[mode][tag][scope]
                if s["wp"] is None:
                    vals.append(0.0)
                    errs.append([0.0, 0.0])
                    continue
                p = s["point"][scenario]
                vals.append(p["p_detect"])
                errs.append([p["p_detect"] - p["p_lo"], p["p_hi"] - p["p_detect"]])
            x = np.arange(len(modes)) + (j - 1.5) * width
            ax.bar(x, vals, width=width, color=color, hatch=hatch, alpha=0.85, edgecolor="white", yerr=np.array(errs).T, ecolor=INK_2, capsize=2, label=f"{tag}, {scope}")
        ax.set_xticks(np.arange(len(modes)))
        ax.set_xticklabels(["shadowing condiviso (field)", "shadowing indipendente"])
        ax.set_ylim(0, 1.08)
        ax.set_title(scenario, fontsize=11)
        ax.set_ylabel(f"P_d entro {config['metrics']['success_distance']:g} m al punto di lavoro")
        style_axes(ax)
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figura 7
# ---------------------------------------------------------------------------


def fig_final_comparison(res, path):
    S, config = res["summaries"], res["config"]
    base = S[("base", "fused", 1)]
    final = S[(res["final_name"], "fused", res["final_mo"])]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, scenario in enumerate(SEPARATED):
        ax = axes[0, col]
        plot_curves(ax, S, scenario, [(base, GREY, "-", "base"), (final, ORANGE, "-", "finale")], config)
        if col == 0:
            ax.legend(frameon=False, fontsize=9, loc="lower right")
        ax = axes[1, col]
        for summary, color, label in ((base, GREY, "base"), (final, ORANGE, "finale")):
            if summary["wp"] is None:
                continue
            d = np.sort(np.where(np.isnan(summary["per_run"][scenario]["distance"]), np.inf, summary["per_run"][scenario]["distance"]))
            y = np.arange(1, len(d) + 1) / len(d)
            ax.step(np.concatenate([[0], d[np.isfinite(d)]]), np.concatenate([[0], y[np.isfinite(d)]]), where="post", color=color, linewidth=1.8, label=label)
        ax.axvline(config["metrics"]["success_distance"], color=INK, linestyle=":", linewidth=1.0)
        ax.set_xlabel("distanza dal gruppo al rilevamento (m)")
        ax.set_ylabel("frazione cumulata di corse")
        ax.set_title(f"{scenario}: distanza al rilevamento (ECDF)", fontsize=11)
        ax.set_ylim(0, 1.03)
        style_axes(ax)
        if col == 0:
            ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default=str(_REPO_ROOT / "config" / "default.yaml"))
    parser.add_argument("--cache-dir", default=str(_REPO_ROOT / "results" / "cache"))
    parser.add_argument("--out-dir", default=str(_REPO_ROOT / "results" / "block5"))
    args = parser.parse_args()
    out = Path(args.out_dir)
    cache_dir = Path(args.cache_dir)
    res = load_pickle(out / "results.pkl")
    fig_fusion_rules(res, out / "fusion_rules.png")
    fig_relative_vs_absolute(res, cache_dir, out / "relative_vs_absolute.png")
    fig_baseline_example(res, cache_dir, out / "baseline_example.png")
    fig_release_turnaround(res, cache_dir, out / "release_turnaround.png")
    fig_robust(res, out / "robust.png")
    fig_shadowing_mode(res, out / "shadowing_mode.png")
    fig_final_comparison(res, out / "final_comparison.png")
    print(f"Figure salvate in {out}")


if __name__ == "__main__":
    main()

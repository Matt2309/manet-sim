#!/usr/bin/env python3
"""Block 6 figures for the thesis (``results/block6/``).

Reads ``<out-dir>/results.pkl`` (from ``run_block6_experiment.py``); the config is the one stored in it (so a run made with ``--set`` is plotted consistently).
Without sensitivity (``--no-sensitivity``) the sensitivity and burst figures are skipped. Labels in Italian, readable in black and white (different line styles and
markers besides color), no long titles in the figure. The ``burst_example`` figure redoes from the cache a run with a long burst.
"""

import math
from pathlib import Path

import numpy as np

from _cli import make_parser  # noqa: E402
from src.common.config import apply_overrides
from src.common.io import load_pickle
from src.common.paths import block_dir
from src.common.plotting import GRID, INK, INK_2, plt, save_figure, SCEN_LABEL, style_axes, ZERO_FA_PLOT
from src.detector import deafness, lost_alarm, reception_features
from src.final import sensitivity_cases
from src.metrics import SEPARATED
from src.simulator import run_from_channel


COLORS = {"media": "#2a78d6", "migliore": "#eb6834", "coppia": "#52514e"}
STYLES = {"media": "-", "migliore": "--", "coppia": ":"}
MARKERS = {"media": "o", "migliore": "s", "coppia": "^"}
DPI = 200


def figure_tradeoff(res, out):
    config = res["config"]
    target = config["metrics"]["target_false_alarms_per_hour"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, scenario in zip(axes, SEPARATED):
        style_axes(ax)
        ax.set_xscale("log")
        for name in res["active_configs"]:
            rows = sorted((r for r in res["sweeps"]["verification"][name] if r["scenario"] == scenario), key=lambda r: r["threshold"])
            fa = np.array([max(r["fa_per_hour"], ZERO_FA_PLOT) for r in rows])
            ax.plot(fa, [r["p_detect"] for r in rows], STYLES[name], color=COLORS[name], linewidth=1.6, label=name)
            frozen = res["verification"][name]["summary"]
            ax.plot(
                [max(frozen["none"]["fa_per_hour"], ZERO_FA_PLOT)], [frozen["sep"][scenario]["p_detect"]], MARKERS[name], color=COLORS[name],
                markersize=9, markeredgecolor=INK, markeredgewidth=1.2, zorder=5,
            )  # fmt: skip
        ax.axvline(target, color=INK, linestyle="-.", linewidth=1.0)
        ax.set_xlabel("falsi allarmi all'ora (scala log)")
        ax.set_ylim(-0.02, 1.03)
        ax.set_title(SCEN_LABEL[scenario], fontsize=10, color=INK)
    axes[0].set_ylabel("$P_d$ entro 30 m")
    axes[0].legend(frameon=False, fontsize=9, loc="lower right", title="curve; marcatore = soglia congelata", title_fontsize=8)
    save_figure(fig, out / "final_tradeoff.png", DPI)


def _cdf(values, xs):
    v = np.where(np.isnan(values), math.inf, values)
    return np.array([(v <= x).mean() for x in xs])


def figure_distance(res, out):
    xs = np.linspace(0, 60, 241)
    final = res["final_name"]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    style_axes(ax)
    for name in dict.fromkeys(n for n in (final, "coppia") if n in res["active_configs"]):
        for scenario, ls in zip(SEPARATED, ("-", "--")):
            d = res["verification"][name]["at"]["sep"][scenario]["distance"]
            ax.plot(xs, _cdf(d, xs), ls, color=COLORS[name], linewidth=2.0 if name == final else 1.5, label=f"{name}, {SCEN_LABEL[scenario]}")
    ax.axvline(res["config"]["metrics"]["success_distance"], color=INK, linestyle=":", linewidth=1.2)
    ax.set_xlabel("distanza dal gruppo al rilevamento (m)")
    ax.set_ylabel("frazione di distacchi rilevati")
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 1.03)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    save_figure(fig, out / "detection_distance.png", DPI)


def figure_delay(res, out):
    final = res["final_name"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharex=True, sharey=True)
    for ax, scenario in zip(axes, SEPARATED):
        style_axes(ax)
        d = res["verification"][final]["at"]["sep"][scenario]["delay"]
        finite = d[np.isfinite(d)]
        edges = np.arange(0, max(60.0, math.ceil(finite.max() / 5.0) * 5.0 + 5.0), 5.0) if len(finite) else np.arange(0, 65, 5.0)
        ax.hist(finite, bins=edges, color="#9a9892", edgecolor=INK, linewidth=0.8, hatch="//")
        s = res["verification"][final]["summary"]["sep"][scenario]
        ax.axvline(s["delay_median"], color=INK, linestyle="-", linewidth=1.4, label=f"mediana {s['delay_median']:.1f} s")
        ax.axvline(s["delay_p90"], color=INK, linestyle="--", linewidth=1.4, label=f"p90 {s['delay_p90']:.1f} s")
        ax.set_xlabel("ritardo dall'inizio del distacco (s)")
        ax.set_title(SCEN_LABEL[scenario], fontsize=10, color=INK)
        ax.legend(frameon=False, fontsize=9)
        missed = int((~np.isfinite(d)).sum())
        if missed:
            ax.text(0.98, 0.55, f"{missed} mai rilevati", transform=ax.transAxes, ha="right", fontsize=9, color=INK_2)
    axes[0].set_ylabel("numero di corse")
    save_figure(fig, out / "detection_delay.png", DPI)


def _errbar(ax, x, y, lo, hi, marker, filled, color="#0b0b0b", offset=0.0, **kw):
    ax.errorbar(
        [x + offset], [y], yerr=[[max(y - lo, 0)], [max(hi - y, 0)]], fmt=marker, color=color, markerfacecolor=color if filled else "white",
        markersize=7, capsize=3, linewidth=1.2, **kw,
    )  # fmt: skip


def figure_tuning_vs_test(res, out):
    names = res["active_configs"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
    panels = [("slowdown", "$P_d$ entro 30 m, rallenta"), ("stop", "$P_d$ entro 30 m, si ferma"), (None, "falsi allarmi all'ora")]
    for ax, (scenario, label) in zip(axes, panels):
        style_axes(ax)
        for i, name in enumerate(names):
            for tag, filled, off in (("tuning", False, -0.12), ("verification", True, 0.12)):
                s = res[tag][name]["summary"]
                if scenario:
                    d = s["sep"][scenario]
                    _errbar(ax, i, d["p_detect"], *d["p_interval"], MARKERS[name], filled, offset=off)
                else:
                    n = s["none"]
                    _errbar(ax, i, n["fa_per_hour"], *n["bootstrap"], MARKERS[name], filled, offset=off)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names)
        ax.set_xlim(-0.6, len(names) - 0.4)
        ax.set_ylabel(label)
        if scenario:
            ax.set_ylim(0.4, 1.03)
        else:
            ax.axhline(res["config"]["metrics"]["target_false_alarms_per_hour"], color=INK, linestyle="-.", linewidth=1.0)
            ax.set_ylim(bottom=0)
    axes[0].plot([], [], "o", mfc="white", color=INK, label="taratura")
    axes[0].plot([], [], "o", color=INK, label="verifica")
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    save_figure(fig, out / "tuning_vs_test.png", DPI)


def figure_sensitivity(res, out):
    sens = res["sensitivity"]
    cases = [(name, rule) for name, _, rules in sensitivity_cases(res["config"]) for rule in rules]
    labels = [name if rule == "any" else f"{name} ({rule.replace('_', ' ')})" for name, rule in cases]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), sharey=True)
    panels = [("stop", "$P_d$ entro 30 m, si ferma"), ("slowdown", "$P_d$ entro 30 m, rallenta"), (None, "falsi allarmi all'ora")]
    ys = np.arange(len(cases))[::-1]
    for ax, (scenario, label) in zip(axes, panels):
        style_axes(ax)
        ax.axhspan(ys[0] - 0.5, ys[0] + 0.5, color=GRID, alpha=0.8, zorder=0)  # the base highlighted
        for y, key in zip(ys, cases):
            r = sens[key]
            for block, filled, off in ((r["frozen"], True, 0.13), (r["retuned"], False, -0.13)):
                if block is None:
                    continue
                s = block["summary"]
                if scenario:
                    d = s["sep"][scenario]
                    val, lo, hi = d["p_detect"], *d["p_interval"]
                else:
                    val, (lo, hi) = s["none"]["fa_per_hour"], s["none"]["bootstrap"]
                ax.errorbar([val], [y + off], xerr=[[max(val - lo, 0)], [max(hi - val, 0)]], fmt="o", color=INK, markerfacecolor=INK if filled else "white",
                            markersize=6, capsize=2.5, linewidth=1.1)  # fmt: skip
        ax.set_xlabel(label)
        if scenario:
            ax.set_xlim(0.3, 1.03)
        else:
            ax.axvline(res["config"]["metrics"]["target_false_alarms_per_hour"], color=INK, linestyle="-.", linewidth=1.0)
            ax.set_xlim(left=0)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(labels, fontsize=9)
    axes[0].plot([], [], "o", color=INK, label="soglia congelata")
    axes[0].plot([], [], "o", mfc="white", color=INK, label="soglia ritarata")
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    save_figure(fig, out / "sensitivity.png", DPI)


def figure_burst_example(res, config, cache_dir, out):
    sens = res["sensitivity"]
    cases = {n: p for n, p, _ in sensitivity_cases(config)}
    bursty = [n for n in cases if config["final"]["sensitivity_points"].get(n, {}).get("packets.reception.background_model") == "bursty"]
    best = max(bursty, key=lambda n: sens[(n, "any")]["diag"]["longest"][0])
    length, node, start, index = sens[(best, "any")]["diag"]["longest"]
    from src.final import run_sets

    _, verification = run_sets(config)
    info = next(r for r in verification if r["scenario"] == "none" and r["index"] == index)
    rcfg = apply_overrides(info["config"], cases[best])
    run = run_from_channel(rcfg, cache_dir)
    t = run["t"]
    n = run["received"].shape[0]
    silence, rx_ok = reception_features(t, run["beacon_times"], run["received"], config)
    deaf_window, deaf_episode = deafness(t, run["beacon_times"], run["received"], silence, config)
    valid = ~np.eye(n, dtype=bool)
    alarms = {}
    for rule in ("any", "not_deaf", "not_deaf_episode"):
        d = {**config["detector"], "lost_rule": rule}
        alarms[rule] = lost_alarm(silence, rx_ok, t, d, valid, deaf_window=deaf_window, deaf_episode=deaf_episode)[:, node, :]
    lo, hi = start - 25.0, start + length + 45.0
    sel = (t >= lo) & (t <= hi)
    others = [k for k in range(n) if k != node]
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.2), sharex=True, gridspec_kw={"height_ratios": [1.2, 1.2, 1.4]})
    state = run["background_state"][:, node]
    for ax in axes:
        style_axes(ax)
        ax.fill_between(t[sel], 0, 1, where=state[sel], transform=ax.get_xaxis_transform(), color="#9a9892", alpha=0.35, linewidth=0)
    for row, k in enumerate(others):
        rx = run["beacon_times"][k][run["received"][k, :, node]]
        rx = rx[(rx >= lo) & (rx <= hi)]
        axes[0].plot(rx, np.full(len(rx), row), "|", color=INK, markersize=9)
    axes[0].set_yticks(range(len(others)))
    axes[0].set_yticklabels([f"vicino {k}" for k in others], fontsize=9)
    axes[0].set_ylabel("beacon ricevuti")
    styles = ["-", "--", "-.", ":"]
    for ls, k in zip(styles, others):
        axes[1].plot(t[sel], silence[sel, node, k], ls, color=INK, linewidth=1.2, label=f"vicino {k}")
    axes[1].axhline(config["detector"]["lost_silence"], color=INK_2, linestyle="-", linewidth=0.8)
    axes[1].text(lo + 1, config["detector"]["lost_silence"] + 0.4, "soglia del perso", fontsize=8, color=INK_2)
    axes[1].set_ylabel("età dei link (s)")
    axes[1].legend(frameon=False, fontsize=8, ncol=4, loc="upper left")
    labels = {"any": "any", "not_deaf": "not deaf", "not_deaf_episode": "not deaf (episodio)"}
    for row, rule in enumerate(alarms):
        for j, k in enumerate(others):
            y = row * (len(others) + 1) + j
            a = alarms[rule][sel][:, k]
            axes[2].fill_between(t[sel], y, y + 0.8 * a, step="post", color=INK, alpha=0.8, linewidth=0)
        axes[2].text(hi, row * (len(others) + 1) + 1.6, labels[rule], ha="right", fontsize=9, color=INK_2)
    axes[2].set_yticks([])
    axes[2].set_ylabel("allarme 'perso'\n(una riga per vicino)")
    axes[2].set_xlabel(f"tempo (s); ricevitore {node}, banda grigia = raffica di {length:.0f} s")
    axes[2].set_xlim(lo, hi)
    save_figure(fig, out / "burst_example.png", DPI)
    return best, node, length


def main() -> None:
    parser = make_parser(__doc__, cache_dir=True, out_dir=block_dir(6))
    args = parser.parse_args()
    out = Path(args.out_dir)
    res = load_pickle(out / "results.pkl")
    config = res["config"]
    res.setdefault("active_configs", config["final"]["configs"])  # results saved before null thresholds existed
    figure_tradeoff(res, out)
    if res["final_name"] is not None:
        figure_distance(res, out)
        figure_delay(res, out)
    figure_tuning_vs_test(res, out)
    if res["sensitivity"]:
        figure_sensitivity(res, out)
        best, node, length = figure_burst_example(res, config, Path(args.cache_dir), out)
        print(f"Figures saved to {out} (burst_example: point {best}, receiver {node}, burst of {length:.1f} s)")
    else:
        print(f"Figures saved to {out} (no sensitivity in the results: sensitivity and burst_example skipped)")


if __name__ == "__main__":
    main()

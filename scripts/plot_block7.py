#!/usr/bin/env python3
"""Figure del Blocco 7 per la tesi (``results/block7/``).

Legge ``results/block7/results.pkl`` (da ``run_block7_experiment.py``). Etichette in italiano, leggibili in bianco e nero (stili di linea e marcatori diversi oltre al colore).
Figure: ``routing_delivery``, ``leader_view_age``, ``separated_contact``, ``route_example``, ``routing_load``, ``predictive_gain``, ``leader_connectivity``, ``range``.
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

from src.channel import path_loss_db
from src.final import bootstrap_rng
from src.mobility import load_config
from src.metrics import nearest_rank
from src.refinement import load_pickle
from src.routing import STRATEGIES
from src.routing_experiment import POSITIONS, connectivity_by_rank, connectivity_summary, delivery_stats, distance_range, median_curve, per_run_values, pooled, select

INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e3e2dd"
COLORS = {"direct": "#52514e", "best_rssi": "#eb6834", "dqrem": "#2a78d6", "dqrem_no_slope": "#2a9d6b"}
STYLES = {"direct": ":", "best_rssi": "--", "dqrem": "-", "dqrem_no_slope": "-."}
MARKERS = {"direct": "^", "best_rssi": "s", "dqrem": "o", "dqrem_no_slope": "D"}
SCEN = ("none", "slowdown", "stop")
SCEN_LABEL = {"none": "nessun distacco", "slowdown": "rallenta", "stop": "si ferma"}
DPI = 200


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


def _bars(ax, values, labels, names, ylabel, errors=None):
    """Barre raggruppate: una per strategia dentro ogni gruppo `labels`; `values[nome][gruppo]`."""
    width = 0.8 / len(names)
    for k, name in enumerate(names):
        x = np.arange(len(labels)) + (k - (len(names) - 1) / 2) * width
        err = None if errors is None else np.array([errors[name][i] for i in range(len(labels))]).T
        ax.bar(x, [values[name][i] for i in range(len(labels))], width, color=COLORS[name], edgecolor=INK, linewidth=0.6, label=name, hatch=["", "//", "..", "xx"][k % 4], yerr=err, capsize=2, error_kw={"elinewidth": 0.8})
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel)
    style_axes(ax)


def figure_delivery(res, out):
    outs, names = res["outs"], list(res["config"]["routing"]["strategies"])
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for row, cls, scens in ((0, "group", SCEN), (1, "sep", ("slowdown", "stop"))):
        stats = {n: [delivery_stats(pooled(select(outs, s), n, cls)) for s in scens] for n in names}
        _bars(axes[row, 0], {n: [x[0] for x in stats[n]] for n in names}, [SCEN_LABEL[s] for s in scens], names, "frazione consegnata")
        axes[row, 0].set_ylim(0, 1.05)
        med = {n: [x[1] for x in stats[n]] for n in names}
        p95 = {n: [x[2] for x in stats[n]] for n in names}
        _bars(axes[row, 1], med, [SCEN_LABEL[s] for s in scens], names, "ritardo mediano (s)")
        for k, n in enumerate(names):  # p95 come trattino
            width = 0.8 / len(names)
            x = np.arange(len(scens)) + (k - (len(names) - 1) / 2) * width
            axes[row, 1].plot(x, p95[n], "_", color=INK, markersize=10, markeredgewidth=1.5)
        axes[row, 0].set_title("nodi nel gruppo" if cls == "group" else "nodo separato", fontsize=10, color=INK, loc="left")
        axes[row, 1].set_title("ritardo (barra = mediana, trattino = p95)", fontsize=10, color=INK, loc="left")
        axes[row, 1].set_yscale("symlog", linthresh=1.0)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out / "routing_delivery.png", dpi=DPI)
    plt.close(fig)


def figure_view_age(res, out):
    outs, names = res["outs"], list(res["config"]["routing"]["strategies"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    xs = np.arange(0, 61)
    for pos, ax in enumerate(axes):
        style_axes(ax)
        sub = select(outs, None, pos)
        for n in names:
            for cls, ls_override in (("group", None), ("sep", "-")):
                hist = sum(o["strategies"][n]["view"][cls]["hist"] for o in sub)
                total = hist.sum()
                if total == 0:
                    continue
                ccdf = 1.0 - np.concatenate([[0.0], np.cumsum(hist)]) / total  # frazione di tempo con età > x
                if cls == "group":
                    ax.plot(xs, np.maximum(ccdf, 1e-6), STYLES[n], color=COLORS[n], linewidth=1.8, label=n)
                else:
                    ax.plot(xs, np.maximum(ccdf, 1e-6), color=COLORS[n], linewidth=0.9, alpha=0.6)
        ax.set_yscale("log")
        ax.set_ylim(1e-4, 1.2)
        ax.axvline(5, color=INK, linestyle=":", linewidth=1.0)
        ax.axvline(10, color=INK, linestyle=":", linewidth=1.0)
        ax.set_xlabel("età della vista (s)")
        ax.set_title(f"capogruppo in {POSITIONS[pos]} ({len(sub)} corse)", fontsize=10, color=INK)
    axes[0].set_ylabel("frazione di tempo con età maggiore")
    axes[0].legend(frameon=False, fontsize=8, loc="lower left", title="linea spessa: nodi nel gruppo; sottile: separato", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "leader_view_age.png", dpi=DPI)
    plt.close(fig)


def figure_separated_contact(res, out):
    outs, names = res["outs"], list(res["config"]["routing"]["strategies"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, scen in zip(axes, ("slowdown", "stop")):
        style_axes(ax)
        sub = select(outs, scen)
        data = [np.array([o["strategies"][n]["sep"]["fresh_dist_group"] for o in sub]) for n in names]
        data = [d[np.isfinite(d)] for d in data]
        bp = ax.boxplot(data, tick_labels=[n.replace("_", "\n") for n in names], patch_artist=True, widths=0.55, medianprops={"color": INK, "linewidth": 1.5})
        for patch, n in zip(bp["boxes"], names):
            patch.set_facecolor(COLORS[n])
            patch.set_alpha(0.7)
        for k, d in enumerate(data):
            ax.plot(np.full(len(d), k + 1) + np.random.default_rng(k).uniform(-0.12, 0.12, len(d)), d, ".", color=INK, markersize=3, alpha=0.5)
        ax.set_title(SCEN_LABEL[scen], fontsize=10, color=INK)
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("distanza del separato dal gruppo all'ultimo record fresco (m)")
    fig.tight_layout()
    fig.savefig(out / "separated_contact.png", dpi=DPI)
    plt.close(fig)


def figure_route_example(res, out):
    outs = res["outs"]
    ref = next((o for o in outs if o["trace"] is not None), None)
    if ref is None:
        print("route_example: nessuna traccia salvata")
        return
    tr = ref["trace"]
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    t = tr["t"]
    ax = axes[0]
    style_axes(ax)
    ax.plot(t, tr["dist_group"], color=INK, linewidth=1.4, label="dal gruppo")
    ax.plot(t, tr["dist_leader"], color=COLORS["dqrem"], linewidth=1.2, linestyle="--", label="dal capogruppo")
    ax.axvline(tr["sep_start"], color=INK_2, linestyle=":")
    ax.set_ylabel("distanza (m)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(f"corsa {ref['scenario']} {ref['index']}, DQREM: separato = nodo {tr['sep_node']}, capogruppo = nodo {tr['leader']}", fontsize=10, color=INK, loc="left")
    ax = axes[1]
    style_axes(ax)
    ax.step(t, np.where(tr["announced"] < 255, tr["announced"], np.nan), where="post", color=INK_2, linewidth=1.2, label="salti annunciati dal separato")
    fresh = tr["rec_arrive"] - tr["rec_create"] < 5.0
    ax.scatter(tr["rec_create"][fresh], tr["rec_hops"][fresh], s=6, color=COLORS["dqrem"], label="record consegnati (fresco)")
    ax.scatter(tr["rec_create"][~fresh], tr["rec_hops"][~fresh], s=14, marker="x", color=COLORS["best_rssi"], label="record consegnati in ritardo (carry)")
    ax.set_ylabel("salti")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    dec = tr["decisions"]
    ax = axes[2]
    style_axes(ax)
    if len(dec):
        ax.plot(dec[:, 0], dec[:, 1], ".", color=INK, markersize=3)
    ax.set_ylabel("ponte scelto (nodo)")
    ax = axes[3]
    style_axes(ax)
    if len(dec):
        ax.plot(dec[:, 0], dec[:, 2], "-", color=COLORS["dqrem"], linewidth=0.9)
    ax.axhline(res["config"]["routing"]["candidate_threshold"], color=INK, linestyle=":", linewidth=1.0)
    for tb in tr.get("breaks", []):
        ax.axvline(tb, color=COLORS["best_rssi"], linewidth=0.6, alpha=0.6)
    ax.set_ylabel("segnale verso il ponte (dBm)")
    ax.set_xlabel("tempo (s); righe arancioni = rotture")
    ax.set_xlim(max(tr["sep_start"] - 60.0, 0.0), min(tr["sep_start"] + 600.0, t[-1]))
    fig.tight_layout()
    fig.savefig(out / "route_example.png", dpi=DPI)
    plt.close(fig)


def figure_load(res, out):
    outs = res["outs"]
    names = ("direct", "best_rssi", "dqrem")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, key, label, scale in ((axes[0], "airtime_per_s", "tempo d'aria per nodo (% del tempo)", 100.0), (axes[1], "packets_per_s", "tentativi di trasmissione al secondo per nodo", 1.0)):
        style_axes(ax)
        for n in names:
            hops = np.concatenate([o["strategies"][n]["hops_median"] for o in outs])
            vals = np.concatenate([o["strategies"][n][key] for o in outs]) * scale
            ok = np.isfinite(hops)
            hops, vals = np.round(hops[ok]).astype(int), vals[ok]
            levels = np.unique(hops)
            ax.plot(levels, [vals[hops == h].mean() for h in levels], STYLES[n], marker=MARKERS[n], color=COLORS[n], linewidth=1.6, label=n)
        ax.set_xlabel("distanza mediana dal capogruppo (salti annunciati)")
        ax.set_ylabel(label)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "routing_load.png", dpi=DPI)
    plt.close(fig)


def _boot_mean(values, config):
    rng = bootstrap_rng(config)
    idx = rng.integers(0, len(values), size=(config["final"]["bootstrap_samples"], len(values)))
    m = values[idx].mean(axis=1)
    return float(values.mean()), tuple(float(x) for x in np.quantile(m, [0.025, 0.975]))


def figure_predictive_gain(res, out):
    outs, config = res["outs"], res["config"]
    names = ("best_rssi", "dqrem_no_slope", "dqrem")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    groups = [("tutte", outs), *[(SCEN_LABEL[s], select(outs, s)) for s in ("slowdown", "stop")]]
    for ax, key, label in ((axes[0], "breaks", "rotture per corsa"), (axes[1], "age5", "frazione di tempo con vista > 5 s (nodi nel gruppo)")):
        vals, errs = {}, {}
        for n in names:
            vals[n], errs[n] = [], []
            for _, sub in groups:
                m, (lo, hi) = _boot_mean(per_run_values(sub, n)[key], config)
                vals[n].append(m)
                errs[n].append((m - lo, hi - m))
        _bars(ax, vals, [g for g, _ in groups], names, label, errs)
    axes[0].legend(frameon=False, fontsize=8)
    ax = axes[2]
    style_axes(ax)
    verdict = res["verdict"]
    rows = []
    for key in ("all", "slowdown", "stop"):
        for ref, short in (("vs_best_rssi", "contro best_rssi"), ("vs_no_slope", "contro dqrem_no_slope")):
            v = verdict[key][ref]
            red = v["break_reduction"]
            rows.append((f"{key}: {short}", 100 * red[0], 100 * red[1][0], 100 * red[1][1]))
    y = np.arange(len(rows))[::-1]
    for yy, (lab, m, lo, hi) in zip(y, rows):
        ax.errorbar(m, yy, xerr=[[m - lo], [hi - m]], fmt="o", color=COLORS["dqrem"], capsize=3, markeredgecolor=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.axvline(100 * config["routing"]["min_break_reduction"], color=INK, linestyle=":", linewidth=1.0)
    ax.set_xlabel("riduzione rotture di dqrem (%), IC bootstrap")
    fig.tight_layout()
    fig.savefig(out / "predictive_gain.png", dpi=DPI)
    plt.close(fig)


def figure_connectivity(res, out):
    outs, config = res["outs"], res["config"]
    n_nodes = config["group"]["n_nodes"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    style_axes(ax)
    for scen, ls, mk in zip(SCEN, ("-", "--", ":"), ("o", "s", "^")):
        frac, _ = connectivity_by_rank(select(outs, scen), n_nodes)
        ax.plot(np.arange(n_nodes), frac, ls, marker=mk, color=INK, linewidth=1.4, label=SCEN_LABEL[scen])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("posizione nel gruppo (0 = coda, %d = testa)" % (n_nodes - 1))
    ax.set_ylabel("frazione di tempo con una strada verso il capogruppo")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    style_axes(ax)
    labels = [SCEN_LABEL[s] for s in SCEN]
    split = [connectivity_summary(select(outs, s), n_nodes)["split"] for s in SCEN]
    ax.bar(np.arange(3), split, 0.5, color=COLORS["dqrem"], edgecolor=INK, linewidth=0.6)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("frazione di tempo con il gruppo spezzato")
    fig.tight_layout()
    fig.savefig(out / "leader_connectivity.png", dpi=DPI)
    plt.close(fig)


def figure_range(res, out):
    outs, config = res["outs"], res["config"]
    dist, med, n = median_curve(outs)
    ok = (n >= 200) & np.isfinite(med)
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    style_axes(ax)
    ax.plot(dist[ok], med[ok], color=COLORS["dqrem"], linewidth=1.8, label="con i corpi (mediana)")
    d = np.linspace(1.0, dist[-1], 400)
    ch = config["channel"]
    level = ch["tx_power"] + ch["antenna_gain_tx"] + ch["antenna_gain_rx"] - path_loss_db(d, ch)
    ax.plot(d, level, "--", color=INK_2, linewidth=1.2, label="solo distanza")
    for lv, ls in ((config["routing"]["candidate_threshold"], ":"), (config["packets"]["reception"]["sensitivity_dbm"], "-.")):
        ax.axhline(lv, color=INK, linestyle=ls, linewidth=1.0)
        ax.text(dist[-1], lv + 0.8, f"{lv:g} dBm: {distance_range(lv, config):.0f} m (solo distanza)", ha="right", fontsize=8, color=INK_2)
    ax.set_xlabel("distanza (m)")
    ax.set_ylabel("RSSI (dBm, senza shadowing e fading)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "range.png", dpi=DPI)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=None, help="default: results/block7/<routing.channel_profile>")
    args = parser.parse_args()
    out = Path(args.out_dir) if args.out_dir else _REPO_ROOT / "results" / "block7" / load_config(_REPO_ROOT / "config" / "default.yaml")["routing"]["channel_profile"]
    res = load_pickle(out / "results.pkl")
    figure_delivery(res, out)
    figure_view_age(res, out)
    figure_separated_contact(res, out)
    figure_route_example(res, out)
    figure_load(res, out)
    figure_predictive_gain(res, out)
    if "connectivity" in res["outs"][0]:
        figure_connectivity(res, out)
        figure_range(res, out)
    print(f"Figure salvate in {out}")


if __name__ == "__main__":
    main()

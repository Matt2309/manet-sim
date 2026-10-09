#!/usr/bin/env python3
"""Script di ispezione visiva del Blocco 4 (rilevatore).

Legge ``results/detection/`` (da ``run_detection_experiment.py``) e rifà una
corsa ``slowdown`` (Blocchi 1-3 dalla cache) per le figure di dettaglio.
Salva le figure in ``results/`` e stampa R, NIS media, tabelle al punto di
lavoro e tempi. Config: config/default.yaml se non indicata.
"""

import argparse
import csv
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
from src.experiment import load_results, print_summary
from src.kalman import predict_links
from src.metrics import best_over_sigma
from src.mobility import load_config
from src.simulator import experiment_runs, load_or_simulate

# Palette categorica (ordine fisso): blu, arancio, acqua; testo e griglia neutri.
C_LEVEL, C_SLOPE, C_BOTH = "#2a78d6", "#eb6834", "#1baf7a"
MODE_COLORS = {"level": C_LEVEL, "slope": C_SLOPE, "both": C_BOTH}
MODE_LABELS = {"level": "livello", "slope": "pendenza", "both": "livello + pendenza"}
SIGMA_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e3e2dd"

# Finestra dei grafici di dettaglio attorno al distacco, in secondi.
BEFORE_S = 60.0
AFTER_S = 180.0
# Un'ascissa log non rappresenta zero: i falsi allarmi nulli si disegnano a questo valore (solo grafica).
ZERO_FA_PLOT = 0.03


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


def read_runs(path: Path) -> list:
    with open(path / "runs.csv", newline="") as f:
        return list(csv.DictReader(f))


def detail_key(sigma_a, scenario, scope, mode, variant, field) -> str:
    return f"{sigma_a}|{scenario}|{scope}|{mode}|{variant}|{field}"


def rows_for(rows, **match):
    return [r for r in rows if all(r[k] == v for k, v in match.items())]


# ---------------------------------------------------------------------------
# Figure 1-2: corsa di dettaglio
# ---------------------------------------------------------------------------


def detail_run(config, cache_dir, details):
    """Rifà la prima corsa ``slowdown``: Blocco 4 con ``sigma_a`` di default."""
    info = next(r for r in experiment_runs(config) if r["scenario"] == "slowdown")
    run = load_or_simulate(info["config"], cache_dir)
    sigma_a = config["kalman"]["sigma_a"]
    inputs, kres = build_inputs(run, config, sigma_a, float(details["R"]))
    return info, run, inputs, kres


def plot_kalman_link(info, run, kres, config, path):
    t = run["t"]
    node, start = info["node"], run["sep_start"]
    others = [k for k in range(kres.times.shape[0]) if k != node]
    links = [
        (f"Link nel gruppo: {others[0]} → {others[1]}", others[0], others[1]),
        (f"Link dal nodo separato: {node} → {others[0]}", node, others[0]),
    ]
    est = predict_links(kres, t)
    lo, hi = start - BEFORE_S, start + AFTER_S
    sel = (t >= lo) & (t <= hi)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    for col, (title, i, j) in enumerate(links):
        ax, ax2 = axes[0, col], axes[1, col]
        beacons = kres.received[i, :, j] & (kres.times[i] >= lo) & (kres.times[i] <= hi)
        ax.plot(kres.times[i, beacons], run["rssi"][i, beacons, j], ".", ms=4, color=INK_2, label="beacon")
        # la stima si traccia solo finché l'ultimo aggiornamento non è più vecchio di fusion.max_age
        fresh = est["age"][sel, i, j] <= config["fusion"]["max_age"]
        r = np.where(fresh, est["r"][sel, i, j], np.nan)
        sd = np.where(fresh, np.sqrt(est["p00"][sel, i, j]), np.nan)
        ax.fill_between(t[sel], r - 2 * sd, r + 2 * sd, color=C_LEVEL, alpha=0.2, linewidth=0, label="±2σ")
        ax.plot(t[sel], r, color=C_LEVEL, linewidth=1.4, label="stima Kalman")
        ax2.plot(t[sel], np.where(fresh, est["s"][sel, i, j], np.nan), color=C_SLOPE, linewidth=1.4)
        ax2.axhline(0.0, color=INK_2, linewidth=0.6)
        for a in (ax, ax2):
            a.axvline(start, color=INK, linestyle=":", linewidth=1.0)
            style_axes(a)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel("RSSI (dBm)")
        ax2.set_ylabel("pendenza (dB/s)")
        ax2.set_xlabel("tempo (s)")
        ax.text(start, ax.get_ylim()[1], " distacco", va="top", ha="left", fontsize=9, color=INK)
        if col == 0:
            ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle(f"Kalman per link (σ_a = {kres.sigma_a:g} dB/s², R = {kres.measurement_noise:.2f} dB²)", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def shade_intervals(ax, t, state, color, label=None):
    edges = np.diff(np.concatenate([[0], state.astype(int), [0]]))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        ax.axvspan(t[a], t[min(b, len(t) - 1)], color=color, alpha=0.12, linewidth=0, label=label)
        label = None


def plot_fused_index(info, run, inputs, config, path):
    t = run["t"]
    node, start = info["node"], run["sep_start"]
    others = [k for k in range(inputs.valid.shape[0]) if k != node]
    d = config["detector"]
    lo, hi = start - BEFORE_S, start + AFTER_S
    sel = (t >= lo) & (t <= hi)
    fused_obs = others[0]
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    for ax, field, thr, label in (
        (axes[0], inputs.level, d["level_threshold"], "livello L (dBm)"),
        (axes[1], inputs.slope, d["slope_threshold"], "pendenza S (dB/s)"),
    ):
        for n, m in enumerate(others):
            ax.plot(t[sel], field["pairwise"][sel, m, node], color=SIGMA_COLORS[n], linewidth=0.8,
                    label=f"pairwise, osservatore {m}")
        ax.plot(t[sel], field["fused"][sel, fused_obs, node], color=INK, linewidth=2.2,
                label=f"fuso, osservatore {fused_obs}")
        ax.axhline(thr, color=INK_2, linestyle="--", linewidth=1.0, label="soglia")
        ax.axvline(start, color=INK, linestyle=":", linewidth=1.0)
        ax.set_ylabel(label)
        style_axes(ax)
    # intervalli di allarme dell'osservatore mostrato (pre-allarme e "perso", ambito fuso)
    pre = pre_alarm(inputs.level["fused"], inputs.slope["fused"], inputs.rx_ok, t, d, valid=inputs.valid)
    lost = lost_alarm(inputs.silence, inputs.rx_ok, t, d, inputs.valid)
    for ax in axes:
        shade_intervals(ax, t, pre[:, fused_obs, node], C_SLOPE, "pre-allarme" if ax is axes[0] else None)
        shade_intervals(ax, t, lost[:, fused_obs, node], C_LEVEL, "nodo perso" if ax is axes[0] else None)
        ax.set_xlim(lo, hi)
    axes[0].legend(frameon=False, fontsize=8, ncol=3, loc="lower left")
    axes[1].set_xlabel("tempo (s)")
    axes[1].set_ylim(-4.0, 3.0)
    axes[0].set_title(f"Bersaglio = nodo separato ({node}); allarmi dell'osservatore {fused_obs} "
                      f"(modalità {d['mode']}, ambito fuso)", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3-6: risultati salvati
# ---------------------------------------------------------------------------


def curve(rows, sigma_a, scope, mode, scenario):
    """Curva (falsi allarmi/ora, P_d) dei soli punti ammissibili, ordinata per soglia."""
    sel = sorted(
        rows_for(rows, sigma_a=sigma_a, scope=scope, mode=mode, variant="both", scenario=scenario, feasible=1),
        key=lambda r: r["threshold"],
    )
    fa = np.array([r["fa_per_hour"] for r in sel])
    return np.where(fa > 0, fa, ZERO_FA_PLOT), np.array([r["p_detect"] for r in sel])


def tradeoff_axes(axes, config):
    for ax, scenario in zip(axes, ("slowdown", "stop")):
        ax.set_xscale("log")
        target = config["metrics"]["target_false_alarms_per_hour"]
        ax.axvline(target, color=INK, linestyle=":", linewidth=1.0)
        ax.set_xlim(ZERO_FA_PLOT * 0.8, target * 1.5)
        ax.set_xlabel(f"falsi allarmi per ora (episodi; zero disegnato a {ZERO_FA_PLOT:g})")
        ax.set_title(scenario, fontsize=11)
        ax.set_ylim(-0.03, 1.03)
        style_axes(ax)
    axes[0].set_ylabel(f"probabilità di rilevamento entro {config['metrics']['success_distance']:g} m")


def plot_tradeoff(rows, config, sigmas, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    tradeoff_axes(axes, config)
    for mode in ("level", "slope", "both"):
        for scope, style in (("fused", "-"), ("pairwise", "--")):
            best = best_over_sigma(rows, scope, mode, "both")
            sigma_a = best["sigma_a"] if best else config["kalman"]["sigma_a"]
            for ax, scenario in zip(axes, ("slowdown", "stop")):
                x, y = curve(rows, sigma_a, scope, mode, scenario)
                if len(x) == 0:
                    continue  # nessun punto ammissibile
                ax.plot(x, y, style, color=MODE_COLORS[mode], linewidth=1.5, marker="o", ms=3,
                        label=f"{MODE_LABELS[mode]}, {'fuso' if scope == 'fused' else 'pairwise'} (σ_a={sigma_a:g})")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle("Compromesso rilevamento / falsi allarmi (pre-allarme + perso, miglior σ_a, solo punti ammissibili)",
                 color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_sigma_sweep(rows, config, sigmas, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    tradeoff_axes(axes, config)
    for n, sigma_a in enumerate(sigmas):
        for ax, scenario in zip(axes, ("slowdown", "stop")):
            x, y = curve(rows, sigma_a, "fused", "both", scenario)
            if len(x) == 0:
                continue  # nessun punto ammissibile
            ax.plot(x, y, color=SIGMA_COLORS[n], linewidth=1.5, marker="o", ms=3, label=f"σ_a = {sigma_a:g} dB/s²")
    axes[0].legend(frameon=False, fontsize=9, loc="lower right")
    fig.suptitle("Effetto di σ_a (modalità livello + pendenza, ambito fuso, solo punti ammissibili)", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_delay(rows, details, config, path):
    mcfg = config["metrics"]
    dist_limit = mcfg["success_distance"]
    best = best_over_sigma(rows, "fused", "both", "both")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    if best is None:
        fig.text(0.5, 0.5, "nessun punto ammissibile", ha="center")
    else:
        sigma_a = best["sigma_a"]
        grid = details[detail_key(sigma_a, "slowdown", "fused", "both", "both", "threshold")]
        n = int(np.argmin(np.abs(grid - best["threshold"])))
        colors = {"slowdown": C_LEVEL, "stop": C_SLOPE}
        data = {
            (scenario, field): details[detail_key(sigma_a, scenario, "fused", "both", "both", field)][:, n]
            for scenario in ("slowdown", "stop")
            for field in ("delay", "distance", "success")
        }
        for ax, field, unit in (
            (axes[0], "delay", "ritardo di rilevamento (s)"),
            (axes[1], "distance", "distanza minima dal gruppo al rilevamento (m)"),
        ):
            finite_all = np.concatenate([v[np.isfinite(v)] for (sc, f), v in data.items() if f == field])
            top = max(float(finite_all.max()) if len(finite_all) else 1.0, dist_limit * 1.5)
            bins = np.linspace(0.0, top, 26)
            for scenario in ("slowdown", "stop"):
                values = data[(scenario, field)]
                ok = data[(scenario, "success")].astype(bool)
                ax.hist(values[np.isfinite(values)], bins=bins, color=colors[scenario], alpha=0.55, linewidth=0,
                        label=f"{scenario} (rilevati entro {dist_limit:g} m: {int(ok.sum())}/{len(ok)})")
            ax.set_xlabel(unit)
            ax.set_ylabel("corse")
            if field == "distance":
                ax.axvline(dist_limit, color=INK, linestyle=":", linewidth=1.0)
            ax.legend(frameon=False, fontsize=9)
            style_axes(ax)
        fig.suptitle(f"Punto di lavoro: modalità livello + pendenza, ambito fuso, σ_a = {sigma_a:g}, "
                     f"soglia {best['threshold']:g} dBm", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_nis(details, config, sigmas, path):
    edges = details["nis_edges"]
    centers = 0.5 * (edges[1:] + edges[:-1])
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.linspace(edges[0] + 1e-3, edges[-1], 600)
    ax.plot(x, chi2.pdf(x, 1), color=INK, linewidth=2.0, label="chi quadro, 1 g.l.")
    for n, sigma_a in enumerate(sigmas):
        hist = details[f"nis_hist|{sigma_a}"]
        total = float(details[f"nis_count|{sigma_a}"])
        density = hist / (total * np.diff(edges))
        mean = float(details[f"nis_mean|{sigma_a}"])
        ax.step(centers, density, where="mid", color=SIGMA_COLORS[n], linewidth=1.2,
                label=f"σ_a = {sigma_a:g} (NIS media {mean:.2f})")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 5.0)
    ax.set_xlabel("NIS")
    ax.set_ylabel("densità")
    ax.set_title("NIS nel gruppo senza distacco", color=INK)
    ax.legend(frameon=False, fontsize=9)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default=str(_REPO_ROOT / "config" / "default.yaml"))
    parser.add_argument("--cache-dir", default=str(_REPO_ROOT / "results" / "cache"))
    parser.add_argument("--in-dir", default=str(_REPO_ROOT / "results" / "detection"))
    parser.add_argument("--out-dir", default=str(_REPO_ROOT / "results"))
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, details = load_results(Path(args.in_dir))
    sigmas = [float(s) for s in details["sigma_values"]]

    print_summary(rows, details, config)
    print(
        f"\nTempi: Blocco 4 su una corsa (un sigma_a) {float(details['time_block4_one_run']):.2f} s;"
        f" sweep {float(details['time_sweep']):.1f} s; Blocchi 1-3 (con cache) {float(details['time_blocks123_cache']):.1f} s;"
        f" esperimento intero {float(details['time_experiment']):.1f} s"
    )

    info, run, inputs, kres = detail_run(config, Path(args.cache_dir), details)
    print(f"\nCorsa di dettaglio: slowdown, nodo {info['node']}, distacco a {run['sep_start']:.1f} s")
    plot_kalman_link(info, run, kres, config, out_dir / "kalman_link.png")
    plot_fused_index(info, run, inputs, config, out_dir / "fused_index.png")
    plot_tradeoff(rows, config, sigmas, out_dir / "detection_tradeoff.png")
    plot_sigma_sweep(rows, config, sigmas, out_dir / "sigma_a_sweep.png")
    plot_delay(rows, details, config, out_dir / "detection_delay.png")
    plot_nis(details, config, sigmas, out_dir / "nis.png")
    print(f"Figure salvate in {out_dir}")


if __name__ == "__main__":
    main()

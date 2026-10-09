#!/usr/bin/env python3
"""Script di ispezione visiva del Blocco 1 (mobilità).

Genera le figure diagnostiche in results/ a partire da un file di
configurazione YAML (config/default.yaml se non specificato altrimenti).
"""

import argparse
import itertools
import subprocess
import sys
import types
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.mobility import (
    Track,
    lateral_offset_speed,
    load_config,
    load_track,
    relative_speed,
    simulate_mobility,
)

# Commit con il modulo di mobilità originale (OU del primo ordine e vincolo
# di distanza minima a proiezione): serve solo al confronto prima/dopo.
LEGACY_MOBILITY_COMMIT = "921bfb8"


def _suffix(cfg: dict) -> str:
    return "_sep" if cfg["separation"]["enabled"] else ""


def _analysis_window(result, cfg: dict) -> tuple[int, int]:
    """Intervallo di indici su cui calcolare le statistiche: prima della separazione se attiva (il nodo separato falserebbe la formazione), altrimenti tutta la simulazione."""
    if cfg["separation"]["enabled"]:
        idx_end = int(np.searchsorted(result.t, cfg["separation"]["start_time"]))
        return 0, max(idx_end, 1)
    return 0, len(result.t)


def plot_path_overview(track: Track, result, out_dir: Path, suffix: str) -> None:
    """Percorso completo in coordinate locali, con il baricentro del gruppo in alcuni istanti."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(track.x, track.y, "-", color="0.7", linewidth=1, label="percorso GPX")

    sample_idx = np.linspace(0, len(result.t) - 1, 6).astype(int)
    centroid_sample = track.query(result.s_centroid[sample_idx])
    ax.scatter(
        centroid_sample.position[:, 0],
        centroid_sample.position[:, 1],
        color="crimson",
        zorder=5,
        label="baricentro (istanti campione)",
    )
    for i, (px, py) in zip(sample_idx, centroid_sample.position):
        ax.annotate(f"t={result.t[i]:.0f}s", (px, py), fontsize=7, xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Percorso completo e posizione del gruppo")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_dir / f"path_overview{suffix}.png", dpi=150)
    plt.close(fig)


def plot_curve_zoom(track: Track, result, out_dir: Path, suffix: str, half_window_m: float = 8.0) -> None:
    """Zoom sull'istante di massimo rapporto fra estensione in ascissa curvilinea e in linea d'aria (gruppo "a cavallo" di un tornante).

    I nodi sono uniti in ordine di ascissa curvilinea. Solo senza separazione attiva.
    """
    n_nodes = result.positions.shape[1]
    s_nodes = result.s_nodes  # (T, N)

    extent_curv = np.max(s_nodes, axis=1) - np.min(s_nodes, axis=1)
    positions = result.positions  # (T, N, 2)
    pair_i, pair_j = np.array(list(itertools.combinations(range(n_nodes), 2))).T
    pairwise = np.linalg.norm(positions[:, pair_i, :] - positions[:, pair_j, :], axis=-1)
    extent_air = np.max(pairwise, axis=1)

    ratio = extent_curv / np.maximum(extent_air, 1e-6)
    t_idx = int(np.argmax(ratio))

    node_positions = positions[t_idx]
    order = np.argsort(s_nodes[t_idx])
    ordered = node_positions[order]
    center = node_positions.mean(axis=0)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(track.x, track.y, "-", color="0.7", linewidth=1.5, label="percorso GPX")
    ax.plot(ordered[:, 0], ordered[:, 1], "-", color="tab:blue", linewidth=1.2, zorder=4,
            label="nodi, in ordine di ascissa curvilinea")
    ax.scatter(node_positions[:, 0], node_positions[:, 1], color="tab:blue", zorder=5)
    for i, (px, py) in enumerate(node_positions):
        ax.annotate(str(i), (px, py), fontsize=9, xytext=(4, 4), textcoords="offset points")

    ax.set_xlim(center[0] - half_window_m, center[0] + half_window_m)
    ax.set_ylim(center[1] - half_window_m, center[1] + half_window_m)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(
        f"Zoom su un tornante (t={result.t[t_idx]:.0f}s)\n"
        f"estensione lungo il percorso: {extent_curv[t_idx]:.2f} m — "
        f"estensione in linea d'aria: {extent_air[t_idx]:.2f} m"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"curve_zoom{suffix}.png", dpi=150)
    plt.close(fig)


def plot_pair_distances_window(result, out_dir: Path, suffix: str, cfg: dict, window_s: float = 120.0) -> None:
    """Distanze fra tutte le coppie su `window_s` s a metà della finestra di analisi: le curve devono essere lisce (un ciclo OU = 80 campioni a correlation_time = 8 s)."""
    n_nodes = cfg["group"]["n_nodes"]
    dt = cfg["simulation"]["dt"]
    idx_a, idx_b = _analysis_window(result, cfg)

    mid = (idx_a + idx_b) // 2
    half_steps = int(round(window_s / dt / 2.0))
    lo = max(idx_a, mid - half_steps)
    hi = min(idx_b, mid + half_steps)

    fig, ax = plt.subplots(figsize=(9, 5))
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            ax.plot(result.t[lo:hi], result.distances[lo:hi, i, j], linewidth=0.9, label=f"{i}-{j}")

    ax.set_xlabel("t [s]")
    ax.set_ylabel("distanza [m]")
    ax.set_title(f"Distanze fra tutte le coppie — finestra di {window_s:.0f} s ({hi - lo} campioni)")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(out_dir / f"pair_distances_window{suffix}.png", dpi=150)
    plt.close(fig)


def plot_group_extent(result, out_dir: Path, suffix: str, cfg: dict) -> None:
    """Estensione testa-coda in ascissa curvilinea nel tempo, con longitudinal_spread e mediana osservata; solo finestra di analisi."""
    idx_a, idx_b = _analysis_window(result, cfg)
    extent = np.max(result.s_nodes[idx_a:idx_b], axis=1) - np.min(result.s_nodes[idx_a:idx_b], axis=1)
    median_extent = float(np.median(extent))
    target = cfg["group"]["longitudinal_spread"]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(result.t[idx_a:idx_b], extent, color="tab:blue", linewidth=0.8)
    ax.axhline(target, color="crimson", linestyle="--", label=f"longitudinal_spread = {target:.1f} m")
    ax.axhline(median_extent, color="0.3", linestyle=":", label=f"mediana osservata = {median_extent:.2f} m")
    ax.set_xlabel("t [s]")
    ax.set_ylabel("estensione testa-coda [m]")
    ax.set_title(
        f"Estensione del gruppo in ascissa curvilinea — "
        f"finestra [{result.t[idx_a]:.0f}, {result.t[idx_b - 1]:.0f}] s ({idx_b - idx_a} campioni)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"group_extent{suffix}.png", dpi=150)
    plt.close(fig)


def plot_distance_histogram(result, out_dir: Path, suffix: str, cfg: dict) -> None:
    """Istogramma delle distanze euclidee fra tutte le coppie sulla finestra di analisi, con mediana, 95° percentile e min_node_gap."""
    idx_a, idx_b = _analysis_window(result, cfg)
    n_nodes = cfg["group"]["n_nodes"]
    iu, ju = np.triu_indices(n_nodes, k=1)
    pair_dist = result.distances[idx_a:idx_b][:, iu, ju].reshape(-1)

    median_d = float(np.median(pair_dist))
    p95_d = float(np.percentile(pair_dist, 95))
    min_gap = cfg["group"]["min_node_gap"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pair_dist, bins=80, color="tab:blue", alpha=0.75)
    ax.axvline(median_d, color="0.2", linestyle=":", label=f"mediana = {median_d:.2f} m")
    ax.axvline(p95_d, color="tab:orange", linestyle="--", label=f"p95 = {p95_d:.2f} m")
    ax.axvline(min_gap, color="crimson", linestyle="-", label=f"min_node_gap = {min_gap:.2f} m")
    ax.set_xlabel("distanza fra coppia di nodi [m]")
    ax.set_ylabel("conteggio")
    ax.set_title(f"Distribuzione delle distanze fra coppie ({len(pair_dist)} campioni)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"distance_histogram{suffix}.png", dpi=150)
    plt.close(fig)


def plot_separation_detail(result, out_dir: Path, suffix: str, cfg: dict, window_s: float = 300.0) -> None:
    """Distanza fra il nodo separato e gli altri quattro nei primi `window_s` s dopo la separazione (finestra in cui il rilevamento è in gioco), asse y logaritmico."""
    sep_cfg = cfg["separation"]
    node_id = sep_cfg["node_id"]
    start_time = sep_cfg["start_time"]
    dt = cfg["simulation"]["dt"]
    n_nodes = cfg["group"]["n_nodes"]
    other_nodes = [i for i in range(n_nodes) if i != node_id]

    idx_start = int(np.searchsorted(result.t, start_time))
    idx_end = min(idx_start + int(round(window_s / dt)), len(result.t))

    fig, ax = plt.subplots(figsize=(9, 5))
    for j in other_nodes:
        ax.plot(
            result.t[idx_start:idx_end] - start_time,
            result.distances[idx_start:idx_end, node_id, j],
            linewidth=1.0,
            label=f"nodo {node_id} — nodo {j}",
        )
    ax.set_yscale("log")
    ax.set_xlabel("tempo dall'inizio della separazione [s]")
    ax.set_ylabel("distanza [m] (scala log)")
    ax.set_title(f"Dettaglio della separazione — primi {window_s:.0f} s")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"separation_detail{suffix}.png", dpi=150)
    plt.close(fig)


def print_group_stats(track: Track, result, cfg: dict) -> None:
    """Stampa le statistiche del gruppo nella finestra di analisi: estensione longitudinale e laterale, distanze fra coppie, velocità 2D (rispetto al baricentro e laterale)."""
    idx_a, idx_b = _analysis_window(result, cfg)
    n_nodes = cfg["group"]["n_nodes"]
    sl = slice(idx_a, idx_b)

    ext_long = np.ptp(result.s_nodes[sl], axis=1)
    ext_lat = np.ptp(result.lateral_offsets[sl], axis=1)
    iu, ju = np.triu_indices(n_nodes, k=1)
    pair = result.distances[sl][:, iu, ju].reshape(-1)

    dt = result.metadata["dt"]
    centroid = result.positions[sl].mean(axis=1, keepdims=True)
    v_full = np.linalg.norm(np.diff(result.positions[sl] - centroid, axis=0), axis=-1) / dt
    v_lat = lateral_offset_speed(result, track)[idx_a : max(idx_b - 1, idx_a + 1)]
    v_road = relative_speed(result)[idx_a : max(idx_b - 1, idx_a + 1)]

    print(f"Statistiche del gruppo, finestra [{result.t[idx_a]:.0f}, {result.t[idx_b - 1]:.0f}] s ({idx_b - idx_a} campioni):")
    print(f"  sigma_long = {result.metadata['sigma_long']:.4f} m, sigma_lat = {result.metadata['sigma_lat']:.4f} m")
    print(f"  estensione longitudinale: mediana {np.median(ext_long):.2f} m, p95 {np.percentile(ext_long, 95):.2f} m")
    print(f"  estensione laterale:      mediana {np.median(ext_lat):.2f} m, p95 {np.percentile(ext_lat, 95):.2f} m")
    print(
        f"  distanza fra coppie: mediana {np.median(pair):.2f} m, p95 {np.percentile(pair, 95):.2f} m, "
        f"p1 {np.percentile(pair, 1):.2f} m; campioni sotto 1 m: {int(np.sum(pair < 1.0))} su {len(pair)} "
        f"({100 * np.mean(pair < 1.0):.4f} %)"
    )
    for label, v in (
        ("2D rispetto al baricentro", v_full),
        ("2D dello scostamento laterale", v_lat),
        ("stradale (s, l)", v_road),
    ):
        print(f"  velocità {label}: p99 {np.percentile(v, 99):.2f} m/s, max {v.max():.2f} m/s")


def _git_show(path: str) -> str:
    return subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "show", f"{LEGACY_MOBILITY_COMMIT}:{path}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def legacy_relative_speed(cfg: dict) -> np.ndarray:
    """Velocità relativa al baricentro col modulo originale (letto dal commit `LEGACY_MOBILITY_COMMIT`), stessa definizione di `relative_speed`. Solleva se git non è disponibile."""
    import yaml

    module = types.ModuleType("mobility_legacy")
    sys.modules[module.__name__] = module
    exec(compile(_git_show("src/mobility.py"), "mobility_legacy", "exec"), module.__dict__)

    legacy_cfg = yaml.safe_load(_git_show("config/default.yaml"))
    legacy_cfg["track"]["gpx_file"] = cfg["track"]["gpx_file"]
    legacy_cfg["simulation"].update(cfg["simulation"])
    legacy_cfg["separation"] = dict(cfg["separation"])
    result = module.simulate_mobility(legacy_cfg)

    track = module.load_track(legacy_cfg)
    sample = track.query(result.s_nodes)
    lateral = np.sum((result.positions - sample.position) * sample.normal, axis=-1)
    dt = legacy_cfg["simulation"]["dt"]
    rel_long = np.diff(result.s_nodes - result.s_centroid[:, None], axis=0) / dt
    rel_lat = np.diff(lateral, axis=0) / dt
    return np.hypot(rel_long, rel_lat)


def plot_relative_speed(result, out_dir: Path, suffix: str, cfg: dict) -> dict:
    """Distribuzione della velocità relativa al baricentro (passo dt, finestra di analisi) con `max_relative_speed_p99`; sovrappone il modulo originale se leggibile da git. Restituisce i 99° percentili."""
    idx_a, idx_b = _analysis_window(result, cfg)
    v_new = relative_speed(result)[idx_a : max(idx_b - 1, idx_a + 1)].ravel()
    limit = cfg["group"]["max_relative_speed_p99"]
    p99 = {"dopo": float(np.percentile(v_new, 99))}

    series = [("dopo (2° ordine + repulsione)", v_new, "tab:blue")]
    try:
        v_old_all = legacy_relative_speed(cfg)
        v_old = v_old_all[idx_a : max(idx_b - 1, idx_a + 1)].ravel()
        series.insert(0, ("prima (OU del 1° ordine + vincolo)", v_old, "tab:red"))
        p99["prima"] = float(np.percentile(v_old, 99))
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"confronto con il modulo originale saltato: {exc}")

    bins = np.logspace(-2, np.log10(max(v.max() for _, v, _ in series) * 1.05), 80)
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, v, color in series:
        ax.hist(v, bins=bins, density=True, alpha=0.5, color=color,
                label=f"{label}: p99 = {np.percentile(v, 99):.2f} m/s")
    ax.axvline(limit, color="k", linestyle="--", label=f"max_relative_speed_p99 = {limit:g} m/s")
    ax.set_xscale("log")
    ax.set_xlabel("velocità relativa al baricentro [m/s] (scala log)")
    ax.set_ylabel("densità")
    ax.set_title(f"Velocità dei nodi relativa al baricentro, a passo dt — {len(v_new)} campioni")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"relative_speed{suffix}.png", dpi=150)
    plt.close(fig)
    return p99


def plot_speed_profile(track: Track, out_dir: Path, suffix: str) -> None:
    """Profilo di velocità lungo il percorso, grezzo e lisciato sovrapposti."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(track.s, track.speed_raw, color="0.7", linewidth=0.7, label="velocità grezza")
    ax.plot(track.s, track.speed, color="tab:blue", linewidth=1.5, label="velocità lisciata")
    ax.set_xlabel("ascissa curvilinea s [m]")
    ax.set_ylabel("velocità [m/s]")
    ax.set_title("Profilo di velocità lungo il percorso")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"speed_profile{suffix}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera le figure di ispezione del modulo di mobilità")
    parser.add_argument(
        "config",
        nargs="?",
        default=str(_REPO_ROOT / "config" / "default.yaml"),
        help="percorso del file di configurazione YAML",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    track = load_track(cfg)
    result = simulate_mobility(cfg)

    out_dir = _REPO_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    suffix = _suffix(cfg)

    plot_path_overview(track, result, out_dir, suffix)
    if cfg["separation"]["enabled"]:
        print("separation.enabled=true: curve_zoom saltato (istante di tornante poco significativo).")
    else:
        plot_curve_zoom(track, result, out_dir, suffix)
    plot_pair_distances_window(result, out_dir, suffix, cfg)
    plot_group_extent(result, out_dir, suffix, cfg)
    plot_distance_histogram(result, out_dir, suffix, cfg)
    if cfg["separation"]["enabled"]:
        plot_separation_detail(result, out_dir, suffix, cfg)
    plot_speed_profile(track, out_dir, suffix)
    p99 = plot_relative_speed(result, out_dir, suffix, cfg)
    print_group_stats(track, result, cfg)
    for label, value in p99.items():
        print(f"p99 della velocità relativa al baricentro, {label}: {value:.2f} m/s")

    print(f"Figure salvate in {out_dir} (suffisso '{suffix}')")


if __name__ == "__main__":
    main()

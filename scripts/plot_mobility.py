#!/usr/bin/env python3
"""Script di ispezione visiva del Blocco 1 (mobilità).

Genera le figure diagnostiche in results/ a partire da un file di
configurazione YAML (config/default.yaml se non specificato altrimenti).
"""

import argparse
import itertools
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.mobility import Track, load_config, load_track, simulate_mobility


def _suffix(cfg: dict) -> str:
    return "_sep" if cfg["separation"]["enabled"] else ""


def _analysis_window(result, cfg: dict) -> tuple[int, int]:
    """Intervallo di indici temporali su cui calcolare le statistiche del
    gruppo: se la separazione è attiva, solo prima del suo inizio (dopo,
    il nodo separato non fa più parte della formazione e ne stravolgerebbe
    le statistiche, calibrate su n_nodes nodi intatti); altrimenti l'intera
    simulazione.
    """
    if cfg["separation"]["enabled"]:
        idx_end = int(np.searchsorted(result.t, cfg["separation"]["start_time"]))
        return 0, max(idx_end, 1)
    return 0, len(result.t)


def plot_path_overview(track: Track, result, out_dir: Path, suffix: str) -> None:
    """Percorso completo in coordinate locali, con la posizione del
    baricentro del gruppo marcata in alcuni istanti campione.
    """
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
    """Zoom sull'istante in cui il gruppo è più "a cavallo" di un tornante:
    quello di massimo rapporto fra estensione in ascissa curvilinea ed
    estensione in linea d'aria. I nodi sono uniti da una spezzata in
    ordine di ascissa curvilinea, per vedere a colpo d'occhio se la catena
    segue la strada o taglia in linea retta.

    Generato solo senza separazione attiva: con la separazione il nodo che
    si allontana smetterebbe rapidamente di essere "a cavallo" di nulla,
    e la ricerca del tornante diventerebbe poco significativa.
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
    """Distanze fra tutte le coppie di nodi su una finestra di window_s
    secondi presa a metà della finestra di analisi: qui le curve devono
    risultare lisce (un ciclo del processo OU, a correlation_time = 8 s,
    occupa 80 campioni). Se appaiono frastagliate campione per campione,
    l'OU non sta funzionando.
    """
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
    """Estensione testa-coda del gruppo in ascissa curvilinea nel tempo,
    con linee di riferimento a longitudinal_spread e alla mediana
    osservata. Limitato alla finestra di analisi (prima della separazione,
    se attiva): dopo, il nodo separato non fa più parte della formazione.
    """
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
    """Istogramma delle distanze euclidee fra tutte le coppie di nodi,
    sulla finestra di analisi, con mediana, 95° percentile e min_node_gap
    marcati: è il modo corretto di leggere una statistica su decine di
    migliaia di campioni, dove un grafico a serie temporale sovraccarico
    di punti per pixel sarebbe illeggibile.
    """
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
    """Distanza fra il nodo separato e gli altri quattro, limitata ai
    primi window_s secondi dopo l'inizio della separazione, asse y
    logaritmico: è l'unica finestra temporale in cui il rilevamento della
    separazione è realmente in gioco.
    """
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

    print(f"Figure salvate in {out_dir} (suffisso '{suffix}')")


if __name__ == "__main__":
    main()

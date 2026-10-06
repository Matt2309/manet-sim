#!/usr/bin/env python3
"""Script di ispezione visiva del Blocco 2 (canale).

Esegue tre simulazioni della corsa intera (senza separazione, con
separazione in modalità `field`, con separazione in modalità
`independent`), salva le figure diagnostiche in results/ e stampa a
console le statistiche principali. Configurazione: config/default.yaml se
non specificato altrimenti.
"""

import argparse
import copy
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.channel import (
    link_shadowing,
    link_variance_theory,
    path_loss_db,
    simulate_channel,
)
from src.mobility import load_config, simulate_mobility

# Finestra, dopo l'inizio della separazione, in cui il rilevamento deve
# avvenire: serve solo alle statistiche e ai grafici, non è un parametro del modello.
DETECTION_WINDOW_S = 60.0


def _group_window(mobility, cfg: dict) -> int:
    """Indice finale della finestra "gruppo intatto": l'inizio della separazione."""
    return max(int(np.searchsorted(mobility.t, cfg["separation"]["start_time"])), 1)


def _offdiag(arr: np.ndarray) -> np.ndarray:
    return arr[:, ~np.eye(arr.shape[1], dtype=bool)]


def plot_rssi_vs_distance(mob, res, cfg: dict, out_dir: Path) -> None:
    """RSSI misurato di tutte le coppie in funzione della distanza, con la
    retta dell'attenuazione, la fascia ±sigma dello shadowing (la deviazione
    standard teorica in funzione della lunghezza del link) e le linee di
    saturazione e di sensibilità. Decimato nel tempo per leggibilità.
    """
    ch = cfg["channel"]
    step = max(len(mob.t) // 3000, 1)
    d = _offdiag(mob.distances[::step]).ravel()
    r = _offdiag(res.rssi_measured[::step]).ravel()

    grid = np.logspace(np.log10(max(d.min(), 0.5)), np.log10(d.max()), 300)
    line = ch["tx_power"] + ch["antenna_gain_tx"] + ch["antenna_gain_rx"] - path_loss_db(grid, ch)
    sh = ch["shadowing"]
    band = np.sqrt(link_variance_theory(grid, sh["sigma"], sh["correlation_distance"]))

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(d, r, s=4, alpha=0.08, color="tab:blue", rasterized=True, label="RSSI misurato")
    ax.plot(grid, line, color="crimson", linewidth=1.5, label="attenuazione con la distanza")
    ax.fill_between(grid, line - band, line + band, color="crimson", alpha=0.15,
                    label="±σ shadowing in funzione della lunghezza del link")
    ax.axhline(ch["measurement"]["saturation_dbm"], color="0.3", linestyle="--", label="saturazione")
    ax.axhline(cfg["packets"]["reception"]["sensitivity_dbm"], color="tab:green", linestyle="--", label="sensibilità (riferimento)")
    ax.set_xscale("log")
    ax.set_xlabel("distanza [m] (scala log)")
    ax.set_ylabel("RSSI [dBm]")
    ax.set_title("RSSI misurato di tutte le coppie in funzione della distanza")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / "rssi_vs_distance.png", dpi=150)
    plt.close(fig)


def plot_link_components(mob, res, cfg: dict, out_dir: Path, window_s: float = 120.0) -> None:
    """Un link del gruppo su una finestra di 120 s, come colonna di pannelli
    con lo stesso asse dei tempi: RSSI della sola distanza (con gli scarti
    delle schede), un pannello per ciascun contributo (shadowing, torso
    proprio, altri corridori, variazioni rapide), ognuno con il proprio asse
    in dB centrato sul suo intervallo, e per ultimo l'RSSI misurato.
    """
    dt = cfg["simulation"]["dt"]
    i, j = 0, 1
    idx_end = _group_window(mob, cfg)
    mid = idx_end // 2
    half = int(round(window_s / dt / 2.0))
    lo, hi = max(mid - half, 0), min(mid + half, idx_end)
    t = mob.t[lo:hi]

    base = res.rssi_path_loss[lo:hi, i, j] + res.tx_offset[i] + res.rx_offset[j]
    panels = [
        ("solo distanza\n(+ scarti schede) [dBm]", base, "tab:blue"),
        ("shadowing [dB]", res.shadowing[lo:hi, i, j], "tab:purple"),
        ("torso proprio [dB]", -res.body_own[lo:hi, i, j], "tab:orange"),
        ("altri corridori [dB]", -res.body_others[lo:hi, i, j], "tab:brown"),
        ("variazioni rapide [dB]", res.fading[lo:hi, i, j], "tab:green"),
    ]
    min_span = 2.0  # dB; intervallo minimo, per i contributi quasi costanti

    fig, axes = plt.subplots(len(panels) + 1, 1, figsize=(10, 12), sharex=True)
    for ax, (label, y, color) in zip(axes, panels):
        ax.plot(t, y, color=color, linewidth=0.9)
        centre = 0.5 * (np.nanmax(y) + np.nanmin(y))
        span = max(np.nanmax(y) - np.nanmin(y), min_span)
        ax.set_ylim(centre - 0.55 * span, centre + 0.55 * span)
        ax.set_ylabel(label, fontsize=8)
        ax.grid(alpha=0.3)
    ax = axes[-1]
    ax.step(t, res.rssi_measured[lo:hi, i, j], where="post", color="k", linewidth=0.8)
    ax.set_ylabel("RSSI misurato [dBm]", fontsize=8)
    ax.set_xlabel("t [s]")
    ax.grid(alpha=0.3)
    fig.suptitle(f"Componenti del link {i}→{j} — finestra di {window_s:.0f} s")
    fig.tight_layout()
    fig.savefig(out_dir / "link_components.png", dpi=150)
    plt.close(fig)


def plot_separation_rssi(mob, res, cfg: dict, out_dir: Path, before_s: float = 60.0, after_s: float = 300.0) -> None:
    """I 4 link verso il nodo separato, da un minuto prima a cinque minuti
    dopo l'inizio della separazione: RSSI misurato e RSSI della sola
    attenuazione con la distanza (senza rumore), linea verticale
    all'inizio della separazione e, su un secondo asse, la distanza.
    """
    sep = cfg["separation"]
    node, t0, dt = sep["node_id"], sep["start_time"], cfg["simulation"]["dt"]
    others = [k for k in range(cfg["group"]["n_nodes"]) if k != node]
    lo = int(np.searchsorted(mob.t, t0 - before_s))
    hi = min(int(np.searchsorted(mob.t, t0 + after_s)), len(mob.t))
    t = mob.t[lo:hi] - t0

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for ax, k in zip(axes.ravel(), others):
        ax.plot(t, res.rssi_measured[lo:hi, k, node], color="tab:blue", linewidth=0.6, label="misurato")
        ax.plot(t, res.rssi_path_loss[lo:hi, k, node], color="crimson", linewidth=1.3, label="senza rumore (solo distanza)")
        ax.axvline(0.0, color="k", linestyle="--", linewidth=1.0, label="inizio separazione")
        ax.set_ylabel("RSSI [dBm]")
        ax.set_title(f"link {k}→{node}", fontsize=10)
        ax2 = ax.twinx()
        ax2.plot(t, mob.distances[lo:hi, k, node], color="tab:green", linewidth=1.0, alpha=0.7, label="distanza")
        ax2.set_ylabel("distanza [m]", color="tab:green")
    for ax in axes[-1]:
        ax.set_xlabel("tempo dall'inizio della separazione [s]")
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle("RSSI dei link verso il nodo separato")
    fig.tight_layout()
    fig.savefig(out_dir / "separation_rssi.png", dpi=150)
    plt.close(fig)


def plot_shadow_field(mob, res, cfg: dict, out_dir: Path, zoom_half_m: float = 300.0) -> None:
    """La mappa di shadowing con il percorso dei nodi sovrapposto: vista
    intera e vista ingrandita attorno al punto di separazione.
    """
    sf = res.shadow_field
    nx, ny = sf.values.shape
    x0, y0 = sf.origin
    extent = (x0, x0 + (nx - 1) * sf.resolution, y0, y0 + (ny - 1) * sf.resolution)
    node = cfg["separation"]["node_id"]
    idx = int(np.searchsorted(mob.t, cfg["separation"]["start_time"]))
    center = mob.positions[idx, node]
    vmax = float(np.percentile(np.abs(sf.values), 99))

    fig, axes = plt.subplots(1, 2, figsize=(13, 8))
    for ax, zoom in zip(axes, (False, True)):
        im = ax.imshow(sf.values.T, origin="lower", extent=extent, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.plot(mob.positions[:, :, 0].mean(axis=1), mob.positions[:, :, 1].mean(axis=1), "k-", linewidth=0.6, label="gruppo")
        ax.plot(mob.positions[:, node, 0], mob.positions[:, node, 1], color="lime", linewidth=0.8, label="nodo separato")
        ax.plot(*center, "k*", markersize=12, label="punto di separazione")
        if zoom:
            ax.set_xlim(center[0] - zoom_half_m, center[0] + zoom_half_m)
            ax.set_ylim(center[1] - zoom_half_m, center[1] + zoom_half_m)
            ax.set_title(f"Ingrandimento (±{zoom_half_m:.0f} m attorno alla separazione)")
        else:
            ax.set_title("Mappa intera")
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.colorbar(im, ax=axes, shrink=0.6, label="p(x) [dB/√m]")
    fig.savefig(out_dir / "shadow_field.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def empirical_std_vs_length(res, cfg: dict, lengths: np.ndarray, n_seg: int, rng: np.random.Generator) -> np.ndarray:
    """Deviazione standard empirica dello shadowing su segmenti casuali
    (interamente dentro la mappa) di data lunghezza, sulla mappa `res`.
    """
    sf = res.shadow_field
    nx, ny = sf.values.shape
    x0, y0 = sf.origin
    x1, y1 = x0 + (nx - 1) * sf.resolution, y0 + (ny - 1) * sf.resolution
    sh = cfg["channel"]["shadowing"]
    out = np.empty(len(lengths))
    for k, length in enumerate(lengths):
        a = np.column_stack([rng.uniform(x0, x1, 6 * n_seg), rng.uniform(y0, y1, 6 * n_seg)])
        ang = rng.uniform(0.0, 2.0 * np.pi, 6 * n_seg)
        b = a + length * np.column_stack([np.cos(ang), np.sin(ang)])
        ok = (b[:, 0] > x0) & (b[:, 0] < x1) & (b[:, 1] > y0) & (b[:, 1] < y1)
        a, b = a[ok][:n_seg], b[ok][:n_seg]
        out[k] = np.std(link_shadowing(sf, a, b, sh))
    return out


def plot_shadowing_vs_length(res, cfg: dict, out_dir: Path) -> None:
    """Deviazione standard dello shadowing in funzione della lunghezza del
    link: curva empirica (sulla mappa) contro la formula teorica.
    """
    sh = cfg["channel"]["shadowing"]
    lengths = np.logspace(0, 3, 16)
    rng = np.random.default_rng(cfg["simulation"]["seed"])
    emp = empirical_std_vs_length(res, cfg, lengths, 400, rng)
    grid = np.logspace(0, 3, 200)
    theory = np.sqrt(link_variance_theory(grid, sh["sigma"], sh["correlation_distance"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grid, theory, color="crimson", linewidth=1.5, label="formula teorica")
    ax.plot(lengths, emp, "o", color="tab:blue", label="empirica (mappa)")
    ax.axhline(sh["sigma"], color="0.4", linestyle=":", label=f"σ = {sh['sigma']:g} dB (modalità independent)")
    ax.set_xscale("log")
    ax.set_xlabel("lunghezza del link [m] (scala log)")
    ax.set_ylabel("deviazione standard dello shadowing [dB]")
    ax.set_title("Shadowing in funzione della lunghezza del link")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "shadowing_vs_length.png", dpi=150)
    plt.close(fig)


def separated_link_correlation(mob, res, cfg: dict, window_s: float = None) -> np.ndarray:
    """Matrice di correlazione dello shadowing fra i 4 link verso il nodo
    separato, sugli istanti dopo l'inizio della separazione: tutti, oppure
    solo i primi `window_s` secondi (la finestra in cui il rilevamento deve
    avvenire).
    """
    sep = cfg["separation"]
    node = sep["node_id"]
    others = [k for k in range(cfg["group"]["n_nodes"]) if k != node]
    idx = int(np.searchsorted(mob.t, sep["start_time"]))
    end = len(mob.t) if window_s is None else min(idx + int(round(window_s / cfg["simulation"]["dt"])), len(mob.t))
    series = np.stack([res.shadowing[idx:end, k, node] for k in others])
    return np.corrcoef(series)


def plot_link_correlation(corrs: dict, cfg: dict, out_dir: Path) -> None:
    """Matrici di correlazione dello shadowing fra i 4 link verso il nodo
    separato, in 2 righe (tutto il periodo dopo la separazione, primi
    `DETECTION_WINDOW_S` secondi) per 2 colonne (field, independent): è la
    figura che motiva il modello a mappa condivisa.
    `corrs` è indicizzato da (riga, colonna) con riga in {"all", "window"} e
    colonna in {"field", "independent"}.
    """
    node = cfg["separation"]["node_id"]
    labels = [f"{k}→{node}" for k in range(cfg["group"]["n_nodes"]) if k != node]
    rows = (("all", "tutto il periodo dopo la separazione"), ("window", f"primi {DETECTION_WINDOW_S:.0f} s"))
    cols = (("field", "field (mappa condivisa)"), ("independent", "independent (per link)"))
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for r, (row_key, row_title) in enumerate(rows):
        for c, (col_key, col_title) in enumerate(cols):
            ax, corr = axes[r, c], corrs[(row_key, col_key)]
            im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
            ax.set_xticks(range(len(labels)), labels)
            ax.set_yticks(range(len(labels)), labels)
            for a in range(len(labels)):
                for b in range(len(labels)):
                    ax.text(b, a, f"{corr[a, b]:.2f}", ha="center", va="center", fontsize=9)
            ax.set_title(f"{col_title}\n{row_title}", fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.8, label="correlazione")
    fig.suptitle("Correlazione dello shadowing fra i link verso il nodo separato")
    fig.savefig(out_dir / "link_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def blocked_fraction(res, cfg: dict) -> np.ndarray:
    """Frazione di tempo in cui ogni coppia ha almeno un corridore in mezzo
    (perdita degli altri corridori almeno metà di `loss`, cioè almeno un
    corridore con il centro entro il raggio dal link).
    """
    threshold = cfg["channel"]["body"]["others"]["loss"] / 2.0
    n = res.body_others.shape[1]
    blocked = np.nan_to_num(res.body_others) >= threshold
    frac = blocked.mean(axis=0)
    frac[np.arange(n), np.arange(n)] = np.nan
    return frac


def plot_body_loss(res, cfg: dict, out_dir: Path) -> None:
    """A sinistra il lobo del torso in coordinate polari (direzione di marcia
    verso destra, angoli positivi in senso antiorario), a destra la frazione
    di tempo in cui ogni coppia ha almeno un corridore in mezzo.
    """
    own = cfg["channel"]["body"]["own"]
    phi_b = np.pi / 2 if own["mount_side"] == "right" else -np.pi / 2
    phi = np.linspace(-np.pi, np.pi, 361)
    loss = own["max_loss"] * ((1.0 + np.cos(phi - phi_b)) / 2.0) ** own["lobe_exponent"]

    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 2, 1, projection="polar")
    ax.plot(phi, loss, color="tab:blue")
    ax.fill(phi, loss, color="tab:blue", alpha=0.25)
    ax.annotate("", xy=(0.0, own["max_loss"] * 1.05), xytext=(0.0, 0.0), arrowprops=dict(arrowstyle="->", color="crimson"))
    ax.set_title(f"Perdita del torso [dB] — freccia rossa = direzione di marcia\n(scheda a {own['mount_side']})", fontsize=10)

    ax2 = fig.add_subplot(1, 2, 2)
    frac = blocked_fraction(res, cfg)
    im = ax2.imshow(100.0 * frac, cmap="viridis")
    n = frac.shape[0]
    ax2.set_xticks(range(n))
    ax2.set_yticks(range(n))
    for a in range(n):
        for b in range(n):
            if a != b:
                ax2.text(b, a, f"{100 * frac[a, b]:.1f}", ha="center", va="center", color="w", fontsize=8)
    ax2.set_xlabel("nodo j")
    ax2.set_ylabel("nodo i")
    ax2.set_title("Tempo con almeno un corridore in mezzo [%]")
    fig.colorbar(im, ax=ax2, shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "body_loss.png", dpi=150)
    plt.close(fig)


def plot_rssi_histogram(res, sat_fraction: float, out_dir: Path, idx_end: int, cfg: dict) -> None:
    """Distribuzione dell'RSSI misurato nel gruppo, prima della separazione,
    con la frazione di campioni saturati scritta nel grafico.
    """
    r = _offdiag(res.rssi_measured[:idx_end]).ravel()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(r, bins=np.arange(r.min() - 0.5, r.max() + 1.5, 1.0), color="tab:blue", alpha=0.8)
    ax.axvline(cfg["channel"]["measurement"]["saturation_dbm"], color="crimson", linestyle="--", label="saturazione")
    ax.text(0.03, 0.95, f"campioni saturati: {100 * sat_fraction:.2f} %", transform=ax.transAxes, va="top")
    ax.set_xlabel("RSSI misurato [dBm]")
    ax.set_ylabel("conteggio")
    ax.set_title(f"Distribuzione dell'RSSI nel gruppo, prima della separazione ({len(r)} campioni)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.03, 0.88))
    fig.tight_layout()
    fig.savefig(out_dir / "rssi_histogram.png", dpi=150)
    plt.close(fig)


def _run(mob, cfg: dict, mode: str, label: str):
    """Esegue e cronometra una simulazione di canale."""
    c = copy.deepcopy(cfg)
    c["channel"]["shadowing"]["mode"] = mode
    t0 = time.perf_counter()
    res = simulate_channel(mob, c)
    elapsed = time.perf_counter() - t0
    print(f"[{label}] simulazione di canale: {elapsed:.1f} s  {res.metadata['timings']}")
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera le figure di ispezione del modulo di canale")
    parser.add_argument(
        "config",
        nargs="?",
        default=str(_REPO_ROOT / "config" / "default.yaml"),
        help="percorso del file di configurazione YAML",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    mode_main = cfg["channel"]["shadowing"]["mode"]

    cfg_nosep = copy.deepcopy(cfg)
    cfg_nosep["separation"]["enabled"] = False
    cfg_sep = copy.deepcopy(cfg)
    cfg_sep["separation"]["enabled"] = True

    t0 = time.perf_counter()
    mob_nosep = simulate_mobility(cfg_nosep)
    mob_sep = simulate_mobility(cfg_sep)
    print(f"mobilità (2 corse): {time.perf_counter() - t0:.1f} s")

    res_nosep = _run(mob_nosep, cfg_nosep, mode_main, "senza separazione")
    res_field = _run(mob_sep, cfg_sep, "field", "separazione, field")
    res_indep = _run(mob_sep, cfg_sep, "independent", "separazione, independent")

    out_dir = _REPO_ROOT / "results"
    out_dir.mkdir(exist_ok=True)

    idx_end = _group_window(mob_sep, cfg_sep)
    corrs = {}
    for row_key, window in (("all", None), ("window", DETECTION_WINDOW_S)):
        corrs[(row_key, "field")] = separated_link_correlation(mob_sep, res_field, cfg_sep, window)
        corrs[(row_key, "independent")] = separated_link_correlation(mob_sep, res_indep, cfg_sep, window)

    plot_rssi_vs_distance(mob_sep, res_field, cfg_sep, out_dir)
    plot_link_components(mob_nosep, res_nosep, cfg_nosep, out_dir)
    plot_separation_rssi(mob_sep, res_field, cfg_sep, out_dir)
    plot_shadow_field(mob_sep, res_field, cfg_sep, out_dir)
    plot_shadowing_vs_length(res_field, cfg_sep, out_dir)
    plot_link_correlation(corrs, cfg_sep, out_dir)
    plot_body_loss(res_nosep, cfg_nosep, out_dir)

    sat_dbm = cfg["channel"]["measurement"]["saturation_dbm"]
    group_meas = _offdiag(res_field.rssi_measured[:idx_end]).ravel()
    sat_fraction = float(np.mean(group_meas >= sat_dbm))
    plot_rssi_histogram(res_field, sat_fraction, out_dir, idx_end, cfg_sep)

    iu = np.triu_indices(corrs[("all", "field")].shape[0], k=1)
    frac_blocked = blocked_fraction(res_nosep, cfg_nosep)
    any_blocked = float(np.nanmean(frac_blocked))

    print(f"\nGruppo prima della separazione (t < {cfg['separation']['start_time']:.0f} s):")
    print(f"  frazione di campioni saturati: {100 * sat_fraction:.3f} %")
    print(f"  RSSI misurato: mediana {np.median(group_meas):.1f} dBm, 95° percentile {np.percentile(group_meas, 95):.1f} dBm")
    print(f"  frazione di tempo (media sulle coppie) con almeno un corridore in mezzo: {100 * any_blocked:.2f} %")
    for row_key, title in (("all", "tutto il periodo dopo la separazione"), ("window", f"primi {DETECTION_WINDOW_S:.0f} s dopo la separazione")):
        print(f"Correlazione media dello shadowing fra i link del nodo separato ({title}):")
        print(f"  field:       {corrs[(row_key, 'field')][iu].mean():.3f}")
        print(f"  independent: {corrs[(row_key, 'independent')][iu].mean():.3f}")
    print(f"Figure salvate in {out_dir}")


if __name__ == "__main__":
    main()

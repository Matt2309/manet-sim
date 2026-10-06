#!/usr/bin/env python3
"""Script di ispezione visiva del Blocco 3 (pacchetti).

Esegue due simulazioni di mobilità e canale (corsa intera senza separazione,
per le statistiche del gruppo, e con separazione). Sulla seconda calcola i
pacchetti con tre sensibilità (-92, -95 dBm e quella di default in
config/default.yaml), a parità di seme. Salva le figure in results/ e stampa
a console le statistiche principali. Configurazione: config/default.yaml se
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

from src.channel import simulate_channel
from src.mobility import load_config, simulate_mobility
from src.packets import reception_probability, simulate_packets

# Sensibilità alternative confrontate negli script (solo per l'ispezione, non
# parametri del modello); la terza è quella di config/default.yaml.
ALT_SENSITIVITIES = (-92.0, -95.0)
COLORS = ("tab:red", "tab:orange", "tab:blue")

# Finestre dei grafici attorno all'inizio della separazione, in secondi.
BEFORE_S = 60.0
AFTER_S = 300.0

# Larghezza della media mobile con cui si liscia la distanza per individuare il rientro.
RETURN_SMOOTHING_S = 10.0
RETURN_SEARCH_S = (60.0, 300.0)  # intervallo, dopo l'inizio, in cui cercare il rientro geometrico


# ---------------------------------------------------------------------------
# Utilità
# ---------------------------------------------------------------------------


def _sens_label(s: float) -> str:
    return f"{s:g} dBm"


def beacon_values(res, ch, mob):
    """Per ogni (i, k, j) valido: RSSI vero, distanza e ricezione.

    Restituisce array 1D (solo elementi con i != j e beacon esistente).
    """
    n = res.beacon_steps.shape[0]
    valid = res.beacon_steps >= 0
    mask = valid[:, :, None] & ~np.eye(n, dtype=bool)[:, None, :]
    ii, kk, jj = np.nonzero(mask)
    steps = res.beacon_steps[ii, kk]
    return {
        "rssi_true": ch.rssi_true[steps, ii, jj],
        "distance": mob.distances[steps, ii, jj],
        "received": res.received[ii, kk, jj],
        "passed_signal": res.passed_signal[ii, kk, jj],
        "tx": ii,
        "rx": jj,
        "time": res.beacon_times[ii, kk],
    }


def loss_run_lengths(res) -> list:
    """Lunghezze delle serie di beacon consecutivi persi, per ogni link i→j."""
    n = res.received.shape[0]
    out = []
    for i in range(n):
        n_valid = int(np.sum(res.beacon_steps[i] >= 0))
        for j in range(n):
            if i == j:
                continue
            lost = np.concatenate([[0], (~res.received[i, :n_valid, j]).astype(int), [0]])
            edges = np.flatnonzero(np.diff(lost))
            out.append(edges[1::2] - edges[0::2])
    return out


def forwarded_ages(res) -> np.ndarray:
    """Età delle informazioni inoltrate: osservatore m, link i→j con j != m, i != j."""
    n = res.knowledge_age.shape[1]
    m, i, j = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    mask = (j != m) & (i != j)
    ages = res.knowledge_age[:, mask]
    return ages[np.isfinite(ages)]


def direct_ages(res) -> np.ndarray:
    """Età dei link diretti: osservatore m = j, i != j."""
    n = res.knowledge_age.shape[1]
    out = [res.knowledge_age[:, m, i, m] for m in range(n) for i in range(n) if i != m]
    ages = np.concatenate(out)
    return ages[np.isfinite(ages)]


def return_window(mob, cfg: dict) -> tuple:
    """Finestra (t_inizio, t_fine) del rientro geometrico del nodo separato.

    La distanza media dal gruppo, lisciata, ha un massimo seguito da un
    minimo locale quando il percorso torna indietro. Si prende il minimo
    entro `RETURN_SEARCH_S` dopo l'inizio e, attorno ad esso, l'intervallo
    contiguo in cui la distanza sta sotto la metà fra il massimo precedente e
    il minimo.
    """
    sep = cfg["separation"]
    node, t0, dt = sep["node_id"], sep["start_time"], cfg["simulation"]["dt"]
    others = [k for k in range(cfg["group"]["n_nodes"]) if k != node]
    d = mob.distances[:, node, others].mean(axis=1)
    w = max(int(round(RETURN_SMOOTHING_S / dt)), 1)
    ds = np.convolve(d, np.ones(w) / w, mode="same")
    lo = int(np.searchsorted(mob.t, t0 + RETURN_SEARCH_S[0]))
    hi = int(np.searchsorted(mob.t, t0 + RETURN_SEARCH_S[1]))
    k_min = lo + int(np.argmin(ds[lo:hi]))
    k0 = int(np.searchsorted(mob.t, t0))
    d_peak = ds[k0:k_min].max()
    level = 0.5 * (d_peak + ds[k_min])
    a = k_min
    while a > k0 and ds[a - 1] < level:
        a -= 1
    b = k_min
    while b < len(ds) - 1 and ds[b + 1] < level:
        b += 1
    return float(mob.t[a]), float(mob.t[b])


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def plot_reception_curve(runs: dict, values: dict, cfg: dict, out_dir: Path) -> None:
    """Curva teorica con le tre sensibilità e, sopra, la frazione empirica di
    beacon ricevuti per intervalli di `rssi_true` (1 dB), dalla corsa con separazione.
    """
    rc = cfg["packets"]["reception"]
    bg = rc["background_loss"]
    x = np.linspace(-125.0, -80.0, 1000)
    edges = np.arange(-125.0, -79.0, 1.0)
    centres = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=(9, 6))
    for (s, res), color in zip(runs.items(), COLORS):
        rc_s = dict(rc, sensitivity_dbm=s)
        ax.plot(x, reception_probability(x, rc_s) * (1.0 - bg), color=color, linewidth=1.5,
                label=f"teoria, sensibilità {_sens_label(s)}")
        ax.axvline(s, color=color, linestyle=":", linewidth=0.9)
        v = values[s]
        idx = np.digitize(v["rssi_true"], edges) - 1
        frac = np.full(len(centres), np.nan)
        for b in range(len(centres)):
            sel = idx == b
            if sel.sum() >= 30:
                frac[b] = v["received"][sel].mean()
        ax.plot(centres, frac, "o", markersize=4, color=color, label=f"empirica, {_sens_label(s)}")
    ax.axhline(1.0 - bg, color="0.5", linestyle="--", linewidth=0.8)
    ax.set_xlabel("RSSI vero [dBm]")
    ax.set_ylabel("frazione di beacon ricevuti")
    ax.set_title("Curva di ricezione: teoria (curva × fondo) ed empirica")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "reception_curve.png", dpi=150)
    plt.close(fig)


def plot_separation_packets(mob, ch, res, cfg: dict, out_dir: Path) -> None:
    """I 4 link dal nodo separato verso il gruppo, da un minuto prima a
    cinque minuti dopo l'inizio della separazione: `rssi_true` sottile,
    beacon ricevuti come punti, persi come segni in basso, sensibilità e,
    su un secondo asse, la distanza.
    """
    sep = cfg["separation"]
    node, t0, dt = sep["node_id"], sep["start_time"], cfg["simulation"]["dt"]
    sens = cfg["packets"]["reception"]["sensitivity_dbm"]
    others = [k for k in range(cfg["group"]["n_nodes"]) if k != node]
    lo = int(np.searchsorted(mob.t, t0 - BEFORE_S))
    hi = min(int(np.searchsorted(mob.t, t0 + AFTER_S)), len(mob.t))
    t = mob.t[lo:hi] - t0
    bt = res.beacon_times[node]
    sel = (bt >= mob.t[lo]) & (bt < mob.t[hi - 1])
    tb = bt[sel] - t0
    sb = res.beacon_steps[node][sel]

    y_min = float(np.nanmin(ch.rssi_true[lo:hi, node, others])) - 4.0
    y_max = float(np.nanmax(ch.rssi_true[lo:hi, node, others])) + 4.0
    mark_y = y_min + 2.0

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for ax, k in zip(axes.ravel(), others):
        rec = res.received[node, sel, k]
        ax.plot(t, ch.rssi_true[lo:hi, node, k], color="tab:blue", linewidth=0.5, alpha=0.7, label="RSSI vero")
        ax.plot(tb[rec], res.rssi[node, sel, k][rec], "o", color="k", markersize=2.5, label="beacon ricevuti")
        ax.plot(tb[~rec], np.full((~rec).sum(), mark_y), "|", color="crimson", markersize=7, label="beacon persi")
        ax.axhline(sens, color="tab:green", linestyle="--", linewidth=1.0, label="sensibilità")
        ax.axvline(0.0, color="k", linestyle="--", linewidth=0.8)
        ax.set_ylim(y_min, y_max)
        ax.set_ylabel("RSSI [dBm]")
        ax.set_title(f"link {node}→{k}", fontsize=10)
        ax2 = ax.twinx()
        ax2.plot(t, mob.distances[lo:hi, node, k], color="tab:purple", linewidth=1.0, alpha=0.6)
        ax2.set_ylabel("distanza [m]", color="tab:purple")
    for ax in axes[-1]:
        ax.set_xlabel("tempo dall'inizio della separazione [s]")
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Beacon dal nodo separato verso il gruppo")
    fig.tight_layout()
    fig.savefig(out_dir / "separation_packets.png", dpi=150)
    plt.close(fig)


def plot_loss_vs_distance(values: dict, out_dir: Path) -> None:
    """Frazione di beacon persi in funzione della distanza, per tutte le coppie."""
    edges = np.logspace(0, np.log10(900.0), 40)
    centres = np.sqrt(edges[:-1] * edges[1:])
    fig, ax = plt.subplots(figsize=(9, 6))
    for (s, v), color in zip(values.items(), COLORS):
        idx = np.digitize(v["distance"], edges) - 1
        frac = np.full(len(centres), np.nan)
        for b in range(len(centres)):
            sel = idx == b
            if sel.sum() >= 30:
                frac[b] = 1.0 - v["received"][sel].mean()
        ax.plot(centres, frac, "o-", markersize=3, color=color, label=f"sensibilità {_sens_label(s)}")
    ax.set_xscale("log")
    ax.set_xlabel("distanza [m] (scala log)")
    ax.set_ylabel("frazione di beacon persi")
    ax.set_title("Beacon persi in funzione della distanza (tutte le coppie)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_vs_distance.png", dpi=150)
    plt.close(fig)


def plot_link_age(mob, runs: dict, res_nosep, cfg: dict, out_dir: Path) -> None:
    """A sinistra l'età dei link diretti verso il nodo separato nel tempo (media
    sui 4 link, vista dal nodo separato), per le tre sensibilità; a destra,
    nel gruppo senza separazione, la distribuzione del numero di beacon
    consecutivi persi.
    """
    sep = cfg["separation"]
    node, t0 = sep["node_id"], sep["start_time"]
    others = [k for k in range(cfg["group"]["n_nodes"]) if k != node]
    lo = int(np.searchsorted(mob.t, t0 - BEFORE_S))
    hi = min(int(np.searchsorted(mob.t, t0 + AFTER_S)), len(mob.t))
    t = mob.t[lo:hi] - t0

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for (s, res), color in zip(runs.items(), COLORS):
        age = np.nanmean(res.knowledge_age[lo:hi, node, others, node], axis=1)
        ax.plot(t, age, color=color, linewidth=1.0, label=f"sensibilità {_sens_label(s)}")
    ax.axvline(0.0, color="k", linestyle="--", linewidth=0.8)
    ax.set_yscale("symlog", linthresh=2.0 * cfg["packets"]["beacon_period"])  # soglia: due periodi
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("tempo dall'inizio della separazione [s]")
    ax.set_ylabel("età media dei link diretti verso il nodo separato [s]")
    ax.set_title("Età dei link diretti verso il nodo separato")
    ax.legend()
    ax.grid(alpha=0.3)

    lengths = np.concatenate(loss_run_lengths(res_nosep))
    top = 8
    counts = np.array([np.sum(lengths == k) for k in range(1, top)] + [np.sum(lengths >= top)])
    ax2.bar(np.arange(1, top + 1), counts, color="tab:blue")
    ax2.set_yscale("log")
    ax2.set_xticks(np.arange(1, top + 1), [str(k) for k in range(1, top)] + [f"≥{top}"])
    ax2.set_xlabel("beacon consecutivi persi")
    ax2.set_ylabel("numero di serie")
    ax2.set_title("Serie di beacon persi nel gruppo (senza separazione)")
    ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_dir / "link_age.png", dpi=150)
    plt.close(fig)


def plot_knowledge_delay(res_nosep, cfg: dict, out_dir: Path) -> None:
    """Distribuzione dell'età delle informazioni inoltrate nel gruppo (il
    ritardo con cui un nodo sa cosa vedono gli altri), a confronto con quella dei link diretti.
    """
    fwd = forwarded_ages(res_nosep)
    dire = direct_ages(res_nosep)
    period = cfg["packets"]["beacon_period"]
    bins = np.arange(0.0, 8.0 * period + period / 10.0, period / 10.0)  # 8 periodi, 10 classi per periodo
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(dire, bins=bins, density=True, alpha=0.55, color="tab:green", label="link diretti")
    ax.hist(fwd, bins=bins, density=True, alpha=0.55, color="tab:blue", label="informazioni inoltrate")
    ax.axvline(fwd.mean(), color="tab:blue", linestyle="--", label=f"media inoltrate {fwd.mean():.2f} s")
    ax.axvline(np.percentile(fwd, 95), color="crimson", linestyle="--",
               label=f"95° percentile inoltrate {np.percentile(fwd, 95):.2f} s")
    ax.set_xlabel("età dell'informazione [s]")
    ax.set_ylabel("densità")
    ax.set_title("Ritardo con cui un nodo sa cosa vedono gli altri (gruppo senza separazione)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "knowledge_delay.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Principale
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera le figure di ispezione del modulo dei pacchetti")
    parser.add_argument(
        "config",
        nargs="?",
        default=str(_REPO_ROOT / "config" / "default.yaml"),
        help="percorso del file di configurazione YAML",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg_nosep = copy.deepcopy(cfg)
    cfg_nosep["separation"]["enabled"] = False
    cfg_sep = copy.deepcopy(cfg)
    cfg_sep["separation"]["enabled"] = True

    t0 = time.perf_counter()
    mob_nosep = simulate_mobility(cfg_nosep)
    mob_sep = simulate_mobility(cfg_sep)
    print(f"mobilità (2 corse): {time.perf_counter() - t0:.1f} s")
    t0 = time.perf_counter()
    ch_nosep = simulate_channel(mob_nosep, cfg_nosep)
    ch_sep = simulate_channel(mob_sep, cfg_sep)
    print(f"canale (2 corse): {time.perf_counter() - t0:.1f} s")

    t0 = time.perf_counter()
    res_nosep = simulate_packets(mob_nosep, ch_nosep, cfg_nosep)
    block3_time = time.perf_counter() - t0

    default_sens = cfg["packets"]["reception"]["sensitivity_dbm"]
    sensitivities = list(ALT_SENSITIVITIES) + [default_sens]
    runs, values = {}, {}
    for s in sensitivities:
        c = copy.deepcopy(cfg_sep)
        c["packets"]["reception"]["sensitivity_dbm"] = s
        runs[s] = simulate_packets(mob_sep, ch_sep, c)
        values[s] = beacon_values(runs[s], ch_sep, mob_sep)
    res_sep = runs[default_sens]

    out_dir = _REPO_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    plot_reception_curve(runs, values, cfg_sep, out_dir)
    plot_separation_packets(mob_sep, ch_sep, res_sep, cfg_sep, out_dir)
    plot_loss_vs_distance(values, out_dir)
    plot_link_age(mob_sep, runs, res_nosep, cfg_sep, out_dir)
    plot_knowledge_delay(res_nosep, cfg_sep, out_dir)

    # --- statistiche del gruppo -------------------------------------------
    md = res_nosep.metadata
    n = cfg["group"]["n_nodes"]
    n_links = n * (n - 1)
    hours = mob_nosep.t[-1] / 3600.0
    print(f"\nGruppo senza separazione ({mob_nosep.t[-1]:.0f} s, {n_links} link, "
          f"beacon ogni {cfg['packets']['beacon_period']:g} s):")
    print(f"  beacon ricevuti: {100 * md['received_fraction']:.2f} %")
    print(f"  persi per segnale debole: {100 * md['lost_signal_fraction']:.2f} %")
    print(f"  persi per fondo (segnale sufficiente): {100 * md['lost_background_only_fraction']:.2f} %")
    lengths = np.concatenate(loss_run_lengths(res_nosep))
    for label, count in (("2", np.sum(lengths == 2)), ("3", np.sum(lengths == 3)), ("≥5", np.sum(lengths >= 5))):
        print(f"  serie di {label} beacon persi consecutivi: {count} totali, "
              f"{count / hours / n_links:.2f} per ora per link")
    fwd = forwarded_ages(res_nosep)
    print(f"Età delle informazioni inoltrate: media {fwd.mean():.2f} s, 95° percentile {np.percentile(fwd, 95):.2f} s")

    # --- distacco, per ogni sensibilità ------------------------------------
    sep = cfg["separation"]
    node, t_sep = sep["node_id"], sep["start_time"]
    others = [k for k in range(n) if k != node]
    w_lo, w_hi = return_window(mob_sep, cfg_sep)
    print(f"\nRientro geometrico del nodo {node}: da {w_lo - t_sep:.0f} s a {w_hi - t_sep:.0f} s dopo l'inizio della separazione")
    for s in sensitivities:
        res = runs[s]
        bt = res.beacon_times[node]
        heard = res.received[node][:, others].any(axis=1) & (bt >= t_sep)
        in_window = (bt >= w_lo) & (bt <= w_hi)
        n_ret_rx = int(np.sum(heard & in_window))
        n_ret_tx = int(np.sum(in_window))
        if heard.any():
            k_last = int(np.flatnonzero(heard)[-1])
            rx_last = [k for k in others if res.received[node, k_last, k]]
            step = res.beacon_steps[node, k_last]
            dist = float(np.mean(mob_sep.distances[step, node, rx_last]))
            print(f"  sensibilità {s:g} dBm: ultimo beacon ricevuto da un compagno {bt[k_last] - t_sep:.1f} s "
                  f"dopo l'inizio, a {dist:.0f} m; beacon ricevuti durante il rientro: {n_ret_rx} su {n_ret_tx}")
        else:
            print(f"  sensibilità {s:g} dBm: nessun beacon ricevuto dopo l'inizio")

    print(f"\nTempo di esecuzione del solo Blocco 3 (corsa intera, senza separazione): {block3_time:.1f} s")
    print(f"Figure salvate in {out_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Visual inspection script for Block 3 (packets).

Two mobility and channel simulations (without separation, for the group
statistics, and with separation); on the second one the packets with three
sensitivities (-92, -95 dBm and the default one), with the same seed. Saves
the figures in results/ and prints the statistics. Config: config/default.yaml
if not given.
"""

import copy
import time
from pathlib import Path

import numpy as np

from _cli import make_parser  # noqa: E402
from src.channel import simulate_channel
from src.common.config import separation_variants
from src.common.paths import block_dir
from src.common.plotting import plt, save_figure
from src.mobility import load_config, simulate_mobility
from src.packets import reception_probability, simulate_packets

# Alternative sensitivities compared in the scripts (inspection only, not
# model parameters); the third is the one in config/default.yaml.


ALT_SENSITIVITIES = (-92.0, -95.0)
COLORS = ("tab:red", "tab:orange", "tab:blue")

# Plot windows around the start of the separation, in seconds.
BEFORE_S = 60.0
AFTER_S = 300.0

# Width of the moving average used to smooth the distance to find the return.
RETURN_SMOOTHING_S = 10.0
RETURN_SEARCH_S = (60.0, 300.0)  # interval, after the start, in which to look for the geometric return


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _sens_label(s: float) -> str:
    return f"{s:g} dBm"


def beacon_values(res, ch, mob):
    """For each valid (i, k, j): true RSSI, distance and reception.

    Returns 1D arrays (only elements with i != j and an existing beacon).
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
    """Lengths of the runs of consecutive lost beacons, for each link i→j."""
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
    """Age of the forwarded information: observer m, link i→j with j != m, i != j."""
    n = res.knowledge_age.shape[1]
    m, i, j = np.meshgrid(np.arange(n), np.arange(n), np.arange(n), indexing="ij")
    mask = (j != m) & (i != j)
    ages = res.knowledge_age[:, mask]
    return ages[np.isfinite(ages)]


def direct_ages(res) -> np.ndarray:
    """Age of the direct links: observer m = j, i != j."""
    n = res.knowledge_age.shape[1]
    out = [res.knowledge_age[:, m, i, m] for m in range(n) for i in range(n) if i != m]
    ages = np.concatenate(out)
    return ages[np.isfinite(ages)]


def return_window(mob, cfg: dict) -> tuple:
    """Window (t_start, t_end) of the geometric return of the separated node.

    Minimum of the smoothed mean distance within `RETURN_SEARCH_S` of the start; the contiguous interval around it with distance below the midpoint between the previous maximum and the minimum.
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
# Figures
# ---------------------------------------------------------------------------


def plot_reception_curve(runs: dict, values: dict, cfg: dict, out_dir: Path) -> None:
    """Theoretical curve with the three sensitivities and empirical fraction of received beacons per `rssi_true` bin (1 dB), from the run with separation."""
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
    save_figure(fig, out_dir / "reception_curve.png", 150)


def plot_separation_packets(mob, ch, res, cfg: dict, out_dir: Path) -> None:
    """The 4 links from the separated node (from -1 to +5 min from the separation): `rssi_true`, received beacons (dots) and lost ones (marks at the bottom), sensitivity, distance on a second axis."""
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
    save_figure(fig, out_dir / "separation_packets.png", 150)


def plot_loss_vs_distance(values: dict, out_dir: Path) -> None:
    """Fraction of lost beacons as a function of distance, for all pairs."""
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
    save_figure(fig, out_dir / "loss_vs_distance.png", 150)


def plot_link_age(mob, runs: dict, res_nosep, cfg: dict, out_dir: Path) -> None:
    """Left: age of the direct links towards the separated node (mean over the 4 links) for the three sensitivities; right: consecutive lost beacons in the group without separation."""
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
    ax.set_yscale("symlog", linthresh=2.0 * cfg["packets"]["beacon_period"])  # threshold: two periods
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
    save_figure(fig, out_dir / "link_age.png", 150)


def plot_knowledge_delay(res_nosep, cfg: dict, out_dir: Path) -> None:
    """Distribution of the age of the forwarded information in the group, compared with the direct links."""
    fwd = forwarded_ages(res_nosep)
    direct = direct_ages(res_nosep)
    period = cfg["packets"]["beacon_period"]
    bins = np.arange(0.0, 8.0 * period + period / 10.0, period / 10.0)  # 8 periods, 10 bins per period
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(direct, bins=bins, density=True, alpha=0.55, color="tab:green", label="link diretti")
    ax.hist(fwd, bins=bins, density=True, alpha=0.55, color="tab:blue", label="informazioni inoltrate")
    ax.axvline(fwd.mean(), color="tab:blue", linestyle="--", label=f"media inoltrate {fwd.mean():.2f} s")
    ax.axvline(np.percentile(fwd, 95), color="crimson", linestyle="--",
               label=f"95° percentile inoltrate {np.percentile(fwd, 95):.2f} s")
    ax.set_xlabel("età dell'informazione [s]")
    ax.set_ylabel("densità")
    ax.set_title("Ritardo con cui un nodo sa cosa vedono gli altri (gruppo senza separazione)")
    ax.legend()
    save_figure(fig, out_dir / "knowledge_delay.png", 150)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = make_parser("Generate the inspection figures of the packets module", config_help="path to the YAML configuration file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg_nosep, cfg_sep = separation_variants(cfg)

    t0 = time.perf_counter()
    mob_nosep = simulate_mobility(cfg_nosep)
    mob_sep = simulate_mobility(cfg_sep)
    print(f"mobility (2 runs): {time.perf_counter() - t0:.1f} s")
    t0 = time.perf_counter()
    ch_nosep = simulate_channel(mob_nosep, cfg_nosep)
    ch_sep = simulate_channel(mob_sep, cfg_sep)
    print(f"channel (2 runs): {time.perf_counter() - t0:.1f} s")

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

    out_dir = block_dir(3)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_reception_curve(runs, values, cfg_sep, out_dir)
    plot_separation_packets(mob_sep, ch_sep, res_sep, cfg_sep, out_dir)
    plot_loss_vs_distance(values, out_dir)
    plot_link_age(mob_sep, runs, res_nosep, cfg_sep, out_dir)
    plot_knowledge_delay(res_nosep, cfg_sep, out_dir)

    # --- group statistics -------------------------------------------
    md = res_nosep.metadata
    n = cfg["group"]["n_nodes"]
    n_links = n * (n - 1)
    hours = mob_nosep.t[-1] / 3600.0
    print(f"\nGroup without separation ({mob_nosep.t[-1]:.0f} s, {n_links} links, "
          f"beacon every {cfg['packets']['beacon_period']:g} s):")
    print(f"  beacons received: {100 * md['received_fraction']:.2f} %")
    print(f"  lost due to weak signal: {100 * md['lost_signal_fraction']:.2f} %")
    print(f"  lost due to background (sufficient signal): {100 * md['lost_background_only_fraction']:.2f} %")
    lengths = np.concatenate(loss_run_lengths(res_nosep))
    for label, count in (("2", np.sum(lengths == 2)), ("3", np.sum(lengths == 3)), ("≥5", np.sum(lengths >= 5))):
        print(f"  runs of {label} consecutive lost beacons: {count} total, "
              f"{count / hours / n_links:.2f} per hour per link")
    fwd = forwarded_ages(res_nosep)
    print(f"Age of the forwarded information: mean {fwd.mean():.2f} s, 95th percentile {np.percentile(fwd, 95):.2f} s")

    # --- separation, for each sensitivity ------------------------------------
    sep = cfg["separation"]
    node, t_sep = sep["node_id"], sep["start_time"]
    others = [k for k in range(n) if k != node]
    w_lo, w_hi = return_window(mob_sep, cfg_sep)
    print(f"\nGeometric return of node {node}: from {w_lo - t_sep:.0f} s to {w_hi - t_sep:.0f} s after the start of the separation")
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
            print(f"  sensitivity {s:g} dBm: last beacon received by a teammate {bt[k_last] - t_sep:.1f} s "
                  f"after the start, at {dist:.0f} m; beacons received during the return: {n_ret_rx} out of {n_ret_tx}")
        else:
            print(f"  sensitivity {s:g} dBm: no beacon received after the start")

    print(f"\nExecution time of Block 3 alone (full run, without separation): {block3_time:.1f} s")
    print(f"Figures saved in {out_dir}")


if __name__ == "__main__":
    main()

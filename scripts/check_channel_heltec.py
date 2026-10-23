#!/usr/bin/env python3
"""Check of the channel against the Heltec measurements (results/channel_check/). It measures and compares, it does not change the model.

3a. Normal run against the run of 13 October: the first ``channel_check.n_runs`` ``none`` runs of the tuning set, whole; all the direct links of the group,
    beacons RECEIVED as in the real data. Statistics against the reference with tolerances; by distance band; mean of the channel components.
3b. Test A (informative only): two static nodes at 5 m, no shadowing, no board offsets, fading on; loss of the body with respect to the boards on supports.
3c. Range table (level of the distance alone) and distance of the separated node from the group at its last beacon received by the group (slowdown, stop).
Exit code 1 if 3a does not respect the tolerances (the tables are written anyway).
Config: config/default.yaml if not given; ``--set key=value`` replaces values.
"""

import math
import sys
from pathlib import Path

import numpy as np

from _cli import load_cli_config, make_parser  # noqa: E402
from src.channel import distance_from_path_loss, path_loss_db, simulate_channel  # noqa: E402
from src.common.config import apply_overrides  # noqa: E402
from src.common.io import save_pickle  # noqa: E402
from src.common.parallel import parallel_map  # noqa: E402
from src.common.paths import RESULTS_DIR  # noqa: E402
from src.metrics import min_distance_to_group, nearest_rank  # noqa: E402
from src.mobility import MobilityResult  # noqa: E402
from src.simulator import experiment_runs, load_or_simulate, simulate_pipeline  # noqa: E402

COMPONENTS = ("own", "others", "shadow")


# ---------------------------------------------------------------------------
# 3a. Normal run
# ---------------------------------------------------------------------------


def link_samples(config: dict) -> dict:
    """One run: every beacon of every direct link ``i -> j``, as arrays over (link, beacon).

    ``link`` (index i * N + j), ``t`` (s), ``received`` (bool), ``rssi`` (dBm, NaN if lost), ``dist`` (m) and the dB of the channel components at the beacon instant
    (``own`` = own body, ``others`` = other runners, ``shadow`` = shadowing). Also ``duration`` (s).
    """
    mob, ch, pk = simulate_pipeline(config)
    n = pk.received.shape[0]
    cols = {k: [] for k in ("link", "t", "received", "rssi", "dist", *COMPONENTS)}
    for i in range(n):
        ok = pk.beacon_steps[i] >= 0
        steps = pk.beacon_steps[i][ok]
        for j in range(n):
            if i == j:
                continue
            cols["link"].append(np.full(len(steps), i * n + j))
            cols["t"].append(pk.beacon_times[i][ok])
            cols["received"].append(pk.received[i][:, j][ok])
            cols["rssi"].append(pk.rssi[i][:, j][ok])
            cols["dist"].append(mob.distances[steps, i, j])
            cols["own"].append(ch.body_own[steps, i, j])
            cols["others"].append(ch.body_others[steps, i, j])
            cols["shadow"].append(ch.shadowing[steps, i, j])
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out["duration"] = float(mob.t[-1] - mob.t[0])
    out["n_nodes"] = n
    return out


def pool(parts: list) -> dict:
    """Concatenates the runs; the run index goes into ``link`` so that a link of each run is a distinct key."""
    n = parts[0]["n_nodes"]
    keys = ("t", "received", "rssi", "dist", *COMPONENTS)
    out = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    out["link"] = np.concatenate([p["link"] + r * n * n for r, p in enumerate(parts)])
    out["link_hours"] = sum(p["duration"] * n * (n - 1) for p in parts) / 3600.0
    return out


def long_gaps(d: dict, seconds: list) -> list:
    """Silences >= each value of `seconds` between consecutive RECEIVED beacons of the same link, per link-hour."""
    counts = np.zeros(len(seconds))
    rec = d["received"]
    link, t = d["link"][rec], d["t"][rec]
    order = np.lexsort((t, link))
    link, t = link[order], t[order]
    same = link[1:] == link[:-1]
    gaps = (t[1:] - t[:-1])[same]
    for k, s in enumerate(seconds):
        counts[k] = (gaps >= s).sum()
    return list(counts / d["link_hours"])


def sample_stats(d: dict, mask: np.ndarray, below_level: float) -> dict:
    """Received fraction, percentiles of the received RSSI, fraction of the received below `below_level` and mean of the components (received beacons)."""
    rec = mask & d["received"]
    r = d["rssi"][rec]
    out = {"emitted": int(mask.sum()), "received_fraction": float(rec.sum() / max(mask.sum(), 1))}
    if len(r) == 0:
        return {**out, "median": math.nan, "p25": math.nan, "p5": math.nan, "below_fraction": math.nan, **{c: math.nan for c in COMPONENTS}}
    out.update(median=float(np.percentile(r, 50)), p25=float(np.percentile(r, 25)), p5=float(np.percentile(r, 5)), below_fraction=float((r < below_level).mean()))
    out.update({c: float(d[c][rec].mean()) for c in COMPONENTS})
    return out


def compare(stats: dict, gaps: list, ccfg: dict) -> list:
    """Rows ``(name, value, reference, tolerance text, passes)`` of the table of 3a."""
    ref, tol = ccfg["reference"], ccfg["tolerance"]
    near = lambda v, refs: all(abs(v - r) <= tol["level_db"] for r in refs)  # noqa: E731
    return [
        ("beacons received", 100 * stats["received_fraction"], [100 * r for r in ref["received_fraction"]], f">= {100 * tol['received_min']:g}%", stats["received_fraction"] >= tol["received_min"]),
        ("median RSSI", stats["median"], ref["median"], f"+/- {tol['level_db']:g} dB", near(stats["median"], ref["median"])),
        ("25th percentile", stats["p25"], ref["p25"], f"+/- {tol['level_db']:g} dB", near(stats["p25"], ref["p25"])),
        ("5th percentile", stats["p5"], ref["p5"], f"+/- {tol['level_db']:g} dB", near(stats["p5"], ref["p5"])),
        (f"received below {ccfg['below_level']:g}", 100 * stats["below_fraction"], [100 * r for r in ref["below_fraction"]], f"<= {100 * tol['below_max']:g}%", stats["below_fraction"] <= tol["below_max"]),
        (f"gaps >= {ccfg['gap_seconds'][0]:g} s per link-hour", gaps[0], [0.0], "indicative", None),
        (f"gaps >= {ccfg['gap_seconds'][1]:g} s per link-hour", gaps[1], [0.0], f"<= {tol['long_gap_max_per_link_hour']:g} (about 0)", gaps[1] <= tol["long_gap_max_per_link_hour"]),
    ]


def table_normal_run(d: dict, ccfg: dict) -> tuple:
    """Text of 3a (comparison, by distance, per link) and whether it passes."""
    below = ccfg["below_level"]
    allmask = np.ones(len(d["t"]), bool)
    stats = sample_stats(d, allmask, below)
    gaps = long_gaps(d, ccfg["gap_seconds"])
    rows = compare(stats, gaps, ccfg)
    lines = [f"{'statistic':<28} {'simulated':>10} {'13 Oct (3->1 / 1->3)':>22} {'tolerance':>22}  result"]
    for name, value, refs, tol, ok in rows:
        lines.append(f"{name:<28} {value:>10.2f} {' / '.join(f'{r:g}' for r in refs):>22} {tol:>22}  {'-' if ok is None else ('PASS' if ok else 'FAIL')}")
    passes = all(ok for *_, ok in rows if ok is not None)
    per_link = [float(np.median(d["rssi"][(d["link"] == k) & d["received"]])) for k in np.unique(d["link"]) if ((d["link"] == k) & d["received"]).any()]
    lines.append(f"median per link (all {len(per_link)} links of the runs): min {min(per_link):.1f}, max {max(per_link):.1f} dBm")
    lines.append(f"mean components over the received beacons: own body {stats['own']:.2f} dB, other runners {stats['others']:.2f} dB, shadowing {stats['shadow']:+.2f} dB")
    lines += ["", "By distance band (received beacons):"]
    lines.append(f"{'band (m)':<10} {'beacons':>9} {'received %':>11} {'median':>7} {'p25':>7} {'p5':>7} {'< ' + format(below, 'g') + ' %':>9} {'own body':>9} {'others':>8} {'shadow':>8}")
    edges = ccfg["distance_bins"]
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = sample_stats(d, (d["dist"] >= lo) & (d["dist"] < hi), below)
        lines.append(
            f"{lo:g}-{hi:g}".ljust(10)
            + f" {s['emitted']:>9d} {100 * s['received_fraction']:>11.1f} {s['median']:>7.1f} {s['p25']:>7.1f} {s['p5']:>7.1f} {100 * s['below_fraction']:>9.1f}"
            + f" {s['own']:>9.2f} {s['others']:>8.2f} {s['shadow']:>+8.2f}"
        )
    return lines, passes, {"stats": stats, "gaps": gaps, "rows": rows}


# ---------------------------------------------------------------------------
# 3b. Test A
# ---------------------------------------------------------------------------


def static_pair(distance: float, heading_a: float, heading_b: float, n_steps: int, dt: float) -> MobilityResult:
    """Node A at the origin, node B at `distance` along x, fixed headings (rad)."""
    positions = np.zeros((n_steps, 2, 2))
    positions[:, 1, 0] = distance
    headings = np.zeros((n_steps, 2, 2))
    headings[:, 0] = [math.cos(heading_a), math.sin(heading_a)]
    headings[:, 1] = [math.cos(heading_b), math.sin(heading_b)]
    dist = np.zeros((n_steps, 2, 2))
    dist[:, 0, 1] = dist[:, 1, 0] = distance
    return MobilityResult(
        t=np.arange(n_steps) * dt, positions=positions, s_nodes=np.zeros((n_steps, 2)), s_centroid=np.zeros(n_steps), distances=dist,
        headings=headings, lateral_offsets=np.zeros((n_steps, 2)), metadata={},
    )  # fmt: skip


def table_test_a(config: dict) -> tuple:
    """Median loss of the model with respect to the boards on supports (same fading), per position, against the measured value."""
    acfg = config["channel_check"]["test_a"]
    cfg = apply_overrides(config, {"channel.shadowing.enabled": False, "channel.device_offset.enabled": False})
    no_body = apply_overrides(cfg, {"channel.body.own.enabled": False})
    dt = config["simulation"]["dt"]
    n_steps = int(round(acfg["duration"] / dt))
    lines = [f"{'position':<30} {'phi A/B':>12} {'A->B':>7} {'B->A':>7} {'mean':>7} {'measured':>9} {'model - measured':>17}"]
    rows = []
    for pos in acfg["positions"]:
        phi_a, phi_b = math.radians(pos["phi_a"]), math.radians(pos["phi_b"])
        mob = static_pair(acfg["distance"], -phi_a, math.pi - phi_b, n_steps, dt)  # phi_A = angle(v) - heading_A, with v = (1, 0) (A->B) and (-1, 0) (B->A)
        with_body, without = simulate_channel(mob, cfg).rssi_true, simulate_channel(mob, no_body).rssi_true
        ab = float(np.median(with_body[:, 0, 1] - without[:, 0, 1]))
        ba = float(np.median(with_body[:, 1, 0] - without[:, 1, 0]))
        mean = (ab + ba) / 2
        rows.append((pos["name"], ab, ba, mean, pos["measured_db"]))
        lines.append(f"{pos['name']:<30} {pos['phi_a']:>5g}/{pos['phi_b']:<6g} {ab:>7.1f} {ba:>7.1f} {mean:>7.1f} {pos['measured_db']:>9.1f} {mean - pos['measured_db']:>17.1f}")
    lines.append("(dB, negative = loss; the last measured position is 'at most -17 dB'. Informative only: the model is not corrected, see 3a.)")
    return lines, rows


# ---------------------------------------------------------------------------
# 3c. Range and separated node
# ---------------------------------------------------------------------------


def table_range(config: dict) -> list:
    ch = config["channel"]
    sensitivity = config["packets"]["reception"]["sensitivity_dbm"]
    budget = ch["tx_power"] + ch["antenna_gain_tx"] + ch["antenna_gain_rx"]
    lines = [f"{'distance (m)':>12} {'RSSI of the distance alone (dBm)':>34}"]
    for d in config["channel_check"]["range_distances"]:
        lines.append(f"{d:>12g} {budget - float(path_loss_db(np.array(d), ch)):>34.1f}")
    lines.append(f"distance at {sensitivity:g} dBm (sensitivity): {distance_from_path_loss(sensitivity, ch):.1f} m")
    return lines


def last_beacon_distance(args: tuple) -> dict:
    """Separated node: distance from the group at its last beacon received by someone in the group (run from the cache), and whether it was received to the end."""
    run_info, cache_dir = args
    run = load_or_simulate(run_info["config"], cache_dir)
    node, t = run["sep_node"], run["t"]
    got = run["received"][node].any(axis=1) & ~np.isnan(run["beacon_times"][node])
    if not got.any():
        return {"distance": math.nan, "to_the_end": False}
    last = float(run["beacon_times"][node][np.flatnonzero(got)[-1]])
    idx = min(int(np.searchsorted(t, last)), len(t) - 1)
    return {"distance": float(min_distance_to_group(run["distances"], node)[idx]), "to_the_end": bool(t[-1] - last < run_info["config"]["detector"]["lost_silence"])}


def table_separated(config: dict, cache_dir: Path, jobs: int) -> tuple:
    n = config["channel_check"]["separated_runs"]
    runs = experiment_runs(config)
    lines = [f"{'scenario':<10} {'runs':>5} {'median (m)':>11} {'p90 (m)':>9} {'still heard at the end':>24}"]
    out = {}
    for scenario in ("slowdown", "stop"):
        sel = [r for r in runs if r["scenario"] == scenario and r["index"] < n]
        res = parallel_map(last_beacon_distance, [(r, cache_dir) for r in sel], jobs)
        dist = np.array([r["distance"] for r in res])
        out[scenario] = dist
        lines.append(f"{scenario:<10} {len(res):>5d} {nearest_rank(dist, 0.5):>11.1f} {nearest_rank(dist, 0.9):>9.1f} {sum(r['to_the_end'] for r in res):>24d}")
    return lines, out


# ---------------------------------------------------------------------------


def main() -> None:
    parser = make_parser(__doc__, jobs=True, cache_dir=True, out_dir=RESULTS_DIR / "channel_check", overrides=True)
    args = parser.parse_args()
    config = load_cli_config(args)
    ccfg = config["channel_check"]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = []

    def emit(title: str, lines: list) -> None:
        text = [f"=== {title} ===", *lines, ""]
        print("\n".join(text))
        report.extend(text)

    ch = config["channel"]
    emit("Channel under test", [
        f"tx_power {ch['tx_power']:g} dBm, antenna gains {ch['antenna_gain_tx']:g} / {ch['antenna_gain_rx']:g} dB, exponent {ch['path_loss']['exponent']:g},"
        f" sensitivity {config['packets']['reception']['sensitivity_dbm']:g} dBm, body {ch['body']['own']['max_loss']:g} dB (lobe {ch['body']['own']['lobe_exponent']:g}),"
        f" others {ch['body']['others']['loss']:g} dB, shadowing sigma {ch['shadowing']['sigma']:g} dB, device offset sigma {ch['device_offset']['sigma']:g} dB"
    ])  # fmt: skip

    runs = [r for r in experiment_runs(config) if r["scenario"] == "none"][: ccfg["n_runs"]]
    parts = parallel_map(link_samples, [r["config"] for r in runs], args.jobs)
    pooled = pool(parts)
    lines, passes, normal = table_normal_run(pooled, ccfg)
    emit(f"3a. Normal run against the run of 13 October ({len(runs)} runs, {pooled['link_hours']:.1f} link-hours)", lines)

    lines, test_a = table_test_a(config)
    emit("3b. Test A: static nodes at %g m, loss with respect to the supports" % ccfg["test_a"]["distance"], lines)
    emit("3c. Range", table_range(config))
    lines, separated = table_separated(config, Path(args.cache_dir), args.jobs)
    emit(f"3c. Distance from the group at the last beacon received from the separated node (first {ccfg['separated_runs']} runs)", lines)

    verdict = "3a RESPECTS the tolerances" if passes else "3a does NOT respect the tolerances"
    print(verdict)
    report.append(verdict)
    (out_dir / "channel_check.md").write_text("\n".join(["# Channel check (generated by `scripts/check_channel_heltec.py`)", "", "```", *report, "```", ""]))
    save_pickle(out_dir / "channel_check.pkl", {"config": config, "normal": normal, "test_a": test_a, "separated": separated, "passes": passes})
    print(f"Results saved to {out_dir}")
    sys.exit(0 if passes else 1)


if __name__ == "__main__":
    main()

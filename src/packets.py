"""Block 3 — Packets and beacons.

Turns the "perfect" RSSI of Block 2 into the real log of each board: only the
received beacons, with their RSSI. Indices ``[i, ..., j]`` = ``i`` transmits, ``j`` receives.

1. ESP-NOW broadcast beacons (1 Mbit/s, no acknowledgment) every ``beacon_period`` s,
   uniform initial phase and uniform jitter on each interval.
2. Collisions NOT modeled: CSMA/CA, ~1 ms beacons, channel occupancy
   ``n_nodes * 1 ms / beacon_period`` (~1 % with 5 nodes, period 0.5 s), negligible
   compared to the background loss.
3. A beacon arrives if it passes two independent trials: signal (logistic in dB on the true
   RSSI, see `logistic_parameters`) and background loss (surrounding Wi-Fi).
4. If it arrives the board reads `rssi_measured` of Block 2, otherwise NaN.
5. Each beacon carries the transmitter's neighbour table (last RSSI and age). One hop only.
6. Knowledge of each observer `m` on the simulator grid.

Parameters: ``packets`` section of the config.

Block 6: bursty background losses (Gilbert-Elliott model per receiver or per link), option ``background_model``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
from scipy.special import expit

from src.common.arrays import run_bounds
from src.common.enums import BackgroundModel, BurstScope, ReceptionModel
from src.common.rng import spawn_rng
from src.channel import ChannelResult
from src.mobility import MobilityResult, load_config

SEED_KEY_TIMES = 4
SEED_KEY_RECEPTION = 5
SEED_KEY_BURST = 10


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PacketResult:
    """Output of Block 3. ``(N, K, N)`` indexed ``[i, k, j]`` = beacon `k` of `i` seen by `j`;
    ``(T, M, N, N)`` indexed ``[t, m, i, j]`` = what observer `m` knows about link `i → j`.
    """

    beacon_times: np.ndarray  # (N, K) s; NaN where a node has fewer beacons
    beacon_steps: np.ndarray  # (N, K) int, grid index; -1 as padding
    p_success: np.ndarray  # (N, K, N) reception probability (curve × background)
    passed_signal: np.ndarray  # (N, K, N) bool, signal trial passed
    passed_background: np.ndarray  # (N, K, N) bool, background trial passed
    background_state: np.ndarray | None  # Block 6: (T, N) or (T, N, N) bool, disturbed state (None with ``independent``)
    received: np.ndarray  # (N, K, N) bool, diagonale False
    rssi: np.ndarray  # (N, K, N) measured dBm, NaN if lost
    knowledge_rssi: np.ndarray  # (T, M, N, N) float32, what m knows about link i→j
    knowledge_age: np.ndarray  # (T, M, N, N) float32 s, age of that information
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reception curve
# ---------------------------------------------------------------------------


def logistic_parameters(sensitivity: float, success: float, width: float) -> tuple:
    """Center `c` and slope `s` (dB) of the logistic ``p(x) = 1 / (1 + exp(-(x - c) / s))``: `width` dB between 10% and 90% (``s = width / (2 ln 9)``)
    and ``p(S) = success`` at sensitivity `S` (``c = S - s ln(success / (1 - success))``). The center (50%) lies below `S`: the datasheet sensitivity is at the level with 8% loss.
    """
    scale = width / (2.0 * math.log(9.0))
    center = sensitivity - scale * math.log(success / (1.0 - success))
    return center, scale


def reception_probability(rssi: np.ndarray, reception_cfg: dict) -> np.ndarray:
    """Probability of passing the signal trial from the true RSSI: logistic (``model: curve``) or hard threshold at the sensitivity (``threshold``)."""
    model = reception_cfg["model"]
    x = np.asarray(rssi, dtype=float)
    if model == ReceptionModel.CURVE:
        center, scale = logistic_parameters(
            reception_cfg["sensitivity_dbm"],
            reception_cfg["sensitivity_success"],
            reception_cfg["transition_width"],
        )
        return expit((x - center) / scale)
    if model == ReceptionModel.THRESHOLD:
        return (x >= reception_cfg["sensitivity_dbm"]).astype(float)
    raise ValueError(f"packets.reception.model must be 'curve' or 'threshold', not {model!r}")


# ---------------------------------------------------------------------------
# Bursty losses (Block 6)
# ---------------------------------------------------------------------------


def gilbert_elliott_states(n_steps: int, shape: tuple, dt: float, bursty_cfg: dict, rng: np.random.Generator) -> np.ndarray:
    """Two-state (good / disturbed) Gilbert-Elliott chains on the `dt` grid.

    Input: `n_steps` steps, `shape` = shape of the independent chains (``(N,)`` per receiver, ``(N, N)`` per link), `bursty_cfg`
    (``mean_loss`` L, ``mean_burst`` T_b), generator `rng`.
    Procedure: mean duration of the good period ``T_g = T_b (1 - L) / L``; per step good → disturbed with probability
    ``1 - exp(-dt / T_g)``, disturbed → good ``1 - exp(-dt / T_b)``. Initial state from the stationary distribution (disturbed with
    probability L). Always ``n_steps + 1`` uniforms are drawn per chain (the first for the initial state).
    Output: bool ``(n_steps, *shape)``, True = disturbed.
    """
    loss, burst = bursty_cfg["mean_loss"], bursty_cfg["mean_burst"]
    if not 0.0 < loss < 1.0 or burst <= 0.0:
        raise ValueError("packets.reception.bursty: mean_loss in (0, 1) and mean_burst > 0")
    good = burst * (1.0 - loss) / loss
    p_go_bad = -math.expm1(-dt / good)
    p_go_good = -math.expm1(-dt / burst)
    u = rng.random((n_steps + 1,) + tuple(shape))
    state = np.empty((n_steps,) + tuple(shape), dtype=bool)
    cur = u[0] < loss
    for k in range(n_steps):
        state[k] = cur
        cur = np.where(cur, u[k + 1] >= p_go_good, u[k + 1] < p_go_bad)
    return state


def burst_lengths(state: np.ndarray, dt: float) -> np.ndarray:
    """Durations (s) of the complete bursts of `state` (time on the first axis): consecutive runs of True between two False."""
    flat = state.reshape(state.shape[0], -1)
    out = []
    for column in flat.T:
        starts, stops = run_bounds(column)
        for a, b in zip(starts, stops):
            if a > 0 and b < len(column):
                out.append((b - a) * dt)
    return np.asarray(out)


# ---------------------------------------------------------------------------
# Beacon instants
# ---------------------------------------------------------------------------


def beacon_schedule(
    n_nodes: int, t_end: float, dt: float, packets_cfg: dict, rng: np.random.Generator
) -> tuple:
    """Beacon instants: (times (N, K) in s, steps (N, K) int). Uniform initial phase in ``[0, period)``, intervals ``period + U[-jitter, +jitter]``;
    beyond `t_end` NaN. `steps` = ``rint(t / dt)``, -1 where NaN.
    """
    period = packets_cfg["beacon_period"]
    jitter = packets_cfg["jitter"]
    if not 0.0 <= jitter < period:
        raise ValueError("packets.jitter must be in [0, beacon_period)")
    n_beacons = int(math.ceil(t_end / (period - jitter))) + 1
    phase = rng.uniform(0.0, period, size=n_nodes)
    intervals = period + rng.uniform(-jitter, jitter, size=(n_nodes, n_beacons - 1))
    times = phase[:, None] + np.concatenate([np.zeros((n_nodes, 1)), np.cumsum(intervals, axis=1)], axis=1)
    times = np.where(times <= t_end, times, np.nan)
    valid = ~np.isnan(times)
    steps = np.where(valid, np.rint(np.where(valid, times, 0.0) / dt), -1).astype(np.int64)
    return times, steps


# ---------------------------------------------------------------------------
# Node knowledge
# ---------------------------------------------------------------------------


def neighbour_tables(times: np.ndarray, received: np.ndarray, payload: dict) -> tuple:
    """Neighbour table carried by each beacon. ``payload[name][i, k, j]`` = value that beacon `k` of `i` carries to `j`.
    For beacon `k` of `j`, for each `i`: the last beacon of `i` received by `j` before ``times[j, k]``. Returns (tables, table_origin) indexed ``[j, k, i]``;
    `table_origin` = origin instant (age is obtained by subtraction: no common clock). NaN if `j` has not yet received anything from `i`.
    """
    n_nodes, n_beacons = times.shape
    tables = {name: np.full((n_nodes, n_beacons, n_nodes), np.nan) for name in payload}
    table_origin = np.full((n_nodes, n_beacons, n_nodes), np.nan)
    for j in range(n_nodes):
        valid_k = np.flatnonzero(~np.isnan(times[j]))
        for i in range(n_nodes):
            if i == j:
                continue
            got = np.flatnonzero(received[i, :, j])
            if len(got) == 0:
                continue
            pos = np.searchsorted(times[i, got], times[j, valid_k], side="left") - 1
            ok = pos >= 0
            src = got[pos[ok]]
            for name, values in payload.items():
                tables[name][j, valid_k[ok], i] = values[i, src, j]
            table_origin[j, valid_k[ok], i] = times[i, src]
    return tables, table_origin


def forward_link_payload(
    t: np.ndarray, times: np.ndarray, received: np.ndarray, payload: dict
) -> tuple:
    """What each observer `m` knows about link `i → j`, for any payload (``payload[name][i, k, j]``, value carried by beacon `k` of `i`).
    A beacon at instant `t_b` is known at `t` if ``t_b <= t``. For ``j == m``: last beacon of `i` received by `m`; for ``j != m``: row `i` of the neighbour table of the last beacon of `j` (one hop only).
    Returns (fields, age, relay_time): `fields` and `age` (s) are (T, M, N, N) float32 ``[t, m, i, j]``; `relay_time` (float64) = instant of the beacon of `j` that relayed (NaN if direct or absent). NaN also for ``i == j``.
    """
    n_steps = len(t)
    n_nodes = times.shape[0]
    tables, table_origin = neighbour_tables(times, received, payload)
    out = {name: np.full((n_steps, n_nodes, n_nodes, n_nodes), np.nan, dtype=np.float32) for name in payload}
    k_age = np.full((n_steps, n_nodes, n_nodes, n_nodes), np.nan, dtype=np.float32)
    relay = np.full((n_steps, n_nodes, n_nodes, n_nodes), np.nan)

    for m in range(n_nodes):
        for j in range(n_nodes):
            if j == m:
                continue
            got = np.flatnonzero(received[j, :, m])
            if len(got) == 0:
                continue
            pos = np.searchsorted(times[j, got], t, side="right") - 1
            ok = pos >= 0
            k = got[pos[ok]]  # most recent beacon of j known to m, for each valid instant
            origin = table_origin[j, k, :]  # (n_ok, i)
            for name in payload:
                out[name][ok, m, :, j] = tables[name][j, k, :]
            k_age[ok, m, :, j] = t[ok, None] - origin
            relay[ok, m, :, j] = np.where(np.isnan(origin), np.nan, times[j, k][:, None])
        for i in range(n_nodes):
            if i == m:
                continue
            got = np.flatnonzero(received[i, :, m])
            if len(got) == 0:
                continue
            pos = np.searchsorted(times[i, got], t, side="right") - 1
            ok = pos >= 0
            k = got[pos[ok]]
            for name, values in payload.items():
                out[name][ok, m, i, m] = values[i, k, m]
            k_age[ok, m, i, m] = t[ok] - times[i, k]

    idx = np.arange(n_nodes)
    for arr in (*out.values(), k_age, relay):
        arr[:, :, idx, idx] = np.nan
    return out, k_age, relay


def _knowledge(t: np.ndarray, times: np.ndarray, received: np.ndarray, rssi: np.ndarray) -> tuple:
    """Special case of `forward_link_payload` with payload = reported RSSI: returns (rssi, age), (T, M, N, N) float32 ``[t, m, i, j]``."""
    fields, age, _ = forward_link_payload(t, times, received, {"rssi": rssi})
    return fields["rssi"], age


# ---------------------------------------------------------------------------
# Packet simulation
# ---------------------------------------------------------------------------


def _burst_state_shape(bcfg: dict, n_nodes: int) -> tuple:
    scope = bcfg["burst_scope"]
    if scope == BurstScope.RECEIVER:
        return (n_nodes,)
    if scope == BurstScope.LINK:
        return (n_nodes, n_nodes)
    raise ValueError("packets.reception.bursty.burst_scope must be 'receiver' or 'link'")


def _background_trial(
    rc: dict,
    seed: int,
    dt: float,
    n_steps: int,
    steps: np.ndarray,
    valid: np.ndarray,
    off_diag: np.ndarray,
    u_background: np.ndarray,
    tx: np.ndarray,
    rx: np.ndarray,
) -> tuple:
    background_model = rc["background_model"]
    if background_model == BackgroundModel.INDEPENDENT:
        passed_background = off_diag & (u_background >= rc["background_loss"])
        return passed_background, None, rc["background_loss"]
    if background_model == BackgroundModel.BURSTY:
        bcfg = rc["bursty"]
        n_nodes = rx.shape[2]
        rng_burst = spawn_rng(seed, SEED_KEY_BURST)
        shape = _burst_state_shape(bcfg, n_nodes)
        background_state = gilbert_elliott_states(n_steps, shape, dt, bcfg, rng_burst)
        steps_safe = np.where(valid, steps, 0)[:, :, None]
        if bcfg["burst_scope"] == BurstScope.RECEIVER:
            bad = background_state[steps_safe, rx]
        else:
            bad = background_state[steps_safe, tx, rx]
        passed_background = off_diag & ~(bad & (u_background < bcfg["loss_bad"]))
        return passed_background, background_state, bcfg["mean_loss"] * bcfg["loss_bad"]
    raise ValueError(f"packets.reception.background_model must be 'independent' or 'bursty', not {background_model!r}")


def simulate_packets(
    mobility: MobilityResult, channel: ChannelResult, config: Union[str, Path, dict]
) -> PacketResult:
    """Block 3 packet simulation (config: YAML path or dict with ``simulation`` and ``packets``).
    Generators independent of Block 2: ``SeedSequence(seed, spawn_key=(4,))`` for the instants, ``(5,)`` for reception, from which two uniform arrays (N, K, N) are always drawn:
    signal (``u < p_curve(rssi_true)``, true unrounded RSSI) and background (``u >= background_loss``), so varying one does not change the other. The beacon arrives if it passes both.
    Block 6: with ``background_model: bursty`` the background trial is replaced by Gilbert-Elliott chains (`gilbert_elliott_states`, dedicated generator
    ``spawn_key=(10,)``, one per receiver or per link according to ``burst_scope``): in the disturbed state the beacon is lost with probability ``loss_bad`` (using
    the background uniform already drawn). With ``independent`` the behaviour is identical to before, bit for bit.
    """
    cfg = load_config(config)
    pk = cfg["packets"]
    rc = pk["reception"]
    seed = cfg["simulation"]["seed"]
    dt = cfg["simulation"]["dt"]
    t = channel.t
    n_nodes = channel.rssi_true.shape[1]
    t0 = time.perf_counter()

    rng_times = spawn_rng(seed, SEED_KEY_TIMES)
    rng_rx = spawn_rng(seed, SEED_KEY_RECEPTION)

    times, steps = beacon_schedule(n_nodes, float(t[-1]), dt, pk, rng_times)
    n_beacons = times.shape[1]
    valid = steps >= 0

    tx = np.arange(n_nodes)[:, None, None]
    rx = np.arange(n_nodes)[None, None, :]
    step_idx = steps[:, :, None]
    off_diag = (tx != rx) & valid[:, :, None]
    rssi_true = np.where(off_diag, channel.rssi_true[step_idx, tx, rx], 0.0)
    rssi_meas = np.where(off_diag, channel.rssi_measured[step_idx, tx, rx], np.nan)

    p_curve = np.where(off_diag, reception_probability(rssi_true, rc), 0.0)
    u_signal = rng_rx.random((n_nodes, n_beacons, n_nodes))
    u_background = rng_rx.random((n_nodes, n_beacons, n_nodes))
    passed_signal = off_diag & (u_signal < p_curve)
    background_model = rc["background_model"]
    passed_background, background_state, background_loss = _background_trial(
        rc, seed, dt, len(t), steps, valid, off_diag, u_background, tx, rx
    )
    received = passed_signal & passed_background

    rssi = np.where(received, rssi_meas, np.nan)
    p_success = p_curve * (1.0 - background_loss)

    k_rssi, k_age = _knowledge(t, times, received, rssi)

    n_trials = int(off_diag.sum())
    metadata = {
        "seed": seed,
        "config": cfg,
        "logistic_center_scale": logistic_parameters(
            rc["sensitivity_dbm"], rc["sensitivity_success"], rc["transition_width"]
        ),
        "n_beacons_per_node": valid.sum(axis=1),
        "received_fraction": float(received.sum() / n_trials),
        "lost_signal_fraction": float((off_diag & ~passed_signal).sum() / n_trials),
        "background_model": background_model,
        "lost_background_only_fraction": float((passed_signal & ~passed_background).sum() / n_trials),
        "elapsed": time.perf_counter() - t0,
    }

    return PacketResult(
        beacon_times=times,
        beacon_steps=steps,
        p_success=p_success,
        passed_signal=passed_signal,
        passed_background=passed_background,
        background_state=background_state,
        received=received,
        rssi=rssi,
        knowledge_rssi=k_rssi,
        knowledge_age=k_age,
        metadata=metadata,
    )

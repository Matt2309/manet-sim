"""Block 2 — Channel model.

RSSI in dBm for every instant and ordered pair of nodes: ``rssi[t, i, j]`` =
power received by ``j`` when ``i`` transmits, NaN diagonal. Sensitivity,
losses and beacon instants belong to Block 3: the RSSI exists even below threshold.

Formula (positive losses, additive random terms)::

    rssi_true = P_tx + G_tx + G_rx - PL(d)      # attenuation with distance
              + S_ij(t)                         # shadowing (symmetric)
              - B_own_ij(t)                     # torso of transmitter and receiver (symmetric)
              - B_others_ij(t)                  # other runners in between (symmetric)
              + F_ij(t)                         # fast variations (NOT symmetric)
              + tx_offset[i] + rx_offset[j]     # fixed offset of each board
              - obstacles_ij(t)                 # buildings (Block 2b)

    rssi_measured = quantize(min(rssi_true, saturation))

Parameters: ``channel`` section of the config.

References: Agrawal and Patwari, IEEE Trans. Wireless Commun., 2009 (network
shadowing); Wang, Tameh and Nix, IEEE Trans. Veh. Technol., 2008 (per-link
shadowing between mobile terminals).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
from scipy.fft import irfft2, next_fast_len, rfft2
from scipy.ndimage import map_coordinates
from scipy.special import expit

from src.common.arrays import off_diagonal
from src.common.enums import ShadowingMode
from src.mobility import MobilityResult, load_config

_SPEED_OF_LIGHT = 299792458.0  # m/s
_N_RNG_STREAMS = 4  # map, independent shadowing, offsets, fading


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ShadowField:
    """2D shadowing map (``field`` mode)."""

    values: np.ndarray  # (nx, ny) float32, index [ix, iy]
    origin: tuple  # (x0, y0) in meters: position of cell [0, 0]
    resolution: float  # m; grid step


@dataclass
class ChannelResult:
    """Output of Block 2. Matrices (T, N, N) indexed ``[t, i, j]`` (``i`` transmits, ``j`` receives), NaN diagonal."""

    t: np.ndarray  # (T,)
    rssi_true: np.ndarray  # (T, N, N) dBm, before saturation and quantization
    rssi_measured: np.ndarray  # (T, N, N) dBm, after saturation and quantization
    rssi_path_loss: np.ndarray  # (T, N, N) P_tx + G_tx + G_rx - PL(d)
    shadowing: np.ndarray  # (T, N, N) dB, additive contribution
    body_own: np.ndarray  # (T, N, N) dB, positive loss
    body_others: np.ndarray  # (T, N, N) dB, positive loss
    fading: np.ndarray  # (T, N, N) dB, additive contribution
    obstacles: np.ndarray  # (T, N, N) dB, positive loss (Block 2b)
    tx_offset: np.ndarray  # (N,) dB
    rx_offset: np.ndarray  # (N,) dB
    shadow_field: Optional[ShadowField]
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Attenuation with distance and reading
# ---------------------------------------------------------------------------


def reference_path_loss(frequency: float, reference_distance: float) -> float:
    """Free-space attenuation at `d0`: ``20 log10(4 pi d0 f / c)`` in dB (about 40.2 dB at 2.437 GHz with d0 = 1 m)."""
    return 20.0 * math.log10(4.0 * math.pi * reference_distance * frequency / _SPEED_OF_LIGHT)


def path_loss_db(distance: np.ndarray, channel_cfg: dict) -> np.ndarray:
    """Log-distance model ``PL0 + 10 n log10(max(d, d0) / d0)``: positive loss in dB, shape of `distance`; below `d0` it equals PL(d0)."""
    pl_cfg = channel_cfg["path_loss"]
    d0 = pl_cfg["reference_distance"]
    pl0 = reference_path_loss(channel_cfg["frequency"], d0)
    d = np.maximum(np.asarray(distance, dtype=float), d0)
    return pl0 + 10.0 * pl_cfg["exponent"] * np.log10(d / d0)


def _link_budget_db(channel_cfg: dict) -> float:
    return channel_cfg["tx_power"] + channel_cfg["antenna_gain_tx"] + channel_cfg["antenna_gain_rx"]


def distance_from_path_loss(level_dbm: float, channel_cfg: dict) -> float:
    """Distance (m) at which path loss alone gives `level_dbm`; analytic inversion of `path_loss_db`."""
    pl_cfg = channel_cfg["path_loss"]
    pl = _link_budget_db(channel_cfg) - level_dbm
    pl0 = reference_path_loss(channel_cfg["frequency"], pl_cfg["reference_distance"])
    return float(pl_cfg["reference_distance"] * 10.0 ** ((pl - pl0) / (10.0 * pl_cfg["exponent"])))


def measure_rssi(rssi_true: np.ndarray, measurement_cfg: dict) -> np.ndarray:
    """Saturation and then quantization ``round(x / step) * step``; NaNs stay NaN."""
    step = measurement_cfg["quantization_step"]
    saturated = np.minimum(rssi_true, measurement_cfg["saturation_dbm"])
    return np.round(saturated / step) * step


# ---------------------------------------------------------------------------
# Shadowing: "field" mode (network shadowing)
# ---------------------------------------------------------------------------


def shadow_field_sigma_p(sigma: float, delta: float) -> float:
    """``sigma_p = sigma / sqrt(2 delta)`` in dB/sqrt(m): the link variance tends to `sigma^2` for ``d >> delta``."""
    return sigma / math.sqrt(2.0 * delta)


def link_variance_theory(distance: np.ndarray, sigma: float, delta: float) -> np.ndarray:
    """Theoretical variance (dB^2) of the shadowing of a link of length `d`: ``sigma^2 [1 - (delta/d)(1 - exp(-d/delta))]``."""
    d = np.asarray(distance, dtype=float)
    return sigma**2 * (1.0 - (delta / d) * (1.0 - np.exp(-d / delta)))


def generate_shadow_field(
    x_range: tuple,
    y_range: tuple,
    sigma_p: float,
    delta: float,
    resolution: float,
    padding: float,
    rng: np.random.Generator,
) -> ShadowField:
    """2D Gaussian shadowing map, zero mean, isotropic, autocorrelation ``exp(-r/delta)``, variance ``sigma_p^2``.
    Generated spectrally (spectrum ``(1 + (2 pi k delta)^2)^(-3/2)``) on an FFT grid enlarged by `padding` m per side, then cropped. float32.
    """
    nx = int(math.ceil((x_range[1] - x_range[0]) / resolution)) + 1
    ny = int(math.ceil((y_range[1] - y_range[0]) / resolution)) + 1
    pad = int(math.ceil(padding / resolution))
    fx = next_fast_len(nx + pad)
    fy = next_fast_len(ny + pad)

    kx = np.fft.fftfreq(fx, d=resolution)[:, None]
    ky_full = np.fft.fftfreq(fy, d=resolution)[None, :]
    ky_half = np.fft.rfftfreq(fy, d=resolution)[None, :]

    def spectrum(ky: np.ndarray) -> np.ndarray:
        return (1.0 + (2.0 * math.pi * delta) ** 2 * (kx**2 + ky**2)) ** (-1.5)

    total = spectrum(ky_full).sum()
    gain = sigma_p * math.sqrt(fx * fy / total)
    filt = gain * np.sqrt(spectrum(ky_half))

    white = rng.standard_normal((fx, fy))
    values = irfft2(rfft2(white) * filt, s=(fx, fy))
    return ShadowField(
        values=np.ascontiguousarray(values[:nx, :ny], dtype=np.float32),
        origin=(float(x_range[0]), float(y_range[0])),
        resolution=float(resolution),
    )


def link_shadowing(
    shadow_field: ShadowField, pos_a: np.ndarray, pos_b: np.ndarray, shadow_cfg: dict
) -> np.ndarray:
    """Shadowing of L links as an integral of the map: ``S = (1/sqrt(d)) * integral_0^d p(x(u)) du`` (trapezoids, bilinear interpolation).
    `pos_a`, `pos_b` (L, 2); returns (L,) in dB, symmetric. Wider step on long links; processed in blocks of whole links.
    """
    delta = shadow_cfg["correlation_distance"]
    step_short = shadow_cfg["integration_step"]
    step_long = shadow_cfg["long_link_step_factor"] * delta
    long_threshold = shadow_cfg["long_link_factor"] * delta

    a = np.asarray(pos_a, dtype=float)
    b = np.asarray(pos_b, dtype=float)
    vec = b - a
    d = np.hypot(vec[:, 0], vec[:, 1])
    h = np.where(d > long_threshold, step_long, step_short)
    n_pts = np.maximum(np.ceil(d / h).astype(np.int64) + 1, 2)

    out = np.zeros(len(d))
    cum = np.cumsum(n_pts)
    chunk_points = int(shadow_cfg["chunk_points"])
    ox, oy = shadow_field.origin
    res = shadow_field.resolution

    start = 0
    n_links = len(d)
    while start < n_links:
        base = cum[start - 1] if start > 0 else 0
        end = int(np.searchsorted(cum, base + chunk_points, side="right"))
        end = min(max(end, start + 1), n_links)

        n_c = n_pts[start:end]
        first = np.concatenate([[0], np.cumsum(n_c)[:-1]])
        rep = np.repeat(np.arange(end - start), n_c)
        k = np.arange(int(n_c.sum())) - first[rep]
        u = k / (n_c[rep] - 1)

        px = a[start:end, 0][rep] + u * vec[start:end, 0][rep]
        py = a[start:end, 1][rep] + u * vec[start:end, 1][rep]
        coords = np.stack([(px - ox) / res, (py - oy) / res])
        vals = map_coordinates(shadow_field.values, coords, order=1, mode="nearest", output=np.float64)

        # trapezoidal weights: effective step d/(n-1), half weight at the ends
        spacing = d[start:end] / (n_c - 1)
        weights = spacing[rep]
        weights = np.where((k == 0) | (k == n_c[rep] - 1), 0.5 * weights, weights)
        integral = np.add.reduceat(vals * weights, first)

        d_c = d[start:end]
        out[start:end] = np.where(d_c > 0.0, integral / np.sqrt(np.where(d_c > 0.0, d_c, 1.0)), 0.0)
        start = end
    return out


def field_shadowing(
    positions: np.ndarray, shadow_field: ShadowField, shadow_cfg: dict
) -> np.ndarray:
    """Shadowing (T, N, N) in dB, field mode, from `positions` (T, N, 2): computed for ``i < j`` and mirrored, zero diagonal."""
    n_steps, n_nodes, _ = positions.shape
    iu, ju = np.triu_indices(n_nodes, k=1)
    a = positions[:, iu, :].reshape(-1, 2)
    b = positions[:, ju, :].reshape(-1, 2)
    s = link_shadowing(shadow_field, a, b, shadow_cfg).reshape(n_steps, len(iu))
    out = np.zeros((n_steps, n_nodes, n_nodes))
    out[:, iu, ju] = s
    out[:, ju, iu] = s
    return out


# ---------------------------------------------------------------------------
# Shadowing: "independent" mode
# ---------------------------------------------------------------------------


def independent_shadowing(
    positions: np.ndarray, sigma: float, delta: float, rng: np.random.Generator
) -> np.ndarray:
    """Shadowing (T, N, N) in dB, independent mode: per pair, Gauss-Markov advancing with the distance travelled by the two endpoints
    (``Delta_k = |dp_i| + |dp_j|``, ``a_k = exp(-Delta_k / delta)``; Wang, Tameh and Nix). Stationary initial state, still nodes = constant. Symmetric, zero diagonal.
    """
    n_steps, n_nodes, _ = positions.shape
    iu, ju = np.triu_indices(n_nodes, k=1)
    n_pairs = len(iu)

    moved = np.zeros((n_steps, n_nodes))
    moved[1:] = np.linalg.norm(np.diff(positions, axis=0), axis=-1)
    delta_k = moved[:, iu] + moved[:, ju]  # (T, P)
    a = np.exp(-delta_k / delta)
    noise_std = sigma * np.sqrt(np.maximum(1.0 - a**2, 0.0))

    s = np.empty((n_steps, n_pairs))
    s[0] = rng.normal(0.0, sigma, size=n_pairs)
    w = rng.standard_normal((n_steps, n_pairs))
    for k in range(1, n_steps):
        s[k] = a[k] * s[k - 1] + noise_std[k] * w[k]

    out = np.zeros((n_steps, n_nodes, n_nodes))
    out[:, iu, ju] = s
    out[:, ju, iu] = s
    return out


# ---------------------------------------------------------------------------
# Bodies
# ---------------------------------------------------------------------------


def own_body_loss(positions: np.ndarray, headings: np.ndarray, own_cfg: dict) -> np.ndarray:
    """Loss (T, N, N) in dB of the torso of the transmitter and of the receiver: ``B_ij = L_i(phi_ij) + L_j(phi_ji)``.
    `phi` = angle of ``p_j - p_i`` relative to ``headings[i]`` (counterclockwise); torso at ``phi_b = +90°`` with ``mount_side: right``, ``-90°`` with ``left``.
    ``L(phi) = max_loss * ((1 + cos(phi - phi_b)) / 2) ** lobe_exponent``.
    """
    side = own_cfg["mount_side"]
    if side not in ("right", "left"):
        raise ValueError(f"channel.body.own.mount_side must be 'right' or 'left', not {side!r}")
    phi_b = math.pi / 2.0 if side == "right" else -math.pi / 2.0

    v = positions[:, None, :, :] - positions[:, :, None, :]  # (T, i, j, 2): p_j - p_i
    h = headings[:, :, None, :]
    cross = h[..., 0] * v[..., 1] - h[..., 1] * v[..., 0]
    dot = h[..., 0] * v[..., 0] + h[..., 1] * v[..., 1]
    phi = np.arctan2(cross, dot)
    lobe = ((1.0 + np.cos(phi - phi_b)) / 2.0) ** own_cfg["lobe_exponent"]
    loss = own_cfg["max_loss"] * lobe  # loss[t, i, j] = L_i(phi_ij)
    return loss + np.transpose(loss, (0, 2, 1))


def other_bodies_loss(positions: np.ndarray, others_cfg: dict, chunk_steps: int = 2000) -> np.ndarray:
    """Loss (T, N, N) in dB of the other runners (detached one included) on link `i–j`: for each `k` with projection ``0 < u < 1`` onto the segment
    and distance `c`, ``loss / (1 + exp((c - radius) / transition_width))``; sum over `k` capped at `max_total_loss`. Computed for ``i < j`` and mirrored.
    """
    n_steps, n_nodes, _ = positions.shape
    out = np.zeros((n_steps, n_nodes, n_nodes))
    idx = np.arange(n_nodes)
    not_endpoint = (
        (idx[None, None, :] != idx[:, None, None]) & (idx[None, None, :] != idx[None, :, None])
    )  # (i, j, k): k different from i and j

    for lo in range(0, n_steps, chunk_steps):
        p = positions[lo : lo + chunk_steps]
        ab = p[:, None, :, :] - p[:, :, None, :]  # (t, i, j, 2)
        ak = p[:, None, None, :, :] - p[:, :, None, None, :]  # (t, i, j, k, 2)
        ab_sq = np.sum(ab**2, axis=-1)
        ab_sq_safe = np.where(ab_sq > 0.0, ab_sq, 1.0)
        u = np.sum(ak * ab[:, :, :, None, :], axis=-1) / ab_sq_safe[..., None]
        closest = ak - u[..., None] * ab[:, :, :, None, :]
        c = np.linalg.norm(closest, axis=-1)
        contribution = others_cfg["loss"] * expit(-(c - others_cfg["radius"]) / others_cfg["transition_width"])
        between = (u > 0.0) & (u < 1.0) & not_endpoint[None]
        total = np.minimum(np.sum(np.where(between, contribution, 0.0), axis=-1), others_cfg["max_total_loss"])
        out[lo : lo + chunk_steps] = total

    upper = np.triu(np.ones((n_nodes, n_nodes), dtype=bool), k=1)[None]
    out = np.where(upper, out, np.transpose(out, (0, 2, 1)))
    out[:, idx, idx] = 0.0
    return out


# ---------------------------------------------------------------------------
# Fast variations and board offsets
# ---------------------------------------------------------------------------


def rician_fading_db(shape: tuple, k_db: float, rng: np.random.Generator) -> np.ndarray:
    """Rician fading with unit mean power: ``F = 20 log10 |h|``, ``h = sqrt(K/(K+1)) + sqrt(1/(K+1)) z``, `z` unit complex Gaussian.
    Independent draws per element: between two steps (0.1 s) the node moves ~0.28 m >> lambda/2 (~6 cm), and the two directions are measured at different instants.
    """
    k = 10.0 ** (k_db / 10.0)
    los = math.sqrt(k / (k + 1.0))
    scatter = math.sqrt(1.0 / (k + 1.0))
    re = rng.standard_normal(shape)
    im = rng.standard_normal(shape)
    h = los + scatter * (re + 1j * im) / math.sqrt(2.0)
    return 20.0 * np.log10(np.abs(h))


def device_offsets(n_nodes: int, sigma: float, rng: np.random.Generator) -> tuple:
    """Fixed offsets of each board, ``N(0, sigma)`` in dB, drawn once: (tx_offset, rx_offset), each (N,)."""
    tx = rng.normal(0.0, sigma, size=n_nodes)
    rx = rng.normal(0.0, sigma, size=n_nodes)
    return tx, rx


def _obstacle_loss(mobility: MobilityResult, obstacles_cfg: dict) -> np.ndarray:
    """Hook for Block 2b (OSM buildings): not yet implemented."""
    raise NotImplementedError("channel.obstacles.enabled: true requires Block 2b (not yet implemented).")


# ---------------------------------------------------------------------------
# Channel simulation
# ---------------------------------------------------------------------------


def _set_diagonal_nan(*arrays: np.ndarray) -> None:
    n_nodes = arrays[0].shape[1]
    idx = np.arange(n_nodes)
    for arr in arrays:
        arr[:, idx, idx] = np.nan


def _shadowing_mode(value: str) -> ShadowingMode:
    try:
        return ShadowingMode(value)
    except ValueError:
        raise ValueError(f"channel.shadowing.mode must be 'field' or 'independent', not {value!r}") from None


def _simulate_shadowing(
    positions: np.ndarray,
    sh: dict,
    rng_field: np.random.Generator,
    rng_indep: np.random.Generator,
    timings: dict,
) -> tuple[np.ndarray, Optional[ShadowField], Optional[float]]:
    t0 = time.perf_counter()
    n_steps, n_nodes, _ = positions.shape
    shadow_field: Optional[ShadowField] = None
    sigma_p = None
    shadowing = np.zeros((n_steps, n_nodes, n_nodes))
    if sh["enabled"]:
        delta = sh["correlation_distance"]
        mode = _shadowing_mode(sh["mode"])
        if mode == ShadowingMode.FIELD:
            margin = sh["field_margin"]
            x_range = (positions[..., 0].min() - margin, positions[..., 0].max() + margin)
            y_range = (positions[..., 1].min() - margin, positions[..., 1].max() + margin)
            sigma_p = shadow_field_sigma_p(sh["sigma"], delta)
            shadow_field = generate_shadow_field(
                x_range, y_range, sigma_p, delta, sh["grid_resolution"], sh["fft_padding"], rng_field
            )
            timings["shadow_field"] = time.perf_counter() - t0
            t0 = time.perf_counter()
            shadowing = field_shadowing(positions, shadow_field, sh)
        else:
            shadowing = independent_shadowing(positions, sh["sigma"], delta, rng_indep)
    timings["shadowing"] = time.perf_counter() - t0
    return shadowing, shadow_field, sigma_p


def _channel_metadata(
    cfg: dict,
    arrays: dict,
    rssi_true: np.ndarray,
    rssi_measured: np.ndarray,
    pl0: float,
    sigma_p: Optional[float],
    timings: dict,
) -> dict:
    ch = cfg["channel"]
    sh = ch["shadowing"]
    n_nodes = rssi_true.shape[1]
    off_diag = off_diagonal(n_nodes)
    true_od = rssi_true[:, off_diag]
    meas_od = rssi_measured[:, off_diag]
    saturation = ch["measurement"]["saturation_dbm"]

    def _std(arr: np.ndarray) -> float:
        return float(np.std(arr[:, off_diag]))

    return {
        "seed": cfg["simulation"]["seed"],
        "config": cfg,
        "pl0": pl0,
        "sigma_p": sigma_p,
        "shadowing_mode": sh["mode"] if sh["enabled"] else None,
        "saturated_fraction": float(np.mean(true_od > saturation)),
        "rssi_measured_median": float(np.median(meas_od)),
        "rssi_measured_p5": float(np.percentile(meas_od, 5)),
        "rssi_measured_p95": float(np.percentile(meas_od, 95)),
        "component_std": {
            "path_loss": _std(arrays["path_loss"]),
            "shadowing": _std(arrays["shadowing"]),
            "body_own": _std(arrays["body_own"]),
            "body_others": _std(arrays["body_others"]),
            "fading": _std(arrays["fading"]),
        },
        "timings": timings,
    }


def simulate_channel(mobility: MobilityResult, config: Union[str, Path, dict]) -> ChannelResult:
    """Block 2 channel simulation (config: YAML path or dict with ``simulation`` and ``channel``).
    Four independent generators from ``SeedSequence(simulation.seed)`` in fixed order (map, independent shadowing, offsets, fading): enabling a component does not change the others. Disabled components = 0.
    """
    cfg = load_config(config)
    ch = cfg["channel"]
    seed = cfg["simulation"]["seed"]
    positions = mobility.positions
    n_steps, n_nodes, _ = positions.shape
    timings: dict = {}

    ss = np.random.SeedSequence(seed)
    rng_field, rng_indep, rng_device, rng_fading = (np.random.default_rng(c) for c in ss.spawn(_N_RNG_STREAMS))

    # attenuation with distance
    t0 = time.perf_counter()
    pl0 = reference_path_loss(ch["frequency"], ch["path_loss"]["reference_distance"])
    rssi_path_loss = _link_budget_db(ch) - path_loss_db(mobility.distances, ch)
    timings["path_loss"] = time.perf_counter() - t0

    # shadowing
    shadowing, shadow_field, sigma_p = _simulate_shadowing(
        positions, ch["shadowing"], rng_field, rng_indep, timings
    )

    # bodies
    t0 = time.perf_counter()
    body_own = np.zeros((n_steps, n_nodes, n_nodes))
    if ch["body"]["own"]["enabled"]:
        body_own = own_body_loss(positions, mobility.headings, ch["body"]["own"])
    timings["body_own"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    body_others = np.zeros((n_steps, n_nodes, n_nodes))
    if ch["body"]["others"]["enabled"]:
        body_others = other_bodies_loss(positions, ch["body"]["others"])
    timings["body_others"] = time.perf_counter() - t0

    # fast variations
    t0 = time.perf_counter()
    fading = np.zeros((n_steps, n_nodes, n_nodes))
    if ch["fading"]["enabled"]:
        fading = rician_fading_db((n_steps, n_nodes, n_nodes), ch["fading"]["rician_k_db"], rng_fading)
    timings["fading"] = time.perf_counter() - t0

    # board offsets
    tx_offset = np.zeros(n_nodes)
    rx_offset = np.zeros(n_nodes)
    if ch["device_offset"]["enabled"]:
        tx_offset, rx_offset = device_offsets(n_nodes, ch["device_offset"]["sigma"], rng_device)

    # obstacles (Block 2b)
    if ch["obstacles"]["enabled"]:
        obstacles = _obstacle_loss(mobility, ch["obstacles"])
    else:
        obstacles = np.zeros((n_steps, n_nodes, n_nodes))

    rssi_true = (
        rssi_path_loss
        + shadowing
        - body_own
        - body_others
        + fading
        + tx_offset[None, :, None]
        + rx_offset[None, None, :]
        - obstacles
    )
    rssi_measured = measure_rssi(rssi_true, ch["measurement"])

    arrays = [rssi_true, rssi_measured, rssi_path_loss, shadowing, body_own, body_others, fading, obstacles]
    _set_diagonal_nan(*arrays)

    metadata = _channel_metadata(
        cfg,
        {
            "path_loss": rssi_path_loss,
            "shadowing": shadowing,
            "body_own": body_own,
            "body_others": body_others,
            "fading": fading,
        },
        rssi_true,
        rssi_measured,
        pl0,
        sigma_p,
        timings,
    )

    return ChannelResult(
        t=mobility.t,
        rssi_true=rssi_true,
        rssi_measured=rssi_measured,
        rssi_path_loss=rssi_path_loss,
        shadowing=shadowing,
        body_own=body_own,
        body_others=body_others,
        fading=fading,
        obstacles=obstacles,
        tx_offset=tx_offset,
        rx_offset=rx_offset,
        shadow_field=shadow_field,
        metadata=metadata,
    )

"""Block 1 — Mobility.

Trajectory of a group of runners along a real GPX route: position of
every node at every instant. Parameters: YAML file (``config/default.yaml``).
"""

from __future__ import annotations

import logging
import math
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Union

import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.ndimage import uniform_filter1d
from scipy.optimize import brentq
from scipy.stats import norm

from src.common.arrays import off_diagonal
from src.common.config import load_config  # noqa: F401

logger = logging.getLogger(__name__)

_GPX_NAMESPACE = {"gpx": "http://www.topografix.com/GPX/1/1"}
_EARTH_RADIUS_M = 6371000.0


# ---------------------------------------------------------------------------
# Loading and cleaning of the GPX track
# ---------------------------------------------------------------------------


@dataclass
class PathSample:
    """Sample of the path queried at a given curvilinear abscissa."""

    position: np.ndarray  # (..., 2) in meters
    tangent: np.ndarray  # (..., 2) unit vector, direction of travel
    normal: np.ndarray  # (..., 2) unit vector, perpendicular to the left
    speed: np.ndarray  # (...,) m/s, from the GPX profile


@dataclass
class Track:
    """GPX track projected onto a local plane in meters, cleaned of
    GPS sampling gaps and equipped with the speed profile.
    """

    x: np.ndarray
    y: np.ndarray
    s: np.ndarray  # curvilinear abscissa, same length as x/y
    ele: np.ndarray
    speed: np.ndarray  # smoothed speed profile, per point
    speed_raw: np.ndarray  # raw speed profile, per point
    length: float
    n_glitches_corrected: int
    n_points_raw: int
    n_points: int
    sample_rate: float
    heading_s: np.ndarray  # segment midpoints, curvilinear abscissa
    heading: np.ndarray  # smoothed heading angle, unwrapped, in radians

    def query(self, s: Union[float, np.ndarray]) -> PathSample:
        """Position, unit vectors and speed at curvilinear abscissae `s` (m, any shape); outside [0, length] clamp with warning.
        Tangent and normal (to the left) come from the smoothed angle (`_smooth_heading`), continuous at the vertices too.
        """
        s_arr = np.asarray(s, dtype=float)
        scalar_input = s_arr.ndim == 0
        s_flat = np.atleast_1d(s_arr).reshape(-1)

        clipped = np.clip(s_flat, 0.0, self.length)
        if not np.array_equal(clipped, s_flat):
            warnings.warn(
                "Curvilinear abscissa outside the path: values clamped to "
                f"[0, {self.length:.1f}] m."
            )

        # index of the segment [s[idx], s[idx+1]] containing each point
        idx = np.searchsorted(self.s, clipped, side="right") - 1
        idx = np.clip(idx, 0, len(self.s) - 2)

        s0, s1 = self.s[idx], self.s[idx + 1]
        seg_len = s1 - s0
        seg_len_safe = np.where(seg_len > 0, seg_len, 1.0)
        frac = np.where(seg_len > 0, (clipped - s0) / seg_len_safe, 0.0)

        x0, x1 = self.x[idx], self.x[idx + 1]
        y0, y1 = self.y[idx], self.y[idx + 1]
        pos_x = x0 + frac * (x1 - x0)
        pos_y = y0 + frac * (y1 - y0)

        theta = np.interp(clipped, self.heading_s, self.heading)
        tx, ty = np.cos(theta), np.sin(theta)
        nx, ny = -ty, tx  # rotation by +90°: normal pointing left

        speed = np.interp(clipped, self.s, self.speed)

        position = np.stack([pos_x, pos_y], axis=-1).reshape(s_arr.shape + (2,))
        tangent = np.stack([tx, ty], axis=-1).reshape(s_arr.shape + (2,))
        normal = np.stack([nx, ny], axis=-1).reshape(s_arr.shape + (2,))
        speed = speed.reshape(s_arr.shape)

        if scalar_input:
            return PathSample(position[()], tangent[()], normal[()], speed[()])
        return PathSample(position, tangent, normal, speed)


def _smooth_heading(
    x: np.ndarray, y: np.ndarray, s: np.ndarray, width: float
) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed heading angle (moving average of width `width` m in `s`): returns segment midpoints and the unwrapped angle there.
    Exact average with unequal segments: Θ(s) is piecewise linear and θ̄ = (Θ(s+w/2) − Θ(s−w/2)) / w; constant angle beyond the ends; null segments ignored.
    """
    dx, dy, ds = np.diff(x), np.diff(y), np.diff(s)
    valid = ds > 0
    theta = np.unwrap(np.arctan2(dy[valid], dx[valid]))
    edges = np.concatenate([[s[0]], s[1:][valid]])
    length = np.diff(edges)
    mid = (edges[:-1] + edges[1:]) / 2.0
    cumulative = np.concatenate([[0.0], np.cumsum(theta * length)])

    def integral(position: np.ndarray) -> np.ndarray:
        inside = np.interp(position, edges, cumulative)
        before = theta[0] * (position - edges[0])
        after = cumulative[-1] + theta[-1] * (position - edges[-1])
        return np.where(position < edges[0], before, np.where(position > edges[-1], after, inside))

    smoothed = (integral(mid + width / 2.0) - integral(mid - width / 2.0)) / width
    return mid, smoothed


def _parse_gpx(path: Union[str, Path]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reads lat, lon, ele from the `<trkpt>` of a GPX 1.1 (explicit namespace needed)."""
    root = ET.parse(path).getroot()
    trkpts = root.findall(".//gpx:trkpt", _GPX_NAMESPACE)
    if not trkpts:
        raise ValueError(f"No <trkpt> found in {path}: empty GPX file or unexpected namespace.")
    lat = np.array([float(p.get("lat")) for p in trkpts])
    lon = np.array([float(p.get("lon")) for p in trkpts])
    ele = np.array(
        [float(p.find("gpx:ele", _GPX_NAMESPACE).text) for p in trkpts]
    )
    return lat, lon, ele


def _step_lengths(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.hypot(np.diff(x), np.diff(y))


def _project_to_local_plane(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Equirectangular projection in meters with origin at the first point; error below one meter on routes of a few km."""
    lat0 = math.radians(lat[0])
    lon0 = math.radians(lon[0])
    x = (np.radians(lon) - lon0) * _EARTH_RADIUS_M * math.cos(lat0)
    y = (np.radians(lat) - lat0) * _EARTH_RADIUS_M
    return x, y


def _repair_glitches(
    x: np.ndarray, y: np.ndarray, ele: np.ndarray, glitch_max_step: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Fills steps longer than `glitch_max_step` with interpolated points; returns x, y, ele and the number of corrected steps."""
    d = _step_lengths(x, y)
    if len(d) == 0:
        return x, y, ele, 0

    median_step = float(np.median(d))
    glitch_idx = set(np.where(d > glitch_max_step)[0].tolist())
    if not glitch_idx:
        return x, y, ele, 0

    new_x, new_y, new_ele = [x[0]], [y[0]], [ele[0]]
    for i in range(len(x) - 1):
        if i in glitch_idx:
            k = max(int(round(d[i] / median_step)) - 1, 1)
            for j in range(1, k + 1):
                frac = j / (k + 1)
                new_x.append(x[i] + frac * (x[i + 1] - x[i]))
                new_y.append(y[i] + frac * (y[i + 1] - y[i]))
                new_ele.append(ele[i] + frac * (ele[i + 1] - ele[i]))
        new_x.append(x[i + 1])
        new_y.append(y[i + 1])
        new_ele.append(ele[i + 1])

    return np.array(new_x), np.array(new_y), np.array(new_ele), len(glitch_idx)


def load_track(cfg: dict) -> Track:
    """Loads and cleans the GPX track (`cfg['track']`). The GPX has no timestamps: uniform sampling assumed (`assumed_sample_rate`),
    with a warning if the median speed is outside `plausible_speed_range`.
    """
    track_cfg = cfg["track"]

    lat, lon, ele = _parse_gpx(track_cfg["gpx_file"])
    n_points_raw = len(lat)
    if n_points_raw < 2:
        raise ValueError("The GPX track must contain at least 2 points.")

    x, y = _project_to_local_plane(lat, lon)
    x, y, ele, n_glitches = _repair_glitches(x, y, ele, track_cfg["glitch_max_step"])
    if n_glitches > 0:
        logger.warning(
            "Corrected %d anomalous points (GPS glitches) in track %s",
            n_glitches,
            track_cfg["gpx_file"],
        )

    d = _step_lengths(x, y)
    s = np.concatenate([[0.0], np.cumsum(d)])

    sample_rate = float(track_cfg["assumed_sample_rate"])
    v_seg = d * sample_rate  # instantaneous speed per segment
    speed_raw = np.empty(len(x))
    speed_raw[0] = v_seg[0]
    speed_raw[-1] = v_seg[-1]
    speed_raw[1:-1] = (v_seg[:-1] + v_seg[1:]) / 2.0  # mean of the two adjacent segments

    window = max(int(round(track_cfg["speed_smoothing_window"] * sample_rate)), 1)
    speed = uniform_filter1d(speed_raw, size=window, mode="nearest")

    lo, hi = track_cfg["plausible_speed_range"]
    median_speed = float(np.median(speed_raw))
    if not (lo <= median_speed <= hi):
        warnings.warn(
            f"Implied median speed ({median_speed:.2f} m/s) outside the plausible "
            f"range [{lo}, {hi}] m/s for running: assumed_sample_rate="
            f"{sample_rate} Hz may be wrong for this file."
        )

    heading_s, heading = _smooth_heading(x, y, s, track_cfg["heading_smoothing"])

    return Track(
        heading_s=heading_s,
        heading=heading,
        x=x,
        y=y,
        s=s,
        ele=ele,
        speed=speed,
        speed_raw=speed_raw,
        length=float(s[-1]),
        n_glitches_corrected=n_glitches,
        n_points_raw=n_points_raw,
        n_points=len(x),
        sample_rate=sample_rate,
    )


# ---------------------------------------------------------------------------
# Critically damped second-order process (node offsets)
# ---------------------------------------------------------------------------


def _exact_discretization(dt: float, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact discretization of the critically damped (position, velocity) system: ``A = [[0, 1], [-1/tau^2, -2/tau]]``, noise on the velocity q^2 = 4/tau^3.
    Returns Phi (2, 2) and the Cholesky of Qd (Van Loan) for sigma = 1; for other sigma it scales with sigma.
    """
    a_mat = np.array([[0.0, 1.0], [-1.0 / tau**2, -2.0 / tau]])
    q_mat = np.array([[0.0, 0.0], [0.0, 4.0 / tau**3]])

    van_loan = np.zeros((4, 4))
    van_loan[:2, :2] = -a_mat
    van_loan[:2, 2:] = q_mat
    van_loan[2:, 2:] = a_mat.T
    blocks = expm(van_loan * dt)
    phi = blocks[2:, 2:].T
    qd = phi @ blocks[:2, 2:]
    qd = (qd + qd.T) / 2.0
    return phi, np.linalg.cholesky(qd + 1e-15 * np.eye(2))


def critically_damped_process(
    n_steps: int,
    dt: float,
    tau: float,
    sigma: float,
    rng: np.random.Generator,
    size: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent realizations of ``x'' + (2/tau)x' + x/tau^2 = noise`` (two cascaded OU). Stationary: var(x) = sigma^2,
    var(v) = sigma^2/tau^2, autocorrelation (1 + |t|/tau)exp(-|t|/tau). Returns (x, v) (n_steps, size), stationary initial state. This is the free process: repulsion is in `simulate_group_offsets`.
    """
    phi, chol = _exact_discretization(dt, tau)
    noise_factor = sigma * chol

    stationary_std = np.array([sigma, sigma / tau])  # diagonal stationary covariance
    state = np.empty((n_steps, 2, size))
    state[0] = stationary_std[:, None] * rng.standard_normal((2, size))
    for k in range(1, n_steps):
        state[k] = phi @ state[k - 1] + noise_factor @ rng.standard_normal((2, size))
    return state[:, 0, :], state[:, 1, :]


@lru_cache(maxsize=None)
def range_statistic_of_normals(n: int, statistic: str = "mean", integration_limit: float = 10.0) -> float:
    """Mean or percentile ("pNN") of the range (max - min) of `n` i.i.d. N(0,1), by numerical integration or `brentq`; scales with sigma."""
    if statistic == "mean":

        def integrand(x: float) -> float:
            cdf = norm.cdf(x)
            return 1.0 - cdf**n - (1.0 - cdf) ** n

        value, _ = quad(integrand, -integration_limit, integration_limit)
        return value

    if not statistic.startswith("p"):
        raise ValueError(f"Unsupported statistic: {statistic!r} (use 'mean' or 'pNN', e.g. 'p95')")
    q = float(statistic[1:]) / 100.0
    if not (0.0 < q < 1.0):
        raise ValueError(f"Percentile out of range: {statistic!r}")

    def range_cdf(r: float) -> float:
        if r <= 0.0:
            return 0.0
        val, _ = quad(
            lambda x: norm.pdf(x) * (norm.cdf(x + r) - norm.cdf(x)) ** (n - 1),
            -integration_limit - 2.0,
            integration_limit + 2.0,
            limit=200,
        )
        return n * val

    return brentq(lambda r: range_cdf(r) - q, 1e-6, 10.0 * integration_limit, xtol=1e-10)


# ---------------------------------------------------------------------------
# Group dynamics: second-order process plus repulsion between runners
# ---------------------------------------------------------------------------


@dataclass
class GroupOffsets:
    """Node offsets from the centroid, with velocities."""

    long: np.ndarray  # (n_steps, size, N) longitudinal offset
    lat: np.ndarray  # (n_steps, size, N) lateral offset
    vel_long: np.ndarray  # (n_steps, size, N) relative longitudinal velocity
    vel_lat: np.ndarray  # (n_steps, size, N) relative lateral velocity


def _straight_frame(s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frame of a straight road (position (s, 0), tangent +x, normal +y), for calibration."""
    zero = np.zeros_like(s)
    one = np.ones_like(s)
    return (
        np.stack([s, zero], axis=-1),
        np.stack([one, zero], axis=-1),
        np.stack([zero, one], axis=-1),
    )


def _repulsion_force(positions: np.ndarray, min_gap: float, strength: float, scale: float, action_distance: float) -> np.ndarray:
    """Repulsive acceleration (size, N, 2) between node pairs in the plane: ``f(d) = A (exp((g - d)/B) - exp((g - R)/B))`` for d < R, otherwise 0,
    along the line joining them (g = `min_gap`, A = `strength` m/s^2, B = `scale` m, R = `action_distance` m). Continuous at R; for coincident nodes it pushes along +x.
    """
    diff = positions[:, :, None, :] - positions[:, None, :, :]  # (size, i, j, 2): from j toward i
    dist = np.linalg.norm(diff, axis=-1)
    magnitude = strength * (np.exp((min_gap - dist) / scale) - np.exp((min_gap - action_distance) / scale))
    n_nodes = positions.shape[1]
    active = (dist < action_distance) & off_diagonal(n_nodes)[None]
    magnitude = np.where(active, magnitude, 0.0)

    safe = np.where(dist > 1e-9, dist, 1.0)
    unit = np.where((dist > 1e-9)[..., None], diff / safe[..., None], np.array([1.0, 0.0]))
    return np.sum(magnitude[..., None] * unit, axis=2)


def _per_axis(long_value: float, lat_value: float) -> np.ndarray:
    return np.array([long_value, lat_value]).reshape(2, 1, 1)


def _advance_detached(detach: dict, k: int, x: np.ndarray, v: np.ndarray, sigma: np.ndarray, base: np.ndarray) -> None:
    node = detach["node"]
    if k == detach["start"]:
        base[:, node] = x[0, :, node]
        x[0, :, node] = 0.0
        v[0, :, node] = 0.0
        sigma[0, :, node] = 0.0
    elif k > detach["start"]:
        base[:, node] += detach["increment"][k]


def _apply_repulsion(
    repulsion: dict, frame, s: np.ndarray, x: np.ndarray, v: np.ndarray, dt: float, detach: Union[dict, None], k: int
) -> None:
    position, tangent, normal = frame(s)
    position = position + x[1][..., None] * normal
    force = _repulsion_force(
        position,
        repulsion["min_gap"],
        repulsion["strength"],
        repulsion["scale"],
        repulsion["action_distance"],
    )
    if detach is not None and k >= detach["start"]:
        force[:, detach["node"], :] = 0.0  # one-way repulsion: the detached node does not feel it
    v[0] += np.sum(force * tangent, axis=-1) * dt
    v[1] += np.sum(force * normal, axis=-1) * dt


def simulate_group_offsets(
    n_steps: int,
    dt: float,
    tau_long: float,
    tau_lat: float,
    sigma_long: float,
    sigma_lat: float,
    rng: np.random.Generator,
    n_nodes: int,
    repulsion: Union[dict, None] = None,
    size: int = 1,
    s_centroid: Union[np.ndarray, None] = None,
    path_frame=None,
    warmup_steps: int = 0,
    detach: Union[dict, None] = None,
) -> GroupOffsets:
    """Offset dynamics: SINGLE function for simulation and calibration. Each offset (long. and lat., different `tau`) follows `critically_damped_process`
    plus the repulsion impulse ``F·dt`` projected onto tangent and normal (semi-implicit Euler; only the linear part is exact). Stationary initial state, `warmup_steps` discarded.
    `path_frame(s)` → (position, tangent, normal), None = straight road; `repulsion` None = independent nodes.
    `detach` (`node`, `start`, `increment` in m/step): from `start` the node advances only by the increment relative to the centroid, does not feel the repulsion but exerts it.
    Returns `GroupOffsets`, arrays (n_steps, size, n_nodes).
    """
    frame = path_frame if path_frame is not None else _straight_frame
    phi_long, chol_long = _exact_discretization(dt, tau_long)
    phi_lat, chol_lat = _exact_discretization(dt, tau_lat)

    p00 = _per_axis(phi_long[0, 0], phi_lat[0, 0])
    p01 = _per_axis(phi_long[0, 1], phi_lat[0, 1])
    p10 = _per_axis(phi_long[1, 0], phi_lat[1, 0])
    p11 = _per_axis(phi_long[1, 1], phi_lat[1, 1])
    c00 = _per_axis(chol_long[0, 0], chol_lat[0, 0])
    c10 = _per_axis(chol_long[1, 0], chol_lat[1, 0])
    c11 = _per_axis(chol_long[1, 1], chol_lat[1, 1])
    tau = _per_axis(tau_long, tau_lat)
    s_c = np.zeros(n_steps) if s_centroid is None else np.asarray(s_centroid, dtype=float)

    sigma = np.empty((2, size, n_nodes))
    sigma[0], sigma[1] = sigma_long, sigma_lat
    x = sigma * rng.standard_normal((2, size, n_nodes))
    v = sigma / tau * rng.standard_normal((2, size, n_nodes))
    base = np.zeros((size, n_nodes))  # deterministic drift of the detached node

    out_long = np.empty((n_steps, size, n_nodes))
    out_lat = np.empty_like(out_long)
    out_vlong = np.empty_like(out_long)
    out_vlat = np.empty_like(out_long)

    for step in range(warmup_steps + n_steps):
        k = step - warmup_steps  # recorded index; negative during warm-up
        kc = max(k, 0)

        if detach is not None and k >= 0:
            _advance_detached(detach, k, x, v, sigma, base)

        if k >= 0:
            out_long[k] = x[0] + base
            out_lat[k] = x[1]
            out_vlong[k] = v[0]
            out_vlat[k] = v[1]

        if repulsion is not None:
            _apply_repulsion(repulsion, frame, s_c[kc] + x[0] + base, x, v, dt, detach, k)

        if k == n_steps - 1:
            break

        z = rng.standard_normal((2, 2, size, n_nodes))
        noise_x = sigma * c00 * z[0]
        noise_v = sigma * (c10 * z[0] + c11 * z[1])
        x, v = p00 * x + p01 * v + noise_x, p10 * x + p11 * v + noise_v

    return GroupOffsets(long=out_long, lat=out_lat, vel_long=out_vlong, vel_lat=out_vlat)


_REPULSION_KEYS = ("min_gap", "strength", "scale", "action_distance")


@dataclass(frozen=True)
class SpreadCalibration:
    n_nodes: int
    statistic: str
    longitudinal_spread: float
    lateral_spread: float
    tau_long: float
    tau_lat: float
    dt: float
    repulsion: tuple
    warmup_steps: int
    calib_groups: int
    calib_steps: int
    calib_seed: int
    calib_max_iterations: int
    calib_tolerance: float


@lru_cache(maxsize=None)
def calibrate_spread_sigmas(spec: SpreadCalibration) -> tuple[float, float]:
    """Calibrates `sigma_long` and `sigma_lat` so that, WITH repulsion, the head-to-tail extent (`statistic`: "mean" or "pNN") equals the requested spreads.
    Fixed point on `simulate_group_offsets` (straight road, fixed seed: deterministic), cached. `repulsion` = (min_gap, strength, scale, action_distance).
    """
    n_nodes = spec.n_nodes
    statistic = spec.statistic
    longitudinal_spread = spec.longitudinal_spread
    lateral_spread = spec.lateral_spread
    tau_long = spec.tau_long
    tau_lat = spec.tau_lat
    dt = spec.dt
    warmup_steps = spec.warmup_steps
    calib_groups = spec.calib_groups
    calib_steps = spec.calib_steps
    calib_seed = spec.calib_seed
    calib_max_iterations = spec.calib_max_iterations
    calib_tolerance = spec.calib_tolerance
    k = range_statistic_of_normals(n_nodes, statistic)
    sigma_long = longitudinal_spread / k
    sigma_lat = lateral_spread / k

    is_percentile = statistic != "mean"
    q = float(statistic[1:]) if is_percentile else None
    repulsion_cfg = dict(zip(_REPULSION_KEYS, spec.repulsion))

    for _ in range(calib_max_iterations):
        rng = np.random.default_rng(calib_seed)
        offsets = simulate_group_offsets(
            calib_steps,
            dt,
            tau_long,
            tau_lat,
            sigma_long,
            sigma_lat,
            rng,
            n_nodes,
            repulsion=repulsion_cfg,
            size=calib_groups,
            warmup_steps=warmup_steps,
        )
        ext_long = (offsets.long.max(axis=2) - offsets.long.min(axis=2)).ravel()
        ext_lat = (offsets.lat.max(axis=2) - offsets.lat.min(axis=2)).ravel()
        obs_long = float(np.percentile(ext_long, q)) if is_percentile else float(ext_long.mean())
        obs_lat = float(np.percentile(ext_lat, q)) if is_percentile else float(ext_lat.mean())

        err_long = abs(obs_long / longitudinal_spread - 1.0)
        err_lat = abs(obs_lat / lateral_spread - 1.0)
        if err_long < calib_tolerance and err_lat < calib_tolerance:
            break

        sigma_long *= longitudinal_spread / obs_long
        sigma_lat *= lateral_spread / obs_lat

    return float(sigma_long), float(sigma_lat)


# ---------------------------------------------------------------------------
# Mobility simulation
# ---------------------------------------------------------------------------


@dataclass
class MobilityResult:
    """Output of Block 1."""

    t: np.ndarray  # (T,)
    positions: np.ndarray  # (T, N, 2) meters
    s_nodes: np.ndarray  # (T, N) curvilinear abscissa of each node
    s_centroid: np.ndarray  # (T,) curvilinear abscissa of the centroid
    distances: np.ndarray  # (T, N, N) Euclidean distances between all pairs
    headings: np.ndarray  # (T, N, 2) path tangent unit vector, direction of travel
    lateral_offsets: np.ndarray  # (T, N) lateral offset from the path, after the constraint
    metadata: dict


def relative_speed(result: MobilityResult) -> np.ndarray:
    """Speed (T-1, N) in m/s of each node relative to the centroid: ``hypot(d(s_i - s_c)/dt, d(l_i)/dt)`` in road coordinates
    (the 2D difference would include the rigid rotation of the group in curves).
    """
    dt = result.metadata["dt"]
    rel_long = np.diff(result.s_nodes - result.s_centroid[:, None], axis=0) / dt
    rel_lat = np.diff(result.lateral_offsets, axis=0) / dt
    return np.hypot(rel_long, rel_lat)


def lateral_offset_speed(result: MobilityResult, track: Track) -> np.ndarray:
    """2D speed (T-1, N) in m/s of the lateral offset alone (``positions − centerline(s_nodes)``): isolates the jumps
    due to the tangent/normal change, without the rigid rotation of the group in curves.
    """
    dt = result.metadata["dt"]
    offset = result.positions - track.query(result.s_nodes).position
    return np.linalg.norm(np.diff(offset, axis=0), axis=-1) / dt


def _integrate_centroid(
    track: Track, dt: float, start_offset: float, duration: Union[float, None], end_margin: float
) -> np.ndarray:
    """Centroid abscissa per step, explicit Euler ``s[k+1] = s[k] + v_track(s[k])*dt``; with `duration` None it integrates up to `length - end_margin`."""
    s_vals = [min(max(start_offset, 0.0), track.length)]

    if duration is not None:
        n_extra_steps = max(int(round(duration / dt)), 0)
        for _ in range(n_extra_steps):
            s_prev = s_vals[-1]
            v = track.query(s_prev).speed
            s_vals.append(min(s_prev + v * dt, track.length))
    else:
        target = max(track.length - end_margin, 0.0)
        # step cap against infinite loops (path covered at the minimum plausible speed)
        min_plausible_speed = _plausible_speed_floor(track)
        max_steps = int(track.length / min_plausible_speed / dt) + 1
        while s_vals[-1] < target and len(s_vals) < max_steps:
            s_prev = s_vals[-1]
            v = track.query(s_prev).speed
            s_vals.append(min(s_prev + v * dt, track.length))

    return np.array(s_vals)


def _plausible_speed_floor(track: Track) -> float:
    """Minimum speed (m/s) for the step cap: 10th percentile of the smoothed profile, never zero."""
    return max(float(np.percentile(track.speed, 10)), 1e-3)


def _repulsion_config(group_cfg: dict) -> dict:
    """Repulsion parameters from the `group` block of the configuration."""
    rep = group_cfg["repulsion"]
    return {
        "min_gap": group_cfg["min_node_gap"],
        "strength": rep["strength"],
        "scale": rep["scale"],
        "action_distance": rep["action_distance"],
    }


def _validate_spread(group_cfg: dict) -> None:
    n_nodes = group_cfg["n_nodes"]
    min_node_gap = group_cfg["min_node_gap"]
    longitudinal_spread = group_cfg["longitudinal_spread"]
    # the formation must hold n_nodes at distance >= min_node_gap, otherwise repulsion dominates
    min_required_spread = (n_nodes - 1) * min_node_gap * group_cfg["spread_margin_factor"]
    if longitudinal_spread <= min_required_spread:
        raise ValueError(
            f"longitudinal_spread ({longitudinal_spread} m) is too small to "
            f"hold {n_nodes} nodes spaced at least min_node_gap={min_node_gap} m "
            f"with margin spread_margin_factor={group_cfg['spread_margin_factor']}: "
            f"longitudinal_spread > {min_required_spread:.2f} m is required."
        )


def _validate_track_margins(track_cfg: dict, margin_sigmas: float, sigma_long: float) -> None:
    # start_offset/end_margin must leave room for margin_sigmas*sigma_long, otherwise the clamp squashes the nodes
    required_margin = margin_sigmas * sigma_long
    if track_cfg["start_offset"] < required_margin:
        raise ValueError(
            f"start_offset ({track_cfg['start_offset']} m) is too small: with "
            f"sigma_long={sigma_long:.3f} m at least {required_margin:.2f} m are needed "
            f"(margin_sigmas={margin_sigmas} · sigma_long), otherwise the tail nodes "
            f"would start with negative curvilinear abscissa."
        )
    if track_cfg["end_margin"] < required_margin:
        raise ValueError(
            f"end_margin ({track_cfg['end_margin']} m) is too small: with "
            f"sigma_long={sigma_long:.3f} m at least {required_margin:.2f} m are needed "
            f"(margin_sigmas={margin_sigmas} · sigma_long), otherwise the head node "
            f"would exceed the end of the available path."
        )


def _separation_detach(
    sep_cfg: dict, track: Track, s_centroid: np.ndarray, t: np.ndarray, dt: float
) -> Union[dict, None]:
    n_steps = len(s_centroid)
    separation_enabled = bool(sep_cfg["enabled"])
    node_id = sep_cfg["node_id"] if separation_enabled else None
    idx_start = int(np.searchsorted(t, sep_cfg["start_time"])) if separation_enabled else None

    # after start_time linear ramp from the group speed to target_speed, without discontinuity;
    # increment[k] = alpha·(target_speed - v_group)·dt, advance per step relative to the centroid
    detach = None
    if separation_enabled and idx_start < n_steps:
        start_time = sep_cfg["start_time"]
        increment = np.zeros(n_steps)
        for k in range(idx_start + 1, n_steps):
            alpha = min(max((t[k - 1] - start_time) / sep_cfg["ramp_duration"], 0.0), 1.0)
            v_group = track.query(s_centroid[k - 1]).speed
            increment[k] = alpha * (sep_cfg["target_speed"] - v_group) * dt
        detach = {"node": node_id, "start": idx_start, "increment": increment}
    return detach


def _make_path_frame(track: Track):
    def path_frame(s: np.ndarray):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the clamp at the ends is reported further below
            sample = track.query(s)
        return sample.position, sample.tangent, sample.normal

    return path_frame


def _project_offsets(
    track: Track, s_centroid: np.ndarray, offsets: GroupOffsets
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_steps, _, n_nodes = offsets.long.shape
    off_long = offsets.long[:, 0, :]
    off_lat = offsets.lat[:, 0, :]
    s_nodes = s_centroid[:, None] + off_long

    s_nodes_before_clip = s_nodes
    s_nodes = np.clip(s_nodes, 0.0, track.length)
    if not np.array_equal(s_nodes_before_clip, s_nodes):
        warnings.warn(
            "s_nodes contained values outside [0, length]: values clamped. With "
            "start_offset/end_margin validated against sigma_long this should "
            "not happen; check the configuration."
        )

    flat_sample = track.query(s_nodes.reshape(-1))
    base_position = flat_sample.position.reshape(n_steps, n_nodes, 2)
    normal = flat_sample.normal.reshape(n_steps, n_nodes, 2)
    headings = flat_sample.tangent.reshape(n_steps, n_nodes, 2)
    positions = base_position + off_lat[:, :, None] * normal

    diff = positions[:, :, None, :] - positions[:, None, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    return s_nodes, off_lat, positions, headings, distances


def simulate_mobility(config: Union[str, Path, dict]) -> MobilityResult:
    """Block 1 mobility simulation (YAML path or config): validation, sigma calibration, track, centroid,
    offsets (`simulate_group_offsets`), projection onto the path. Single generator ``default_rng(seed)``, never global `np.random`.
    """
    cfg = load_config(config)
    sim_cfg = cfg["simulation"]
    track_cfg = cfg["track"]
    group_cfg = cfg["group"]
    sep_cfg = cfg["separation"]

    dt = sim_cfg["dt"]
    seed = sim_cfg["seed"]
    rng = np.random.default_rng(seed)

    n_nodes = group_cfg["n_nodes"]
    tau = group_cfg["correlation_time"]
    tau_lat = group_cfg["lateral_correlation_time"]
    margin_sigmas = group_cfg["margin_sigmas"]
    calib_cfg = group_cfg["calibration"]
    repulsion = _repulsion_config(group_cfg)
    warmup_steps = int(round(group_cfg["warmup_taus"] * max(tau, tau_lat) / dt))

    _validate_spread(group_cfg)

    # calibration on the extent observed WITH repulsion (range_statistic_of_normals is not enough)
    sigma_long, sigma_lat = calibrate_spread_sigmas(
        SpreadCalibration(
            n_nodes,
            group_cfg["spread_statistic"],
            group_cfg["longitudinal_spread"],
            group_cfg["lateral_spread"],
            tau,
            tau_lat,
            dt,
            tuple(repulsion[key] for key in _REPULSION_KEYS),
            warmup_steps,
            calib_cfg["groups"],
            int(round(calib_cfg["duration"] / dt)),
            calib_cfg["seed"],
            calib_cfg["max_iterations"],
            calib_cfg["tolerance"],
        )
    )

    _validate_track_margins(track_cfg, margin_sigmas, sigma_long)

    track = load_track(cfg)

    s_centroid = _integrate_centroid(
        track,
        dt,
        start_offset=track_cfg["start_offset"],
        duration=sim_cfg["duration"],
        end_margin=track_cfg["end_margin"],
    )
    n_steps = len(s_centroid)
    t = np.arange(n_steps) * dt

    separation_enabled = bool(sep_cfg["enabled"])
    detach = _separation_detach(sep_cfg, track, s_centroid, t, dt)
    path_frame = _make_path_frame(track)

    # each node has its own curvilinear abscissa: in a hairpin bend the tail node stays behind the corner
    # (loss of visibility); in-plane repulsion, one-way from the detachment
    offsets = simulate_group_offsets(
        n_steps,
        dt,
        tau,
        tau_lat,
        sigma_long,
        sigma_lat,
        rng,
        n_nodes,
        repulsion=repulsion,
        s_centroid=s_centroid,
        path_frame=path_frame,
        warmup_steps=warmup_steps,
        detach=detach,
    )
    s_nodes, off_lat, positions, headings, distances = _project_offsets(track, s_centroid, offsets)
    min_pair_distance = float(np.min(distances + np.eye(n_nodes)[None, :, :] * 1e9))

    metadata = {
        "seed": seed,
        "config": cfg,
        "n_glitches_corrected": track.n_glitches_corrected,
        "track_length": track.length,
        "n_points_raw": track.n_points_raw,
        "dt": dt,
        "duration": float(t[-1]) if n_steps > 0 else 0.0,
        "sigma_long": sigma_long,
        "sigma_lat": sigma_lat,
        "min_pair_distance": min_pair_distance,
        "separation_start_time": sep_cfg["start_time"] if separation_enabled else None,
        "separation_node_id": sep_cfg["node_id"] if separation_enabled else None,
        "separation_ramp_duration": sep_cfg["ramp_duration"] if separation_enabled else None,
        "separation_target_speed": sep_cfg["target_speed"] if separation_enabled else None,
    }

    return MobilityResult(
        t=t,
        positions=positions,
        s_nodes=s_nodes,
        s_centroid=s_centroid,
        distances=distances,
        headings=headings,
        lateral_offsets=off_lat,
        metadata=metadata,
    )

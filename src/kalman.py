"""Block 4 — Per-link Kalman.

Node `j` keeps one filter for each neighbour `i`, on the link `i → j`, updated only
when it receives a beacon from `i`. Indices ``[i, ..., j]`` = `i` transmits, `j` receives.

- state ``x = [r, s]``: RSSI (dBm) and slope (dB/s); measurement ``z`` = beacon RSSI, ``H = [1, 0]``;
- ``F = [[1, Δt], [0, 1]]`` with `Δt` ACTUAL since the last update (irregular due to jitter and lost beacons);
- continuous random acceleration (slope = Brownian motion of intensity ``σ_a²``)::

      Q = σ_a² · [[Δt³/3, Δt²/2],
                  [Δt²/2, Δt   ]]

  chosen because it is ADDITIVE in time (the predicted covariance does not depend on how
  the interval is split), useful with the irregular gaps of lost beacons;
- measurement noise `R`: only Block 2 fading plus quantization; torso,
  other runners and shadowing are slow and are absorbed by `Q`.

Robustness to body jumps (Block 5, ``kalman.robust``): ``none`` (base), ``inflate`` (R
multiplied by ``r_inflation``) and ``nis_gate`` (if the NIS exceeds ``nis_gate`` the update
uses ``R·NIS/nis_gate``: the innovation weighs less but is not discarded).

Parameters: ``kalman`` section of the config.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.common.enums import RobustMode

ROBUST_OPTIONS = tuple(m.value for m in RobustMode)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class KalmanResult:
    """Posterior estimates aligned to the beacons: ``[i, k, j]`` = link `i → j` after the update from beacon `k` of `i`; NaN if `j` did not receive it."""

    times: np.ndarray  # (N, K) beacon times (NaN past the end)
    received: np.ndarray  # (N, K, N) bool
    r: np.ndarray  # (N, K, N) dBm
    s: np.ndarray  # (N, K, N) dB/s
    P: np.ndarray  # (N, K, N, 2, 2) posterior covariance
    nis: np.ndarray  # (N, K, N) normalized innovation (NaN at initialization)
    reinit: np.ndarray  # (N, K, N) bool: the filter (re)started from this beacon
    sigma_a: float
    measurement_noise: float


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def transition(dt: np.ndarray) -> np.ndarray:
    """``F = [[1, Δt], [0, 1]]`` with `dt` in s (scalar or array): shape ``dt.shape + (2, 2)``."""
    dt = np.asarray(dt, dtype=float)
    f = np.zeros(dt.shape + (2, 2))
    f[..., 0, 0] = 1.0
    f[..., 1, 1] = 1.0
    f[..., 0, 1] = dt
    return f


def process_noise(dt: np.ndarray, sigma_a: float) -> np.ndarray:
    """``Q = σ_a² [[Δt³/3, Δt²/2], [Δt²/2, Δt]]`` (`sigma_a` in dB/s²), shape ``dt.shape + (2, 2)``. Additive: ``Φ(b)Q(a)Φ(b)^T + Q(b) = Q(a+b)``."""
    dt = np.asarray(dt, dtype=float)
    q = np.zeros(dt.shape + (2, 2))
    q[..., 0, 0] = dt**3 / 3.0
    q[..., 0, 1] = dt**2 / 2.0
    q[..., 1, 0] = dt**2 / 2.0
    q[..., 1, 1] = dt
    return sigma_a**2 * q


def predict(x: np.ndarray, P: np.ndarray, dt: np.ndarray, sigma_a: float) -> tuple:
    """``x⁻ = F x``, ``P⁻ = F P Fᵀ + Q`` over `dt` s; explicit 2×2 forms, vectorized, same shapes as the inputs."""
    x = np.asarray(x, dtype=float)
    P = np.asarray(P, dtype=float)
    dt = np.asarray(dt, dtype=float)
    q = process_noise(dt, sigma_a)
    x_new = np.stack([x[..., 0] + dt * x[..., 1], x[..., 1]], axis=-1)
    p00 = P[..., 0, 0] + dt * (P[..., 0, 1] + P[..., 1, 0]) + dt**2 * P[..., 1, 1]
    p01 = P[..., 0, 1] + dt * P[..., 1, 1]
    p10 = P[..., 1, 0] + dt * P[..., 1, 1]
    p11 = P[..., 1, 1]
    P_new = np.stack([np.stack([p00, p01], axis=-1), np.stack([p10, p11], axis=-1)], axis=-2) + q
    return x_new, P_new


def measurement_noise(config: dict, fading_variance: float) -> float:
    """Measurement noise `R` in dB²: with ``auto``, ``var(fading) + quantization_step²/12`` (`fading_variance` from Block 2 in the no-separation group); otherwise the config value."""
    value = config["kalman"]["measurement_noise"]
    if isinstance(value, str):
        if value != "auto":
            raise ValueError(f"kalman.measurement_noise must be 'auto' or a number, not {value!r}")
        step = config["channel"]["measurement"]["quantization_step"]
        return float(fading_variance + step**2 / 12.0)
    return float(value)


# ---------------------------------------------------------------------------
# Single-link filter
# ---------------------------------------------------------------------------


def filter_link(
    times: np.ndarray,
    z: np.ndarray,
    R: float,
    sigma_a: float,
    initial_slope_std: float,
    reinit_gap: float,
    nis_gate: float = math.inf,
) -> tuple:
    """Kalman filter of one link on increasing `times` (n,) and received-beacon RSSI `z` (n,). At the first beacon or after `reinit_gap` s it restarts
    (``r = z``, ``s = 0``, ``P = diag(R, initial_slope_std²)``, no innovation); otherwise it predicts with the actual `Δt` and updates (P in Joseph form).
    NIS = ``(z - r⁻)² / S`` with ``S = P⁻₀₀ + R``. With a finite `nis_gate`, if the NIS exceeds it the update (gain and Joseph) uses ``R·NIS/nis_gate``
    in place of `R`; the reported NIS remains the one computed with `R`. Returns (r, s, P (n, 2, 2), nis, reinit); lost beacons produce nothing (see `predict_links`).
    """
    n = len(times)
    r_out = np.empty(n)
    s_out = np.empty(n)
    p_out = np.empty((n, 2, 2))
    nis_out = np.full(n, np.nan)
    reinit_out = np.zeros(n, dtype=bool)
    slope_var0 = initial_slope_std**2
    sa2 = sigma_a**2

    r = s = p00 = p01 = p11 = 0.0
    t_last = 0.0
    started = False
    for idx in range(n):
        t_now = float(times[idx])
        zz = float(z[idx])
        if (not started) or (t_now - t_last > reinit_gap):
            r, s = zz, 0.0
            p00, p01, p11 = R, 0.0, slope_var0
            reinit_out[idx] = True
            started = True
        else:
            dt = t_now - t_last
            dt2 = dt * dt
            r = r + dt * s
            p00 = p00 + 2.0 * dt * p01 + dt2 * p11 + sa2 * dt2 * dt / 3.0
            p01 = p01 + dt * p11 + sa2 * dt2 / 2.0
            p11 = p11 + sa2 * dt
            innovation = zz - r
            innovation_var = p00 + R
            nis = innovation * innovation / innovation_var
            nis_out[idx] = nis
            noise = R * nis / nis_gate if nis > nis_gate else R
            if noise != R:
                innovation_var = p00 + noise
            k0 = p00 / innovation_var
            k1 = p01 / innovation_var
            r = r + k0 * innovation
            s = s + k1 * innovation
            a = 1.0 - k0
            new_p00 = a * a * p00 + noise * k0 * k0
            new_p01 = a * (p01 - k1 * p00) + noise * k0 * k1
            new_p11 = k1 * k1 * p00 - 2.0 * k1 * p01 + p11 + noise * k1 * k1
            p00, p01, p11 = new_p00, new_p01, new_p11
        t_last = t_now
        r_out[idx] = r
        s_out[idx] = s
        p_out[idx, 0, 0] = p00
        p_out[idx, 0, 1] = p_out[idx, 1, 0] = p01
        p_out[idx, 1, 1] = p11
    return r_out, s_out, p_out, nis_out, reinit_out


def filter_all(
    times: np.ndarray,
    received: np.ndarray,
    rssi: np.ndarray,
    R: float,
    sigma_a: float,
    kalman_cfg: dict,
) -> KalmanResult:
    """Applies `filter_link` to every pair `(i, j)` on the beacons of `i` received by `j`; `times` (N, K), `received` and `rssi` (N, K, N) as in `PacketResult`.

    `kalman_cfg["robust"]`: ``none`` (base), ``inflate`` (the filter uses ``R·r_inflation``, also at startup) or ``nis_gate`` (see `filter_link`).
    The result keeps the nominal `R`; the NIS is the filter's (with ``R·r_inflation`` in ``inflate``).
    """
    robust = kalman_cfg["robust"]
    if robust not in ROBUST_OPTIONS:
        raise ValueError(f"kalman.robust must be one of {ROBUST_OPTIONS}, not {robust!r}")
    R_filter = R * kalman_cfg["r_inflation"] if robust == RobustMode.INFLATE else R
    gate = float(kalman_cfg["nis_gate"]) if robust == RobustMode.NIS_GATE else math.inf
    n_nodes, n_beacons = times.shape
    r = np.full((n_nodes, n_beacons, n_nodes), np.nan)
    s = np.full_like(r, np.nan)
    nis = np.full_like(r, np.nan)
    P = np.full((n_nodes, n_beacons, n_nodes, 2, 2), np.nan)
    reinit = np.zeros((n_nodes, n_beacons, n_nodes), dtype=bool)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            got = np.flatnonzero(received[i, :, j])
            if len(got) == 0:
                continue
            rr, ss, pp, nn, ri = filter_link(
                times[i, got],
                rssi[i, got, j],
                R_filter,
                sigma_a,
                kalman_cfg["initial_slope_std"],
                kalman_cfg["reinit_gap"],
                gate,
            )
            r[i, got, j], s[i, got, j], P[i, got, j], nis[i, got, j], reinit[i, got, j] = rr, ss, pp, nn, ri
    return KalmanResult(times, received.copy(), r, s, P, nis, reinit, float(sigma_a), float(R))


# ---------------------------------------------------------------------------
# Estimate on the grid
# ---------------------------------------------------------------------------


def predict_links(kres: KalmanResult, t: np.ndarray) -> dict:
    """Estimate of each link predicted at every time `t` (T,): last update ``<= t`` plus `predict` over its age.
    Dictionary ``r, s, p00, p01, p11, age``, each (T, N, N) ``[t, i, j]``; NaN before the first update and on the diagonal.
    """
    n_nodes = kres.times.shape[0]
    shape = (len(t), n_nodes, n_nodes)
    out = {name: np.full(shape, np.nan) for name in ("r", "s", "p00", "p01", "p11", "age")}
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i == j:
                continue
            got = np.flatnonzero(kres.received[i, :, j])
            if len(got) == 0:
                continue
            pos = np.searchsorted(kres.times[i, got], t, side="right") - 1
            ok = pos >= 0
            k = got[pos[ok]]
            age = t[ok] - kres.times[i, k]
            x = np.stack([kres.r[i, k, j], kres.s[i, k, j]], axis=-1)
            x_pred, p_pred = predict(x, kres.P[i, k, j], age, kres.sigma_a)
            out["r"][ok, i, j] = x_pred[:, 0]
            out["s"][ok, i, j] = x_pred[:, 1]
            out["p00"][ok, i, j] = p_pred[:, 0, 0]
            out["p01"][ok, i, j] = p_pred[:, 0, 1]
            out["p11"][ok, i, j] = p_pred[:, 1, 1]
            out["age"][ok, i, j] = age
    return out


def mean_nis(kres: KalmanResult, mask: np.ndarray | None = None) -> float:
    """Mean NIS of the finite values (1 if the filter is consistent), with optional (N, K, N) mask."""
    values = kres.nis if mask is None else np.where(mask, kres.nis, np.nan)
    return float(np.nanmean(values))


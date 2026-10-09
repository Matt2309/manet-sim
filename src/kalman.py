"""Blocco 4 — Kalman per link.

Il nodo `j` tiene un filtro per ogni vicino `i`, sul link `i → j`, aggiornato solo
quando riceve un beacon di `i`. Indici ``[i, ..., j]`` = trasmette `i`, riceve `j`.

- stato ``x = [r, s]``: RSSI (dBm) e pendenza (dB/s); misura ``z`` = RSSI del beacon, ``H = [1, 0]``;
- ``F = [[1, Δt], [0, 1]]`` con `Δt` EFFETTIVO dall'ultimo aggiornamento (irregolare per jitter e beacon persi);
- accelerazione casuale continua (pendenza = moto browniano di intensità ``σ_a²``)::

      Q = σ_a² · [[Δt³/3, Δt²/2],
                  [Δt²/2, Δt   ]]

  scelta perché ADDITIVA nel tempo (la covarianza predetta non dipende da come
  si spezza l'intervallo), utile con i buchi irregolari dei beacon persi;
- rumore di misura `R`: solo fading del Blocco 2 più quantizzazione; torso,
  altri corridori e shadowing sono lenti e li assorbe `Q`.

Parametri: sezione ``kalman`` della config.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Strutture dati
# ---------------------------------------------------------------------------


@dataclass
class KalmanResult:
    """Stime a posteriori allineate ai beacon: ``[i, k, j]`` = link `i → j` dopo l'aggiornamento dal beacon `k` di `i`; NaN se `j` non l'ha ricevuto."""

    times: np.ndarray  # (N, K) istanti dei beacon (NaN oltre la fine)
    received: np.ndarray  # (N, K, N) bool
    r: np.ndarray  # (N, K, N) dBm
    s: np.ndarray  # (N, K, N) dB/s
    P: np.ndarray  # (N, K, N, 2, 2) covarianza a posteriori
    nis: np.ndarray  # (N, K, N) innovazione normalizzata (NaN all'inizializzazione)
    reinit: np.ndarray  # (N, K, N) bool: il filtro è (ri)partito da questo beacon
    sigma_a: float
    measurement_noise: float


# ---------------------------------------------------------------------------
# Modello
# ---------------------------------------------------------------------------


def transition(dt: np.ndarray) -> np.ndarray:
    """``F = [[1, Δt], [0, 1]]`` con `dt` in s (scalare o array): forma ``dt.shape + (2, 2)``."""
    dt = np.asarray(dt, dtype=float)
    f = np.zeros(dt.shape + (2, 2))
    f[..., 0, 0] = 1.0
    f[..., 1, 1] = 1.0
    f[..., 0, 1] = dt
    return f


def process_noise(dt: np.ndarray, sigma_a: float) -> np.ndarray:
    """``Q = σ_a² [[Δt³/3, Δt²/2], [Δt²/2, Δt]]`` (`sigma_a` in dB/s²), forma ``dt.shape + (2, 2)``. Additiva: ``Φ(b)Q(a)Φ(b)^T + Q(b) = Q(a+b)``."""
    dt = np.asarray(dt, dtype=float)
    q = np.zeros(dt.shape + (2, 2))
    q[..., 0, 0] = dt**3 / 3.0
    q[..., 0, 1] = dt**2 / 2.0
    q[..., 1, 0] = dt**2 / 2.0
    q[..., 1, 1] = dt
    return sigma_a**2 * q


def predict(x: np.ndarray, P: np.ndarray, dt: np.ndarray, sigma_a: float) -> tuple:
    """``x⁻ = F x``, ``P⁻ = F P Fᵀ + Q`` su `dt` s; forme 2×2 esplicite, vettorizzate, stesse forme degli ingressi."""
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
    """Rumore di misura `R` in dB²: con ``auto``, ``var(fading) + quantization_step²/12`` (`fading_variance` del Blocco 2 nel gruppo senza separazione); altrimenti il valore in config."""
    value = config["kalman"]["measurement_noise"]
    if isinstance(value, str):
        if value != "auto":
            raise ValueError(f"kalman.measurement_noise deve essere 'auto' o un numero, non {value!r}")
        step = config["channel"]["measurement"]["quantization_step"]
        return float(fading_variance + step**2 / 12.0)
    return float(value)


# ---------------------------------------------------------------------------
# Filtro di un link
# ---------------------------------------------------------------------------


def filter_link(
    times: np.ndarray,
    z: np.ndarray,
    R: float,
    sigma_a: float,
    initial_slope_std: float,
    reinit_gap: float,
) -> tuple:
    """Kalman di un link su `times` (n,) crescenti e RSSI `z` (n,) dei beacon ricevuti. Al primo beacon o dopo `reinit_gap` s riparte
    (``r = z``, ``s = 0``, ``P = diag(R, initial_slope_std²)``, senza innovazione); altrimenti predice con `Δt` effettivo e aggiorna (P in forma di Joseph).
    NIS = ``(z - r⁻)² / S``. Ritorna (r, s, P (n, 2, 2), nis, reinit); i beacon persi non producono nulla (vedi `predict_links`).
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
            k0 = p00 / innovation_var
            k1 = p01 / innovation_var
            nis_out[idx] = innovation * innovation / innovation_var
            r = r + k0 * innovation
            s = s + k1 * innovation
            a = 1.0 - k0
            new_p00 = a * a * p00 + R * k0 * k0
            new_p01 = a * (p01 - k1 * p00) + R * k0 * k1
            new_p11 = k1 * k1 * p00 - 2.0 * k1 * p01 + p11 + R * k1 * k1
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
    """Applica `filter_link` a ogni coppia `(i, j)` sui beacon di `i` ricevuti da `j`; `times` (N, K), `received` e `rssi` (N, K, N) come in `PacketResult`."""
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
                R,
                sigma_a,
                kalman_cfg["initial_slope_std"],
                kalman_cfg["reinit_gap"],
            )
            r[i, got, j], s[i, got, j], P[i, got, j], nis[i, got, j], reinit[i, got, j] = rr, ss, pp, nn, ri
    return KalmanResult(times, received.copy(), r, s, P, nis, reinit, float(sigma_a), float(R))


# ---------------------------------------------------------------------------
# Stima sulla griglia
# ---------------------------------------------------------------------------


def predict_links(kres: KalmanResult, t: np.ndarray) -> dict:
    """Stima di ogni link predetta a ogni istante `t` (T,): ultimo aggiornamento ``<= t`` più `predict` per la sua età.
    Dizionario ``r, s, p00, p01, p11, age``, ognuno (T, N, N) ``[t, i, j]``; NaN prima del primo aggiornamento e in diagonale.
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
    """NIS media dei valori finiti (1 se il filtro è consistente), con maschera (N, K, N) opzionale."""
    values = kres.nis if mask is None else np.where(mask, kres.nis, np.nan)
    return float(np.nanmean(values))


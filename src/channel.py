"""Blocco 2 — Modello di canale.

RSSI in dBm per ogni istante e coppia ordinata di nodi: ``rssi[t, i, j]`` =
potenza ricevuta da ``j`` quando trasmette ``i``, diagonale NaN. Sensibilità,
perdite e istanti dei beacon sono del Blocco 3: l'RSSI c'è anche sotto soglia.

Formula (perdite positive, termini casuali additivi)::

    rssi_true = P_tx + G_tx + G_rx - PL(d)      # attenuazione con la distanza
              + S_ij(t)                         # shadowing (simmetrico)
              - B_own_ij(t)                     # torso di chi trasmette e riceve (simmetrico)
              - B_others_ij(t)                  # altri corridori in mezzo (simmetrico)
              + F_ij(t)                         # variazioni rapide (NON simmetrico)
              + tx_offset[i] + rx_offset[j]     # scarto fisso di ogni scheda
              - obstacles_ij(t)                 # edifici (Blocco 2b)

    rssi_measured = quantizza(min(rssi_true, saturazione))

Parametri: sezione ``channel`` della config.

Riferimenti: Agrawal e Patwari, IEEE Trans. Wireless Commun., 2009 (network
shadowing); Wang, Tameh e Nix, IEEE Trans. Veh. Technol., 2008 (shadowing per
link fra terminali mobili).
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

from src.mobility import MobilityResult, load_config

_SPEED_OF_LIGHT = 299792458.0  # m/s


# ---------------------------------------------------------------------------
# Strutture dati
# ---------------------------------------------------------------------------


@dataclass
class ShadowField:
    """Mappa 2D di shadowing (modalità ``field``)."""

    values: np.ndarray  # (nx, ny) float32, indice [ix, iy]
    origin: tuple  # (x0, y0) in metri: posizione della cella [0, 0]
    resolution: float  # m; passo della griglia


@dataclass
class ChannelResult:
    """Output del Blocco 2. Matrici (T, N, N) indicizzate ``[t, i, j]`` (trasmette ``i``, riceve ``j``), diagonale NaN."""

    t: np.ndarray  # (T,)
    rssi_true: np.ndarray  # (T, N, N) dBm, prima di saturazione e quantizzazione
    rssi_measured: np.ndarray  # (T, N, N) dBm, dopo saturazione e quantizzazione
    rssi_path_loss: np.ndarray  # (T, N, N) P_tx + G_tx + G_rx - PL(d)
    shadowing: np.ndarray  # (T, N, N) dB, contributo additivo
    body_own: np.ndarray  # (T, N, N) dB, perdita positiva
    body_others: np.ndarray  # (T, N, N) dB, perdita positiva
    fading: np.ndarray  # (T, N, N) dB, contributo additivo
    obstacles: np.ndarray  # (T, N, N) dB, perdita positiva (Blocco 2b)
    tx_offset: np.ndarray  # (N,) dB
    rx_offset: np.ndarray  # (N,) dB
    shadow_field: Optional[ShadowField]
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Attenuazione con la distanza e lettura
# ---------------------------------------------------------------------------


def reference_path_loss(frequency: float, reference_distance: float) -> float:
    """Attenuazione di spazio libero a `d0`: ``20 log10(4 pi d0 f / c)`` in dB (circa 40,2 dB a 2,437 GHz con d0 = 1 m)."""
    return 20.0 * math.log10(4.0 * math.pi * reference_distance * frequency / _SPEED_OF_LIGHT)


def path_loss_db(distance: np.ndarray, channel_cfg: dict) -> np.ndarray:
    """Modello log-distanza ``PL0 + 10 n log10(max(d, d0) / d0)``: perdita positiva in dB, forma di `distance`; sotto `d0` vale PL(d0)."""
    pl_cfg = channel_cfg["path_loss"]
    d0 = pl_cfg["reference_distance"]
    pl0 = reference_path_loss(channel_cfg["frequency"], d0)
    d = np.maximum(np.asarray(distance, dtype=float), d0)
    return pl0 + 10.0 * pl_cfg["exponent"] * np.log10(d / d0)


def measure_rssi(rssi_true: np.ndarray, measurement_cfg: dict) -> np.ndarray:
    """Saturazione e poi quantizzazione ``round(x / step) * step``; i NaN restano NaN."""
    step = measurement_cfg["quantization_step"]
    saturated = np.minimum(rssi_true, measurement_cfg["saturation_dbm"])
    return np.round(saturated / step) * step


# ---------------------------------------------------------------------------
# Shadowing: modalità "field" (network shadowing)
# ---------------------------------------------------------------------------


def shadow_field_sigma_p(sigma: float, delta: float) -> float:
    """``sigma_p = sigma / sqrt(2 delta)`` in dB/sqrt(m): la varianza del link tende a `sigma^2` per ``d >> delta``."""
    return sigma / math.sqrt(2.0 * delta)


def link_variance_theory(distance: np.ndarray, sigma: float, delta: float) -> np.ndarray:
    """Varianza teorica (dB^2) dello shadowing di un link di lunghezza `d`: ``sigma^2 [1 - (delta/d)(1 - exp(-d/delta))]``."""
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
    """Mappa gaussiana 2D di shadowing, a media nulla, isotropa, autocorrelazione ``exp(-r/delta)``, varianza ``sigma_p^2``.
    Generata per via spettrale (spettro ``(1 + (2 pi k delta)^2)^(-3/2)``) su griglia FFT allargata di `padding` m per lato, poi ritagliata. float32.
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
    """Shadowing di L link come integrale della mappa: ``S = (1/sqrt(d)) * integrale_0^d p(x(u)) du`` (trapezi, interpolazione bilineare).
    `pos_a`, `pos_b` (L, 2); ritorna (L,) in dB, simmetrico. Passo più largo sui link lunghi; elaborazione a blocchi di link interi.
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

        # pesi trapezoidali: passo effettivo d/(n-1), metà peso agli estremi
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
    """Shadowing (T, N, N) in dB, modalità field, da `positions` (T, N, 2): calcolato per ``i < j`` e specchiato, diagonale nulla."""
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
# Shadowing: modalità "independent"
# ---------------------------------------------------------------------------


def independent_shadowing(
    positions: np.ndarray, sigma: float, delta: float, rng: np.random.Generator
) -> np.ndarray:
    """Shadowing (T, N, N) in dB, modalità independent: per coppia, Gauss-Markov che avanza con la distanza percorsa dai due estremi
    (``Delta_k = |dp_i| + |dp_j|``, ``a_k = exp(-Delta_k / delta)``; Wang, Tameh e Nix). Stato iniziale stazionario, nodi fermi = costante. Simmetrico, diagonale nulla.
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
# Corpi
# ---------------------------------------------------------------------------


def own_body_loss(positions: np.ndarray, headings: np.ndarray, own_cfg: dict) -> np.ndarray:
    """Perdita (T, N, N) in dB del torso di chi trasmette e di chi riceve: ``B_ij = L_i(phi_ij) + L_j(phi_ji)``.
    `phi` = angolo di ``p_j - p_i`` rispetto a ``headings[i]`` (antiorario); torso a ``phi_b = +90°`` con ``mount_side: right``, ``-90°`` con ``left``.
    ``L(phi) = max_loss * ((1 + cos(phi - phi_b)) / 2) ** lobe_exponent``.
    """
    side = own_cfg["mount_side"]
    if side not in ("right", "left"):
        raise ValueError(f"channel.body.own.mount_side deve essere 'right' o 'left', non {side!r}")
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
    """Perdita (T, N, N) in dB degli altri corridori (separato compreso) sul link `i–j`: per ogni `k` con proiezione ``0 < u < 1`` sul segmento
    e distanza `c`, ``loss / (1 + exp((c - radius) / transition_width))``; somma su `k` con tetto `max_total_loss`. Calcolata per ``i < j`` e specchiata.
    """
    n_steps, n_nodes, _ = positions.shape
    out = np.zeros((n_steps, n_nodes, n_nodes))
    idx = np.arange(n_nodes)
    not_endpoint = (
        (idx[None, None, :] != idx[:, None, None]) & (idx[None, None, :] != idx[None, :, None])
    )  # (i, j, k): k diverso da i e da j

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
# Variazioni rapide e scarti delle schede
# ---------------------------------------------------------------------------


def rician_fading_db(shape: tuple, k_db: float, rng: np.random.Generator) -> np.ndarray:
    """Fading di Rice a potenza media unitaria: ``F = 20 log10 |h|``, ``h = sqrt(K/(K+1)) + sqrt(1/(K+1)) z``, `z` gaussiana complessa unitaria.
    Estrazioni indipendenti per elemento: fra due passi (0,1 s) il nodo si sposta ~0,28 m >> lambda/2 (~6 cm), e le due direzioni sono misurate in istanti diversi.
    """
    k = 10.0 ** (k_db / 10.0)
    los = math.sqrt(k / (k + 1.0))
    scatter = math.sqrt(1.0 / (k + 1.0))
    re = rng.standard_normal(shape)
    im = rng.standard_normal(shape)
    h = los + scatter * (re + 1j * im) / math.sqrt(2.0)
    return 20.0 * np.log10(np.abs(h))


def device_offsets(n_nodes: int, sigma: float, rng: np.random.Generator) -> tuple:
    """Scarti fissi di ogni scheda, ``N(0, sigma)`` in dB, estratti una volta: (tx_offset, rx_offset), ciascuno (N,)."""
    tx = rng.normal(0.0, sigma, size=n_nodes)
    rx = rng.normal(0.0, sigma, size=n_nodes)
    return tx, rx


def _obstacle_loss(mobility: MobilityResult, obstacles_cfg: dict) -> np.ndarray:
    """Aggancio per il Blocco 2b (edifici OSM): non ancora implementato."""
    raise NotImplementedError("channel.obstacles.enabled: true richiede il Blocco 2b (non ancora implementato).")


# ---------------------------------------------------------------------------
# Simulazione del canale
# ---------------------------------------------------------------------------


def _set_diagonal_nan(*arrays: np.ndarray) -> None:
    n_nodes = arrays[0].shape[1]
    idx = np.arange(n_nodes)
    for arr in arrays:
        arr[:, idx, idx] = np.nan


def simulate_channel(mobility: MobilityResult, config: Union[str, Path, dict]) -> ChannelResult:
    """Simulazione di canale del Blocco 2 (config: percorso YAML o dict con ``simulation`` e ``channel``).
    Quattro generatori indipendenti da ``SeedSequence(simulation.seed)`` in ordine fisso (mappa, shadowing indipendente, scarti, fading): attivare una componente non cambia le altre. Componenti disattivate = 0.
    """
    cfg = load_config(config)
    ch = cfg["channel"]
    seed = cfg["simulation"]["seed"]
    positions = mobility.positions
    n_steps, n_nodes, _ = positions.shape
    timings: dict = {}

    ss = np.random.SeedSequence(seed)
    rng_field, rng_indep, rng_device, rng_fading = (np.random.default_rng(c) for c in ss.spawn(4))

    # attenuazione con la distanza
    t0 = time.perf_counter()
    pl0 = reference_path_loss(ch["frequency"], ch["path_loss"]["reference_distance"])
    rssi_path_loss = ch["tx_power"] + ch["antenna_gain_tx"] + ch["antenna_gain_rx"] - path_loss_db(mobility.distances, ch)
    timings["path_loss"] = time.perf_counter() - t0

    # shadowing
    t0 = time.perf_counter()
    sh = ch["shadowing"]
    shadow_field: Optional[ShadowField] = None
    sigma_p = None
    shadowing = np.zeros((n_steps, n_nodes, n_nodes))
    if sh["enabled"]:
        delta = sh["correlation_distance"]
        if sh["mode"] == "field":
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
        elif sh["mode"] == "independent":
            shadowing = independent_shadowing(positions, sh["sigma"], delta, rng_indep)
        else:
            raise ValueError(f"channel.shadowing.mode deve essere 'field' o 'independent', non {sh['mode']!r}")
    timings["shadowing"] = time.perf_counter() - t0

    # corpi
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

    # variazioni rapide
    t0 = time.perf_counter()
    fading = np.zeros((n_steps, n_nodes, n_nodes))
    if ch["fading"]["enabled"]:
        fading = rician_fading_db((n_steps, n_nodes, n_nodes), ch["fading"]["rician_k_db"], rng_fading)
    timings["fading"] = time.perf_counter() - t0

    # scarti delle schede
    tx_offset = np.zeros(n_nodes)
    rx_offset = np.zeros(n_nodes)
    if ch["device_offset"]["enabled"]:
        tx_offset, rx_offset = device_offsets(n_nodes, ch["device_offset"]["sigma"], rng_device)

    # ostacoli (Blocco 2b)
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

    off_diag = ~np.eye(n_nodes, dtype=bool)
    true_od = rssi_true[:, off_diag]
    meas_od = rssi_measured[:, off_diag]
    saturation = ch["measurement"]["saturation_dbm"]

    def _std(arr: np.ndarray) -> float:
        return float(np.std(arr[:, off_diag]))

    metadata = {
        "seed": seed,
        "config": cfg,
        "pl0": pl0,
        "sigma_p": sigma_p,
        "shadowing_mode": sh["mode"] if sh["enabled"] else None,
        "saturated_fraction": float(np.mean(true_od > saturation)),
        "rssi_measured_median": float(np.median(meas_od)),
        "rssi_measured_p5": float(np.percentile(meas_od, 5)),
        "rssi_measured_p95": float(np.percentile(meas_od, 95)),
        "component_std": {
            "path_loss": _std(rssi_path_loss),
            "shadowing": _std(shadowing),
            "body_own": _std(body_own),
            "body_others": _std(body_others),
            "fading": _std(fading),
        },
        "timings": timings,
    }

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

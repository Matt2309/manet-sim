"""Blocco 2 — Modello di canale.

Modulo che calcola, per ogni istante della griglia di mobilità e per ogni
coppia ordinata di nodi, l'RSSI in dBm. Convenzione: ``rssi[t, i, j]`` è la
potenza ricevuta dal nodo ``j`` quando trasmette il nodo ``i``; la
diagonale vale NaN.

Il modulo non decide se un pacchetto arriva (sensibilità, perdite e
istanti dei beacon sono compito del Blocco 3): l'RSSI si calcola anche
quando è sotto la sensibilità.

Formula complessiva (perdite memorizzate come numeri positivi, termini
casuali come contributi additivi)::

    rssi_true = P_tx + G_tx + G_rx - PL(d)      # attenuazione con la distanza
              + S_ij(t)                         # shadowing (simmetrico)
              - B_own_ij(t)                     # torso di chi trasmette e di chi riceve (simmetrico)
              - B_others_ij(t)                  # altri corridori in mezzo (simmetrico)
              + F_ij(t)                         # variazioni rapide (NON simmetrico)
              + tx_offset[i] + rx_offset[j]     # scarto fisso di ogni scheda
              - obstacles_ij(t)                 # edifici (Blocco 2b)

    rssi_measured = quantizza(min(rssi_true, saturazione))

Nessun valore numerico del modello è salvato in questo file: tutti i
parametri vengono letti dalla sezione ``channel`` della configurazione YAML
(vedi ``config/default.yaml``). Le costanti fisiche (velocità della luce,
``10/ln 10``) sono ammesse.

Riferimenti:
- Agrawal e Patwari, "Correlated link shadow fading in multi-hop wireless
  networks", IEEE Trans. Wireless Commun., 2009 (network shadowing).
- Wang, Tameh e Nix, "Joint shadowing process in urban peer-to-peer radio
  channels", IEEE Trans. Veh. Technol., 2008 (shadowing per link fra
  terminali mobili).
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
    """Output della simulazione di canale (Blocco 2).

    Tutte le matrici (T, N, N) seguono la convenzione ``[t, i, j]`` =
    trasmette ``i``, riceve ``j``, con diagonale NaN.
    """

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
    """Attenuazione di spazio libero alla distanza di riferimento.

    Input: frequenza in Hz e distanza di riferimento `d0` in metri.
    Procedimento: ``PL0 = 20 log10(4 pi d0 f / c)``, calcolata dalla
    frequenza e non scritta a mano.
    Output: PL0 in dB (circa 40,2 dB a 2,437 GHz con d0 = 1 m).
    """
    return 20.0 * math.log10(4.0 * math.pi * reference_distance * frequency / _SPEED_OF_LIGHT)


def path_loss_db(distance: np.ndarray, channel_cfg: dict) -> np.ndarray:
    """Attenuazione con la distanza (modello log-distanza).

    Input: distanze in metri (qualunque forma) e sezione ``channel``.
    Procedimento: ``PL(d) = PL0 + 10 n log10(max(d, d0) / d0)``: sotto `d0`
    l'attenuazione resta quella a `d0`.
    Output: perdita positiva in dB, stessa forma di `distance`.
    """
    pl_cfg = channel_cfg["path_loss"]
    d0 = pl_cfg["reference_distance"]
    pl0 = reference_path_loss(channel_cfg["frequency"], d0)
    d = np.maximum(np.asarray(distance, dtype=float), d0)
    return pl0 + 10.0 * pl_cfg["exponent"] * np.log10(d / d0)


def measure_rssi(rssi_true: np.ndarray, measurement_cfg: dict) -> np.ndarray:
    """Lettura dell'RSSI da parte della scheda.

    Input: RSSI vero in dBm e sezione ``channel.measurement``.
    Procedimento: prima il tetto di saturazione, poi la quantizzazione
    ``round(x / step) * step``. I NaN restano NaN.
    Output: RSSI misurato in dBm.
    """
    step = measurement_cfg["quantization_step"]
    saturated = np.minimum(rssi_true, measurement_cfg["saturation_dbm"])
    return np.round(saturated / step) * step


# ---------------------------------------------------------------------------
# Shadowing: modalità "field" (network shadowing)
# ---------------------------------------------------------------------------


def shadow_field_sigma_p(sigma: float, delta: float) -> float:
    """Deviazione standard della mappa.

    Input: `sigma` (dB, sui link lunghi) e distanza di correlazione `delta`.
    Procedimento: ``sigma_p^2 = sigma^2 / (2 delta)``, così la varianza del
    link tende a `sigma^2` per ``d >> delta``.
    Output: `sigma_p` in dB / sqrt(m).
    """
    return sigma / math.sqrt(2.0 * delta)


def link_variance_theory(distance: np.ndarray, sigma: float, delta: float) -> np.ndarray:
    """Varianza teorica dello shadowing di un link in funzione della lunghezza.

    Input: lunghezza `d` del link, `sigma`, `delta`.
    Procedimento: per l'integrale normalizzato di un campo con covarianza
    esponenziale, ``var(d) = sigma_p^2 * 2 delta * [1 - (delta/d)(1 - exp(-d/delta))]``
    con ``sigma_p^2 * 2 delta = sigma^2``.
    Output: varianza in dB^2.
    """
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
    """Genera la mappa gaussiana 2D di shadowing.

    Input: estremi del rettangolo da coprire (`x_range`, `y_range`),
    `sigma_p`, `delta`, passo della griglia, `padding` (metri) e
    generatore `rng`.
    Procedimento: campo a media nulla, isotropo, con autocorrelazione
    esponenziale ``exp(-r/delta)``, generato per via spettrale. Lo spettro
    2D della covarianza esponenziale è proporzionale a
    ``(1 + (2 pi k delta)^2)^(-3/2)``. Rumore bianco → FFT → prodotto con
    la radice dello spettro → FFT inversa. La griglia della FFT è
    allargata di `padding` metri per lato, per evitare che la periodicità
    correli i bordi opposti, e poi ritagliata. La normalizzazione è fatta
    sulla somma discreta dello spettro, così la varianza della mappa vale
    esattamente ``sigma_p^2`` (in media). Conservata in float32.
    Output: `ShadowField`.
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
    """Shadowing di una lista di link, come integrale della mappa.

    Input: mappa, estremi dei link `pos_a`, `pos_b` (L, 2) e sezione
    ``channel.shadowing``.
    Procedimento: ``S = (1/sqrt(d)) * integrale_0^d p(x(u)) du``, con
    integrale trapezoidale e interpolazione bilineare della mappa
    (`map_coordinates`, ordine 1). Passo di integrazione
    `integration_step`; per i link più lunghi di ``long_link_factor * delta``
    il passo sale a ``long_link_step_factor * delta`` (errore
    trascurabile, perché la mappa varia su scala `delta`). Tutti i punti di
    tutti i link sono vettorializzati e processati per blocchi di link
    interi: il risultato di ogni link non dipende dal blocco. Il valore è
    simmetrico per costruzione (l'integrale non dipende dal verso).
    Output: array (L,) in dB.
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
    """Shadowing di tutte le coppie di nodi a tutti gli istanti (modalità field).

    Input: posizioni (T, N, 2), mappa e sezione ``channel.shadowing``.
    Procedimento: calcolato solo per le coppie non ordinate ``i < j`` e
    poi specchiato, quindi ``S_ij = S_ji`` esattamente.
    Output: array (T, N, N) in dB, diagonale nulla.
    """
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
    """Shadowing indipendente per ogni coppia non ordinata (modalità independent).

    Input: posizioni (T, N, 2), `sigma`, `delta`, generatore `rng`.
    Procedimento: ogni coppia ha un processo di Gauss-Markov che non
    avanza nel tempo ma nella distanza percorsa dai due estremi (modello di
    Wang, Tameh e Nix per link fra terminali mobili):
        ``Delta_k = |dp_i| + |dp_j|``, ``a_k = exp(-Delta_k / delta)``,
        ``S_k = a_k S_{k-1} + sigma sqrt(1 - a_k^2) w_k``.
    È la stessa discretizzazione esatta dell'Ornstein-Uhlenbeck del
    Blocco 1, con stato iniziale estratto dalla distribuzione stazionaria.
    Con nodi fermi `Delta` = 0 e lo shadowing resta costante. Varianza
    `sigma^2` indipendente dalla lunghezza del link.
    Output: array (T, N, N) in dB simmetrico, diagonale nulla.
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
    """Perdita dovuta al torso di chi trasmette e di chi riceve.

    Input: posizioni (T, N, 2), direzioni di marcia (T, N, 2) e sezione
    ``channel.body.own``.
    Procedimento: per il nodo `i`, `phi` è l'angolo del vettore ``p_j - p_i``
    rispetto a ``headings[i]``, positivo in senso antiorario (verso
    sinistra). Con ``mount_side: right`` il torso sta a ``phi_b = +90°``
    (``-90°`` con ``left``). ``L(phi) = max_loss * ((1 + cos(phi - phi_b)) / 2) ** lobe_exponent``.
    La perdita del link è la somma dei due estremi,
    ``B_ij = L_i(phi_ij) + L_j(phi_ji)``, simmetrica per costruzione.
    Output: perdita positiva in dB, (T, N, N).
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
    """Perdita dovuta agli altri corridori che stanno in mezzo al link.

    Input: posizioni (T, N, 2) e sezione ``channel.body.others``.
    Procedimento: per il link `i–j` e ogni altro nodo `k` (nodo separato
    compreso) si calcolano il parametro di proiezione `u` di ``p_k`` sul
    segmento e la distanza `c` di ``p_k`` dal segmento. Se ``0 < u < 1`` il
    contributo è ``loss / (1 + exp((c - radius) / transition_width))``,
    altrimenti 0. Somma su `k` e tetto a `max_total_loss`. Calcolato per
    ``i < j`` e specchiato, quindi simmetrico esattamente.
    Output: perdita positiva in dB, (T, N, N), diagonale nulla.
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
    """Variazioni rapide (fading di Rice) a potenza media unitaria.

    Input: forma dell'array, fattore K di Rice in dB, generatore `rng`.
    Procedimento: per ogni elemento si estrae in modo indipendente un
    guadagno complesso ``h = sqrt(K/(K+1)) + sqrt(1/(K+1)) z`` con `z`
    gaussiana complessa circolare a varianza unitaria, e ``F = 20 log10 |h|``.
    La potenza media ``E|h|^2`` vale 1. L'indipendenza fra istanti è
    giustificata perché fra due passi (0,1 s) ogni nodo si sposta di circa
    0,28 m, molto più di lambda/2 (circa 6 cm): il canale si è
    decorrelato. L'indipendenza fra le due direzioni di una coppia è
    giustificata perché le due misure avvengono in istanti diversi.
    Output: F in dB, forma `shape`.
    """
    k = 10.0 ** (k_db / 10.0)
    los = math.sqrt(k / (k + 1.0))
    scatter = math.sqrt(1.0 / (k + 1.0))
    re = rng.standard_normal(shape)
    im = rng.standard_normal(shape)
    h = los + scatter * (re + 1j * im) / math.sqrt(2.0)
    return 20.0 * np.log10(np.abs(h))


def device_offsets(n_nodes: int, sigma: float, rng: np.random.Generator) -> tuple:
    """Scarti fissi di ogni scheda.

    Input: numero di nodi, `sigma` in dB, generatore `rng`.
    Procedimento: due vettori di lunghezza N estratti una volta sola per
    simulazione da ``N(0, sigma)``: prima `tx_offset`, poi `rx_offset`.
    Output: (tx_offset, rx_offset).
    """
    tx = rng.normal(0.0, sigma, size=n_nodes)
    rx = rng.normal(0.0, sigma, size=n_nodes)
    return tx, rx


def _obstacle_loss(mobility: MobilityResult, obstacles_cfg: dict) -> np.ndarray:
    """Aggancio per il Blocco 2b (edifici da OpenStreetMap): non ancora implementato."""
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
    """Esegue la simulazione di canale del Blocco 2.

    Input: `MobilityResult` del Blocco 1 e configurazione (percorso YAML o
    dizionario; deve contenere le sezioni ``simulation`` e ``channel``).
    Procedimento: da ``SeedSequence(simulation.seed)`` si derivano, in
    ordine fisso, quattro generatori indipendenti (mappa di shadowing,
    shadowing indipendente, scarti delle schede, variazioni rapide): così
    attivare o disattivare una componente non cambia le estrazioni delle
    altre. Poi si calcolano le componenti (distanza, shadowing, torso,
    altri corridori, fading, scarti, ostacoli), si assemblano secondo la
    formula del modulo e si applicano saturazione e quantizzazione. Ogni
    componente disattivata contribuisce zero.
    Output: `ChannelResult`.
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

"""Blocco 1 — Mobilità.

Modulo che genera la traiettoria di un gruppo podistico lungo un percorso GPX
reale: dove si trova ogni nodo a ogni istante.

Nessun valore numerico è salvato in questo file: tutti i parametri del
modello vengono letti da un file di configurazione YAML (vedi
``config/default.yaml``).
"""

from __future__ import annotations

import copy
import logging
import math
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Union

import numpy as np
import yaml
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.ndimage import uniform_filter1d
from scipy.optimize import brentq
from scipy.stats import norm

logger = logging.getLogger(__name__)

_GPX_NAMESPACE = {"gpx": "http://www.topografix.com/GPX/1/1"}
_EARTH_RADIUS_M = 6371000.0


# ---------------------------------------------------------------------------
# Caricamento e pulizia della traccia GPX
# ---------------------------------------------------------------------------


@dataclass
class PathSample:
    """Campione del percorso interrogato a una data ascissa curvilinea."""

    position: np.ndarray  # (..., 2) in metri
    tangent: np.ndarray  # (..., 2) versore, direzione di marcia
    normal: np.ndarray  # (..., 2) versore, perpendicolare a sinistra
    speed: np.ndarray  # (...,) m/s, dal profilo GPX


@dataclass
class Track:
    """Traccia GPX proiettata su un piano locale in metri, ripulita dai
    buchi di campionamento GPS e corredata del profilo di velocità.
    """

    x: np.ndarray
    y: np.ndarray
    s: np.ndarray  # ascissa curvilinea, stessa lunghezza di x/y
    ele: np.ndarray
    speed: np.ndarray  # profilo di velocità lisciato, per punto
    speed_raw: np.ndarray  # profilo di velocità grezzo, per punto
    length: float
    n_glitches_corrected: int
    n_points_raw: int
    n_points: int
    sample_rate: float
    heading_s: np.ndarray  # punti medi dei segmenti, ascissa curvilinea
    heading: np.ndarray  # angolo di direzione lisciato, srotolato, in radianti

    def query(self, s: Union[float, np.ndarray]) -> PathSample:
        """Interroga il percorso a una o più ascisse curvilinee.

        Input: `s`, metri dall'inizio del percorso (scalare o array di
        qualunque forma). Valori fuori da [0, length] vengono bloccati ai
        capi, con un warning.
        Procedimento: posizione e velocità per interpolazione lineare lungo
        la polilinea, vettorializzata. Tangente e normale NON sono costanti
        a tratti: l'angolo di direzione lisciato (`_smooth_heading`) è
        interpolato linearmente in `s`, quindi il riferimento locale varia
        con continuità anche ai vertici. La linea centrale resta quella
        della polilinea.
        Output: `PathSample` con posizione, versore tangente, versore
        normale (a sinistra della marcia) e velocità del profilo GPX.
        """
        s_arr = np.asarray(s, dtype=float)
        scalar_input = s_arr.ndim == 0
        s_flat = np.atleast_1d(s_arr).reshape(-1)

        clipped = np.clip(s_flat, 0.0, self.length)
        if not np.array_equal(clipped, s_flat):
            warnings.warn(
                "Ascissa curvilinea fuori dal percorso: valori clampati a "
                f"[0, {self.length:.1f}] m."
            )

        # indice del segmento [s[idx], s[idx+1]] che contiene ciascun punto
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
        nx, ny = -ty, tx  # rotazione di +90°: normale verso sinistra

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
    """Angolo di direzione del percorso, continuo lungo l'ascissa curvilinea.

    Input: x, y, s della polilinea e larghezza `width` (m) della media mobile.
    Procedimento: l'angolo di ogni segmento è attribuito al suo punto medio e
    srotolato con `np.unwrap`. La media mobile di larghezza `width` in `s` è
    esatta anche con segmenti di lunghezza diversa: l'integrale Θ(s) di
    un angolo costante a tratti è lineare a tratti (somma cumulata di
    θ_j·lunghezza_j), quindi θ̄(s) = (Θ(s+w/2) − Θ(s−w/2)) / w. Oltre i capi
    l'angolo è prolungato costante.
    Output: punti medi dei segmenti e angolo lisciato in quei punti. I
    segmenti di lunghezza nulla sono ignorati.
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
    """Legge i punti di un file GPX 1.1.

    Input: percorso del file GPX.
    Procedimento: cerca i `<trkpt>` passando il namespace esplicito (senza,
    `findall` non trova nulla).
    Output: array lat, lon, ele.
    """
    root = ET.parse(path).getroot()
    trkpts = root.findall(".//gpx:trkpt", _GPX_NAMESPACE)
    if not trkpts:
        raise ValueError(f"Nessun <trkpt> trovato in {path}: file GPX vuoto o namespace inatteso.")
    lat = np.array([float(p.get("lat")) for p in trkpts])
    lon = np.array([float(p.get("lon")) for p in trkpts])
    ele = np.array(
        [float(p.find("gpx:ele", _GPX_NAMESPACE).text) for p in trkpts]
    )
    return lat, lon, ele


def _project_to_local_plane(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Proietta lat/lon su un piano locale in metri.

    Input: array lat e lon in gradi.
    Procedimento: proiezione equirettangolare con origine sul primo punto;
    l'errore resta sotto il metro perché il percorso è lungo poche migliaia
    di metri.
    Output: coordinate x, y in metri.
    """
    lat0 = math.radians(lat[0])
    lon0 = math.radians(lon[0])
    x = (np.radians(lon) - lon0) * _EARTH_RADIUS_M * math.cos(lat0)
    y = (np.radians(lat) - lat0) * _EARTH_RADIUS_M
    return x, y


def _repair_glitches(
    x: np.ndarray, y: np.ndarray, ele: np.ndarray, glitch_max_step: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Colma i buchi di campionamento GPS.

    Input: x, y, ele della traccia e soglia `glitch_max_step`.
    Procedimento: i passi più lunghi della soglia sono considerati anomali
    e riempiti con punti interpolati linearmente, così geometria e
    lunghezza totale restano invariate.
    Output: x, y, ele corretti e numero di passi anomali corretti.
    """
    d = np.hypot(np.diff(x), np.diff(y))
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
    """Carica e ripulisce la traccia GPX.

    Input: configurazione `cfg['track']` (file, soglia dei glitch,
    `assumed_sample_rate`, finestra di lisciatura, intervallo di velocità
    plausibile).
    Procedimento: parsing, proiezione, correzione dei glitch, ascissa
    curvilinea e profilo di velocità (grezzo e lisciato). Il GPX non ha
    timestamp, quindi si assume campionamento uniforme; l'assunzione è
    verificata confrontando la velocità mediana con
    `plausible_speed_range`, con warning se fuori intervallo.
    Output: `Track`.
    """
    track_cfg = cfg["track"]

    lat, lon, ele = _parse_gpx(track_cfg["gpx_file"])
    n_points_raw = len(lat)
    if n_points_raw < 2:
        raise ValueError("La traccia GPX deve contenere almeno 2 punti.")

    x, y = _project_to_local_plane(lat, lon)
    x, y, ele, n_glitches = _repair_glitches(x, y, ele, track_cfg["glitch_max_step"])
    if n_glitches > 0:
        logger.warning(
            "Corretti %d punti anomali (glitch GPS) nella traccia %s",
            n_glitches,
            track_cfg["gpx_file"],
        )

    d = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0.0], np.cumsum(d)])

    sample_rate = float(track_cfg["assumed_sample_rate"])
    v_seg = d * sample_rate  # velocità istantanea per segmento
    speed_raw = np.empty(len(x))
    speed_raw[0] = v_seg[0]
    speed_raw[-1] = v_seg[-1]
    speed_raw[1:-1] = (v_seg[:-1] + v_seg[1:]) / 2.0  # media dei due segmenti adiacenti

    window = max(int(round(track_cfg["speed_smoothing_window"] * sample_rate)), 1)
    speed = uniform_filter1d(speed_raw, size=window, mode="nearest")

    lo, hi = track_cfg["plausible_speed_range"]
    median_speed = float(np.median(speed_raw))
    if not (lo <= median_speed <= hi):
        warnings.warn(
            f"Velocità mediana implicita ({median_speed:.2f} m/s) fuori dal range "
            f"plausibile [{lo}, {hi}] m/s per la corsa: assumed_sample_rate="
            f"{sample_rate} Hz potrebbe essere sbagliato per questo file."
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
# Processo del secondo ordine a smorzamento critico (scostamenti dei nodi)
# ---------------------------------------------------------------------------


def _exact_discretization(dt: float, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Discretizzazione esatta del sistema a 2 stati (posizione, velocità).

    Input: passo `dt` e costante di tempo `tau`.
    Procedimento: A = [[0, 1], [-1/tau^2, -2/tau]], rumore solo sulla
    velocità con intensità q^2 = 4/tau^3 (cioè sigma = 1). Matrice di
    transizione Phi = expm(A·dt); covarianza del rumore Qd con il metodo di
    Van Loan, sull'esponenziale della matrice a blocchi
    [[-A, G q^2 G^T], [0, A^T]]·dt. Per una sigma qualunque Qd scala con
    sigma^2, quindi il fattore di Cholesky scala con sigma.
    Output: Phi (2, 2) e fattore di Cholesky triangolare inferiore di Qd per
    sigma = 1.
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
    """Genera realizzazioni indipendenti di un processo del secondo ordine
    a smorzamento critico (due stadi OU in cascata con la stessa `tau`).

    Input: numero di passi, passo `dt`, costante di tempo `tau`, deviazione
    standard stazionaria della posizione `sigma`, generatore `rng`, numero
    di realizzazioni `size`.
    Procedimento: equazione x'' + (2/tau)·x' + x/tau^2 = rumore bianco.
    Varianze stazionarie: var(x) = sigma^2, var(v) = sigma^2/tau^2, covarianza
    nulla; autocorrelazione della posizione (1 + |t|/tau)·exp(-|t|/tau).
    Discretizzazione esatta (vedi `_exact_discretization`), valida per
    qualunque `dt`. Lo stato iniziale è estratto dalla covarianza stazionaria
    congiunta, non posto a zero.
    Output: (x, v), ciascuno di forma (n_steps, size). La posizione è
    derivabile, quindi `v` è la velocità regolare dello scostamento.
    È il processo lineare "libero": la dinamica del gruppo, con la repulsione
    fra i corridori, sta in `simulate_group_offsets`.
    """
    phi, chol = _exact_discretization(dt, tau)
    noise_factor = sigma * chol

    stationary_std = np.array([sigma, sigma / tau])  # covarianza stazionaria diagonale
    state = np.empty((n_steps, 2, size))
    state[0] = stationary_std[:, None] * rng.standard_normal((2, size))
    for k in range(1, n_steps):
        state[k] = phi @ state[k - 1] + noise_factor @ rng.standard_normal((2, size))
    return state[:, 0, :], state[:, 1, :]


@lru_cache(maxsize=None)
def range_statistic_of_normals(n: int, statistic: str = "mean", integration_limit: float = 10.0) -> float:
    """Statistica del range (max - min) di `n` variabili N(0,1) i.i.d.

    Input: `n`, `statistic` ("mean" o "pNN", es. "p95"), limite di
    integrazione.
    Procedimento: calcolo numerico (media per integrazione, percentile
    invertendo la CDF del range con `brentq`). Per N(0, sigma^2) il valore
    scala con `sigma`, così la calibrazione di sigma resta corretta al
    variare di `n_nodes`.
    Output: valore della statistica per sigma = 1.
    """
    if statistic == "mean":

        def integrand(x: float) -> float:
            cdf = norm.cdf(x)
            return 1.0 - cdf**n - (1.0 - cdf) ** n

        value, _ = quad(integrand, -integration_limit, integration_limit)
        return value

    if not statistic.startswith("p"):
        raise ValueError(f"Statistica non supportata: {statistic!r} (usare 'mean' o 'pNN', es. 'p95')")
    q = float(statistic[1:]) / 100.0
    if not (0.0 < q < 1.0):
        raise ValueError(f"Percentile fuori range: {statistic!r}")

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
# Dinamica del gruppo: processo del secondo ordine più repulsione fra corridori
# ---------------------------------------------------------------------------


@dataclass
class GroupOffsets:
    """Scostamenti dei nodi dal baricentro, con le velocità."""

    long: np.ndarray  # (n_steps, size, N) scostamento longitudinale
    lat: np.ndarray  # (n_steps, size, N) scostamento laterale
    vel_long: np.ndarray  # (n_steps, size, N) velocità longitudinale relativa
    vel_lat: np.ndarray  # (n_steps, size, N) velocità laterale relativa


def _straight_frame(s: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sistema di riferimento di una strada rettilinea: posizione (s, 0),
    tangente +x, normale +y. Serve alla calibrazione, che non ha percorso.
    """
    zero = np.zeros_like(s)
    one = np.ones_like(s)
    return (
        np.stack([s, zero], axis=-1),
        np.stack([one, zero], axis=-1),
        np.stack([zero, one], axis=-1),
    )


def _repulsion_force(positions: np.ndarray, min_gap: float, strength: float, scale: float, action_distance: float) -> np.ndarray:
    """Forza repulsiva (accelerazione) fra tutte le coppie di nodi, nel piano.

    Input: posizioni (size, N, 2), distanza minima `min_gap`, `strength` A
    (m/s^2), `scale` B (m), `action_distance` R (m).
    Procedimento: per ogni coppia, modulo
        f(d) = A·(exp((g - d)/B) - exp((g - R)/B))   per d < R, altrimenti 0
    diretto lungo la congiungente, verso l'allontanamento. Il termine
    sottratto annulla la forza esattamente in R, quindi f è continua ovunque
    e senza gradini. A d = 0 la forza resta finita (A·exp(g/B)); per due nodi
    coincidenti la direzione è indefinita e si spinge lungo +x.
    Output: forza risultante su ogni nodo (size, N, 2).
    """
    diff = positions[:, :, None, :] - positions[:, None, :, :]  # (size, i, j, 2): da j verso i
    dist = np.linalg.norm(diff, axis=-1)
    magnitude = strength * (np.exp((min_gap - dist) / scale) - np.exp((min_gap - action_distance) / scale))
    n_nodes = positions.shape[1]
    active = (dist < action_distance) & ~np.eye(n_nodes, dtype=bool)[None]
    magnitude = np.where(active, magnitude, 0.0)

    safe = np.where(dist > 1e-9, dist, 1.0)
    unit = np.where((dist > 1e-9)[..., None], diff / safe[..., None], np.array([1.0, 0.0]))
    return np.sum(magnitude[..., None] * unit, axis=2)


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
    """Dinamica degli scostamenti dei nodi: UNICA funzione usata sia dalla
    simulazione sia dalla calibrazione di `sigma_long` e `sigma_lat`.

    Input: numero di passi registrati, `dt`, costanti di tempo longitudinale
    `tau_long` e laterale `tau_lat`, sigma longitudinale e laterale della
    posizione, generatore `rng`, numero di nodi, parametri
    della repulsione `repulsion` (dizionario con `min_gap`, `strength`,
    `scale`, `action_distance`; None = nodi indipendenti), numero di gruppi
    indipendenti `size`, ascissa del baricentro `s_centroid` (n_steps,),
    funzione `path_frame(s) -> (posizione, tangente, normale)` (None = strada
    rettilinea), passi di riscaldamento `warmup_steps` scartati, e
    `detach` (None oppure dizionario con `node`, `start` e `increment`, vedi
    sotto).
    Procedimento: ogni scostamento, longitudinale e laterale, segue il sistema
    lineare a 2 stati di `critically_damped_process`, con la propria costante
    di tempo. Le due sono diverse: il riallineamento laterale, cioè tornare
    sulla propria linea dopo una schivata, è molto più rapido della deriva
    longitudinale rispetto al gruppo; con la stessa tau un calcio laterale
    della repulsione persisterebbe per decine di secondi e l'estensione
    laterale non sarebbe più calibrabile. A ogni passo, per
    l'intero gruppo: (1) si calcolano le posizioni nel piano (punto del
    percorso più scostamento laterale lungo la normale); (2) la forza di
    repulsione fra le coppie, calcolata nel piano lungo la congiungente, si
    proietta sulla tangente e sulla normale locali di ciascun nodo e si somma
    alla velocità come impulso `F·dt` (Eulero semi-implicito: la velocità si
    aggiorna prima della posizione); (3) lo stato avanza con la
    discretizzazione esatta della parte lineare. L'esattezza vale SOLO per la
    parte lineare: la forza è integrata a passo `dt`. Lo stato iniziale è
    estratto dalla covarianza stazionaria del processo libero; il
    riscaldamento lascia che la repulsione porti il gruppo alla propria
    distribuzione.
    Nodo separato: da `detach["start"]` lo scostamento longitudinale del nodo
    `detach["node"]` non ha più il richiamo né il rumore del gruppo: avanza
    con `detach["increment"][k]` (metri per passo, relativi al baricentro) e
    subisce comunque la forza (risposta lineare smorzata, senza rumore), ed è
    sorgente di forza per gli altri. La repulsione non lo trattiene.
    Output: `GroupOffsets`, ogni array di forma (n_steps, size, n_nodes).
    """
    frame = path_frame if path_frame is not None else _straight_frame
    phi_long, chol_long = _exact_discretization(dt, tau_long)
    phi_lat, chol_lat = _exact_discretization(dt, tau_lat)

    def per_axis(long_value: float, lat_value: float) -> np.ndarray:
        return np.array([long_value, lat_value]).reshape(2, 1, 1)

    p00 = per_axis(phi_long[0, 0], phi_lat[0, 0])
    p01 = per_axis(phi_long[0, 1], phi_lat[0, 1])
    p10 = per_axis(phi_long[1, 0], phi_lat[1, 0])
    p11 = per_axis(phi_long[1, 1], phi_lat[1, 1])
    c00 = per_axis(chol_long[0, 0], chol_lat[0, 0])
    c10 = per_axis(chol_long[1, 0], chol_lat[1, 0])
    c11 = per_axis(chol_long[1, 1], chol_lat[1, 1])
    tau = per_axis(tau_long, tau_lat)
    s_c = np.zeros(n_steps) if s_centroid is None else np.asarray(s_centroid, dtype=float)

    sigma = np.empty((2, size, n_nodes))
    sigma[0], sigma[1] = sigma_long, sigma_lat
    x = sigma * rng.standard_normal((2, size, n_nodes))
    v = sigma / tau * rng.standard_normal((2, size, n_nodes))
    base = np.zeros((size, n_nodes))  # deriva deterministica del nodo separato

    out_long = np.empty((n_steps, size, n_nodes))
    out_lat = np.empty_like(out_long)
    out_vlong = np.empty_like(out_long)
    out_vlat = np.empty_like(out_long)

    for step in range(warmup_steps + n_steps):
        k = step - warmup_steps  # indice registrato; negativo durante il riscaldamento
        kc = max(k, 0)

        if detach is not None and k >= 0:
            node = detach["node"]
            if k == detach["start"]:
                base[:, node] = x[0, :, node]
                x[0, :, node] = 0.0
                v[0, :, node] = 0.0
                sigma[0, :, node] = 0.0
            elif k > detach["start"]:
                base[:, node] += detach["increment"][k]

        if k >= 0:
            out_long[k] = x[0] + base
            out_lat[k] = x[1]
            out_vlong[k] = v[0]
            out_vlat[k] = v[1]

        if repulsion is not None:
            s = s_c[kc] + x[0] + base
            position, tangent, normal = frame(s)
            position = position + x[1][..., None] * normal
            force = _repulsion_force(
                position,
                repulsion["min_gap"],
                repulsion["strength"],
                repulsion["scale"],
                repulsion["action_distance"],
            )
            v[0] += np.sum(force * tangent, axis=-1) * dt
            v[1] += np.sum(force * normal, axis=-1) * dt

        if k == n_steps - 1:
            break

        z = rng.standard_normal((2, 2, size, n_nodes))
        noise_x = sigma * c00 * z[0]
        noise_v = sigma * (c10 * z[0] + c11 * z[1])
        x, v = p00 * x + p01 * v + noise_x, p10 * x + p11 * v + noise_v

    return GroupOffsets(long=out_long, lat=out_lat, vel_long=out_vlong, vel_lat=out_vlat)


@lru_cache(maxsize=None)
def calibrate_spread_sigmas(
    n_nodes: int,
    statistic: str,
    longitudinal_spread: float,
    lateral_spread: float,
    tau_long: float,
    tau_lat: float,
    dt: float,
    repulsion: tuple,
    warmup_steps: int,
    calib_groups: int,
    calib_steps: int,
    calib_seed: int,
    calib_max_iterations: int,
    calib_tolerance: float,
) -> tuple[float, float]:
    """Calibra `sigma_long` e `sigma_lat` del processo degli scostamenti.

    Input: numero di nodi, `statistic` ("mean" o "pNN"), spread
    longitudinale e laterale desiderati, `tau_long`, `tau_lat`, `dt`, parametri della
    repulsione come tupla (min_gap, strength, scale, action_distance),
    passi di riscaldamento e parametri della calibrazione (gruppi
    indipendenti, passi, seme, iterazioni, tolleranza).
    Procedimento: sigma iniziale = spread / `range_statistic_of_normals`
    (esatto senza repulsione); poi punto fisso: si simula con
    `simulate_group_offsets` (la stessa funzione della simulazione, su una
    strada rettilinea) `calib_groups` gruppi indipendenti, si misura la
    statistica dell'estensione testa-coda osservata e si riscala sigma finché
    l'errore relativo scende sotto `calib_tolerance`. Ogni iterazione usa lo
    stesso seme, quindi gli stessi numeri casuali: il punto fisso è
    deterministico. Il risultato è in cache.
    Output: (sigma_long, sigma_lat) tali che, CON la repulsione, l'estensione
    osservata valga gli spread richiesti.
    """
    k = range_statistic_of_normals(n_nodes, statistic)
    sigma_long = longitudinal_spread / k
    sigma_lat = lateral_spread / k

    is_percentile = statistic != "mean"
    q = float(statistic[1:]) if is_percentile else None
    repulsion_cfg = dict(zip(("min_gap", "strength", "scale", "action_distance"), repulsion))

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
# Simulazione della mobilità
# ---------------------------------------------------------------------------


@dataclass
class MobilityResult:
    """Output della simulazione di mobilità (Blocco 1)."""

    t: np.ndarray  # (T,)
    positions: np.ndarray  # (T, N, 2) metri
    s_nodes: np.ndarray  # (T, N) ascissa curvilinea di ogni nodo
    s_centroid: np.ndarray  # (T,) ascissa curvilinea del baricentro
    distances: np.ndarray  # (T, N, N) distanze euclidee fra tutte le coppie
    headings: np.ndarray  # (T, N, 2) versore tangente del percorso, direzione di marcia
    lateral_offsets: np.ndarray  # (T, N) scostamento laterale dal percorso, dopo il vincolo
    metadata: dict


def relative_speed(result: MobilityResult) -> np.ndarray:
    """Velocità di ogni nodo relativa al baricentro, a passo `dt`.

    Input: `MobilityResult`.
    Procedimento: in coordinate stradali, `hypot(d(s_i - s_c)/dt, d(l_i)/dt)`.
    Si evita la differenza in 2D perché nelle curve un nodo davanti al
    baricentro ha direzione di marcia diversa: ne risulterebbe una velocità
    relativa apparente del tutto fisica, che non è il rumore da misurare.
    Output: array (T-1, N) in m/s.
    """
    dt = result.metadata["dt"]
    rel_long = np.diff(result.s_nodes - result.s_centroid[:, None], axis=0) / dt
    rel_lat = np.diff(result.lateral_offsets, axis=0) / dt
    return np.hypot(rel_long, rel_lat)


def lateral_offset_speed(result: MobilityResult, track: Track) -> np.ndarray:
    """Velocità 2D dello scostamento laterale di ogni nodo, a passo `dt`.

    Input: `MobilityResult` e la `Track` da cui proviene.
    Procedimento: derivata a passo `dt` del vettore `positions − linea
    centrale(s_nodes)`, cioè del solo scostamento laterale, misurato nel
    piano. Non si usa la velocità 2D rispetto al baricentro delle posizioni:
    in curva include la rotazione rigida del gruppo (un nodo davanti al
    baricentro ha direzione di marcia diversa), che dipende dalla geometria
    e non dal rumore; resta ~1-3 m/s anche con scostamenti laterali nulli.
    Questa misura isola ciò che il riferimento locale può rompere: i salti
    dello scostamento quando tangente e normale cambiano di colpo.
    Output: array (T-1, N) in m/s.
    """
    dt = result.metadata["dt"]
    offset = result.positions - track.query(result.s_nodes).position
    return np.linalg.norm(np.diff(offset, axis=0), axis=-1) / dt


def load_config(config: Union[str, Path, dict]) -> dict:
    """Carica la configurazione.

    Input: percorso di un file YAML oppure dizionario già caricato (comodo
    per i test).
    Procedimento: per un percorso, legge il YAML e risolve `track.gpx_file`
    rispetto alla cartella del file YAML (non alla working directory),
    così funziona ovunque venga lanciato lo script; per un dizionario ne
    fa una copia profonda.
    Output: dizionario di configurazione.
    """
    if isinstance(config, (str, Path)):
        config_path = Path(config).resolve()
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        gpx_path = Path(cfg["track"]["gpx_file"])
        if not gpx_path.is_absolute():
            gpx_path = (config_path.parent / gpx_path).resolve()
        cfg["track"]["gpx_file"] = str(gpx_path)
        return cfg
    return copy.deepcopy(config)


def _integrate_centroid(
    track: Track, dt: float, start_offset: float, duration: Union[float, None], end_margin: float
) -> np.ndarray:
    """Integra nel tempo l'ascissa curvilinea del baricentro del gruppo.

    Input: traccia, passo `dt`, `start_offset`, `duration` (o None),
    `end_margin`.
    Procedimento: Eulero esplicito, s[k+1] = s[k] + v_track(s[k])*dt. Con
    `duration` None si integra finché il baricentro raggiunge
    `length - end_margin`, così il nodo di testa non esce dal tracciato.
    Output: array delle ascisse del baricentro, una per passo.
    """
    s_vals = [min(max(start_offset, 0.0), track.length)]

    if duration is not None:
        n_extra_steps = max(int(round(duration / dt)), 0)
        for _ in range(n_extra_steps):
            s_prev = s_vals[-1]
            v = track.query(s_prev).speed
            s_vals.append(min(s_prev + v * dt, track.length))
    else:
        target = max(track.length - end_margin, 0.0)
        # Procedimento: tetto ai passi contro loop infiniti (percorso
        # coperto alla velocità minima plausibile).
        min_plausible_speed = _plausible_speed_floor(track)
        max_steps = int(track.length / min_plausible_speed / dt) + 1
        while s_vals[-1] < target and len(s_vals) < max_steps:
            s_prev = s_vals[-1]
            v = track.query(s_prev).speed
            s_vals.append(min(s_prev + v * dt, track.length))

    return np.array(s_vals)


def _plausible_speed_floor(track: Track) -> float:
    """Velocità minima di sicurezza per il tetto ai passi di integrazione.

    Input: traccia.
    Procedimento: decimo percentile del profilo di velocità lisciato,
    comunque mai nullo.
    Output: velocità in m/s.
    """
    return max(float(np.percentile(track.speed, 10)), 1e-3)


def _repulsion_config(group_cfg: dict) -> dict:
    """Parametri della repulsione dal blocco `group` della configurazione."""
    rep = group_cfg["repulsion"]
    return {
        "min_gap": group_cfg["min_node_gap"],
        "strength": rep["strength"],
        "scale": rep["scale"],
        "action_distance": rep["action_distance"],
    }


def simulate_mobility(config: Union[str, Path, dict]) -> MobilityResult:
    """Esegue la simulazione di mobilità del Blocco 1.

    Input: percorso di un file YAML o configurazione già caricata.
    Procedimento: validazione della formazione, calibrazione degli sigma,
    caricamento della traccia, moto del baricentro, dinamica degli scostamenti
    dei nodi (processo del secondo ordine con repulsione fra i corridori,
    eventuale separazione di un nodo) in `simulate_group_offsets`, proiezione
    sul percorso. Il generatore casuale è unico (`default_rng(seed)`) e
    passato esplicitamente, mai `np.random` globale, per la riproducibilità.
    Output: `MobilityResult`.
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
    spread_statistic = group_cfg["spread_statistic"]
    longitudinal_spread = group_cfg["longitudinal_spread"]
    lateral_spread = group_cfg["lateral_spread"]
    min_node_gap = group_cfg["min_node_gap"]
    margin_sigmas = group_cfg["margin_sigmas"]
    calib_cfg = group_cfg["calibration"]
    repulsion = _repulsion_config(group_cfg)
    warmup_steps = int(round(group_cfg["warmup_taus"] * max(tau, tau_lat) / dt))

    # Input: spread e min_node_gap. Procedimento: la formazione deve poter
    # contenere n_nodes distanziati almeno min_node_gap, altrimenti la
    # repulsione domina la dinamica. Output: errore se lo spread è troppo piccolo.
    min_required_spread = (n_nodes - 1) * min_node_gap * group_cfg["spread_margin_factor"]
    if longitudinal_spread <= min_required_spread:
        raise ValueError(
            f"longitudinal_spread ({longitudinal_spread} m) è troppo piccolo per "
            f"contenere {n_nodes} nodi distanziati almeno min_node_gap={min_node_gap} m "
            f"con margine spread_margin_factor={group_cfg['spread_margin_factor']}: "
            f"serve longitudinal_spread > {min_required_spread:.2f} m."
        )

    # Input: spread desiderati e parametri della repulsione. Procedimento:
    # calibrazione sull'estensione osservata CON la repulsione (dividere per
    # range_statistic_of_normals non basta). Output: sigma_long, sigma_lat.
    sigma_long, sigma_lat = calibrate_spread_sigmas(
        n_nodes,
        spread_statistic,
        longitudinal_spread,
        lateral_spread,
        tau,
        tau_lat,
        dt,
        tuple(repulsion[key] for key in ("min_gap", "strength", "scale", "action_distance")),
        warmup_steps,
        calib_cfg["groups"],
        int(round(calib_cfg["duration"] / dt)),
        calib_cfg["seed"],
        calib_cfg["max_iterations"],
        calib_cfg["tolerance"],
    )

    # Input: start_offset, end_margin, sigma_long. Procedimento: devono
    # lasciare spazio a margin_sigmas*sigma_long, altrimenti il clamp
    # finale schiaccerebbe in silenzio i nodi di coda/testa. Output:
    # errore se il margine è insufficiente.
    required_margin = margin_sigmas * sigma_long
    if track_cfg["start_offset"] < required_margin:
        raise ValueError(
            f"start_offset ({track_cfg['start_offset']} m) è troppo piccolo: con "
            f"sigma_long={sigma_long:.3f} m servono almeno {required_margin:.2f} m "
            f"(margin_sigmas={margin_sigmas} · sigma_long), altrimenti i nodi in coda "
            f"partirebbero con ascissa curvilinea negativa."
        )
    if track_cfg["end_margin"] < required_margin:
        raise ValueError(
            f"end_margin ({track_cfg['end_margin']} m) è troppo piccolo: con "
            f"sigma_long={sigma_long:.3f} m servono almeno {required_margin:.2f} m "
            f"(margin_sigmas={margin_sigmas} · sigma_long), altrimenti il nodo di testa "
            f"supererebbe la fine del percorso disponibile."
        )

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
    node_id = sep_cfg["node_id"] if separation_enabled else None
    idx_start = int(np.searchsorted(t, sep_cfg["start_time"])) if separation_enabled else None

    # Input: nodo separato, start_time, ramp_duration, target_speed.
    # Procedimento: dopo start_time il nodo lascia il baricentro con una
    # rampa lineare dalla velocità di gruppo a target_speed, che parte dal
    # valore che aveva nel gruppo, senza discontinuità. `increment[k]` è
    # l'avanzamento per passo rispetto al baricentro, cioè
    # alpha·(target_speed - v_gruppo)·dt. Output: dizionario per
    # `simulate_group_offsets`, o None.
    detach = None
    if separation_enabled and idx_start < n_steps:
        start_time = sep_cfg["start_time"]
        increment = np.zeros(n_steps)
        for k in range(idx_start + 1, n_steps):
            alpha = min(max((t[k - 1] - start_time) / sep_cfg["ramp_duration"], 0.0), 1.0)
            v_group = track.query(s_centroid[k - 1]).speed
            increment[k] = alpha * (sep_cfg["target_speed"] - v_group) * dt
        detach = {"node": node_id, "start": idx_start, "increment": increment}

    def path_frame(s: np.ndarray):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # il clamp ai capi viene segnalato più sotto
            sample = track.query(s)
        return sample.position, sample.tangent, sample.normal

    # Input: ascissa del baricentro, sigma, parametri della repulsione.
    # Procedimento: ogni nodo ha una propria ascissa curvilinea, non uno
    # scostamento euclideo; così in una curva a gomito il nodo in coda
    # resta dietro l'angolo e si riproduce la perdita di visibilità, caso
    # d'uso principale. La repulsione, calcolata nel piano, impedisce che i
    # corridori si compenetrino e agisce SEMPRE su tutti i nodi, separato
    # incluso. Output: scostamenti longitudinale e laterale.
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
    off_long = offsets.long[:, 0, :]
    off_lat = offsets.lat[:, 0, :]
    s_nodes = s_centroid[:, None] + off_long

    s_nodes_before_clip = s_nodes
    s_nodes = np.clip(s_nodes, 0.0, track.length)
    if not np.array_equal(s_nodes_before_clip, s_nodes):
        warnings.warn(
            "s_nodes conteneva valori fuori da [0, length]: valori clampati. Con "
            "start_offset/end_margin validati contro sigma_long non dovrebbe "
            "accadere; verificare la configurazione."
        )

    flat_sample = track.query(s_nodes.reshape(-1))
    base_position = flat_sample.position.reshape(n_steps, n_nodes, 2)
    normal = flat_sample.normal.reshape(n_steps, n_nodes, 2)
    headings = flat_sample.tangent.reshape(n_steps, n_nodes, 2)
    positions = base_position + off_lat[:, :, None] * normal

    diff = positions[:, :, None, :] - positions[:, None, :, :]
    distances = np.linalg.norm(diff, axis=-1)
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

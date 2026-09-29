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

    def query(self, s: Union[float, np.ndarray]) -> PathSample:
        """Interroga il percorso a una o più ascisse curvilinee.

        Input: `s`, metri dall'inizio del percorso (scalare o array di
        qualunque forma). Valori fuori da [0, length] vengono bloccati ai
        capi, con un warning.
        Procedimento: interpolazione lineare lungo la polilinea,
        vettorializzata.
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

        dx, dy = x1 - x0, y1 - y0
        seg_norm = np.hypot(dx, dy)
        seg_norm_safe = np.where(seg_norm > 0, seg_norm, 1.0)
        tx, ty = dx / seg_norm_safe, dy / seg_norm_safe
        nx, ny = -ty, tx  # rotazione di +90°: normale verso sinistra

        speed = np.interp(clipped, self.s, self.speed)

        position = np.stack([pos_x, pos_y], axis=-1).reshape(s_arr.shape + (2,))
        tangent = np.stack([tx, ty], axis=-1).reshape(s_arr.shape + (2,))
        normal = np.stack([nx, ny], axis=-1).reshape(s_arr.shape + (2,))
        speed = speed.reshape(s_arr.shape)

        if scalar_input:
            return PathSample(position[()], tangent[()], normal[()], speed[()])
        return PathSample(position, tangent, normal, speed)


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

    return Track(
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
# Processo di Ornstein-Uhlenbeck (scostamenti dei nodi)
# ---------------------------------------------------------------------------


def ou_process(
    n_steps: int,
    dt: float,
    tau: float,
    sigma: float,
    rng: np.random.Generator,
    size: int = 1,
) -> np.ndarray:
    """Genera realizzazioni indipendenti di un processo di Ornstein-Uhlenbeck.

    Input: numero di passi, passo `dt`, tempo di correlazione `tau`,
    deviazione standard a regime `sigma`, generatore `rng`, numero di
    realizzazioni `size`.
    Procedimento: discretizzazione esatta (valida per qualunque `dt`):
        a = exp(-dt/tau)
        X[k+1] = a*X[k] + sigma*sqrt(1-a^2)*N(0,1)
    Lo stato iniziale è estratto dalla distribuzione stazionaria
    N(0, sigma^2), non posto a zero, per non "aprire" il gruppo nei primi
    secondi.
    Output: array di forma (n_steps, size).
    """
    a = math.exp(-dt / tau)
    noise_std = sigma * math.sqrt(max(1.0 - a**2, 0.0))

    x = np.empty((n_steps, size))
    x[0] = rng.normal(0.0, sigma, size=size)
    for k in range(1, n_steps):
        x[k] = a * x[k - 1] + noise_std * rng.normal(0.0, 1.0, size=size)
    return x


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
# Vincolo di distanza minima fra nodi
# ---------------------------------------------------------------------------


def _random_gap_margin(
    rng: np.random.Generator,
    n_steps: int,
    n_nodes: int,
    tau: float,
    dt: float,
    softness: float,
) -> np.ndarray:
    """Margine casuale da sommare a `min_node_gap`.

    Input: generatore, numero di passi e di nodi, `tau`, `dt`, `softness`
    (sigma del margine).
    Procedimento: per ciascuna coppia di nodi, `|X(t)|` con `X` processo
    OU indipendente (stesso `tau` del gruppo, tramite `ou_process`).
    Output: matrice simmetrica (n_steps, n_nodes, n_nodes) con diagonale
    nulla (tutta zero se `softness` <= 0).
    """
    if softness <= 0.0:
        return np.zeros((n_steps, n_nodes, n_nodes))

    iu, ju = np.triu_indices(n_nodes, k=1)
    n_pairs = len(iu)
    raw = ou_process(n_steps, dt, tau, softness, rng, size=n_pairs)

    margin = np.zeros((n_steps, n_nodes, n_nodes))
    margin[:, iu, ju] = np.abs(raw)
    margin[:, ju, iu] = np.abs(raw)
    return margin


def _enforce_min_distance(
    off_long: np.ndarray,
    off_lat: np.ndarray,
    min_gap: float,
    margin: np.ndarray,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Impone una distanza euclidea minima fra tutti i nodi, a ogni istante.

    Input: scostamenti longitudinali `off_long` e laterali `off_lat`
    (T, N), distanza minima `min_gap`, `margin` casuale, `tolerance`,
    `max_iterations`.
    Procedimento: rilassamento a "sfere soffici" nel piano delle
    coordinate stradali; a ogni iterazione le coppie più vicine del
    bersaglio (`min_gap + margin`) vengono allontanate simmetricamente
    lungo la congiungente, fino a violazione residua sotto `tolerance` o
    `max_iterations`. Vettorializzato su tutti gli istanti (nessun ciclo
    su T).
    Output: scostamenti corretti (off_long, off_lat), spostamento indotto
    per nodo-istante (diagnostica di quanto il vincolo perturba la
    formazione) e violazione residua massima.
    """
    s = off_long.copy()
    l = off_lat.copy()
    n_nodes = s.shape[1]
    eye = np.eye(n_nodes, dtype=bool)[None, :, :]
    target = min_gap + margin

    for _ in range(max_iterations):
        ds = s[:, :, None] - s[:, None, :]
        dl = l[:, :, None] - l[:, None, :]
        d = np.hypot(ds, dl)
        overlap = (d < target - tolerance) & ~eye
        if not overlap.any():
            break

        d_safe = np.where(d > 1e-9, d, 1.0)
        ux = np.where(d > 1e-9, ds / d_safe, 1.0)  # fallback se d=0: spinta lungo +s
        uy = np.where(d > 1e-9, dl / d_safe, 0.0)
        push = np.where(overlap, (target - d) / 2.0, 0.0)

        s = s + (push * ux).sum(axis=2)
        l = l + (push * uy).sum(axis=2)

    ds = s[:, :, None] - s[:, None, :]
    dl = l[:, :, None] - l[:, None, :]
    d = np.hypot(ds, dl)
    residual = float(np.max(np.where(~eye, target - d, 0.0)))

    displacement = np.hypot(s - off_long, l - off_lat)
    return s, l, displacement, residual


@lru_cache(maxsize=None)
def calibrate_spread_sigmas(
    n_nodes: int,
    statistic: str,
    longitudinal_spread: float,
    lateral_spread: float,
    min_node_gap: float,
    min_gap_softness: float,
    tau: float,
    dt: float,
    calib_samples: int,
    calib_seed: int,
    calib_max_iterations: int,
    calib_tolerance: float,
    min_gap_tolerance: float,
    min_gap_max_iterations: int,
) -> tuple[float, float]:
    """Calibra `sigma_long` e `sigma_lat` del processo OU.

    Input: numero di nodi, `statistic` ("mean" o "pNN"), spread
    longitudinale e laterale desiderati, parametri del vincolo di distanza
    minima (gap, softness, tolleranza, iterazioni), `tau`, `dt` e
    parametri della calibrazione (campioni, seed, iterazioni, tolleranza).
    Procedimento: sigma iniziale = spread / `range_statistic_of_normals`
    (esatto senza vincolo); poi punto fisso su `calib_samples` campioni
    i.i.d. N(0, sigma^2) (le marginali stazionarie dell'OU): si applica
    `_enforce_min_distance`, si misura la statistica osservata e si
    riscala sigma finché l'errore relativo scende sotto `calib_tolerance`.
    Il risultato è in cache (tipicamente poche iterazioni, sotto il
    secondo).
    Output: (sigma_long, sigma_lat) tali che, DOPO il vincolo, l'estensione
    testa-coda osservata valga gli spread richiesti.
    """
    k = range_statistic_of_normals(n_nodes, statistic)
    sigma_long = longitudinal_spread / k
    sigma_lat = lateral_spread / k

    is_percentile = statistic != "mean"
    q = float(statistic[1:]) if is_percentile else None

    rng = np.random.default_rng(calib_seed)
    z_long = rng.standard_normal((calib_samples, n_nodes))
    z_lat = rng.standard_normal((calib_samples, n_nodes))
    margin = _random_gap_margin(rng, calib_samples, n_nodes, tau, dt, min_gap_softness)

    for _ in range(calib_max_iterations):
        s, l, _, _ = _enforce_min_distance(
            z_long * sigma_long, z_lat * sigma_lat, min_node_gap, margin, min_gap_tolerance, min_gap_max_iterations
        )
        ext_long = s.max(axis=1) - s.min(axis=1)
        ext_lat = l.max(axis=1) - l.min(axis=1)
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
    metadata: dict


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


def simulate_mobility(config: Union[str, Path, dict]) -> MobilityResult:
    """Esegue la simulazione di mobilità del Blocco 1.

    Input: percorso di un file YAML o configurazione già caricata.
    Procedimento: validazione della formazione, calibrazione degli sigma,
    caricamento della traccia, moto del baricentro, scostamenti OU dei
    nodi, eventuale separazione di un nodo, vincolo di distanza minima,
    proiezione sul percorso. Il generatore casuale è unico
    (`default_rng(seed)`) e passato esplicitamente, mai `np.random`
    globale, per la riproducibilità.
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
    spread_statistic = group_cfg["spread_statistic"]
    longitudinal_spread = group_cfg["longitudinal_spread"]
    lateral_spread = group_cfg["lateral_spread"]
    min_node_gap = group_cfg["min_node_gap"]
    min_gap_tolerance = group_cfg["min_gap_tolerance"]
    min_gap_max_iterations = group_cfg["min_gap_max_iterations"]
    margin_sigmas = group_cfg["margin_sigmas"]
    calib_cfg = group_cfg["calibration"]

    # Input: spread e min_node_gap. Procedimento: la formazione deve poter
    # contenere n_nodes distanziati almeno min_node_gap, altrimenti il
    # vincolo domina la dinamica. Output: errore se lo spread è troppo piccolo.
    min_required_spread = (n_nodes - 1) * min_node_gap * group_cfg["spread_margin_factor"]
    if longitudinal_spread <= min_required_spread:
        raise ValueError(
            f"longitudinal_spread ({longitudinal_spread} m) è troppo piccolo per "
            f"contenere {n_nodes} nodi distanziati almeno min_node_gap={min_node_gap} m "
            f"con margine spread_margin_factor={group_cfg['spread_margin_factor']}: "
            f"serve longitudinal_spread > {min_required_spread:.2f} m."
        )

    # Input: spread desiderati e parametri del vincolo. Procedimento:
    # calibrazione sull'estensione osservata DOPO il vincolo (dividere per
    # range_statistic_of_normals non basta: con min_node_gap vicino allo
    # spread il vincolo è il regime dominante). Output: sigma_long, sigma_lat.
    sigma_long, sigma_lat = calibrate_spread_sigmas(
        n_nodes,
        spread_statistic,
        longitudinal_spread,
        lateral_spread,
        min_node_gap,
        group_cfg["min_gap_softness"],
        tau,
        dt,
        calib_cfg["samples"],
        calib_cfg["seed"],
        calib_cfg["max_iterations"],
        calib_cfg["tolerance"],
        min_gap_tolerance,
        min_gap_max_iterations,
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

    off_long = ou_process(n_steps, dt, tau, sigma_long, rng, size=n_nodes)
    off_lat = ou_process(n_steps, dt, tau, sigma_lat, rng, size=n_nodes)

    # Input: ascissa del baricentro e scostamenti OU. Procedimento: ogni
    # nodo ha una propria ascissa curvilinea, non uno scostamento euclideo;
    # così in una curva a gomito il nodo in coda resta dietro l'angolo e si
    # riproduce la perdita di visibilità, caso d'uso principale. Output:
    # s_nodes ideale (non ancora vincolato).
    s_nodes = s_centroid[:, None] + off_long  # (T, N), traiettoria "ideale" (non ancora vincolata)

    separation_enabled = bool(sep_cfg["enabled"])
    node_id = sep_cfg["node_id"] if separation_enabled else None
    idx_start = int(np.searchsorted(t, sep_cfg["start_time"])) if separation_enabled else None

    if separation_enabled and idx_start < n_steps:
        start_time = sep_cfg["start_time"]
        ramp_duration = sep_cfg["ramp_duration"]
        target_speed = sep_cfg["target_speed"]

        # Input: nodo separato, start_time, ramp_duration, target_speed.
        # Procedimento: dopo start_time il nodo lascia il baricentro e
        # integra la propria ascissa con una rampa lineare dalla velocità di
        # gruppo a target_speed (parte dal valore che aveva nel gruppo, senza
        # discontinuità). Lo scostamento OU longitudinale del nodo viene
        # ignorato da qui in poi. Output: s_nodes[:, node_id] sovrascritto;
        # il vincolo di distanza minima è applicato dopo, sulla traiettoria
        # reale (rampa inclusa).
        s_sep = s_nodes[idx_start, node_id]
        for k in range(idx_start, n_steps):
            if k > idx_start:
                alpha = min(max((t[k - 1] - start_time) / ramp_duration, 0.0), 1.0)
                v_group = track.query(s_centroid[k - 1]).speed
                v = (1.0 - alpha) * v_group + alpha * target_speed
                s_sep = s_sep + v * dt
            s_nodes[k, node_id] = s_sep

    # Input: s_nodes (rampa inclusa), off_lat, margine casuale.
    # Procedimento: vincolo di distanza minima (i corridori non si
    # compenetrano) applicato SEMPRE a tutti i nodi e istanti, sulla
    # traiettoria REALE del nodo separato e non su quella ideale: se
    # applicato prima della rampa, questa lo scavalcherebbe subito dopo
    # start_time, quando il nodo è ancora dentro il gruppo. È puramente
    # repulsivo, quindi inerte quando il nodo si allontana.
    # Output: s_nodes e off_lat corretti, spostamento indotto e residuo.
    off_long_effective = s_nodes - s_centroid[:, None]
    margin = _random_gap_margin(rng, n_steps, n_nodes, tau, dt, group_cfg["min_gap_softness"])
    off_long_effective, off_lat, min_gap_displacement, min_gap_residual = _enforce_min_distance(
        off_long_effective, off_lat, min_node_gap, margin, min_gap_tolerance, min_gap_max_iterations
    )
    s_nodes = s_centroid[:, None] + off_long_effective

    s_nodes_before_clip = s_nodes
    s_nodes = np.clip(s_nodes, 0.0, track.length)
    if not np.array_equal(s_nodes_before_clip, s_nodes):
        warnings.warn(
            "s_nodes conteneva valori fuori da [0, length] dopo i vincoli di "
            "margine e distanza minima: valori clampati. Con start_offset/"
            "end_margin validati contro sigma_long non dovrebbe accadere; "
            "verificare la configurazione."
        )

    flat_sample = track.query(s_nodes.reshape(-1))
    base_position = flat_sample.position.reshape(n_steps, n_nodes, 2)
    normal = flat_sample.normal.reshape(n_steps, n_nodes, 2)
    positions = base_position + off_lat[:, :, None] * normal

    diff = positions[:, :, None, :] - positions[:, None, :, :]
    distances = np.linalg.norm(diff, axis=-1)
    min_pair_distance = float(np.min(distances + np.eye(n_nodes)[None, :, :] * 1e9))

    min_gap_displacement_mean = float(np.mean(min_gap_displacement))
    min_gap_displacement_p95 = float(np.percentile(min_gap_displacement, 95))
    if min_gap_displacement_p95 > group_cfg["max_displacement_p95"]:
        warnings.warn(
            f"Il vincolo di distanza minima induce uno spostamento p95 di "
            f"{min_gap_displacement_p95:.2f} m, oltre max_displacement_p95="
            f"{group_cfg['max_displacement_p95']} m: min_node_gap potrebbe essere "
            f"troppo grande rispetto a sigma_long ({sigma_long:.3f} m), distorcendo "
            f"la formazione del gruppo invece di limitarsi alla repulsione di contatto."
        )

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
        "min_gap_displacement_mean": min_gap_displacement_mean,
        "min_gap_displacement_p95": min_gap_displacement_p95,
        "min_gap_residual": min_gap_residual,
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
        metadata=metadata,
    )

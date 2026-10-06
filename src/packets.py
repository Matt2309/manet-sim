"""Blocco 3 — Pacchetti e beacon.

Trasforma l'RSSI "perfetto" del Blocco 2 nel registro che ogni scheda
avrebbe davvero: solo i beacon che arrivano, ciascuno con il suo RSSI, e i
buchi dove un beacon si è perso. Convenzione degli indici: ``[i, ..., j]``
= trasmette ``i``, riceve ``j``.

Modello:

1. Ogni nodo trasmette un beacon broadcast ESP-NOW (1 Mbit/s, senza
   conferma né ritrasmissione) ogni ``beacon_period`` s, con fase iniziale
   uniforme e jitter uniforme su ogni intervallo.
2. Collisioni: NON modellate. ESP-NOW usa CSMA/CA; un beacon dura circa 1 ms
   a 1 Mbit/s, quindi il canale è occupato per una frazione
   ``n_nodi * 1 ms / beacon_period`` del tempo (dell'ordine dell'1 % con
   5 nodi e un periodo di 0,5 s): la probabilità che due beacon si
   sovrappongano, già ridotta dall'ascolto del canale, è trascurabile
   rispetto alla perdita di fondo.
3. Un beacon arriva se superano entrambe due prove indipendenti: segnale
   sufficiente (curva logistica in dB sull'RSSI vero, vedi
   `logistic_parameters`) e perdita di fondo (Wi-Fi circostante).
4. Se arriva, la scheda legge `rssi_measured` del Blocco 2, altrimenti NaN.
5. Ogni beacon porta la tabella dei vicini del trasmettitore: per ogni altro
   nodo, l'RSSI dell'ultimo beacon ricevuto e la sua età. Un solo salto.
6. Conoscenza di ogni osservatore `m` sulla griglia del simulatore.

Nessun valore numerico del modello è salvato in questo file: tutti i
parametri vengono letti dalla sezione ``packets`` della configurazione YAML
(vedi ``config/default.yaml``).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
from scipy.special import expit

from src.channel import ChannelResult
from src.mobility import MobilityResult, load_config


# ---------------------------------------------------------------------------
# Strutture dati
# ---------------------------------------------------------------------------


@dataclass
class PacketResult:
    """Output della simulazione dei pacchetti (Blocco 3).

    Gli array ``(N, K, N)`` seguono la convenzione ``[i, k, j]`` = il beacon
    `k` del nodo `i`, visto dal nodo `j`. Gli array ``(T, M, N, N)`` seguono
    ``[t, m, i, j]`` = cosa sa l'osservatore `m` del link `i → j`.
    """

    beacon_times: np.ndarray  # (N, K) s; NaN dove un nodo ha meno beacon
    beacon_steps: np.ndarray  # (N, K) int, indice sulla griglia; -1 come riempimento
    p_success: np.ndarray  # (N, K, N) probabilità di ricezione (curva × fondo)
    passed_signal: np.ndarray  # (N, K, N) bool, prova del segnale superata
    passed_background: np.ndarray  # (N, K, N) bool, prova di fondo superata
    received: np.ndarray  # (N, K, N) bool, diagonale False
    rssi: np.ndarray  # (N, K, N) dBm misurato, NaN se perso
    knowledge_rssi: np.ndarray  # (T, M, N, N) float32, cosa sa m del link i→j
    knowledge_age: np.ndarray  # (T, M, N, N) float32 s, età di quell'informazione
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Curva di ricezione
# ---------------------------------------------------------------------------


def logistic_parameters(sensitivity: float, success: float, width: float) -> tuple:
    """Centro e pendenza della curva logistica di ricezione.

    Input: sensibilità `S` in dBm, probabilità `success` alla sensibilità,
    larghezza `width` in dB fra il 10% e il 90% di ricezione.
    Procedimento: ``p(x) = 1 / (1 + exp(-(x - c) / s))``. Il livello al quale
    la probabilità vale `p` è ``x = c + s ln(p / (1 - p))``. Fra 10% e 90%:
    ``width = s [ln 9 - ln(1/9)] = 2 s ln 9``, quindi ``s = width / (2 ln 9)``.
    Alla sensibilità ``p(S) = success``: ``S = c + s ln(success / (1 - success))``,
    quindi ``c = S - s ln(success / (1 - success))``. Per questo il centro (50%)
    sta sotto la sensibilità: la sensibilità del datasheet è il livello al
    quale l'8% dei pacchetti va perso, non il 50%.
    Output: (c, s) in dB.
    """
    scale = width / (2.0 * math.log(9.0))
    center = sensitivity - scale * math.log(success / (1.0 - success))
    return center, scale


def reception_probability(rssi: np.ndarray, reception_cfg: dict) -> np.ndarray:
    """Probabilità che la prova del segnale sia superata.

    Input: RSSI vero in dBm (qualunque forma) e sezione ``packets.reception``.
    Procedimento: con ``model: curve`` la logistica di `logistic_parameters`;
    con ``model: threshold`` una soglia netta alla sensibilità (1 se
    ``rssi >= sensibilità``, 0 altrimenti), per i confronti.
    Output: probabilità, stessa forma di `rssi`.
    """
    model = reception_cfg["model"]
    x = np.asarray(rssi, dtype=float)
    if model == "curve":
        center, scale = logistic_parameters(
            reception_cfg["sensitivity_dbm"],
            reception_cfg["sensitivity_success"],
            reception_cfg["transition_width"],
        )
        return expit((x - center) / scale)
    if model == "threshold":
        return (x >= reception_cfg["sensitivity_dbm"]).astype(float)
    raise ValueError(f"packets.reception.model deve essere 'curve' o 'threshold', non {model!r}")


# ---------------------------------------------------------------------------
# Istanti dei beacon
# ---------------------------------------------------------------------------


def beacon_schedule(
    n_nodes: int, t_end: float, dt: float, packets_cfg: dict, rng: np.random.Generator
) -> tuple:
    """Istanti di trasmissione dei beacon di tutti i nodi.

    Input: numero di nodi, durata `t_end` in s, passo `dt` della griglia,
    sezione ``packets`` e generatore `rng`.
    Procedimento: la fase iniziale di ogni nodo è uniforme in
    ``[0, period)``; ogni intervallo successivo vale ``period + u`` con `u`
    uniforme in ``[-jitter, +jitter]``. Si estrae un numero di intervalli
    sufficiente a coprire `t_end` anche con tutti gli intervalli minimi, si
    accumula e i beacon oltre `t_end` diventano NaN. L'indice della griglia è
    quello più vicino al tempo di trasmissione (``rint(t / dt)``), -1 dove NaN.
    Output: (times (N, K) in s, steps (N, K) int).
    """
    period = packets_cfg["beacon_period"]
    jitter = packets_cfg["jitter"]
    if not 0.0 <= jitter < period:
        raise ValueError("packets.jitter deve essere in [0, beacon_period)")
    n_beacons = int(math.ceil(t_end / (period - jitter))) + 1
    phase = rng.uniform(0.0, period, size=n_nodes)
    intervals = period + rng.uniform(-jitter, jitter, size=(n_nodes, n_beacons - 1))
    times = phase[:, None] + np.concatenate([np.zeros((n_nodes, 1)), np.cumsum(intervals, axis=1)], axis=1)
    times = np.where(times <= t_end, times, np.nan)
    valid = ~np.isnan(times)
    steps = np.where(valid, np.rint(np.where(valid, times, 0.0) / dt), -1).astype(np.int64)
    return times, steps


# ---------------------------------------------------------------------------
# Conoscenza dei nodi
# ---------------------------------------------------------------------------


def _neighbour_tables(times: np.ndarray, received: np.ndarray, rssi: np.ndarray) -> tuple:
    """Tabella dei vicini portata da ogni beacon.

    Input: istanti (N, K), ricezioni (N, K, N) e RSSI riportati (N, K, N).
    Procedimento: nel beacon `k` del nodo `j`, per ogni altro nodo `i`, si
    prende l'ultimo beacon di `i` ricevuto da `j` prima di `times[j, k]`
    (ricerca binaria sui soli beacon ricevuti). L'età nel beacon è la
    differenza fra due tempi misurati dal solo nodo `j` (nessun orologio
    comune); qui si conserva l'istante d'origine, da cui l'età a ogni
    istante successivo si ricava per sottrazione.
    Output: (table_rssi, table_origin), entrambi (N, K, N) indicizzati
    ``[j, k, i]``; NaN se `j` non aveva ancora ricevuto nulla da `i`.
    """
    n_nodes, n_beacons = times.shape
    table_rssi = np.full((n_nodes, n_beacons, n_nodes), np.nan)
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
            table_rssi[j, valid_k[ok], i] = rssi[i, src, j]
            table_origin[j, valid_k[ok], i] = times[i, src]
    return table_rssi, table_origin


def _knowledge(t: np.ndarray, times: np.ndarray, received: np.ndarray, rssi: np.ndarray) -> tuple:
    """Cosa sa ogni osservatore `m` di ogni link `i → j`, sulla griglia.

    Input: istanti della griglia `t` (T,), istanti dei beacon, ricezioni e
    RSSI riportati.
    Procedimento: un beacon trasmesso a `t_b` è noto all'istante di griglia
    `t` se ``t_b <= t``. Per ``j == m``: ultimo beacon di `i` ricevuto da `m`,
    età ``t - t_b``. Per ``j != m`` (anche ``i == m``): ultimo beacon di `j`
    ricevuto da `m`, e da esso la riga della tabella dei vicini per `i`; età
    ``t - istante d'origine``, quindi include il ritardo di inoltro. Un solo
    salto. NaN se l'informazione non è mai arrivata e sulla diagonale ``i == j``.
    Output: (rssi, età), entrambi (T, M, N, N) float32 indicizzati ``[t, m, i, j]``.
    """
    n_steps = len(t)
    n_nodes = times.shape[0]
    table_rssi, table_origin = _neighbour_tables(times, received, rssi)
    k_rssi = np.full((n_steps, n_nodes, n_nodes, n_nodes), np.nan, dtype=np.float32)
    k_age = np.full_like(k_rssi, np.nan)

    for m in range(n_nodes):
        for j in range(n_nodes):
            if j == m:
                continue
            got = np.flatnonzero(received[j, :, m])
            if len(got) == 0:
                continue
            pos = np.searchsorted(times[j, got], t, side="right") - 1
            ok = pos >= 0
            k = got[pos[ok]]  # beacon di j più recente noto a m, per ogni istante valido
            origin = table_origin[j, k, :]  # (n_ok, i)
            k_rssi[ok, m, :, j] = table_rssi[j, k, :]
            k_age[ok, m, :, j] = t[ok, None] - origin
        for i in range(n_nodes):
            if i == m:
                continue
            got = np.flatnonzero(received[i, :, m])
            if len(got) == 0:
                continue
            pos = np.searchsorted(times[i, got], t, side="right") - 1
            ok = pos >= 0
            k = got[pos[ok]]
            k_rssi[ok, m, i, m] = rssi[i, k, m]
            k_age[ok, m, i, m] = t[ok] - times[i, k]

    idx = np.arange(n_nodes)
    k_rssi[:, :, idx, idx] = np.nan
    k_age[:, :, idx, idx] = np.nan
    return k_rssi, k_age


# ---------------------------------------------------------------------------
# Simulazione dei pacchetti
# ---------------------------------------------------------------------------


def simulate_packets(
    mobility: MobilityResult, channel: ChannelResult, config: Union[str, Path, dict]
) -> PacketResult:
    """Esegue la simulazione dei pacchetti del Blocco 3.

    Input: `MobilityResult`, `ChannelResult` e configurazione (percorso YAML
    o dizionario; sezioni ``simulation`` e ``packets``).
    Procedimento: due generatori indipendenti da quelli del Blocco 2,
    ``SeedSequence(seed, spawn_key=(4,))`` per gli istanti e ``(5,)`` per la
    ricezione. Si generano gli istanti dei beacon; il canale si legge
    all'istante della griglia più vicino. Dal generatore di ricezione si
    estraggono sempre, in quest'ordine, due array uniformi (N, K, N): uno per
    la prova del segnale, uno per la perdita di fondo. La prova del segnale è
    superata se ``u < p_curve(rssi_true)`` (si usa l'RSSI vero, non quello
    arrotondato); quella di fondo se ``u >= background_loss``. Così cambiare
    ``background_loss`` non cambia quali beacon superano la prova del
    segnale, e con la stessa estrazione una sensibilità più bassa dà un
    sovrainsieme di ricezioni. Il beacon arriva se superano entrambe; l'RSSI
    riportato è `rssi_measured` all'indice del beacon, NaN se perso. Infine
    si calcola la conoscenza di ogni nodo.
    Output: `PacketResult`.
    """
    cfg = load_config(config)
    pk = cfg["packets"]
    rc = pk["reception"]
    seed = cfg["simulation"]["seed"]
    dt = cfg["simulation"]["dt"]
    t = channel.t
    n_nodes = channel.rssi_true.shape[1]
    t0 = time.perf_counter()

    rng_times = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(4,)))
    rng_rx = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(5,)))

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
    passed_background = off_diag & (u_background >= rc["background_loss"])
    received = passed_signal & passed_background

    rssi = np.where(received, rssi_meas, np.nan)
    p_success = p_curve * (1.0 - rc["background_loss"])

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
        "lost_background_only_fraction": float((passed_signal & ~passed_background).sum() / n_trials),
        "elapsed": time.perf_counter() - t0,
    }

    return PacketResult(
        beacon_times=times,
        beacon_steps=steps,
        p_success=p_success,
        passed_signal=passed_signal,
        passed_background=passed_background,
        received=received,
        rssi=rssi,
        knowledge_rssi=k_rssi,
        knowledge_age=k_age,
        metadata=metadata,
    )

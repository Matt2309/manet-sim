"""Blocco 3 — Pacchetti e beacon.

Trasforma l'RSSI "perfetto" del Blocco 2 nel registro reale di ogni scheda: solo i
beacon ricevuti, con il loro RSSI. Indici ``[i, ..., j]`` = trasmette ``i``, riceve ``j``.

1. Beacon broadcast ESP-NOW (1 Mbit/s, senza conferma) ogni ``beacon_period`` s,
   fase iniziale uniforme e jitter uniforme su ogni intervallo.
2. Collisioni NON modellate: CSMA/CA, beacon da ~1 ms, occupazione del canale
   ``n_nodi * 1 ms / beacon_period`` (~1 % con 5 nodi, periodo 0,5 s), trascurabile
   rispetto alla perdita di fondo.
3. Arriva se superano due prove indipendenti: segnale (logistica in dB sull'RSSI
   vero, vedi `logistic_parameters`) e perdita di fondo (Wi-Fi circostante).
4. Se arriva la scheda legge `rssi_measured` del Blocco 2, altrimenti NaN.
5. Ogni beacon porta la tabella dei vicini del trasmettitore (ultimo RSSI e età). Un solo salto.
6. Conoscenza di ogni osservatore `m` sulla griglia del simulatore.

Parametri: sezione ``packets`` della config.
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
    """Output del Blocco 3. ``(N, K, N)`` indicizzato ``[i, k, j]`` = beacon `k` di `i` visto da `j`;
    ``(T, M, N, N)`` indicizzato ``[t, m, i, j]`` = cosa sa l'osservatore `m` del link `i → j`.
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
    """Centro `c` e pendenza `s` (dB) della logistica ``p(x) = 1 / (1 + exp(-(x - c) / s))``: `width` dB fra 10% e 90% (``s = width / (2 ln 9)``)
    e ``p(S) = success`` alla sensibilità `S` (``c = S - s ln(success / (1 - success))``). Il centro (50%) sta sotto `S`: la sensibilità di datasheet è al livello con l'8% di perdita.
    """
    scale = width / (2.0 * math.log(9.0))
    center = sensitivity - scale * math.log(success / (1.0 - success))
    return center, scale


def reception_probability(rssi: np.ndarray, reception_cfg: dict) -> np.ndarray:
    """Probabilità di superare la prova del segnale dall'RSSI vero: logistica (``model: curve``) o soglia netta alla sensibilità (``threshold``)."""
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
    """Istanti dei beacon: (times (N, K) in s, steps (N, K) int). Fase iniziale uniforme in ``[0, period)``, intervalli ``period + U[-jitter, +jitter]``;
    oltre `t_end` NaN. `steps` = ``rint(t / dt)``, -1 dove NaN.
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


def _neighbour_tables(times: np.ndarray, received: np.ndarray, payload: dict) -> tuple:
    """Tabella dei vicini portata da ogni beacon. ``payload[nome][i, k, j]`` = valore che il beacon `k` di `i` porta a `j`.
    Per il beacon `k` di `j`, per ogni `i`: l'ultimo beacon di `i` ricevuto da `j` prima di ``times[j, k]``. Ritorna (tabelle, table_origin) indicizzati ``[j, k, i]``;
    `table_origin` = istante d'origine (l'età si ricava per sottrazione: nessun orologio comune). NaN se `j` non ha ancora ricevuto nulla da `i`.
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
    """Cosa sa ogni osservatore `m` del link `i → j`, per un carico qualsiasi (``payload[nome][i, k, j]``, valore portato dal beacon `k` di `i`).
    Un beacon di istante `t_b` è noto a `t` se ``t_b <= t``. Per ``j == m``: ultimo beacon di `i` ricevuto da `m`; per ``j != m``: riga `i` della tabella dei vicini dell'ultimo beacon di `j` (un solo salto).
    Ritorna (campi, age, relay_time): `campi` e `age` (s) sono (T, M, N, N) float32 ``[t, m, i, j]``; `relay_time` (float64) = istante del beacon di `j` che ha inoltrato (NaN se diretto o assente). NaN anche per ``i == j``.
    """
    n_steps = len(t)
    n_nodes = times.shape[0]
    tables, table_origin = _neighbour_tables(times, received, payload)
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
            k = got[pos[ok]]  # beacon di j più recente noto a m, per ogni istante valido
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
    """Caso particolare di `forward_link_payload` con carico = RSSI riportato: ritorna (rssi, età), (T, M, N, N) float32 ``[t, m, i, j]``."""
    fields, age, _ = forward_link_payload(t, times, received, {"rssi": rssi})
    return fields["rssi"], age


# ---------------------------------------------------------------------------
# Simulazione dei pacchetti
# ---------------------------------------------------------------------------


def simulate_packets(
    mobility: MobilityResult, channel: ChannelResult, config: Union[str, Path, dict]
) -> PacketResult:
    """Simulazione dei pacchetti del Blocco 3 (config: percorso YAML o dict con ``simulation`` e ``packets``).
    Generatori indipendenti dal Blocco 2: ``SeedSequence(seed, spawn_key=(4,))`` per gli istanti, ``(5,)`` per la ricezione, da cui si estraggono sempre due array uniformi (N, K, N):
    segnale (``u < p_curve(rssi_true)``, RSSI vero non arrotondato) e fondo (``u >= background_loss``), così variare uno non cambia l'altro. Il beacon arriva se superano entrambe.
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

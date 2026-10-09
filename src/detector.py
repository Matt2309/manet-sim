"""Blocco 4 — Tabella dei vicini, fusione e rilevatore.

Catena su ogni nodo: beacon ricevuto → Kalman del link (`src/kalman.py`) →
tabella dei vicini → fusione per bersaglio → rilevatore → allarme.
Convenzione degli indici: ``[i, ..., j]`` = trasmette `i`, riceve `j`.

Tabella dei vicini (5 byte per vicino): id, RSSI stimato (int8, 1 dB),
pendenza (int8, 0,1 dB/s, ±12,7), deviazione standard (uint8, 0,1 dB), età
(uint8, 0,1 s, satura a 25,5 s). Il beacon di `j` porta, per ogni vicino `i`,
la stima a posteriori del Kalman del link `i → j`; chi riceve aggiunge il
tempo trascorso dalla ricezione (esatto).

Fusione per bersaglio `i` (osservatore `m`): solo i link in cui trasmette `i`.
Semplificazioni, da raffinare:

- la tabella non porta la covarianza completa: la voce ``i → k`` avanza con
  ``r + s·età`` e varianza ``std² + σ_a²·età³/3`` (ignora la correlazione
  livello-pendenza);
- per la pendenza si usano gli stessi pesi ``1/varianza`` dell'RSSI;
- lo shadowing è condiviso fra i link (correlazione ~0,9): la media pesata
  SOTTOSTIMA la varianza vera, quindi le soglie si tarano sui dati (sweep).

Parametri: sezioni ``neighbour_table``, ``fusion``, ``detector`` della config.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.kalman import KalmanResult, filter_all, predict_links
from src.packets import forward_link_payload


# ---------------------------------------------------------------------------
# Tabella dei vicini a 5 byte
# ---------------------------------------------------------------------------

# Limiti dei tipi interi; la pendenza è simmetrica (±127 passi).
_INT8 = (-128, 127)
_SLOPE = (-127, 127)
_UINT8 = (0, 255)


def _to_code(values: np.ndarray, step: float, limits: tuple, dtype) -> np.ndarray:
    return np.clip(np.rint(np.asarray(values, dtype=float) / step), limits[0], limits[1]).astype(dtype)


def encode_table(r, s, std, age, table_cfg: dict) -> dict:
    """Codifica una voce (4 byte, id escluso): arrotonda a passo e satura ai limiti del tipo.

    Campi in unità fisiche (dBm, dB/s, dB, s); restituisce ``r``, ``s`` (int8), ``std``, ``age`` (uint8).
    """
    return {
        "r": _to_code(r, table_cfg["rssi_step"], _INT8, np.int8),
        "s": _to_code(s, table_cfg["slope_step"], _SLOPE, np.int8),
        "std": _to_code(std, table_cfg["std_step"], _UINT8, np.uint8),
        "age": _to_code(age, table_cfg["age_step"], _UINT8, np.uint8),
    }


def decode_table(codes: dict, table_cfg: dict) -> dict:
    """Inverso di `encode_table`: codici → unità fisiche (errore ≤ mezzo passo, salvo saturazione)."""
    return {
        "r": codes["r"].astype(float) * table_cfg["rssi_step"],
        "s": codes["s"].astype(float) * table_cfg["slope_step"],
        "std": codes["std"].astype(float) * table_cfg["std_step"],
        "age": codes["age"].astype(float) * table_cfg["age_step"],
    }


def _quantize(values: np.ndarray, step: float, limits: tuple) -> np.ndarray:
    """Arrotonda e satura come la scheda; i NaN restano NaN."""
    out = np.clip(np.rint(values / step), limits[0], limits[1]) * step
    return np.where(np.isnan(values), np.nan, out)


def neighbour_knowledge(t: np.ndarray, kres: KalmanResult, table_cfg: dict) -> dict:
    """Cosa sa l'osservatore `m` del link `i → j` dalle tabelle dei vicini.

    Carico dei beacon = stima a posteriori quantizzata come sulla scheda; età = età alla trasmissione
    (quantizzata) + tempo dalla ricezione (esatto). Restituisce ``r``, ``s``, ``std``, ``age``
    (T, M, N, N) indicizzati ``[t, m, i, j]``; per ``j == m`` l'età è esatta (stima propria).
    """
    payload = {
        "r": _quantize(kres.r, table_cfg["rssi_step"], _INT8),
        "s": _quantize(kres.s, table_cfg["slope_step"], _SLOPE),
        "std": _quantize(np.sqrt(kres.P[..., 0, 0]), table_cfg["std_step"], _UINT8),
    }
    fields, age, relay = forward_link_payload(t, kres.times, kres.received, payload)
    since_rx = t[:, None, None, None] - relay  # NaN sul link diretto
    age_tx = np.maximum(age.astype(float) - since_rx, 0.0)
    age_q = _quantize(age_tx, table_cfg["age_step"], _UINT8) + since_rx
    fields["age"] = np.where(np.isnan(relay), age.astype(float), age_q)
    return fields


# ---------------------------------------------------------------------------
# Fusione
# ---------------------------------------------------------------------------


def fuse(
    values: np.ndarray,
    slopes: np.ndarray,
    variances: np.ndarray,
    ages: np.ndarray,
    max_age: float,
    min_variance: float = 0.0,
) -> tuple:
    """Media pesata delle voci di un bersaglio (voci sull'ULTIMO asse, `max_age` in s).

    Voce valida: campi finiti ed ``età <= max_age``; peso ``w = 1/max(var, min_variance)``,
    ``L = Σ w r / Σ w``, ``S = Σ w s / Σ w``, varianza ``1/Σ w``.
    Restituisce (L, S, varianza, n voci valide); NaN dove nessuna voce è valida.
    """
    values = np.asarray(values, dtype=float)
    slopes = np.asarray(slopes, dtype=float)
    variances = np.asarray(variances, dtype=float)
    ages = np.asarray(ages, dtype=float)
    valid = (
        np.isfinite(values) & np.isfinite(slopes) & np.isfinite(variances) & np.isfinite(ages) & (ages <= max_age)
    )
    floor = max(min_variance, np.finfo(float).tiny)
    weight = np.where(valid, 1.0 / np.maximum(np.where(valid, variances, 1.0), floor), 0.0)
    total = weight.sum(axis=-1)
    has = total > 0.0
    safe = np.where(has, total, 1.0)
    level = np.where(has, (weight * np.where(valid, values, 0.0)).sum(axis=-1) / safe, np.nan)
    slope = np.where(has, (weight * np.where(valid, slopes, 0.0)).sum(axis=-1) / safe, np.nan)
    variance = np.where(has, 1.0 / safe, np.nan)
    return level, slope, variance, valid.sum(axis=-1)


def fusion_indices(
    t: np.ndarray, kres: KalmanResult, config: dict, sigma_a: float | None = None
) -> dict:
    """Livello e pendenza di ogni bersaglio per ogni osservatore, fusi e non (`sigma_a` default del filtro).

    Voci per ``(m, i)``: stima propria ``i → m`` predetta all'istante corrente (`predict_links`) e,
    per ogni compagno `k`, la voce ``i → k`` dall'ultima tabella di `k` ricevuta da `m`. Escluse
    quelle con età > ``fusion.max_age`` (la propria solo se ``fusion.max_age_own``). Varianza minima
    in tabella ``std_step²/12``. Restituisce ``fused`` (tutte le voci) e ``pairwise`` (solo la propria),
    ciascuno ``(L, S, var, n)`` con array (T, M, I) ``[t, m, i]``; diagonale ``m == i`` NaN.
    """
    fcfg = config["fusion"]
    tcfg = config["neighbour_table"]
    sigma_a = kres.sigma_a if sigma_a is None else sigma_a
    n_nodes = kres.times.shape[0]
    know = neighbour_knowledge(t, kres, tcfg)
    own = predict_links(kres, t)

    age = know["age"]
    value = know["r"].astype(float) + know["s"].astype(float) * age
    slope = know["s"].astype(float)
    var = know["std"].astype(float) ** 2 + sigma_a**2 * age**3 / 3.0
    mask_age = age.copy()

    own_age = own["age"] if fcfg["max_age_own"] else np.where(np.isnan(own["age"]), np.nan, 0.0)
    for m in range(n_nodes):  # voce propria: indice k = m
        value[:, m, :, m] = own["r"][:, :, m]
        slope[:, m, :, m] = own["s"][:, :, m]
        var[:, m, :, m] = own["p00"][:, :, m]
        mask_age[:, m, :, m] = own_age[:, :, m]

    min_var = tcfg["std_step"] ** 2 / 12.0
    fused = fuse(value, slope, var, mask_age, fcfg["max_age"], min_var)

    idx = np.arange(n_nodes)
    own_entry = lambda a: a[:, idx, :, idx].transpose(1, 0, 2)[..., None]  # (T, M, I, 1)
    pair = fuse(
        own_entry(value), own_entry(slope), own_entry(var), own_entry(mask_age), fcfg["max_age"], min_var
    )
    out = {"fused": fused, "pairwise": pair}
    for name, parts in out.items():
        for a in parts:
            a[:, idx, idx] = np.nan if a.dtype.kind == "f" else 0
    return out


# ---------------------------------------------------------------------------
# Ricezione diretta
# ---------------------------------------------------------------------------


def reception_features(t: np.ndarray, times: np.ndarray, received: np.ndarray, config: dict) -> tuple:
    """Silenzio e ricezione recente di ogni link diretto ``i → m`` (`times` (N, K), `received` (N, K, N)).

    silence = tempo dall'ultima ricezione (da 0 se nessuna); ``rx_ok`` = beacon ricevuti in
    ``(t − release_time, t]`` ≥ ``release_reception · release_time / beacon_period`` (periodo nominale).
    Restituisce (silence (s), rx_ok) (T, M, I) ``[t, m, i]``; diagonale 0 e False.
    """
    dcfg = config["detector"]
    window = dcfg["release_time"]
    needed = dcfg["release_reception"] * window / config["packets"]["beacon_period"]
    n_nodes = times.shape[0]
    silence = np.zeros((len(t), n_nodes, n_nodes))
    rx_ok = np.zeros((len(t), n_nodes, n_nodes), dtype=bool)
    for i in range(n_nodes):
        for m in range(n_nodes):
            if i == m:
                continue
            rx_times = times[i, received[i, :, m]]
            count_now = np.searchsorted(rx_times, t, side="right")
            count_before = np.searchsorted(rx_times, t - window, side="right")
            last = np.where(count_now > 0, rx_times[np.maximum(count_now - 1, 0)] if len(rx_times) else 0.0, 0.0)
            silence[:, m, i] = t - last
            rx_ok[:, m, i] = (count_now - count_before) >= needed - 1e-9
    return silence, rx_ok


# ---------------------------------------------------------------------------
# Allarmi
# ---------------------------------------------------------------------------

_TIME_TOLERANCE = 1e-9  # s; errore di arrotondamento sui confronti fra durate


def run_duration(cond: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Durata in s da cui `cond` (tempo sul primo asse) è vera senza interruzioni; 0 se falsa o al primo campione vero."""
    shape = (-1,) + (1,) * (cond.ndim - 1)
    idx = np.arange(len(t)).reshape(shape)
    last_false = np.maximum.accumulate(np.where(cond, -1, idx), axis=0)
    start = t[np.minimum(last_false + 1, len(t) - 1)]
    return np.where(cond, t.reshape(shape) - start, 0.0)


def latched_alarm(on: np.ndarray, off: np.ndarray) -> np.ndarray:
    """Stato con isteresi (tempo sul primo asse): acceso se l'ultimo `on` è più recente dell'ultimo `off`.

    A pari indice vince lo spegnimento; parte spento.
    """
    idx = np.arange(on.shape[0]).reshape((-1,) + (1,) * (on.ndim - 1))
    last_on = np.maximum.accumulate(np.where(on, idx, -1), axis=0)
    last_off = np.maximum.accumulate(np.where(off, idx, -1), axis=0)
    return last_on > last_off


def pre_alarm(
    level: np.ndarray,
    slope: np.ndarray,
    rx_ok: np.ndarray,
    t: np.ndarray,
    detector_cfg: dict,
    mode: str | None = None,
    level_threshold: float | None = None,
    slope_threshold: float | None = None,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Pre-allarme di ogni coppia (osservatore, bersaglio); `mode` e soglie sostituibili per gli sweep.

    Condizione: ``level`` ``L < soglia``, ``slope`` ``S < soglia``, ``both`` entrambe (NaN = falso).
    Si accende dopo ``min_duration``; si spegne dopo ``release_time`` di rientro con margine
    (``release_*_margin``; in ``both`` solo sul livello) e con `rx_ok`. Soglie più larghe ⇒ allarmi
    in numero monotono crescente (vedi `src/metrics.py`). `valid` esclude ``m == i``.
    """
    mode = detector_cfg["mode"] if mode is None else mode
    thr_l = detector_cfg["level_threshold"] if level_threshold is None else level_threshold
    thr_s = detector_cfg["slope_threshold"] if slope_threshold is None else slope_threshold
    with np.errstate(invalid="ignore"):
        low_l = level < thr_l
        low_s = slope < thr_s
        back_l = level > thr_l + detector_cfg["release_level_margin"]
        back_s = slope > thr_s + detector_cfg["release_slope_margin"]
    if mode == "level":
        cond, back = low_l, back_l
    elif mode == "slope":
        cond, back = low_s, back_s
    elif mode == "both":
        cond, back = low_l & low_s, back_l
    else:
        raise ValueError(f"detector.mode deve essere 'level', 'slope' o 'both', non {mode!r}")
    on = run_duration(cond, t) >= detector_cfg["min_duration"] - _TIME_TOLERANCE
    off = (run_duration(back, t) >= detector_cfg["release_time"] - _TIME_TOLERANCE) & rx_ok
    state = latched_alarm(on, off)
    return state if valid is None else state & valid


def lost_alarm(
    silence: np.ndarray,
    rx_ok: np.ndarray,
    t: np.ndarray,
    detector_cfg: dict,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Allarme "nodo perso" di ogni coppia: acceso con silenzio > ``lost_silence`` s, spento da `rx_ok`.

    Un beacon isolato non lo spegne. `valid` esclude ``m == i``.
    """
    on = silence > detector_cfg["lost_silence"]
    state = latched_alarm(on, rx_ok)
    return state if valid is None else state & valid


def system_alarm(observer_alarm: np.ndarray, min_observers: int) -> np.ndarray:
    """Allarme di sistema (ciò che arriva al capogruppo): osservatori in allarme ≥ `min_observers`.

    Ingresso (tempo, osservatore, ...); uscita (tempo, ...).
    """
    return observer_alarm.sum(axis=1) >= min_observers


def count_episodes(system: np.ndarray, t: np.ndarray, warmup: float) -> np.ndarray:
    """Numero di fronti di salita dell'allarme di sistema con ``t[k] >= warmup`` (s), per asse non temporale.

    Un allarme già acceso al termine del `warmup` non è un episodio.
    """
    edges = system[1:] & ~system[:-1]
    first = int(np.searchsorted(t, warmup))
    return edges[max(first - 1, 0) :].sum(axis=0)


# ---------------------------------------------------------------------------
# Ingressi del rilevatore per una corsa
# ---------------------------------------------------------------------------


@dataclass
class DetectionInputs:
    """Tutto ciò che il rilevatore usa di una corsa, indipendente dalle soglie.

    Gli array hanno il tempo sul primo asse e, per l'intera corsa,
    ``[t, m, i]`` (osservatore, bersaglio); `target_slice` ne ritaglia uno solo.
    """

    t: np.ndarray
    level: dict  # scope -> (T, M, I)
    slope: dict
    n_entries: dict
    silence: np.ndarray
    rx_ok: np.ndarray
    valid: np.ndarray  # (M, I) False sulla diagonale

    def target_slice(self, target: int) -> "DetectionInputs":
        sel = (slice(None), slice(None), target)
        return DetectionInputs(
            t=self.t,
            level={k: v[sel] for k, v in self.level.items()},
            slope={k: v[sel] for k, v in self.slope.items()},
            n_entries={k: v[sel] for k, v in self.n_entries.items()},
            silence=self.silence[sel],
            rx_ok=self.rx_ok[sel],
            valid=self.valid[:, target],
        )


def build_inputs(run: dict, config: dict, sigma_a: float, R: float) -> tuple:
    """`filter_all` → `fusion_indices` → `reception_features` su una corsa (chiavi come `PacketResult`).

    Restituisce (`DetectionInputs`, `KalmanResult`).
    """
    kres = filter_all(run["beacon_times"], run["received"], run["rssi"], R, sigma_a, config["kalman"])
    t = run["t"]
    indices = fusion_indices(t, kres, config, sigma_a)
    silence, rx_ok = reception_features(t, run["beacon_times"], run["received"], config)
    n_nodes = kres.times.shape[0]
    inputs = DetectionInputs(
        t=t,
        level={k: v[0] for k, v in indices.items()},
        slope={k: v[1] for k, v in indices.items()},
        n_entries={k: v[3] for k, v in indices.items()},
        silence=silence,
        rx_ok=rx_ok,
        valid=~np.eye(n_nodes, dtype=bool),
    )
    return inputs, kres

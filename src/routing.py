"""Blocco 7 — Routing verso il capogruppo: ponte, DQREM, carry.

Chi non sente più il capogruppo fa passare record di telemetria, allarmi e SOS di salto in salto attraverso i compagni. Indici ``[i, ..., j]`` = trasmette
`i`, riceve `j`; ``[t, m, j]`` = nodo `m` che guarda il vicino `j` all'istante `t` della griglia.

Pezzi (ognuno testabile da solo):

- salti annunciati nel beacon (`announce_hops`) e stato dei collegamenti visto da ogni nodo (`link_state`);
- criteri C1-C5, pesi AHP e da entropia, punteggio (`ahp_weights`, `entropy_weights`, `time_to_break`, `dqrem_scores`, `Chooser`);
- unicast con conferma (`unicast_attempts`), pacchetti (`select_packet`), code, buffer e carry (`simulate_strategy`);
- preparazione di una corsa dai Blocchi 1-3 e dal Kalman del Blocco 4 (`prepare_run`).

Il rilevatore NON si valuta qui: gli allarmi da trasportare vengono dalla ground truth (allarme oracolo) e dal SOS del separato.
Collisioni non modellate; si conta il tempo d'aria. Parametri: sezione ``routing`` della config.
"""

from __future__ import annotations

import math
from bisect import insort
from collections import defaultdict, deque

import numpy as np

from src.channel import simulate_channel
from src.detector import _INT8, _SLOPE, _quantize
from src.kalman import filter_all, measurement_noise, predict_links
from src.metrics import min_distance_to_group
from src.mobility import simulate_mobility
from src.packets import _neighbour_tables, reception_probability, simulate_packets

ISOLATED = 255  # salti annunciati da un nodo isolato
STRATEGIES = ("direct", "best_rssi", "dqrem", "dqrem_no_slope")
CRITERIA = ("segnale", "tempo_rottura", "salti", "ricezione", "coda")
BENEFIT = np.array([True, True, False, True, False])  # C3 (salti) e C5 (coda) sono costi
NO_SLOPE_COLUMNS = (0, 2, 3, 4)  # `dqrem_no_slope`: senza C2
KIND_NORMAL, KIND_ALARM, KIND_SOS = 0, 1, 2
ALARM_SEQ_BASE, SOS_SEQ_BASE = 1_000_000, 2_000_000  # numeri progressivi dei messaggi di allarme e SOS (non collidono con i record)

# indici dei campi di un record: [t_creazione, origine, progressivo, tipo, ttl, salti, carry, t_in_coda]
R_T, R_ORIGIN, R_SEQ, R_KIND, R_TTL, R_HOPS, R_CARRY, R_ENQ = range(8)


# ---------------------------------------------------------------------------
# Dimensioni e tempo d'aria
# ---------------------------------------------------------------------------


def record_bytes(n_nodes: int) -> int:
    """Byte di un record di telemetria: come il beacon, ``8 + 5·d`` con ``d = n_nodes − 1`` vicini in tabella (53 con 10 nodi)."""
    return 8 + 5 * (n_nodes - 1)


def records_per_packet(rcfg: dict, n_nodes: int, max_packet_bytes: int | None = None) -> int:
    """Record che entrano in un pacchetto: ``min(max_records, (max_packet_bytes − intestazione) // byte per record)`` (4 con 250 B e 10 nodi, 10 con 1470 B)."""
    limit = rcfg["max_packet_bytes"] if max_packet_bytes is None else max_packet_bytes
    fit = (limit - rcfg["packet_header_bytes"]) // record_bytes(n_nodes)
    return int(max(1, min(rcfg["max_records"], fit)))


def airtime(n_bytes: int, rcfg: dict) -> float:
    """Tempo d'aria in s di una trama: preambolo + byte (802.11b a 1 Mbit/s: 192 µs + 8 µs per byte)."""
    a = rcfg["airtime"]
    return (a["preamble_us"] + a["us_per_byte"] * n_bytes) * 1e-6


# ---------------------------------------------------------------------------
# Pesi, criteri e punteggio DQREM
# ---------------------------------------------------------------------------


def ahp_weights(matrix, random_index: dict) -> tuple:
    """Pesi AHP e rapporto di consistenza dalla matrice dei confronti a coppie (scala di Saaty).

    Input: `matrix` (n, n) e `random_index` (indice casuale di Saaty per n).
    Procedimento: autovettore principale (autovalore reale massimo) normalizzato a somma 1; ``CI = (λmax − n)/(n − 1)``, ``CR = CI / RI``.
    Output: (pesi (n,), CR).
    """
    a = np.asarray(matrix, dtype=float)
    n = a.shape[0]
    values, vectors = np.linalg.eig(a)
    k = int(np.argmax(values.real))
    w = np.abs(vectors[:, k].real)
    w = w / w.sum()
    ci = (float(values[k].real) - n) / (n - 1)
    return w, ci / float(random_index[n])


def entropy_weights(samples: np.ndarray, benefit: np.ndarray) -> np.ndarray:
    """Pesi da entropia di Shannon dei criteri sui campioni raccolti.

    Input: `samples` (n, c) valori dei criteri (una riga per candidato visto), `benefit` (c,) bool.
    Procedimento: per colonna, normalizzazione min-max sulle righe (costi invertiti); proporzioni ``p = x / Σx``; ``E = −Σ p ln p / ln n``; peso ∝ ``1 − E``.
    Una colonna che non varia (o con meno di 2 righe) ha peso 0.
    Output: (c,) con somma 1, oppure tutti zeri se nessun criterio varia.
    """
    x = np.asarray(samples, dtype=float)
    n, c = x.shape
    w = np.zeros(c)
    if n < 2:
        return w
    lo, hi = x.min(axis=0), x.max(axis=0)
    for k in range(c):
        if hi[k] - lo[k] <= 0.0:
            continue
        v = (x[:, k] - lo[k]) / (hi[k] - lo[k])
        if not benefit[k]:
            v = 1.0 - v
        total = v.sum()
        if total <= 0.0:
            continue
        p = v / total
        nz = p > 0
        entropy = -float((p[nz] * np.log(p[nz])).sum()) / math.log(n)
        w[k] = max(1.0 - entropy, 0.0)
    s = w.sum()
    return w / s if s > 0 else w


def combine_weights(w_ahp: np.ndarray, w_entropy: np.ndarray, alpha: float) -> np.ndarray:
    """``w = α·w_AHP + (1 − α)·w_entropia``, rinormalizzato a somma 1."""
    w = alpha * np.asarray(w_ahp) + (1.0 - alpha) * np.asarray(w_entropy)
    return w / w.sum()


def time_to_break(signal, slope, threshold: float, cap: float):
    """Tempo alla rottura C2 in s: ``(segnale − soglia) / (−pendenza)`` se la pendenza è negativa, altrimenti `cap`; limitato a ``[0, cap]``."""
    signal = np.asarray(signal, dtype=float)
    slope = np.asarray(slope, dtype=float)
    falling = slope < 0
    with np.errstate(divide="ignore", invalid="ignore"):
        ttb = np.where(falling, (signal - threshold) / np.where(falling, -slope, 1.0), cap)
    return np.clip(np.nan_to_num(ttb, nan=cap), 0.0, cap)


def dqrem_scores(x: np.ndarray, weights: np.ndarray, benefit: np.ndarray) -> np.ndarray:
    """Punteggio DQREM dei candidati: ogni criterio normalizzato min-max sui candidati del momento (costi invertiti; criterio uguale per tutti, o un solo candidato → 1), poi somma pesata.

    Input: `x` (n, c) criteri dei candidati, `weights` (c,), `benefit` (c,) bool. Output: (n,) punteggi (più alto = meglio).
    """
    x = np.asarray(x, dtype=float)
    lo, hi = x.min(axis=0), x.max(axis=0)
    span = hi - lo
    norm = np.ones_like(x)
    var = span > 0
    norm[:, var] = (x[:, var] - lo[var]) / span[var]
    cost = var & ~benefit
    norm[:, cost] = 1.0 - norm[:, cost]
    return norm @ weights


class Chooser:
    """Sceglie il prossimo salto fra i candidati con la strategia `strategy` (`best_rssi`, `dqrem`, `dqrem_no_slope`).

    Input: `rcfg` (sezione ``routing``), numero di nodi. Stato: per nodo, i criteri dei candidati visti negli ultimi ``entropy_window`` s.
    `dqrem_no_slope` toglie C2 e ricalcola i pesi AHP sulla sottomatrice 4×4 (stessi confronti a coppie), entropia sui 4 criteri.
    """

    def __init__(self, strategy: str, rcfg: dict, n_nodes: int):
        self.strategy = strategy
        self.rcfg = rcfg
        self.columns = list(NO_SLOPE_COLUMNS) if strategy == "dqrem_no_slope" else list(range(len(CRITERIA)))
        matrix = np.asarray(rcfg["ahp_matrix"], dtype=float)[np.ix_(self.columns, self.columns)]
        self.w_ahp, self.cr = ahp_weights(matrix, rcfg["ahp_random_index"])
        self.benefit = BENEFIT[self.columns]
        self.samples = [deque() for _ in range(n_nodes)]
        self.last_weights = None

    def weights(self, node: int) -> np.ndarray:
        """Pesi finali del nodo: solo AHP con meno di ``entropy_min_samples`` campioni, altrimenti ``α·AHP + (1 − α)·entropia``."""
        rows = np.vstack([x for _, x in self.samples[node]])
        if len(rows) < self.rcfg["entropy_min_samples"]:
            return self.w_ahp
        return combine_weights(self.w_ahp, entropy_weights(rows, self.benefit), self.rcfg["alpha"])

    def rank(self, node: int, t_now: float, criteria: np.ndarray, hops: np.ndarray) -> np.ndarray:
        """Indici dei candidati dal migliore al peggiore. `criteria` (n, 5) è ``[C1..C5]``, `hops` (n,) i salti annunciati (spareggio di `best_rssi`)."""
        if self.strategy == "best_rssi":
            return np.lexsort((hops, -criteria[:, 0]))
        x = criteria[:, self.columns]
        window = self.samples[node]
        window.append((t_now, x))
        while window[0][0] < t_now - self.rcfg["entropy_window"]:
            window.popleft()
        w = self.weights(node)
        self.last_weights = w
        return np.argsort(-dqrem_scores(x, w, self.benefit), kind="stable")


# ---------------------------------------------------------------------------
# Salti annunciati e stato dei collegamenti
# ---------------------------------------------------------------------------


def announce_hops(times: np.ndarray, steps: np.ndarray, received: np.ndarray, est_r: np.ndarray, leader: int, rcfg: dict) -> np.ndarray:
    """Distanza in salti dal capogruppo che ogni nodo annuncia a ogni beacon.

    Input: `times`, `steps` (N, K) (istanti e indici di griglia dei beacon, NaN / −1 oltre la fine), `received` (N, K, N), `est_r` (T, N, N) ``[t, i, j]`` stima di `j` sul
    link ``i → j`` (NaN se nessuna), `leader`, `rcfg`.
    Procedimento: si percorrono i beacon in ordine di tempo. Il capogruppo annuncia 0; ogni altro nodo ``1 + min`` dei salti annunciati dai suoi vicini validi (beacon ricevuto da
    meno di ``neighbour_max_age`` s e segnale stimato ≥ ``candidate_threshold``). Senza vicini validi, o oltre ``max_hops``, annuncia 255 (isolato).
    Output: (N, K) int16, 255 dove il beacon non esiste.
    """
    n_nodes, n_beacons = times.shape
    max_age, threshold, max_hops = rcfg["neighbour_max_age"], rcfg["candidate_threshold"], rcfg["max_hops"]
    ann = np.full((n_nodes, n_beacons), ISOLATED, dtype=np.int16)
    last_hops = np.full((n_nodes, n_nodes), ISOLATED, dtype=np.int16)
    last_rx = np.full((n_nodes, n_nodes), -np.inf)
    nodes, ks = np.nonzero(~np.isnan(times))
    for idx in np.argsort(times[nodes, ks], kind="stable"):
        i, k = int(nodes[idx]), int(ks[idx])
        tb = times[i, k]
        if i == leader:
            value = 0
        else:
            sig = est_r[steps[i, k], :, i]
            ok = (tb - last_rx[i] <= max_age) & (sig >= threshold)
            value = int(1 + last_hops[i, ok].min()) if ok.any() else ISOLATED
            if value > max_hops:
                value = ISOLATED
        ann[i, k] = value
        got = np.flatnonzero(received[i, k])
        last_hops[got, i] = value
        last_rx[got, i] = tb
    return ann


def announced_grid(t: np.ndarray, times: np.ndarray, ann: np.ndarray) -> np.ndarray:
    """Salti correnti di ogni nodo sulla griglia: valore dell'ultimo beacon proprio ``<= t``, 255 prima del primo. Output (T, N) int16."""
    n_nodes = times.shape[0]
    out = np.full((len(t), n_nodes), ISOLATED, dtype=np.int16)
    for j in range(n_nodes):
        ks = np.flatnonzero(~np.isnan(times[j]))
        pos = np.searchsorted(times[j, ks], t, side="right") - 1
        out[:, j] = np.where(pos >= 0, ann[j, ks[np.maximum(pos, 0)]], ISOLATED)
    return out


def link_state(
    t: np.ndarray, times: np.ndarray, received: np.ndarray, est_r: np.ndarray, est_s: np.ndarray, pay_r: np.ndarray, pay_s: np.ndarray, rcfg: dict, beacon_period: float
) -> dict:
    """Che cosa sa il nodo `m` del vicino `j` a ogni istante, per scegliere il prossimo salto.

    Input: griglia `t` (T,), beacon (`times` (N, K), `received` (N, K, N)), stime del Kalman sulla griglia `est_r`, `est_s` (T, N, N) ``[t, i, j]`` (link ``i → j`` visto da `j`),
    carico dei beacon `pay_r`, `pay_s` (N, K, N) ``[i, k, j]`` (stima di ``i → j`` dopo il beacon `k` di `i`, già quantizzata come nella tabella dei vicini), `rcfg`, periodo dei beacon.
    Procedimento: per ogni coppia ``(m, j)``: ``lastk`` = ultimo beacon di `j` ricevuto da `m`; ``valid`` se ha meno di ``neighbour_max_age`` s e il segnale stimato ``j → m`` è
    ≥ ``candidate_threshold``; C1 e pendenza di ``m → j`` dalla voce di `m` nella tabella di quel beacon (``r + s·età``), se più vecchia di ``neighbour_max_age`` s o assente dalla
    stima propria di ``j → m`` (reciprocità); C4 = beacon di `j` ricevuti da `m` in ``reception_window`` s / attesi.
    Output: dizionario di array (T, N, N) ``[t, m, j]``: ``lastk`` int32 (−1 se nessuno), ``valid`` bool, ``c1`` e ``slope`` float32, ``c4`` float32; più ``any_valid`` (T, N).
    """
    n_nodes = times.shape[0]
    n_steps = len(t)
    max_age, threshold, window = rcfg["neighbour_max_age"], rcfg["candidate_threshold"], rcfg["reception_window"]
    tables, origin = _neighbour_tables(times, received, {"r": pay_r, "s": pay_s})
    shape = (n_steps, n_nodes, n_nodes)
    lastk = np.full(shape, -1, dtype=np.int32)
    valid = np.zeros(shape, dtype=bool)
    c1 = np.full(shape, np.nan, dtype=np.float32)
    slope = np.full(shape, np.nan, dtype=np.float32)
    c4 = np.zeros(shape, dtype=np.float32)
    expected = window / beacon_period
    for m in range(n_nodes):
        for j in range(n_nodes):
            if j == m:
                continue
            got = np.flatnonzero(received[j, :, m])
            if len(got) == 0:
                continue
            rx = times[j, got]
            pos = np.searchsorted(rx, t, side="right") - 1
            has = pos >= 0
            safe = np.maximum(pos, 0)
            kk = got[safe]
            sig = est_r[:, j, m]  # segnale ``j → m`` stimato da `m`
            valid[:, m, j] = has & (t - rx[safe] <= max_age) & (sig >= threshold)
            lastk[:, m, j] = np.where(has, kk, -1)
            org = origin[j, kk, m]
            tab_age = t - org
            use = has & np.isfinite(org) & (tab_age <= max_age)
            c1[:, m, j] = np.where(use, tables["r"][j, kk, m] + tables["s"][j, kk, m] * np.where(use, tab_age, 0.0), sig)
            slope[:, m, j] = np.where(use, tables["s"][j, kk, m], est_s[:, j, m])
            count = np.searchsorted(rx, t, side="right") - np.searchsorted(rx, t - window, side="right")
            c4[:, m, j] = np.minimum(1.0, count / expected)
    return {"lastk": lastk, "valid": valid, "c1": c1, "slope": slope, "c4": c4, "any_valid": valid.any(axis=2)}


# ---------------------------------------------------------------------------
# Unicast, pacchetti
# ---------------------------------------------------------------------------


class UniformStream:
    """Uniformi (0, 1) da un generatore, a blocchi (stessa sequenza dei singoli ``random()``)."""

    def __init__(self, rng: np.random.Generator, chunk: int = 1 << 16):
        self.rng, self.chunk = rng, chunk
        self.buf, self.pos = rng.random(chunk).tolist(), 0

    def next(self) -> float:
        if self.pos >= len(self.buf):
            self.buf, self.pos = self.rng.random(self.chunk).tolist(), 0
        self.pos += 1
        return self.buf[self.pos - 1]


def unicast_attempts(p_forward: float, p_return: float, max_retries: int, stream: UniformStream) -> tuple:
    """Unicast con conferma: ogni tentativo riesce se il pacchetto arriva (probabilità `p_forward`) **e** la conferma torna (`p_return`), con estrazioni nuove a ogni tentativo.

    Input: probabilità di ricezione dell'andata e del ritorno, `max_retries` ritrasmissioni, flusso di uniformi (due estrazioni per tentativo).
    Output: (riuscito, tentativi fatti ≤ ``1 + max_retries``, quante volte il pacchetto è arrivato). Se il pacchetto arriva ma la conferma no, il mittente ritrasmette.
    """
    arrivals = 0
    for attempt in range(1, max_retries + 2):
        arrived = stream.next() < p_forward
        confirmed = stream.next() < p_return
        arrivals += arrived
        if arrived and confirmed:
            return True, attempt, arrivals
    return False, max_retries + 1, arrivals


def select_packet(queue: list, capacity: int, n_priority: int) -> list:
    """Record del prossimo pacchetto (senza toglierli dalla coda): prima gli allarmi e gli SOS (coda prioritaria), poi i più vecchi, fino a `capacity`.

    `queue` è ordinata per istante di creazione; `n_priority` = quanti record prioritari contiene.
    """
    if n_priority == 0:
        return queue[:capacity]
    chosen = [r for r in queue if r[R_KIND] != KIND_NORMAL][:capacity]
    if len(chosen) < capacity:
        chosen += [r for r in queue if r[R_KIND] == KIND_NORMAL][: capacity - len(chosen)]
    return chosen


# ---------------------------------------------------------------------------
# Simulazione di una strategia
# ---------------------------------------------------------------------------


def simulate_strategy(rd: dict, strategy: str, rcfg: dict, max_packet_bytes: int | None = None, trace: bool = False) -> dict:
    """Trasporto dei record verso il capogruppo con la strategia `strategy`, sulla griglia di simulazione.

    Input: `rd` (corsa preparata: `prepare_run` o sintetica, vedi sotto), `strategy` in `STRATEGIES`, `rcfg`, `max_packet_bytes` (default ``routing.max_packet_bytes``), `trace`.
    `rd` contiene: ``dt``, ``t``, ``times``, ``steps``, ``ann``, ``ann_grid``, i campi di `link_state`, ``p`` (T, N, N) probabilità di ricezione del link ``i → j``, ``leader``,
    ``events`` (lista ``(passo, nodo, tipo)`` di allarmi e SOS), ``seed`` (della corsa), ``sep_node`` (−1 se nessuno).
    Procedimento: a ogni passo ogni nodo crea 1 record/s (``record_period``; a buffer pieno solo a ``degraded_period``, sovrascrivendo i più vecchi non-SOS); trasmette se ha un
    prioritario, un pacchetto pieno o il primo record in coda da ``max_wait`` s. Candidati = vicini validi con salti strettamente minori dei propri; `direct` solo il capogruppo.
    Un tentativo = unicast con conferma (`unicast_attempts`), fino a ``max_retries`` ritrasmissioni al primo candidato, poi una prova sul secondo; se fallisce anche quello i record
    restano in coda (carry). Il ricevente inserisce i record nella propria coda (scarta quelli già visti; TTL ``max_hops``); il capogruppo scarta i duplicati (origine, progressivo).
    Un nodo non supera ``dt`` s di tempo d'aria per passo. Il generatore degli unicast è ``SeedSequence(seme della corsa, spawn_key=(seed_keys.unicast,))``.
    Output: dizionario con il registro del capogruppo (``log``: liste origine, progressivo, t_creazione, t_arrivo, salti, tipo, carry), i record creati (``created``), contatori
    (``breaks``, ``break_episodes``, ``fail_both``, ``bridge_changes``, ``ttl_drops``, ``duplicates``, ``max_packet_bytes_sent``) e per nodo ``tx_airtime``, ``attempts``,
    ``max_occupancy``, ``lost_buffer``; con `trace` anche ``decisions`` del nodo separato.
    """
    n_nodes = rd["ann_grid"].shape[1]
    n_steps, dt, leader = len(rd["t"]), rd["dt"], rd["leader"]
    capacity = records_per_packet(rcfg, n_nodes, max_packet_bytes)
    rec_bytes = record_bytes(n_nodes)
    buffer_size, max_hops, max_wait, retries = rcfg["buffer_records"], rcfg["max_hops"], rcfg["max_wait"], rcfg["max_retries"]
    rec_step = int(round(rcfg["record_period"] / dt))
    deg_step = int(round(rcfg["degraded_period"] / dt))
    ack_air = airtime(rcfg["airtime"]["ack_bytes"], rcfg)
    p, valid, any_valid, lastk = rd["p"], rd["valid"], rd["any_valid"], rd["lastk"]
    c1, slope, c4, ann, ann_grid = rd["c1"], rd["slope"], rd["c4"], rd["ann"], rd["ann_grid"]
    fill = np.zeros(ann.shape, dtype=np.int16)  # riempimento della coda annunciato in ogni beacon (4 bit)
    stream = UniformStream(np.random.default_rng(np.random.SeedSequence(rd["seed"], spawn_key=(rcfg["seed_keys"]["unicast"],))))
    chooser = None if strategy in ("direct", "best_rssi") else Chooser(strategy, rcfg, n_nodes)
    ranker = Chooser("best_rssi", rcfg, n_nodes) if strategy == "best_rssi" else chooser

    beacons_at = defaultdict(list)
    for j in range(n_nodes):
        for k in np.flatnonzero(rd["steps"][j] >= 0):
            beacons_at[int(rd["steps"][j, k])].append((j, int(k)))
    events = defaultdict(list)
    for step, node, kind in rd["events"]:
        events[int(step)].append((int(node), int(kind)))

    queues = [[] for _ in range(n_nodes)]
    n_pri = [0] * n_nodes
    seen = [set() for _ in range(n_nodes)]
    wait_since = [None] * n_nodes
    stuck_last = [-math.inf] * n_nodes
    seq = [0] * n_nodes
    last_primary = [-1] * n_nodes
    last_failed = [False] * n_nodes
    tx_airtime = np.zeros(n_nodes)
    attempts_count = np.zeros(n_nodes, dtype=np.int64)
    max_occ = np.zeros(n_nodes, dtype=np.int64)
    lost_buffer = np.zeros(n_nodes, dtype=np.int64)
    counters = dict(breaks=0, break_episodes=0, fail_both=0, bridge_changes=0, ttl_drops=0, duplicates=0, max_packet_bytes_sent=0, cycles=0)
    log = {k: [] for k in ("origin", "seq", "t_create", "t_arrive", "hops", "kind", "carry")}
    leader_seen = set()
    created = []
    decisions = []
    break_times = []
    alarm_counter = [0, 0]

    def insert(m: int, rec: list, t_now: float) -> None:
        """Mette `rec` nella coda di `m` (ordinata per creazione); a buffer pieno sovrascrive il più vecchio non-SOS, mai un SOS (se sono tutti SOS scarta il nuovo)."""
        q = queues[m]
        if len(q) >= buffer_size:
            for idx, old in enumerate(q):
                if old[R_KIND] != KIND_SOS:
                    if old[R_KIND] != KIND_NORMAL:
                        n_pri[m] -= 1
                    del q[idx]
                    lost_buffer[m] += 1
                    break
            else:
                lost_buffer[m] += 1
                return
        rec[R_ENQ] = t_now
        insort(q, rec)
        if rec[R_KIND] != KIND_NORMAL:
            n_pri[m] += 1
        if wait_since[m] is None:
            wait_since[m] = t_now
        if len(q) > max_occ[m]:
            max_occ[m] = len(q)

    def deliver_to_leader(rec: list, t_now: float) -> None:
        key = (rec[R_ORIGIN], rec[R_SEQ])
        if key in leader_seen:
            counters["duplicates"] += 1
            return
        leader_seen.add(key)
        for name, value in zip(log, (rec[R_ORIGIN], rec[R_SEQ], rec[R_T], t_now, rec[R_HOPS], rec[R_KIND], rec[R_CARRY])):
            log[name].append(value)

    def create(m: int, kind: int, t_now: float, step: int) -> None:
        if kind == KIND_NORMAL:
            s = seq[m]
            seq[m] += 1
        else:
            base = ALARM_SEQ_BASE if kind == KIND_ALARM else SOS_SEQ_BASE
            s = base + alarm_counter[kind - 1]
            alarm_counter[kind - 1] += 1
        rec = [t_now, m, s, kind, max_hops, 0, False, t_now]
        created.append((m, s, t_now, kind, step))
        if m == leader:
            deliver_to_leader(rec, t_now)
        else:
            insert(m, rec, t_now)

    def receive(pkt: list, m: int, j: int, t_now: float) -> None:
        for rec in pkt:
            carry = rec[R_CARRY] or stuck_last[m] >= rec[R_ENQ]
            new = [rec[R_T], rec[R_ORIGIN], rec[R_SEQ], rec[R_KIND], rec[R_TTL] - 1, rec[R_HOPS] + 1, carry, t_now]
            if j == leader:
                deliver_to_leader(new, t_now)
                continue
            key = (rec[R_ORIGIN], rec[R_SEQ])
            if key in seen[j]:
                continue
            seen[j].add(key)
            if new[R_TTL] <= 0:
                counters["ttl_drops"] += 1
                continue
            insert(j, new, t_now)

    for step in range(n_steps):
        t_now = step * dt
        for j, k in beacons_at.get(step, ()):
            fill[j, k] = min(15, len(queues[j]))
        if step % rec_step == 0:
            for m in range(n_nodes):
                if m != leader and (len(queues[m]) < buffer_size or step % deg_step == 0):
                    create(m, KIND_NORMAL, t_now, step)
        for node, kind in events.get(step, ()):
            create(node, kind, t_now, step)

        for m in range(n_nodes):
            q = queues[m]
            if m == leader or not q:
                continue
            if not (n_pri[m] > 0 or len(q) >= capacity or t_now - wait_since[m] >= max_wait - 1e-9):
                continue
            if not any_valid[step, m]:
                stuck_last[m] = t_now
                continue
            if strategy == "direct":
                js = np.array([leader]) if valid[step, m, leader] else np.empty(0, dtype=int)
                kk = lastk[step, m, js]
            else:
                js = np.flatnonzero(valid[step, m])
                kk = lastk[step, m, js]
                hj = ann[js, kk]
                keep = hj < ann_grid[step, m]
                js, kk = js[keep], kk[keep]
            if len(js) == 0:
                stuck_last[m] = t_now
                continue
            hj = ann[js, kk].astype(float)
            if strategy == "direct" or len(js) == 1 and chooser is None:
                order = np.arange(len(js))
            else:
                sig = c1[step, m, js].astype(float)
                crit = np.column_stack([sig, time_to_break(sig, slope[step, m, js], rcfg["candidate_threshold"], rcfg["ttb_cap"]), hj, c4[step, m, js], fill[js, kk]])
                order = ranker.rank(m, t_now, crit, hj)
            order = order[:1] if strategy == "direct" else order[:2]
            primary = int(js[order[0]])
            if last_primary[m] not in (-1, primary):
                counters["bridge_changes"] += 1
            last_primary[m] = primary
            if trace and m == rd["sep_node"]:
                decisions.append((t_now, primary, float(c1[step, m, primary]), float(ann_grid[step, m])))

            used = 0.0
            while q and used < dt and (n_pri[m] > 0 or len(q) >= capacity or t_now - wait_since[m] >= max_wait - 1e-9):
                pkt = select_packet(q, capacity, n_pri[m])
                pkt_bytes = rcfg["packet_header_bytes"] + len(pkt) * rec_bytes
                counters["max_packet_bytes_sent"] = max(counters["max_packet_bytes_sent"], pkt_bytes)
                data_air = airtime(pkt_bytes, rcfg)
                counters["cycles"] += 1
                done = False
                for rank, ci in enumerate(order):
                    j = int(js[ci])
                    ok, tries, arrivals = unicast_attempts(float(p[step, m, j]), float(p[step, j, m]), retries, stream)
                    spent = tries * data_air + arrivals * ack_air
                    used += spent
                    tx_airtime[m] += tries * data_air
                    tx_airtime[j] += arrivals * ack_air
                    attempts_count[m] += tries
                    if arrivals:
                        receive(pkt, m, j, t_now)
                    if ok:
                        done = True
                        break
                    if rank == 0:
                        counters["breaks"] += 1
                        if trace and m == rd["sep_node"]:
                            break_times.append(t_now)
                        if not last_failed[m]:
                            counters["break_episodes"] += 1
                if not done:
                    counters["fail_both"] += 1
                    last_failed[m] = True
                    break
                last_failed[m] = False
                for rec in pkt:
                    if rec[R_KIND] != KIND_NORMAL:
                        n_pri[m] -= 1
                if n_pri[m] == 0 and pkt == q[: len(pkt)]:
                    del q[: len(pkt)]
                else:
                    for rec in pkt:
                        q.remove(rec)
                wait_since[m] = t_now if q else None

    out = {
        "log": {k: np.asarray(v) for k, v in log.items()},
        "created": np.asarray(created, dtype=float).reshape(-1, 5),
        "tx_airtime": tx_airtime,
        "attempts": attempts_count,
        "max_occupancy": max_occ,
        "lost_buffer": lost_buffer,
        "capacity": capacity,
        "queue_lengths": [len(q) for q in queues],
        **counters,
    }
    if trace:
        out["decisions"] = decisions
        out["break_times"] = break_times
    if chooser is not None:
        out["ahp_weights"], out["cr"] = chooser.w_ahp, chooser.cr
    return out


# ---------------------------------------------------------------------------
# Preparazione di una corsa
# ---------------------------------------------------------------------------


def position_class(mean_offset: np.ndarray, node: int) -> tuple:
    """Posizione di `node` nel gruppo: rango della posizione longitudinale media (0 = coda) e classe 0/1/2 = coda/centro/testa (terzi: ranghi ``< round(N/3)``, ``< round(2N/3)``, il resto)."""
    n = len(mean_offset)
    rank = int(np.argsort(np.argsort(mean_offset))[node])
    return rank, int(rank >= round(n / 3)) + int(rank >= round(2 * n / 3))


def wrong_hops_episodes(t: np.ndarray, ann_grid: np.ndarray, rssi_true: np.ndarray, leader: int, rcfg: dict, stride: int) -> dict:
    """Quanto a lungo un nodo annuncia un numero di salti finito pur non avendo una strada vera verso il capogruppo.

    Procedimento: ogni `stride` passi, grafo vero con arco ``m → j`` se ``rssi_true[m, j] ≥ candidate_threshold``; raggiungibilità del capogruppo per iterazione ``N`` volte sui vicini.
    Sbagliato = salti annunciati finiti (≠ 255) e capogruppo non raggiungibile. Episodio = campioni consecutivi sbagliati di un nodo.
    Output: ``durations`` (s) e ``max_announced`` di ogni episodio, ``wrong_fraction`` (frazione di campioni nodo-tempo sbagliati), ``samples``.
    """
    idx = np.arange(0, len(t), stride)
    n_nodes = ann_grid.shape[1]
    adj = np.nan_to_num(rssi_true[idx], nan=-np.inf) >= rcfg["candidate_threshold"]
    reach = np.zeros((len(idx), n_nodes), dtype=bool)
    reach[:, leader] = True
    for _ in range(n_nodes):
        reach |= (adj & reach[:, None, :]).any(axis=2)
    ann = ann_grid[idx]
    wrong = (ann != ISOLATED) & ~reach
    wrong[:, leader] = False
    step_s = float(t[idx[1]] - t[idx[0]]) if len(idx) > 1 else 0.0
    durations, peaks = [], []
    for node in range(n_nodes):
        col = np.concatenate([[False], wrong[:, node], [False]]).astype(np.int8)
        edges = np.flatnonzero(np.diff(col))
        for a, b in zip(edges[0::2], edges[1::2]):
            durations.append((b - a) * step_s)
            peaks.append(int(ann[a:b, node].max()))
    return {"durations": np.asarray(durations), "max_announced": np.asarray(peaks), "wrong_fraction": float(wrong.mean()), "samples": int(wrong.size)}


RANGE_STEP, RANGE_MAX = 0.5, 100.0  # m; passo e massimo dell'istogramma distanza-segnale
RANGE_RSSI_EDGES = np.arange(-150.0, -9.0, 1.0)  # dBm


def link_components(adj: np.ndarray) -> np.ndarray:
    """Componente connessa di ogni nodo, per ogni campione: etichetta = indice più basso della componente.

    Input: `adj` (S, N, N) bool simmetrico (nodi esclusi: righe e colonne tutte False, restano isolati). Procedimento: propagazione del minimo sui vicini, ``N`` iterazioni. Output (S, N) int.
    """
    n_nodes = adj.shape[1]
    lab = np.broadcast_to(np.arange(n_nodes), adj.shape[:2]).copy()
    for _ in range(n_nodes):
        new = np.minimum(lab, np.where(adj, lab[:, None, :], n_nodes).min(axis=2))
        if np.array_equal(new, lab):
            break
        lab = new
    return lab


def connectivity_stats(t: np.ndarray, rssi_true: np.ndarray, leader: int, sep_node: int, sep_start: float, rcfg: dict, stride: int) -> dict:
    """Connettività verso il capogruppo (anche a più salti), campionata ogni `stride` passi, con il grafo VERO.

    Arco ``i — j`` se il segnale vero è ≥ ``candidate_threshold`` in entrambe le direzioni (l'unicast ha bisogno anche della conferma). Dopo il distacco il nodo separato esce dal gruppo:
    i nodi del gruppo si valutano sul grafo senza di lui, il separato sul grafo completo. Output: ``in``/``n`` (N,) campioni con strada / campioni totali per nodo (nodi nel gruppo; il capogruppo
    conta sempre), ``sep_in``/``sep_n`` per il nodo separato dopo il distacco, ``split``/``split_n`` campioni con il gruppo (senza il separato) in più componenti / campioni totali.
    """
    idx = np.arange(0, len(t), stride)
    n_nodes = rssi_true.shape[1]
    ok = np.nan_to_num(rssi_true[idx], nan=-np.inf) >= rcfg["candidate_threshold"]
    adj = ok & ok.transpose(0, 2, 1)
    out_sep = (t[idx] >= sep_start) if sep_node >= 0 else np.zeros(len(idx), dtype=bool)
    adj_group = adj.copy()
    if sep_node >= 0:
        adj_group[out_sep, sep_node, :] = False
        adj_group[out_sep, :, sep_node] = False
    lab_full, lab_group = link_components(adj), link_components(adj_group)
    active = np.ones((len(idx), n_nodes), dtype=bool)
    if sep_node >= 0:
        active[out_sep, sep_node] = False
    reach = lab_group == lab_group[:, [leader]]
    comps = ((lab_group == np.arange(n_nodes)) & active).sum(axis=1)
    out = {"in": (reach & active).sum(axis=0), "n": active.sum(axis=0), "split": int((comps > 1).sum()), "split_n": int(len(idx)), "sep_in": 0, "sep_n": 0}
    if sep_node >= 0:
        out["sep_in"] = int((lab_full[out_sep, sep_node] == lab_full[out_sep, leader]).sum())
        out["sep_n"] = int(out_sep.sum())
    return out


def range_histogram(distances: np.ndarray, level: np.ndarray, stride: int) -> np.ndarray:
    """Istogramma 2D (distanza, segnale) di ``level`` = ``rssi_path_loss − corpi`` sulle coppie ordinate, ogni `stride` passi: distanza a passi di `RANGE_STEP` fino a `RANGE_MAX`, dBm a 1 dB."""
    n_nodes = distances.shape[1]
    off = ~np.eye(n_nodes, dtype=bool)
    d, v = distances[::stride][:, off].ravel(), level[::stride][:, off].ravel()
    edges = np.arange(0.0, RANGE_MAX + RANGE_STEP / 2, RANGE_STEP)
    return np.histogram2d(d, v, bins=[edges, RANGE_RSSI_EDGES])[0].astype(np.int32)


def prepare_run(config: dict, leader: int, sos_fraction: float) -> dict:
    """Simula una corsa (Blocchi 1-3, Kalman del Blocco 4) e la riduce a ciò che serve al trasporto. Niente cache su disco.

    Input: `config` della corsa (sostituzioni del Blocco 7 e scenario già applicati), indice del `leader`, `sos_fraction` in [0, 1) (istante del SOS = ``sos_window · sos_fraction`` dopo il distacco).
    Procedimento: `simulate_mobility` → `simulate_channel` → `simulate_packets`; ``R`` dal fading della corsa (``kalman.measurement_noise``); `filter_all`, `predict_links`;
    carico dei beacon quantizzato come la tabella dei vicini; `link_state`, `announce_hops`; probabilità di ricezione ``p = curva(rssi_vero)·(1 − perdita di fondo)`` (con perdite a raffiche si usa la
    perdita media); eventi: SOS del separato e allarme oracolo (primo passo con il separato oltre ``oracle_distance`` dal gruppo, generato dal nodo del gruppo con il segnale stimato più alto verso di lui).
    Output: dizionario `rd` per `simulate_strategy`, con in più ``distances``, ``mean_offset``, ``rank``/``pos_class`` del capogruppo, ``wrong`` (`wrong_hops_episodes`), ``extent`` e i campi del distacco.
    """
    rcfg = config["routing"]
    mob = simulate_mobility(config)
    ch = simulate_channel(mob, config)
    pk = simulate_packets(mob, ch, config)
    t = mob.t
    n_nodes = mob.positions.shape[1]
    dt = config["simulation"]["dt"]
    fading_var = float(np.var(ch.fading[:, ~np.eye(n_nodes, dtype=bool)]))
    R = measurement_noise(config, fading_var)
    kcfg, tcfg = config["kalman"], config["neighbour_table"]
    kres = filter_all(pk.beacon_times, pk.received, pk.rssi, R, kcfg["sigma_a"], kcfg)
    est = predict_links(kres, t)
    pay_r = _quantize(kres.r, tcfg["rssi_step"], _INT8)
    pay_s = _quantize(kres.s, tcfg["slope_step"], _SLOPE)
    ls = link_state(t, pk.beacon_times, pk.received, est["r"], est["s"], pay_r, pay_s, rcfg, config["packets"]["beacon_period"])
    ann = announce_hops(pk.beacon_times, pk.beacon_steps, pk.received, est["r"], leader, rcfg)
    ann_grid = announced_grid(t, pk.beacon_times, ann)

    rc = config["packets"]["reception"]
    background = rc["background_loss"] if rc["background_model"] == "independent" else rc["bursty"]["mean_loss"] * rc["bursty"]["loss_bad"]
    p = np.nan_to_num(reception_probability(ch.rssi_true, rc) * (1.0 - background)).astype(np.float32)

    sep = config["separation"]
    sep_enabled = bool(sep["enabled"])
    sep_node = int(sep["node_id"]) if sep_enabled else -1
    sep_start = float(sep["start_time"]) if sep_enabled else float("nan")
    offsets = mob.s_nodes - mob.s_centroid[:, None]
    before = slice(0, int(np.searchsorted(t, sep_start))) if sep_enabled else slice(None)
    mean_offset = offsets[before].mean(axis=0)
    rank, pos_class = position_class(mean_offset, leader)

    events = []
    oracle_step = None
    if sep_enabled:
        idx0 = int(np.searchsorted(t, sep_start))
        over = np.flatnonzero(min_distance_to_group(mob.distances, sep_node)[idx0:] > rcfg["oracle_distance"])
        if len(over):
            oracle_step = idx0 + int(over[0])
            others = [x for x in range(n_nodes) if x != sep_node]
            sig = np.array([est["r"][oracle_step, sep_node, x] for x in others])
            best = others[int(np.nanargmax(sig))] if np.isfinite(sig).any() else others[int(np.argmin(mob.distances[oracle_step, sep_node, others]))]
            events.append((oracle_step, best, KIND_ALARM))
        sos_step = int(round((sep_start + sos_fraction * rcfg["sos_window"]) / dt))
        if sos_step < len(t):
            events.append((sos_step, sep_node, KIND_SOS))

    ext_long = offsets.max(axis=1) - offsets.min(axis=1)
    ext_lat = mob.lateral_offsets.max(axis=1) - mob.lateral_offsets.min(axis=1)
    rd = {
        "dt": dt, "t": t, "times": pk.beacon_times, "steps": pk.beacon_steps, "received": pk.received, "ann": ann, "ann_grid": ann_grid, "p": p,
        "leader": leader, "events": events, "seed": config["simulation"]["seed"], "sep_node": sep_node, "sep_start": sep_start, "oracle_step": oracle_step,
        "distances": mob.distances.astype(np.float32), "mean_offset": mean_offset, "rank": rank, "pos_class": pos_class,
        "extent": {"long_median": float(np.median(ext_long)), "long_p95": float(np.percentile(ext_long, 95)), "lat_median": float(np.median(ext_lat)),
                   "lat_p95": float(np.percentile(ext_lat, 95)), "sigma_long": mob.metadata["sigma_long"], "sigma_lat": mob.metadata["sigma_lat"]},
        "wrong": wrong_hops_episodes(t, ann_grid, ch.rssi_true, leader, rcfg, int(round(1.0 / dt))),
        "node_rank": np.argsort(np.argsort(mean_offset)),
        "connectivity": connectivity_stats(t, ch.rssi_true, leader, sep_node, sep_start, rcfg, int(round(1.0 / dt))),
        "range_hist": range_histogram(mob.distances, ch.rssi_path_loss - ch.body_own - ch.body_others, int(round(1.0 / dt))),
        "R": R, **ls,
    }  # fmt: skip
    return rd

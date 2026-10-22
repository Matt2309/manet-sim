"""Block 7 — Routing to the group leader: bridge, DQREM, carry.

Whoever no longer hears the group leader relays telemetry records, alarms and SOS hop by hop through its companions. Indices ``[i, ..., j]`` = transmits
`i`, receives `j`; ``[t, m, j]`` = node `m` looking at neighbour `j` at grid instant `t`.

Pieces (each testable on its own):

- hops announced in the beacon (`announce_hops`) and link state seen by each node (`link_state`);
- criteria C1-C5, AHP and entropy weights, score (`ahp_weights`, `entropy_weights`, `time_to_break`, `dqrem_scores`, `Chooser`);
- unicast with acknowledgement (`unicast_attempts`), packets (`select_packet`), queues, buffer and carry (`simulate_strategy`);
- preparation of a run from Blocks 1-3 and the Block 4 Kalman filter (`prepare_run`).

The detector is NOT evaluated here: the alarms to carry come from the ground truth (oracle alarm) and from the SOS of the separated node.
Collisions are not modelled; airtime is counted. Parameters: ``routing`` section of the config.
"""

from __future__ import annotations

import math
from bisect import insort
from collections import defaultdict, deque

import numpy as np

from src.common.enums import Strategy
from src.common.rng import spawn_rng
from src.detector import INT8_RANGE, SLOPE_CODE_RANGE, quantize
from src.kalman import filter_all, measurement_noise, predict_links
from src.metrics import min_distance_to_group
from src.packets import neighbour_tables, reception_probability
from src.simulator import off_diagonal_variance, separation_fields, simulate_pipeline

ISOLATED = 255  # hops announced by an isolated node
STRATEGIES = tuple(s.value for s in Strategy)
CRITERIA = ("segnale", "tempo_rottura", "salti", "ricezione", "coda")
BENEFIT = np.array([True, True, False, True, False])  # C3 (hops) and C5 (queue) are costs
NO_SLOPE_COLUMNS = (0, 2, 3, 4)  # `dqrem_no_slope`: without C2
KIND_NORMAL, KIND_ALARM, KIND_SOS = 0, 1, 2
ALARM_SEQ_BASE, SOS_SEQ_BASE = 1_000_000, 2_000_000  # sequence numbers of alarm and SOS messages (do not collide with records)

# indices of a record's fields: [t_creation, origin, sequence, kind, ttl, hops, carry, t_enqueued]
R_T, R_ORIGIN, R_SEQ, R_KIND, R_TTL, R_HOPS, R_CARRY, R_ENQ = range(8)


# ---------------------------------------------------------------------------
# Sizes and airtime
# ---------------------------------------------------------------------------


def record_bytes(n_nodes: int) -> int:
    """Bytes of a telemetry record: like the beacon, ``8 + 5·d`` with ``d = n_nodes − 1`` neighbours in the table (53 with 10 nodes)."""
    return 8 + 5 * (n_nodes - 1)


def records_per_packet(rcfg: dict, n_nodes: int, max_packet_bytes: int | None = None) -> int:
    """Records that fit in a packet: ``min(max_records, (max_packet_bytes − header) // bytes per record)`` (4 with 250 B and 10 nodes, 10 with 1470 B)."""
    limit = rcfg["max_packet_bytes"] if max_packet_bytes is None else max_packet_bytes
    fit = (limit - rcfg["packet_header_bytes"]) // record_bytes(n_nodes)
    return int(max(1, min(rcfg["max_records"], fit)))


def airtime(n_bytes: int, rcfg: dict) -> float:
    """Airtime in s of a frame: preamble + bytes (802.11b at 1 Mbit/s: 192 µs + 8 µs per byte)."""
    a = rcfg["airtime"]
    return (a["preamble_us"] + a["us_per_byte"] * n_bytes) * 1e-6


# ---------------------------------------------------------------------------
# Weights, criteria and DQREM score
# ---------------------------------------------------------------------------


def ahp_weights(matrix, random_index: dict) -> tuple:
    """AHP weights and consistency ratio from the pairwise comparison matrix (Saaty scale).

    Input: `matrix` (n, n) and `random_index` (Saaty random index for n).
    Procedure: principal eigenvector (largest real eigenvalue) normalised to sum 1; ``CI = (λmax − n)/(n − 1)``, ``CR = CI / RI``.
    Output: (weights (n,), CR).
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
    """Shannon entropy weights of the criteria over the collected samples.

    Input: `samples` (n, c) criteria values (one row per candidate seen), `benefit` (c,) bool.
    Procedure: per column, min-max normalisation over the rows (costs inverted); proportions ``p = x / Σx``; ``E = −Σ p ln p / ln n``; weight ∝ ``1 − E``.
    A column that does not vary (or with fewer than 2 rows) has weight 0.
    Output: (c,) summing to 1, or all zeros if no criterion varies.
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
    """``w = α·w_AHP + (1 − α)·w_entropy``, renormalised to sum 1."""
    w = alpha * np.asarray(w_ahp) + (1.0 - alpha) * np.asarray(w_entropy)
    return w / w.sum()


def time_to_break(signal, slope, threshold: float, cap: float):
    """Time to break C2 in s: ``(signal − threshold) / (−slope)`` if the slope is negative, otherwise `cap`; clipped to ``[0, cap]``."""
    signal = np.asarray(signal, dtype=float)
    slope = np.asarray(slope, dtype=float)
    falling = slope < 0
    with np.errstate(divide="ignore", invalid="ignore"):
        ttb = np.where(falling, (signal - threshold) / np.where(falling, -slope, 1.0), cap)
    return np.clip(np.nan_to_num(ttb, nan=cap), 0.0, cap)


def dqrem_scores(x: np.ndarray, weights: np.ndarray, benefit: np.ndarray) -> np.ndarray:
    """DQREM score of the candidates: each criterion min-max normalised over the current candidates (costs inverted; criterion equal for all, or a single candidate → 1), then weighted sum.

    Input: `x` (n, c) candidate criteria, `weights` (c,), `benefit` (c,) bool. Output: (n,) scores (higher = better).
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
    """Picks the next hop among the candidates with strategy `strategy` (`best_rssi`, `dqrem`, `dqrem_no_slope`).

    Input: `rcfg` (``routing`` section), number of nodes. State: per node, the criteria of the candidates seen in the last ``entropy_window`` s.
    `dqrem_no_slope` drops C2 and recomputes the AHP weights on the 4×4 submatrix (same pairwise comparisons), entropy over the 4 criteria.
    """

    def rank(self, node: int, t_now: float, criteria: np.ndarray, hops: np.ndarray) -> np.ndarray:
        """Candidate indices from best to worst. `criteria` (n, 5) is ``[C1..C5]``, `hops` (n,) the announced hops (tie-break of `best_rssi`)."""
        raise NotImplementedError


class BestRssiChooser(Chooser):
    def rank(self, node: int, t_now: float, criteria: np.ndarray, hops: np.ndarray) -> np.ndarray:
        return np.lexsort((hops, -criteria[:, 0]))


class DqremChooser(Chooser):
    def __init__(self, strategy: Strategy, rcfg: dict, n_nodes: int):
        self.strategy = strategy
        self.rcfg = rcfg
        self.columns = list(NO_SLOPE_COLUMNS) if strategy == Strategy.DQREM_NO_SLOPE else list(range(len(CRITERIA)))
        matrix = np.asarray(rcfg["ahp_matrix"], dtype=float)[np.ix_(self.columns, self.columns)]
        self.w_ahp, self.cr = ahp_weights(matrix, rcfg["ahp_random_index"])
        self.benefit = BENEFIT[self.columns]
        self.samples = [deque() for _ in range(n_nodes)]
        self.last_weights = None

    def weights(self, node: int) -> np.ndarray:
        """Final weights of the node: AHP only with fewer than ``entropy_min_samples`` samples, otherwise ``α·AHP + (1 − α)·entropy``."""
        rows = np.vstack([x for _, x in self.samples[node]])
        if len(rows) < self.rcfg["entropy_min_samples"]:
            return self.w_ahp
        return combine_weights(self.w_ahp, entropy_weights(rows, self.benefit), self.rcfg["alpha"])

    def rank(self, node: int, t_now: float, criteria: np.ndarray, hops: np.ndarray) -> np.ndarray:
        x = criteria[:, self.columns]
        window = self.samples[node]
        window.append((t_now, x))
        while window[0][0] < t_now - self.rcfg["entropy_window"]:
            window.popleft()
        w = self.weights(node)
        self.last_weights = w
        return np.argsort(-dqrem_scores(x, w, self.benefit), kind="stable")


def make_chooser(strategy: str, rcfg: dict, n_nodes: int) -> Chooser | None:
    strategy = Strategy(strategy)
    if strategy == Strategy.DIRECT:
        return None
    if strategy == Strategy.BEST_RSSI:
        return BestRssiChooser()
    return DqremChooser(strategy, rcfg, n_nodes)


# ---------------------------------------------------------------------------
# Announced hops and link state
# ---------------------------------------------------------------------------


def announce_hops(times: np.ndarray, steps: np.ndarray, received: np.ndarray, est_r: np.ndarray, leader: int, rcfg: dict) -> np.ndarray:
    """Distance in hops from the group leader that each node announces at every beacon.

    Input: `times`, `steps` (N, K) (beacon instants and grid indices, NaN / −1 past the end), `received` (N, K, N), `est_r` (T, N, N) ``[t, i, j]`` estimate by `j` of the
    link ``i → j`` (NaN if none), `leader`, `rcfg`.
    Procedure: the beacons are walked in time order. The leader announces 0; every other node announces ``1 + min`` of the hops announced by its valid neighbours (beacon received
    less than ``neighbour_max_age`` s ago and estimated signal ≥ ``candidate_threshold``). With no valid neighbours, or beyond ``max_hops``, it announces 255 (isolated).
    Output: (N, K) int16, 255 where the beacon does not exist.
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
    """Current hops of each node on the grid: value of its own last beacon ``<= t``, 255 before the first. Output (T, N) int16."""
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
    """What node `m` knows about neighbour `j` at every instant, to choose the next hop.

    Input: grid `t` (T,), beacons (`times` (N, K), `received` (N, K, N)), Kalman estimates on the grid `est_r`, `est_s` (T, N, N) ``[t, i, j]`` (link ``i → j`` seen by `j`),
    beacon payload `pay_r`, `pay_s` (N, K, N) ``[i, k, j]`` (estimate of ``i → j`` after beacon `k` of `i`, already quantised as in the neighbour table), `rcfg`, beacon period.
    Procedure: for each pair ``(m, j)``: ``lastk`` = last beacon of `j` received by `m`; ``valid`` if it is less than ``neighbour_max_age`` s old and the estimated signal ``j → m`` is
    ≥ ``candidate_threshold``; C1 and slope of ``m → j`` from `m`'s entry in that beacon's table (``r + s·age``), or, if older than ``neighbour_max_age`` s or missing, from the
    own estimate of ``j → m`` (reciprocity); C4 = beacons of `j` received by `m` in ``reception_window`` s / expected.
    Output: dict of (T, N, N) arrays ``[t, m, j]``: ``lastk`` int32 (−1 if none), ``valid`` bool, ``c1`` and ``slope`` float32, ``c4`` float32; plus ``any_valid`` (T, N).
    """
    n_nodes = times.shape[0]
    n_steps = len(t)
    max_age, threshold, window = rcfg["neighbour_max_age"], rcfg["candidate_threshold"], rcfg["reception_window"]
    tables, origin = neighbour_tables(times, received, {"r": pay_r, "s": pay_s})
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
            sig = est_r[:, j, m]  # signal ``j → m`` estimated by `m`
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
# Unicast, packets
# ---------------------------------------------------------------------------


class UniformStream:
    """Uniforms (0, 1) from a generator, in blocks (same sequence as single ``random()`` calls)."""

    def __init__(self, rng: np.random.Generator, chunk: int = 1 << 16):
        self.rng, self.chunk = rng, chunk
        self.buf, self.pos = rng.random(chunk).tolist(), 0

    def next(self) -> float:
        if self.pos >= len(self.buf):
            self.buf, self.pos = self.rng.random(self.chunk).tolist(), 0
        self.pos += 1
        return self.buf[self.pos - 1]


def unicast_attempts(p_forward: float, p_return: float, max_retries: int, stream: UniformStream) -> tuple:
    """Unicast with acknowledgement: each attempt succeeds if the packet arrives (probability `p_forward`) **and** the ack comes back (`p_return`), with fresh draws at every attempt.

    Input: reception probability of the forward and return legs, `max_retries` retransmissions, stream of uniforms (two draws per attempt).
    Output: (succeeded, attempts made ≤ ``1 + max_retries``, how many times the packet arrived). If the packet arrives but the ack does not, the sender retransmits.
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
    """Records of the next packet (without removing them from the queue): alarms and SOS first (priority queue), then the oldest, up to `capacity`.

    `queue` is sorted by creation instant; `n_priority` = how many priority records it holds.
    """
    if n_priority == 0:
        return queue[:capacity]
    chosen = [r for r in queue if r[R_KIND] != KIND_NORMAL][:capacity]
    if len(chosen) < capacity:
        chosen += [r for r in queue if r[R_KIND] == KIND_NORMAL][: capacity - len(chosen)]
    return chosen


# ---------------------------------------------------------------------------
# Simulation of a strategy
# ---------------------------------------------------------------------------


def second_stride(dt: float) -> int:
    return int(round(1.0 / dt))


def _beacons_by_step(steps: np.ndarray, n_nodes: int) -> defaultdict:
    beacons_at = defaultdict(list)
    for j in range(n_nodes):
        for k in np.flatnonzero(steps[j] >= 0):
            beacons_at[int(steps[j, k])].append((j, int(k)))
    return beacons_at


def _events_by_step(raw_events) -> defaultdict:
    events = defaultdict(list)
    for step, node, kind in raw_events:
        events[int(step)].append((int(node), int(kind)))
    return events


class _Transport:
    def __init__(self, rd: dict, strategy: Strategy, rcfg: dict, max_packet_bytes: int | None, trace: bool):
        n_nodes = rd["ann_grid"].shape[1]
        dt = rd["dt"]
        self.rd, self.strategy, self.rcfg, self.trace = rd, strategy, rcfg, trace
        self.n_nodes, self.n_steps, self.dt, self.leader = n_nodes, len(rd["t"]), dt, rd["leader"]
        self.capacity = records_per_packet(rcfg, n_nodes, max_packet_bytes)
        self.rec_bytes = record_bytes(n_nodes)
        self.buffer_size, self.max_hops, self.max_wait, self.retries = rcfg["buffer_records"], rcfg["max_hops"], rcfg["max_wait"], rcfg["max_retries"]
        self.rec_step = int(round(rcfg["record_period"] / dt))
        self.deg_step = int(round(rcfg["degraded_period"] / dt))
        self.ack_air = airtime(rcfg["airtime"]["ack_bytes"], rcfg)
        self.p, self.valid, self.any_valid, self.lastk = rd["p"], rd["valid"], rd["any_valid"], rd["lastk"]
        self.c1, self.slope, self.c4, self.ann, self.ann_grid = rd["c1"], rd["slope"], rd["c4"], rd["ann"], rd["ann_grid"]
        self.fill = np.zeros(self.ann.shape, dtype=np.int16)  # queue fill announced in each beacon (4 bits)
        self.stream = UniformStream(spawn_rng(rd["seed"], rcfg["seed_keys"]["unicast"]))
        self.ranker = make_chooser(strategy, rcfg, n_nodes)
        self.chooser = self.ranker if isinstance(self.ranker, DqremChooser) else None

        self.beacons_at = _beacons_by_step(rd["steps"], n_nodes)
        self.events = _events_by_step(rd["events"])

        self.queues = [[] for _ in range(n_nodes)]
        self.n_pri = [0] * n_nodes
        self.seen = [set() for _ in range(n_nodes)]
        self.wait_since = [None] * n_nodes
        self.stuck_last = [-math.inf] * n_nodes
        self.seq = [0] * n_nodes
        self.last_primary = [-1] * n_nodes
        self.last_failed = [False] * n_nodes
        self.tx_airtime = np.zeros(n_nodes)
        self.attempts_count = np.zeros(n_nodes, dtype=np.int64)
        self.max_occ = np.zeros(n_nodes, dtype=np.int64)
        self.lost_buffer = np.zeros(n_nodes, dtype=np.int64)
        self.counters = dict(breaks=0, break_episodes=0, fail_both=0, bridge_changes=0, ttl_drops=0, duplicates=0, max_packet_bytes_sent=0, cycles=0)
        self.log = {k: [] for k in ("origin", "seq", "t_create", "t_arrive", "hops", "kind", "carry")}
        self.leader_seen = set()
        self.created = []
        self.decisions = []
        self.break_times = []
        self.alarm_counter = [0, 0]

    def insert(self, m: int, rec: list, t_now: float) -> None:
        """Puts `rec` in the queue of `m` (sorted by creation); with a full buffer it overwrites the oldest non-SOS, never an SOS (if all are SOS it drops the new one)."""
        q = self.queues[m]
        if len(q) >= self.buffer_size:
            for idx, old in enumerate(q):
                if old[R_KIND] != KIND_SOS:
                    if old[R_KIND] != KIND_NORMAL:
                        self.n_pri[m] -= 1
                    del q[idx]
                    self.lost_buffer[m] += 1
                    break
            else:
                self.lost_buffer[m] += 1
                return
        rec[R_ENQ] = t_now
        insort(q, rec)
        if rec[R_KIND] != KIND_NORMAL:
            self.n_pri[m] += 1
        if self.wait_since[m] is None:
            self.wait_since[m] = t_now
        if len(q) > self.max_occ[m]:
            self.max_occ[m] = len(q)

    def deliver_to_leader(self, rec: list, t_now: float) -> None:
        key = (rec[R_ORIGIN], rec[R_SEQ])
        if key in self.leader_seen:
            self.counters["duplicates"] += 1
            return
        self.leader_seen.add(key)
        for name, value in zip(self.log, (rec[R_ORIGIN], rec[R_SEQ], rec[R_T], t_now, rec[R_HOPS], rec[R_KIND], rec[R_CARRY])):
            self.log[name].append(value)

    def create(self, m: int, kind: int, t_now: float, step: int) -> None:
        if kind == KIND_NORMAL:
            s = self.seq[m]
            self.seq[m] += 1
        else:
            base = ALARM_SEQ_BASE if kind == KIND_ALARM else SOS_SEQ_BASE
            s = base + self.alarm_counter[kind - 1]
            self.alarm_counter[kind - 1] += 1
        rec = [t_now, m, s, kind, self.max_hops, 0, False, t_now]
        self.created.append((m, s, t_now, kind, step))
        if m == self.leader:
            self.deliver_to_leader(rec, t_now)
        else:
            self.insert(m, rec, t_now)

    def receive(self, pkt: list, m: int, j: int, t_now: float) -> None:
        for rec in pkt:
            carry = rec[R_CARRY] or self.stuck_last[m] >= rec[R_ENQ]
            new = [rec[R_T], rec[R_ORIGIN], rec[R_SEQ], rec[R_KIND], rec[R_TTL] - 1, rec[R_HOPS] + 1, carry, t_now]
            if j == self.leader:
                self.deliver_to_leader(new, t_now)
                continue
            key = (rec[R_ORIGIN], rec[R_SEQ])
            if key in self.seen[j]:
                continue
            self.seen[j].add(key)
            if new[R_TTL] <= 0:
                self.counters["ttl_drops"] += 1
                continue
            self.insert(j, new, t_now)

    def generate(self, step: int, t_now: float) -> None:
        for j, k in self.beacons_at.get(step, ()):
            self.fill[j, k] = min(15, len(self.queues[j]))
        if step % self.rec_step == 0:
            for m in range(self.n_nodes):
                if m != self.leader and (len(self.queues[m]) < self.buffer_size or step % self.deg_step == 0):
                    self.create(m, KIND_NORMAL, t_now, step)
        for node, kind in self.events.get(step, ()):
            self.create(node, kind, t_now, step)

    def ready(self, m: int, t_now: float) -> bool:
        return self.n_pri[m] > 0 or len(self.queues[m]) >= self.capacity or t_now - self.wait_since[m] >= self.max_wait - 1e-9

    def candidates(self, step: int, m: int):
        if self.strategy == Strategy.DIRECT:
            js = np.array([self.leader]) if self.valid[step, m, self.leader] else np.empty(0, dtype=int)
            kk = self.lastk[step, m, js]
        else:
            js = np.flatnonzero(self.valid[step, m])
            kk = self.lastk[step, m, js]
            hj = self.ann[js, kk]
            keep = hj < self.ann_grid[step, m]
            js, kk = js[keep], kk[keep]
        return js, kk

    def choose_order(self, step: int, m: int, t_now: float, js: np.ndarray, kk: np.ndarray) -> np.ndarray:
        hj = self.ann[js, kk].astype(float)
        if self.strategy == Strategy.DIRECT or len(js) == 1 and self.chooser is None:
            order = np.arange(len(js))
        else:
            sig = self.c1[step, m, js].astype(float)
            crit = np.column_stack([sig, time_to_break(sig, self.slope[step, m, js], self.rcfg["candidate_threshold"], self.rcfg["ttb_cap"]), hj, self.c4[step, m, js], self.fill[js, kk]])
            order = self.ranker.rank(m, t_now, crit, hj)
        return order[:1] if self.strategy == Strategy.DIRECT else order[:2]

    def serve(self, step: int, m: int, t_now: float) -> None:
        q = self.queues[m]
        if m == self.leader or not q:
            return
        if not self.ready(m, t_now):
            return
        if not self.any_valid[step, m]:
            self.stuck_last[m] = t_now
            return
        js, kk = self.candidates(step, m)
        if len(js) == 0:
            self.stuck_last[m] = t_now
            return
        order = self.choose_order(step, m, t_now, js, kk)
        primary = int(js[order[0]])
        if self.last_primary[m] not in (-1, primary):
            self.counters["bridge_changes"] += 1
        self.last_primary[m] = primary
        if self.trace and m == self.rd["sep_node"]:
            self.decisions.append((t_now, primary, float(self.c1[step, m, primary]), float(self.ann_grid[step, m])))
        self.transmit(step, m, t_now, js, order)

    def transmit(self, step: int, m: int, t_now: float, js: np.ndarray, order: np.ndarray) -> None:
        q = self.queues[m]
        rcfg, counters = self.rcfg, self.counters
        used = 0.0
        while q and used < self.dt and self.ready(m, t_now):
            pkt = select_packet(q, self.capacity, self.n_pri[m])
            pkt_bytes = rcfg["packet_header_bytes"] + len(pkt) * self.rec_bytes
            counters["max_packet_bytes_sent"] = max(counters["max_packet_bytes_sent"], pkt_bytes)
            data_air = airtime(pkt_bytes, rcfg)
            counters["cycles"] += 1
            done = False
            for rank, ci in enumerate(order):
                j = int(js[ci])
                ok, tries, arrivals = unicast_attempts(float(self.p[step, m, j]), float(self.p[step, j, m]), self.retries, self.stream)
                spent = tries * data_air + arrivals * self.ack_air
                used += spent
                self.tx_airtime[m] += tries * data_air
                self.tx_airtime[j] += arrivals * self.ack_air
                self.attempts_count[m] += tries
                if arrivals:
                    self.receive(pkt, m, j, t_now)
                if ok:
                    done = True
                    break
                if rank == 0:
                    counters["breaks"] += 1
                    if self.trace and m == self.rd["sep_node"]:
                        self.break_times.append(t_now)
                    if not self.last_failed[m]:
                        counters["break_episodes"] += 1
            if not done:
                counters["fail_both"] += 1
                self.last_failed[m] = True
                break
            self.last_failed[m] = False
            for rec in pkt:
                if rec[R_KIND] != KIND_NORMAL:
                    self.n_pri[m] -= 1
            if self.n_pri[m] == 0 and pkt == q[: len(pkt)]:
                del q[: len(pkt)]
            else:
                for rec in pkt:
                    q.remove(rec)
            self.wait_since[m] = t_now if q else None

    def result(self) -> dict:
        out = {
            "log": {k: np.asarray(v) for k, v in self.log.items()},
            "created": np.asarray(self.created, dtype=float).reshape(-1, 5),
            "tx_airtime": self.tx_airtime,
            "attempts": self.attempts_count,
            "max_occupancy": self.max_occ,
            "lost_buffer": self.lost_buffer,
            "capacity": self.capacity,
            "queue_lengths": [len(q) for q in self.queues],
            **self.counters,
        }
        if self.trace:
            out["decisions"] = self.decisions
            out["break_times"] = self.break_times
        if self.chooser is not None:
            out["ahp_weights"], out["cr"] = self.chooser.w_ahp, self.chooser.cr
        return out


def simulate_strategy(rd: dict, strategy: str, rcfg: dict, max_packet_bytes: int | None = None, trace: bool = False) -> dict:
    """Transport of the records to the group leader with strategy `strategy`, on the simulation grid.

    Input: `rd` (prepared run: `prepare_run` or synthetic, see below), `strategy` in `STRATEGIES`, `rcfg`, `max_packet_bytes` (default ``routing.max_packet_bytes``), `trace`.
    `rd` contains: ``dt``, ``t``, ``times``, ``steps``, ``ann``, ``ann_grid``, the fields of `link_state`, ``p`` (T, N, N) reception probability of link ``i → j``, ``leader``,
    ``events`` (list ``(step, node, kind)`` of alarms and SOS), ``seed`` (of the run), ``sep_node`` (−1 if none).
    Procedure: at each step every node creates 1 record/s (``record_period``; with a full buffer only at ``degraded_period``, overwriting the oldest non-SOS); it transmits if it has a
    priority record, a full packet or the first queued record has waited ``max_wait`` s. Candidates = valid neighbours with strictly fewer hops than its own; `direct` only the leader.
    One attempt = unicast with ack (`unicast_attempts`), up to ``max_retries`` retransmissions to the first candidate, then one try on the second; if that also fails the records
    stay in the queue (carry). The receiver inserts the records in its own queue (drops those already seen; TTL ``max_hops``); the leader drops duplicates (origin, sequence).
    A node does not exceed ``dt`` s of airtime per step. The unicast generator is ``SeedSequence(run seed, spawn_key=(seed_keys.unicast,))``.
    Output: dict with the leader's log (``log``: lists origin, seq, t_create, t_arrive, hops, kind, carry), the created records (``created``), counters
    (``breaks``, ``break_episodes``, ``fail_both``, ``bridge_changes``, ``ttl_drops``, ``duplicates``, ``max_packet_bytes_sent``) and per node ``tx_airtime``, ``attempts``,
    ``max_occupancy``, ``lost_buffer``; with `trace` also ``decisions`` of the separated node.
    """
    strategy = Strategy(strategy)
    transport = _Transport(rd, strategy, rcfg, max_packet_bytes, trace)
    for step in range(transport.n_steps):
        t_now = step * transport.dt
        transport.generate(step, t_now)
        for m in range(transport.n_nodes):
            transport.serve(step, m, t_now)
    return transport.result()



# ---------------------------------------------------------------------------
# Preparation of a run
# ---------------------------------------------------------------------------


def position_class(mean_offset: np.ndarray, node: int) -> tuple:
    """Position of `node` in the group: rank of the mean longitudinal position (0 = tail) and class 0/1/2 = tail/centre/head (thirds: ranks ``< round(N/3)``, ``< round(2N/3)``, the rest)."""
    n = len(mean_offset)
    rank = int(np.argsort(np.argsort(mean_offset))[node])
    return rank, int(rank >= round(n / 3)) + int(rank >= round(2 * n / 3))


def wrong_hops_episodes(t: np.ndarray, ann_grid: np.ndarray, rssi_true: np.ndarray, leader: int, rcfg: dict, stride: int) -> dict:
    """How long a node announces a finite hop count while having no real path to the group leader.

    Procedure: every `stride` steps, true graph with edge ``m → j`` if ``rssi_true[m, j] ≥ candidate_threshold``; leader reachability by iterating ``N`` times over the neighbours.
    Wrong = finite announced hops (≠ 255) and leader not reachable. Episode = consecutive wrong samples of a node.
    Output: ``durations`` (s) and ``max_announced`` of each episode, ``wrong_fraction`` (fraction of wrong node-time samples), ``samples``.
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


RANGE_STEP, RANGE_MAX = 0.5, 100.0  # m; step and maximum of the distance-signal histogram
RANGE_RSSI_EDGES = np.arange(-150.0, -9.0, 1.0)  # dBm


def link_components(adj: np.ndarray) -> np.ndarray:
    """Connected component of each node, for each sample: label = lowest index of the component.

    Input: `adj` (S, N, N) symmetric bool (excluded nodes: rows and columns all False, they stay isolated). Procedure: propagation of the minimum over the neighbours, ``N`` iterations. Output (S, N) int.
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
    """Connectivity to the group leader (also over multiple hops), sampled every `stride` steps, on the TRUE graph.

    Edge ``i — j`` if the true signal is ≥ ``candidate_threshold`` in both directions (the unicast also needs the ack). After the detachment the separated node leaves the group:
    the group's nodes are evaluated on the graph without it, the separated one on the full graph. Output: ``in``/``n`` (N,) samples with a path / total samples per node (nodes in the group; the leader
    always counts), ``sep_in``/``sep_n`` for the separated node after the detachment, ``split``/``split_n`` samples with the group (without the separated node) in several components / total samples.
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
    """2D histogram (distance, signal) of ``level`` = ``rssi_path_loss − bodies`` over the ordered pairs, every `stride` steps: distance in steps of `RANGE_STEP` up to `RANGE_MAX`, dBm at 1 dB."""
    n_nodes = distances.shape[1]
    off = ~np.eye(n_nodes, dtype=bool)
    d, v = distances[::stride][:, off].ravel(), level[::stride][:, off].ravel()
    edges = np.arange(0.0, RANGE_MAX + RANGE_STEP / 2, RANGE_STEP)
    return np.histogram2d(d, v, bins=[edges, RANGE_RSSI_EDGES])[0].astype(np.int32)


def _reception_p(ch, rc: dict) -> np.ndarray:
    background = rc["background_loss"] if rc["background_model"] == "independent" else rc["bursty"]["mean_loss"] * rc["bursty"]["loss_bad"]
    return np.nan_to_num(reception_probability(ch.rssi_true, rc) * (1.0 - background)).astype(np.float32)


def _group_events(mob, est: dict, t: np.ndarray, dt: float, sep_enabled: bool, sep_node: int, sep_start: float, sos_fraction: float, rcfg: dict) -> tuple:
    n_nodes = mob.positions.shape[1]
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
    return events, oracle_step


def prepare_run(config: dict, leader: int, sos_fraction: float) -> dict:
    """Simulates a run (Blocks 1-3, Block 4 Kalman filter) and reduces it to what the transport needs. No disk cache.

    Input: run `config` (Block 7 overrides and scenario already applied), index of the `leader`, `sos_fraction` in [0, 1) (SOS instant = ``sos_window · sos_fraction`` after the detachment).
    Procedure: `simulate_mobility` → `simulate_channel` → `simulate_packets`; ``R`` from the run's fading (``kalman.measurement_noise``); `filter_all`, `predict_links`;
    beacon payload quantised like the neighbour table; `link_state`, `announce_hops`; reception probability ``p = curve(true_rssi)·(1 − background loss)`` (with bursty losses the
    mean loss is used); events: SOS of the separated node and oracle alarm (first step with the separated node beyond ``oracle_distance`` from the group, generated by the group node with the highest estimated signal towards it).
    Output: dict `rd` for `simulate_strategy`, plus ``distances``, ``mean_offset``, ``rank``/``pos_class`` of the leader, ``wrong`` (`wrong_hops_episodes`), ``extent`` and the detachment fields.
    """
    rcfg = config["routing"]
    mob, ch, pk = simulate_pipeline(config)
    t = mob.t
    n_nodes = mob.positions.shape[1]
    dt = config["simulation"]["dt"]
    fading_var = off_diagonal_variance(ch.fading, n_nodes)
    R = measurement_noise(config, fading_var)
    kcfg, tcfg = config["kalman"], config["neighbour_table"]
    kres = filter_all(pk.beacon_times, pk.received, pk.rssi, R, kcfg["sigma_a"], kcfg)
    est = predict_links(kres, t)
    pay_r = quantize(kres.r, tcfg["rssi_step"], INT8_RANGE)
    pay_s = quantize(kres.s, tcfg["slope_step"], SLOPE_CODE_RANGE)
    ls = link_state(t, pk.beacon_times, pk.received, est["r"], est["s"], pay_r, pay_s, rcfg, config["packets"]["beacon_period"])
    ann = announce_hops(pk.beacon_times, pk.beacon_steps, pk.received, est["r"], leader, rcfg)
    ann_grid = announced_grid(t, pk.beacon_times, ann)

    p = _reception_p(ch, config["packets"]["reception"])

    sep = separation_fields(config)
    sep_enabled, sep_node, sep_start = sep["sep_enabled"], sep["sep_node"], sep["sep_start"]
    offsets = mob.s_nodes - mob.s_centroid[:, None]
    before = slice(0, int(np.searchsorted(t, sep_start))) if sep_enabled else slice(None)
    mean_offset = offsets[before].mean(axis=0)
    rank, pos_class = position_class(mean_offset, leader)

    events, oracle_step = _group_events(mob, est, t, dt, sep_enabled, sep_node, sep_start, sos_fraction, rcfg)

    ext_long = offsets.max(axis=1) - offsets.min(axis=1)
    ext_lat = mob.lateral_offsets.max(axis=1) - mob.lateral_offsets.min(axis=1)
    rd = {
        "dt": dt, "t": t, "times": pk.beacon_times, "steps": pk.beacon_steps, "received": pk.received, "ann": ann, "ann_grid": ann_grid, "p": p,
        "leader": leader, "events": events, "seed": config["simulation"]["seed"], "sep_node": sep_node, "sep_start": sep_start, "oracle_step": oracle_step,
        "distances": mob.distances.astype(np.float32), "mean_offset": mean_offset, "rank": rank, "pos_class": pos_class,
        "extent": {"long_median": float(np.median(ext_long)), "long_p95": float(np.percentile(ext_long, 95)), "lat_median": float(np.median(ext_lat)),
                   "lat_p95": float(np.percentile(ext_lat, 95)), "sigma_long": mob.metadata["sigma_long"], "sigma_lat": mob.metadata["sigma_lat"]},
        "wrong": wrong_hops_episodes(t, ann_grid, ch.rssi_true, leader, rcfg, second_stride(dt)),
        "node_rank": np.argsort(np.argsort(mean_offset)),
        "connectivity": connectivity_stats(t, ch.rssi_true, leader, sep_node, sep_start, rcfg, second_stride(dt)),
        "range_hist": range_histogram(mob.distances, ch.rssi_path_loss - ch.body_own - ch.body_others, second_stride(dt)),
        "R": R, **ls,
    }  # fmt: skip
    return rd

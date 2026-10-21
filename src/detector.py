"""Block 4 — Neighbour table, fusion and detector.

Chain on each node: received beacon → link Kalman (`src/kalman.py`) →
neighbour table → per-target fusion → detector → alarm.
Index convention: ``[i, ..., j]`` = `i` transmits, `j` receives.

Neighbour table (5 bytes per neighbour): id, estimated RSSI (int8, 1 dB),
slope (int8, 0.1 dB/s, ±12.7), standard deviation (uint8, 0.1 dB), age
(uint8, 0.1 s, saturates at 25.5 s). The beacon of `j` carries, for each neighbour `i`,
the Kalman posterior estimate of the link `i → j`; the receiver adds the
time elapsed since reception (exact).

Fusion per target `i` (observer `m`): only the links on which `i` transmits.
Simplifications, to be refined:

- the table does not carry the full covariance: the entry ``i → k`` advances with
  ``r + s·age`` and variance ``std² + σ_a²·age³/3`` (ignores the level-slope
  correlation);
- for the slope the same ``1/variance`` weights as for the RSSI are used;
- shadowing is shared among the links (correlation ~0.9): the weighted mean
  UNDERESTIMATES the true variance, so the thresholds are tuned on the data (sweep).

Parameters: ``neighbour_table``, ``fusion``, ``detector`` sections of the config.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.common.arrays import rising_edges, warmup_start
from src.common.enums import DetectorMode, LostRule, Reference
from src.fusion import combine
from src.kalman import KalmanResult, filter_all, predict_links
from src.packets import forward_link_payload


# ---------------------------------------------------------------------------
# 5-byte neighbour table
# ---------------------------------------------------------------------------

# Integer type limits; the slope is symmetric (±127 steps).
INT8_RANGE = (-128, 127)
SLOPE_CODE_RANGE = (-127, 127)
_UINT8 = (0, 255)


def _to_code(values: np.ndarray, step: float, limits: tuple, dtype) -> np.ndarray:
    return np.clip(np.rint(np.asarray(values, dtype=float) / step), limits[0], limits[1]).astype(dtype)


def encode_table(r, s, std, age, table_cfg: dict) -> dict:
    """Encodes an entry (4 bytes, id excluded): rounds to step and saturates at the type limits.

    Fields in physical units (dBm, dB/s, dB, s); returns ``r``, ``s`` (int8), ``std``, ``age`` (uint8).
    """
    return {
        "r": _to_code(r, table_cfg["rssi_step"], INT8_RANGE, np.int8),
        "s": _to_code(s, table_cfg["slope_step"], SLOPE_CODE_RANGE, np.int8),
        "std": _to_code(std, table_cfg["std_step"], _UINT8, np.uint8),
        "age": _to_code(age, table_cfg["age_step"], _UINT8, np.uint8),
    }


def decode_table(codes: dict, table_cfg: dict) -> dict:
    """Inverse of `encode_table`: codes → physical units (error ≤ half a step, except for saturation)."""
    return {
        "r": codes["r"].astype(float) * table_cfg["rssi_step"],
        "s": codes["s"].astype(float) * table_cfg["slope_step"],
        "std": codes["std"].astype(float) * table_cfg["std_step"],
        "age": codes["age"].astype(float) * table_cfg["age_step"],
    }


def quantize(values: np.ndarray, step: float, limits: tuple) -> np.ndarray:
    """Rounds and saturates like the board; NaNs stay NaN."""
    out = np.clip(np.rint(values / step), limits[0], limits[1]) * step
    return np.where(np.isnan(values), np.nan, out)


def neighbour_knowledge(t: np.ndarray, kres: KalmanResult, table_cfg: dict) -> dict:
    """What observer `m` knows about the link `i → j` from the neighbour tables.

    Beacon payload = posterior estimate quantized as on the board; age = age at transmission
    (quantized) + time since reception (exact). Returns ``r``, ``s``, ``std``, ``age``
    (T, M, N, N) indexed ``[t, m, i, j]``; for ``j == m`` the age is exact (own estimate).
    """
    return neighbour_knowledge_relay(t, kres, table_cfg)[0]


def neighbour_knowledge_relay(t: np.ndarray, kres: KalmanResult, table_cfg: dict) -> tuple:
    """Like `neighbour_knowledge`, but also returns ``relay`` (T, M, N, N) float64: time of the beacon of `j` that forwarded
    the table (NaN for ``j == m``, i.e. the own estimate). Used to recognize a NEW value of the link (Block 5, relative level)."""
    payload = {
        "r": quantize(kres.r, table_cfg["rssi_step"], INT8_RANGE),
        "s": quantize(kres.s, table_cfg["slope_step"], SLOPE_CODE_RANGE),
        "std": quantize(np.sqrt(kres.P[..., 0, 0]), table_cfg["std_step"], _UINT8),
    }
    fields, age, relay = forward_link_payload(t, kres.times, kres.received, payload)
    since_rx = t[:, None, None, None] - relay  # NaN on the direct link
    age_tx = np.maximum(age.astype(float) - since_rx, 0.0)
    age_q = quantize(age_tx, table_cfg["age_step"], _UINT8) + since_rx
    fields["age"] = np.where(np.isnan(relay), age.astype(float), age_q)
    return fields, relay


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


def fuse(
    values: np.ndarray,
    slopes: np.ndarray,
    variances: np.ndarray,
    ages: np.ndarray,
    max_age: float,
    min_variance: float = 0.0,
    rule: str = "weighted_mean",
) -> tuple:
    """Fuses the entries of a target (entries on the LAST axis, `max_age` in s) with rule `rule` (`src/fusion.py`).

    Valid entry: finite fields and ``age <= max_age``. With ``weighted_mean`` (base): weight ``w = 1/max(var, min_variance)``,
    ``L = Σ w r / Σ w``, ``S = Σ w s / Σ w``, variance ``1/Σ w``. Returns (L, S, variance, n valid entries);
    NaN where no entry is valid.
    """
    values = np.asarray(values, dtype=float)
    slopes = np.asarray(slopes, dtype=float)
    variances = np.asarray(variances, dtype=float)
    ages = np.asarray(ages, dtype=float)
    valid = _valid_entries(values, slopes, variances, ages, max_age)
    return combine(values, slopes, variances, valid, rule, min_variance)


def _valid_entries(values, slopes, variances, ages, max_age: float) -> np.ndarray:
    """Valid entries: finite fields and ``age <= max_age``."""
    return np.isfinite(values) & np.isfinite(slopes) & np.isfinite(variances) & np.isfinite(ages) & (ages <= max_age)


def _entry_arrays(t: np.ndarray, kres: KalmanResult, config: dict, sigma_a: float) -> tuple:
    """Entries ``(T, M, I, K)`` ``[t, m, i, k]`` of the link ``i → k`` as known by `m`: (value, slope, variance, age for validity, relay, own age).

    Value ``r + s·age``, variance ``std² + σ_a²·age³/3``; for ``k == m`` the predicted own estimate (`predict_links`).
    ``relay`` is the time of the beacon of `k` that forwarded the entry (NaN for the own one); ``own age`` ``[t, i, m]`` is the age of the own estimate.
    """
    fcfg = config["fusion"]
    tcfg = config["neighbour_table"]
    n_nodes = kres.times.shape[0]
    know, relay = neighbour_knowledge_relay(t, kres, tcfg)
    own = predict_links(kres, t)

    age = know["age"]
    value = know["r"].astype(float) + know["s"].astype(float) * age
    slope = know["s"].astype(float)
    var = know["std"].astype(float) ** 2 + sigma_a**2 * age**3 / 3.0
    mask_age = age.copy()

    own_age = own["age"] if fcfg["max_age_own"] else np.where(np.isnan(own["age"]), np.nan, 0.0)
    for m in range(n_nodes):  # own entry: index k = m
        value[:, m, :, m] = own["r"][:, :, m]
        slope[:, m, :, m] = own["s"][:, :, m]
        var[:, m, :, m] = own["p00"][:, :, m]
        mask_age[:, m, :, m] = own_age[:, :, m]
    return value, slope, var, mask_age, relay, own["age"]


def fusion_indices(
    t: np.ndarray, kres: KalmanResult, config: dict, sigma_a: float | None = None
) -> dict:
    """Level and slope of every target for every observer, fused and unfused (`sigma_a` defaults to the filter's).

    Entries for ``(m, i)``: own estimate ``i → m`` predicted at the current time (`predict_links`) and,
    for each companion `k`, the entry ``i → k`` from the last table of `k` received by `m`. Excluded
    are those with age > ``fusion.max_age`` (the own one only if ``fusion.max_age_own``). Minimum variance
    in the table ``std_step²/12``. Fusion rule: ``fusion.rule`` (`src/fusion.py`). Returns ``fused``
    (all entries) and ``pairwise`` (own only), each ``(L, S, var, n)`` with (T, M, I) arrays
    ``[t, m, i]``; diagonal ``m == i`` NaN.
    """
    fcfg = config["fusion"]
    tcfg = config["neighbour_table"]
    sigma_a = kres.sigma_a if sigma_a is None else sigma_a
    n_nodes = kres.times.shape[0]
    value, slope, var, mask_age, _, _ = _entry_arrays(t, kres, config, sigma_a)

    min_var = tcfg["std_step"] ** 2 / 12.0
    rule = fcfg["rule"]
    fused = fuse(value, slope, var, mask_age, fcfg["max_age"], min_var, rule)

    idx = np.arange(n_nodes)
    own_entry = lambda a: a[:, idx, :, idx].transpose(1, 0, 2)[..., None]  # (T, M, I, 1)
    pair = fuse(
        own_entry(value), own_entry(slope), own_entry(var), own_entry(mask_age), fcfg["max_age"], min_var, rule
    )
    out = {"fused": fused, "pairwise": pair}
    for name, parts in out.items():
        for a in parts:
            a[:, idx, idx] = np.nan if a.dtype.kind == "f" else 0
    return out


def fusion_entries(t: np.ndarray, kres: KalmanResult, config: dict, sigma_a: float | None = None) -> dict:
    """Unfused entries for the relative level (Block 5): ``value``, ``var``, ``valid``, ``new``, all ``(T, M, I, K)`` ``[t, m, i, k]``.

    As in `fusion_indices` (same entries, same validity by age). ``new`` is true at the grid instant when `m` has a NEW
    value of the link ``i → k``: for ``k == m`` a new beacon of `i` received by `m`; for ``k != m`` a new beacon of `k` (with its
    table). The beacon format does not change: `m` derives everything by itself.
    """
    fcfg = config["fusion"]
    sigma_a = kres.sigma_a if sigma_a is None else sigma_a
    n_nodes = kres.times.shape[0]
    value, slope, var, mask_age, relay, own_age = _entry_arrays(t, kres, config, sigma_a)
    valid = _valid_entries(value, slope, var, mask_age, fcfg["max_age"])
    new = np.zeros(value.shape, dtype=bool)
    with np.errstate(invalid="ignore"):
        new[1:] = np.isfinite(relay[1:]) & ~(relay[1:] == relay[:-1])
        new[0] = np.isfinite(relay[0])
        dt = float(t[1] - t[0]) if len(t) > 1 else 0.0
        for m in range(n_nodes):
            age = own_age[:, :, m]  # (T, I)
            fresh = np.zeros(age.shape, dtype=bool)
            fresh[1:] = np.isfinite(age[1:]) & (np.isnan(age[:-1]) | (age[1:] < age[:-1] + dt - TIME_TOLERANCE))
            fresh[0] = np.isfinite(age[0])
            new[:, m, :, m] = fresh
    return {"value": value, "var": var, "valid": valid, "new": new & valid}


# ---------------------------------------------------------------------------
# Direct reception
# ---------------------------------------------------------------------------


def reception_features(t: np.ndarray, times: np.ndarray, received: np.ndarray, config: dict) -> tuple:
    """Silence and recent reception of every direct link ``i → m`` (`times` (N, K), `received` (N, K, N)).

    silence = time since the last reception (from 0 if none); ``rx_ok`` = beacons received in
    ``(t − release_time, t]`` ≥ ``release_reception · release_time / beacon_period`` (nominal period).
    Returns (silence (s), rx_ok) (T, M, I) ``[t, m, i]``; diagonal 0 and False.
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
            rx_ok[:, m, i] = (count_now - count_before) >= needed - TIME_TOLERANCE
    return silence, rx_ok


def deafness(t: np.ndarray, times: np.ndarray, received: np.ndarray, silence: np.ndarray, config: dict) -> tuple:
    """Deafness of the observers (Block 6, D4): receiver `m` that hears nobody, not a node `i` that is truly lost.

    Input: `t`, `times` (N, K), `received` (N, K, N), `silence` (T, M, I) from `reception_features`, `config`.
    Procedure:

    - ``deaf_window`` (T, M): `m` has received beacons from NO neighbour in the last ``lost_silence`` s (literal definition of ``not_deaf``);
    - ``deaf_episode`` (T, M, I): for the silence episode of `i` that started at the last beacon `last` of `i`, `m` has received no beacon from any other
      neighbour in ``(last + beacon_period, last + lost_silence]``. The decision is therefore a property of the episode (constant while `i` is silent) and does not
      depend on the instant: it avoids the "lost" firing for the neighbour with the oldest last beacon (the silence of `m` is still hidden by the beacon
      of another neighbour that arrived less than one period later) or when `m` comes back (it hears a neighbour before `i`). Used only with ``not_deaf_episode``.
    Output: (deaf_window, deaf_episode); diagonal False.
    """
    dcfg = config["detector"]
    silence_limit = dcfg["lost_silence"]
    guard = config["packets"]["beacon_period"]
    n_nodes = times.shape[0]
    off = ~np.eye(n_nodes, dtype=bool)
    deaf_window = np.where(off[None], silence, np.inf).min(axis=2) > silence_limit
    deaf_episode = np.zeros(silence.shape, dtype=bool)
    for m in range(n_nodes):
        heard = {k: times[k, received[k, :, m]] for k in range(n_nodes) if k != m}
        for i in range(n_nodes):
            if i == m:
                continue
            others = np.sort(np.concatenate([heard[k] for k in heard if k != i]))
            last = t - silence[:, m, i]
            count = np.searchsorted(others, last + silence_limit, side="right") - np.searchsorted(others, last + guard, side="right")
            deaf_episode[:, m, i] = count == 0
    return deaf_window, deaf_episode


# ---------------------------------------------------------------------------
# Alarms
# ---------------------------------------------------------------------------

TIME_TOLERANCE = 1e-9  # s; rounding error on duration comparisons


def run_duration(cond: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Duration in s for which `cond` (time on the first axis) has been true without interruption; 0 if false or at the first true sample."""
    shape = (-1,) + (1,) * (cond.ndim - 1)
    idx = np.arange(len(t)).reshape(shape)
    last_false = np.maximum.accumulate(np.where(cond, -1, idx), axis=0)
    start = t[np.minimum(last_false + 1, len(t) - 1)]
    return np.where(cond, t.reshape(shape) - start, 0.0)


def latched_alarm(on: np.ndarray, off: np.ndarray) -> np.ndarray:
    """Hysteresis state (time on the first axis): on if the last `on` is more recent than the last `off`.

    At the same index turning off wins; starts off.
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
    """Pre-alarm of every (observer, target) pair; `mode` and thresholds can be overridden for the sweeps.

    Condition: ``level`` ``L < threshold``, ``slope`` ``S < threshold``, ``both`` both (NaN = false).
    It turns on after ``min_duration``; it turns off after ``release_time`` of return with margin
    (``release_*_margin``; in ``both`` only on the level) and with `rx_ok`. Wider thresholds ⇒ alarms
    in monotonically increasing number (see `src/metrics.py`). `valid` excludes ``m == i``.
    """
    mode = detector_cfg["mode"] if mode is None else mode
    thr_l = detector_cfg["level_threshold"] if level_threshold is None else level_threshold
    thr_s = detector_cfg["slope_threshold"] if slope_threshold is None else slope_threshold
    with np.errstate(invalid="ignore"):
        low_l = level < thr_l
        low_s = slope < thr_s
        back_l = level > thr_l + detector_cfg["release_level_margin"]
        back_s = slope > thr_s + detector_cfg["release_slope_margin"]
    if mode == DetectorMode.LEVEL:
        cond, back = low_l, back_l
    elif mode == DetectorMode.SLOPE:
        cond, back = low_s, back_s
    elif mode == DetectorMode.BOTH:
        cond, back = low_l & low_s, back_l
    else:
        raise ValueError(f"detector.mode must be 'level', 'slope' or 'both', not {mode!r}")
    on = run_duration(cond, t) >= detector_cfg["min_duration"] - TIME_TOLERANCE
    off = (run_duration(back, t) >= detector_cfg["release_time"] - TIME_TOLERANCE) & rx_ok
    state = latched_alarm(on, off)
    return state if valid is None else state & valid


def lost_alarm(
    silence: np.ndarray,
    rx_ok: np.ndarray,
    t: np.ndarray,
    detector_cfg: dict,
    valid: np.ndarray | None = None,
    deaf_window: np.ndarray | None = None,
    deaf_episode: np.ndarray | None = None,
) -> np.ndarray:
    """"Lost node" alarm of every pair: on with silence > ``lost_silence`` s, off by `rx_ok`.

    An isolated beacon does not turn it off. `valid` excludes ``m == i``. ``detector.lost_rule`` (Block 6): ``any`` = base; ``not_deaf`` does not turn on
    the "lost" if `m` hears no neighbour (`deaf_window` (T, M), see `deafness`); ``not_deaf_episode`` uses `deaf_episode` (T, M, I or T, M).
    Turning off does not change.
    """
    on = silence > detector_cfg["lost_silence"]
    rule = detector_cfg.get("lost_rule", "any")
    if rule == LostRule.NOT_DEAF:
        on = on & ~(deaf_window if silence.ndim == 2 else deaf_window[:, :, None])
    elif rule == LostRule.NOT_DEAF_EPISODE:
        on = on & ~deaf_episode
    elif rule != LostRule.ANY:
        raise ValueError(f"detector.lost_rule must be 'any', 'not_deaf' or 'not_deaf_episode', not {rule!r}")
    state = latched_alarm(on, rx_ok)
    return state if valid is None else state & valid


def system_alarm(observer_alarm: np.ndarray, min_observers: int) -> np.ndarray:
    """System alarm (what reaches the group leader): observers in alarm ≥ `min_observers`.

    Input (time, observer, ...); output (time, ...).
    """
    return observer_alarm.sum(axis=1) >= min_observers


def count_episodes(system: np.ndarray, t: np.ndarray, warmup: float) -> np.ndarray:
    """Number of rising edges of the system alarm with ``t[k] >= warmup`` (s), per non-time axis.

    An alarm already on at the end of `warmup` is not an episode.
    """
    return rising_edges(system)[warmup_start(t, warmup) :].sum(axis=0)


# ---------------------------------------------------------------------------
# Detector inputs for a run
# ---------------------------------------------------------------------------


@dataclass
class DetectionInputs:
    """Everything the detector uses from a run, independent of the thresholds.

    The arrays have time on the first axis and, for the whole run,
    ``[t, m, i]`` (observer, target); `target_slice` slices out a single one.
    """

    t: np.ndarray
    level: dict  # scope -> (T, M, I)
    slope: dict
    n_entries: dict
    silence: np.ndarray
    rx_ok: np.ndarray
    valid: np.ndarray  # (M, I) False on the diagonal
    entries: dict | None = None  # Block 5: unfused entries (T, M, I, K) for the relative level (`fusion_entries`)
    deaf_window: np.ndarray | None = None  # Block 6: (T, M), `deafness`
    deaf_episode: np.ndarray | None = None  # Block 6: (T, M, I) whole, (T, M) after `target_slice`

    def lost_kwargs(self) -> dict:
        """Arguments of `lost_alarm` that depend on deafness."""
        return {"deaf_window": self.deaf_window, "deaf_episode": self.deaf_episode}

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
            entries=None if self.entries is None else {k: v[:, :, target, :] for k, v in self.entries.items()},
            deaf_window=self.deaf_window,
            deaf_episode=None if self.deaf_episode is None else self.deaf_episode[sel],
        )


def build_inputs(run: dict, config: dict, sigma_a: float, R: float, kres: KalmanResult | None = None) -> tuple:
    """`filter_all` → `fusion_indices` → `reception_features` on a run (keys as in `PacketResult`).

    `kres` (already computed with the same Kalman parameters) avoids redoing the filter. With ``detector.reference == "relative"``
    it also adds the unfused entries (`fusion_entries`). Returns (`DetectionInputs`, `KalmanResult`).
    """
    if kres is None:
        kres = filter_all(run["beacon_times"], run["received"], run["rssi"], R, sigma_a, config["kalman"])
    t = run["t"]
    indices = fusion_indices(t, kres, config, sigma_a)
    silence, rx_ok = reception_features(t, run["beacon_times"], run["received"], config)
    n_nodes = kres.times.shape[0]
    deaf_window, deaf_episode = deafness(t, run["beacon_times"], run["received"], silence, config)
    inputs = DetectionInputs(
        t=t,
        level={k: v[0] for k, v in indices.items()},
        slope={k: v[1] for k, v in indices.items()},
        n_entries={k: v[3] for k, v in indices.items()},
        silence=silence,
        rx_ok=rx_ok,
        valid=~np.eye(n_nodes, dtype=bool),
        entries=fusion_entries(t, kres, config, sigma_a) if config["detector"]["reference"] == Reference.RELATIVE else None,
        deaf_window=deaf_window,
        deaf_episode=deaf_episode,
    )
    return inputs, kres

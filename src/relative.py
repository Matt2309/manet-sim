"""Block 5 (B) — Relative level: how much the signal has dropped relative to the group.

With ``detector.reference = relative`` the value in dBm is not used, but the DROP relative to a
BASELINE. For each observer `m`, target `i` and link used in the fusion (its own
``i → m`` and the links ``i → k`` forwarded by the companions) `m` computes the baseline by itself: the
beacon format does not change.

- **Baseline**: median of the link estimates over the last ``baseline_window`` s. Only
  ACCEPTED samples count, taken at the instants when `m` has a new value of the link (an
  own update, or a table received from `k`: `fusion_entries`).
- **Freezing**: a sample is accepted only if it has not fallen more than ``freeze_drop`` dB below the
  current baseline and only if `m` has no active alarm (pre-alarm or "lost") on the target. If it
  is not accepted the baseline stays put; it resumes on its own when the signal returns within
  ``freeze_drop`` dB. With fewer than ``baseline_min_samples`` accepted samples in the window the
  previous baseline stays put.
- **Startup**: until a link has a baseline it is excluded from the fusion; if no entry remains,
  there is no alarm.
- **Drop** of the link = baseline − current estimate (positive = weaker signal). The link drops are
  fused with the rule of `src/fusion.py` applied to the index ``−drop`` (``max`` = the smallest
  drop = the best observer).
- **Pre-alarm**: fused drop > ``drop_threshold`` for at least ``min_duration``; it turns off after
  ``release_time`` with fused drop < ``drop_threshold − release_level_margin`` and good reception
  (same logic as `src/detector.py::pre_alarm`, at the same instant turning off wins).

Why it helps: the relative drop cancels the fixed offsets of the boards (the link always has the same
tx/rx pair) and the part of shadowing that changes little between the point where the runner was in the
group and the one where he breaks away. It does not cancel shadowing that changes along the path.

The baseline depends on the alarms (the freezing), so the state is sequential: the loop is in
numba (`@njit`; without numba it runs, much slower, in pure Python).
"""

from __future__ import annotations

import numpy as np

try:  # numba is optional: without it, the same code runs in pure Python
    from numba import njit
except ImportError:  # pragma: no cover

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f


from src.detector import TIME_TOLERANCE, lost_alarm
from src.fusion import RULES

_RULE_CODE = {name: code for code, name in enumerate(RULES)}


@njit(cache=True)
def _run_pairs(
    t, value, var, valid, new, lost, rx_ok, obs, tgt, thresholds, only_own, rule, min_var,
    window, min_samples, freeze_drop, margin, min_duration, release_time, tol, cap, trace,
    out_state, out_drop, out_frozen, out_base,
):  # fmt: skip
    """Relative level on every (observer, target) pair of `obs`/`tgt` and every threshold (see the module docstring).

    `value`, `var`, `valid`, `new`: (T, M, I, K); `lost`, `rx_ok`: (T, M, I); `out_state`: (pairs, thresholds, T) bool.
    With `trace` it also writes `out_drop` (T,), `out_frozen` (T, K) and `out_base` (T, K) (only for the first pair and the first threshold).
    """
    n_steps = len(t)
    n_links = value.shape[3]
    for p in range(len(obs)):
        m = obs[p]
        i = tgt[p]
        for n in range(len(thresholds)):
            thr = thresholds[n]
            base = np.full(n_links, np.nan)
            has_base = np.zeros(n_links, dtype=np.bool_)
            frozen = np.zeros(n_links, dtype=np.bool_)
            buf_t = np.zeros((n_links, cap))
            buf_v = np.zeros((n_links, cap))
            head = np.zeros(n_links, dtype=np.int64)
            count = np.zeros(n_links, dtype=np.int64)
            drops = np.empty(n_links)
            weights = np.empty(n_links)
            state = False
            cond_on = False
            cond_t0 = 0.0
            back_on = False
            back_t0 = 0.0
            for s in range(n_steps):
                # 1. fused drop
                n_used = 0
                for k in range(n_links):
                    if only_own and k != m:
                        continue
                    if valid[s, m, i, k] and has_base[k]:
                        drops[n_used] = base[k] - value[s, m, i, k]
                        weights[n_used] = 1.0 / max(var[s, m, i, k], min_var)
                        n_used += 1
                fused = np.nan
                if n_used > 0:
                    if rule == 0:
                        total = 0.0
                        acc = 0.0
                        for u in range(n_used):
                            total += weights[u]
                            acc += weights[u] * drops[u]
                        fused = acc / total
                    elif rule == 1:
                        ordered = np.sort(drops[:n_used])
                        fused = 0.5 * (ordered[(n_used - 1) // 2] + ordered[n_used // 2])
                    else:
                        fused = drops[0]
                        for u in range(1, n_used):
                            if drops[u] < fused:
                                fused = drops[u]
                # 2. pre-alarm (same logic as `pre_alarm`)
                cond = fused > thr  # NaN -> False
                back = fused < thr - margin
                if cond:
                    if not cond_on:
                        cond_on = True
                        cond_t0 = t[s]
                else:
                    cond_on = False
                if back:
                    if not back_on:
                        back_on = True
                        back_t0 = t[s]
                else:
                    back_on = False
                on = cond_on and (t[s] - cond_t0 >= min_duration - tol)
                off = back_on and (t[s] - back_t0 >= release_time - tol) and rx_ok[s, m, i]
                if off:
                    state = False
                elif on:
                    state = True
                out_state[p, n, s] = state
                # 3. baseline
                active = state or lost[s, m, i]
                for k in range(n_links):
                    if only_own and k != m:
                        continue
                    if new[s, m, i, k]:
                        v = value[s, m, i, k]
                        accept = (not active) and ((not has_base[k]) or (base[k] - v <= freeze_drop))
                        frozen[k] = not accept
                        if accept:
                            while count[k] > 0 and buf_t[k, head[k]] < t[s] - window - tol:
                                head[k] = (head[k] + 1) % cap
                                count[k] -= 1
                            if count[k] == cap:
                                head[k] = (head[k] + 1) % cap
                                count[k] -= 1
                            pos = (head[k] + count[k]) % cap
                            buf_t[k, pos] = t[s]
                            buf_v[k, pos] = v
                            count[k] += 1
                            if count[k] >= min_samples:
                                tmp = np.empty(count[k])
                                for u in range(count[k]):
                                    tmp[u] = buf_v[k, (head[k] + u) % cap]
                                tmp = np.sort(tmp)
                                base[k] = 0.5 * (tmp[(count[k] - 1) // 2] + tmp[count[k] // 2])
                                has_base[k] = True
                if trace and p == 0 and n == 0:
                    out_drop[s] = fused
                    for k in range(n_links):
                        out_frozen[s, k] = frozen[k] and has_base[k]
                        out_base[s, k] = base[k]


def _arrays(inputs, config: dict, scope: str, targets: list | None):
    """Prepares the kernel inputs: 4D entries ``(T, M, I, K)``, "lost" alarm and 3D reception, (observer, target) pairs."""
    dcfg = config["detector"]
    entries = inputs.entries
    if entries is None:
        raise ValueError("unfused entries are required: build the inputs with detector.reference = 'relative'")
    sliced = entries["value"].ndim == 3  # `target_slice`: (T, M, K)
    expand = (lambda a: a[:, :, None, ...]) if sliced else (lambda a: a)
    lost = lost_alarm(inputs.silence, inputs.rx_ok, inputs.t, dcfg, inputs.valid, **inputs.lost_kwargs())
    lost, rx_ok = (lost[:, :, None], inputs.rx_ok[:, :, None]) if sliced else (lost, inputs.rx_ok)
    n_obs, n_tgt = rx_ok.shape[1], rx_ok.shape[2]
    pairs = [(m, i) for i in range(n_tgt) for m in range(n_obs) if sliced or m != i]
    if sliced:
        pairs = [(m, 0) for m in range(n_obs) if inputs.valid[m]]
    return (
        expand(entries["value"]), expand(entries["var"]), expand(entries["valid"]), expand(entries["new"]),
        np.ascontiguousarray(lost), np.ascontiguousarray(rx_ok), pairs, sliced,
    )  # fmt: skip


def _call(inputs, config, scope, thresholds, trace_pair=None):
    dcfg = config["detector"]
    rule = config["fusion"]["rule"]
    if rule not in _RULE_CODE:
        raise ValueError(f"fusion.rule must be one of {RULES}, not {rule!r}")
    value, var, valid, new, lost, rx_ok, pairs, sliced = _arrays(inputs, config, scope, None)
    if trace_pair is not None:
        pairs = [trace_pair]
    t = np.ascontiguousarray(inputs.t, dtype=float)
    dt = float(t[1] - t[0])
    cap = int(dcfg["baseline_window"] / dt) + 2
    n_steps, n_links = len(t), value.shape[3]
    obs = np.array([m for m, _ in pairs], dtype=np.int64)
    tgt = np.array([i for _, i in pairs], dtype=np.int64)
    thresholds = np.ascontiguousarray(thresholds, dtype=float)
    trace = trace_pair is not None
    out_state = np.zeros((len(pairs), len(thresholds), n_steps), dtype=np.bool_)
    out_drop = np.full(n_steps if trace else 1, np.nan)
    out_frozen = np.zeros((n_steps if trace else 1, n_links), dtype=np.bool_)
    out_base = np.full((n_steps if trace else 1, n_links), np.nan)
    min_var = config["neighbour_table"]["std_step"] ** 2 / 12.0
    _run_pairs(
        t, np.ascontiguousarray(value), np.ascontiguousarray(var), np.ascontiguousarray(valid), np.ascontiguousarray(new),
        lost, rx_ok, obs, tgt, thresholds, scope == "pairwise", _RULE_CODE[rule], max(min_var, np.finfo(float).tiny),
        float(dcfg["baseline_window"]), int(dcfg["baseline_min_samples"]), float(dcfg["freeze_drop"]),
        float(dcfg["release_level_margin"]), float(dcfg["min_duration"]), float(dcfg["release_time"]), TIME_TOLERANCE, cap,
        trace, out_state, out_drop, out_frozen, out_base,
    )  # fmt: skip
    return out_state, pairs, sliced, (out_drop, out_frozen, out_base)


def relative_alarms(inputs, config: dict, scope: str, thresholds: np.ndarray) -> np.ndarray:
    """Relative pre-alarm of every (observer, target) pair for every drop threshold.

    Input: `inputs` (`DetectionInputs` with ``entries``, whole or sliced to one target), `scope` (``fused`` or ``pairwise``:
    own link only), `thresholds` (dB).
    Procedure: sequential kernel (module docstring) with the ``fusion.rule`` rule and the ``detector`` parameters.
    Output: bool ``(thresholds, T, M, I)``, or ``(thresholds, T, M)`` if sliced; False on the diagonal.
    """
    out_state, pairs, sliced, _ = _call(inputs, config, scope, thresholds)
    n_thr, n_steps = out_state.shape[1], out_state.shape[2]
    n_obs = inputs.rx_ok.shape[1]
    if sliced:
        states = np.zeros((n_thr, n_steps, n_obs), dtype=bool)
        for p, (m, _) in enumerate(pairs):
            states[:, :, m] = out_state[p]
        return states
    n_tgt = inputs.rx_ok.shape[2]
    states = np.zeros((n_thr, n_steps, n_obs, n_tgt), dtype=bool)
    for p, (m, i) in enumerate(pairs):
        states[:, :, m, i] = out_state[p]
    return states


def relative_trace(inputs, config: dict, scope: str, threshold: float, observer: int, target: int = 0) -> dict:
    """Trace of an (observer, target) pair at one threshold: ``state`` (T,), ``drop`` (T,) fused drop, ``frozen`` (T, K) and ``base`` (T, K) per link.

    `target` is the index in the array (0 if `inputs` is sliced to one target). Used for diagnostics and plots.
    """
    out_state, _, _, (drop, frozen, base) = _call(inputs, config, scope, np.array([threshold]), (observer, target))
    return {"state": out_state[0, 0], "drop": drop, "frozen": frozen, "base": base}

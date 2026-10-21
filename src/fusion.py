"""Block 5 (A) — Fusion rules for the entries of a target.

The entries are the links ``i → k`` (index `k` on the last axis) that observer `m` knows about
target `i`. Each entry has a SIGNAL INDEX (higher = closer) and a slope. With the
absolute level the index is `L` (dBm); with the relative level it is ``−drop``. This way every rule has the
same meaning in both cases.

Rules (``fusion.rule``):

- ``weighted_mean``: weighted mean with ``1/variance`` weights (Block 4);
- ``median``: median of the valid entries; with an even number of entries (4) it is the mean of the two middle ones;
- ``max``: best observer, i.e. the entry with the strongest signal ("if at least one hears it well,
  it is with the group"). The strongest link is usually the one with the closest companion: so
  ``max`` approximates the DISTANCE FROM THE CLOSEST COMPANION, which is exactly the quantity of the
  30 m criterion. The maximum of noisy values is biased upwards (more entries, more bias), but
  the threshold sweep compensates for it.

For ``median`` and ``max`` the slope and variance are those of the chosen entry (``median``: mean
of the two middle ones, or the middle entry if odd).
"""

from __future__ import annotations

import numpy as np

from src.common.enums import FusionRule

RULES = tuple(r.value for r in FusionRule)


def combine(
    index: np.ndarray,
    slope: np.ndarray,
    variance: np.ndarray,
    valid: np.ndarray,
    rule: str = "weighted_mean",
    min_variance: float = 0.0,
) -> tuple:
    """Fuses the valid entries on the LAST axis with rule `rule`.

    Input: `index`, `slope`, `variance` (same shape, entries on the last axis; the values of the
    invalid entries are ignored), `valid` (bool, same shape), `min_variance` (dB², only for the
    ``weighted_mean`` weights).

    Procedure: ``weighted_mean``: ``w = 1/max(var, min_variance)``, ``L = Σ w r / Σ w``,
    ``S = Σ w s / Σ w``, variance ``1/Σ w``. ``median`` and ``max``: see the module docstring.

    Output: (fused index, fused slope, variance, number of valid entries) with the last axis
    removed; NaN (and 0 entries) where no entry is valid.
    """
    index = np.asarray(index, dtype=float)
    slope = np.asarray(slope, dtype=float)
    variance = np.asarray(variance, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    count = valid.sum(axis=-1)
    if rule == FusionRule.WEIGHTED_MEAN:
        floor = max(min_variance, np.finfo(float).tiny)
        weight = np.where(valid, 1.0 / np.maximum(np.where(valid, variance, 1.0), floor), 0.0)
        total = weight.sum(axis=-1)
        has = total > 0.0
        safe = np.where(has, total, 1.0)
        level = np.where(has, (weight * np.where(valid, index, 0.0)).sum(axis=-1) / safe, np.nan)
        fused_slope = np.where(has, (weight * np.where(valid, slope, 0.0)).sum(axis=-1) / safe, np.nan)
        fused_var = np.where(has, 1.0 / safe, np.nan)
        return level, fused_slope, fused_var, count
    has = count > 0
    if rule == FusionRule.MAX:
        pos = np.argmax(np.where(valid, index, -np.inf), axis=-1)[..., None]
        pick = lambda a: np.take_along_axis(a, pos, axis=-1)[..., 0]
        out = (pick(index), pick(slope), pick(variance))
    elif rule == FusionRule.MEDIAN:
        order = np.argsort(np.where(valid, index, np.inf), axis=-1, kind="stable")
        lo = np.maximum((count - 1) // 2, 0)[..., None]
        hi = (count // 2)[..., None]
        hi = np.minimum(hi, order.shape[-1] - 1)
        pos_lo = np.take_along_axis(order, lo, axis=-1)
        pos_hi = np.take_along_axis(order, hi, axis=-1)
        pick = lambda a: 0.5 * (np.take_along_axis(a, pos_lo, axis=-1)[..., 0] + np.take_along_axis(a, pos_hi, axis=-1)[..., 0])
        out = (pick(index), pick(slope), pick(variance))
    else:
        raise ValueError(f"fusion.rule must be one of {RULES}, not {rule!r}")
    return tuple(np.where(has, o, np.nan) for o in out) + (count,)

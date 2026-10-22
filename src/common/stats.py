import numpy as np


def percentile_interval(values: np.ndarray, level: float) -> tuple:
    alpha = 1.0 - level
    lo, hi = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def resample_indices(rng: np.random.Generator, n_items: int, n_samples: int) -> np.ndarray:
    return rng.integers(0, n_items, size=(n_samples, n_items))

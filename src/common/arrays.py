import numpy as np


def run_bounds(mask: np.ndarray) -> tuple:
    padded = np.concatenate([[0], np.asarray(mask).astype(np.int8), [0]])
    edges = np.flatnonzero(np.diff(padded))
    return edges[0::2], edges[1::2]


def rising_edges(mask: np.ndarray) -> np.ndarray:
    return mask[1:] & ~mask[:-1]


def warmup_start(t: np.ndarray, warmup: float) -> int:
    return max(int(np.searchsorted(t, warmup)) - 1, 0)


def nearest_index(grid, value: float) -> int:
    return int(np.argmin(np.abs(np.asarray(grid) - value)))


def off_diagonal(n: int) -> np.ndarray:
    return ~np.eye(n, dtype=bool)

import numpy as np


def seed_sequence(seed: int, *keys: int) -> np.random.SeedSequence:
    return np.random.SeedSequence(seed, spawn_key=keys)


def spawn_rng(seed: int, *keys: int) -> np.random.Generator:
    return np.random.default_rng(seed_sequence(seed, *keys))

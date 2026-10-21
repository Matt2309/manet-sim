import numpy as np


def seed_sequence(seed: int, *keys: int) -> np.random.SeedSequence:
    return np.random.SeedSequence(seed, spawn_key=keys)


def spawn_rng(seed: int, *keys: int) -> np.random.Generator:
    return np.random.default_rng(seed_sequence(seed, *keys))


def bootstrap_rng(config: dict, seed: int | None = None) -> np.random.Generator:
    """Bootstrap generator: ``SeedSequence(simulation.seed, spawn_key=(final.seed_keys.bootstrap,))``, rebuilt at every evaluation (order-independent)."""
    seed = config["simulation"]["seed"] if seed is None else seed
    return spawn_rng(seed, config["final"]["seed_keys"]["bootstrap"])

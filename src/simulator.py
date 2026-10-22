"""Scenarios and runs of the Block 4 experiment.

Each run executes Blocks 1-3 with its own seed; the results the detector
needs are cached on disk, because the Kalman filter changes with
``σ_a`` but Blocks 1-3 do not.

Three scenarios (``metrics.scenarios``): ``none`` (no separation), ``slowdown``
(the chosen node switches to ``separation.target_speed`` with the configured
ramp) and ``stop`` (target speed ``metrics.stop_speed``, e.g.
a fall). In the separation runs the node and the start instant
are drawn from a dedicated generator; ``slowdown`` and ``stop`` (and ``none``)
use the same seeds, nodes and instants, so the comparison is paired.

Parameters: ``simulation``, ``separation``, ``metrics`` sections of the config.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from src.common.enums import Scenario
from src.common.io import load_npz_run, save_npz_atomic
from src.common.parallel import parallel_map
from src.common.rng import seed_sequence, spawn_rng
from src.channel import simulate_channel
from src.mobility import simulate_mobility
from src.packets import simulate_packets

# Config sections that the results of Blocks 1-3 depend on.
CACHE_SECTIONS = ("simulation", "track", "group", "separation", "channel", "packets")
CACHE_VERSION = 1  # to be incremented if the content of the cache files changes


TUNING_KEYS = {"runs": 7, "separation": 6}  # Blocks 4-5: tuning set


def run_seeds(config: dict, key: int = TUNING_KEYS["runs"], n_runs: int | None = None) -> np.ndarray:
    """Seeds (n_runs,) from ``SeedSequence(simulation.seed, spawn_key=(key,))`` (default 7: tuning), the same for all scenarios."""
    ss = seed_sequence(config["simulation"]["seed"], key)
    return ss.generate_state(config["metrics"]["n_runs"] if n_runs is None else n_runs).astype(np.int64)


def draw_separations(config: dict, key: int = TUNING_KEYS["separation"], n_runs: int | None = None) -> tuple:
    """Node (n_runs,) and start instant (n_runs,) s of the separation, uniform, from a dedicated generator (``spawn_key=(key,)``, default 6: tuning)."""
    rng = spawn_rng(config["simulation"]["seed"], key)
    n_runs = config["metrics"]["n_runs"] if n_runs is None else n_runs
    lo, hi = config["metrics"]["separation_start_range"]
    nodes = rng.integers(0, config["group"]["n_nodes"], size=n_runs)
    starts = rng.uniform(lo, hi, size=n_runs)
    return nodes, starts


def scenario_config(config: dict, scenario: str, seed: int, node: int, start: float) -> dict:
    """Copy of the config with the given seed: ``none`` without separation, ``slowdown`` with the config speed, ``stop`` with ``metrics.stop_speed``."""
    cfg = copy.deepcopy(config)
    cfg["simulation"]["seed"] = int(seed)
    sep = cfg["separation"]
    if scenario == Scenario.NONE:
        sep["enabled"] = False
    elif scenario in (Scenario.SLOWDOWN, Scenario.STOP):
        sep["enabled"] = True
        sep["node_id"] = int(node)
        sep["start_time"] = float(start)
        if scenario == Scenario.STOP:
            sep["target_speed"] = cfg["metrics"]["stop_speed"]
    else:
        raise ValueError(f"unknown scenario: {scenario!r}")
    return cfg


def _hash_parts(config: dict, sections: tuple) -> str:
    parts = {name: copy.deepcopy(config[name]) for name in sections}
    rc = parts.get("packets", {}).get("reception")
    if rc is not None and rc.get("background_model", "independent") == "independent":
        rc.pop("background_model", None)  # Block 6: with "independent" the key stays as before (Blocks 4-5 cache valid)
        rc.pop("bursty", None)
    gpx = Path(parts["track"]["gpx_file"])
    parts["track"]["gpx_file"] = hashlib.sha256(gpx.read_bytes()).hexdigest()
    parts["cache_version"] = CACHE_VERSION
    digest = hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return f"{config['simulation']['seed']}_{digest}"


def cache_key(config: dict) -> str:
    """Key ``<seed>_<hash>``: sha256 of the Blocks 1-3 sections only (GPX replaced by its content) and of the version.

    With ``packets.reception.background_model: independent`` the Block 6 options do not enter the hash.
    """
    return _hash_parts(config, CACHE_SECTIONS)


def channel_cache_key(config: dict) -> str:
    """Like `cache_key` but only for Blocks 1-2 (without the ``packets`` section): the Block 6 sensitivity points share it."""
    return _hash_parts(config, tuple(n for n in CACHE_SECTIONS if n != "packets"))


def simulate_pipeline(config: dict) -> tuple:
    mob = simulate_mobility(config)
    ch = simulate_channel(mob, config)
    return mob, ch, simulate_packets(mob, ch, config)


def simulate_run(config: dict) -> dict:
    """`simulate_mobility` → `simulate_channel` → `simulate_packets`, reduced to what the detector needs.

    ``distances`` is float32; ``fading_var`` = off-diagonal fading variance (dB²);
    without separation ``sep_node = -1`` and ``sep_start`` NaN.
    """
    mob, ch, pk = simulate_pipeline(config)
    return _reduce(config, mob.t, mob.distances, off_diagonal_variance(ch.fading, mob.positions.shape[1]), pk)


def off_diagonal_variance(fading, n_nodes: int) -> float:
    return float(np.var(fading[:, ~np.eye(n_nodes, dtype=bool)]))


def separation_fields(config: dict) -> dict:
    sep = config["separation"]
    return {
        "sep_enabled": bool(sep["enabled"]),
        "sep_node": int(sep["node_id"]) if sep["enabled"] else -1,
        "sep_start": float(sep["start_time"]) if sep["enabled"] else float("nan"),
    }


def _reduce(config: dict, t, distances, fading_var: float, pk) -> dict:
    return {
        "t": t,
        "distances": np.asarray(distances).astype(np.float32),
        "beacon_times": pk.beacon_times,
        "received": pk.received,
        "rssi": pk.rssi,
        "fading_var": fading_var,
        **separation_fields(config),
    }


def channel_cache_path(config: dict, cache_dir: Path) -> Path:
    return Path(cache_dir) / "channel" / f"{channel_cache_key(config)}.npz"


def load_or_simulate_channel(config: dict, cache_dir: Path) -> dict:
    """Blocks 1-2 of a run, from the cache (``<cache>/channel/``) or simulated and saved: ``t``, ``distances``, ``rssi_true``,
    ``rssi_measured`` (saved as float32 only if exact), ``fading_var`` and the separation fields. Does not depend on the ``packets`` section."""
    path = channel_cache_path(config, cache_dir)
    if path.exists():
        ch = load_npz_run(path)
        ch["rssi_measured"] = ch["rssi_measured"].astype(np.float64)
        return ch
    mob = simulate_mobility(config)
    res = simulate_channel(mob, config)
    measured = res.rssi_measured
    narrow = measured.astype(np.float32)
    stored = narrow if np.array_equal(narrow.astype(measured.dtype), measured) else measured
    out = {
        "t": mob.t,
        "distances": mob.distances.astype(np.float32),
        "rssi_true": res.rssi_true,
        "rssi_measured": stored,
        "fading_var": off_diagonal_variance(res.fading, mob.positions.shape[1]),
        **separation_fields(config),
    }
    save_npz_atomic(path, out)
    out["rssi_measured"] = measured
    return out


def run_from_channel(config: dict, cache_dir: Path) -> dict:
    """Full run (same shape as `load_or_simulate`) with Block 3 simulated in memory on top of the cached Blocks 1-2.

    Does not write to the run cache: the sensitivity points (which change only Block 3) cannot overwrite the base.
    Also returns ``background_state`` (None with ``independent``) for diagnostics.
    """
    from types import SimpleNamespace

    ch = load_or_simulate_channel(config, cache_dir)
    chres = SimpleNamespace(t=ch["t"], rssi_true=ch["rssi_true"], rssi_measured=ch["rssi_measured"])
    pk = simulate_packets(None, chres, config)
    run = _reduce(config, ch["t"], ch["distances"], ch["fading_var"], pk)
    run["background_state"] = pk.background_state
    return run


def cache_path(config: dict, cache_dir: Path) -> Path:
    return Path(cache_dir) / f"{cache_key(config)}.npz"


def load_or_simulate(config: dict, cache_dir: Path) -> dict:
    """Reads the run from the cache or simulates and saves it (temporary file + rename: no half-written files)."""
    path = cache_path(config, cache_dir)
    if path.exists():
        return load_npz_run(path)
    run = simulate_run(config)
    save_npz_atomic(path, run)
    return run


def _ensure_cached(args: tuple) -> str:
    config, cache_dir, *rest = args
    path = cache_path(config, cache_dir)
    if rest and rest[0]:  # Block 6: also the Blocks 1-2 cache; the run is derived from there (a single pass over the channel)
        load_or_simulate_channel(config, cache_dir)
        if not path.exists():
            run = run_from_channel(config, cache_dir)
            run.pop("background_state", None)
            save_npz_atomic(path, run)
    elif not path.exists():
        load_or_simulate(config, cache_dir)
    return str(path)


def experiment_runs(config: dict, keys: dict | None = None, n_runs: int | None = None) -> list:
    """Runs for each scenario of ``metrics.scenarios`` and index, with seed, node and instant shared across scenarios.

    `keys` = ``{"runs": k, "separation": k}`` (default: tuning, keys 7 and 6) and `n_runs` choose the set of runs.
    Dicts ``scenario``, ``index``, ``seed``, ``node``, ``start``, ``config``.
    """
    keys = TUNING_KEYS if keys is None else keys
    seeds = run_seeds(config, keys["runs"], n_runs)
    nodes, starts = draw_separations(config, keys["separation"], n_runs)
    runs = []
    for scenario in config["metrics"]["scenarios"]:
        for index, (seed, node, start) in enumerate(zip(seeds, nodes, starts)):
            runs.append(
                {
                    "scenario": scenario,
                    "index": index,
                    "seed": int(seed),
                    "node": int(node),
                    "start": float(start),
                    "config": scenario_config(config, scenario, seed, node, start),
                }
            )
    return runs


def ensure_all_cached(runs: list, cache_dir: Path, jobs: int = 1, with_channel=None) -> None:
    """Simulates and caches the missing runs (parallel if ``jobs > 1``; the result does not depend on `jobs`).

    `with_channel` (function ``run -> bool``, Block 6) chooses the runs for which the Blocks 1-2 cache is also saved (`load_or_simulate_channel`).
    """
    tasks = [(r["config"], cache_dir, bool(with_channel and with_channel(r))) for r in runs]
    parallel_map(_ensure_cached, tasks, jobs)

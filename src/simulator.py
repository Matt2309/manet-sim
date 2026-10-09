"""Scenari e corse dell'esperimento del Blocco 4.

Ogni corsa esegue i Blocchi 1-3 con un proprio seme; i risultati che servono
al rilevatore si mettono in cache su disco, perché il Kalman cambia con
``σ_a`` ma i Blocchi 1-3 no.

Tre scenari (``metrics.scenarios``): ``none`` (nessun distacco), ``slowdown``
(il nodo scelto passa a ``separation.target_speed`` con la rampa di
configurazione) e ``stop`` (velocità obiettivo ``metrics.stop_speed``, per
esempio una caduta). Nelle corse con distacco il nodo e l'istante d'inizio
sono estratti da un generatore dedicato; ``slowdown`` e ``stop`` (e ``none``)
usano gli stessi semi, nodi e istanti, quindi il confronto è a coppie.

Parametri: sezioni ``simulation``, ``separation``, ``metrics`` della config.
"""

from __future__ import annotations

import copy
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from src.channel import simulate_channel
from src.mobility import simulate_mobility
from src.packets import simulate_packets

# Sezioni di configurazione da cui dipendono i risultati dei Blocchi 1-3.
CACHE_SECTIONS = ("simulation", "track", "group", "separation", "channel", "packets")
CACHE_VERSION = 1  # da incrementare se cambia il contenuto dei file di cache


def run_seeds(config: dict) -> np.ndarray:
    """Semi (n_runs,) da ``SeedSequence(simulation.seed, spawn_key=(7,))``, uguali per tutti gli scenari."""
    ss = np.random.SeedSequence(config["simulation"]["seed"], spawn_key=(7,))
    return ss.generate_state(config["metrics"]["n_runs"]).astype(np.int64)


def draw_separations(config: dict) -> tuple:
    """Nodo (n_runs,) e istante d'inizio (n_runs,) s del distacco, uniformi, da un generatore dedicato (``spawn_key=(6,)``)."""
    rng = np.random.default_rng(np.random.SeedSequence(config["simulation"]["seed"], spawn_key=(6,)))
    n_runs = config["metrics"]["n_runs"]
    lo, hi = config["metrics"]["separation_start_range"]
    nodes = rng.integers(0, config["group"]["n_nodes"], size=n_runs)
    starts = rng.uniform(lo, hi, size=n_runs)
    return nodes, starts


def scenario_config(config: dict, scenario: str, seed: int, node: int, start: float) -> dict:
    """Copia della config con il seme dato: ``none`` senza separazione, ``slowdown`` con la velocità di config, ``stop`` con ``metrics.stop_speed``."""
    cfg = copy.deepcopy(config)
    cfg["simulation"]["seed"] = int(seed)
    sep = cfg["separation"]
    if scenario == "none":
        sep["enabled"] = False
    elif scenario in ("slowdown", "stop"):
        sep["enabled"] = True
        sep["node_id"] = int(node)
        sep["start_time"] = float(start)
        if scenario == "stop":
            sep["target_speed"] = cfg["metrics"]["stop_speed"]
    else:
        raise ValueError(f"scenario sconosciuto: {scenario!r}")
    return cfg


def cache_key(config: dict) -> str:
    """Chiave ``<seme>_<hash>``: sha256 delle sole sezioni dei Blocchi 1-3 (GPX sostituito dal suo contenuto) e della versione."""
    parts = {name: copy.deepcopy(config[name]) for name in CACHE_SECTIONS}
    gpx = Path(parts["track"]["gpx_file"])
    parts["track"]["gpx_file"] = hashlib.sha256(gpx.read_bytes()).hexdigest()
    parts["cache_version"] = CACHE_VERSION
    digest = hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return f"{config['simulation']['seed']}_{digest}"


def simulate_run(config: dict) -> dict:
    """`simulate_mobility` → `simulate_channel` → `simulate_packets`, ridotti a ciò che serve al rilevatore.

    ``distances`` è float32; ``fading_var`` = varianza del fading fuori diagonale (dB²);
    senza separazione ``sep_node = -1`` e ``sep_start`` NaN.
    """
    mob = simulate_mobility(config)
    ch = simulate_channel(mob, config)
    pk = simulate_packets(mob, ch, config)
    off_diag = ~np.eye(mob.positions.shape[1], dtype=bool)
    sep = config["separation"]
    return {
        "t": mob.t,
        "distances": mob.distances.astype(np.float32),
        "beacon_times": pk.beacon_times,
        "received": pk.received,
        "rssi": pk.rssi,
        "fading_var": float(np.var(ch.fading[:, off_diag])),
        "sep_enabled": bool(sep["enabled"]),
        "sep_node": int(sep["node_id"]) if sep["enabled"] else -1,
        "sep_start": float(sep["start_time"]) if sep["enabled"] else float("nan"),
    }


def cache_path(config: dict, cache_dir: Path) -> Path:
    return Path(cache_dir) / f"{cache_key(config)}.npz"


def load_or_simulate(config: dict, cache_dir: Path) -> dict:
    """Legge la corsa dalla cache o la simula e la salva (file temporaneo + rinomina: niente file a metà)."""
    path = cache_path(config, cache_dir)
    if path.exists():
        with np.load(path) as f:
            run = {k: f[k] for k in f.files}
        for key in ("fading_var", "sep_start"):
            run[key] = float(run[key])
        run["sep_enabled"] = bool(run["sep_enabled"])
        run["sep_node"] = int(run["sep_node"])
        return run
    run = simulate_run(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **run)
    tmp.replace(path)
    return run


def _ensure_cached(args: tuple) -> str:
    config, cache_dir = args
    path = cache_path(config, cache_dir)
    if not path.exists():
        load_or_simulate(config, cache_dir)
    return str(path)


def experiment_runs(config: dict) -> list:
    """Corse per ogni scenario di ``metrics.scenarios`` e indice, con seme, nodo e istante comuni fra scenari.

    Dizionari ``scenario``, ``index``, ``seed``, ``node``, ``start``, ``config``.
    """
    seeds = run_seeds(config)
    nodes, starts = draw_separations(config)
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


def ensure_all_cached(runs: list, cache_dir: Path, jobs: int = 1) -> None:
    """Simula e salva in cache le corse mancanti (parallelo se ``jobs > 1``; il risultato non dipende da `jobs`)."""
    tasks = [(r["config"], cache_dir) for r in runs]
    if jobs <= 1:
        for task in tasks:
            _ensure_cached(task)
        return
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        list(pool.map(_ensure_cached, tasks))

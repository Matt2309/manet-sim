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
from pathlib import Path

import numpy as np

from src.common.parallel import parallel_map
from src.common.rng import seed_sequence, spawn_rng
from src.channel import simulate_channel
from src.mobility import simulate_mobility
from src.packets import simulate_packets

# Sezioni di configurazione da cui dipendono i risultati dei Blocchi 1-3.
CACHE_SECTIONS = ("simulation", "track", "group", "separation", "channel", "packets")
CACHE_VERSION = 1  # da incrementare se cambia il contenuto dei file di cache


TUNING_KEYS = {"runs": 7, "separation": 6}  # Blocchi 4-5: insieme di taratura


def run_seeds(config: dict, key: int = TUNING_KEYS["runs"], n_runs: int | None = None) -> np.ndarray:
    """Semi (n_runs,) da ``SeedSequence(simulation.seed, spawn_key=(key,))`` (default 7: taratura), uguali per tutti gli scenari."""
    ss = seed_sequence(config["simulation"]["seed"], key)
    return ss.generate_state(config["metrics"]["n_runs"] if n_runs is None else n_runs).astype(np.int64)


def draw_separations(config: dict, key: int = TUNING_KEYS["separation"], n_runs: int | None = None) -> tuple:
    """Nodo (n_runs,) e istante d'inizio (n_runs,) s del distacco, uniformi, da un generatore dedicato (``spawn_key=(key,)``, default 6: taratura)."""
    rng = spawn_rng(config["simulation"]["seed"], key)
    n_runs = config["metrics"]["n_runs"] if n_runs is None else n_runs
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


def _hash_parts(config: dict, sections: tuple) -> str:
    parts = {name: copy.deepcopy(config[name]) for name in sections}
    rc = parts.get("packets", {}).get("reception")
    if rc is not None and rc.get("background_model", "independent") == "independent":
        rc.pop("background_model", None)  # Blocco 6: con "independent" la chiave resta quella di prima (cache dei Blocchi 4-5 valida)
        rc.pop("bursty", None)
    gpx = Path(parts["track"]["gpx_file"])
    parts["track"]["gpx_file"] = hashlib.sha256(gpx.read_bytes()).hexdigest()
    parts["cache_version"] = CACHE_VERSION
    digest = hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return f"{config['simulation']['seed']}_{digest}"


def cache_key(config: dict) -> str:
    """Chiave ``<seme>_<hash>``: sha256 delle sole sezioni dei Blocchi 1-3 (GPX sostituito dal suo contenuto) e della versione.

    Con ``packets.reception.background_model: independent`` le opzioni del Blocco 6 non entrano nell'hash.
    """
    return _hash_parts(config, CACHE_SECTIONS)


def channel_cache_key(config: dict) -> str:
    """Come `cache_key` ma solo per i Blocchi 1-2 (senza la sezione ``packets``): i punti di sensibilità del Blocco 6 la condividono."""
    return _hash_parts(config, tuple(n for n in CACHE_SECTIONS if n != "packets"))


def simulate_run(config: dict) -> dict:
    """`simulate_mobility` → `simulate_channel` → `simulate_packets`, ridotti a ciò che serve al rilevatore.

    ``distances`` è float32; ``fading_var`` = varianza del fading fuori diagonale (dB²);
    senza separazione ``sep_node = -1`` e ``sep_start`` NaN.
    """
    mob = simulate_mobility(config)
    ch = simulate_channel(mob, config)
    pk = simulate_packets(mob, ch, config)
    return _reduce(config, mob.t, mob.distances, float(np.var(ch.fading[:, ~np.eye(mob.positions.shape[1], dtype=bool)])), pk)


def _reduce(config: dict, t, distances, fading_var: float, pk) -> dict:
    sep = config["separation"]
    return {
        "t": t,
        "distances": np.asarray(distances).astype(np.float32),
        "beacon_times": pk.beacon_times,
        "received": pk.received,
        "rssi": pk.rssi,
        "fading_var": fading_var,
        "sep_enabled": bool(sep["enabled"]),
        "sep_node": int(sep["node_id"]) if sep["enabled"] else -1,
        "sep_start": float(sep["start_time"]) if sep["enabled"] else float("nan"),
    }


def channel_cache_path(config: dict, cache_dir: Path) -> Path:
    return Path(cache_dir) / "channel" / f"{channel_cache_key(config)}.npz"


def load_or_simulate_channel(config: dict, cache_dir: Path) -> dict:
    """Blocchi 1-2 di una corsa, dalla cache (``<cache>/channel/``) o simulati e salvati: ``t``, ``distances``, ``rssi_true``,
    ``rssi_measured`` (salvato float32 solo se esatto), ``fading_var`` e i campi del distacco. Non dipende dalla sezione ``packets``."""
    path = channel_cache_path(config, cache_dir)
    if path.exists():
        with np.load(path) as f:
            ch = {k: f[k] for k in f.files}
        ch["rssi_measured"] = ch["rssi_measured"].astype(np.float64)
        for key in ("fading_var", "sep_start"):
            ch[key] = float(ch[key])
        ch["sep_enabled"], ch["sep_node"] = bool(ch["sep_enabled"]), int(ch["sep_node"])
        return ch
    mob = simulate_mobility(config)
    res = simulate_channel(mob, config)
    sep = config["separation"]
    measured = res.rssi_measured
    narrow = measured.astype(np.float32)
    stored = narrow if np.array_equal(narrow.astype(measured.dtype), measured) else measured
    out = {
        "t": mob.t,
        "distances": mob.distances.astype(np.float32),
        "rssi_true": res.rssi_true,
        "rssi_measured": stored,
        "fading_var": float(np.var(res.fading[:, ~np.eye(mob.positions.shape[1], dtype=bool)])),
        "sep_enabled": bool(sep["enabled"]),
        "sep_node": int(sep["node_id"]) if sep["enabled"] else -1,
        "sep_start": float(sep["start_time"]) if sep["enabled"] else float("nan"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **out)
    tmp.replace(path)
    out["rssi_measured"] = measured
    return out


def run_from_channel(config: dict, cache_dir: Path) -> dict:
    """Corsa completa (stessa forma di `load_or_simulate`) con il Blocco 3 simulato in memoria sopra i Blocchi 1-2 in cache.

    Non scrive nella cache delle corse: i punti di sensibilità (che cambiano solo il Blocco 3) non possono sovrascrivere la base.
    Restituisce anche ``background_state`` (None con ``independent``) per le diagnostiche.
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
    config, cache_dir, *rest = args
    path = cache_path(config, cache_dir)
    if rest and rest[0]:  # Blocco 6: anche la cache dei Blocchi 1-2; la corsa si ricava da lì (un solo passaggio sul canale)
        load_or_simulate_channel(config, cache_dir)
        if not path.exists():
            run = run_from_channel(config, cache_dir)
            run.pop("background_state", None)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp.npz")
            np.savez(tmp, **run)
            tmp.replace(path)
    elif not path.exists():
        load_or_simulate(config, cache_dir)
    return str(path)


def experiment_runs(config: dict, keys: dict | None = None, n_runs: int | None = None) -> list:
    """Corse per ogni scenario di ``metrics.scenarios`` e indice, con seme, nodo e istante comuni fra scenari.

    `keys` = ``{"runs": k, "separation": k}`` (default: taratura, chiavi 7 e 6) e `n_runs` scelgono l'insieme di corse.
    Dizionari ``scenario``, ``index``, ``seed``, ``node``, ``start``, ``config``.
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
    """Simula e salva in cache le corse mancanti (parallelo se ``jobs > 1``; il risultato non dipende da `jobs`).

    `with_channel` (funzione ``corsa -> bool``, Blocco 6) sceglie le corse per cui si salva anche la cache dei Blocchi 1-2 (`load_or_simulate_channel`).
    """
    tasks = [(r["config"], cache_dir, bool(with_channel and with_channel(r))) for r in runs]
    parallel_map(_ensure_cached, tasks, jobs)

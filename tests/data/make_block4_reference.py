"""Genera `block4_reference.npz`: uscite del Blocco 4 su tre corse brevi (una per scenario).

Va eseguito con il codice del Blocco 4 (commit 16c6180), PRIMA delle modifiche del Blocco 5;
`tests/test_block5.py::test_default_reproduces_block4` confronta in modo esatto.
Uso: python tests/data/make_block4_reference.py
"""

import copy
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.detector import build_inputs  # noqa: E402
from src.metrics import evaluate_run, min_distance_to_group, summarize_none, summarize_separated  # noqa: E402
from src.mobility import load_config  # noqa: E402
from src.simulator import scenario_config, simulate_run  # noqa: E402

DURATION, SEED, NODE, START, SIGMA_A, R = 150.0, 5, 2, 60.0, 0.25, 9.0


def reference_config() -> dict:
    cfg = load_config(_ROOT / "config" / "default.yaml")
    for dotted, value in cfg["history"]["block4"].items():  # Blocco 6: i default sono cambiati, il Blocco 4 si fissa esplicitamente
        node = cfg
        *path, last = dotted.split(".")
        for key in path:
            node = node[key]
        node[last] = value
    cfg["simulation"]["duration"] = DURATION
    cfg["metrics"]["level_threshold_sweep"] = [-90.0, -65.0, 5.0]
    cfg["metrics"]["slope_threshold_sweep"] = [-3.0, -0.5, 0.5]
    return cfg


def reference_outputs(scenario: str, config: dict) -> dict:
    cfg = scenario_config(config, scenario, SEED, NODE, START)
    run = simulate_run(cfg)
    inputs, kres = build_inputs(run, config, SIGMA_A, R)
    t = run["t"]
    out = {"kalman_r": kres.r, "kalman_nis": kres.nis}
    for scope in ("fused", "pairwise"):
        out[f"level_{scope}"] = inputs.level[scope]
        out[f"slope_{scope}"] = inputs.slope[scope]
    if scenario == "none":
        results = evaluate_run(inputs, config, lambda s: summarize_none(s, t, config["detector"]["warmup"]))
    else:
        idx_start = int(np.searchsorted(t, run["sep_start"]))
        distance = min_distance_to_group(run["distances"], run["sep_node"])
        results = evaluate_run(
            inputs.target_slice(run["sep_node"]),
            config,
            lambda s: summarize_separated(s, t, idx_start, distance, config["metrics"]["success_distance"]),
        )
    for key, value in results.items():
        for field, arr in value.items():
            out["|".join(key) + "|" + field] = np.asarray(arr)
    return out


if __name__ == "__main__":
    config = reference_config()
    arrays = {}
    for scenario in ("none", "slowdown", "stop"):
        for name, arr in reference_outputs(scenario, copy.deepcopy(config)).items():
            arrays[f"{scenario}/{name}"] = arr
    np.savez_compressed(Path(__file__).with_name("block4_reference.npz"), **arrays)
    print(f"{len(arrays)} array salvati")

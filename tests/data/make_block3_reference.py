"""Genera `block3_reference.npz`: uscite del Blocco 3 su tre corse brevi (una per scenario).

Va eseguito con il codice del Blocco 3 PRIMA dell'opzione ``background_model`` (commit 0b8c38d);
`tests/test_block6.py::test_independent_model_reproduces_block3` confronta bit per bit.
Uso: python tests/data/make_block3_reference.py
"""

import copy
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.channel import simulate_channel  # noqa: E402
from src.mobility import load_config, simulate_mobility  # noqa: E402
from src.packets import simulate_packets  # noqa: E402
from src.simulator import scenario_config  # noqa: E402

DURATION, SEED, NODE, START = 150.0, 5, 2, 60.0


def reference_config() -> dict:
    cfg = load_config(_ROOT / "config" / "default.yaml")
    cfg["simulation"]["duration"] = DURATION
    return cfg


def reference_outputs(scenario: str, config: dict) -> dict:
    cfg = scenario_config(config, scenario, SEED, NODE, START)
    mob = simulate_mobility(cfg)
    ch = simulate_channel(mob, cfg)
    pk = simulate_packets(mob, ch, cfg)
    return {
        "beacon_times": pk.beacon_times,
        "received": pk.received,
        "rssi": pk.rssi,
        "passed_signal": pk.passed_signal,
        "passed_background": pk.passed_background,
        "p_success": pk.p_success,
        "knowledge_rssi": pk.knowledge_rssi,
        "knowledge_age": pk.knowledge_age,
    }


if __name__ == "__main__":
    config = reference_config()
    arrays = {}
    for scenario in ("none", "slowdown", "stop"):
        for name, arr in reference_outputs(scenario, copy.deepcopy(config)).items():
            arrays[f"{scenario}/{name}"] = arr
    np.savez_compressed(Path(__file__).with_name("block3_reference.npz"), **arrays)
    print(f"{len(arrays)} array salvati")

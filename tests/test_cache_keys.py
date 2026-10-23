from pathlib import Path

import pytest

from src.final import run_sets
from src.mobility import load_config
from src.refinement import block4_config
from src.simulator import cache_path, channel_cache_path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "results" / "cache"
CONFIG_PATH = ROOT / "config" / "default.yaml"

pytestmark = pytest.mark.skipif(not CACHE.exists(), reason="no local cache")


def test_block4_run_keys_match_existing_cache():
    config = block4_config(load_config(CONFIG_PATH))
    tuning, _ = run_sets(config)
    assert tuning
    if not any(cache_path(r["config"], CACHE).exists() for r in tuning):
        pytest.skip("no cache of the Block 4 channel (history.channel_v1) in the local cache")
    assert all(cache_path(r["config"], CACHE).exists() for r in tuning)


def test_block6_run_and_channel_keys_match_existing_cache():
    config = load_config(CONFIG_PATH)
    tuning, verification = run_sets(config)
    runs = tuning + verification
    if not any(cache_path(r["config"], CACHE).exists() for r in runs):
        pytest.skip("no cache of the current channel in the local cache")
    assert all(cache_path(r["config"], CACHE).exists() for r in runs)
    assert sum(channel_cache_path(r["config"], CACHE).exists() for r in runs) == 150

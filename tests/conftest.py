import copy

import pytest

from src.common.config import load_config
from src.common.paths import DEFAULT_CONFIG


@pytest.fixture(scope="session")
def base_config():
    return load_config(DEFAULT_CONFIG)


@pytest.fixture
def cfg(base_config):
    return copy.deepcopy(base_config)

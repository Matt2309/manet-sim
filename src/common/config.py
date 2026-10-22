import copy
from pathlib import Path
from typing import Union

import yaml


def load_config(config: Union[str, Path, dict]) -> dict:
    """Load the config from YAML or a dict (deep copy); `track.gpx_file` is resolved relative to the YAML folder."""
    if isinstance(config, (str, Path)):
        config_path = Path(config).resolve()
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        gpx_path = Path(cfg["track"]["gpx_file"])
        if not gpx_path.is_absolute():
            gpx_path = (config_path.parent / gpx_path).resolve()
        cfg["track"]["gpx_file"] = str(gpx_path)
        return cfg
    return copy.deepcopy(config)


def separation_variants(config: dict) -> tuple:
    cfg_nosep = apply_overrides(config, {"separation.enabled": False})
    cfg_sep = apply_overrides(config, {"separation.enabled": True})
    return cfg_nosep, cfg_sep


def apply_overrides(config: dict, overrides: dict) -> dict:
    """Copy of the config with the ``{"section.key": value}`` replacements (dotted keys)."""
    cfg = copy.deepcopy(config)
    for dotted, value in overrides.items():
        node = cfg
        *path, last = dotted.split(".")
        for key in path:
            node = node[key]
        if last not in node:
            raise KeyError(f"unknown config key: {dotted}")
        node[last] = value
    return cfg

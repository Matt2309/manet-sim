import copy
from pathlib import Path
from typing import Union

import yaml


def load_config(config: Union[str, Path, dict]) -> dict:
    """Carica la config da YAML o dizionario (copia profonda); `track.gpx_file` è risolto rispetto alla cartella del YAML."""
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


def apply_overrides(config: dict, overrides: dict) -> dict:
    """Copia della config con le sostituzioni ``{"sezione.chiave": valore}`` (chiavi puntate)."""
    cfg = copy.deepcopy(config)
    for dotted, value in overrides.items():
        node = cfg
        *path, last = dotted.split(".")
        for key in path:
            node = node[key]
        if last not in node:
            raise KeyError(f"chiave di config sconosciuta: {dotted}")
        node[last] = value
    return cfg

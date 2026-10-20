import pickle
from pathlib import Path

import numpy as np


def save_pickle(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_npz_atomic(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)


def load_npz_run(path: Path) -> dict:
    with np.load(path) as f:
        run = {k: f[k] for k in f.files}
    for key in ("fading_var", "sep_start"):
        run[key] = float(run[key])
    run["sep_enabled"] = bool(run["sep_enabled"])
    run["sep_node"] = int(run["sep_node"])
    return run

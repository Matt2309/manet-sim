from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = REPO_ROOT / "results"
CACHE_DIR = RESULTS_DIR / "cache"
DEFAULT_CONFIG = REPO_ROOT / "config" / "default.yaml"

NOMINAL_PROFILE = "nominal"


def block_dir(block: int, profile: str | None = None) -> Path:
    path = RESULTS_DIR / f"block{block}"
    if profile is not None and profile != NOMINAL_PROFILE:
        path = path / profile
    return path

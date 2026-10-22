import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.config import load_config  # noqa: E402

from src.common.paths import CACHE_DIR, DEFAULT_CONFIG  # noqa: E402
from src.refinement import block4_config  # noqa: E402


def make_parser(
    description: str | None,
    *,
    config: bool = True,
    config_help: str | None = None,
    jobs: bool = False,
    cache_dir: bool = False,
    out_dir: Path | None = None,
    force_help: str | None = None,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    if config:
        parser.add_argument("config", nargs="?", default=str(DEFAULT_CONFIG), help=config_help)
    if jobs:
        parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="parallel processes")
    if cache_dir:
        parser.add_argument("--cache-dir", default=str(CACHE_DIR))
    if out_dir is not None:
        parser.add_argument("--out-dir", default=str(out_dir))
    if force_help is not None:
        parser.add_argument("--force", action="store_true", help=force_help)
    return parser


def load_block4_config(path) -> dict:
    return block4_config(load_config(path))

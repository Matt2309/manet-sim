import argparse
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common.config import apply_overrides, load_config  # noqa: E402

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
    overrides: bool = False,
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
    if overrides:
        parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE", help="config replacement with a dotted key (e.g. channel.tx_power=20); repeatable")
    return parser


def parse_overrides(items: list) -> dict:
    """``["a.b=1", "c.d=[1, 2]"]`` → ``{"a.b": 1, "c.d": [1, 2]}``; values are read as YAML."""
    out = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        out[key.strip()] = yaml.safe_load(value)
    return out


def load_cli_config(args) -> dict:
    """Config from the command line: the file (`args.config`) with the ``--set`` replacements applied (unknown keys are an error)."""
    return apply_overrides(load_config(args.config), parse_overrides(getattr(args, "overrides", [])))


def load_block4_config(path) -> dict:
    return block4_config(load_config(path))

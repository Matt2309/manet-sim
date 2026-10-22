import subprocess

from src.common.paths import REPO_ROOT


def git_output(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True, cwd=REPO_ROOT).stdout


def git_or_unknown(*args: str) -> str:
    try:
        return git_output(*args).strip()
    except Exception:  # noqa: BLE001
        return "unknown"

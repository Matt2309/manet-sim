import importlib.util

import numpy as np

from src.channel import ChannelResult
from src.mobility import MobilityResult


def short_config(cfg: dict, duration: float, drop_sweep: bool = False) -> dict:
    cfg["simulation"]["duration"] = duration
    cfg["metrics"]["level_threshold_sweep"] = [-90.0, -65.0, 5.0]
    if drop_sweep:
        cfg["metrics"]["drop_threshold_sweep"] = [10.0, 30.0, 10.0]
    cfg["detector"]["warmup"] = 5.0
    return cfg


def load_module(name: str, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def static_mobility(n_steps: int, n_nodes: int, dt: float) -> MobilityResult:
    positions = np.zeros((n_steps, n_nodes, 2))
    positions[:, :, 0] = np.arange(n_nodes)[None, :]
    diff = positions[:, :, None, :] - positions[:, None, :, :]
    return MobilityResult(
        t=np.arange(n_steps) * dt,
        positions=positions,
        s_nodes=np.zeros((n_steps, n_nodes)),
        s_centroid=np.zeros(n_steps),
        distances=np.linalg.norm(diff, axis=-1),
        headings=np.tile([1.0, 0.0], (n_steps, n_nodes, 1)),
        lateral_offsets=np.zeros((n_steps, n_nodes)),
        metadata={},
    )


def manual_channel(rssi: np.ndarray, dt: float) -> ChannelResult:
    rssi = np.array(rssi, dtype=float)
    n_steps, n, _ = rssi.shape
    idx = np.arange(n)
    rssi[:, idx, idx] = np.nan
    zeros = np.zeros_like(rssi)
    return ChannelResult(
        t=np.arange(n_steps) * dt,
        rssi_true=rssi,
        rssi_measured=np.round(rssi),
        rssi_path_loss=rssi.copy(),
        shadowing=zeros,
        body_own=zeros,
        body_others=zeros,
        fading=zeros,
        obstacles=zeros,
        tx_offset=np.zeros(n),
        rx_offset=np.zeros(n),
        shadow_field=None,
        metadata={},
    )

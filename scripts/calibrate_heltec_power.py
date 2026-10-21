#!/usr/bin/env python3
"""Calibration of the equivalent power of the ``heltec_run`` profile (Block 7).

Searches for ``channel.tx_power`` such that the median RSSI in the compact 5-node group (scenario none, base channel) equals ``--target`` dBm (−78, October 13 Heltec run).
It is an EQUIVALENT power (EIRP with directional antenna, mounting and bodies), not settable on the ESP32. The value must be written by hand in the config, not recomputed on every run.
"""

import copy

from _cli import make_parser  # noqa: E402
from src.mobility import load_config  # noqa: E402
from src.routing_experiment import group_median_rssi  # noqa: E402


def main() -> None:
    parser = make_parser(__doc__)
    parser.add_argument("--target", type=float, default=-78.0)
    parser.add_argument("--tolerance", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=300.0)
    args = parser.parse_args()
    base = load_config(args.config)
    power = base["channel"]["tx_power"]
    for it in range(5):
        cfg = copy.deepcopy(base)
        cfg["channel"]["tx_power"] = power
        med = group_median_rssi(cfg, args.duration)
        print(f"tx_power {power:+.1f} dBm -> median RSSI in the group {med:.1f} dBm")
        if abs(med - args.target) <= args.tolerance and it > 0:
            break
        power += args.target - med
        power = round(power)
    print(f"\nEquivalent power found: {power:.0f} dBm (median {med:.1f} dBm, target {args.target:g} ± {args.tolerance:g})")


if __name__ == "__main__":
    main()

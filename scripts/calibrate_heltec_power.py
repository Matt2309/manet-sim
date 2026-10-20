#!/usr/bin/env python3
"""Taratura della potenza equivalente del profilo ``heltec_run`` (Blocco 7).

Cerca ``channel.tx_power`` tale che la mediana dell'RSSI nel gruppo da 5 compatti (scenario none, canale base) sia ``--target`` dBm (−78, corsa Heltec del 13 ottobre).
È una potenza EQUIVALENTE (EIRP con antenna direzionale, montaggio e corpi), non impostabile sull'ESP32. Il valore va scritto a mano in config, non ricalcolato a ogni esecuzione.
"""

import argparse
import copy
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.mobility import load_config  # noqa: E402
from src.routing_experiment import group_median_rssi  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default=str(_REPO_ROOT / "config" / "default.yaml"))
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
        print(f"tx_power {power:+.1f} dBm -> mediana RSSI nel gruppo {med:.1f} dBm")
        if abs(med - args.target) <= args.tolerance and it > 0:
            break
        power += args.target - med
        power = round(power)
    print(f"\nPotenza equivalente trovata: {power:.0f} dBm (mediana {med:.1f} dBm, obiettivo {args.target:g} ± {args.tolerance:g})")


if __name__ == "__main__":
    main()

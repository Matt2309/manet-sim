# manet-sim

Python simulator to validate, before the hardware is available, the separation-detection
algorithm of a predictive MANET for monitoring and safety of running groups.
The network is a mesh of ESP32-based wearable devices that detects when a runner
separates from the group by analysing the trend of the RSSI of the radio links.


## Setup

Python 3.13.1 (developed and tested with this version), libraries with exact versions in `requirements.txt`.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Tests

```bash
pytest
```

## Visual inspection

```bash
python scripts/plot_mobility.py                    # uses config/default.yaml
python scripts/plot_mobility.py path/to/config.yaml # alternative configuration
python scripts/plot_channel.py                     # Block 2: RSSI, shadowing, bodies (3 simulations)
python scripts/plot_packets.py                     # Block 3: beacons, reception, node knowledge
python scripts/run_detection_experiment.py         # Block 4: runs, sweeps, results in results/block4/
python scripts/run_detection_experiment.py --jobs 4  # number of processes (default: all cores)
python scripts/plot_detector.py                    # Block 4: figures and console summary
python scripts/run_block5_experiment.py            # Block 5: variants A-F, results in results/block5/
python scripts/run_block5_experiment.py --jobs 8   # number of processes
python scripts/plot_block5.py                      # Block 5: seven figures in results/block5/
python scripts/run_block6_experiment.py            # Block 6: frozen thresholds, verification, verdict, sensitivity (results/block6/)
python scripts/plot_block6.py                      # Block 6: six figures in results/block6/
python scripts/reproduce_all.py                    # EVERYTHING: figures of Blocks 1-7 and results/final_summary.md
python scripts/reproduce_all.py --clean            # same, starting without cache
```

The Block 4-5 scripts restore the Block 4 values (`history.block4` in `config/default.yaml`): since Block 6, the defaults of `config/default.yaml`
describe the final detector (frozen threshold, 30 s release).

`run_detection_experiment.py` caches Blocks 1-3 of every run in `results/cache/`
(key = seed + hash of the configuration used): the first execution is slow, later ones are not.

The figures are saved in `results/blockN/` (folder ignored by git).

## Configuration

All the numeric parameters of the model live in `config/default.yaml` — no
constant is hard-coded in the code.

## Status

- [x] **Block 1 — Mobility**: node trajectories along a real GPX track,
  group formation with a critically damped second-order process and repulsion between runners, separation event.
- [x] **Block 2 — Channel model**: RSSI per ordered pair (distance attenuation,
  correlated shadowing on a shared map, torso and other runners, Rician fading, board offsets,
  saturation and quantization).
- [x] **Block 3 — Packets and beacons**: ESP-NOW beacon times with jitter, reception
  (logistic curve on the sensitivity × background loss), reported RSSI, neighbour table in the beacons
  and knowledge of each node (with age of the information).
- [x] **Block 4 — Detector**: per-link Kalman (continuous random acceleration), 5-byte neighbour
  table, per-target fusion, pre-alarm and "lost node" alarm, system alarm, metrics
  (detection probability, delay, false alarms per hour) and sweeps over `sigma_a`, mode, scope and threshold.
  From the separation onwards the repulsion of Block 1 is one-way: the separated node does not feel it
  (the group goes around it), so its speed is exactly the imposed one (even 0).
- [x] **Block 5 — Refined detector**: five optional changes, each with a default that exactly reproduces
  Block 4 (test `test_default_reproduces_block4`, on a reference generated with the Block 4 code):
  (A) fusion rule `weighted_mean` | `median` | `max` (`src/fusion.py`) and `min_observers`;
  (B) relative level, i.e. drop with respect to a per-link baseline that freezes during the separation
  (`src/relative.py`, sequential loop in numba); (C) slow release (`release_time`); (D) Kalman robustness
  to body jumps (`kalman.robust`: `inflate` | `nis_gate`); (E) independent shadowing instead of shared.
  Each variant is compared with the base pairwise (runs gained and lost) on the same 150 cached runs; the
  selection rule is in `src/refinement.py`. Then the winners are combined (F).
- [x] **Block 6 — Final numbers, sensitivity and wrap-up** (see below)
- [x] Block 7 — Routing

`numba` is a dependency of the relative level only (`pip install -r requirements.txt`); without numba the code
runs in pure Python, much more slowly.

**One-command reproduction:** `python scripts/reproduce_all.py` (about 1.5 minutes with the cache and results already present; with `--clean`
it recomputes everything: cache, experiments of Blocks 4-6, see the printed timings; the cache takes about 5 GB).

| Figure | Script |
| --- | --- |
| Blocks 1-3 (`results/block1/`, `block2/`, `block3/`) | `plot_mobility.py`, `plot_channel.py`, `plot_packets.py` |
| Block 4 (`results/block4/*.png`) | `plot_detector.py` (after `run_detection_experiment.py`) |
| Block 5 (`results/block5/*.png`) | `plot_block5.py` (after `run_block5_experiment.py`) |
| `final_tradeoff.png`, `detection_distance.png`, `detection_delay.png`, `tuning_vs_test.png`, `sensitivity.png`, `burst_example.png` | `plot_block6.py` (after `run_block6_experiment.py`) |
| `results/final_summary.md` | `reproduce_all.py` (generated by the code) |

**Random generators** (`SeedSequence(simulation.seed, spawn_key=(k,))`, or the run seed for Block 3):

| Key | Use |
| --- | --- |
| 0-3 | Block 2: shadowing map, independent shadowing, board offsets, fading (`ss.spawn(4)` of `SeedSequence(seed)`) |
| 4 | Block 3: beacon times |
| 5 | Block 3: reception (signal and background) |
| 6 | Blocks 4-5 (tuning): node and time of the separation |
| 7 | Blocks 4-5 (tuning): run seeds |
| 8 | Block 6 (verification): run seeds |
| 9 | Block 6 (verification): node and time of the separation |
| 10 | Block 3: Gilbert-Elliott bursts (`background_model: bursty`) |
| 11 | Block 6: bootstrap resamplings (my choice, not foreseen by the specification) |

Block 1 uses a single generator `default_rng(seed)` (and `calibration.seed` for the calibration).

# manet-sim

Simulatore Python per validare, prima di avere l'hardware, l'algoritmo di rilevamento
della separazione di una MANET predittiva per il monitoraggio e la sicurezza di gruppi
podistici. La rete è una mesh di dispositivi indossabili basati su ESP32 che rileva
quando un corridore si separa dal gruppo, analizzando l'andamento dell'RSSI dei link
radio.


## Setup

Python 3.13.1 (sviluppato e testato con questa versione), librerie con versioni esatte in `requirements.txt`.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Test

```bash
pytest
```

## Ispezione visiva

```bash
python scripts/plot_mobility.py                    # usa config/default.yaml
python scripts/plot_mobility.py path/al/config.yaml # configurazione alternativa
python scripts/plot_channel.py                     # Blocco 2: RSSI, shadowing, corpi (3 simulazioni)
python scripts/plot_packets.py                     # Blocco 3: beacon, ricezione, conoscenza dei nodi
python scripts/run_detection_experiment.py         # Blocco 4: corse, sweep, risultati in results/block4/
python scripts/run_detection_experiment.py --jobs 4  # numero di processi (default: tutti i core)
python scripts/plot_detector.py                    # Blocco 4: figure e riepilogo a console
python scripts/run_block5_experiment.py            # Blocco 5: varianti A-F, risultati in results/block5/
python scripts/run_block5_experiment.py --jobs 8   # numero di processi
python scripts/plot_block5.py                      # Blocco 5: sette figure in results/block5/
python scripts/run_block6_experiment.py            # Blocco 6: soglie congelate, verifica, verdetto, sensibilità (results/block6/)
python scripts/plot_block6.py                      # Blocco 6: sei figure in results/block6/
python scripts/reproduce_all.py                    # TUTTO: figure dei Blocchi 1-6 e results/final_summary.md
python scripts/reproduce_all.py --clean            # idem, ripartendo senza cache
```

Gli script dei Blocchi 4-5 rimettono i valori del Blocco 4 (`history.block4` in `config/default.yaml`): i default di `config/default.yaml`
descrivono dal Blocco 6 il rilevatore finale (soglia congelata, spegnimento 30 s).

`run_detection_experiment.py` mette in cache i Blocchi 1-3 di ogni corsa in `results/cache/`
(chiave = seme + hash della configurazione usata): la prima esecuzione è lenta, le successive no.

Le figure vengono salvate in `results/blockN/` (cartella ignorata da git).

## Configurazione

Tutti i parametri numerici del modello vivono in `config/default.yaml` — nessuna
costante è cablata nel codice.

## Stato

- [x] **Blocco 1 — Mobilità**: traiettorie dei nodi lungo una traccia GPX reale,
  formazione del gruppo con un processo del secondo ordine a smorzamento critico e repulsione fra i corridori, evento di separazione.
- [x] **Blocco 2 — Modello di canale**: RSSI per coppia ordinata (attenuazione con la distanza,
  shadowing correlato a mappa condivisa, torso e altri corridori, fading di Rice, scarti delle schede,
  saturazione e quantizzazione).
- [x] **Blocco 3 — Pacchetti e beacon**: istanti dei beacon ESP-NOW con jitter, ricezione
  (curva logistica sulla sensibilità × perdita di fondo), RSSI riportato, tabella dei vicini nei beacon
  e conoscenza di ogni nodo (con età dell'informazione).
- [x] **Blocco 4 — Rilevatore**: Kalman per link (accelerazione casuale continua), tabella dei vicini
  a 5 byte, fusione per bersaglio, pre-allarme e allarme "nodo perso", allarme di sistema, metriche
  (probabilità di rilevamento, ritardo, falsi allarmi per ora) e sweep su `sigma_a`, modalità, ambito e soglia.
  Dal distacco in poi la repulsione del Blocco 1 è a senso unico: il nodo separato non la subisce
  (il gruppo lo aggira), così la sua velocità è esattamente quella imposta (anche 0).
- [x] **Blocco 5 — Rilevatore rifinito**: cinque modifiche opzionali, ognuna con default che riproduce
  esattamente il Blocco 4 (test `test_default_reproduces_block4`, su un riferimento generato con il codice del
  Blocco 4): (A) regola di fusione `weighted_mean` | `median` | `max` (`src/fusion.py`) e `min_observers`;
  (B) livello relativo, cioè calo rispetto a una linea di base per link che si congela durante il distacco
  (`src/relative.py`, ciclo sequenziale in numba); (C) spegnimento lento (`release_time`); (D) robustezza del
  Kalman ai salti da corpo (`kalman.robust`: `inflate` | `nis_gate`); (E) shadowing indipendente invece che condiviso.
  Ogni variante si confronta con la base a coppie (corse guadagnate e perse) sulle stesse 150 corse in cache; la
  regola di scelta è in `src/refinement.py`. Poi si combinano le vincenti (F).
- [x] **Blocco 6 — Numeri finali, sensibilità e chiusura** (vedi sotto)
- [x] Blocco 7 — Routing

`numba` è una dipendenza del solo livello relativo (`pip install -r requirements.txt`); senza numba il codice
gira in Python puro, molto più lento.

**Riproduzione con un comando:** `python scripts/reproduce_all.py` (circa 1,5 minuti con la cache e i risultati già presenti; con `--clean`
ricalcola tutto: cache, esperimenti dei Blocchi 4-6, vedi i tempi stampati; la cache occupa circa 5 GB).

| Figura | Script |
| --- | --- |
| Blocchi 1-3 (`results/block1/`, `block2/`, `block3/`) | `plot_mobility.py`, `plot_channel.py`, `plot_packets.py` |
| Blocco 4 (`results/block4/*.png`) | `plot_detector.py` (dopo `run_detection_experiment.py`) |
| Blocco 5 (`results/block5/*.png`) | `plot_block5.py` (dopo `run_block5_experiment.py`) |
| `final_tradeoff.png`, `detection_distance.png`, `detection_delay.png`, `tuning_vs_test.png`, `sensitivity.png`, `burst_example.png` | `plot_block6.py` (dopo `run_block6_experiment.py`) |
| `results/final_summary.md` | `reproduce_all.py` (generato dal codice) |

**Generatori casuali** (`SeedSequence(simulation.seed, spawn_key=(k,))`, o il seme della corsa per i Blocchi 3):

| Chiave | Uso |
| --- | --- |
| 0-3 | Blocco 2: mappa di shadowing, shadowing indipendente, scarti delle schede, fading (`ss.spawn(4)` di `SeedSequence(seed)`) |
| 4 | Blocco 3: istanti dei beacon |
| 5 | Blocco 3: ricezione (segnale e fondo) |
| 6 | Blocchi 4-5 (taratura): nodo e istante del distacco |
| 7 | Blocchi 4-5 (taratura): semi delle corse |
| 8 | Blocco 6 (verifica): semi delle corse |
| 9 | Blocco 6 (verifica): nodo e istante del distacco |
| 10 | Blocco 3: raffiche di Gilbert-Elliott (`background_model: bursty`) |
| 11 | Blocco 6: ricampionamenti del bootstrap (scelta mia, non prevista dalla specifica) |

Il Blocco 1 usa un generatore unico `default_rng(seed)` (e `calibration.seed` per la calibrazione).

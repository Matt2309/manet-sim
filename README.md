# manet-sim

Simulatore Python per validare, prima di avere l'hardware, l'algoritmo di rilevamento
della separazione di una MANET predittiva per il monitoraggio e la sicurezza di gruppi
podistici. La rete è una mesh di dispositivi indossabili basati su ESP32 che rileva
quando un corridore si separa dal gruppo, analizzando l'andamento dell'RSSI dei link
radio.


## Setup

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
python scripts/run_detection_experiment.py         # Blocco 4: corse, sweep, risultati in results/detection/
python scripts/run_detection_experiment.py --jobs 4  # numero di processi (default: tutti i core)
python scripts/plot_detector.py                    # Blocco 4: figure e riepilogo a console
python scripts/run_block5_experiment.py            # Blocco 5: varianti A-F, risultati in results/block5/
python scripts/run_block5_experiment.py --jobs 8   # numero di processi
python scripts/plot_block5.py                      # Blocco 5: sette figure in results/block5/
```

`run_detection_experiment.py` mette in cache i Blocchi 1-3 di ogni corsa in `results/cache/`
(chiave = seme + hash della configurazione usata): la prima esecuzione è lenta, le successive no.

Le figure vengono salvate in `results/` (cartella ignorata da git).

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
- [ ] Blocco 6 — Valutazione dell'algoritmo di rilevamento (corse nuove, altri semi)
- [ ] Blocco 7 — Routing (era il Blocco 5 nell'elenco originale: slitta dopo la rifinitura del rilevatore)

### Nota sul Blocco 5 (da riportare nella tesi)

Le varianti, le soglie e la configurazione finale del Blocco 5 sono scelte confrontando molte alternative
sulle **stesse 150 corse**: i numeri di questo blocco sono quindi **ottimistici**. Con 50 corse per scenario una
differenza di 1-2 corse sta dentro il rumore, e per questo una variante vince solo se guadagna almeno
`refinement.min_net_gain_runs` corse nette. I numeri finali verranno dal Blocco 6, su **corse nuove** (altri semi).

`numba` è una dipendenza del solo livello relativo (`pip install -r requirements.txt`); senza numba il codice
gira in Python puro, molto più lento.

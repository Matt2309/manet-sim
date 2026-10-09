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
- [ ] Blocco 5 — Routing
- [ ] Blocco 6 — Valutazione dell'algoritmo di rilevamento

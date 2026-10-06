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
```

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
- [ ] Blocco 4 — Filtro di Kalman
- [ ] Blocco 5 — Routing
- [ ] Blocco 6 — Valutazione dell'algoritmo di rilevamento

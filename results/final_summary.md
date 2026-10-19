# Riepilogo finale (generato da `scripts/reproduce_all.py`)

Commit: `0b8c38d1876709e661b4c4ed607936d4f8e6f02c` (con modifiche non committate)  
Python 3.13.1; numpy 2.5.3, scipy 1.18.1, matplotlib 3.11.2, PyYAML 6.0.3, numba 0.68.0, llvmlite 0.50.0, pytest 9.1.1

## Configurazione finale

- configurazione: **media** (ambito `fused`, regola di fusione `weighted_mean`)
- soglia di livello congelata: **-75 dBm** (taratura: 2026-10-19T14:44:40, commit `0b8c38d1876709e661b4c4ed607936d4f8e6f02c`)
- spegnimento: 30 s; `sigma_a` 0.25; `min_observers` 1; `lost_rule` any; shadowing `field`

## Soglie congelate (sulla taratura)

| configurazione | soglia (dBm) | P_d slowdown | P_d stop | FA/h | tempo in allarme |
| --- | --- | --- | --- | --- | --- |
| media | -75 | 1.00 | 0.96 | 0.96 | 0.20% |
| migliore | -67.5 | 1.00 | 0.98 | 0.77 | 0.15% |
| coppia | -87.5 | 0.74 | 0.70 | 0.98 | 0.19% |

## Valutazione sulle corse di verifica

```
config    sc.      soglia   P_d 30 m [Wilson]   P_d 30 s [Wilson]  rit.med/p90 s  dist.med/p90 m  tar.  diff.
media     slowdown  -75.0    0.99 [0.95-1.00]    0.71 [0.61-0.79]   24.4/36.9       13.9/23.2     1.00  -0.01
media     stop      -75.0    1.00 [0.96-1.00]    1.00 [0.96-1.00]   12.0/16.5       17.1/23.7     0.96  +0.04
migliore  slowdown  -67.5    1.00 [0.96-1.00]    0.70 [0.60-0.78]   23.8/35.4       14.5/19.9     1.00  +0.00
migliore  stop      -67.5    1.00 [0.96-1.00]    1.00 [0.96-1.00]   12.3/17.4       18.2/23.8     0.98  +0.02
coppia    slowdown  -87.5    0.74 [0.65-0.82]    0.43 [0.34-0.53]   32.5/49.2       24.7/40.2     0.74  +0.00
coppia    stop      -87.5    0.63 [0.53-0.72]    0.98 [0.93-0.99]   15.2/21.5       26.4/44.7     0.70  -0.07
perso     slowdown    nan    0.00 [0.00-0.04]    0.00 [0.00-0.04]  142.8/218.5     130.1/194.4    0.00  +0.00
perso     stop        nan    0.00 [0.00-0.04]    0.00 [0.00-0.04]   70.0/108.9     163.8/271.7    0.00  +0.00

config           FA/h Poisson      FA/h bootstrap livello/h  perso/h in allarme FA/h taratura  diff.  torti (corse/n)  ripassaggi (spenti)
media        0.92 [0.73-1.15]    0.92 [0.71-1.14]      0.92     0.00      0.19%          0.96  -0.04        0/0                19 (0)
migliore     0.86 [0.67-1.08]    0.86 [0.69-1.04]      0.86     0.00      0.18%          0.77  +0.10        0/0                19 (0)
coppia       0.87 [0.68-1.10]    0.87 [0.59-1.20]      0.87     0.00      0.18%          0.98  -0.11        1/1                19 (0)
perso        0.00 [0.00-0.04]    0.00 [0.00-0.00]      0.00     0.00      0.00%          0.00  +0.00        8/8                 8 (4)
Target falsi allarmi: 1/h. Rispettato da tutte le configurazioni sulla verifica.
```

## Verdetto media contro migliore

netto = +1 (migliore guadagna 1, perde 0; soglia 3); FA/h media 0.92, migliore 0.86; McNemar esatto p = 1.000 (informativo) -> finale = media

## Sensibilità

```
punto       regola             soglia      P_d slowdown          P_d stop  g/p sl  g/p st  dist.med/p90 sl  dist.med/p90 st    FA/h [bootstrap] livello  perso in all.
base        any              congelata -75.0  1.00 [0.93-1.00]  1.00 [0.93-1.00]   +0/-0   +0/-0    13.4/22.8       17.4/23.2       0.96 [0.69-1.24]    0.96   0.00   0.20%
base        any               ritarata -75.0  1.00 [0.93-1.00]  1.00 [0.93-1.00]   +0/-0   +0/-0    13.4/22.8       17.4/23.2       0.96 [0.69-1.24]    0.96   0.00   0.20%
base        not_deaf         congelata -75.0  1.00 [0.93-1.00]  1.00 [0.93-1.00]   +0/-0   +0/-0    13.4/22.8       17.4/23.2       0.96 [0.69-1.24]    0.96   0.00   0.20%
base        not_deaf          ritarata -75.0  1.00 [0.93-1.00]  1.00 [0.93-1.00]   +0/-0   +0/-0    13.4/22.8       17.4/23.2       0.96 [0.69-1.24]    0.96   0.00   0.20%
base        not_deaf_episode congelata -75.0  1.00 [0.93-1.00]  1.00 [0.93-1.00]   +0/-0   +0/-0    13.4/22.8       17.4/23.2       0.96 [0.69-1.24]    0.96   0.00   0.20%
base        not_deaf_episode  ritarata -75.0  1.00 [0.93-1.00]  1.00 [0.93-1.00]   +0/-0   +0/-0    13.4/22.8       17.4/23.2       0.96 [0.69-1.24]    0.96   0.00   0.20%
sens_96     any              congelata -75.0  1.00 [0.93-1.00]  1.00 [0.93-1.00]   +0/-0   +0/-0    13.4/22.8       17.4/23.2       0.96 [0.69-1.24]    0.96   0.00   0.20%
sens_96     any               ritarata -75.0  1.00 [0.93-1.00]  1.00 [0.93-1.00]   +0/-0   +0/-0    13.4/22.8       17.4/23.2       0.96 [0.69-1.24]    0.96   0.00   0.20%
bursty_2_1  any              congelata -75.0  1.00 [0.93-1.00]  1.00 [0.93-1.00]   +0/-0   +0/-0    13.6/22.8       17.4/23.2       1.17 [0.84-1.53]    1.05   0.12   0.27%
bursty_2_1  any               ritarata -77.5  0.96 [0.87-0.99]  0.92 [0.81-0.97]   +0/-2   +0/-4    15.9/25.9       20.2/26.9       0.72 [0.50-0.96]    0.60   0.12   0.17%
bursty_2_1  not_deaf         congelata -75.0  1.00 [0.93-1.00]  1.00 [0.93-1.00]   +0/-0   +0/-0    13.6/22.8       17.4/23.2       1.17 [0.84-1.53]    1.05   0.12   0.27%
bursty_2_1  not_deaf          ritarata -77.5  0.96 [0.87-0.99]  0.92 [0.81-0.97]   +0/-2   +0/-4    15.9/25.9       20.2/26.9       0.72 [0.50-0.96]    0.60   0.12   0.17%
bursty_2_1  not_deaf_episode congelata -75.0  1.00 [0.93-1.00]  1.00 [0.93-1.00]   +0/-0   +0/-0    13.6/22.8       17.4/23.2       1.08 [0.77-1.39]    1.05   0.02   0.25%
bursty_2_1  not_deaf_episode  ritarata -77.5  0.96 [0.87-0.99]  0.92 [0.81-0.97]   +0/-2   +0/-4    15.9/25.9       20.2/26.9       0.62 [0.45-0.79]    0.60   0.02   0.15%
bursty_2_3  any              congelata -75.0  1.00 [0.93-1.00]  1.00 [0.93-1.00]   +0/-0   +0/-0    13.0/22.8       17.4/23.5    20.29 [17.58-23.25]    1.00  19.28   3.20%
bursty_2_3  any               ritarata  nessun punto ammissibile
bursty_2_3  not_deaf         congelata -75.0  1.00 [0.93-1.00]  1.00 [0.93-1.00]   +0/-0   +0/-0    13.0/23.2       17.4/23.5    18.47 [15.98-21.17]    1.03  17.44   2.84%
bursty_2_3  not_deaf          ritarata  nessun punto ammissibile
bursty_2_3  not_deaf_episode congelata -75.0  1.00 [0.93-1.00]  1.00 [0.93-1.00]   +0/-0   +0/-0    13.7/23.2       17.4/23.5       2.11 [1.65-2.58]    1.05   1.05   0.41%
bursty_2_3  not_deaf_episode  ritarata  nessun punto ammissibile
bursty_5_3  any              congelata -75.0  1.00 [0.93-1.00]  1.00 [0.93-1.00]   +0/-0   +0/-0    12.4/21.7       17.2/23.5    48.47 [44.57-52.51]    1.10  47.37   8.14%
bursty_5_3  any               ritarata  nessun punto ammissibile
bursty_5_3  not_deaf         congelata -75.0  1.00 [0.93-1.00]  1.00 [0.93-1.00]   +0/-0   +0/-0    12.4/21.7       17.4/23.5    44.38 [40.67-48.23]    1.12  43.25   7.23%
bursty_5_3  not_deaf          ritarata  nessun punto ammissibile
bursty_5_3  not_deaf_episode congelata -75.0  1.00 [0.93-1.00]  1.00 [0.93-1.00]   +0/-0   +0/-0    13.6/22.8       17.5/23.5       3.88 [3.30-4.47]    1.17   2.70   0.75%
bursty_5_3  not_deaf_episode  ritarata  nessun punto ammissibile
pessimista  any              congelata -75.0  1.00 [0.93-1.00]  1.00 [0.93-1.00]   +0/-0   +0/-0    12.4/21.7       17.2/23.5    48.44 [44.52-52.51]    1.08  47.37   8.13%
pessimista  any               ritarata  nessun punto ammissibile
pessimista  not_deaf         congelata -75.0  1.00 [0.93-1.00]  1.00 [0.93-1.00]   +0/-0   +0/-0    12.4/21.7       17.4/23.5    44.35 [40.62-48.21]    1.10  43.25   7.22%
pessimista  not_deaf          ritarata  nessun punto ammissibile
pessimista  not_deaf_episode congelata -75.0  1.00 [0.93-1.00]  1.00 [0.93-1.00]   +0/-0   +0/-0    13.6/22.8       17.5/23.5       3.85 [3.28-4.47]    1.15   2.70   0.74%
pessimista  not_deaf_episode  ritarata  nessun punto ammissibile

punto        beacon ricevuti raffiche > 10 s osservate   attese  episodi di sordità
base                  97.99%                        -        -                   0
sens_96               97.99%                        -        -                   0
bursty_2_1            97.89%                        1      0.7                   1
bursty_2_3            97.94%                      177    179.5                 188
bursty_5_3            94.91%                      463    448.8                 490
pessimista            94.91%                      463    448.8                 490
```

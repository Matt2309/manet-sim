"""Blocco 5 (A) — Regole di fusione delle voci di un bersaglio.

Le voci sono i link ``i → k`` (indice `k` sull'ultimo asse) che l'osservatore `m` conosce del
bersaglio `i`. Ogni voce ha un INDICE DI SEGNALE (più alto = più vicino) e una pendenza. Con il
livello assoluto l'indice è `L` (dBm); con il livello relativo è ``−calo``. Così ogni regola ha lo
stesso significato nei due casi.

Regole (``fusion.rule``):

- ``weighted_mean``: media pesata con pesi ``1/varianza`` (il Blocco 4);
- ``median``: mediana delle voci valide; con un numero pari di voci (4) è la media delle due centrali;
- ``max``: miglior osservatore, cioè la voce con il segnale più forte ("se almeno uno lo sente bene,
  è con il gruppo"). Il collegamento più forte è di solito quello col compagno più vicino: quindi
  ``max`` approssima la DISTANZA DAL COMPAGNO PIÙ VICINO, che è proprio la grandezza del criterio
  dei 30 m. Il massimo di valori rumorosi è distorto verso l'alto (più voci, più distorsione), ma
  lo sweep della soglia lo compensa.

Per ``median`` e ``max`` la pendenza e la varianza sono quelle della voce scelta (``median``: media
delle due centrali, o la voce centrale se dispari).
"""

from __future__ import annotations

import numpy as np

RULES = ("weighted_mean", "median", "max")


def combine(
    index: np.ndarray,
    slope: np.ndarray,
    variance: np.ndarray,
    valid: np.ndarray,
    rule: str = "weighted_mean",
    min_variance: float = 0.0,
) -> tuple:
    """Fonde le voci valide sull'ULTIMO asse con la regola `rule`.

    Input: `index`, `slope`, `variance` (stessa forma, voci sull'ultimo asse; i valori delle voci
    non valide sono ignorati), `valid` (bool, stessa forma), `min_variance` (dB², solo per i pesi
    di ``weighted_mean``).

    Procedimento: ``weighted_mean``: ``w = 1/max(var, min_variance)``, ``L = Σ w r / Σ w``,
    ``S = Σ w s / Σ w``, varianza ``1/Σ w``. ``median`` e ``max``: vedi la docstring del modulo.

    Output: (indice fuso, pendenza fusa, varianza, numero di voci valide) con l'ultimo asse
    eliminato; NaN (e 0 voci) dove nessuna voce è valida.
    """
    index = np.asarray(index, dtype=float)
    slope = np.asarray(slope, dtype=float)
    variance = np.asarray(variance, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    count = valid.sum(axis=-1)
    if rule == "weighted_mean":
        floor = max(min_variance, np.finfo(float).tiny)
        weight = np.where(valid, 1.0 / np.maximum(np.where(valid, variance, 1.0), floor), 0.0)
        total = weight.sum(axis=-1)
        has = total > 0.0
        safe = np.where(has, total, 1.0)
        level = np.where(has, (weight * np.where(valid, index, 0.0)).sum(axis=-1) / safe, np.nan)
        fused_slope = np.where(has, (weight * np.where(valid, slope, 0.0)).sum(axis=-1) / safe, np.nan)
        fused_var = np.where(has, 1.0 / safe, np.nan)
        return level, fused_slope, fused_var, count
    has = count > 0
    if rule == "max":
        pos = np.argmax(np.where(valid, index, -np.inf), axis=-1)[..., None]
        pick = lambda a: np.take_along_axis(a, pos, axis=-1)[..., 0]
        out = (pick(index), pick(slope), pick(variance))
    elif rule == "median":
        order = np.argsort(np.where(valid, index, np.inf), axis=-1, kind="stable")
        lo = np.maximum((count - 1) // 2, 0)[..., None]
        hi = (count // 2)[..., None]
        hi = np.minimum(hi, order.shape[-1] - 1)
        pos_lo = np.take_along_axis(order, lo, axis=-1)
        pos_hi = np.take_along_axis(order, hi, axis=-1)
        pick = lambda a: 0.5 * (np.take_along_axis(a, pos_lo, axis=-1)[..., 0] + np.take_along_axis(a, pos_hi, axis=-1)[..., 0])
        out = (pick(index), pick(slope), pick(variance))
    else:
        raise ValueError(f"fusion.rule deve essere uno fra {RULES}, non {rule!r}")
    return tuple(np.where(has, o, np.nan) for o in out) + (count,)

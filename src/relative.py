"""Blocco 5 (B) — Livello relativo: quanto è calato il segnale rispetto al gruppo.

Con ``detector.reference = relative`` non si guarda il valore in dBm ma il CALO rispetto a una
LINEA DI BASE. Per ogni osservatore `m`, bersaglio `i` e link usato nella fusione (il proprio
``i → m`` e i link ``i → k`` inoltrati dai compagni) `m` calcola da sé la base: il formato del
beacon non cambia.

- **Linea di base**: mediana delle stime del link negli ultimi ``baseline_window`` s. Contano solo
  i campioni ACCETTATI, presi agli istanti in cui `m` ha un valore nuovo del link (un
  aggiornamento proprio, oppure una tabella ricevuta da `k`: `fusion_entries`).
- **Congelamento**: un campione è accettato solo se non è sceso più di ``freeze_drop`` dB sotto la
  base corrente e solo se `m` non ha un allarme attivo (pre-allarme o "perso") sul bersaglio. Se
  non è accettato la base resta ferma; riprende da sola quando il segnale torna entro
  ``freeze_drop`` dB. Con meno di ``baseline_min_samples`` campioni accettati in finestra la base
  precedente resta ferma.
- **Avvio**: finché un link non ha una base è escluso dalla fusione; se non resta nessuna voce,
  non c'è allarme.
- **Calo** del link = base − stima attuale (positivo = segnale più debole). I cali dei link si
  fondono con la regola di `src/fusion.py` applicata all'indice ``−calo`` (``max`` = il calo più
  piccolo = il miglior osservatore).
- **Pre-allarme**: calo fuso > ``drop_threshold`` per almeno ``min_duration``; si spegne dopo
  ``release_time`` con calo fuso < ``drop_threshold − release_level_margin`` e buona ricezione
  (stessa logica di `src/detector.py::pre_alarm`, a pari istante vince lo spegnimento).

Perché aiuta: il calo relativo annulla gli scarti fissi delle schede (il link ha sempre la stessa
coppia tx/rx) e la parte di shadowing che cambia poco fra il punto in cui il corridore stava nel
gruppo e quello in cui si stacca. Lo shadowing che cambia lungo il percorso non lo annulla.

La base dipende dagli allarmi (il congelamento), quindi lo stato è sequenziale: il ciclo è in
numba (`@njit`; senza numba gira, molto più lento, in Python puro).
"""

from __future__ import annotations

import numpy as np

try:  # numba è opzionale: senza, lo stesso codice gira in Python puro
    from numba import njit
except ImportError:  # pragma: no cover

    def njit(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda f: f


from src.detector import _TIME_TOLERANCE, lost_alarm
from src.fusion import RULES

_RULE_CODE = {name: code for code, name in enumerate(RULES)}


@njit(cache=True)
def _run_pairs(
    t, value, var, valid, new, lost, rx_ok, obs, tgt, thresholds, only_own, rule, min_var,
    window, min_samples, freeze_drop, margin, min_duration, release_time, tol, cap, trace,
    out_state, out_drop, out_frozen, out_base,
):  # fmt: skip
    """Livello relativo su ogni coppia (osservatore, bersaglio) di `obs`/`tgt` e ogni soglia (vedi la docstring del modulo).

    `value`, `var`, `valid`, `new`: (T, M, I, K); `lost`, `rx_ok`: (T, M, I); `out_state`: (coppie, soglie, T) bool.
    Con `trace` scrive anche `out_drop` (T,), `out_frozen` (T, K) e `out_base` (T, K) (solo per la prima coppia e la prima soglia).
    """
    n_steps = len(t)
    n_links = value.shape[3]
    for p in range(len(obs)):
        m = obs[p]
        i = tgt[p]
        for n in range(len(thresholds)):
            thr = thresholds[n]
            base = np.full(n_links, np.nan)
            has_base = np.zeros(n_links, dtype=np.bool_)
            frozen = np.zeros(n_links, dtype=np.bool_)
            buf_t = np.zeros((n_links, cap))
            buf_v = np.zeros((n_links, cap))
            head = np.zeros(n_links, dtype=np.int64)
            count = np.zeros(n_links, dtype=np.int64)
            drops = np.empty(n_links)
            weights = np.empty(n_links)
            state = False
            cond_on = False
            cond_t0 = 0.0
            back_on = False
            back_t0 = 0.0
            for s in range(n_steps):
                # 1. calo fuso
                n_used = 0
                for k in range(n_links):
                    if only_own and k != m:
                        continue
                    if valid[s, m, i, k] and has_base[k]:
                        drops[n_used] = base[k] - value[s, m, i, k]
                        weights[n_used] = 1.0 / max(var[s, m, i, k], min_var)
                        n_used += 1
                fused = np.nan
                if n_used > 0:
                    if rule == 0:
                        total = 0.0
                        acc = 0.0
                        for u in range(n_used):
                            total += weights[u]
                            acc += weights[u] * drops[u]
                        fused = acc / total
                    elif rule == 1:
                        ordered = np.sort(drops[:n_used])
                        fused = 0.5 * (ordered[(n_used - 1) // 2] + ordered[n_used // 2])
                    else:
                        fused = drops[0]
                        for u in range(1, n_used):
                            if drops[u] < fused:
                                fused = drops[u]
                # 2. pre-allarme (stessa logica di `pre_alarm`)
                cond = fused > thr  # NaN -> False
                back = fused < thr - margin
                if cond:
                    if not cond_on:
                        cond_on = True
                        cond_t0 = t[s]
                else:
                    cond_on = False
                if back:
                    if not back_on:
                        back_on = True
                        back_t0 = t[s]
                else:
                    back_on = False
                on = cond_on and (t[s] - cond_t0 >= min_duration - tol)
                off = back_on and (t[s] - back_t0 >= release_time - tol) and rx_ok[s, m, i]
                if off:
                    state = False
                elif on:
                    state = True
                out_state[p, n, s] = state
                # 3. linea di base
                active = state or lost[s, m, i]
                for k in range(n_links):
                    if only_own and k != m:
                        continue
                    if new[s, m, i, k]:
                        v = value[s, m, i, k]
                        accept = (not active) and ((not has_base[k]) or (base[k] - v <= freeze_drop))
                        frozen[k] = not accept
                        if accept:
                            while count[k] > 0 and buf_t[k, head[k]] < t[s] - window - tol:
                                head[k] = (head[k] + 1) % cap
                                count[k] -= 1
                            if count[k] == cap:
                                head[k] = (head[k] + 1) % cap
                                count[k] -= 1
                            pos = (head[k] + count[k]) % cap
                            buf_t[k, pos] = t[s]
                            buf_v[k, pos] = v
                            count[k] += 1
                            if count[k] >= min_samples:
                                tmp = np.empty(count[k])
                                for u in range(count[k]):
                                    tmp[u] = buf_v[k, (head[k] + u) % cap]
                                tmp = np.sort(tmp)
                                base[k] = 0.5 * (tmp[(count[k] - 1) // 2] + tmp[count[k] // 2])
                                has_base[k] = True
                if trace and p == 0 and n == 0:
                    out_drop[s] = fused
                    for k in range(n_links):
                        out_frozen[s, k] = frozen[k] and has_base[k]
                        out_base[s, k] = base[k]


def _arrays(inputs, config: dict, scope: str, targets: list | None):
    """Prepara gli ingressi del kernel: voci 4D ``(T, M, I, K)``, allarme "perso" e ricezione 3D, coppie (osservatore, bersaglio)."""
    dcfg = config["detector"]
    entries = inputs.entries
    if entries is None:
        raise ValueError("servono le voci non fuse: costruire gli ingressi con detector.reference = 'relative'")
    sliced = entries["value"].ndim == 3  # `target_slice`: (T, M, K)
    expand = (lambda a: a[:, :, None, ...]) if sliced else (lambda a: a)
    lost = lost_alarm(inputs.silence, inputs.rx_ok, inputs.t, dcfg, inputs.valid)
    lost, rx_ok = (lost[:, :, None], inputs.rx_ok[:, :, None]) if sliced else (lost, inputs.rx_ok)
    n_obs, n_tgt = rx_ok.shape[1], rx_ok.shape[2]
    pairs = [(m, i) for i in range(n_tgt) for m in range(n_obs) if sliced or m != i]
    if sliced:
        pairs = [(m, 0) for m in range(n_obs) if inputs.valid[m]]
    return (
        expand(entries["value"]), expand(entries["var"]), expand(entries["valid"]), expand(entries["new"]),
        np.ascontiguousarray(lost), np.ascontiguousarray(rx_ok), pairs, sliced,
    )  # fmt: skip


def _call(inputs, config, scope, thresholds, trace_pair=None):
    dcfg = config["detector"]
    rule = config["fusion"]["rule"]
    if rule not in _RULE_CODE:
        raise ValueError(f"fusion.rule deve essere uno fra {RULES}, non {rule!r}")
    value, var, valid, new, lost, rx_ok, pairs, sliced = _arrays(inputs, config, scope, None)
    if trace_pair is not None:
        pairs = [trace_pair]
    t = np.ascontiguousarray(inputs.t, dtype=float)
    dt = float(t[1] - t[0])
    cap = int(dcfg["baseline_window"] / dt) + 2
    n_steps, n_links = len(t), value.shape[3]
    obs = np.array([m for m, _ in pairs], dtype=np.int64)
    tgt = np.array([i for _, i in pairs], dtype=np.int64)
    thresholds = np.ascontiguousarray(thresholds, dtype=float)
    trace = trace_pair is not None
    out_state = np.zeros((len(pairs), len(thresholds), n_steps), dtype=np.bool_)
    out_drop = np.full(n_steps if trace else 1, np.nan)
    out_frozen = np.zeros((n_steps if trace else 1, n_links), dtype=np.bool_)
    out_base = np.full((n_steps if trace else 1, n_links), np.nan)
    min_var = config["neighbour_table"]["std_step"] ** 2 / 12.0
    _run_pairs(
        t, np.ascontiguousarray(value), np.ascontiguousarray(var), np.ascontiguousarray(valid), np.ascontiguousarray(new),
        lost, rx_ok, obs, tgt, thresholds, scope == "pairwise", _RULE_CODE[rule], max(min_var, np.finfo(float).tiny),
        float(dcfg["baseline_window"]), int(dcfg["baseline_min_samples"]), float(dcfg["freeze_drop"]),
        float(dcfg["release_level_margin"]), float(dcfg["min_duration"]), float(dcfg["release_time"]), _TIME_TOLERANCE, cap,
        trace, out_state, out_drop, out_frozen, out_base,
    )  # fmt: skip
    return out_state, pairs, sliced, (out_drop, out_frozen, out_base)


def relative_alarms(inputs, config: dict, scope: str, thresholds: np.ndarray) -> np.ndarray:
    """Pre-allarme relativo di ogni coppia (osservatore, bersaglio) per ogni soglia del calo.

    Input: `inputs` (`DetectionInputs` con ``entries``, intero o ritagliato su un bersaglio), `scope` (``fused`` o ``pairwise``:
    solo il link proprio), `thresholds` (dB).
    Procedimento: kernel sequenziale (docstring del modulo) con la regola ``fusion.rule`` e i parametri ``detector``.
    Output: bool ``(soglie, T, M, I)``, oppure ``(soglie, T, M)`` se ritagliato; False sulla diagonale.
    """
    out_state, pairs, sliced, _ = _call(inputs, config, scope, thresholds)
    n_thr, n_steps = out_state.shape[1], out_state.shape[2]
    n_obs = inputs.rx_ok.shape[1]
    if sliced:
        states = np.zeros((n_thr, n_steps, n_obs), dtype=bool)
        for p, (m, _) in enumerate(pairs):
            states[:, :, m] = out_state[p]
        return states
    n_tgt = inputs.rx_ok.shape[2]
    states = np.zeros((n_thr, n_steps, n_obs, n_tgt), dtype=bool)
    for p, (m, i) in enumerate(pairs):
        states[:, :, m, i] = out_state[p]
    return states


def relative_trace(inputs, config: dict, scope: str, threshold: float, observer: int, target: int = 0) -> dict:
    """Traccia di una coppia (osservatore, bersaglio) a una soglia: ``state`` (T,), ``drop`` (T,) calo fuso, ``frozen`` (T, K) e ``base`` (T, K) per link.

    `target` è l'indice nell'array (0 se `inputs` è ritagliato su un bersaglio). Serve alla diagnostica e ai grafici.
    """
    out_state, _, _, (drop, frozen, base) = _call(inputs, config, scope, np.array([threshold]), (observer, target))
    return {"state": out_state[0, 0], "drop": drop, "frozen": frozen, "base": base}

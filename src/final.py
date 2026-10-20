"""Blocco 6 — Numeri finali, soglie congelate, verdetto e sensibilità.

Due insiemi di corse, mai mescolati (`run_sets`):

- **taratura**: le 150 corse dei Blocchi 4-5 (chiavi 6-7); qui, e solo qui, si sceglie la soglia (`freeze_thresholds`) e si scrive in
  ``frozen_thresholds.yaml``;
- **verifica**: ``final.n_runs`` corse per scenario con chiavi 8-9, mai viste; le soglie si LEGGONO dal file (`evaluate_at` non sceglie nulla).

Una *configurazione* è ``media`` (``fused`` + ``weighted_mean``), ``migliore`` (``fused`` + ``max``) o ``coppia`` (``pairwise``); tutte con livello
assoluto, σ_a e spegnimento fissati in ``final.common``. Chiave dei risultati per corsa: ``(nome, regola_perso)``, con ``nome`` anche ``"perso"``
(il solo allarme "nodo perso", senza soglia).

Parametri: sezioni ``final``, ``detector``, ``metrics``, ``packets`` della config.
"""

from __future__ import annotations

import datetime
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import binom

from src.common.parallel import parallel_map
from src.common.rng import spawn_rng
from src.detector import build_inputs, lost_alarm, system_alarm
from src.kalman import filter_all
from src.metrics import (
    _stack,
    aggregate,
    min_distance_to_group,
    nearest_rank,
    observer_states,
    paired_counts,
    poisson_interval,
    summarize_none,
    summarize_separated_ext,
    wilson_interval,
    working_point,
)
from src.packets import burst_lengths
from src.refinement import PER_RUN_FIELDS, SEPARATED, apply_overrides
from src.simulator import experiment_runs, load_or_simulate, run_from_channel

NONE_FIELDS = ("episodes", "hours", "active", "samples", "ep_level", "ep_lost")
LOST = "perso"  # nome della "configurazione" che contiene il solo allarme "nodo perso"


# ---------------------------------------------------------------------------
# Insiemi di corse e configurazioni
# ---------------------------------------------------------------------------


def run_sets(config: dict) -> tuple:
    """Insiemi di taratura e di verifica.

    Output: (taratura, verifica), liste di corse come `experiment_runs`: taratura = ``metrics.n_runs`` corse per scenario con chiavi 7 (semi) e 6
    (nodo e istante); verifica = ``final.n_runs`` corse con ``final.seed_keys.runs`` e ``final.seed_keys.separation``.
    """
    keys = config["final"]["seed_keys"]
    tuning = experiment_runs(config)
    verification = experiment_runs(config, {"runs": keys["runs"], "separation": keys["separation"]}, config["final"]["n_runs"])
    return tuning, verification


def config_overrides(config: dict, name: str, point: dict | None = None) -> dict:
    """Sostituzioni di config della configurazione `name` (``final.common`` + ``final.config_defs[name]``) e di un punto di sensibilità."""
    fcfg = config["final"]
    return {**fcfg["common"], **fcfg["config_defs"][name], **(point or {})}


def result_key(config: dict, name: str) -> tuple:
    """Chiave delle righe di `aggregate` per la configurazione `name`: ``(ambito, "level", "both")``, o quella del solo "perso"."""
    if name == LOST:
        return ("-", "-", "lost")
    return (config["final"]["config_defs"][name]["detector.scope"], "level", "both")


# ---------------------------------------------------------------------------
# Lavoro per corsa
# ---------------------------------------------------------------------------


def episodes_by_type(pre_system: np.ndarray, system: np.ndarray, t: np.ndarray, warmup: float) -> tuple:
    """Falsi allarmi (fronti di salita dell'allarme di sistema dopo il `warmup`, come `count_episodes`) divisi per tipo.

    Input: `pre_system` e `system` (T, I) bool (pre-allarme sul livello; pre-allarme o "perso"), `t`.
    Procedimento: un fronte di salita è di tipo "livello" se nell'istante in cui scatta il pre-allarme di sistema è acceso, altrimenti "perso".
    Output: (episodi livello, episodi perso); la somma è ``count_episodes(system).sum()``.
    """
    edges = system[1:] & ~system[:-1]
    first = max(int(np.searchsorted(t, warmup)) - 1, 0)
    edges, level = edges[first:], pre_system[1:][first:]
    return int((edges & level).sum()), int((edges & ~level).sum())


def _summ_none(pre_system, system, t, warmup):
    out = summarize_none(system, t, warmup)
    out["ep_level"], out["ep_lost"] = episodes_by_type(pre_system, system, t, warmup)
    return out


def _diagnostics(run: dict, inputs, config: dict) -> dict:
    """Corsa ``none``: beacon ricevuti nel gruppo, raffiche (osservate) e sordità. ``longest`` = (durata s, ricevitore, istante d'inizio s)."""
    t = run["t"]
    dt = float(t[1] - t[0])
    n_nodes = run["received"].shape[0]
    trials = int((~np.isnan(run["beacon_times"])).sum()) * (n_nodes - 1)
    out = {"received": int(run["received"].sum()), "trials": trials, "duration": float(t[-1] - t[0]), "n_long": 0, "n_bursts": 0, "longest": (0.0, -1, math.nan)}
    state = run.get("background_state")
    if state is not None and state.ndim == 2:
        lengths = burst_lengths(state, dt)
        out["n_bursts"] = int(len(lengths))
        out["n_long"] = int((lengths > config["final"]["burst_long"]).sum())
        best = (0.0, -1, math.nan)
        for node in range(state.shape[1]):
            padded = np.concatenate([[0], state[:, node].astype(np.int8), [0]])
            edges = np.flatnonzero(np.diff(padded))
            for a, b in zip(edges[0::2], edges[1::2]):
                if (b - a) * dt > best[0]:
                    best = ((b - a) * dt, node, float(t[a]))
        out["longest"] = best
    dw = inputs.deaf_window
    out["deaf_episodes"] = int((dw[1:] & ~dw[:-1]).sum())
    return out


def process_run_final(args: tuple) -> dict:
    """Valuta le configurazioni di `spec` su una corsa; `args` = (corsa, config, cartella di cache, ``R``, spec).

    Input: `spec` = ``{"mode": "cache" | "channel", "point": {sostituzioni di config}, "configs": [nomi], "rules": [regole del perso]}``.
    ``cache``: Blocchi 1-3 dalla cache delle corse; ``channel``: Blocchi 1-2 dalla cache del canale e Blocco 3 rifatto con le sostituzioni del punto
    (senza scrivere nella cache delle corse).
    Procedimento: Kalman una volta (σ_a di ``final.common``); per ogni regola di fusione un `build_inputs`; per ogni configurazione `observer_states`
    su tutta la griglia di soglie; per ogni ``lost_rule`` allarme "perso" e allarme di sistema (pre | perso) a ogni soglia, con `summarize_none`
    (più episodi per tipo) o `summarize_separated_ext`; il solo "perso" una volta per regola.
    Output: ``scenario``, ``index``, ``results`` (``(nome, regola)`` → `_stack` per soglia), ``diag`` (solo ``none``), ``elapsed``.
    """
    run_info, config, cache_dir, R, spec = args
    t0 = time.perf_counter()
    point = spec.get("point") or {}
    rcfg = apply_overrides(run_info["config"], point) if point else run_info["config"]
    run = run_from_channel(rcfg, cache_dir) if spec["mode"] == "channel" else load_or_simulate(rcfg, cache_dir)
    t = run["t"]
    is_none = run_info["scenario"] == "none"
    warmup = config["detector"]["warmup"]
    common = apply_overrides(config, {**config["final"]["common"], **point})
    sigma_a = common["kalman"]["sigma_a"]
    kres = filter_all(run["beacon_times"], run["received"], run["rssi"], R, sigma_a, common["kalman"])
    if is_none:
        summarize = lambda pre, system: _summ_none(pre, system, t, warmup)  # noqa: E731
    else:
        node = run["sep_node"]
        idx_start = int(np.searchsorted(t, run["sep_start"]))
        distance = min_distance_to_group(run["distances"], node)
        sd, rd = config["metrics"]["success_distance"], config["refinement"]["repass_distance"]
        summarize = lambda pre, system: summarize_separated_ext(system, t, idx_start, distance, sd, rd)  # noqa: E731
    out = {"scenario": run_info["scenario"], "index": run_info["index"], "results": {}, "diag": None}
    inputs_by_rule = {}
    first_inputs = None
    for name in spec["configs"]:
        vcfg = apply_overrides(config, config_overrides(config, name, point))
        fusion_rule = vcfg["fusion"]["rule"]
        if fusion_rule not in inputs_by_rule:
            inputs, _ = build_inputs(run, vcfg, sigma_a, R, kres)
            inputs_by_rule[fusion_rule] = inputs if is_none else inputs.target_slice(node)
        inputs = inputs_by_rule[fusion_rule]
        first_inputs = first_inputs or inputs
        grid, states = observer_states(inputs, vcfg, vcfg["detector"]["scope"])
        mo = vcfg["detector"]["min_observers"]
        pre_systems = [system_alarm(states[n], mo) for n in range(len(grid))]
        for rule in spec["rules"]:
            dcfg = {**vcfg["detector"], "lost_rule": rule}
            lost_obs = lost_alarm(inputs.silence, inputs.rx_ok, t, dcfg, inputs.valid, **inputs.lost_kwargs())
            out["results"][(name, rule)] = _stack(
                grid, [summarize(pre_systems[n], system_alarm(states[n] | lost_obs, mo)) for n in range(len(grid))]
            )
    for rule in spec["rules"]:
        dcfg = {**common["detector"], "lost_rule": rule}
        lost_sys = system_alarm(
            lost_alarm(first_inputs.silence, first_inputs.rx_ok, t, dcfg, first_inputs.valid, **first_inputs.lost_kwargs()), dcfg["min_observers"]
        )
        out["results"][(LOST, rule)] = _stack([math.nan], [summarize(np.zeros_like(lost_sys), lost_sys)])
    if is_none:
        out["diag"] = _diagnostics(run, first_inputs, config)
    out["elapsed"] = time.perf_counter() - t0
    return out


def run_final_phase(config: dict, runs: list, cache_dir: Path, R: float, spec: dict, jobs: int) -> list:
    """`process_run_final` su tutte le corse (in parallelo se ``jobs > 1``; il risultato non dipende da `jobs`)."""
    tasks = [(r, config, cache_dir, R, spec) for r in runs]
    return parallel_map(process_run_final, tasks, jobs)


# ---------------------------------------------------------------------------
# Aggregazione e soglie congelate
# ---------------------------------------------------------------------------


def split_outs(outs: list) -> tuple:
    """(corse ``none`` ordinate per indice, scenario → corse con distacco ordinate)."""
    none = sorted((o for o in outs if o["scenario"] == "none"), key=lambda o: o["index"])
    sep = {s: sorted((o for o in outs if o["scenario"] == s), key=lambda o: o["index"]) for s in SEPARATED}
    return none, sep


def sweep_rows(outs: list, name: str, rule: str, config: dict) -> list:
    """Righe di `aggregate` (una per scenario e soglia) per la configurazione `name` e la regola del perso `rule`."""
    key = result_key(config, name)
    none, sep = split_outs(outs)
    pick = lambda o: {key: o["results"][(name, rule)]}  # noqa: E731
    return aggregate([pick(o) for o in none], {s: [pick(o) for o in v] for s, v in sep.items()}, config)


def freeze_thresholds(tuning_outs: list, config: dict) -> dict:
    """Sceglie le soglie dalle corse di TARATURA: regola del Blocco 4 (`working_point`): P_d media massima con falsi allarmi/h e tempo in allarme ammissibili.

    Output: ``{nome: {"threshold", "scope", "fusion_rule", "tuning": {p_detect_slowdown, p_detect_stop, fa_per_hour, active_fraction}}}`` per le
    configurazioni di ``final.configs``. Unico punto del codice in cui si sceglie una soglia che poi si congela.
    """
    frozen = {}
    for name in config["final"]["configs"]:
        rows = sweep_rows(tuning_outs, name, "any", config)
        scope = config["final"]["config_defs"][name]["detector.scope"]
        wp = working_point(rows, scope, "level", "both")
        if wp is None:
            raise RuntimeError(f"configurazione {name}: nessun punto ammissibile sulla taratura")
        at = {s: next(r for r in rows if r["scenario"] == s and r["threshold"] == wp["threshold"]) for s in SEPARATED}
        frozen[name] = {
            "threshold": float(wp["threshold"]),
            "scope": scope,
            "fusion_rule": config["final"]["config_defs"][name]["fusion.rule"],
            "tuning": {
                "p_detect_slowdown": float(at["slowdown"]["p_detect"]),
                "p_detect_stop": float(at["stop"]["p_detect"]),
                "fa_per_hour": float(at["slowdown"]["fa_per_hour"]),
                "active_fraction": float(at["slowdown"]["active_fraction"]),
            },
        }
    return frozen


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True, cwd=Path(__file__).resolve().parent.parent).stdout.strip()
    except Exception:  # noqa: BLE001
        return "sconosciuto"


def write_frozen(path: Path, frozen: dict, config: dict, R: float) -> None:
    """Scrive `frozen` in YAML con data, hash del commit (e ``dirty`` se ci sono modifiche non committate), ``R`` e le condizioni della taratura."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "tuning_set": {"runs_per_scenario": config["metrics"]["n_runs"], "seed_keys": {"runs": 7, "separation": 6}},
        "R": float(R),
        "release_time": float(config["final"]["common"]["detector.release_time"]),
        "thresholds": frozen,
    }
    with open(path, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)


def load_frozen(path: Path) -> dict:
    """Legge ``frozen_thresholds.yaml``: ritorna il documento (le soglie sono in ``["thresholds"][nome]["threshold"]``)."""
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Valutazione a soglia data
# ---------------------------------------------------------------------------


def threshold_index(grid: np.ndarray, threshold: float) -> int:
    """Indice della soglia `threshold` nella griglia (deve coincidere); NaN = il solo "perso" (indice 0)."""
    if math.isnan(threshold):
        return 0
    n = int(np.argmin(np.abs(grid - threshold)))
    if abs(grid[n] - threshold) > 1e-9:
        raise ValueError(f"la soglia {threshold} non sta nella griglia dello sweep")
    return n


def at_threshold(outs: list, name: str, rule: str, threshold: float) -> dict:
    """Dati per corsa alla soglia `threshold`: ``{"none": {campo: array (corse,)}, "sep": {scenario: {campo: array}}}``."""
    none, sep = split_outs(outs)
    n = threshold_index(next(iter(none))["results"][(name, rule)]["threshold"], threshold)
    return {
        "none": {f: np.array([o["results"][(name, rule)][f][n] for o in none]) for f in NONE_FIELDS},
        "sep": {s: {f: np.array([o["results"][(name, rule)][f][n] for o in v]) for f in PER_RUN_FIELDS} for s, v in sep.items()},
    }


def bootstrap_rate_interval(counts: np.ndarray, hours: np.ndarray, n_samples: int, level: float, rng: np.random.Generator) -> tuple:
    """Intervallo bootstrap (percentile) per la frequenza di eventi/ora con le CORSE come unità indipendente.

    Input: `counts` e `hours` per corsa, numero di ricampionamenti, livello, generatore.
    Procedimento: si ricampionano le corse con reinserimento; frequenza = somma eventi / somma ore; percentili ``(1-level)/2`` e ``1-(1-level)/2``.
    """
    counts, hours = np.asarray(counts, dtype=float), np.asarray(hours, dtype=float)
    idx = rng.integers(0, len(counts), size=(n_samples, len(counts)))
    rates = counts[idx].sum(axis=1) / hours[idx].sum(axis=1)
    alpha = 1.0 - level
    lo, hi = np.quantile(rates, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def bootstrap_rng(config: dict, seed: int | None = None) -> np.random.Generator:
    """Generatore del bootstrap: ``SeedSequence(simulation.seed, spawn_key=(final.seed_keys.bootstrap,))``, rifatto a ogni valutazione (non dipende dall'ordine)."""
    seed = config["simulation"]["seed"] if seed is None else seed
    return spawn_rng(seed, config["final"]["seed_keys"]["bootstrap"])


def summarize_at(at: dict, config: dict) -> dict:
    """Metriche di una configurazione a una soglia (nessuna scelta di soglia qui).

    Output: ``sep[scenario]``: P_d entro ``success_distance`` e entro ``success_window`` con Wilson, ritardo e distanza (mediana, p90; mancati = inf),
    spegnimenti a torto (corse, totale), ripassaggi (totale, con allarme spento); ``none``: falsi allarmi/h con Poisson e bootstrap per corsa, per tipo,
    frazione di tempo in allarme.
    """
    mcfg, level = config["metrics"], config["metrics"]["confidence_level"]
    out = {"sep": {}, "none": {}}
    for s, d in at["sep"].items():
        n = len(d["success"])
        k = int(d["success"].sum())
        kw = int((d["delay"] <= mcfg["success_window"]).sum())
        dist = np.where(np.isnan(d["distance"]), math.inf, d["distance"])
        out["sep"][s] = {
            "n_runs": n,
            "p_detect": k / n,
            "p_interval": wilson_interval(k, n, level),
            "p_window": kw / n,
            "p_window_interval": wilson_interval(kw, n, level),
            "delay_median": nearest_rank(d["delay"], 0.5),
            "delay_p90": nearest_rank(d["delay"], 0.9),
            "distance_median": nearest_rank(dist, 0.5),
            "distance_p90": nearest_rank(dist, 0.9),
            "wrong_off_runs": int((d["n_wrong_off"] > 0).sum()),
            "wrong_off_total": int(d["n_wrong_off"].sum()),
            "repass": int(d["repass"].sum()),
            "repass_off": int((d["repass"].astype(bool) & d["repass_off"].astype(bool)).sum()),
        }
    nn = at["none"]
    hours = nn["hours"].sum()
    episodes = int(nn["episodes"].sum())
    out["none"] = {
        "n_runs": len(nn["episodes"]),
        "episodes": episodes,
        "hours": float(hours),
        "fa_per_hour": episodes / hours,
        "poisson": poisson_interval(episodes, float(hours), level),
        "bootstrap": bootstrap_rate_interval(nn["episodes"], nn["hours"], mcfg_bootstrap(config), level, bootstrap_rng(config)),
        "fa_level": float(nn["ep_level"].sum() / hours),
        "fa_lost": float(nn["ep_lost"].sum() / hours),
        "active_fraction": float(nn["active"].sum() / nn["samples"].sum()),
        "episodes_per_run": nn["episodes"].astype(int),
    }
    return out


def mcfg_bootstrap(config: dict) -> int:
    return int(config["final"]["bootstrap_samples"])


def evaluate_at(outs: list, name: str, rule: str, threshold: float, config: dict) -> dict:
    """`at_threshold` + `summarize_at`: valutazione alla soglia DATA (letta dal file congelato). Restituisce ``summary`` e ``at`` (dati per corsa)."""
    at = at_threshold(outs, name, rule, threshold)
    return {"summary": summarize_at(at, config), "at": at}


def total_success(at: dict) -> np.ndarray:
    """Esiti (slowdown poi stop) concatenati: bool (corse con distacco,)."""
    return np.concatenate([at["sep"][s]["success"].astype(bool) for s in SEPARATED])


# ---------------------------------------------------------------------------
# Verdetto media contro migliore
# ---------------------------------------------------------------------------


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """p-value bilaterale del test esatto di McNemar sulle corse discordanti (binomiale con p = 1/2): ``min(1, 2 P(X <= min(b, c)))``."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    return float(min(1.0, 2.0 * binom.cdf(min(only_a, only_b), n, 0.5)))


def decide_verdict(success_media: np.ndarray, success_migliore: np.ndarray, fa_media: float, fa_migliore: float, min_net: int) -> dict:
    """Verdetto fissato prima di vedere i dati.

    Input: esiti per corsa (slowdown + stop insieme, stessa corsa nella stessa posizione) di `media` e `migliore` entro 30 m; falsi allarmi/h
    (stima puntuale) di ciascuna sulle corse ``none`` di verifica; `min_net` (``final.min_net_gain``).
    Procedimento: ``netto`` = corse rilevate da `migliore` e non da `media`, meno quelle rilevate da `media` e non da `migliore`.
    Output: si adotta ``migliore`` solo se ``netto >= min_net`` E ``fa_migliore <= fa_media``; altrimenti ``media``. Riporta anche il p-value di McNemar (solo informativo).
    """
    gained, lost = paired_counts(success_migliore, success_media)
    net = gained - lost
    adopt = "migliore" if (net >= min_net and fa_migliore <= fa_media) else "media"
    return {
        "adopted": adopt,
        "net": int(net),
        "gained": int(gained),
        "lost": int(lost),
        "min_net_gain": int(min_net),
        "fa_media": float(fa_media),
        "fa_migliore": float(fa_migliore),
        "mcnemar_p": mcnemar_exact(gained, lost),
    }


# ---------------------------------------------------------------------------
# Sensibilità
# ---------------------------------------------------------------------------


def sensitivity_cases(config: dict) -> list:
    """Punti di sensibilità: lista di ``(nome, sostituzioni, regole del perso)``. ``base`` (nessuna sostituzione) e i punti di ``final.sensitivity_points``;
    ``any`` ovunque, più ``not_deaf`` e ``not_deaf_episode`` nei punti di ``final.not_deaf_points``."""
    fcfg = config["final"]
    cases = [("base", {})] + list(fcfg["sensitivity_points"].items())
    return [
        (name, point, ["any", "not_deaf", "not_deaf_episode"] if name in fcfg["not_deaf_points"] else ["any"]) for name, point in cases
    ]


def expected_long_bursts(config: dict, point: dict, duration: float) -> float | None:
    """Raffiche più lunghe di ``final.burst_long`` attese per corsa (tutti i ricevitori): ``N · durata · L / T_b · exp(-burst_long / T_b)``; None se non ``bursty``."""
    cfg = apply_overrides(config, point)
    rc = cfg["packets"]["reception"]
    if rc["background_model"] != "bursty":
        return None
    b = rc["bursty"]
    n_scope = cfg["group"]["n_nodes"] if b["burst_scope"] == "receiver" else cfg["group"]["n_nodes"] ** 2
    return n_scope * duration * b["mean_loss"] / b["mean_burst"] * math.exp(-config["final"]["burst_long"] / b["mean_burst"])


def retuned_point(outs: list, name: str, rule: str, config: dict) -> dict | None:
    """SOLO INFORMATIVO (sezione D): soglia che si sceglierebbe ritarando sul punto di sensibilità, con la regola solita. Non si scrive mai nel file congelato."""
    rows = sweep_rows(outs, name, rule, config)
    scope = config["final"]["config_defs"][name]["detector.scope"]
    wp = working_point(rows, scope, "level", "both")
    return None if wp is None else {"threshold": float(wp["threshold"])}


def gain_loss_vs(at: dict, base_at: dict) -> dict:
    """Corse guadagnate e perse (entro 30 m) rispetto alla base, per scenario e totale."""
    out = {s: paired_counts(at["sep"][s]["success"], base_at["sep"][s]["success"]) for s in SEPARATED}
    out["total"] = tuple(sum(out[s][k] for s in SEPARATED) for k in (0, 1))
    return out


def evaluate_sensitivity(case_outs: dict, final_name: str, frozen_threshold: float, config: dict) -> dict:
    """Valuta ogni punto di sensibilità alla soglia congelata e a quella ritarata.

    Input: `case_outs` = ``{punto: corse}`` (le prime ``final.sensitivity_runs`` corse per scenario, stesse per tutti i punti), configurazione finale,
    soglia congelata.
    Output: ``{(punto, regola): {"frozen": {summary, at}, "retuned": {threshold, summary, at} | None, "gain_loss": {...}, "diag": {...}}}``;
    ``gain_loss`` è rispetto alla base (regola ``any``) alla soglia congelata.
    """
    base_at = at_threshold(case_outs["base"], final_name, "any", frozen_threshold)
    out = {}
    for name, point, rules in sensitivity_cases(config):
        outs = case_outs[name]
        none = split_outs(outs)[0]
        diags = [o["diag"] for o in none]
        duration = float(np.mean([d["duration"] for d in diags]))
        longest = max((d["longest"] + (i,) for i, d in enumerate(diags)), key=lambda x: x[0])
        diag = {
            "received": int(sum(d["received"] for d in diags)),
            "trials": int(sum(d["trials"] for d in diags)),
            "n_long": int(sum(d["n_long"] for d in diags)),
            "n_bursts": int(sum(d["n_bursts"] for d in diags)),
            "deaf_episodes": int(sum(d["deaf_episodes"] for d in diags)),
            "expected_long": None,
            "longest": longest,  # (durata, ricevitore, istante, indice della corsa none)
        }
        exp = expected_long_bursts(config, point, duration)
        if exp is not None:
            diag["expected_long"] = exp * len(none)
        for rule in rules:
            frozen = evaluate_at(outs, final_name, rule, frozen_threshold, config)
            retuned = None
            wp = retuned_point(outs, final_name, rule, config)
            if wp is not None:
                retuned = {**wp, **evaluate_at(outs, final_name, rule, wp["threshold"], config)}
            lost_only = evaluate_at(outs, LOST, rule, math.nan, config)
            out[(name, rule)] = {
                "frozen": frozen,
                "retuned": retuned,
                "lost_only": lost_only,
                "gain_loss": gain_loss_vs(frozen["at"], base_at),
                "diag": diag,
            }
    return out


# ---------------------------------------------------------------------------
# Formattazione (console e final_summary.md)
# ---------------------------------------------------------------------------


def _pd(p: float, interval: tuple) -> str:
    return f"{p:.2f} [{interval[0]:.2f}-{interval[1]:.2f}]"


def _ci(value: float, interval: tuple) -> str:
    return f"{value:.2f} [{interval[0]:.2f}-{interval[1]:.2f}]"


def evaluation_table(results: dict) -> str:
    """Tabella della sezione C: per configurazione e per il solo "perso", P_d (30 m e 30 s), ritardo, distanza, falsi allarmi, ottimismo."""
    verif, tuning, frozen = results["verification"], results["tuning"], results["frozen"]["thresholds"]
    lines = []
    head = (
        f"{'config':<9} {'sc.':<8} {'soglia':>6} {'P_d 30 m [Wilson]':>19} {'P_d 30 s [Wilson]':>19} {'rit.med/p90 s':>14} {'dist.med/p90 m':>15}"
        f" {'tar.':>5} {'diff.':>6}"
    )
    lines.append(head)
    for name in [*results["config"]["final"]["configs"], LOST]:
        s = verif[name]["summary"]
        thr = frozen[name]["threshold"] if name in frozen else math.nan
        for sc in SEPARATED:
            d = s["sep"][sc]
            tun = tuning[name]["summary"]["sep"][sc]["p_detect"]
            lines.append(
                f"{name:<9} {sc:<8} {thr:>6.1f} {_pd(d['p_detect'], d['p_interval']):>19} {_pd(d['p_window'], d['p_window_interval']):>19}"
                f" {d['delay_median']:>6.1f}/{d['delay_p90']:<7.1f} {d['distance_median']:>7.1f}/{d['distance_p90']:<7.1f} {tun:>5.2f} {d['p_detect'] - tun:>+6.2f}"
            )
    lines.append("")
    lines.append(
        f"{'config':<9} {'FA/h Poisson':>19} {'FA/h bootstrap':>19} {'livello/h':>9} {'perso/h':>8} {'in allarme':>10} {'FA/h taratura':>13} {'diff.':>6}"
        f" {'torti (corse/n)':>16} {'ripassaggi (spenti)':>20}"
    )
    for name in [*results["config"]["final"]["configs"], LOST]:
        s, st = verif[name]["summary"], tuning[name]["summary"]
        n = s["none"]
        wrong = sum(s["sep"][sc]["wrong_off_runs"] for sc in SEPARATED)
        total = sum(s["sep"][sc]["wrong_off_total"] for sc in SEPARATED)
        rep = sum(s["sep"][sc]["repass"] for sc in SEPARATED)
        repo = sum(s["sep"][sc]["repass_off"] for sc in SEPARATED)
        lines.append(
            f"{name:<9} {_ci(n['fa_per_hour'], n['poisson']):>19} {_ci(n['fa_per_hour'], n['bootstrap']):>19} {n['fa_level']:>9.2f} {n['fa_lost']:>8.2f}"
            f" {100 * n['active_fraction']:>9.2f}% {st['none']['fa_per_hour']:>13.2f} {n['fa_per_hour'] - st['none']['fa_per_hour']:>+6.2f}"
            f" {wrong:>8d}/{total:<7d} {rep:>11d} ({repo})"
        )
    target = results["config"]["metrics"]["target_false_alarms_per_hour"]
    over = [n for n in results["config"]["final"]["configs"] if verif[n]["summary"]["none"]["fa_per_hour"] > target]
    lines.append(
        f"Target falsi allarmi: {target:g}/h. "
        + (f"SUPERATO sulla verifica da: {', '.join(over)} (soglie non corrette, come richiesto)." if over else "Rispettato da tutte le configurazioni sulla verifica.")
    )
    return "\n".join(lines)


def verdict_text(v: dict) -> str:
    return (
        f"netto = {v['net']:+d} (migliore guadagna {v['gained']}, perde {v['lost']}; soglia {v['min_net_gain']}); FA/h media {v['fa_media']:.2f}, migliore"
        f" {v['fa_migliore']:.2f}; McNemar esatto p = {v['mcnemar_p']:.3f} (informativo) -> finale = {v['adopted']}"
    )


def sensitivity_table(results: dict) -> str:
    """Tabella della sezione D: per ogni punto e regola, soglia congelata e ritarata."""
    sens, config = results["sensitivity"], results["config"]
    lines = [
        f"{'punto':<11} {'regola':<16} {'soglia':>8} {'P_d slowdown':>17} {'P_d stop':>17} {'g/p sl':>7} {'g/p st':>7} {'dist.med/p90 sl':>16} {'dist.med/p90 st':>16}"
        f" {'FA/h [bootstrap]':>19} {'livello':>7} {'perso':>6} {'in all.':>7}"
    ]
    for (name, rule), r in sens.items():
        for tag, block in (("congelata", r["frozen"]), ("ritarata", r["retuned"])):
            label = f"{name:<11} {rule:<16}"
            if block is None:
                lines.append(f"{label} {tag:>9}  nessun punto ammissibile")
                continue
            s = block["summary"]
            sl, st, nn = s["sep"]["slowdown"], s["sep"]["stop"], s["none"]
            thr = block.get("threshold", results["frozen"]["thresholds"][results["final_name"]]["threshold"])
            gl = r["gain_loss"] if tag == "congelata" else gain_loss_vs(block["at"], sens[("base", "any")]["frozen"]["at"])
            lines.append(
                f"{label} {tag:>9} {thr:>5.1f} {_pd(sl['p_detect'], sl['p_interval']):>17} {_pd(st['p_detect'], st['p_interval']):>17}"
                f" {'+%d/-%d' % gl['slowdown']:>7} {'+%d/-%d' % gl['stop']:>7}"
                f" {sl['distance_median']:>7.1f}/{sl['distance_p90']:<7.1f} {st['distance_median']:>7.1f}/{st['distance_p90']:<7.1f}"
                f" {_ci(nn['fa_per_hour'], nn['bootstrap']):>19} {nn['fa_level']:>7.2f} {nn['fa_lost']:>6.2f} {100 * nn['active_fraction']:>6.2f}%"
            )
    lines.append("")
    lines.append(f"{'punto':<11} {'beacon ricevuti':>16} {'raffiche > ' + format(config['final']['burst_long'], 'g') + ' s osservate':>24} {'attese':>8} {'episodi di sordità':>19}")
    for name, _, _ in sensitivity_cases(config):
        d = sens[(name, "any")]["diag"]
        exp = "-" if d["expected_long"] is None else f"{d['expected_long']:.1f}"
        obs = "-" if d["expected_long"] is None else str(d["n_long"])
        lines.append(f"{name:<11} {100 * d['received'] / d['trials']:>15.2f}% {obs:>24} {exp:>8} {d['deaf_episodes']:>19d}")
    return "\n".join(lines)


def write_final_summary(results: dict, path: Path, versions: dict) -> None:
    """Scrive ``final_summary.md`` dai risultati (nessun numero scritto a mano): config finale, soglie, valutazione, verdetto, sensibilità, versioni."""
    config, frozen = results["config"], results["frozen"]
    final = results["final_name"]
    dcfg = config["detector"]
    doc = [
        "# Riepilogo finale (generato da `scripts/reproduce_all.py`)",
        "",
        f"Commit: `{versions['commit']}`{' (con modifiche non committate)' if versions['dirty'] else ''}  ",
        f"Python {versions['python']}; " + ", ".join(f"{k} {v}" for k, v in versions["libraries"].items()),
        "",
        "## Configurazione finale",
        "",
        f"- configurazione: **{final}** (ambito `{frozen['thresholds'][final]['scope']}`, regola di fusione `{frozen['thresholds'][final]['fusion_rule']}`)",
        f"- soglia di livello congelata: **{frozen['thresholds'][final]['threshold']:g} dBm** (taratura: {frozen['date']}, commit `{frozen['commit']}`)",
        f"- spegnimento: {dcfg['release_time']:g} s; `sigma_a` {config['kalman']['sigma_a']:g}; `min_observers` {dcfg['min_observers']};"
        f" `lost_rule` {dcfg['lost_rule']}; shadowing `{config['channel']['shadowing']['mode']}`",
        "",
        "## Soglie congelate (sulla taratura)",
        "",
        "| configurazione | soglia (dBm) | P_d slowdown | P_d stop | FA/h | tempo in allarme |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, f in frozen["thresholds"].items():
        t = f["tuning"]
        doc.append(f"| {name} | {f['threshold']:g} | {t['p_detect_slowdown']:.2f} | {t['p_detect_stop']:.2f} | {t['fa_per_hour']:.2f} | {100 * t['active_fraction']:.2f}% |")
    doc += ["", "## Valutazione sulle corse di verifica", "", "```", evaluation_table(results), "```", "", "## Verdetto media contro migliore", "", verdict_text(results["verdict"]), ""]
    doc += ["## Sensibilità", "", "```", sensitivity_table(results), "```", ""]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(doc))

"""Block 7 tests: routing to the group leader (hops, AHP/entropy, score, time to break, unicast, packets, carry, duplicates, `direct`, reproducibility).

Each test checks a known fact (hand calculation, closed formula, comparison with an independent calculation). The topologies are synthetic: the network is described by a function
``up(i, j, t)`` (link ``i → j`` exists at that instant) and from it beacons, link estimates and reception probabilities are built with the same functions as the experiment.
"""

import copy
import math

import numpy as np
import pytest

from src.common.paths import REPO_ROOT as ROOT
from tests.helpers import short_config
from src.packets import beacon_schedule
from src.routing import (
    ISOLATED,
    KIND_NORMAL,
    KIND_SOS,
    STRATEGIES,
    UniformStream,
    ahp_weights,
    airtime,
    announce_hops,
    announced_grid,
    combine_weights,
    dqrem_scores,
    entropy_weights,
    link_state,
    position_class,
    record_bytes,
    records_per_packet,
    select_packet,
    simulate_strategy,
    time_to_break,
    unicast_attempts,
)


def synthetic_rd(cfg, n, duration, leader, up, events=(), seed=3, p_override=None):
    """Synthetic run: real beacons (`beacon_schedule`), received if ``up(i, j, t)``; estimate -60 dBm on the links that exist, -120 on the others; slope 0; ``p`` = 1 where the link exists."""
    rcfg = cfg["routing"]
    dt = cfg["simulation"]["dt"]
    t = np.arange(0.0, duration, dt)
    times, steps = beacon_schedule(n, float(t[-1]), dt, cfg["packets"], np.random.default_rng(seed))
    k = times.shape[1]
    ok = steps >= 0
    received = np.zeros((n, k, n), dtype=bool)
    p = np.zeros((len(t), n, n), dtype=np.float32)
    est_r = np.full((len(t), n, n), -120.0)
    level = np.full((n, k, n), -120.0)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            received[i, :, j] = up(i, j, np.where(ok[i], times[i], 0.0)) & ok[i]
            exists = up(i, j, t)
            p[:, i, j] = exists
            if exists.any():
                est_r[:, i, j] = -60.0
                level[i, :, j] = -60.0
    ls = link_state(t, times, received, est_r, np.zeros_like(est_r), level, np.zeros_like(level), rcfg, cfg["packets"]["beacon_period"])
    ann = announce_hops(times, steps, received, est_r, leader, rcfg)
    rd = {
        "dt": dt, "t": t, "times": times, "steps": steps, "received": received, "ann": ann, "ann_grid": announced_grid(t, times, ann), "p": p,
        "leader": leader, "events": list(events), "seed": 11, "sep_node": -1, **ls,
    }  # fmt: skip
    if p_override is not None:
        p_override(rd)
    return rd


def chain(i, j, tt):
    return np.full(np.shape(tt), abs(i - j) == 1)


# ---------------------------------------------------------------------------
# 1. No regression
# ---------------------------------------------------------------------------


def test_default_config_is_the_block6_detector_and_block7_does_not_change_cache_keys(cfg):
    """The default config remains the final 5-node detector one; the ``routing`` section does not enter the cache keys of Blocks 1-3."""
    from src.simulator import cache_key, channel_cache_key

    assert cfg["group"]["n_nodes"] == 5 and cfg["group"]["longitudinal_spread"] == 10.0
    assert cfg["detector"]["level_threshold"] == -75.0 and cfg["detector"]["mode"] == "level"
    without = copy.deepcopy(cfg)
    del without["routing"]
    assert cache_key(cfg) == cache_key(without) and channel_cache_key(cfg) == channel_cache_key(without)
    from src.routing_experiment import block7_config

    long = block7_config(cfg)
    assert long["group"]["n_nodes"] == 10 and cache_key(long) != cache_key(cfg)  # the long group has its own cache (never the 5-node one)


def test_block6_results_identical_with_or_without_routing_section(cfg, tmp_path):
    """`process_run_final` gives the same result with and without the ``routing`` section: Block 7 does not touch Blocks 1-6."""
    from src.final import process_run_final
    from src.simulator import scenario_config

    short_config(cfg, 120.0)
    without = copy.deepcopy(cfg)
    del without["routing"]
    spec = {"mode": "cache", "point": {}, "configs": ["media"], "rules": ["any"]}
    info = {"scenario": "stop", "index": 0, "config": scenario_config(cfg, "stop", 7, 1, 40.0)}
    info_no = {"scenario": "stop", "index": 0, "config": scenario_config(without, "stop", 7, 1, 40.0)}
    a = process_run_final((info, cfg, tmp_path / "a", 9.0, spec))
    b = process_run_final((info_no, without, tmp_path / "b", 9.0, spec))
    for key, res in a["results"].items():
        for field, arr in res.items():
            np.testing.assert_array_equal(arr, b["results"][key][field], err_msg=str((key, field)))


def test_real_run_reproducible_and_writes_nothing_to_run_cache(cfg):
    """A short real run (long group): same seed, same result; the run cache (``results/cache``) does not change."""
    from src.routing_experiment import block7_config, process_run_routing
    from src.simulator import scenario_config

    cfg["group"]["calibration"].update({"groups": 10, "duration": 60.0})
    c7 = block7_config(cfg)
    c7["simulation"]["duration"] = 150.0
    run = {"scenario": "stop", "index": 0, "leader": 2, "sos": 0.3, "config": scenario_config(c7, "stop", 5, 6, 40.0)}
    cache = ROOT / "results" / "cache"
    before = sorted(p.name for p in cache.glob("*")) if cache.exists() else []
    a = process_run_routing((run, ["direct", "dqrem"], [], False))
    b = process_run_routing((run, ["direct", "dqrem"], [], False))
    after = sorted(p.name for p in cache.glob("*")) if cache.exists() else []
    assert before == after
    for name in ("direct", "dqrem"):
        sa, sb = a["strategies"][name], b["strategies"][name]
        for key in ("breaks", "bridge_changes", "ttl_drops", "cycles"):
            assert sa[key] == sb[key]
        np.testing.assert_array_equal(sa["classes"]["group"]["delays"], sb["classes"]["group"]["delays"])
        assert sa["classes"]["group"]["created"] > 0


# ---------------------------------------------------------------------------
# 2. Hops
# ---------------------------------------------------------------------------


def test_hops_on_a_chain_and_isolation_when_a_link_is_cut(cfg):
    """Chain 0-1-2-3: hops 0, 1, 2, 3. With link 1-2 removed, nodes 2 and 3 become isolated within ``max_hops`` beacon periods (bounded count to infinity)."""
    rcfg = cfg["routing"]
    rd = synthetic_rd(cfg, 4, 80.0, 0, chain)
    times, ann = rd["times"], rd["ann"]
    last = [ann[i, np.flatnonzero(~np.isnan(times[i]))[-1]] for i in range(4)]
    assert last == [0, 1, 2, 3]

    cut = 30.0
    cut_chain = lambda i, j, tt: chain(i, j, tt) & ~(({i, j} == {1, 2}) & (tt >= cut))  # noqa: E731
    rd = synthetic_rd(cfg, 4, 80.0, 0, cut_chain)
    times, ann = rd["times"], rd["ann"]
    assert [ann[i, np.flatnonzero(~np.isnan(times[i]))[-1]] for i in range(4)] == [0, 1, ISOLATED, ISOLATED]
    period, jitter = cfg["packets"]["beacon_period"], cfg["packets"]["jitter"]
    bound = cut + rcfg["neighbour_max_age"] + (rcfg["max_hops"] + 2) * (period + jitter)
    for node in (2, 3):
        ks = np.flatnonzero(~np.isnan(times[node]))
        ok = ks[(ann[node, ks] == ISOLATED) & (times[node, ks] > cut)]
        assert times[node, ok[0]] <= bound, f"node {node}: isolated only at {times[node, ok[0]]:.1f} s (limit {bound:.1f} s)"
    # in the meantime the wrong value never exceeds max_hops
    finite = ann[2:, :][(ann[2:, :] != ISOLATED)]
    assert finite.max() <= rcfg["max_hops"]


def test_no_ttl_drops_on_a_still_topology(cfg):
    """On a still topology no packet is dropped by TTL and all records arrive."""
    rd = synthetic_rd(cfg, 4, 60.0, 0, chain)
    out = simulate_strategy(rd, "best_rssi", cfg["routing"])
    assert out["ttl_drops"] == 0
    assert set(np.unique(out["log"]["origin"])) == {1, 2, 3}
    hops = {o: out["log"]["hops"][out["log"]["origin"] == o] for o in (1, 2, 3)}
    assert all(np.all(h == o) for o, h in hops.items())  # the node `o` hops from the leader takes exactly `o` hops


# ---------------------------------------------------------------------------
# 3. AHP, entropy
# ---------------------------------------------------------------------------


def test_ahp_weights_equal_independent_eigenvector_and_consistent(cfg):
    """AHP weights = principal eigenvector computed separately (power iteration), CR < 0.1, expected values from the brief."""
    rcfg = cfg["routing"]
    a = np.array(rcfg["ahp_matrix"], dtype=float)
    w, cr = ahp_weights(a, rcfg["ahp_random_index"])
    v = np.ones(5)
    for _ in range(500):
        v = a @ v
        v /= v.sum()
    np.testing.assert_allclose(w, v, atol=1e-9)
    np.testing.assert_allclose(w, [0.364, 0.364, 0.131, 0.070, 0.070], atol=2e-3)
    assert w.sum() == pytest.approx(1.0) and cr < 0.1 and cr == pytest.approx(0.001, abs=1e-3)


def test_entropy_weight_zero_for_constant_criterion_and_known_case():
    """A constant criterion has weight 0; with known columns the weight goes to the one that varies most (hand calculation: 3 rows, column A concentrated, column B uniform)."""
    benefit = np.array([True, True, True])
    x = np.array([[1.0, 5.0, 3.0], [2.0, 5.0, 9.0], [3.0, 5.0, 4.0], [4.0, 5.0, 8.0]])
    w = entropy_weights(x, benefit)
    assert w[1] == 0.0 and w.sum() == pytest.approx(1.0) and np.all(w >= 0)
    # by hand for one column: v = (x - min)/(max - min) = [0, 1/3, 2/3, 1] -> p = [0, 1/6, 2/6, 3/6], E = -Σ p ln p / ln 4
    p = np.array([1 / 6, 2 / 6, 3 / 6])
    e = -(p * np.log(p)).sum() / math.log(4)
    only = entropy_weights(x[:, [0, 1]], benefit[:2])
    assert only[0] == pytest.approx(1.0) and 1 - e > 0  # a single varying column: all the weight
    two = entropy_weights(np.column_stack([x[:, 0], [0.0, 0.0, 0.0, 9.0]]), benefit[:2])
    assert two[1] > two[0]  # the column concentrated on a single value has lower entropy, hence higher weight
    assert np.all(entropy_weights(np.ones((5, 3)), benefit) == 0)
    np.testing.assert_allclose(combine_weights(np.array([0.6, 0.4]), np.array([0.0, 0.0]), 0.5), [0.6, 0.4])


# ---------------------------------------------------------------------------
# 4. Score
# ---------------------------------------------------------------------------


def test_score_dominant_candidate_wins_and_directions(cfg):
    """A candidate better in all criteria always wins; with only one it is chosen; high benefits and low costs raise the score."""
    benefit = np.array([True, True, False, True, False])
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(2, 6))
        x = np.column_stack([rng.uniform(-90, -50, n), rng.uniform(0, 60, n), rng.integers(1, 6, n), rng.uniform(0, 1, n), rng.integers(0, 16, n)]).astype(float)
        best = int(rng.integers(0, n))
        x[best] = [x[:, 0].max() + 1, x[:, 1].max() + 1, x[:, 2].min() - 1, x[:, 3].max() + 0.01, x[:, 4].min() - 1]
        w = rng.dirichlet(np.ones(5)) + 1e-3
        assert int(np.argmax(dqrem_scores(x, w / w.sum(), benefit))) == best
    assert dqrem_scores(np.array([[-70.0, 10.0, 2.0, 0.9, 3.0]]), np.full(5, 0.2), benefit)[0] == pytest.approx(1.0)
    w = np.array([1.0, 0, 0, 0, 0])
    x = np.array([[-80.0, 0, 1, 1, 0], [-60.0, 0, 1, 1, 0]])
    assert dqrem_scores(x, w, benefit)[1] > dqrem_scores(x, w, benefit)[0]  # benefit: higher signal is better
    w = np.array([0, 0, 0, 0, 1.0])
    x = np.array([[-70.0, 0, 1, 1, 2.0], [-70.0, 0, 1, 1, 9.0]])
    assert dqrem_scores(x, w, benefit)[0] > dqrem_scores(x, w, benefit)[1]  # cost: shorter queue is better


# ---------------------------------------------------------------------------
# 5. Time to break
# ---------------------------------------------------------------------------


def test_time_to_break(cfg):
    """Slope ≥ 0 → cap; −80 dBm with −0.5 dB/s and threshold −90 → (−80 − (−90)) / 0.5 = 20 s; clipped to the cap and never negative."""
    rcfg = cfg["routing"]
    thr, cap = rcfg["candidate_threshold"], rcfg["ttb_cap"]
    assert time_to_break(-70.0, 0.0, thr, cap) == cap and time_to_break(-70.0, 0.4, thr, cap) == cap
    assert time_to_break(-80.0, -0.5, thr, cap) == pytest.approx(20.0)
    assert time_to_break(-60.0, -0.1, thr, cap) == cap  # 300 s clipped to 60
    assert time_to_break(-95.0, -0.5, thr, cap) == 0.0  # already below the threshold


# ---------------------------------------------------------------------------
# 6. Unicast
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p_fwd,p_ret,retries", [(0.7, 0.8, 3), (0.5, 0.5, 1), (0.9, 0.6, 0)])
def test_unicast_delivery_matches_closed_form(p_fwd, p_ret, retries):
    """With probability ``p = forward × return`` per attempt, the Monte Carlo delivery equals ``1 − (1 − p)^(r + 1)`` within 4 standard errors."""
    stream = UniformStream(np.random.default_rng(5))
    n = 40000
    ok = sum(unicast_attempts(p_fwd, p_ret, retries, stream)[0] for _ in range(n))
    p = p_fwd * p_ret
    expect = 1 - (1 - p) ** (retries + 1)
    assert abs(ok / n - expect) < 4 * math.sqrt(expect * (1 - expect) / n)


# ---------------------------------------------------------------------------
# 7. Aggregation
# ---------------------------------------------------------------------------


def test_packet_sizes_and_priority(cfg):
    """Never beyond ``max_packet_bytes`` (4 records with 250 B and 10 nodes, 10 with 1470 B); an SOS goes before normal records; airtime of 250 B ≈ 2.5 ms with the ack."""
    rcfg = cfg["routing"]
    assert record_bytes(10) == 53
    assert records_per_packet(rcfg, 10) == 4 and records_per_packet(rcfg, 10, 1470) == 10
    assert rcfg["packet_header_bytes"] + records_per_packet(rcfg, 10) * 53 <= rcfg["max_packet_bytes"]
    assert airtime(250, rcfg) + airtime(rcfg["airtime"]["ack_bytes"], rcfg) == pytest.approx(2.5e-3, abs=0.05e-3)
    queue = [[float(k), 1, k, KIND_NORMAL, 8, 0, False, 0.0] for k in range(6)]
    queue.append([7.0, 1, 2_000_000, KIND_SOS, 8, 0, False, 7.0])
    pkt = select_packet(queue, 4, 1)
    assert pkt[0][3] == KIND_SOS and [r[2] for r in pkt[1:]] == [0, 1, 2] and len(pkt) == 4
    assert [r[2] for r in select_packet(queue[:6], 4, 0)] == [0, 1, 2, 3]  # without priority records: the oldest
    # in the simulation: no packet beyond the limit, and a lone record leaves after ``max_wait``
    rd = synthetic_rd(cfg, 3, 60.0, 0, chain)
    out = simulate_strategy(rd, "best_rssi", rcfg)
    assert out["max_packet_bytes_sent"] <= rcfg["max_packet_bytes"]
    delay = out["log"]["t_arrive"] - out["log"]["t_create"]
    one_hop = delay[out["log"]["hops"] == 1]
    assert one_hop.max() == pytest.approx(rcfg["max_wait"], abs=1e-6)  # a lone record waits ``max_wait`` and then leaves
    assert delay.max() <= 2 * rcfg["max_wait"] + 1e-6  # at most ``max_wait`` of waiting per hop


def test_sos_leaves_immediately(cfg):
    """An SOS does not wait ``max_wait``: it reaches the leader in the same step it is born (direct link, p = 1)."""
    rcfg = cfg["routing"]
    step = 355  # 35.5 s: far from the instant of the records (integers)
    rd = synthetic_rd(cfg, 3, 60.0, 0, chain, events=[(step, 1, KIND_SOS)])
    out = simulate_strategy(rd, "best_rssi", rcfg)
    sel = out["log"]["kind"] == KIND_SOS
    assert sel.sum() == 1 and (out["log"]["t_arrive"][sel] - out["log"]["t_create"][sel])[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 8. Carry
# ---------------------------------------------------------------------------


def test_isolated_node_fills_buffer_degrades_keeps_sos_and_delivers_in_time_order(cfg):
    """Node 2 isolated until 100 s: fills the buffer (30), then creates records at 0.2 Hz overwriting the oldest non-SOS and keeps the SOS; on return it delivers in time order."""
    cfg["routing"]["buffer_records"] = 30
    rcfg = cfg["routing"]
    back = 100.0
    topo = lambda i, j, tt: chain(i, j, tt) & ~(({i, j} == {1, 2}) & (tt < back))  # noqa: E731
    rd = synthetic_rd(cfg, 3, 140.0, 0, topo, events=[(100, 2, KIND_SOS)])  # SOS at 10 s, during the isolation
    out = simulate_strategy(rd, "best_rssi", rcfg)
    created = out["created"]
    mine = created[(created[:, 0] == 2) & (created[:, 3] == KIND_NORMAL)]
    during = mine[mine[:, 2] < back]
    # 29 records at 1 per second (with the SOS the 30-record buffer is full), then one every 5 s from 30 to 95 s: 14 records
    assert len(during) == 29 + 14
    late = during[during[:, 2] > 40.0]
    assert np.all(np.diff(late[:, 2]) >= 5.0 - 1e-9)  # with a full buffer: 0.2 Hz
    assert out["max_occupancy"][2] == 30 and out["lost_buffer"][2] > 0
    log = out["log"]
    sos = log["kind"] == KIND_SOS
    assert sos.sum() == 1 and log["t_arrive"][sos][0] >= back  # the SOS survives and leaves on return
    arrive = (log["origin"] == 2) & (log["kind"] == KIND_NORMAL) & (log["t_arrive"] >= back)
    assert arrive.sum() >= 20
    tc = log["t_create"][arrive]
    assert np.all(np.diff(tc) > 0)  # time order
    buffered = arrive & (log["t_create"] < back)
    assert buffered.sum() >= 20 and np.all(log["carry"][buffered])  # the buffered records arrived via carry
    assert not log["carry"][(log["origin"] == 2) & (log["t_create"] > back + 5.0)].any()  # those born after the return did not


# ---------------------------------------------------------------------------
# 9. Duplicates
# ---------------------------------------------------------------------------


def test_duplicate_records_counted_once(cfg):
    """With the ack always lost (return 0) the leader receives each record several times, but each (node, sequence) appears only once in the log."""

    def kill_ack(rd):
        rd["p"][:, 0, 1] = 0.0

    rd = synthetic_rd(cfg, 2, 40.0, 0, chain, p_override=kill_ack)
    out = simulate_strategy(rd, "best_rssi", cfg["routing"])
    keys = list(zip(out["log"]["origin"].tolist(), out["log"]["seq"].tolist()))
    cap = records_per_packet(cfg["routing"], 2)
    assert out["duplicates"] > 0 and len(keys) == len(set(keys)) == cap  # without the ack the sender keeps resending the `cap` oldest records
    assert out["fail_both"] > 0  # the sender considers them failed


# ---------------------------------------------------------------------------
# 10. direct
# ---------------------------------------------------------------------------


def test_direct_delivers_only_on_the_direct_link(cfg):
    """Chain 0-1-2-3: `direct` delivers only the records of node 1 (neighbour of the leader); `best_rssi` also those of 2 and 3 (via bridge)."""
    rd = synthetic_rd(cfg, 4, 60.0, 0, chain)
    d = simulate_strategy(rd, "direct", cfg["routing"])
    assert set(np.unique(d["log"]["origin"])) == {1}
    assert np.all(d["log"]["hops"] == 1) and d["bridge_changes"] == 0
    b = simulate_strategy(rd, "best_rssi", cfg["routing"])
    assert set(np.unique(b["log"]["origin"])) == {1, 2, 3}
    assert d["max_occupancy"][3] > 0 and max(d["queue_lengths"][2:]) > 30  # without a bridge the records of the far nodes stay in the queue (carry)


# ---------------------------------------------------------------------------
# 11. Reproducibility
# ---------------------------------------------------------------------------


def test_reproducible_same_seed_same_result_and_position_class(cfg):
    """Same seed, same result for all strategies, with probabilistic receptions; a different seed changes the unicasts."""

    def lossy(rd):
        rd["p"][:] = np.where(rd["p"] > 0, 0.6, 0.0)

    rd = synthetic_rd(cfg, 4, 80.0, 0, chain, p_override=lossy)
    for name in STRATEGIES:
        a, b = simulate_strategy(rd, name, cfg["routing"]), simulate_strategy(rd, name, cfg["routing"])
        for key in a["log"]:
            np.testing.assert_array_equal(a["log"][key], b["log"][key])
        assert a["breaks"] == b["breaks"] and a["bridge_changes"] == b["bridge_changes"]
    other = dict(rd, seed=12)
    assert not np.array_equal(simulate_strategy(rd, "dqrem", cfg["routing"])["log"]["t_arrive"], simulate_strategy(other, "dqrem", cfg["routing"])["log"]["t_arrive"])
    # leader position: thirds of the rank (10 nodes: tail 0-2, centre 3-6, head 7-9)
    offsets = np.arange(10, dtype=float)[::-1]
    assert [position_class(offsets, 9)[1], position_class(offsets, 6)[1], position_class(offsets, 0)[1]] == [0, 1, 2]


# ---------------------------------------------------------------------------
# 12. Channel profiles, connectivity, range
# ---------------------------------------------------------------------------


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        out.update(_flatten(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v})
    return out


def test_nominal_profile_is_the_previous_block7_config_and_heltec_changes_three_keys(cfg):
    """`nominal` = only ``scenario_overrides`` (the first operating point config, hence identical results); `heltec_run` changes only power, sensitivity and candidate threshold."""
    from src.refinement import apply_overrides
    from src.routing_experiment import block7_config

    previous = apply_overrides(cfg, {**cfg["routing"]["scenario_overrides"], **cfg["history"]["channel_v1"]})
    nominal = block7_config(cfg, "nominal")
    assert nominal == previous
    heltec = block7_config(cfg, "heltec_run")
    a, b = _flatten(nominal), _flatten(heltec)
    changed = {k for k in a if a[k] != b[k]}
    assert changed == {"channel.tx_power", "packets.reception.sensitivity_dbm", "routing.candidate_threshold"}
    assert (b["packets.reception.sensitivity_dbm"], b["routing.candidate_threshold"]) == (-96.0, -88.0)
    assert b["routing.candidate_threshold"] - b["packets.reception.sensitivity_dbm"] == pytest.approx(8.0)
    assert block7_config(cfg)["channel"]["tx_power"] == heltec["channel"]["tx_power"]  # default = routing.channel_profile
    with pytest.raises(KeyError):
        block7_config(cfg, "nonexistent")


def test_heltec_profile_median_rssi_in_compact_group_of_five(cfg):
    """With the equivalent power of the ``heltec_run`` profile the median RSSI in the compact 5-node group (scenario none, base channel) is −78 ± 1 dBm; at 10 dBm it is −51."""
    from src.refinement import apply_overrides
    from src.routing_experiment import group_median_rssi

    channel_only = {k: v for k, v in cfg["routing"]["channel_profiles"]["heltec_run"].items() if k.startswith("channel.")}
    assert group_median_rssi(apply_overrides(cfg, {**cfg["history"]["channel_v1"], **channel_only})) == pytest.approx(-78.0, abs=1.0)
    assert group_median_rssi(apply_overrides(cfg, cfg["history"]["channel_v1"])) == pytest.approx(-51.0, abs=1.0)


def test_connectivity_on_a_cut_chain(cfg):
    """Chain 0-1-2-3 with leader 0; halfway through, link 1-2 drops. Before, all connected and group whole; after, nodes 2 and 3 have no path and the group is split in 2 parts (1 s samples)."""
    from src.routing import connectivity_stats

    t = np.arange(0.0, 100.0, 0.1)
    rssi = np.full((len(t), 4, 4), -120.0)
    for i in range(3):
        rssi[:, i, i + 1] = rssi[:, i + 1, i] = -70.0
    rssi[t >= 50.0, 1, 2] = rssi[t >= 50.0, 2, 1] = -120.0
    c = connectivity_stats(t, rssi, 0, -1, float("nan"), cfg["routing"], 10)
    np.testing.assert_allclose(c["in"] / c["n"], [1.0, 1.0, 0.5, 0.5])
    assert c["split"] == 50 and c["split_n"] == 100
    # a single direction above threshold is not enough (the ack is needed too)
    rssi[:, 0, 1] = -120.0
    c = connectivity_stats(t, rssi, 0, -1, float("nan"), cfg["routing"], 10)
    assert c["in"][1] == 0


def test_connectivity_excludes_separated_node_from_the_group(cfg):
    """After the detachment the separated node leaves the group: it does not break the group's connectivity and is counted separately on the full graph."""
    from src.routing import connectivity_stats

    t = np.arange(0.0, 100.0, 0.1)
    rssi = np.full((len(t), 3, 3), -70.0)
    rssi[t >= 60.0, 2, :] = rssi[t >= 60.0, :, 2] = -120.0  # node 2 detaches at 60 s
    c = connectivity_stats(t, rssi, 0, 2, 60.0, cfg["routing"], 10)
    assert c["split"] == 0 and c["n"][2] == 60 and c["in"][2] == 60
    assert c["sep_n"] == 40 and c["sep_in"] == 0


def test_range_inversion_matches_path_loss(cfg):
    """The distance-only range inverts `path_loss_db`: at that distance the level equals the requested value."""
    from src.channel import path_loss_db
    from src.routing_experiment import block7_config, distance_range

    c = block7_config(cfg, "heltec_run")
    for level in (-88.0, -96.0):
        d = distance_range(level, c)
        ch = c["channel"]
        assert ch["tx_power"] + ch["antenna_gain_tx"] + ch["antenna_gain_rx"] - float(path_loss_db(np.array(d), ch)) == pytest.approx(level, abs=1e-6)

import os
import sys
import json
import tempfile
import unittest

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import honeycomb_topology as ht                                     # noqa: E402
from honeycomb_topology import HoneycombTopology, K_ACTIONS, TX_RANGE   # noqa: E402
from routing_env import (HoneycombRoutingEnv, EnvConfig, STATE_DIM, SLOT_FEATURES,  # noqa: E402
                         NODE_FEATURES, link_prr)
import tldqn_routing as T                                            # noqa: E402

torch.set_num_threads(1)


class TestTopology(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.S, cls.K = ht.parse_positions()
        cls.topo = {n: HoneycombTopology(n) for n in ht.NODE_SWEEP}

    def test_coordinates_from_ns3_source(self):
        self.assertEqual(self.S.shape, (100, 2))
        self.assertEqual(self.K.shape, (4, 2))
        self.assertEqual(len({tuple(p) for p in self.S}), 100)

    def test_colocated_sensors_documented(self):
        self.assertEqual(self.topo[100].colocated_sensor_sink(), [(23, 3), (37, 2)])

    def test_destination_is_nearest_sink(self):
        t = self.topo[100]
        for c in range(t.n):
            d = t.dist[c, t.n:]
            self.assertEqual(int(t.dest[c]), t.n + int(np.argmin(d)))

    def test_candidates_obey_range_progress_order_and_slot_limit(self):
        for n, t in self.topo.items():
            for c, d in t.reachable_pairs():
                cand = t.candidates(c, d)
                self.assertGreater(len(cand), 0)
                self.assertLessEqual(len(cand), K_ACTIONS)
                for j in cand:
                    self.assertLessEqual(t.dist[c, j], TX_RANGE)
                    self.assertTrue(j == d or (not t.is_sink(j) and t.dist[j, d] < t.dist[c, d]))
                    self.assertNotEqual(j, c)
                keys = [(t.dist[j, d], j) for j in cand]
                self.assertEqual(keys, sorted(keys))

    def test_every_sensor_reaches_its_sink_loop_free(self):
        for n, t in self.topo.items():
            for s in range(t.n):
                for pick in (lambda sl, c: sl[0], lambda sl, c: sl[-1]):
                    path, ok = t.greedy_path(s, pick)
                    self.assertTrue(ok, f"n={n} s={s}")
                    self.assertEqual(len(path), len(set(path)), "loop")
                    self.assertEqual(path[-1], int(t.dest[s]))

    def test_slot_array(self):
        t = self.topo[100]
        c, d = t.reachable_pairs()[0]
        sl = t.slots(c, d)
        self.assertEqual(sl.shape, (K_ACTIONS,))
        n_valid = len(t.candidates(c, d))
        self.assertTrue(np.all(sl[:n_valid] >= 0) and np.all(sl[n_valid:] == -1))

    def test_schema_matches_state_and_action_dimensions(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.json")
            ht.save_routing_schema(p, SLOT_FEATURES, NODE_FEATURES)
            with open(p) as fh:
                s = json.load(fh)
        self.assertEqual(s["action_dim"], K_ACTIONS)
        self.assertEqual(s["tx_range_m"], TX_RANGE)
        self.assertEqual(K_ACTIONS * len(s["slot_features"]) + len(s["node_features"]), STATE_DIM)


class TestEnvironment(unittest.TestCase):
    def setUp(self):
        self.env = HoneycombRoutingEnv(100, scenario_seed=2001)

    def test_scenario_has_ten_percent_malicious_sensors_only(self):
        self.assertEqual(int(self.env.malicious.sum()), 10)
        self.assertFalse(self.env.malicious[self.env.n:].any())

    def test_scenario_seed_reproducible_and_distinct(self):
        a = HoneycombRoutingEnv(100, scenario_seed=2001)
        b = HoneycombRoutingEnv(100, scenario_seed=2002)
        np.testing.assert_array_equal(a.malicious, self.env.malicious)
        np.testing.assert_array_equal(a.stability, self.env.stability)
        self.assertFalse(np.array_equal(a.malicious, b.malicious))

    def test_link_model_and_symmetry(self):
        np.testing.assert_allclose(self.env.stability, self.env.stability.T)
        p = link_prr(np.array([50.0, 120.0, 150.0]), EnvConfig())
        self.assertGreater(p[0], 0.99)
        self.assertAlmostEqual(p[1], 0.5, places=6)
        self.assertLess(p[2], 0.1)

    def test_trust_lower_for_malicious(self):
        rng = np.random.default_rng(0)
        mal, nor = [], []
        for _ in range(50):
            _, trust = self.env.sample_network_state(rng)
            mal.append(trust[:self.env.n][self.env.malicious[:self.env.n]].mean())
            nor.append(trust[:self.env.n][~self.env.malicious[:self.env.n]].mean())
        self.assertLess(np.mean(mal), np.mean(nor) - 0.3)

    def test_observation_layout(self):
        obs = self.env.reset(source=0, rng=np.random.default_rng(1))
        self.assertEqual(obs.shape, (STATE_DIM,))
        feats = obs[:K_ACTIONS * len(SLOT_FEATURES)].reshape(K_ACTIONS, len(SLOT_FEATURES))
        mask = self.env.action_mask()
        np.testing.assert_array_equal(feats[:, 0].astype(bool), mask)
        self.assertTrue(np.all(feats[~mask] == 0.0))
        self.assertTrue(np.all(feats[mask, 5] > 0.0))                     # progres positif

    def test_invalid_action_rejected(self):
        self.env.reset(source=0, rng=np.random.default_rng(1))
        bad = np.flatnonzero(~self.env.action_mask())
        if len(bad):
            with self.assertRaises(ValueError):
                self.env.step(int(bad[0]))
        with self.assertRaises(ValueError):
            self.env.step(K_ACTIONS)

    def test_transition_moves_to_chosen_neighbour_and_terminates(self):
        env = self.env
        env.cfg = EnvConfig(prr_d50=1e6, p_fwd_normal=1.0, p_fwd_malicious=1.0)  # tanpa kehilangan
        env.prr = link_prr(env.topo.dist, env.cfg)
        env.stability[:] = 1.0
        env.reset(source=0, rng=np.random.default_rng(1))
        visited, total = [0], 0.0
        while True:
            j = int(env.slots()[0])
            _, r, done, info = env.step(0)
            total += r
            if done:
                break
            self.assertEqual(env.cur, j)
            visited.append(j)
        self.assertEqual(info["outcome"], "delivered")
        self.assertEqual(info["to"], int(env.topo.dest[0]))
        self.assertEqual(len(visited), len(set(visited)))

    def test_malicious_relay_drops_more(self):
        env = self.env
        env.stability[:] = 1.0
        env.prr[:] = 1.0
        outcomes = {True: [], False: []}
        rng = np.random.default_rng(0)
        for _ in range(4000):
            env.reset(rng=rng)
            sl = env.slots()
            relays = [k for k, j in enumerate(sl) if j >= 0 and j != env.dest]
            if not relays:
                continue
            k = relays[0]
            j = int(sl[k])
            _, _, done, info = env.step(k)
            outcomes[bool(env.malicious[j])].append(info["outcome"] in ("malicious_drop", "relay_drop"))
        self.assertGreater(np.mean(outcomes[True]), 0.6)
        self.assertLess(np.mean(outcomes[False]), 0.05)


class TestDQNComponents(unittest.TestCase):
    def test_masked_argmax_never_picks_invalid(self):
        q = torch.tensor([[5.0, 9.0, 1.0], [0.0, -1.0, -2.0]])
        m = torch.tensor([[True, False, True], [False, True, True]])
        self.assertEqual(T.masked_argmax(q, m).tolist(), [0, 1])
        with self.assertRaises(ValueError):
            T.masked_argmax(q, torch.zeros_like(m))

    def test_bellman_targets_equation_17(self):
        r = torch.tensor([1.0, -0.1, -0.1])
        done = torch.tensor([True, False, False])
        q_next = torch.tensor([[9.0, 9.0], [2.0, 7.0], [3.0, 1.0]])
        m = torch.tensor([[False, False], [True, False], [True, True]])
        y = T.bellman_targets(r, done, q_next, m, gamma=0.9)
        np.testing.assert_allclose(y.numpy(), [1.0, -0.1 + 0.9 * 2.0, -0.1 + 0.9 * 3.0], rtol=1e-6)

    def test_transfer_initialisation_algorithm1(self):
        src = T.QNet(STATE_DIM, K_ACTIONS)
        ag = T.DQNAgent(T.DQNConfig(seed=3), init_state_dict=src.state_dict())
        for p, q, r in zip(src.parameters(), ag.online.parameters(), ag.target.parameters()):
            self.assertTrue(torch.equal(p, q) and torch.equal(q, r))
        scratch = T.DQNAgent(T.DQNConfig(seed=3))
        self.assertFalse(all(torch.equal(p, q) for p, q in zip(src.parameters(), scratch.online.parameters())))

    def test_target_network_synced_every_C_steps_only(self):
        cfg = T.DQNConfig(seed=1, target_update_steps=5, batch_size=8)
        ag = T.DQNAgent(cfg)
        buf = T.ReplayBuffer(100, STATE_DIM, K_ACTIONS, 0)
        rng = np.random.default_rng(0)
        for _ in range(50):
            buf.push(rng.normal(size=STATE_DIM), int(rng.integers(K_ACTIONS)), rng.normal(),
                     rng.normal(size=STATE_DIM), bool(rng.random() < 0.3), np.ones(K_ACTIONS, bool))
        for step in range(1, 13):
            before = [p.clone() for p in ag.target.parameters()]
            ag.update(buf.sample(8))
            changed = not all(torch.equal(a, b) for a, b in zip(before, ag.target.parameters()))
            self.assertEqual(changed, step % 5 == 0, f"langkah {step}")
        self.assertEqual(ag.target_syncs, 2)

    def test_sgd_optimizer_is_default(self):
        self.assertEqual(T.DQNConfig().optimizer, "sgd")
        self.assertIsInstance(T.DQNAgent(T.DQNConfig()).opt, torch.optim.SGD)

    def test_epsilon_greedy_respects_mask(self):
        ag = T.DQNAgent(T.DQNConfig(seed=0))
        mask = np.array([False, True, False, True, False, False])
        acts = {ag.act(np.zeros(STATE_DIM, np.float32), mask, eps=1.0) for _ in range(200)}
        self.assertEqual(acts, {1, 3})


class TestEvaluationAndLearning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = HoneycombRoutingEnv(100, scenario_seed=2001)
        cls.ev = T.make_eval_set(cls.env, n_states=1, seed=5000)
        cls.refs = {k: T.evaluate_reference(cls.env, cls.ev, k) for k in ("random", "oracle")}

    def test_exact_evaluation_matches_monte_carlo(self):
        exact = T.evaluate_policy(self.env, T.greedy_progress_policy, self.ev)
        st = self.ev[0][1]
        self.env.rng = np.random.default_rng(9)
        deliv, rets = [], []
        for _ in range(40):
            for s, _ in self.ev:
                self.env.reset(source=s, network_state=st)
                done, R = False, 0.0
                while not done:
                    _, r, done, info = self.env.step(0)
                    R += r
                deliv.append(info["outcome"] == "delivered"); rets.append(R)
        se = np.sqrt(exact["pdr"] * (1 - exact["pdr"]) / len(deliv))
        self.assertLess(abs(np.mean(deliv) - exact["pdr"]), 4 * se)
        self.assertLess(abs(np.mean(rets) - exact["expected_return"]), 0.05)

    def test_oracle_dominates_and_greedy_is_not_optimal(self):
        g = T.evaluate_policy(self.env, T.greedy_progress_policy, self.ev)
        self.assertGreater(self.refs["oracle"]["expected_return"], g["expected_return"] + 0.2)
        self.assertGreater(self.refs["oracle"]["expected_return"], self.refs["random"]["expected_return"])

    def test_dp_random_matches_rollout_of_uniform_policy(self):
        env = self.env
        st = self.ev[0][1]
        rng = np.random.default_rng(3)
        env.rng = np.random.default_rng(4)
        rets = []
        for _ in range(30):
            for s, _ in self.ev:
                env.reset(source=s, network_state=st)
                done, R = False, 0.0
                while not done:
                    a = int(rng.choice(np.flatnonzero(env.action_mask())))
                    _, r, done, _ = env.step(a)
                    R += r
                rets.append(R)
        self.assertLess(abs(np.mean(rets) - self.refs["random"]["expected_return"]), 0.05)

    def test_references_use_no_future_information(self):
        """Referensi = nilai harapan: deterministik, tidak bergantung/menyentuh RNG lingkungan."""
        for kind in ("random", "oracle", "bayes"):
            self.env.rng = np.random.default_rng(1)
            a = T.evaluate_reference(self.env, self.ev, kind)
            self.env.rng = np.random.default_rng(999)
            state = self.env.rng.bit_generator.state
            b = T.evaluate_reference(self.env, self.ev, kind)
            self.assertEqual(a, b, kind)
            self.assertEqual(self.env.rng.bit_generator.state, state, kind)

    def test_bayes_uses_only_observed_trust(self):
        """Keputusan Bayes tidak berubah bila label malicious diubah sementara trust tetap
        (nilai evaluasinya boleh berubah karena dievaluasi pada dinamika sebenarnya)."""
        st = self.ev[0][1]
        self.env.set_network_state(*st)
        post = T.posterior_pfwd(self.env, self.env.trust)
        saved = self.env.malicious.copy()
        try:
            self.env.malicious[:] = False
            post2 = T.posterior_pfwd(self.env, self.env.trust)
        finally:
            self.env.malicious[:] = saved
        np.testing.assert_array_equal(post, post2)

    def test_longest_route_within_ttl(self):
        """DP referensi tidak memodelkan TTL: valid hanya jika rute terpanjang <= max_hops."""
        depths = {n: T.longest_route_hops(HoneycombTopology(n)) for n in ht.NODE_SWEEP}
        self.assertLessEqual(max(depths.values()), EnvConfig().max_hops, depths)
        self.assertEqual(depths[100], 12)        # dicatat: tepat di batas

    def test_bayes_decisions_independent_of_malicious_labels(self):
        """Tingkat KEPUTUSAN: kebijakan Bayes identik bila label malicious diacak sementara
        trust teramati tetap; kebijakan oracle berubah (kontrol kewarasan)."""
        st = self.ev[0][1]
        self.env.set_network_state(*st)
        _, _, pol_b = T._dp_values(self.env, "bayes", return_policy=True)
        _, _, pol_o = T._dp_values(self.env, "oracle", return_policy=True)
        saved = self.env.malicious.copy()
        try:
            rng = np.random.default_rng(0)
            perm = saved.copy()
            perm[:self.env.n] = rng.permutation(saved[:self.env.n])
            self.env.malicious[:] = perm
            _, _, pol_b2 = T._dp_values(self.env, "bayes", return_policy=True)
            _, _, pol_o2 = T._dp_values(self.env, "oracle", return_policy=True)
        finally:
            self.env.malicious[:] = saved
        self.assertEqual(pol_b, pol_b2)
        self.assertNotEqual(pol_o, pol_o2)

    def test_posterior_is_finite_and_ordered(self):
        st = self.ev[0][1]
        post = T.posterior_pfwd(self.env, st[1])
        self.assertTrue(np.all(np.isfinite(post)))
        cfg = self.env.cfg
        self.assertTrue(np.all((post >= cfg.p_fwd_malicious - 1e-12) & (post <= 1.0)))
        mal = self.env.malicious[:self.env.n]
        self.assertLess(post[:self.env.n][mal].mean(), post[:self.env.n][~mal].mean())

    def test_reference_ordering(self):
        b = T.evaluate_reference(self.env, self.ev, "bayes")
        self.assertLessEqual(b["expected_return"], self.refs["oracle"]["expected_return"] + 1e-9)
        self.assertGreater(b["expected_return"], self.refs["random"]["expected_return"])

    def test_dqn_learns_beyond_random_and_greedy(self):
        cfg = T.DQNConfig(seed=1, lr=0.05, eps_decay_episodes=600)
        _, hist = T.train(self.env, cfg, 900, eval_set=self.ev, eval_every=300,
                          refs=self.refs, log_fn=None)
        g = T.evaluate_policy(self.env, T.greedy_progress_policy, self.ev)
        self.assertGreater(hist[-1]["expected_return"], g["expected_return"] + 0.1)
        self.assertGreater(hist[-1]["normalized_gain"], 0.4)

    def test_training_reproducible_with_seed(self):
        cfg = T.DQNConfig(seed=2, lr=0.05, eps_decay_episodes=200)
        a, _ = T.train(self.env, cfg, 250, log_fn=None)
        b, _ = T.train(self.env, cfg, 250, log_fn=None)
        for p, q in zip(a.online.parameters(), b.online.parameters()):
            self.assertTrue(torch.equal(p, q))


if __name__ == "__main__":
    unittest.main()

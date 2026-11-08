import os
import sys
import csv
import json
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import export_routes as X                                   # noqa: E402
from honeycomb_topology import HoneycombTopology, TX_RANGE   # noqa: E402

HAVE_MODEL = os.path.exists(X.DEFAULT_MODEL)


def write_scenario(d, n, seed, malicious):
    with open(os.path.join(d, f"bigan_decisions_{n}_s{seed}.csv"), "w") as f:
        f.write("node_id,actual_mal,anomaly_score,threshold,pred_intrusion\n")
        for i in range(n):
            f.write(f"{i},{int(i in malicious)},0.0,0.0,0\n")


def read_table(path):
    with open(path) as f:
        return [{k: (float(v) if k == "q_value" else int(v)) for k, v in r.items()}
                for r in csv.DictReader(f)]


@unittest.skipUnless(HAVE_MODEL, "checkpoint TLDQN arsip tidak tersedia")
class TestExportRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.d = cls.tmp.name
        cls.cases = {(20, 1): {3, 7}, (60, 2): {1, 5, 9, 14, 30, 41}, (100, 1): set(range(0, 100, 10))}
        for (n, s), mal in cls.cases.items():
            write_scenario(cls.d, n, s, mal)
        for pol, sub in (("tldqn", "t"), ("greedy", "g")):
            for (n, s) in cls.cases:
                X.main(["--policy", pol, "--nodes", str(n), "--seeds", str(s),
                        "--scenario-dir", cls.d, "--out-dir", os.path.join(cls.d, sub)])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def table(self, sub, n, s):
        return read_table(os.path.join(self.d, sub, f"tldqn_nexthop_{n}_s{s}.csv"))

    def test_schema_columns(self):
        with open(os.path.join(self.d, "t", "tldqn_nexthop_20_s1.csv")) as f:
            self.assertEqual(next(csv.reader(f)), X.COLUMNS)

    def test_covers_exactly_the_reachable_pairs(self):
        for (n, s) in self.cases:
            topo = HoneycombTopology(n)
            got = {(r["node_id"], n + r["dest_sink"]) for r in self.table("t", n, s)}
            self.assertEqual(got, set(topo.reachable_pairs()))

    def test_next_hop_is_valid_candidate_in_range(self):
        for (n, s) in self.cases:
            topo = HoneycombTopology(n)
            for r in self.table("t", n, s):
                c, t = r["node_id"], n + r["dest_sink"]
                j = n + r["next_hop"] if r["next_hop_is_sink"] else r["next_hop"]
                self.assertEqual(int(topo.slots(c, t)[r["slot"]]), j)
                self.assertLessEqual(topo.dist[c, j], TX_RANGE)
                self.assertEqual(r["n_candidates"], len(topo.candidates(c, t)))
                self.assertEqual(r["own_dest"], int(topo.dest[c] == t))
                if r["next_hop_is_sink"]:
                    self.assertEqual(r["next_hop"], r["dest_sink"])

    def test_every_sensor_reaches_own_sink_loop_free(self):
        for (n, s) in self.cases:
            rows = self.table("t", n, s)
            topo = HoneycombTopology(n)
            for src in range(n):
                path = X.follow(rows, n, src)
                self.assertEqual(path[-1], int(topo.dest[src]))
                self.assertEqual(len(path), len(set(path)))

    def test_ground_truth_comes_from_scenario_file(self):
        mal, _ = X.load_ground_truth(self.d, 60, 2)
        self.assertEqual(set(np.flatnonzero(mal)), self.cases[(60, 2)])
        env = X.build_env(60, 2, mal, X.EnvConfig())
        self.assertEqual(set(np.flatnonzero(env.malicious)), self.cases[(60, 2)])
        self.assertTrue(np.all(env.energy == 1.0))

    def test_node_malicious_column_matches_scenario(self):
        for (n, s), mal in self.cases.items():
            rows = self.table("t", n, s)
            self.assertEqual({r["node_id"] for r in rows if r["node_malicious"]}, mal)
            self.assertTrue(all(r["node_malicious"] == int(r["node_id"] in mal) for r in rows))

    def test_incomplete_scenario_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "bigan_decisions_20_s1.csv"), "w") as f:
                f.write("node_id,actual_mal\n0,1\n")
            with self.assertRaises(ValueError):
                X.load_ground_truth(d, 20, 1)

    def test_greedy_reference_picks_slot_zero_and_tldqn_differs(self):
        g = self.table("g", 100, 1)
        t = self.table("t", 100, 1)
        self.assertTrue(all(r["slot"] == 0 for r in g))
        diff = np.mean([a["slot"] != b["slot"] for a, b in zip(t, g)])
        self.assertGreater(diff, 0.2)

    def test_deterministic_export(self):
        with tempfile.TemporaryDirectory() as out:
            X.main(["--nodes", "60", "--seeds", "2", "--scenario-dir", self.d, "--out-dir", out])
            a = open(os.path.join(out, "tldqn_nexthop_60_s2.csv")).read()
        self.assertEqual(a, open(os.path.join(self.d, "t", "tldqn_nexthop_60_s2.csv")).read())

    def test_meta_records_model_hash_and_assumptions(self):
        with open(os.path.join(self.d, "t", "tldqn_nexthop_meta.json")) as f:
            meta = json.load(f)
        self.assertEqual(meta["model_sha256"], X.sha256(X.DEFAULT_MODEL))
        self.assertIn("malicious_relay", meta["assumptions"])
        self.assertEqual(meta["schema"]["action_dim"], 6)


if __name__ == "__main__":
    unittest.main()

"""Uji telemetry langkah 2 dengan telemetry SINTETIS (tanpa NS-3).

Generator di bawah meniru keluaran honeycomb-sdwsn.cc: watch acak (observer i ->
tetangga j) dengan kategori yang bergantung pada perilaku j, lalu diagregasi ke
berkas node / link / audit / meta / metrics yang konsisten satu sama lain."""
import csv
import json
import os
import re
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

import telemetry_schema as S          # noqa: E402
import telemetry_trust as T           # noqa: E402
import telemetry_eval as E            # noqa: E402
import export_routes as X             # noqa: E402

CC = os.path.join(HERE, "..", "..", "ns3-sim", "honeycomb-sdwsn.cc")


def synth(d, n=20, seed=1, malicious=(3, 11), n_win=4, watches=4000, p_fwd_mal=0.30,
          p_fwd_norm=0.98, b=0.40, c=0.10, rng_seed=0):
    """Tulis telemetry sintetis yang konsisten. Mengembalikan label (bool array).

    Model observasi (sama dengan asumsi koreksi di telemetry_trust):
      b = peluang ada kegagalan terima KEBETULAN di window watch (independen dari j)
      c = peluang frame penerusan j sendiri gagal di observer (tabrakan)
    Diteruskan: gagal (c) -> unobserved_rxfail; selain itu terdengar (+ heard_rxfail
    dgn peluang b). Di-drop: unobserved_rxfail dgn peluang b, selain itu clean."""
    rng = np.random.default_rng(rng_seed)
    mal = np.zeros(n, bool)
    mal[list(malicious)] = True
    W = 30.0
    node = {(i, w): {c: 0 for c in S.NODE_COLUMNS} for i in range(n) for w in range(n_win)}
    for (i, w), r in node.items():
        r.update(node_id=i, window=w, t_start=w * W, t_end=(w + 1) * W)
        r["self_energy_j"] = 0.01 * (1 + i)
        r["self_ip_orig_pkts"] = 5
    links, audit = {}, {}
    cats = {c: 0 for c in S.WD_CATEGORIES}
    for _ in range(watches):
        i, j = rng.choice(n, 2, replace=False)
        w = int(rng.integers(n_win))
        p = p_fwd_mal if mal[j] else p_fwd_norm
        tx = rng.random() < p
        coincidental = rng.random() < b
        heard_rx = False
        if tx and rng.random() < c:
            cat, cause = "unobserved_rxfail", "j_tx_rxfail_at_observer"
        elif tx:
            cat = "forwarded" if rng.random() < 0.9 else "forwarded_late"
            cause, heard_rx = "j_tx_heard", coincidental
        else:
            cat = "unobserved_rxfail" if coincidental else "unobserved_clean"
            cause = "j_ip_drop_route_error" if mal[j] else "j_ip_forwarded_not_tx"
        cats[cat] += 1
        lk = links.setdefault((w, i, "sensor", j), {k: 0 for k in S.LINK_COLUMNS})
        lk.update(window=w, observer=i, neighbor_type="sensor", neighbor_id=j)
        lk["wd_started"] += 1
        lk[f"wd_{cat}"] += 1
        lk["wd_heard_rxfail"] += int(heard_rx)
        lk["mac_handoff_acked"] += 1
        lk["rx_ok_from"] += 1
        node[(j, w)]["obs_watched"] += 1
        node[(j, w)][f"obs_{cat}"] += 1
        node[(i, w)]["wd_started_as_observer"] += 1
        node[(i, w)]["self_mac_acked"] += 1
        audit[(w, i, j, cat, cause)] = audit.get((w, i, j, cat, cause), 0) + 1
    t = S.tag(n, seed)
    with open(os.path.join(d, f"telemetry_{t}.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=S.NODE_COLUMNS)
        wr.writeheader()
        for k in sorted(node):
            wr.writerow(node[k])
    with open(os.path.join(d, f"telemetry_links_{t}.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=S.LINK_COLUMNS)
        wr.writeheader()
        for k in sorted(links):
            r = links[k]
            r.update(rssi_mean_dbm=-14.0, snr_mean_db=80.0, wd_delay_mean_s=0.002)
            wr.writerow(r)
    with open(os.path.join(d, f"watchdog_audit_{t}.csv"), "w", newline="") as f:
        f.write(",".join(S.AUDIT_COLUMNS) + "\n")
        for (w, i, j, cat, cause), c in sorted(audit.items()):
            f.write(f"{w},{i},{j},{cat},{cause},{c}\n")
    meta = {"num_nodes": n, "seed": seed, "window_s": W, "n_windows": n_win,
            "watches": watches, **{f"wd_{c}": v for c, v in cats.items()}}
    with open(os.path.join(d, f"telemetry_meta_{t}.json"), "w") as f:
        json.dump(meta, f)
    energy = sum(r["self_energy_j"] for r in node.values())
    with open(os.path.join(d, f"metrics_{t}.csv"), "w") as f:
        f.write("metric,value\n")
        for k, v in [("telemetry_ip_forward_udp_total", 930),
                     ("telemetry_ip_forward_udp_delivered", 900),
                     ("flowmon_udp_times_forwarded", 900),
                     ("telemetry_ip_forward_tcp_total", 5), ("flowmon_tcp_times_forwarded", 4),
                     ("telemetry_ip_orig_udp_total", 400), ("flowmon_udp_tx_packets", 400),
                     ("telemetry_ip_local_udp_total", 380), ("flowmon_udp_rx_packets", 380),
                     ("sensor_energy_consumed_J", energy), ("grayhole_drop_prob", 0.7)]:
            f.write(f"{k},{v!r}\n")
    with open(os.path.join(d, f"bigan_decisions_{t}.csv"), "w") as f:
        f.write("node_id,actual_mal,anomaly_score,threshold,pred_intrusion\n")
        for i in range(n):
            f.write(f"{i},{int(mal[i])},0.0,0.0,0\n")
    return mal


def rewrite(path, fn):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        cols, rows = r.fieldnames, list(r)
    cols, rows = fn(cols, rows)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


class TestSchemaAndValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        self.mal = synth(self.d)
        self.metrics = os.path.join(self.d, "metrics_20_s1.csv")

    def tearDown(self):
        self.tmp.cleanup()

    def test_consistent_synthetic_passes(self):
        self.assertEqual(S.validate_dir(self.d, 20, 1, self.metrics), [])

    def test_detects_node_vs_link_mismatch(self):
        def tamper(cols, rows):
            rows[5]["obs_watched"] = str(int(rows[5]["obs_watched"]) + 1)
            rows[5]["obs_forwarded"] = str(int(rows[5]["obs_forwarded"]) + 1)
            return cols, rows
        rewrite(S.paths(self.d, 20, 1)["node"], tamper)
        errs = S.validate_dir(self.d, 20, 1, self.metrics)
        self.assertTrue(any("agregasi" in e for e in errs), errs)

    def test_detects_category_sum_error(self):
        def tamper(cols, rows):
            rows[0]["wd_forwarded"] = str(int(rows[0]["wd_forwarded"]) + 1)
            return cols, rows
        rewrite(S.paths(self.d, 20, 1)["links"], tamper)
        errs = S.validate_dir(self.d, 20, 1, self.metrics)
        self.assertTrue(any("kategori" in e for e in errs), errs)

    def test_detects_flowmon_and_energy_mismatch(self):
        def tamper(cols, rows):
            for r in rows:
                if r["metric"] == "flowmon_udp_times_forwarded":
                    r["value"] = "901"
                if r["metric"] == "sensor_energy_consumed_J":
                    r["value"] = str(float(r["value"]) + 0.01)
            return cols, rows
        rewrite(self.metrics, tamper)
        errs = S.validate_dir(self.d, 20, 1, self.metrics)
        self.assertTrue(any("flowmon_udp_times_forwarded" in e for e in errs), errs)
        # selisih TCP (5 vs 4 di sintetis) bukan kegagalan: tidak diklaim sama
        self.assertFalse(any("tcp" in e for e in errs), errs)
        self.assertTrue(any("energi" in e for e in errs), errs)

    def test_forward_definition_legacy_and_delivered(self):
        base = {"flowmon_udp_times_forwarded": 900, "telemetry_ip_orig_udp_total": 400,
                "telemetry_ip_local_udp_total": 380}
        # baru: delivered harus sama persis; total boleh lebih besar (paket hilang)
        self.assertEqual(S._check_forward_definition(
            {**base, "telemetry_ip_forward_udp_total": 950,
             "telemetry_ip_forward_udp_delivered": 900}), [])
        self.assertTrue(S._check_forward_definition(
            {**base, "telemetry_ip_forward_udp_total": 950,
             "telemetry_ip_forward_udp_delivered": 899}))
        # lama: selisih >= 0 dan nol bila tidak ada yang hilang
        self.assertEqual(S._check_forward_definition(
            {**base, "telemetry_ip_forward_udp_total": 950}), [])
        self.assertTrue(S._check_forward_definition(
            {**base, "telemetry_ip_forward_udp_total": 899}))
        nolost = {**base, "telemetry_ip_local_udp_total": 400}
        self.assertTrue(S._check_forward_definition({**nolost, "telemetry_ip_forward_udp_total": 901}))
        self.assertEqual(S._check_forward_definition({**nolost, "telemetry_ip_forward_udp_total": 900}), [])
        self.assertEqual(S.forward_definition({"telemetry_ip_forward_udp_delivered": 1}), "delivered")

    def test_energy_tolerance_is_quantization_bound(self):
        nodes = S.load_nodes(S.paths(self.d, 20, 1)["node"])
        meta = {"num_nodes": 20, "sim_time_s": 600.0}
        for r in nodes:
            r["self_phy_tx_mpdu"] = 100
        tol = S.energy_tolerance(nodes, meta)
        # 1e-6 + 1e-9 * (20*600 + 4 * 80 baris * 100)
        self.assertAlmostEqual(tol, 1e-6 + 1e-9 * (12000 + 4 * 8000))
        self.assertLess(tol, 1e-3)                  # masih jauh di bawah 0.01 J (tamper)

    def test_label_column_rejected(self):
        def add_label(cols, rows):
            for r in rows:
                r["node_malicious"] = "0"
            return cols + ["node_malicious"], rows
        rewrite(S.paths(self.d, 20, 1)["node"], add_label)
        with self.assertRaises(ValueError):
            S.load_nodes(S.paths(self.d, 20, 1)["node"])
        self.assertTrue(S.check_no_leakage(S.NODE_COLUMNS + ["node_malicious"]))
        self.assertEqual(S.check_no_leakage(S.NODE_COLUMNS, S.LINK_COLUMNS), [])

    def test_incomplete_windows_rejected(self):
        rewrite(S.paths(self.d, 20, 1)["node"], lambda c, r: (c, r[:-1]))
        errs = S.validate_dir(self.d, 20, 1)
        self.assertTrue(any("tidak lengkap" in e for e in errs), errs)


class TestTrust(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = self.tmp.name
        self.mal = synth(self.d, watches=6000)
        self.nodes = S.load_nodes(S.paths(self.d, 20, 1)["node"])
        self.links = S.load_links(S.paths(self.d, 20, 1)["links"])

    def tearDown(self):
        self.tmp.cleanup()

    def test_discard_formula_and_prior(self):
        rows = [dict(r) for r in self.nodes if r["node_id"] in (0, 1)]
        for r in rows:
            for c in S.TRUST_INPUT_COLUMNS:
                r[c] = 0
        rows[0].update(obs_forwarded=7, obs_forwarded_late=1, obs_unobserved_clean=2,
                       obs_unobserved_rxfail=50)
        t = T.beta_trust(rows, 2)
        self.assertAlmostEqual(t["trust"][0], (8 + 1) / (8 + 2 + 2))
        self.assertAlmostEqual(t["trust"][1], 0.5)            # tanpa observasi = prior
        self.assertEqual(t["n_ambiguous"][0], 50)

    def _link(self, obs, nbr, fwd, rxf, clean, hrx, ntype="sensor"):
        r = {c: 0 for c in S.LINK_COLUMNS}
        r.update(window=0, observer=obs, neighbor_type=ntype, neighbor_id=nbr, wd_forwarded=fwd,
                 wd_unobserved_rxfail=rxf, wd_unobserved_clean=clean, wd_heard_rxfail=hrx,
                 wd_started=fwd + rxf + clean)
        return r

    def test_corrected_formula(self):
        # observer 0: b = (25 + 15) / (100 + 60) = 0.25 (semua watch terdengar observer).
        # Tetangga 1: N=100, U=30 -> D = 30/0.75 = 40.
        links = [self._link(0, 2, 100, 0, 0, 25), self._link(0, 1, 60, 10, 30, 15),
                 self._link(0, 2, 0, 0, 0, 0, ntype="sink")]
        t = T.corrected_trust(links, 3)
        self.assertAlmostEqual(t["drops_est"][1], 40.0)
        self.assertAlmostEqual(t["trust"][1], (100 - 40 + 1) / (100 + 2))
        self.assertAlmostEqual(t["trust"][0], 0.5)            # tanpa observasi = prior
        # D dibatasi A + U pada agregat node (b = 0.9 -> U/(1-b) = 50 > A + U = 6)
        t2 = T.corrected_trust([self._link(0, 2, 10, 0, 0, 9), self._link(0, 1, 0, 1, 5, 0)], 3)
        self.assertAlmostEqual(t2["drops_est"][1], 6.0)
        # ... tetapi TIDAK per sel: dua window, sel kedua melampaui batasnya sendiri
        l1, l2 = self._link(0, 1, 0, 0, 1, 0), self._link(0, 1, 0, 9, 1, 0)
        l2["window"] = 1
        t4 = T.corrected_trust([self._link(0, 2, 10, 0, 0, 5), l1, l2], 3)   # b = 0.5
        self.assertAlmostEqual(t4["drops_est"][1], 4.0)                     # 2 + 2, batas 11
        # b = 0 -> semua ambigu dihitung terusan (berasal dari tabrakan frame j)
        t3 = T.corrected_trust([self._link(0, 1, 50, 10, 5, 0)], 2)
        self.assertAlmostEqual(t3["drops_est"][1], 5.0)

    def test_corrected_removes_rxfail_bias(self):
        tc = T.corrected_trust(self.links, 20)
        td = T.beta_trust(self.nodes, 20)
        pm = 1 - tc["drops_est"][self.mal].sum() / tc["n_watched"][self.mal].sum()
        pd = td["n_forwarded"][self.mal].sum() / td["n_watched"][self.mal].sum()
        self.assertAlmostEqual(pm, 0.30, delta=0.03)          # tanpa bias
        self.assertGreater(pd, 0.36)                          # discard: bias naik (b > c)
        self.assertAlmostEqual(tc["b_pooled"], 0.40, delta=0.03)

    def test_self_columns_never_reach_trust(self):
        base = T.corrected_trust(self.links, 20)["trust"]
        bd = T.beta_trust(self.nodes, 20)["trust"]
        mod = [dict(r) for r in self.nodes]
        for r in mod:
            r["obs_unobserved_rxfail"] += 100             # discard: ambigu tak berpengaruh
            for c in S.NODE_COLUMNS:
                if c.startswith("self_"):
                    r[c] = 12345
        np.testing.assert_array_equal(T.beta_trust(mod, 20)["trust"], bd)
        lmod = [dict(r) for r in self.links]
        for r in lmod:
            for c in S.LINK_COLUMNS:
                if c not in S.TRUST_INPUT_LINK_COLUMNS and c not in ("neighbor_type",):
                    r[c] = 999
        np.testing.assert_array_equal(T.corrected_trust(lmod, 20)["trust"], base)

    def test_trust_separates_classes_on_synthetic(self):
        for tr in (T.corrected_trust(self.links, 20)["trust"], T.beta_trust(self.nodes, 20)["trust"]):
            self.assertGreater(E.auc(1 - tr, self.mal), 0.95)
            self.assertLess(tr[self.mal].max(), tr[~self.mal].min())

    def test_window_subset(self):
        t_all = T.corrected_trust(self.links, 20)
        t0 = T.corrected_trust(self.links, 20, windows={0})
        self.assertLess(t0["n_watched"].sum(), t_all["n_watched"].sum())

    def test_cli_roundtrip(self):
        for method, ref in (("corrected", T.corrected_trust(self.links, 20)["trust"]),
                            ("discard", T.beta_trust(self.nodes, 20)["trust"])):
            out = os.path.join(self.d, "trust_" + method)
            T.main(["--telem-dir", self.d, "--out-dir", out, "--nodes", "20", "--seeds", "1",
                    "--method", method])
            tr = T.load_trust(os.path.join(out, "trust_telemetry_20_s1.csv"), 20)
            np.testing.assert_allclose(tr, ref, atol=1e-6)
            with open(os.path.join(out, "trust_telemetry_meta.json")) as f:
                self.assertEqual(json.load(f)["method"], method)


class TestEval(unittest.TestCase):
    def test_forwarding_ratio_and_repeat(self):
        with tempfile.TemporaryDirectory() as d:
            synth(d, watches=20000, malicious=(1, 4, 9, 15))
            rep = E.evaluate(_Args(d))
            self.assertTrue(rep["all_valid"])
            m, nm = rep["pooled"]["malicious"], rep["pooled"]["normal"]
            self.assertAlmostEqual(m["est_corrected"], 0.30, delta=0.03)
            self.assertAlmostEqual(nm["est_corrected"], 0.98, delta=0.01)
            self.assertGreater(m["est"], m["est_corrected"])      # bias discard terlihat
            for r in (m, nm):
                self.assertLessEqual(r["lo"], r["est"])
                self.assertLessEqual(r["est"], r["hi"])
                self.assertLessEqual(r["lo"], r["est_corrected"] + 1e-12)
                self.assertLessEqual(r["est_corrected"], r["hi"] + 1e-12)
            sc = rep["scenarios"][0]
            self.assertTrue(sc["repeat_identical"])
            self.assertTrue(rep["repeat_all_identical"])
            self.assertAlmostEqual(sc["audit"]["malicious"]["true_tx_ratio"], 0.30, delta=0.03)


class _Args:
    def __init__(self, d):
        self.telem_dir = self.scenario_dir = self.metrics_dir = self.repeat_dir = d
        self.metrics_tag, self.nodes, self.seeds, self.trust_dir = "", [20], [1], None
        self.passive_tag = None


class TestPassiveCompare(unittest.TestCase):
    def test_wallclock_and_telemetry_rows_ignored(self):
        ref = {"pdr_percent": 97.5, "rt_compute_ms": 0.1, "telemetry_enabled": 0.0}
        tel = {"pdr_percent": 97.5, "rt_compute_ms": 0.3, "telemetry_enabled": 1.0,
               "wd_watches": 10.0}
        self.assertTrue(E.compare_passive(ref, tel)["identical"])
        tel["pdr_percent"] = 97.4
        r = E.compare_passive(ref, tel)
        self.assertFalse(r["identical"])
        self.assertIn("pdr_percent", r["differences"])


class TestExportTrustSource(unittest.TestCase):
    def test_none_is_uninformative_and_label_independent(self):
        n = 40
        a = X.build_env(n, 1, np.isin(np.arange(n), [2, 5, 7, 30]), X.EnvConfig(), "none")
        b = X.build_env(n, 1, np.isin(np.arange(n), [0, 1, 3, 4]), X.EnvConfig(), "none")
        self.assertTrue(np.all(a.trust[:n] == 0.5) and np.all(a.trust[n:] == 1.0))
        for c, t in a.topo.reachable_pairs():
            for e in (a, b):
                e.cur, e.dest = c, t
            np.testing.assert_array_equal(a.observe(), b.observe())

    def test_trust_feature_index(self):
        # analyze_stage2_tables mengubah obs[k * len(SLOT_FEATURES) + idx_trust]
        from routing_env import SLOT_FEATURES
        n = 40
        st = np.linspace(0.05, 0.95, n)
        env = X.build_env(n, 1, np.zeros(n, bool), X.EnvConfig(), "telemetry", st)
        tix = SLOT_FEATURES.index("trust")
        checked = 0
        for c, t in env.topo.reachable_pairs():
            env.cur, env.dest = c, t
            obs = env.observe()
            for k, j in enumerate(env.topo.slots(c, t)):
                if j >= 0:
                    self.assertAlmostEqual(obs[k * len(SLOT_FEATURES) + tix], env.trust[j], places=6)
                    checked += 1
        self.assertGreater(checked, 50)

    def test_telemetry_trust_injected(self):
        n = 20
        st = np.linspace(0.1, 0.9, n)
        env = X.build_env(n, 1, np.zeros(n, bool), X.EnvConfig(), "telemetry", st)
        np.testing.assert_allclose(env.trust[:n], st)
        self.assertTrue(np.all(env.trust[n:] == 1.0))
        with self.assertRaises(ValueError):
            X.build_env(n, 1, np.zeros(n, bool), X.EnvConfig(), "telemetry", st[:-1])

    def test_simulated_unchanged_default(self):
        n, mal = 30, np.isin(np.arange(30), [1, 2, 3])
        a = X.build_env(n, 2, mal, X.EnvConfig())
        b = X.build_env(n, 2, mal, X.EnvConfig(), "simulated")
        np.testing.assert_array_equal(a.trust, b.trust)
        rng = np.random.default_rng(20000 * 2 + n)
        ref = X.HoneycombRoutingEnv(n, 10000 * 2 + n, X.EnvConfig())
        ref.malicious[:] = False
        ref.malicious[:n] = mal
        np.testing.assert_array_equal(a.trust, ref.sample_network_state(rng)[1])

    @unittest.skipUnless(os.path.exists(X.DEFAULT_MODEL), "checkpoint TLDQN tidak tersedia")
    def test_cli_telemetry_routes_label_independent(self):
        with tempfile.TemporaryDirectory() as d:
            synth(d, n=20, malicious=(3, 11))
            tdir = os.path.join(d, "trust")
            T.main(["--telem-dir", d, "--out-dir", tdir, "--nodes", "20", "--seeds", "1"])
            outs = []
            for k, mal in enumerate(((3, 11), (0, 19))):
                sd = os.path.join(d, f"scen{k}")
                os.makedirs(sd)
                with open(os.path.join(sd, "bigan_decisions_20_s1.csv"), "w") as f:
                    f.write("node_id,actual_mal\n")
                    for i in range(20):
                        f.write(f"{i},{int(i in mal)}\n")
                od = os.path.join(d, f"out{k}")
                X.main(["--nodes", "20", "--seeds", "1", "--scenario-dir", sd, "--out-dir", od,
                        "--trust-source", "telemetry", "--trust-dir", tdir])
                with open(os.path.join(od, "tldqn_nexthop_20_s1.csv")) as f:
                    outs.append([r[:5] for r in csv.reader(f)])    # tanpa q/node_malicious
                with open(os.path.join(od, "tldqn_nexthop_meta.json")) as f:
                    self.assertEqual(json.load(f)["trust_source"], "telemetry")
            self.assertEqual(outs[0], outs[1])


def _cc_header(src, first):
    """Gabungkan literal string C++ yang membentuk header CSV mulai dari `first`."""
    i = src.index('"' + first)
    out = []
    for m in re.finditer(r'"((?:[^"\\]|\\.)*)"', src[i:]):
        out.append(m.group(1))
        if out[-1].endswith("\\n"):
            break
    return "".join(out)[:-2].split(",")


class TestCppContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(CC) as f:
            cls.src = f.read()
        a = cls.src.index("namespace telem {")
        b = cls.src.index("}  // namespace telem")
        cls.telem = cls.src[a:b]

    def test_headers_match_schema(self):
        self.assertEqual(_cc_header(self.telem, "node_id,window,"), S.NODE_COLUMNS)
        self.assertEqual(_cc_header(self.telem, "window,observer,neighbor_type"), S.LINK_COLUMNS)
        self.assertEqual(_cc_header(self.telem, "window,observer,neighbor,wd_category"),
                         S.AUDIT_COLUMNS)

    def test_telemetry_code_reads_no_labels(self):
        # Komentar dibuang dulu (dokumentasi boleh menyebut nama-nama ini).
        code = re.sub(r"//[^\n]*", "", self.telem)
        for tok in ("g_bigan", "actualMal", "GrayholeRouting", "g_grayhole", "ExportMalicious",
                    "g_malicious", "g_blocked", "node_malicious"):
            self.assertNotIn(tok, code, tok)

    def test_telemetry_code_is_passive(self):
        code = re.sub(r"//[^\n]*", "", self.telem)
        for tok in ("Simulator::Schedule", "RandomVariable", "SetPromisc", "RegisterProtocolHandler",
                    "GetRemainingEnergy"):
            self.assertNotIn(tok, code, tok)


if __name__ == "__main__":
    unittest.main()

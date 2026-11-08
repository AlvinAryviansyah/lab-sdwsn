import os
import sys
import math
import json
import inspect
import tempfile
import subprocess
import unittest

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ML = os.path.join(HERE, "..")
sys.path.insert(0, ML)
sys.path.insert(0, HERE)

import synth                                                       # noqa: E402
import bigan_ids                                                   # noqa: E402
from bigan_model import (BiGANConfig, Generator, Encoder, Discriminator, build_models,  # noqa: E402
                         d_loss, ge_loss_terms, discriminator_logit, anomaly_from_logit,
                         select_threshold, binary_metrics, prevalence_adjusted, score_auc,
                         make_mlp, validate_direction)
from bigan_infer import BiGANDetector                              # noqa: E402
from nslkdd_data import NSLKDDPreprocessor, Splits                 # noqa: E402

torch.set_num_threads(1)


def tiny_cfg(x_dim, **kw):
    base = dict(x_dim=x_dim, z_dim=8, hidden=32, n_hidden=1, epochs=12,
                batch_size=128, seed=3)
    base.update(kw)
    return BiGANConfig(**base)


def synth_problem(seed=0, n_train=1500, n_val=800):
    """NSL-KDD sintetis: train (semua normal untuk model), validasi campuran."""
    df_tr, y_tr = synth.make_df(n_train, seed=seed, attack_fraction=0.25)
    pre = NSLKDDPreprocessor().fit(df_tr, y_tr, scaler_fit="normal")
    df_val, y_val = synth.make_df(n_val, seed=seed + 1, attack_fraction=0.3)
    return pre, pre.transform(df_tr), y_tr, pre.transform(df_val), y_val, df_tr, df_val


class TestArchitecture(unittest.TestCase):
    def test_layout_matches_equations_32_to_34(self):
        cfg = tiny_cfg(20)
        G, E, D = build_models(cfg)
        # pers. (33)/(34): Linear -> ReLU -> Linear (keluaran linear), satu lapisan tersembunyi
        for net, i, o in ((G.net, cfg.z_dim, cfg.x_dim), (E.net, cfg.x_dim, cfg.z_dim)):
            mods = list(net)
            self.assertEqual(len(mods), 3)
            self.assertIsInstance(mods[0], torch.nn.Linear)
            self.assertIsInstance(mods[1], torch.nn.ReLU)
            self.assertIsInstance(mods[2], torch.nn.Linear)
            self.assertEqual((mods[0].in_features, mods[0].out_features), (i, cfg.hidden))
            self.assertEqual((mods[2].in_features, mods[2].out_features), (cfg.hidden, o))
        # pers. (32): D menerima concatenate(x,z); keluaran 1 logit (sigma di kerugian)
        mods = list(D.net)
        self.assertEqual(len(mods), 3)
        self.assertEqual(mods[0].in_features, cfg.x_dim + cfg.z_dim)
        self.assertEqual(mods[2].out_features, 1)

    def test_default_config_is_paper_aligned(self):
        cfg = BiGANConfig(x_dim=10)
        self.assertEqual(cfg.n_hidden, 1)
        self.assertEqual(cfg.activation, "relu")

    def test_forward_shapes(self):
        cfg = tiny_cfg(20)
        G, E, D = build_models(cfg)
        x = torch.randn(7, 20); z = torch.randn(7, 8)
        self.assertEqual(tuple(G(z).shape), (7, 20))
        self.assertEqual(tuple(E(x).shape), (7, 8))
        self.assertEqual(tuple(D(x, z).shape), (7, 1))

    def test_unsupported_activation_and_deeper_net(self):
        with self.assertRaises(ValueError):
            make_mlp(4, 8, 1, 2, activation="tanh")
        self.assertEqual(len(list(make_mlp(4, 8, 2, 2))), 5)

    def test_legacy_constructor_default_is_two_hidden(self):
        """Shim kompatibilitas export_decisions.py/checkpoint lama (didokumentasikan)."""
        self.assertEqual(len(list(Generator(8, 20).net)), 5)
        self.assertEqual(len(list(Encoder(20, 8).net)), 5)
        self.assertEqual(len(list(Discriminator(20, 8).net)), 5)


class TestLosses(unittest.TestCase):
    def test_match_paper_formulas(self):
        torch.manual_seed(0)
        r, f = torch.randn(64, 1), torch.randn(64, 1)
        sr, sf = torch.sigmoid(r), torch.sigmoid(f)
        expect_d = -(torch.log(sr).mean() + torch.log(1 - sf).mean())     # pers. (35)
        self.assertAlmostEqual(float(d_loss(r, f)), float(expect_d), places=5)
        lg, le = ge_loss_terms(r, f)
        self.assertAlmostEqual(float(lg), float(-torch.log(sf).mean()), places=5)       # pers. (36)
        self.assertAlmostEqual(float(le), float(-torch.log(1 - sr).mean()), places=5)   # pers. (37) non-saturating

    def test_numerically_stable_for_extreme_logits(self):
        r, f = torch.full((4, 1), 80.0), torch.full((4, 1), -80.0)
        for v in (d_loss(r, f), *ge_loss_terms(r, f)):
            self.assertTrue(torch.isfinite(v))


class TestTrainingSteps(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.cfg = tiny_cfg(20)
        self.G, self.E, self.D = build_models(self.cfg)
        self.opt_d = torch.optim.Adam(self.D.parameters(), lr=1e-2)
        self.opt_ge = torch.optim.Adam(list(self.G.parameters()) + list(self.E.parameters()), lr=1e-2)
        self.x, self.z = torch.randn(32, 20), torch.randn(32, 8)

    @staticmethod
    def snap(m):
        return [p.detach().clone() for p in m.parameters()]

    @staticmethod
    def same(a, b):
        return all(torch.equal(p, q) for p, q in zip(a, b))

    def test_d_step_updates_only_discriminator(self):
        g0, e0, d0 = self.snap(self.G), self.snap(self.E), self.snap(self.D)
        bigan_ids.d_step(self.G, self.E, self.D, self.opt_d, self.x, self.z)
        self.assertTrue(self.same(g0, self.snap(self.G)))
        self.assertTrue(self.same(e0, self.snap(self.E)))
        self.assertFalse(self.same(d0, self.snap(self.D)))

    def test_ge_step_updates_generator_and_encoder_but_not_discriminator(self):
        g0, e0, d0 = self.snap(self.G), self.snap(self.E), self.snap(self.D)
        bigan_ids.ge_step(self.G, self.E, self.D, self.opt_ge, self.x, self.z)
        self.assertFalse(self.same(g0, self.snap(self.G)))
        self.assertFalse(self.same(e0, self.snap(self.E)))
        self.assertTrue(self.same(d0, self.snap(self.D)))

    def test_algorithm3_schedule_k_d_steps_then_one_ge_step(self):
        calls = {"d": 0, "ge": 0, "order": []}
        d0, g0 = bigan_ids.d_step, bigan_ids.ge_step

        def cd(*a, **k):
            calls["d"] += 1; calls["order"].append("d"); return d0(*a, **k)

        def cg(*a, **k):
            calls["ge"] += 1; calls["order"].append("g"); return g0(*a, **k)

        bigan_ids.d_step, bigan_ids.ge_step = cd, cg
        try:
            cfg = tiny_cfg(20, d_steps=3, epochs=2)
            X = np.random.default_rng(0).normal(size=(300, 20)).astype(np.float32)
            bigan_ids.train(cfg, X, log_every=10 ** 9, log_fn=lambda *_: None)
        finally:
            bigan_ids.d_step, bigan_ids.ge_step = d0, g0
        n_iter = cfg.epochs * math.ceil(300 / cfg.batch_size)
        self.assertEqual(calls["ge"], n_iter)
        self.assertEqual(calls["d"], 3 * n_iter)
        self.assertEqual(calls["order"][:4], ["d", "d", "d", "g"])


class TestReproducibility(unittest.TestCase):
    def test_same_seed_same_weights_different_seed_differs(self):
        X = np.random.default_rng(0).normal(size=(400, 20)).astype(np.float32)
        quiet = dict(log_every=10 ** 9, log_fn=lambda *_: None)
        a = bigan_ids.train(tiny_cfg(20, seed=5, epochs=3), X, **quiet)
        b = bigan_ids.train(tiny_cfg(20, seed=5, epochs=3), X, **quiet)
        c = bigan_ids.train(tiny_cfg(20, seed=6, epochs=3), X, **quiet)
        for m1, m2 in zip(a[:3], b[:3]):
            for p, q in zip(m1.parameters(), m2.parameters()):
                self.assertTrue(torch.allclose(p, q, atol=1e-6))
        diff = max(float((p - q).abs().max().detach()) for p, q in zip(a[0].parameters(), c[0].parameters()))
        self.assertGreater(diff, 1e-4)


class TestScoreAndThreshold(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (cls.pre, cls.Xtr, cls.ytr, cls.Xval, cls.yval, _, _) = synth_problem(seed=0)
        cls.cfg = tiny_cfg(cls.pre.n_features, epochs=25)
        cls.G, cls.E, cls.D, _ = bigan_ids.train(
            cls.cfg, cls.Xtr[cls.ytr == 0], log_every=10 ** 9, log_fn=lambda *_: None)

    def test_score_is_negative_discriminator_logit(self):
        logit = discriminator_logit(self.E, self.D, self.Xval)
        np.testing.assert_allclose(anomaly_from_logit(logit), -logit)

    def test_one_class_model_generalises_to_heldout_normals(self):
        """Model belajar distribusi normal: skor normal held-out ~ normal data latih."""
        held, _ = synth.make_df(1500, seed=777, attack_fraction=0.0)
        s_train = discriminator_logit(self.E, self.D, self.Xtr[self.ytr == 0])
        s_held = discriminator_logit(self.E, self.D, self.pre.transform(held))
        self.assertLess(abs(s_train.mean() - s_held.mean()), 0.5 * s_train.std())

    def test_validate_direction_flags_reversed_scores(self):
        y = np.array([0] * 50 + [1] * 50)
        good = np.concatenate([np.random.default_rng(0).normal(0, 1, 50),
                               np.random.default_rng(1).normal(3, 1, 50)])
        self.assertTrue(validate_direction(good, y)["ok"])
        bad = validate_direction(-good, y)
        self.assertFalse(bad["ok"])
        self.assertLess(bad["auc"], 0.1)

    def test_threshold_rule_is_strictly_greater_than(self):
        rng = np.random.default_rng(0)
        y = np.array([0] * 900 + [1] * 100)
        score = np.concatenate([rng.normal(0, 1, 900), rng.normal(4, 1, 100)])
        thr = select_threshold(score, y, policy="f1_weighted", prevalence=0.1)
        m = binary_metrics(y, score > thr["value"])
        self.assertGreater(m["TPR"], 0.8)
        self.assertLess(m["FPR"], 0.1)
        edge = np.array([thr["value"]])                       # tepat di ambang -> bukan malicious
        self.assertFalse(bool((edge > thr["value"])[0]))

    def test_threshold_selection_api_takes_validation_only(self):
        params = list(inspect.signature(select_threshold).parameters)
        self.assertEqual(params[:2], ["score_val", "y_val"])
        self.assertFalse(any("test" in p for p in params))
        self.assertEqual(select_threshold(np.arange(10.0), np.array([0] * 8 + [1] * 2))["fit_on"],
                         "validation")

    def test_fpr_policy_hits_target_on_validation_normals(self):
        score = anomaly_from_logit(discriminator_logit(self.E, self.D, self.Xval))
        thr = select_threshold(score, self.yval, policy="fpr", target_fpr=0.05)
        fpr = float((score[self.yval == 0] > thr["value"]).mean())
        self.assertAlmostEqual(fpr, 0.05, delta=0.01)

    def test_threshold_input_validation(self):
        with self.assertRaises(ValueError):
            select_threshold(np.array([1.0, 2.0]), np.array([1, 1]))       # tanpa normal
        with self.assertRaises(ValueError):
            select_threshold(np.array([1.0, 2.0]), np.array([0, 0]))       # tanpa serangan
        with self.assertRaises(ValueError):
            select_threshold(np.array([1.0, 2.0]), np.array([0, 1]), policy="x")

    def test_metrics_helpers(self):
        m = binary_metrics([1, 1, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0])
        self.assertEqual((m["TP"], m["FN"], m["FP"], m["TN"]), (1, 1, 1, 3))
        self.assertAlmostEqual(m["precision"], 0.5)
        self.assertAlmostEqual(m["recall"], 0.5)
        self.assertAlmostEqual(m["FPR"], 0.25)
        e = prevalence_adjusted(0.5, 0.02, 0.10)
        self.assertAlmostEqual(e["precision"], 0.05 / (0.05 + 0.9 * 0.02), places=6)


class TestKnownLimitations(unittest.TestCase):
    """Mendokumentasikan batasan yang DITEMUKAN saat validasi arah skor.

    Skor Discriminator (-logit) TIDAK dijamin searah dengan anomali: pada data
    sintetis dengan anomali jauh (geser 4 sigma) AUC(-logit) ~ 0 (terbalik),
    sedangkan reconstruction error mendapat AUC ~ 1. Itulah alasan pipeline
    memeriksa arah di validasi dan menolak checkpoint dengan arah terbalik.
    """
    def test_discriminator_score_direction_is_data_dependent(self):
        df_tr, y_tr = synth.make_df(3000, seed=0, attack_fraction=0.25, shift=4.0)
        pre = NSLKDDPreprocessor().fit(df_tr, y_tr, scaler_fit="normal")
        Xtr = pre.transform(df_tr)
        df_v, y_v = synth.make_df(1500, seed=1, attack_fraction=0.3, shift=4.0)
        Xv = pre.transform(df_v)
        G, E, D, _ = bigan_ids.train(tiny_cfg(pre.n_features, epochs=12),
                                     Xtr[y_tr == 0], log_every=10 ** 9, log_fn=lambda *_: None)
        d = validate_direction(anomaly_from_logit(discriminator_logit(E, D, Xv)), y_v)
        recon_auc = score_auc(bigan_ids.anomaly_score(G, E, Xv), y_v)
        self.assertFalse(d["ok"], "arah skor D terbalik pada data ini (batasan terdokumentasi)")
        self.assertLess(d["auc"], 0.2)
        self.assertGreater(recon_auc, 0.9)


class TestCheckpointAndInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (cls.pre, cls.Xtr, cls.ytr, cls.Xval, cls.yval, cls.df_tr, cls.df_val) = synth_problem(seed=1)
        cls.cfg = tiny_cfg(cls.pre.n_features, epochs=10)
        cls.G, cls.E, cls.D, cls.hist = bigan_ids.train(
            cls.cfg, cls.Xtr[cls.ytr == 0], log_every=20, log_fn=lambda *_: None)
        score = anomaly_from_logit(discriminator_logit(cls.E, cls.D, cls.Xval))
        cls.thr = select_threshold(score, cls.yval)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = cls.tmp.name
        idx = np.arange(len(cls.Xtr))
        sp = Splits(cls.Xtr, cls.ytr, cls.Xval, cls.yval, cls.Xval, cls.yval, cls.pre,
                    idx, np.arange(len(cls.Xval)), seed=1, val_fraction=0.2)
        bigan_ids.save_checkpoint(cls.dir, cls.cfg, cls.G, cls.E, cls.D, sp, cls.thr,
                                  {"F1": 0.0}, 0.9, cls.hist, "normal")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_artifacts_written(self):
        for f in ("model.pt", "inference_config.json", "splits.npz", "train_log.json",
                  os.path.join("preprocessor", "schema.json"),
                  os.path.join("preprocessor", "scaler.npz")):
            self.assertTrue(os.path.exists(os.path.join(self.dir, f)), f)
        with open(os.path.join(self.dir, "inference_config.json")) as fh:
            cfgj = json.load(fh)
        for key in ("model", "score", "threshold", "preprocessing", "data_split", "validation"):
            self.assertIn(key, cfgj)
        self.assertEqual(cfgj["score"]["direction"], "malicious_if_score_above_threshold")
        self.assertEqual(cfgj["threshold"]["fit_on"], "validation")
        self.assertEqual(cfgj["preprocessing"]["n_features"], self.pre.n_features)
        self.assertGreaterEqual(len(self.hist), 1)

    def test_loaded_detector_reproduces_scores_and_decisions(self):
        det = BiGANDetector.load(self.dir)
        mem = anomaly_from_logit(discriminator_logit(self.E, self.D, self.Xval))
        np.testing.assert_allclose(det.score(self.Xval), mem, atol=1e-5)
        np.testing.assert_array_equal(det.predict(self.Xval), mem > self.thr["value"])
        np.testing.assert_allclose(det.p_real(self.Xval), 1 / (1 + np.exp(mem)), atol=1e-5)

    def test_inference_from_raw_dataframe_uses_saved_schema(self):
        det = BiGANDetector.load(self.dir)
        X_raw_path = det.transform_raw(self.df_val)
        np.testing.assert_allclose(X_raw_path, self.Xval, atol=1e-6)
        np.testing.assert_array_equal(det.predict(X_raw_path), det.predict(self.Xval))

    def test_feature_dimension_mismatch_rejected(self):
        det = BiGANDetector.load(self.dir)
        with self.assertRaises(ValueError):
            det.score(self.Xval[:, :-1])
        with self.assertRaises(ValueError):
            det.score(self.Xval[0])

    def test_corrupted_config_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            for name in os.listdir(self.dir):
                src = os.path.join(self.dir, name)
                if os.path.isdir(src):
                    os.makedirs(os.path.join(d, name))
                    for f in os.listdir(src):
                        open(os.path.join(d, name, f), "wb").write(open(os.path.join(src, f), "rb").read())
                else:
                    open(os.path.join(d, name), "wb").write(open(src, "rb").read())
            p = os.path.join(d, "inference_config.json")
            cfgj = json.load(open(p))
            cfgj["preprocessing"]["schema_hash"] = "deadbeef"
            json.dump(cfgj, open(p, "w"))
            with self.assertRaises(ValueError):
                BiGANDetector.load(d)

    def test_inverted_direction_checkpoint_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            idx = np.arange(len(self.Xtr))
            sp = Splits(self.Xtr, self.ytr, self.Xval, self.yval, self.Xval, self.yval, self.pre,
                        idx, np.arange(len(self.Xval)), seed=1, val_fraction=0.2)
            bigan_ids.save_checkpoint(d, self.cfg, self.G, self.E, self.D, sp, self.thr,
                                      {"F1": 0.0}, 0.2, self.hist, "normal")   # AUC validasi 0.2
            with self.assertRaises(ValueError):
                BiGANDetector.load(d)
            self.assertIsNotNone(BiGANDetector.load(d, allow_inverted=True))

    def test_inference_module_does_not_import_training_code(self):
        code = ("import sys, bigan_infer; "
                "assert 'bigan_ids' not in sys.modules, 'inferensi mengimpor pelatihan'; "
                "assert not hasattr(bigan_infer, 'train'); print('ok')")
        out = subprocess.run([sys.executable, "-c", code], cwd=ML, capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)


if __name__ == "__main__":
    unittest.main()

import os
import sys
import json
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import synth                                           # noqa: E402
from nslkdd_data import (NSLKDDPreprocessor, NUMERIC, CATEGORICAL, DATA_DIR,  # noqa: E402
                         make_splits, load, read_raw, binary_labels, TRAIN_FILE)

HAVE_DATA = os.path.exists(os.path.join(DATA_DIR, TRAIN_FILE))


class TestPreprocessor(unittest.TestCase):
    def setUp(self):
        self.df, self.y = synth.make_df(600, seed=1, attack_fraction=0.3)
        self.pre = NSLKDDPreprocessor().fit(self.df, self.y, scaler_fit="normal")

    def test_feature_order_and_dimension(self):
        names = self.pre.feature_names
        self.assertEqual(names[:len(NUMERIC)], NUMERIC)
        expected_cat = [f"{c}_{v}" for c in CATEGORICAL for v in self.pre.categories[c]]
        self.assertEqual(names[len(NUMERIC):], expected_cat)
        X = self.pre.transform(self.df)
        self.assertEqual(X.shape, (600, len(names)))
        self.assertEqual(X.dtype, np.float32)
        for c in CATEGORICAL:                       # kategori terurut alfabet
            self.assertEqual(self.pre.categories[c], sorted(self.pre.categories[c]))

    def test_scaler_uses_normal_rows_only(self):
        raw = self.pre._raw_matrix(self.df)
        np.testing.assert_allclose(self.pre.mean, raw[self.y == 0].mean(axis=0), rtol=1e-9)
        self.assertGreater(np.abs(self.pre.mean[:len(NUMERIC)]
                                  - raw[:, :len(NUMERIC)].mean(axis=0)).max(), 0.5,
                           "statistik harus berbeda dari statistik semua baris")
        Xn = self.pre.transform(self.df[self.y == 0])
        np.testing.assert_allclose(Xn[:, :len(NUMERIC)].mean(axis=0), 0.0, atol=1e-5)

    def test_scaler_fit_train_option(self):
        pre_all = NSLKDDPreprocessor().fit(self.df, self.y, scaler_fit="train")
        raw = pre_all._raw_matrix(self.df)
        np.testing.assert_allclose(pre_all.mean, raw.mean(axis=0), rtol=1e-9)
        with self.assertRaises(ValueError):
            NSLKDDPreprocessor().fit(self.df, None, scaler_fit="normal")
        with self.assertRaises(ValueError):
            NSLKDDPreprocessor().fit(self.df, self.y, scaler_fit="x")

    def test_transform_does_not_change_fit_state(self):
        """Tidak ada kebocoran: transform data lain tidak mengubah statistik fit."""
        mean0, scale0 = self.pre.mean.copy(), self.pre.scale.copy()
        other, _ = synth.make_df(300, seed=99, attack_fraction=0.9, shift=50.0)
        self.pre.transform(other)
        np.testing.assert_array_equal(self.pre.mean, mean0)
        np.testing.assert_array_equal(self.pre.scale, scale0)

    def test_unseen_category_gives_zeros_same_dimension(self):
        other, _ = synth.make_df(50, seed=5)
        other["service"] = "never_seen_service"
        X = self.pre.transform(other)
        self.assertEqual(X.shape[1], self.pre.n_features)
        raw = self.pre._raw_matrix(other)
        svc_cols = [i for i, n in enumerate(self.pre.feature_names) if n.startswith("service_")]
        self.assertTrue(np.all(raw[:, svc_cols] == 0.0))

    def test_column_order_independent(self):
        shuffled = self.df[list(reversed(self.df.columns))]
        np.testing.assert_array_equal(self.pre.transform(self.df),
                                      self.pre.transform(shuffled))

    def test_missing_column_raises(self):
        with self.assertRaises(ValueError):
            self.pre.transform(self.df.drop(columns=["src_bytes"]))

    def test_unfitted_raises(self):
        with self.assertRaises(RuntimeError):
            NSLKDDPreprocessor().transform(self.df)

    def test_save_load_roundtrip_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as d:
            self.pre.save(d)
            loaded = NSLKDDPreprocessor.load(d)
            self.assertEqual(loaded.schema_hash, self.pre.schema_hash)
            np.testing.assert_array_equal(loaded.transform(self.df), self.pre.transform(self.df))
            path = os.path.join(d, "schema.json")
            schema = json.load(open(path))
            schema["feature_names"][0] = "tampered"
            json.dump(schema, open(path, "w"))
            with self.assertRaises(ValueError):
                NSLKDDPreprocessor.load(d)


@unittest.skipUnless(HAVE_DATA, "dataset NSL-KDD tidak tersedia")
class TestRealSplits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sp = make_splits(seed=1, val_fraction=0.2, scaler_fit="normal")

    def test_disjoint_and_complete(self):
        sp = self.sp
        self.assertEqual(np.intersect1d(sp.train_idx, sp.val_idx).size, 0)
        self.assertEqual(len(sp.train_idx) + len(sp.val_idx), 125973)
        self.assertEqual(sp.X_test.shape[0], 22544)

    def test_stratified_validation(self):
        sp = self.sp
        self.assertAlmostEqual(sp.y_train.mean(), sp.y_val.mean(), places=2)

    def test_fixed_122_features(self):
        sp = self.sp
        self.assertEqual(sp.n_features, 122)
        self.assertEqual(sp.X_val.shape[1], 122)
        self.assertEqual(sp.X_test.shape[1], 122)

    def test_fit_statistics_exclude_val_and_test(self):
        sp = self.sp
        tr = read_raw(TRAIN_FILE)
        y = binary_labels(tr)
        raw_norm_train = sp.preproc._raw_matrix(tr.iloc[sp.train_idx][y[sp.train_idx] == 0])
        np.testing.assert_allclose(sp.preproc.mean, raw_norm_train.mean(axis=0), rtol=1e-9)
        raw_norm_all = sp.preproc._raw_matrix(tr[y == 0])
        self.assertFalse(np.allclose(sp.preproc.mean, raw_norm_all.mean(axis=0), rtol=1e-12),
                         "statistik tidak boleh memakai baris validasi")

    def test_deterministic_split(self):
        sp2 = make_splits(seed=1, val_fraction=0.2, scaler_fit="normal")
        np.testing.assert_array_equal(self.sp.val_idx, sp2.val_idx)
        sp3 = make_splits(seed=2, val_fraction=0.2, scaler_fit="normal")
        self.assertFalse(np.array_equal(self.sp.val_idx, sp3.val_idx))

    def test_matches_legacy_when_scaler_fit_train(self):
        Xtr_legacy, _, _, _, n = load()
        tr = read_raw(TRAIN_FILE)
        pre = NSLKDDPreprocessor().fit(tr, binary_labels(tr), scaler_fit="train")
        Xnew = pre.transform(tr)
        self.assertEqual(Xnew.shape, Xtr_legacy.shape)
        self.assertLess(float(np.abs(Xnew - Xtr_legacy).max()), 1e-3)


if __name__ == "__main__":
    unittest.main()

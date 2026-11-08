#!/usr/bin/env python3
"""
bigan_infer.py  --  INFERENSI Bi-GAN
-------------------------------------
Memuat checkpoint hasil bigan_ids.py dan menghitung skor/keputusan deteksi.
Modul ini TIDAK mengimpor kode pelatihan (bigan_ids.py) dan tidak melatih apa pun.

Aturan keputusan (tersimpan di inference_config.json):
  skor anomali = -logit D(x, E(x))      (Fig. 6 paper: Encoder -> Concatenate ->
                                          Discriminator -> Normal/Malicious)
  malicious jika skor > ambang          (ambang dipilih di data validasi)

Contoh:
  det = BiGANDetector.load("checkpoints/bigan_v2")
  Xs  = det.transform_raw(df_nslkdd)          # preprocessor tersimpan
  y   = det.predict(Xs)                       # bool, True = malicious
"""

import os
import json

import numpy as np
import torch

from bigan_model import (BiGANConfig, build_models, discriminator_logit,
                         anomaly_from_logit, SCORE_NAME, SCORE_DIRECTION)
from nslkdd_data import NSLKDDPreprocessor

SUPPORTED_FORMAT = 1


class BiGANDetector:
    def __init__(self, cfg, G, E, D, threshold, config, preproc=None, device="cpu"):
        self.cfg = cfg
        self.G, self.E, self.D = G, E, D
        self.threshold = float(threshold)
        self.config = config
        self.preproc = preproc
        self.device = device

    # -- pemuatan -------------------------------------------------------------
    @classmethod
    def load(cls, ckpt_dir, device="cpu", allow_inverted=False):
        with open(os.path.join(ckpt_dir, "inference_config.json")) as f:
            config = json.load(f)
        if config.get("format_version") != SUPPORTED_FORMAT:
            raise ValueError(f"format_version tidak didukung: {config.get('format_version')}")
        score = config["score"]
        if score["name"] != SCORE_NAME or score["direction"] != SCORE_DIRECTION:
            raise ValueError("definisi skor/arah di checkpoint tidak cocok dengan kode")
        if not config.get("validation", {}).get("direction_ok", False) and not allow_inverted:
            raise ValueError("checkpoint ini gagal verifikasi arah skor (AUC validasi <= 0.5); "
                             "tolak pemakaian (gunakan allow_inverted=True hanya utk diagnostik)")
        cfg = BiGANConfig.from_dict(config["model"])
        G, E, D = build_models(cfg)
        state = torch.load(os.path.join(ckpt_dir, "model.pt"), map_location=device)
        G.load_state_dict(state["G"]); E.load_state_dict(state["E"])
        D.load_state_dict(state["D"])
        for m in (G, E, D):
            m.to(device).eval()
        pre_dir = os.path.join(ckpt_dir, "preprocessor")
        preproc = None
        if os.path.isdir(pre_dir):
            preproc = NSLKDDPreprocessor.load(pre_dir)
            pp = config["preprocessing"]
            if preproc.schema_hash != pp["schema_hash"] or preproc.n_features != pp["n_features"]:
                raise ValueError("preprocessor tidak cocok dengan inference_config.json")
            if preproc.n_features != cfg.x_dim:
                raise ValueError("dimensi fitur preprocessor != x_dim model")
        return cls(cfg, G, E, D, config["threshold"]["value"], config, preproc, device)

    # -- inferensi ------------------------------------------------------------
    @property
    def n_features(self):
        return self.cfg.x_dim

    def _check(self, X):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(f"bentuk fitur {X.shape} != (n, {self.n_features})")
        return X

    def transform_raw(self, df):
        """DataFrame NSL-KDD mentah -> fitur terstandardisasi (skema tersimpan)."""
        if self.preproc is None:
            raise RuntimeError("checkpoint tanpa preprocessor")
        return self.preproc.transform(df)

    def logit(self, X):
        return discriminator_logit(self.E, self.D, self._check(X), self.device)

    def score(self, X):
        """Skor anomali (tinggi = malicious)."""
        return anomaly_from_logit(self.logit(X))

    def p_real(self, X):
        """sigma(logit): probabilitas D menilai pasangan sebagai nyata/normal."""
        return 1.0 / (1.0 + np.exp(-self.logit(X).astype(np.float64)))

    def predict(self, X):
        """True = malicious (skor > ambang)."""
        return self.score(X) > self.threshold

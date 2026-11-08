#!/usr/bin/env python3
"""
bigan_model.py
--------------
Definisi MODEL Bi-GAN yang dipakai bersama oleh pelatihan (bigan_ids.py) dan
inferensi (bigan_infer.py). Modul ini TIDAK berisi loop pelatihan.

Rujukan paper (Kipongo et al., IEEE Access vol. 12, 2024), Sec. III-D-2 dan
Algoritma 3. Status verifikasi tiap butir (V = terverifikasi dari teks paper,
A = ambigu/tidak konsisten di paper, U = tidak disebut paper):

  Arsitektur
    D(x)  = sigma(W_d h_d + b'_d), h_d = act(theta_d x + b_d)     pers. (32)  V
    G(z)  = W_g h_g + b'_g,        h_g = act(theta_g z + b_g)     pers. (33)  V
    E(x)  = W_e h_e + b'_e,        h_e = act(theta_e x + b_e)     pers. (34)  V
      -> satu lapisan tersembunyi + lapisan keluaran (G dan E keluaran linear,
         D keluaran sigmoid). act = ReLU "such as a ReLU" (teks paper).
      -> lebar lapisan tersembunyi, dimensi latent z: TIDAK disebut        U
  Pasangan masukan Discriminator
    Teks paper: "concatenate()" menggabungkan fitur dari encoder (sampel nyata)
    dan generator; Fig. 6 (training): data -> Encoder -> Concatenate <- Generator
    <- noise; (testing): data -> Encoder -> Concatenate -> Discriminator ->
    Normal/Malicious. Pers. (32),(35)-(37) menulis D dengan satu argumen.
    Alg. 3 baris 14 memakai concatenate(E(x), G(z)) yang tidak sejajar secara     A
    dimensi/semantik. Implementasi di sini: Bi-GAN standar (Donahue et al.;
    Xu et al., rujukan [56],[57] paper): D menerima PASANGAN (x, z); pasangan
    nyata = (x, E(x)), palsu = (G(z), z).
  Fungsi kerugian
    loss_D = -[mean log D(real) + mean log(1 - D(fake))]          pers. (35), Alg.3 b.15  V
    loss_G = -mean log D(fake)  (non-saturating)                  pers. (36), Alg.3 b.8   V
    loss_E : pers. (37) minimasi E terhadap V; Alg.3 b.11/25 menulis                      A
             -log D(x) - log(1 - D(G(E(x)))) yang bertanda kebalikan dari pers. (37).
             Di sini: E memakai bentuk non-saturating dari pers. (37) pada pasangan
             nyata: -mean log(1 - D(x, E(x))). G dan E diperbarui bersama (Alg.3 b.26-27).
  Pelatihan
    N iterasi; tiap iterasi K langkah D, lalu 1 langkah G/E dengan batch baru       V
    learning rate alpha, N, K, ukuran batch, optimizer                              U
    ("gradient descent with learning rate alpha")
  Skor deteksi dan ambang
    Fig. 6 testing: Encoder -> Concatenate -> Discriminator -> Normal/Malicious.    V
    Definisi skor numerik dan nilai ambang                                          U
    Di sini: skor anomali = -logit D(x, E(x)); malicious jika skor > ambang;
    ambang dipilih pada data VALIDASI saja (select_threshold).
"""

from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

SCORE_NAME = "neg_logit_discriminator_pair"
SCORE_DIRECTION = "malicious_if_score_above_threshold"


# ---------------------------------------------------------------------------
# Konfigurasi
# ---------------------------------------------------------------------------
@dataclass
class BiGANConfig:
    x_dim: int
    z_dim: int = 32            # U: tidak disebut paper (asumsi)
    hidden: int = 128          # U: tidak disebut paper (asumsi)
    n_hidden: int = 1          # V: pers. (32)-(34) = satu lapisan tersembunyi
    activation: str = "relu"   # V: "such as a ReLU"
    lr: float = 2e-4           # U: alpha tidak disebut (asumsi)
    beta1: float = 0.5         # U: paper menyebut "gradient descent"; Adam = asumsi
    beta2: float = 0.999
    batch_size: int = 256      # U
    epochs: int = 20           # U: N = epochs x ceil(n / batch_size)
    d_steps: int = 1           # U: K tidak disebut (asumsi)
    seed: int = 1
    latent_prior: str = "normal"   # U: p(z) tidak disebut (asumsi N(0, I))

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)


# ---------------------------------------------------------------------------
# Jaringan (pers. 32-34)
# ---------------------------------------------------------------------------
def _act(name):
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"aktivasi tidak didukung: {name} (paper: ReLU)")


def make_mlp(in_dim, hidden, n_hidden, out_dim, activation="relu"):
    """n_hidden lapisan [Linear -> act] diikuti Linear keluaran (linear)."""
    layers, d = [], in_dim
    for _ in range(n_hidden):
        layers += [nn.Linear(d, hidden), _act(activation)]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class Generator(nn.Module):
    """G(z) -> x_fake (pers. 33). Keluaran linear.

    CATATAN KOMPATIBILITAS: default n_hidden=2 dipertahankan hanya agar kode lama
    (export_decisions.py, checkpoint lama) tetap berjalan. Pelatihan baru memakai
    BiGANConfig.n_hidden=1 (sesuai pers. 32-34) lewat build_models().
    """
    def __init__(self, z_dim, x_dim, h=128, n_hidden=2, activation="relu"):
        super().__init__()
        self.net = make_mlp(z_dim, h, n_hidden, x_dim, activation)

    def forward(self, z):
        return self.net(z)


class Encoder(nn.Module):
    """E(x) -> z (pers. 34). Keluaran linear. Lihat catatan default di Generator."""
    def __init__(self, x_dim, z_dim, h=128, n_hidden=2, activation="relu"):
        super().__init__()
        self.net = make_mlp(x_dim, h, n_hidden, z_dim, activation)

    def forward(self, x):
        return self.net(x)


class Discriminator(nn.Module):
    """D(x, z) -> logit (pers. 32). sigma() diterapkan di fungsi kerugian/inferensi.

    Masukan = concatenate(x, z) (Fig. 6; lihat catatan A di docstring modul).
    """
    def __init__(self, x_dim, z_dim, h=128, n_hidden=2, activation="relu"):
        super().__init__()
        self.net = make_mlp(x_dim + z_dim, h, n_hidden, 1, activation)

    def forward(self, x, z):
        return self.net(torch.cat([x, z], dim=1))


def build_models(cfg):
    """Bangun (G, E, D) sesuai BiGANConfig (n_hidden dari cfg, default paper = 1)."""
    G = Generator(cfg.z_dim, cfg.x_dim, cfg.hidden, cfg.n_hidden, cfg.activation)
    E = Encoder(cfg.x_dim, cfg.z_dim, cfg.hidden, cfg.n_hidden, cfg.activation)
    D = Discriminator(cfg.x_dim, cfg.z_dim, cfg.hidden, cfg.n_hidden, cfg.activation)
    return G, E, D


# ---------------------------------------------------------------------------
# Fungsi kerugian (pers. 35-37, Alg. 3)
# ---------------------------------------------------------------------------
def d_loss(real_logits, fake_logits):
    """Pers. (35) / Alg.3 b.15 (tanda dibalik untuk minimisasi):
    -mean log D(real) - mean log(1 - D(fake))."""
    return (F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
            + F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits)))


def ge_loss_terms(real_logits, fake_logits):
    """(loss_G, loss_E) non-saturating.
    loss_G = -mean log D(G(z), z)            pers. (36), Alg.3 b.8/22
    loss_E = -mean log(1 - D(x, E(x)))       bentuk non-saturating pers. (37)
    G dan E diperbarui bersama dari jumlah keduanya (Alg.3 b.26-27)."""
    loss_g = F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
    loss_e = F.binary_cross_entropy_with_logits(real_logits, torch.zeros_like(real_logits))
    return loss_g, loss_e


# ---------------------------------------------------------------------------
# Skor deteksi (inferensi murni; tanpa gradient)
# ---------------------------------------------------------------------------
@torch.no_grad()
def discriminator_logit(E, D, X, device="cpu", batch=8192):
    """logit D(x, E(x)) untuk tiap baris X. Tinggi = mirip trafik normal (nyata)."""
    E.eval(); D.eval()
    out = []
    for i in range(0, len(X), batch):
        x = torch.as_tensor(X[i:i + batch], dtype=torch.float32, device=device)
        out.append(D(x, E(x)).squeeze(1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def anomaly_from_logit(logit):
    """Skor anomali = -logit. Lebih tinggi = lebih mungkin malicious."""
    return -np.asarray(logit)


# ---------------------------------------------------------------------------
# Ambang (hanya dari data validasi) dan metrik
# ---------------------------------------------------------------------------
def score_auc(score, y):
    """AUC skor anomali terhadap label serangan (1). >0.5 berarti arah benar."""
    y = np.asarray(y)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, score))


def validate_direction(score_val, y_val):
    """Periksa arah skor pada data VALIDASI: AUC(skor anomali, label serangan) > 0.5.

    Arah skor Discriminator TIDAK dijamin oleh metode: Discriminator ReLU
    mengekstrapolasi linear di luar distribusi latih, sehingga logit anomali bisa
    lebih TINGGI daripada normal (terbukti pada data sintetis; lihat
    tests/test_bigan.py::TestKnownLimitations). Karena itu arah harus diverifikasi
    untuk setiap model/dataset; skor tidak pernah dibalik secara otomatis.
    """
    auc = score_auc(score_val, y_val)
    return {"auc": auc, "ok": bool(auc > 0.5), "rule": SCORE_DIRECTION}


def select_threshold(score_val, y_val, policy="f1_weighted",
                     prevalence=0.10, target_fpr=0.05, n_grid=400):
    """Pilih ambang pada data VALIDASI (tidak menerima data test).

    policy:
      "f1_weighted": memaksimalkan F1 dengan bobot kelas sehingga prevalensi
                     serangan = `prevalence` (skenario paper: 10% node malicious).
      "fpr"        : ambang = persentil (1 - target_fpr) skor normal validasi.
    Aturan keputusan: malicious jika skor > ambang.
    """
    score_val = np.asarray(score_val, dtype=np.float64)
    y_val = np.asarray(y_val)
    norm = score_val[y_val == 0]
    if len(norm) == 0:
        raise ValueError("validasi tidak memiliki sampel normal")
    info = {"policy": policy, "prevalence": prevalence, "target_fpr": target_fpr,
            "fit_on": "validation"}
    if policy == "fpr":
        info["value"] = float(np.percentile(norm, 100.0 * (1.0 - target_fpr)))
        return info
    if policy != "f1_weighted":
        raise ValueError(f"policy tidak dikenal: {policy}")
    n_att = int((y_val == 1).sum())
    if n_att == 0:
        raise ValueError("f1_weighted membutuhkan serangan di validasi")
    w_att = (prevalence / (1 - prevalence)) * len(norm) / n_att
    best_t, best_f = None, -1.0
    for q in np.linspace(50.0, 99.9, n_grid):
        t = np.percentile(norm, q)
        pred = score_val > t
        tp = w_att * float((pred & (y_val == 1)).sum())
        fp = float((pred & (y_val == 0)).sum())
        fn = w_att * float((~pred & (y_val == 1)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        if f > best_f:
            best_f, best_t = f, t
    info["value"] = float(best_t)
    info["val_weighted_f1"] = float(best_f)
    return info


def binary_metrics(y_true, y_pred):
    """Matriks konfusi dan metrik turunan (positif = serangan/malicious)."""
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    tp = int((y_pred & y_true).sum()); tn = int((~y_pred & ~y_true).sum())
    fp = int((y_pred & ~y_true).sum()); fn = int((~y_pred & y_true).sum())
    pos, neg = tp + fn, tn + fp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / pos if pos else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "accuracy": (tp + tn) / max(pos + neg, 1),
            "precision": prec, "recall": rec, "TPR": rec,
            "FPR": fp / neg if neg else 0.0, "F1": f1}


def prevalence_adjusted(tpr, fpr, prevalence):
    """Precision/recall/F1/accuracy yang DIHARAPKAN bila prevalensi serangan = pi,
    dari TPR dan FPR terukur (TPR/FPR tidak bergantung prevalensi)."""
    pi = prevalence
    denom = pi * tpr + (1 - pi) * fpr
    prec = pi * tpr / denom if denom > 0 else 0.0
    f1 = 2 * prec * tpr / (prec + tpr) if prec + tpr > 0 else 0.0
    acc = pi * tpr + (1 - pi) * (1 - fpr)
    return {"prevalence": pi, "precision": prec, "recall": tpr, "F1": f1,
            "accuracy": acc}

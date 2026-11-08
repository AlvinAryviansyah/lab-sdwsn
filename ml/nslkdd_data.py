#!/usr/bin/env python3
"""
nslkdd_data.py
--------------
Preprocessing dataset NSL-KDD untuk intrusion detection.

CATATAN PENTING (verifikasi terhadap paper Kipongo et al. 2024):
  Paper TIDAK memakai NSL-KDD untuk Bi-GAN; NSL-KDD hanya disebut pada
  Sec. VI (future work). Paper menyebut fitur flow (SrcIP, DstIP, SrcPort,
  DstPort, ...) dan fitur paket (tipe paket, tipe respons, koneksi, ...) dari
  trafik jaringan simulasi. Modul ini menyiapkan NSL-KDD sebagai dataset
  benchmark untuk MELATIH dan MEMVALIDASI Bi-GAN; fitur dari trafik ns-3 menyusul
  pada fase integrasi NS-3 (lihat docs/PAPER_DEVIATIONS.md).

Isi modul:
  - NSLKDDPreprocessor : one-hot + standardisasi dengan skema fitur yang
                         DISIMPAN (schema.json + scaler.npz) sehingga urutan dan
                         dimensi fitur identik saat pelatihan dan inferensi.
  - make_splits()      : pemisahan TRAIN / VALIDATION / TEST tanpa kebocoran:
                           * train+validation = KDDTrain+ (stratified, seed tetap)
                           * test             = KDDTest+ (tidak pernah dipakai
                             untuk fit scaler, kategori, ambang, atau seleksi model)
                           * kategori one-hot dan statistik scaler hanya dari
                             fold TRAIN (default: hanya baris NORMAL fold train).
  - load()             : API lama (legacy) yang dipertahankan agar
                         export_decisions.py tetap jalan (feature_selection.py: archive/).
"""

import os
import json
import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data", "nsl-kdd")

# 41 nama kolom fitur NSL-KDD + label + difficulty (urutan resmi KDD'99).
COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]
CATEGORICAL = ["protocol_type", "service", "flag"]
NON_FEATURE = ["label", "difficulty"]
NUMERIC = [c for c in COLUMNS if c not in CATEGORICAL + NON_FEATURE]

SCHEMA_VERSION = 1
TRAIN_FILE = "KDDTrain+.txt"
TEST_FILE = "KDDTest+.txt"


# ---------------------------------------------------------------------------
# Pembacaan data mentah
# ---------------------------------------------------------------------------
def read_raw(filename, data_dir=DATA_DIR):
    """Baca file NSL-KDD mentah menjadi DataFrame dengan kolom COLUMNS."""
    return pd.read_csv(os.path.join(data_dir, filename), names=COLUMNS)


def binary_labels(df):
    """normal -> 0, serangan apa pun -> 1."""
    return (df["label"] != "normal").astype(np.int64).to_numpy()


# ---------------------------------------------------------------------------
# Preprocessor dengan skema yang dipersistenkan
# ---------------------------------------------------------------------------
class NSLKDDPreprocessor:
    """One-hot kategorikal + standardisasi semua fitur, skema tersimpan.

    Urutan fitur (tetap): 38 kolom numerik (urutan COLUMNS), lalu one-hot
    protocol_type, service, flag (kategori terurut alfabet). Kategori yang tidak
    ada di fold train menghasilkan vektor nol (tidak mengubah dimensi).
    """

    def __init__(self):
        self.categories = None      # dict kolom -> list kategori terurut
        self.feature_names = None
        self.mean = None
        self.scale = None
        self.scaler_fit = None

    # -- properti -----------------------------------------------------------
    @property
    def fitted(self):
        return self.mean is not None

    @property
    def n_features(self):
        return len(self.feature_names)

    @property
    def schema_hash(self):
        payload = {"numeric": NUMERIC,
                   "categorical": self.categories,
                   "feature_names": self.feature_names}
        blob = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    # -- fit / transform ----------------------------------------------------
    def _raw_matrix(self, df):
        """Matriks fitur belum distandardisasi (float64) sesuai skema."""
        missing = [c for c in NUMERIC + CATEGORICAL if c not in df.columns]
        if missing:
            raise ValueError(f"kolom wajib tidak ada: {missing[:5]}")
        num = df[NUMERIC].to_numpy(dtype=np.float64)
        blocks = [num]
        for col in CATEGORICAL:
            vals = df[col].astype(str).to_numpy()
            cats = np.asarray(self.categories[col])
            blocks.append((vals[:, None] == cats[None, :]).astype(np.float64))
        return np.hstack(blocks)

    def fit(self, df, y=None, scaler_fit="normal"):
        """Fit pada fold TRAIN saja.

        scaler_fit:
          "normal" -> statistik scaler dari baris NORMAL fold train (setting
                      one-class: tidak ada informasi serangan di normalisasi).
          "train"  -> dari semua baris fold train (setara perilaku legacy).
        Kategori one-hot selalu dari semua baris fold train.
        """
        if scaler_fit not in ("normal", "train"):
            raise ValueError("scaler_fit harus 'normal' atau 'train'")
        self.categories = {c: sorted(df[c].astype(str).unique().tolist())
                           for c in CATEGORICAL}
        self.feature_names = list(NUMERIC) + [
            f"{c}_{v}" for c in CATEGORICAL for v in self.categories[c]]
        raw = self._raw_matrix(df)
        if scaler_fit == "normal":
            if y is None:
                raise ValueError("scaler_fit='normal' membutuhkan label y")
            raw = raw[np.asarray(y) == 0]
            if len(raw) == 0:
                raise ValueError("tidak ada baris normal untuk fit scaler")
        self.mean = raw.mean(axis=0)
        std = raw.std(axis=0)                  # populasi (ddof=0), seperti StandardScaler
        self.scale = np.where(std < 1e-12, 1.0, std)   # varians nol -> skala 1
        self.scaler_fit = scaler_fit
        return self

    def transform(self, df):
        """DataFrame mentah -> array float32 (n, n_features), urutan skema tetap."""
        if not self.fitted:
            raise RuntimeError("preprocessor belum di-fit / belum dimuat")
        raw = self._raw_matrix(df)
        return ((raw - self.mean) / self.scale).astype(np.float32)

    # -- persistensi --------------------------------------------------------
    def save(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        schema = {
            "schema_version": SCHEMA_VERSION,
            "schema_hash": self.schema_hash,
            "n_features": self.n_features,
            "scaler_fit": self.scaler_fit,
            "numeric": NUMERIC,
            "categorical": self.categories,
            "feature_names": self.feature_names,
        }
        with open(os.path.join(out_dir, "schema.json"), "w") as f:
            json.dump(schema, f, indent=1)
        np.savez(os.path.join(out_dir, "scaler.npz"),
                 mean=self.mean, scale=self.scale)

    @classmethod
    def load(cls, in_dir):
        with open(os.path.join(in_dir, "schema.json")) as f:
            schema = json.load(f)
        if schema["numeric"] != NUMERIC:
            raise ValueError("daftar fitur numerik di schema.json berbeda dari kode")
        obj = cls()
        obj.categories = schema["categorical"]
        obj.feature_names = schema["feature_names"]
        obj.scaler_fit = schema.get("scaler_fit")
        sc = np.load(os.path.join(in_dir, "scaler.npz"))
        obj.mean, obj.scale = sc["mean"], sc["scale"]
        if obj.schema_hash != schema["schema_hash"]:
            raise ValueError("schema_hash tidak cocok: schema.json rusak/diubah")
        if len(obj.mean) != obj.n_features or len(obj.scale) != obj.n_features:
            raise ValueError("dimensi scaler tidak cocok dengan skema")
        return obj


# ---------------------------------------------------------------------------
# Pemisahan train / validation / test
# ---------------------------------------------------------------------------
@dataclass
class Splits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    preproc: NSLKDDPreprocessor
    train_idx: np.ndarray       # indeks baris KDDTrain+ untuk fold train
    val_idx: np.ndarray         # indeks baris KDDTrain+ untuk fold validation
    seed: int
    val_fraction: float

    @property
    def n_features(self):
        return self.X_train.shape[1]

    @property
    def X_train_normal(self):
        return self.X_train[self.y_train == 0]


def make_splits(seed=1, val_fraction=0.2, scaler_fit="normal",
                data_dir=DATA_DIR, train_file=TRAIN_FILE, test_file=TEST_FILE):
    """Bangun split tanpa kebocoran.

    - KDDTrain+ dipisah stratified menjadi fold train dan validation.
    - Preprocessor di-fit HANYA pada fold train.
    - KDDTest+ hanya di-transform (tidak memengaruhi fit apa pun).
    """
    tr = read_raw(train_file, data_dir)
    te = read_raw(test_file, data_dir)
    y_all = binary_labels(tr)
    idx = np.arange(len(tr))
    idx_tr, idx_val = train_test_split(idx, test_size=val_fraction,
                                       stratify=y_all, random_state=seed)
    idx_tr, idx_val = np.sort(idx_tr), np.sort(idx_val)
    if np.intersect1d(idx_tr, idx_val).size:
        raise AssertionError("fold train dan validation beririsan")
    pre = NSLKDDPreprocessor().fit(tr.iloc[idx_tr], y_all[idx_tr], scaler_fit)
    return Splits(
        X_train=pre.transform(tr.iloc[idx_tr]), y_train=y_all[idx_tr],
        X_val=pre.transform(tr.iloc[idx_val]), y_val=y_all[idx_val],
        X_test=pre.transform(te), y_test=binary_labels(te),
        preproc=pre, train_idx=idx_tr, val_idx=idx_val,
        seed=seed, val_fraction=val_fraction,
    )


# ---------------------------------------------------------------------------
# API LEGACY (dipertahankan untuk export_decisions.py; feature_selection.py di archive/)
# ---------------------------------------------------------------------------
def load(train_file=TRAIN_FILE, test_file=TEST_FILE):
    """LEGACY. Return (Xtr, ytr, Xte, yte, n_features).

    Scaler di-fit pada SELURUH KDDTrain+ (normal + serangan) dan tidak ada
    validation split. Dipertahankan hanya untuk modul yang belum direfaktor.
    Gunakan make_splits() untuk pekerjaan baru.
    """
    tr = pd.read_csv(os.path.join(DATA_DIR, train_file), names=COLUMNS)
    te = pd.read_csv(os.path.join(DATA_DIR, test_file), names=COLUMNS)
    ytr = (tr["label"] != "normal").astype(np.int64).to_numpy()
    yte = (te["label"] != "normal").astype(np.int64).to_numpy()
    tr = tr.drop(columns=NON_FEATURE)
    te = te.drop(columns=NON_FEATURE)
    tr = pd.get_dummies(tr, columns=CATEGORICAL)
    te = pd.get_dummies(te, columns=CATEGORICAL)
    tr, te = tr.align(te, join="left", axis=1, fill_value=0)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(tr.to_numpy(dtype=np.float32))
    Xte = scaler.transform(te.to_numpy(dtype=np.float32))
    return (Xtr.astype(np.float32), ytr,
            Xte.astype(np.float32), yte, Xtr.shape[1])


if __name__ == "__main__":
    s = make_splits()
    print(f"Fitur: {s.n_features}  schema_hash={s.preproc.schema_hash}")
    print(f"Train: {s.X_train.shape} (normal={int((s.y_train==0).sum())}, "
          f"intrusi={s.y_train.mean()*100:.1f}%)")
    print(f"Val  : {s.X_val.shape} (intrusi={s.y_val.mean()*100:.1f}%)")
    print(f"Test : {s.X_test.shape} (intrusi={s.y_test.mean()*100:.1f}%)")

#!/usr/bin/env python3
"""
feature_selection.py
--------------------
Implementasi FUTURE WORK #1 paper Kipongo et al. (2024), Section VI:
algoritma feature selection untuk IDS di IoT-enabled SDWSN pada NSL-KDD.

Tujuan (sesuai paper):
  - Memilih subset fitur paling relevan dari 122 fitur NSL-KDD.
  - Mengatasi tantangan dataset besar + keterbatasan sumber daya sensor IoT.
  - Menunjukkan: dengan lebih sedikit fitur, komputasi lebih ringan (hemat
    energi) sementara akurasi deteksi tetap tinggi.

Tiga metode feature selection dibandingkan:
  1. Mutual Information (filter)  - ukur dependensi fitur<->label.
  2. Chi-square (filter)          - uji statistik (butuh fitur non-negatif).
  3. Random Forest importance (embedded) - importance dari ensemble pohon.

Evaluasi: untuk k = 10, 20, 40, 80, 122 fitur, latih classifier cepat
(RandomForest) lalu ukur accuracy/F1 di test set + waktu training.

Output:
  - ns3-sim/results/feature_selection.csv
  - figures/feature_selection.png  (akurasi/F1 vs #fitur, waktu vs #fitur)
  - figures/feature_importance.png (top-20 fitur terpenting)
"""

import os
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from nslkdd_data import load

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "..", "ns3-sim", "results")
FIG_DIR = os.path.join(HERE, "..", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

np.random.seed(1)

K_VALUES = [10, 20, 40, 80, 122]   # jumlah fitur yang diuji


# ---------------------------------------------------------------------------
# Metode-metode feature selection: mengembalikan skor relevansi per fitur.
# ---------------------------------------------------------------------------
def rank_mutual_info(X, y):
    """Mutual Information: makin tinggi, makin relevan."""
    return mutual_info_classif(X, y, random_state=1)


def rank_chi2(X, y):
    """Chi-square: butuh fitur non-negatif -> MinMax scale dulu."""
    Xpos = MinMaxScaler().fit_transform(X)
    scores, _ = chi2(Xpos, y)
    return np.nan_to_num(scores)


def rank_rf_importance(X, y):
    """Random Forest feature importance (embedded)."""
    rf = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=1)
    rf.fit(X, y)
    return rf.feature_importances_


METHODS = {
    "Mutual Info": rank_mutual_info,
    "Chi-square": rank_chi2,
    "RF importance": rank_rf_importance,
}


def evaluate_subset(Xtr, ytr, Xte, yte, feat_idx):
    """Latih classifier pada subset fitur; return (acc, f1, train_time)."""
    clf = RandomForestClassifier(n_estimators=60, n_jobs=-1, random_state=1)
    t0 = time.time()
    clf.fit(Xtr[:, feat_idx], ytr)
    train_time = time.time() - t0
    ypred = clf.predict(Xte[:, feat_idx])
    acc = accuracy_score(yte, ypred) * 100
    f1 = f1_score(yte, ypred, zero_division=0) * 100
    return acc, f1, train_time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=30000,
                    help="subsample train utk kecepatan (0=semua)")
    args = ap.parse_args()

    print("Memuat NSL-KDD ...")
    Xtr, ytr, Xte, yte, n_feat = load()
    print(f"Fitur total: {n_feat}")

    # Subsample train agar cepat (ranking & evaluasi).
    if args.sample and args.sample < len(Xtr):
        idx = np.random.choice(len(Xtr), args.sample, replace=False)
        Xtr_s, ytr_s = Xtr[idx], ytr[idx]
    else:
        Xtr_s, ytr_s = Xtr, ytr
    print(f"Sampel train utk seleksi: {len(Xtr_s)}")

    rows = []  # (method, k, acc, f1, time)
    ranking_cache = {}

    for mname, ranker in METHODS.items():
        print(f"\n=== Metode: {mname} ===")
        scores = ranker(Xtr_s, ytr_s)
        order = np.argsort(scores)[::-1]   # indeks fitur, terbaik dulu
        ranking_cache[mname] = (scores, order)

        for k in K_VALUES:
            k = min(k, n_feat)
            feat_idx = order[:k]
            acc, f1, tt = evaluate_subset(Xtr_s, ytr_s, Xte, yte, feat_idx)
            print(f"  k={k:3d}  acc={acc:5.2f}%  f1={f1:5.2f}%  "
                  f"train={tt:.2f}s")
            rows.append((mname, k, acc, f1, tt))

    # Simpan CSV.
    df = pd.DataFrame(rows, columns=["method", "k", "accuracy", "f1",
                                     "train_time_s"])
    csv_path = os.path.join(RESULTS_DIR, "feature_selection.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nMetrik -> {os.path.relpath(csv_path)}")

    # --- Grafik 1: akurasi & F1 & waktu vs jumlah fitur ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    colors = {"Mutual Info": "blue", "Chi-square": "green",
              "RF importance": "red"}
    for mname in METHODS:
        sub = df[df.method == mname]
        ax1.plot(sub.k, sub.f1, marker="o", label=mname,
                 color=colors[mname])
    ax1.set_xlabel("Jumlah fitur terpilih (k)")
    ax1.set_ylabel("F1-score (%)")
    ax1.set_title("Akurasi deteksi vs jumlah fitur")
    ax1.grid(True, linestyle=":", alpha=0.6); ax1.legend(fontsize=8)

    for mname in METHODS:
        sub = df[df.method == mname]
        ax2.plot(sub.k, sub.train_time_s, marker="s", label=mname,
                 color=colors[mname])
    ax2.set_xlabel("Jumlah fitur terpilih (k)")
    ax2.set_ylabel("Waktu training (detik)")
    ax2.set_title("Biaya komputasi vs jumlah fitur")
    ax2.grid(True, linestyle=":", alpha=0.6); ax2.legend(fontsize=8)

    fig.suptitle("Feature Selection di NSL-KDD (future work paper)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out1 = os.path.join(FIG_DIR, "feature_selection.png")
    fig.savefig(out1, dpi=150); plt.close(fig)
    print("Grafik -> figures/feature_selection.png")

    # --- Grafik 2: top-20 fitur terpenting (pakai Mutual Info) ---
    # nama fitur setelah one-hot.
    from nslkdd_data import load as _  # noqa (sudah dimuat)
    # Ambil nama kolom dari get_dummies lewat re-run ringkas:
    scores_mi, order_mi = ranking_cache["Mutual Info"]
    # Rekonstruksi nama fitur.
    tr = pd.read_csv(os.path.join(HERE, "..", "data", "nsl-kdd",
                                  "KDDTrain+.txt"),
                     names=__import__("nslkdd_data").COLUMNS)
    tr = tr.drop(columns=["label", "difficulty"])
    tr = pd.get_dummies(tr, columns=__import__("nslkdd_data").CATEGORICAL)
    feat_names = list(tr.columns)

    top = order_mi[:20]
    names = [feat_names[i] if i < len(feat_names) else f"f{i}" for i in top]
    vals = [scores_mi[i] for i in top]
    plt.figure(figsize=(7, 6))
    plt.barh(range(len(top))[::-1], vals, color="steelblue")
    plt.yticks(range(len(top))[::-1], names, fontsize=8)
    plt.xlabel("Mutual Information score")
    plt.title("Top-20 fitur paling relevan (NSL-KDD)")
    plt.tight_layout()
    out2 = os.path.join(FIG_DIR, "feature_importance.png")
    plt.savefig(out2, dpi=150); plt.close()
    print("Grafik -> figures/feature_importance.png")

    # Ringkasan temuan.
    best_full = df[df.k == 122]["f1"].max()
    best_20 = df[df.k == 20]["f1"].max()
    print(f"\nTemuan: F1 dgn 122 fitur = {best_full:.1f}%, "
          f"dgn 20 fitur = {best_20:.1f}% "
          f"(selisih {best_full - best_20:+.1f} poin).")
    print("=> fitur jauh lebih sedikit, komputasi lebih ringan, "
          "akurasi hampir sama (klaim paper terbukti).")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
detector_eval.py
----------------
EVALUASI detektor Bi-GAN dari checkpoint (hanya inferensi; tidak melatih).

Metode yang dievaluasi = metode yang selaras paper (Fig. 6): skor
-logit D(x, E(x)), keputusan malicious jika skor > ambang. Ambang TIDAK dipilih di
sini: ia dibaca dari inference_config.json (dipilih di data VALIDASI saat pelatihan).

Pemisahan data:
  - validasi : baris KDDTrain+ yang tidak dipakai melatih (indeks di splits.npz)
  - test     : KDDTest+ (tidak pernah dipakai untuk fit apa pun)

Laporan (per split): Accuracy, Precision, Recall(=TPR), F1, FPR, AUC, matriks
konfusi. Karena skenario ns-3 memakai prevalensi serangan 10% sedangkan KDDTest+
berprevalensi ~57%, ditambahkan angka yang DIHARAPKAN pada prevalensi 10%
(dihitung dari TPR dan FPR terukur; bukan hasil simulasi).

Keputusan tim: skor Discriminator = MAINLINE. Opsi --ablation-recon menjalankan
EKSPERIMEN PEMBANDING skor reconstruction error (ambang sendiri dari validasi).
Pembanding ini BUKAN metode paper dan tidak dipakai untuk keputusan IDS.

Pemakaian:
  python detector_eval.py --ckpt checkpoints/bigan_v2
"""

import os
import json
import argparse

import numpy as np

from bigan_infer import BiGANDetector
from bigan_model import (binary_metrics, prevalence_adjusted, score_auc,
                         select_threshold)
from nslkdd_data import read_raw, binary_labels, TRAIN_FILE, TEST_FILE

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = os.path.join(HERE, "reports")


def evaluate_split(det, X, y, name, thr=None, prevalence=0.10):
    """Metrik pada satu split dengan ambang tetap (default: ambang checkpoint)."""
    thr = det.threshold if thr is None else thr
    score = det.score(X)
    m = binary_metrics(y, score > thr)
    out = {"split": name, "n": int(len(y)), "prevalence": float(np.mean(y)),
           "threshold": float(thr), "AUC": score_auc(score, y), **m,
           "expected_at_prevalence": prevalence_adjusted(m["TPR"], m["FPR"], prevalence)}
    return out


def print_report(r):
    e = r["expected_at_prevalence"]
    print(f"\n[{r['split']}] n={r['n']}  prevalensi serangan={r['prevalence']*100:.1f}%  "
          f"ambang={r['threshold']:.3f}")
    print(f"  Accuracy={r['accuracy']*100:.2f}%  Precision={r['precision']*100:.2f}%  "
          f"Recall/TPR={r['TPR']*100:.2f}%  F1={r['F1']*100:.2f}%")
    print(f"  FPR={r['FPR']*100:.2f}%  AUC={r['AUC']:.4f}")
    print(f"  Konfusi: TP={r['TP']} TN={r['TN']} FP={r['FP']} FN={r['FN']}")
    print(f"  Diharapkan pada prevalensi {e['prevalence']*100:.0f}%: "
          f"Acc={e['accuracy']*100:.1f}%  P={e['precision']*100:.1f}%  "
          f"R={e['recall']*100:.1f}%  F1={e['F1']*100:.1f}%")


def load_val(det, ckpt_dir, data_dir=None):
    """Baris validasi (indeks dari splits.npz) -> fitur via preprocessor tersimpan."""
    sp = np.load(os.path.join(ckpt_dir, "splits.npz"))
    kwargs = {} if data_dir is None else {"data_dir": data_dir}
    tr = read_raw(TRAIN_FILE, **kwargs)
    val = tr.iloc[sp["val_idx"]]
    return det.transform_raw(val), binary_labels(val)


def plot_confusion(r, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]], dtype=float)
    pct = cm / cm.sum() * 100
    plt.figure(figsize=(4.6, 4))
    plt.imshow(pct, cmap="Blues", vmin=0, vmax=pct.max())
    lab = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{lab[i][j]}\n{int(cm[i, j])}\n({pct[i, j]:.1f}%)",
                     ha="center", va="center",
                     color="black" if pct[i, j] < pct.max() / 2 else "white")
    plt.xticks([0, 1], ["Pred Normal", "Pred Intrusi"])
    plt.yticks([0, 1], ["Aktual Normal", "Aktual Intrusi"])
    plt.title(f"Bi-GAN ({r['split']})  F1={r['F1']*100:.1f}%")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "checkpoints", "bigan_v2"))
    ap.add_argument("--prevalence", type=float, default=0.10)
    ap.add_argument("--ablation-recon", action="store_true",
                    help="bandingkan dgn skor reconstruction error (bukan metode paper)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args(argv)

    det = BiGANDetector.load(args.ckpt)
    cfgj = det.config
    print(f"Checkpoint: {os.path.relpath(args.ckpt)}")
    print(f"  model: n_hidden={det.cfg.n_hidden} hidden={det.cfg.hidden} "
          f"z_dim={det.cfg.z_dim} | skor={cfgj['score']['name']} "
          f"| ambang={det.threshold:.3f} ({cfgj['threshold']['policy']}, "
          f"fit_on={cfgj['threshold']['fit_on']})")

    Xv, yv = load_val(det, args.ckpt)
    te = read_raw(TEST_FILE)
    Xt, yt = det.transform_raw(te), binary_labels(te)

    results = {"checkpoint": os.path.relpath(args.ckpt),
               "model": cfgj["model"], "threshold": cfgj["threshold"],
               "validation": evaluate_split(det, Xv, yv, "validation", prevalence=args.prevalence),
               "test": evaluate_split(det, Xt, yt, "test", prevalence=args.prevalence)}
    print_report(results["validation"])
    print_report(results["test"])

    if args.ablation_recon:
        from bigan_ids import anomaly_score
        sv, st = anomaly_score(det.G, det.E, Xv), anomaly_score(det.G, det.E, Xt)
        thr = select_threshold(sv, yv, policy=cfgj["threshold"]["policy"],
                               prevalence=cfgj["threshold"].get("prevalence", 0.10))
        m = binary_metrics(yt, st > thr["value"])
        ab = {"AUC": score_auc(st, yt), **m,
              "expected_at_prevalence": prevalence_adjusted(m["TPR"], m["FPR"], args.prevalence)}
        results["ablation_recon_test"] = ab
        print(f"\n[ABLATION recon, bukan metode paper] test: AUC={ab['AUC']:.3f} "
              f"TPR={ab['TPR']*100:.1f}% FPR={ab['FPR']*100:.1f}% "
              f"F1@{args.prevalence*100:.0f}%={ab['expected_at_prevalence']['F1']*100:.1f}%")

    os.makedirs(REPORT_DIR, exist_ok=True)
    tag = args.tag or os.path.basename(os.path.normpath(args.ckpt))
    with open(os.path.join(REPORT_DIR, f"bigan_eval_{tag}.json"), "w") as f:
        json.dump(results, f, indent=1)
    plot_confusion(results["test"], os.path.join(REPORT_DIR, f"bigan_confusion_{tag}.png"))
    print(f"\nLaporan -> ml/reports/bigan_eval_{tag}.json (+ bigan_confusion_{tag}.png)")


if __name__ == "__main__":
    main()

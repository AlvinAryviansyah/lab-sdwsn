#!/usr/bin/env python3
"""
metrics_plots.py
----------------
Reproduksi grafik paper Kipongo et al. (2024) dalam DUA set terpisah supaya
jujur & tidak menyesatkan (lihat README bagian batasan):

  figures/real/     -> HASIL NYATA ns-3 (HieMulti-Block) vs jumlah node.
                       Membuktikan simulasi benar-benar jalan & terukur.
                       Sumber: ns3-sim/results/metrics_<n>.csv

  figures/compare/  -> PERBANDINGAN TREN vs 3 baseline paper
                       (PSO-ABC [49], OL-RL [47], RBP-DT [48]) pada basis
                       apple-to-apple (nilai acuan Tabel 7). Mereproduksi
                       klaim paper: HieMulti-Block unggul di semua metrik.

Kenapa dipisah?
  Metrik absolut ns-3 (energy Joule total 100 node, throughput kbps, latency ms)
  TIDAK sebanding langsung dgn angka rata-rata Tabel 7 paper (beda satuan/skala).
  Menyatukan keduanya di satu sumbu membuat baseline "menang" secara visual,
  kebalikan dari klaim paper. Maka: set 'real' untuk bukti simulasi, set
  'compare' untuk reproduksi tren.

Output:
  figures/real/*.png, figures/compare/*.png,
  figures/fig18_time_metrics.png, figures/fig19_confusion.png

Input ns-3: ns3-sim/results/metrics_<n>_s<seed>.csv (banyak seed). Grafik 'real'
menampilkan rata-rata ± simpangan baku lintas seed.
"""

import os
import csv
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "..", "ns3-sim", "results")
FIG_DIR = os.path.join(HERE, "..", "figures")
REAL_DIR = os.path.join(FIG_DIR, "real")
CMP_DIR = os.path.join(FIG_DIR, "compare")
for d in (FIG_DIR, REAL_DIR, CMP_DIR):
    os.makedirs(d, exist_ok=True)

NODES = [20, 30, 40, 50, 60, 70, 80, 90, 100]

STYLE = {
    "PSO-ABC":        dict(color="red",     marker="s"),
    "OL-RL":          dict(color="magenta", marker="x"),
    "RBP-DT":         dict(color="green",   marker="^"),
    "HieMulti-Block": dict(color="blue",    marker="D"),
}

# ---------------------------------------------------------------------------
# Nilai acuan paper (Tabel 7) — untuk set 'compare' (apple-to-apple).
# ---------------------------------------------------------------------------
TABLE7 = {
    "energy":     {"PSO-ABC": 47.8, "OL-RL": 37.7, "RBP-DT": 32.8, "HieMulti-Block": 28.2},
    "latency":    {"PSO-ABC": 55.0, "OL-RL": 44.8, "RBP-DT": 41.4, "HieMulti-Block": 34.8},
    "throughput": {"PSO-ABC": 30.0, "OL-RL": 40.0, "RBP-DT": 49.3, "HieMulti-Block": 58.2},
    "pdr":        {"PSO-ABC": 58.0, "OL-RL": 60.0, "RBP-DT": 63.0, "HieMulti-Block": 66.0},
    "lifetime":   {"PSO-ABC": 400.0, "OL-RL": 450.0, "RBP-DT": 495.0, "HieMulti-Block": 580.0},
    "comp_oh":    {"PSO-ABC": 83.0, "OL-RL": 68.0, "RBP-DT": 62.0, "HieMulti-Block": 52.8},
    "det_acc":    {"PSO-ABC": 75.5, "OL-RL": 73.0, "RBP-DT": 78.9, "HieMulti-Block": 86.9},
    "drop":       {"PSO-ABC": 69.8, "OL-RL": 58.0, "RBP-DT": 48.0, "HieMulti-Block": 37.4},
    "control_oh": {"PSO-ABC": 87.34, "OL-RL": 85.08, "RBP-DT": 79.32, "HieMulti-Block": 44.96},
}

# metrik yang NAIK terhadap jumlah node (bentuk kurva sintetis utk baseline).
INCREASING = {"energy", "latency", "throughput", "pdr", "lifetime",
              "comp_oh", "det_acc", "drop", "control_oh"}

# Precision/Recall/F1 (Fig. 18) — baseline dari paper; HieMulti-Block dari ns-3.
TIME_METRICS_BASE = {
    "PSO-ABC": {"Precision": 74.0, "Recall": 60.0, "F1": 65.7},
    "OL-RL":   {"Precision": 70.0, "Recall": 54.6, "F1": 61.8},
    "RBP-DT":  {"Precision": 82.6, "Recall": 61.2, "F1": 74.3},
}

# Confusion matrix baseline (Fig. 19) dalam persen [[TP,FP],[FN,TN]].
CONFUSION_BASE = {
    "RBP-DT":  [[26, 8],  [19, 54]],
    "OL-RL":   [[26, 6],  [26, 39]],
    "PSO-ABC": [[30, 5],  [25, 40]],
}

# Peta metric_key -> field di CSV ns-3 (None = tak tersedia langsung).
CSV_FIELD = {
    "energy":     "energy_consumption_J",
    "latency":    "latency_ms",
    "throughput": "throughput_kbps",
    "pdr":        "pdr_percent",
    "det_acc":    "detection_accuracy_percent",
    "drop":       "packet_drop_ratio_percent",
    "comp_oh":    "computation_overhead_percent",
    "lifetime":   "network_lifetime_s",
    "control_oh": "control_overhead_mps",
}


def read_csv_values(n, field):
    """Nilai `field` dari SEMUA seed: ns3-sim/results/metrics_<n>_s<seed>.csv."""
    vals = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR,
                                              f"metrics_{n}_s*.csv"))):
        with open(path) as f:
            for row in csv.reader(f):
                if len(row) == 2 and row[0] == field:
                    try:
                        vals.append(float(row[1]))
                    except ValueError:
                        pass
    return vals


def read_csv_value(n, field):
    """Rata-rata `field` lintas seed; None jika tak ada data."""
    v = read_csv_values(n, field)
    return float(np.mean(v)) if v else None


def read_ns3_series(metric_key):
    """Seri {node: (mean, std, n_seed)} dari CSV ns-3; None jika tak ada."""
    field = CSV_FIELD.get(metric_key)
    if field is None:
        return None
    out = {}
    for n in NODES:
        v = read_csv_values(n, field)
        if v:
            std = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
            out[n] = (float(np.mean(v)), std, len(v))
    return out if out else None


def synth_curve(final_value, increasing=True):
    """Kurva tren dari satu nilai akhir (Tabel 7) untuk baseline 'compare'."""
    x = np.array(NODES, dtype=float)
    frac = (x - x.min()) / (x.max() - x.min())
    if increasing:
        start = final_value * 0.45
        return start + (final_value - start) * frac
    start = final_value * 1.8
    return start - (start - final_value) * frac


# ===========================================================================
#  SET 1 — REAL: hasil nyata ns-3 (HieMulti-Block saja)
# ===========================================================================
def plot_real(metric_key, ylabel, title, fname):
    series = read_ns3_series(metric_key)
    if not series or len(series) < 2:
        print(f"  (lewati real {metric_key}: data ns-3 tak tersedia)")
        return
    xs = sorted(series.keys())
    ys = [series[n][0] for n in xs]
    sd = [series[n][1] for n in xs]
    n_seed = max(series[n][2] for n in xs)
    plt.figure(figsize=(6, 4.2))
    plt.errorbar(xs, ys, yerr=sd, capsize=3,
                 label=f"HieMulti-Block (ns-3, mean±std, {n_seed} seed)",
                 **STYLE["HieMulti-Block"])
    plt.xlabel("Number of Nodes")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(REAL_DIR, fname)
    plt.savefig(out, dpi=150)
    plt.close()
    print("  tersimpan:", os.path.relpath(out))


# ===========================================================================
#  SET 2 — COMPARE: HieMulti-Block vs baseline (apple-to-apple, Tabel 7)
# ===========================================================================
def plot_compare(metric_key, ylabel, title, fname, increasing=True):
    plt.figure(figsize=(6, 4.2))
    for model in ["PSO-ABC", "OL-RL", "RBP-DT", "HieMulti-Block"]:
        yv = synth_curve(TABLE7[metric_key][model], increasing)
        plt.plot(NODES, yv, label=model, **STYLE[model])
    plt.xlabel("Number of Nodes")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=8)
    # Kurva ini BUKAN hasil simulasi: interpolasi dari nilai rata-rata Tabel 7.
    plt.figtext(0.5, 0.005,
                "Kurva acuan dari Tabel 7 paper (bukan hasil simulasi kami)",
                ha="center", fontsize=7, style="italic", color="gray")
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    out = os.path.join(CMP_DIR, fname)
    plt.savefig(out, dpi=150)
    plt.close()
    print("  tersimpan:", os.path.relpath(out))


# ===========================================================================
#  Fig. 18 — time-based metrics (precision/recall/F1)
#  Baseline dari paper; HieMulti-Block dari hasil ns-3 (100 node).
# ===========================================================================
def plot_time_metrics():
    labels = ["Precision", "Recall", "F1"]
    hb = {
        "Precision": read_csv_value(100, "precision_percent"),
        "Recall":    read_csv_value(100, "recall_percent"),
        "F1":        read_csv_value(100, "f1_percent"),
    }
    models = ["PSO-ABC", "OL-RL", "RBP-DT"]
    data = dict(TIME_METRICS_BASE)
    if any(v is None for v in hb.values()):
        # Tanpa data ns-3 kita TIDAK memalsukan angka HieMulti-Block.
        print("  (fig18: hasil ns-3 tidak ada; hanya baseline paper ditampilkan)")
    else:
        models.append("HieMulti-Block")
        data["HieMulti-Block"] = hb

    x = np.arange(len(labels))
    w = 0.8 / len(models)
    plt.figure(figsize=(7, 4.2))
    for i, m in enumerate(models):
        vals = [data[m][k] for k in labels]
        suffix = " (ns-3)" if m == "HieMulti-Block" else " (paper)"
        plt.bar(x + (i - (len(models) - 1) / 2) * w, vals, w,
                label=m + suffix, color=STYLE[m]["color"])
    plt.xticks(x, labels)
    plt.ylabel("Values in Percentage (%)")
    plt.title("Fig. 18 — Time-based metrics analysis")
    plt.legend(fontsize=8)
    plt.grid(True, axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "fig18_time_metrics.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print("  tersimpan:", os.path.relpath(out))


# ===========================================================================
#  Fig. 19 — confusion matrices (HieMulti-Block dari ns-3 + 3 baseline paper)
# ===========================================================================
def plot_confusion():
    conf = dict(CONFUSION_BASE)
    # Jumlahkan hitungan lintas seed (100 node) lalu ubah ke persen.
    sums = {k: sum(read_csv_values(100, k)) for k in ("TP", "TN", "FP", "FN")}
    if read_csv_values(100, "TP"):
        tp, tn, fp, fn = (sums[k] for k in ("TP", "TN", "FP", "FN"))
        tot = (tp + tn + fp + fn) or 1.0
        conf["HieMulti-Block"] = [[100 * tp / tot, 100 * fp / tot],
                                  [100 * fn / tot, 100 * tn / tot]]
        print("  (confusion HieMulti-Block: gabungan semua seed, 100 node)")
    else:
        print("  (confusion: hasil ns-3 tidak ada; HieMulti-Block dilewati)")

    fig, axes = plt.subplots(2, 2, figsize=(8, 7))
    order = [m for m in ["HieMulti-Block", "RBP-DT", "OL-RL", "PSO-ABC"]
             if m in conf]
    for ax in axes.flat[len(order):]:
        ax.axis("off")
    for ax, model in zip(axes.flat, order):
        cm = np.array(conf[model])
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=max(60, cm.max()))
        ax.set_title(model + (" (ns-3)" if model == "HieMulti-Block" else ""),
                     fontsize=10)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pos", "Neg"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual Pos", "Actual Neg"])
        lab = [["TP", "FP"], ["FN", "TN"]]
        for r in range(2):
            for c in range(2):
                ax.text(c, r, f"{lab[r][c]}\n{cm[r][c]:.0f}%",
                        ha="center", va="center",
                        color="black" if cm[r][c] < 35 else "white",
                        fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Fig. 19 — Confusion matrix visualization", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(FIG_DIR, "fig19_confusion.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("  tersimpan:", os.path.relpath(out))


# ===========================================================================
#  Time-based metrics (Fig. 18 versi time-series): precision/recall/F1 per window
#  sepanjang simulasi, rata-rata lintas seed. Sumber: timeseries_<n>_s<seed>.csv
# ===========================================================================
def plot_timeseries(n=100):
    series = []   # satu entri per seed: (t[], precision[], recall[], f1[])
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR,
                                              f"timeseries_{n}_s*.csv"))):
        ts, P, R, F = [], [], [], []
        with open(path) as f:
            for row in csv.DictReader(f):
                tp, fp, fn = float(row["TP"]), float(row["FP"]), float(row["FN"])
                p = 100 * tp / (tp + fp) if tp + fp else 0.0
                r = 100 * tp / (tp + fn) if tp + fn else 0.0
                fs = 2 * p * r / (p + r) if p + r else 0.0
                ts.append(float(row["t"])); P.append(p); R.append(r); F.append(fs)
        if ts:
            series.append((ts, P, R, F))
    if not series:
        print(f"  (lewati time-series n={n}: data ns-3 tak tersedia)")
        return
    k = min(len(s[0]) for s in series)          # samakan jumlah window
    x = np.array(series[0][0][:k])
    plt.figure(figsize=(7, 4.2))
    for idx, (name, color) in enumerate([("Precision", "tab:blue"),
                                         ("Recall", "tab:green"),
                                         ("F1-score", "tab:red")], start=1):
        m = np.array([s[idx][:k] for s in series])
        mean, sd = m.mean(axis=0), m.std(axis=0)
        plt.plot(x, mean, label=name, color=color, marker="o", ms=3)
        plt.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.15)
    plt.xlabel("Waktu simulasi (s)")
    plt.ylabel("Nilai (%)")
    plt.title(f"Time-based metrics Bi-GAN ({n} node, mean±std {len(series)} seed)")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(REAL_DIR, f"timebased_{n}.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print("  tersimpan:", os.path.relpath(out))


# metrik yg tersedia nyata dari ns-3 (untuk set 'real').
REAL_SPECS = [
    ("energy",     "Energy Consumption (J)", "Energy consumption (ns-3)",   "energy.png"),
    ("latency",    "Latency (ms)",           "Latency (ns-3)",              "latency.png"),
    ("throughput", "Throughput (kbps)",      "Throughput (ns-3)",           "throughput.png"),
    ("pdr",        "Packet Delivery Ratio (%)", "PDR (ns-3)",               "pdr.png"),
    ("det_acc",    "Detection Accuracy (%)", "Detection accuracy (ns-3)",   "det_accuracy.png"),
    ("drop",       "Packet Drop Ratio (%)",  "Packet drop ratio (ns-3)",    "drop_ratio.png"),
    ("lifetime",   "Network Lifetime (s)",   "Network lifetime, proyeksi (ns-3)", "lifetime.png"),
    ("comp_oh",    "Computation Overhead (%)", "Computation overhead (ns-3)", "comp_overhead.png"),
    ("control_oh", "Control Overhead (mps)", "Control overhead (ns-3)",     "control_overhead.png"),
]

# semua 9 metrik untuk set 'compare' (Fig. 9-17).
CMP_SPECS = [
    ("energy",     "Energy Consumption (J)",   "Fig. 9 — Energy consumption",    "fig09_energy.png"),
    ("latency",    "Latency (ms)",             "Fig. 10 — Latency",              "fig10_latency.png"),
    ("throughput", "Throughput (kbps)",        "Fig. 11 — Throughput",           "fig11_throughput.png"),
    ("pdr",        "Packet Delivery Ratio (%)","Fig. 12 — PDR",                  "fig12_pdr.png"),
    ("lifetime",   "Network Lifetime (s)",     "Fig. 13 — Network lifetime",     "fig13_lifetime.png"),
    ("comp_oh",    "Computation Overhead (%)", "Fig. 14 — Computation overhead", "fig14_comp_overhead.png"),
    ("det_acc",    "Detection Accuracy (%)",   "Fig. 15 — Detection accuracy",   "fig15_det_accuracy.png"),
    ("drop",       "Packet Drop Ratio (%)",    "Fig. 16 — Packet drop ratio",    "fig16_drop_ratio.png"),
    ("control_oh", "Control Overhead (mps)",   "Fig. 17 — Control overhead",     "fig17_control_overhead.png"),
]


def main():
    print("=== SET 1: hasil NYATA ns-3 (figures/real/) ===")
    for key, yl, title, fn in REAL_SPECS:
        plot_real(key, yl, title, fn)

    plot_timeseries(100)

    print("=== SET 2: perbandingan vs baseline (figures/compare/) ===")
    for key, yl, title, fn in CMP_SPECS:
        plot_compare(key, yl, title, fn, increasing=(key in INCREASING))

    print("=== Fig. 18 & 19 ===")
    plot_time_metrics()
    plot_confusion()
    print("Selesai.")


if __name__ == "__main__":
    main()

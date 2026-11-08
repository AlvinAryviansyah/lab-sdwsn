#!/usr/bin/env python3
"""
analyze_routing.py
------------------
Analisis BERPASANGAN hasil sweep NS-3 integrasi langkah 1: tabel next-hop TLDQN
vs tabel referensi greedy (progres terbesar) pada skenario yang sama (n, seed).

Masukan : ns3-sim/results/metrics_<n>_s<seed>.csv dan metrics_<n>_s<seed>_greedy.csv
Keluaran: ml/reports/routing_ns3_step1.json, figures/real/routing_tldqn_vs_greedy.png

Uji: uji tanda dua sisi pada selisih per skenario (tanpa asumsi distribusi);
skenario seri dikeluarkan dari uji.
"""

import os
import csv
import json
import argparse
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "ns3-sim", "results")
NODES = [20, 30, 40, 50, 60, 70, 80, 90, 100]
METRICS = [  # (kunci, label, arah lebih baik: +1 naik, -1 turun, 0 tanpa arah)
    ("pdr_percent", "PDR (%)", +1),
    ("packet_drop_ratio_percent", "Packet drop ratio (%)", -1),
    ("throughput_kbps", "Throughput (kbps)", +1),
    ("latency_ms", "Latency (ms)", -1),
    ("grayhole_drops", "Paket dijatuhkan grayhole", -1),
    ("energy_consumption_J", "Energi (J)", -1),
    ("benign_hops_per_rx_est", "Hop per paket (sensor normal)", 0),
]
IDS_KEYS = ("detection_accuracy_percent", "precision_percent", "recall_percent", "f1_percent",
            "TP", "TN", "FP", "FN")


def load(n, s, tag=""):
    path = os.path.join(RES, f"metrics_{n}_s{s}{tag}.csv")
    d = {}
    with open(path) as f:
        for r in csv.reader(f):
            if len(r) == 2:
                try:
                    d[r[0]] = float(r[1])
                except ValueError:
                    pass
    if "grayhole_drops" not in d:
        raise ValueError(f"{path} bukan keluaran langkah 1 (kolom grayhole_drops tidak ada)")
    return d


def sign_test(diff, better):
    win = int(sum(1 for d in diff if d * better > 0))
    lose = int(sum(1 for d in diff if d * better < 0))
    tie = len(diff) - win - lose
    m = win + lose
    p = min(1.0, 2 * sum(comb(m, i) for i in range(min(win, lose) + 1)) / 2 ** m) if m else 1.0
    return {"better": win, "tie": tie, "worse": lose, "p_sign_test": p}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args(argv)
    S = args.seeds
    T = {(n, s): load(n, s) for n in NODES for s in S}
    G = {(n, s): load(n, s, "_greedy") for n in NODES for s in S}

    out = {"seeds": S, "nodes": NODES, "n_scenarios": len(T), "metrics": {}, "per_n": {}}
    for k, label, better in METRICS:
        diff = [T[x][k] - G[x][k] for x in T]
        m = {"label": label, "mean_tldqn": float(np.mean([T[x][k] for x in T])),
             "mean_greedy": float(np.mean([G[x][k] for x in T])),
             "mean_diff": float(np.mean(diff)), "sd_diff": float(np.std(diff, ddof=1))}
        if better:
            m.update(sign_test(diff, better))
        out["metrics"][k] = m
    for n in NODES:
        out["per_n"][n] = {k: {"tldqn_mean": float(np.mean([T[(n, s)][k] for s in S])),
                               "tldqn_std": float(np.std([T[(n, s)][k] for s in S], ddof=1)),
                               "greedy_mean": float(np.mean([G[(n, s)][k] for s in S])),
                               "greedy_std": float(np.std([G[(n, s)][k] for s in S], ddof=1))}
                           for k, _, _ in METRICS}
    out["checks"] = {
        "ids_metrics_identical_across_routing": all(T[x][k] == G[x][k] for x in T for k in IDS_KEYS),
        "grayhole_drop_ratio_pooled": float(
            sum(d[x]["grayhole_drops"] for d in (T, G) for x in T)
            / sum(d[x]["grayhole_drops"] + d[x]["grayhole_not_dropped"] for d in (T, G) for x in T)),
        "grayhole_drop_prob_config": T[(NODES[0], S[0])]["grayhole_drop_prob"],
        "dto": {name: {"TP": float(sum(d[x]["dto_TP"] for x in T)),
                       "FP": float(sum(d[x]["dto_FP"] for x in T)),
                       "attackers": float(sum(d[x]["dto_TP"] + d[x]["dto_FN"] for x in T))}
                for name, d in (("tldqn", T), ("greedy", G))},
    }
    os.makedirs(os.path.join(HERE, "reports"), exist_ok=True)
    with open(os.path.join(HERE, "reports", "routing_ns3_step1.json"), "w") as f:
        json.dump(out, f, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = ["pdr_percent", "grayhole_drops", "throughput_kbps", "latency_ms"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    for ax, k in zip(axes, show):
        for name, d, c, mk in (("TLDQN", "tldqn", "tab:blue", "D"), ("greedy (referensi)", "greedy", "tab:orange", "s")):
            mu = [out["per_n"][n][k][f"{d}_mean"] for n in NODES]
            sd = [out["per_n"][n][k][f"{d}_std"] for n in NODES]
            ax.errorbar(NODES, mu, yerr=sd, capsize=3, color=c, marker=mk, label=name)
        mm = out["metrics"][k]
        ax.set_title(f"{mm['label']}\nTLDQN lebih baik {mm['better']}/{mm['better'] + mm['worse']} "
                     f"(uji tanda p={mm['p_sign_test']:.2g})", fontsize=9)
        ax.set_xlabel("Jumlah node")
        ax.grid(True, ls=":", alpha=0.6)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"NS-3 integrasi langkah 1: routing TLDQN vs greedy (mean±std, {len(S)} seed; "
                 "trust saat ekspor masih tersimulasi)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    os.makedirs(os.path.join(HERE, "..", "figures", "real"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "..", "figures", "real", "routing_tldqn_vs_greedy.png"), dpi=150)
    plt.close(fig)

    for k, m in out["metrics"].items():
        tail = (f" | lebih baik {m['better']}, seri {m['tie']}, lebih buruk {m['worse']}, "
                f"p={m['p_sign_test']:.2g}") if "better" in m else ""
        print(f"{m['label']:32s} TLDQN {m['mean_tldqn']:9.3f}  greedy {m['mean_greedy']:9.3f}  "
              f"selisih {m['mean_diff']:+9.3f}{tail}")
    print("cek:", json.dumps(out["checks"]))
    print("-> ml/reports/routing_ns3_step1.json, figures/real/routing_tldqn_vs_greedy.png")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
analyze_stage2_runs.py  --  dampak trust telemetry pada jaringan NS-3 (langkah 2)
--------------------------------------------------------------------------------
Perbandingan BERPASANGAN per skenario (n, seed), uji tanda dua sisi (analyze_routing):
  stage2 (trust telemetry)  vs  stage1 (trust 0.5, tanpa informasi)      <- utama
  stage2                    vs  langkah-1 TLDQN (trust tersimulasi DARI LABEL;
                                batas atas informasi, bukan metode yang sah)
  stage2                    vs  langkah-1 greedy (referensi tanpa trust)
Label tidak dipakai di sini; semua metrik dari metrics_*.csv NS-3.

Masukan : ns3-sim/results/step2/metrics_<n>_s<seed>_{stage1,stage2}.csv,
          ns3-sim/results/metrics_<n>_s<seed>{,_greedy}.csv (langkah 1)
Keluaran: ml/reports/telemetry_step2_network.json, figures/real/step2_stage1_vs_stage2.png
"""

import json
import os
import sys

import numpy as np

from analyze_routing import METRICS, NODES, sign_test
from telemetry_schema import load_metrics

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "ns3-sim", "results")
S2 = os.path.join(RES, "step2")
SEEDS = [1, 2, 3]
EXTRA = [("dto_TP", "DTO TP", +1), ("dto_FP", "DTO FP", -1)]


def load_set(d, tag):
    return {(n, s): load_metrics(os.path.join(d, f"metrics_{n}_s{s}{tag}.csv"))
            for n in NODES for s in SEEDS}


def compare(A, B, name_a, name_b):
    out = {}
    for k, label, better in METRICS + EXTRA:
        diff = [A[x][k] - B[x][k] for x in A]
        m = {"label": label, f"mean_{name_a}": float(np.mean([A[x][k] for x in A])),
             f"mean_{name_b}": float(np.mean([B[x][k] for x in A])),
             "mean_diff": float(np.mean(diff)), "sd_diff": float(np.std(diff, ddof=1))}
        if better:
            m.update(sign_test(diff, better))
        out[k] = m
    return out


def main(argv=None):
    st1, st2 = load_set(S2, "_stage1"), load_set(S2, "_stage2")
    l1, gr = load_set(RES, ""), load_set(RES, "_greedy")
    rep = {"n_scenarios": len(st2), "seeds": SEEDS, "nodes": NODES,
           "stage2_vs_stage1": compare(st2, st1, "stage2", "stage1"),
           "stage2_vs_step1_label_trust": compare(st2, l1, "stage2", "step1"),
           "stage2_vs_step1_greedy": compare(st2, gr, "stage2", "greedy"),
           "per_n": {n: {k: {t: float(np.mean([d[(n, s)][k] for s in SEEDS]))
                             for t, d in (("stage1", st1), ("stage2", st2), ("step1", l1),
                                          ("greedy", gr))}
                         for k, _, _ in METRICS} for n in NODES},
           "note": "step1 = trust tersimulasi dari label (referensi batas atas, bukan metode)"}
    os.makedirs(os.path.join(HERE, "reports"), exist_ok=True)
    with open(os.path.join(HERE, "reports", "telemetry_step2_network.json"), "w") as f:
        json.dump(rep, f, indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = ["pdr_percent", "grayhole_drops", "throughput_kbps", "latency_ms"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    style = (("stage1", "tahap-1 (trust 0.5)", "tab:gray", "o"),
             ("stage2", "tahap-2 (trust telemetry)", "tab:blue", "D"),
             ("step1", "langkah-1 (trust dari label)", "tab:green", "^"),
             ("greedy", "greedy", "tab:orange", "s"))
    for ax, k in zip(axes, show):
        for t, lab, c, mk in style:
            ax.plot(NODES, [rep["per_n"][n][k][t] for n in NODES], color=c, marker=mk, label=lab,
                    ls="--" if t in ("step1", "greedy") else "-")
        m = rep["stage2_vs_stage1"][k]
        ax.set_title(f"{m['label']}\ntahap-2 lebih baik {m['better']}/{m['better'] + m['worse']}"
                     f" vs tahap-1 (p={m['p_sign_test']:.2g})", fontsize=9)
        ax.set_xlabel("Jumlah node")
        ax.grid(True, ls=":", alpha=0.6)
    axes[0].legend(fontsize=7)
    fig.suptitle("NS-3 langkah 2: dampak trust telemetry (rerata 3 seed, berpasangan)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    os.makedirs(os.path.join(HERE, "..", "figures", "real"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "..", "figures", "real", "step2_stage1_vs_stage2.png"), dpi=150)
    plt.close(fig)

    for title, key, a, b in (("tahap-2 vs tahap-1", "stage2_vs_stage1", "stage2", "stage1"),
                             ("tahap-2 vs langkah-1 (trust label)", "stage2_vs_step1_label_trust",
                              "stage2", "step1"),
                             ("tahap-2 vs greedy", "stage2_vs_step1_greedy", "stage2", "greedy")):
        print(f"== {title}")
        for k, m in rep[key].items():
            tail = (f" | lebih baik {m['better']}, seri {m['tie']}, lebih buruk {m['worse']}, "
                    f"p={m['p_sign_test']:.2g}") if "better" in m else ""
            print(f"  {m['label']:30s} {m[f'mean_{a}']:9.3f} vs {m[f'mean_{b}']:9.3f}  "
                  f"selisih {m['mean_diff']:+8.3f}{tail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

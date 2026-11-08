#!/usr/bin/env python3
"""
dto_proxy_validation.py  --  validasi pengukuran kandidat bukti DTO (langkah 3, EVALUASI)
--------------------------------------------------------------------------------------
Skrip EKSPLORATIF/validasi. Menghasilkan tabel docs/STEP3_DTO_DESIGN.md bagian 4.1 dan 10.2.
BUKAN bagian detektor: memakai label (AUC) dan kolom self_* (sebagai alat ukur kebenaran
proksi). Detektor (dto_detect.py) tidak membaca keluaran skrip ini.

  4.1   AUC per skenario (malicious vs normal) untuk PLR watchdog, aktivitas TX mentah,
        dan A_j = aktivitas - penerusan terdengar        (run: stage1)
  10.2  Spearman per skenario: A_j vs paket asli node (self_ip_orig_pkts),
        rx_max vs MPDU dipancarkan (self_phy_tx_mpdu), keduanya vs energi radio
        (self_energy_j)                                   (run: stage1, stage2, ctrl0)
Keluaran: ml/reports/dto_step3_proxy_validation.json
"""

import json
import os
import sys

import numpy as np

import telemetry_eval as E
import telemetry_schema as S
from telemetry_trust import corrected_trust

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "ns3-sim", "results")
S2 = os.path.join(RES, "step2")
ALL = [20, 30, 40, 50, 60, 70, 80, 90, 100]


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def proxies(links, n):
    rx, F = np.zeros(n), np.zeros(n)
    acc = {}
    for r in links:
        if r["neighbor_type"] != "sensor":
            continue
        k = (r["observer"], r["neighbor_id"])
        acc[k] = acc.get(k, 0) + r["rx_ok_from"]
        F[r["neighbor_id"]] += r["wd_forwarded"] + r["wd_forwarded_late"]
    for (i, j), v in acc.items():
        rx[j] = max(rx[j], v)
    return rx, rx - F


def stat(v):
    return {"median": float(np.median(v)), "min": float(np.min(v)), "n_scenarios": len(v)}


def main():
    rep = {"note": "evaluasi saja: memakai label dan self_* sebagai alat ukur", "auc_4_1": {},
           "spearman_10_2": {}}
    auc = {"plr": [], "tx_raw": [], "A_j": []}
    cov = {"malicious": 0, "with_plr": 0, "with_tx": 0}
    for n in ALL:
        for s in (1, 2, 3):
            links = S.load_links(S.paths(os.path.join(S2, "telemetry_stage1"), n, s)["links"])
            mal = E.load_labels(RES, n, s)
            ct = corrected_trust(links, n)
            N = ct["n_watched"]
            L = np.where(N > 0, ct["drops_est"] / np.maximum(N, 1), np.nan)
            rx, A = proxies(links, n)
            o = ~np.isnan(L)
            if mal[o].any() and (~mal[o]).any():
                auc["plr"].append(E.auc(L[o], mal[o]))
            auc["tx_raw"].append(E.auc(rx, mal))
            auc["A_j"].append(E.auc(A, mal))
            cov["malicious"] += int(mal.sum())
            cov["with_plr"] += int((mal & o).sum())
            cov["with_tx"] += int((mal & (rx > 0)).sum())
    rep["auc_4_1"] = {k: stat(v) for k, v in auc.items()}
    rep["auc_4_1"]["coverage_stage1"] = cov
    for tag, nodes in (("stage1", ALL), ("stage2", ALL), ("ctrl0", [20, 60, 100])):
        r = {"A_vs_orig_pkts": [], "rx_vs_tx_mpdu": [], "A_vs_energy": [], "rx_vs_energy": []}
        for n in nodes:
            for s in (1, 2, 3):
                p = S.paths(os.path.join(S2, f"telemetry_{tag}"), n, s)
                tot = S.node_totals(S.load_nodes(p["node"]), n)
                rx, A = proxies(S.load_links(p["links"]), n)
                r["A_vs_orig_pkts"].append(spearman(A, tot["self_ip_orig_pkts"]))
                r["rx_vs_tx_mpdu"].append(spearman(rx, tot["self_phy_tx_mpdu"]))
                r["A_vs_energy"].append(spearman(A, tot["self_energy_j"]))
                r["rx_vs_energy"].append(spearman(rx, tot["self_energy_j"]))
        rep["spearman_10_2"][tag] = {k: stat(v) for k, v in r.items()}
    os.makedirs(os.path.join(HERE, "reports"), exist_ok=True)
    out = os.path.join(HERE, "reports", "dto_step3_proxy_validation.json")
    with open(out, "w") as f:
        json.dump(rep, f, indent=1)
    for k, v in rep["auc_4_1"].items():
        print("AUC", k, v)
    for t, v in rep["spearman_10_2"].items():
        print("Spearman", t, {k: round(x["median"], 3) for k, x in v.items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())

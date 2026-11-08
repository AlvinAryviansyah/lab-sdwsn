#!/usr/bin/env python3
"""
telemetry_trust.py  --  trust dari telemetry NS-3 (integrasi langkah 2, tahap offline)
------------------------------------------------------------------------------------
Alur dua tahap (belum routing online):
  1. NS-3 tahap-1 dijalankan dengan tabel rute yang TIDAK memakai trust berlabel
     (export_routes.py --trust-source none) dan menulis telemetry.
  2. Skrip ini menghitung trust per sensor HANYA dari kolom watchdog (lewat
     telemetry_schema.trust_view), lalu export_routes.py --trust-source telemetry
     membuat tabel untuk NS-3 tahap-2.

Per watch (observer i -> tetangga j): F = forwarded + forwarded_late (terdengar),
A = unobserved_rxfail (tidak terdengar, observer sempat gagal terima), U =
unobserved_clean, N = F + A + U.

Metode `corrected` (default). Kategori A tidak netral: paket yang benar-benar di-drop
masuk A bila KEBETULAN ada kegagalan terima di window, sedangkan paket yang diteruskan
masuk A hanya bila frame itu sendiri gagal. Membuang A membuat trust relay malicious
naik seiring beban kanal. Koreksinya:
  b_i  = P(kegagalan di window | watch terdengar) di observer i
         = sum wd_heard_rxfail / sum (wd_forwarded + wd_forwarded_late)
         (observer tanpa watch terdengar memakai b gabungan)
  D_j  = min(A_j + U_j, sum_i U_ij / (1 - b_i))   estimasi drop sebenarnya
         (drop -> U dgn peluang 1 - b_i, bila waktu drop independen dari kegagalan;
          batas A + U diterapkan pada agregat per node, bukan per sel)
  trust_j = (N_j - D_j + 1) / (N_j + 2)        Beta(1,1); tanpa observasi = 0.5
Metode `discard` (pembanding): trust_j = (F_j + 1) / (F_j + U_j + 2), A dibuang.
Kolom self_* dan watchdog_audit TIDAK dibaca.

Keluaran: <out-dir>/trust_telemetry_<n>_s<seed>.csv
  node_id,trust,alpha,beta,n_watched,n_forwarded,n_ambiguous,n_unobserved_clean,drops_est
          + trust_telemetry_meta.json (metode, sumber, SHA-256, rentang window)
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np

from telemetry_schema import (TRUST_INPUT_COLUMNS, TRUST_INPUT_LINK_COLUMNS, load_links,
                              load_nodes, paths, trust_view)

TRUST_COLUMNS = ["node_id", "trust", "alpha", "beta", "n_watched", "n_forwarded",
                 "n_ambiguous", "n_unobserved_clean", "drops_est"]
METHODS = ("corrected", "discard")


def _empty(n):
    return {k: np.zeros(n) for k in ("n_watched", "n_forwarded", "n_ambiguous",
                                      "n_unobserved_clean", "drops_est")}


def _finish(c):
    alpha = c["n_watched"] - c["drops_est"] + 1.0
    beta = c["drops_est"] + 1.0
    c.update(alpha=alpha, beta=beta, trust=alpha / (alpha + beta))
    return c


def coincidence_rates(links, windows=None):
    """b_i per observer dan b gabungan, dari watch yang terdengar."""
    heard, hrx = {}, {}
    for r in trust_view(links, TRUST_INPUT_LINK_COLUMNS):
        if r["neighbor_type"] != "sensor" or (windows is not None and r["window"] not in windows):
            continue
        i = r["observer"]
        heard[i] = heard.get(i, 0) + r["wd_forwarded"] + r["wd_forwarded_late"]
        hrx[i] = hrx.get(i, 0) + r["wd_heard_rxfail"]
    tot_h, tot_x = sum(heard.values()), sum(hrx.values())
    pooled = tot_x / tot_h if tot_h else 0.0
    b = {i: (hrx[i] / heard[i] if heard[i] else pooled) for i in heard}
    return b, pooled


def corrected_trust(links, n, windows=None):
    b, pooled = coincidence_rates(links, windows)
    c = _empty(n)
    for r in trust_view(links, TRUST_INPUT_LINK_COLUMNS):
        if r["neighbor_type"] != "sensor" or (windows is not None and r["window"] not in windows):
            continue
        j, bi = r["neighbor_id"], b.get(r["observer"], pooled)
        F = r["wd_forwarded"] + r["wd_forwarded_late"]
        A, U = r["wd_unobserved_rxfail"], r["wd_unobserved_clean"]
        c["n_watched"][j] += F + A + U
        c["n_forwarded"][j] += F
        c["n_ambiguous"][j] += A
        c["n_unobserved_clean"][j] += U
        c["drops_est"][j] += (A + U) if bi >= 1.0 else U / (1.0 - bi)
    # Batas fisik D <= A + U diterapkan pada AGREGAT per node: memotong per sel kecil
    # (i, j, window) membuat estimator bias turun.
    cap = c["n_ambiguous"] + c["n_unobserved_clean"]
    c["cap_hit"] = (c["drops_est"] > cap) & (c["n_watched"] > 0)   # estimasi jenuh
    c["drops_est"] = np.minimum(c["drops_est"], cap)
    c = _finish(c)
    c["b_pooled"] = pooled
    return c


def beta_trust(nodes, n, windows=None):
    """Metode discard dari telemetry_<n>_s<seed>.csv: A dibuang."""
    c = _empty(n)
    for r in trust_view(nodes, ["node_id", "window"] + TRUST_INPUT_COLUMNS):
        if windows is not None and r["window"] not in windows:
            continue
        j = r["node_id"]
        F = r["obs_forwarded"] + r["obs_forwarded_late"]
        A, U = r["obs_unobserved_rxfail"], r["obs_unobserved_clean"]
        c["n_watched"][j] += F + U                     # A tidak ikut dihitung
        c["n_forwarded"][j] += F
        c["n_ambiguous"][j] += A
        c["n_unobserved_clean"][j] += U
        c["drops_est"][j] += U
    return _finish(c)


def compute(telem_dir, n, seed, method="corrected", windows=None):
    p = paths(telem_dir, n, seed)
    if method == "corrected":
        return corrected_trust(load_links(p["links"]), n, windows), p["links"]
    if method == "discard":
        return beta_trust(load_nodes(p["node"]), n, windows), p["node"]
    raise ValueError(method)


def write_trust(path, t):
    n = len(t["trust"])
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(TRUST_COLUMNS)
        for i in range(n):
            w.writerow([i] + [f"{t[c][i]:.6f}" for c in TRUST_COLUMNS[1:]])


def load_trust(path, n):
    trust = np.full(n, np.nan)
    with open(path) as f:
        r = csv.DictReader(f)
        if r.fieldnames != TRUST_COLUMNS:
            raise ValueError(f"{path}: header trust tidak dikenal: {r.fieldnames}")
        for row in r:
            trust[int(row["node_id"])] = float(row["trust"])
    if np.isnan(trust).any():
        raise ValueError(f"{path}: trust tidak lengkap untuk 0..{n - 1}")
    return trust


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Trust Beta dari watchdog NS-3 (offline)")
    ap.add_argument("--telem-dir", required=True, help="folder telemetry NS-3 tahap-1")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--nodes", type=int, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--method", choices=METHODS, default="corrected")
    ap.add_argument("--windows", type=int, nargs="*", default=None,
                    help="indeks window yang dipakai (default semua)")
    a = ap.parse_args(argv)
    os.makedirs(a.out_dir, exist_ok=True)
    wins = set(a.windows) if a.windows else None
    scen = []
    for n in a.nodes:
        for s in a.seeds:
            t, src = compute(a.telem_dir, n, s, a.method, wins)
            out = os.path.join(a.out_dir, f"trust_telemetry_{n}_s{s}.csv")
            write_trust(out, t)
            scen.append({"n": n, "seed": s, "source_file": os.path.basename(src),
                         "source_sha256": sha256(src), "trust_sha256": sha256(out),
                         "nodes_observed": int((t["n_watched"] > 0).sum()),
                         "watches": int(t["n_watched"].sum()),
                         "ambiguous_total": int(t["n_ambiguous"].sum()),
                         "b_pooled": t.get("b_pooled"),
                         "cap_hit_nodes": (sorted(np.flatnonzero(t["cap_hit"]).tolist())
                                           if "cap_hit" in t else None)})
            print(f"n={n:3d} s={s}: {scen[-1]['nodes_observed']}/{n} sensor teramati, "
                  f"watch={scen[-1]['watches']} ambigu={scen[-1]['ambiguous_total']}"
                  + (f" b={t['b_pooled']:.3f}" if "b_pooled" in t else ""))
    meta = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "method": a.method,
            "formula": {"corrected": "D=min(A+U, U/(1-b_obs)); trust=(N-D+1)/(N+2)",
                        "discard": "trust=(F+1)/(F+U+2); A dibuang"}[a.method],
            "input_columns": (TRUST_INPUT_LINK_COLUMNS if a.method == "corrected"
                              else TRUST_INPUT_COLUMNS),
            "windows": sorted(wins) if wins else "all", "telem_dir": a.telem_dir,
            "scenarios": scen}
    with open(os.path.join(a.out_dir, "trust_telemetry_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

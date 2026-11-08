#!/usr/bin/env python3
"""
dto_detect.py  --  deteksi node malicious dengan DTO di atas telemetry NS-3 (langkah 3)
-------------------------------------------------------------------------------------
Offline (K8). Hanya membaca telemetry_links lewat whitelist DTO_INPUT_LINK_COLUMNS: tanpa
self_*, tanpa label, tanpa watchdog_audit. Label hanya dipakai di dto_eval.py.

Keputusan yang disetujui (docs/STEP3_DTO_DESIGN.md bagian 9):
  K1  satu kandidat solusi = satu node
  K2  satu indikator delivery reliability: PLR_j = D_j / N_j (PDR = 1 - PLR, tidak dihitung lagi)
  K3  tanpa self_*; kriteria energi paper = NOT AVAILABLE (tidak ada proksi tervalidasi:
      frame-terdengar berkorelasi 0.98 dgn MPDU terkirim tetapi ~0.1 dgn energi radio).
      Aktivitas TX hanya sebagai ablasi berlabel "bukan energi".
  K4  aturan berhenti statistik tanpa label
  K5  m = 30, Y = 100, N_min = 10 (konfigurasi reproduksi)
  K6  Rand (22) ~ U(0,1)
  K9  observer dgn laju kegagalan terima kebetulan b_i >= 0.8 tidak memberi bukti N_j
      (bukti tidak informatif); target tanpa bukti cukup -> insufficient_evidence

Alur per skenario:
  1. bukti: D_j, N_j (estimator drop terkoreksi langkah 2, tanpa observer K9)
  2. kandidat C = sensor dgn N_j >= N_min, diacak (permutasi seeded)
  3. ulangi: DTO meminimalkan OF(z) = -S(C[floor(z-1)]) -> node j*;
            terima bila lolos uji (accept); keluarkan dari C; berhenti bila gagal
  4. keputusan: malicious / normal (punya bukti) / insufficient_evidence

Varian:
  main         S = PLR; uji loss; berhenti pada kegagalan pertama
  skip         seperti main, tetapi node yang gagal dikeluarkan dan pencarian berlanjut
  tx_ablation  S = 0.5 PLR~ + 0.5 TX~ (TX = log(1 + frame dari j yang terdengar));
               uji loss ATAU robust-z TX > 3.5. TX = aktivitas transmisi, BUKAN energi.
  paper_k      isolasi tepat k = ceil(0.1 n) node teratas tanpa uji (klaim paper "10%";
               k = jumlah penyerang skenario -> hanya pembanding, bukan metode tanpa prior)
"""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
from scipy import stats

from dto import DTO, DTOConfig
from telemetry_schema import DTO_INPUT_LINK_COLUMNS, load_links, paths, trust_view
from telemetry_trust import coincidence_rates, corrected_trust

VARIANTS = ("main", "skip", "tx_ablation", "paper_k")
B_MAX_OBSERVER = 0.8      # K9 (disetujui reviewer): observer dgn b_i >= 0.8 tidak memberi bukti
DECISION_COLUMNS = ["node_id", "decision", "isolation_order", "n_watched", "drops_est", "plr",
                    "tx_frames", "score", "posterior_exceed", "tx_robust_z", "accepted_by",
                    "n_watched_excluded"]


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _corrected_excluding(view, n, windows, b, pooled, excluded):
    """Estimator drop terkoreksi langkah 2 (telemetry_trust.corrected_trust), dengan
    watch dari observer di `excluded` tidak dihitung. b dan pooled diambil dari SEMUA
    observer (sama dengan langkah 2), sehingga b observer yang tersisa tidak berubah.
    Dengan `excluded` kosong hasilnya identik dengan corrected_trust (diuji)."""
    N, A, U, D, Nx = (np.zeros(n) for _ in range(5))
    for r in view:
        if r["neighbor_type"] != "sensor" or (windows is not None and r["window"] not in windows):
            continue
        i, j = r["observer"], r["neighbor_id"]
        F = r["wd_forwarded"] + r["wd_forwarded_late"]
        a, u = r["wd_unobserved_rxfail"], r["wd_unobserved_clean"]
        if i in excluded:
            Nx[j] += F + a + u
            continue
        bi = b.get(i, pooled)
        N[j] += F + a + u
        A[j] += a
        U[j] += u
        D[j] += (a + u) if bi >= 1.0 else u / (1.0 - bi)
    D = np.minimum(D, A + U)          # batas fisik pada agregat per node (sama dgn langkah 2)
    return N, D, Nx


def evidence(links, n, windows=None, b_max=B_MAX_OBSERVER):
    """Bukti per sensor dari observasi tetangga saja.

    K9: observer i dengan laju kegagalan terima kebetulan b_i >= b_max tidak memberi
    bukti untuk N_j / D_j. b_i = sum wd_heard_rxfail / sum (wd_forwarded + wd_forwarded_late)
    diukur HANYA pada watch yang terdengar (penerusan yang benar-benar terjadi) dan hanya
    dari kegagalan terima PHY observer i sendiri; jadi b_i adalah kegagalan observasi
    milik observer, bukan packet loss jaringan (drop tidak pernah masuk pembilang/penyebut).
    Observer tanpa watch terdengar tidak punya b_i terukur (memakai b gabungan langkah 2)
    dan tidak dikecualikan. b_max=None -> perilaku pra-K9 (corrected_trust apa adanya).
    Trust dan routing langkah 2 tidak berubah: fungsi ini hanya dipakai DTO."""
    view = trust_view(links, DTO_INPUT_LINK_COLUMNS)
    b, pooled = coincidence_rates(view, windows)
    # observer dgn b terukur = punya >= 1 watch terdengar (selain itu b = b gabungan)
    heard = {}
    for r in view:
        if r["neighbor_type"] == "sensor" and (windows is None or r["window"] in windows):
            heard[r["observer"]] = (heard.get(r["observer"], 0)
                                    + r["wd_forwarded"] + r["wd_forwarded_late"])
    excluded = ({} if b_max is None else
                {i: v for i, v in b.items() if heard.get(i, 0) > 0 and v >= b_max})
    if b_max is None:
        ct = corrected_trust(view, n, windows)
        N, D, Nx = ct["n_watched"], ct["drops_est"], np.zeros(n)
    else:
        N, D, Nx = _corrected_excluding(view, n, windows, b, pooled, excluded)
    # frame dari j yang di-decode benar oleh observer i (kumulatif atas window); maks antar
    # observer = batas bawah jumlah frame yang dipancarkan j (aktivitas TX, bukan energi)
    tx = np.zeros(n)
    acc = {}
    for r in view:
        if r["neighbor_type"] != "sensor" or (windows is not None and r["window"] not in windows):
            continue
        key = (r["observer"], r["neighbor_id"])
        acc[key] = acc.get(key, 0) + r["rx_ok_from"]
    for (i, j), v in acc.items():
        tx[j] = max(tx[j], v)
    plr = np.where(N > 0, D / np.maximum(N, 1e-12), np.nan)
    return {"N": N, "D": D, "plr": plr, "tx": tx, "b_pooled": pooled, "N_excluded": Nx,
            "excluded_observers": {int(i): float(v) for i, v in sorted(excluded.items())},
            "b_max": b_max}


def robust_z(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        return np.zeros_like(x)
    return 0.6745 * (x - med) / mad


def minmax(x):
    lo, hi = np.nanmin(x), np.nanmax(x)
    return np.zeros_like(x) if hi - lo < 1e-12 else (x - lo) / (hi - lo)


def detect(ev, n, variant="main", n_min=10, delta=0.1, alpha=0.01, z_thr=3.5,
           dto_cfg=None, seed=0):
    """Mengembalikan baris keputusan per node + log langkah DTO. Tanpa label."""
    if variant not in VARIANTS:
        raise ValueError(variant)
    dto_cfg = dto_cfg or DTOConfig()
    N, D, plr, tx = ev["N"], ev["D"], ev["plr"], ev["tx"]
    Nx = ev.get("N_excluded", np.zeros(n))
    has_loss = N >= n_min
    logtx = np.log1p(tx)
    use_tx = variant == "tx_ablation"
    cand = np.flatnonzero(has_loss | (use_tx & (tx > 0)))

    # baseline jaringan (asumsi mayoritas jujur) dan uji per node
    q0 = float(np.median(plr[has_loss])) if has_loss.any() else 0.0
    M = max(1, int(has_loss.sum()))
    post = np.full(n, np.nan)
    for j in np.flatnonzero(has_loss):
        a, b = D[j] + 1.0, N[j] - D[j] + 1.0          # Beta(1,1) prior
        post[j] = stats.beta.sf(min(1.0, q0 + delta), a, b)   # P(q_j > q0 + delta)
    loss_ok = post >= 1.0 - alpha / M
    tx_on = tx > 0
    tz = np.zeros(n)
    if tx_on.any():
        tz[tx_on] = robust_z(logtx[tx_on])
    tx_ok = use_tx & tx_on & (tz > z_thr)

    # skor kecurigaan S (tanpa label)
    if use_tx:
        Lt = np.where(has_loss, plr, np.nan)
        Lt = np.where(np.isnan(Lt), np.nanmedian(Lt) if has_loss.any() else 0.0, Lt)
        S = 0.5 * minmax(Lt) + 0.5 * minmax(logtx)
    else:
        S = np.where(has_loss, plr, -np.inf)

    rng = np.random.default_rng(seed)
    C = list(rng.permutation(cand))
    k_paper = math.ceil(0.1 * n)
    decided = {}
    steps = []
    order = 0
    step = 0
    while C:
        cfg = DTOConfig(m=dto_cfg.m, Y=dto_cfg.Y, seed=seed * 1000 + step)
        arr = np.array(C)
        res = DTO(cfg, [1.0], [len(C) + 1.0]).minimize(
            lambda z: -S[arr[int(np.floor(z[0] - 1.0))]])
        j = int(arr[int(np.floor(res.x_best[0] - 1.0))])
        is_argmax = bool(S[j] >= np.max(S[arr]) - 1e-12)
        if variant == "paper_k":
            accept, why = order < k_paper, "paper_k"
        else:
            accept = bool(loss_ok[j] or tx_ok[j])
            why = "+".join(w for w, ok in (("loss", loss_ok[j]), ("tx", tx_ok[j])) if ok)
        steps.append({"step": step, "node": j, "score": float(S[j]), "accepted": accept,
                      "dto_found_argmax": is_argmax, "n_candidates": len(C),
                      "n_eval": res.n_eval})
        C.remove(j)
        step += 1
        if accept:
            order += 1
            decided[j] = ("malicious", order, why)
            continue
        if variant == "skip":
            continue
        break
    rows = []
    for j in range(n):
        if j in decided:
            dec, ordr, why = decided[j]
        elif has_loss[j] or (use_tx and tx[j] > 0):
            dec, ordr, why = "normal", "", ""
        else:
            dec, ordr, why = "insufficient_evidence", "", ""
        rows.append({"node_id": j, "decision": dec, "isolation_order": ordr,
                     "n_watched": f"{N[j]:.0f}", "drops_est": f"{D[j]:.4f}",
                     "plr": "" if np.isnan(plr[j]) else f"{plr[j]:.6f}",
                     "tx_frames": f"{tx[j]:.0f}",
                     "score": "" if not np.isfinite(S[j]) else f"{S[j]:.6f}",
                     "posterior_exceed": "" if np.isnan(post[j]) else f"{post[j]:.6g}",
                     "tx_robust_z": f"{tz[j]:.4f}", "accepted_by": why,
                     "n_watched_excluded": f"{Nx[j]:.0f}"})
    info = {"q0_baseline_plr": q0, "bonferroni_M": M, "candidates": int(len(cand)),
            "steps": steps, "k_paper": k_paper if variant == "paper_k" else None}
    return rows, info


def run_dir(telem_dir, n, s, variant, out_dir, run_tag, windows=None, dto_seed=0,
            b_max=B_MAX_OBSERVER, **kw):
    p = paths(telem_dir, n, s)
    ev = evidence(load_links(p["links"]), n, windows, b_max)
    seed = dto_seed * 1_000_003 + n * 101 + s * 7 + VARIANTS.index(variant)
    rows, info = detect(ev, n, variant, seed=seed, **kw)
    wtag = "all" if windows is None else "w" + "-".join(str(w) for w in sorted(windows))
    base = f"dto_decisions_{n}_s{s}_{run_tag}_{variant}_{wtag}"
    if dto_seed:
        base += f"_seed{dto_seed}"
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, base + ".csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DECISION_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    info.update(source=os.path.basename(p["links"]), source_sha256=sha256(p["links"]),
                decisions_sha256=sha256(out), b_pooled=ev["b_pooled"], dto_seed=dto_seed,
                seed=seed, b_max_observer=b_max,
                excluded_observers=ev["excluded_observers"])
    return out, info


def main(argv=None):
    ap = argparse.ArgumentParser(description="DTO offline di atas telemetry NS-3")
    ap.add_argument("--telem-dir", required=True)
    ap.add_argument("--run-tag", required=True, help="mis. stage2, ctrl0")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--nodes", type=int, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    ap.add_argument("--windows", type=int, nargs="*", default=None)
    ap.add_argument("--dto-seed", type=int, default=0)
    ap.add_argument("--m", type=int, default=30)
    ap.add_argument("--Y", type=int, default=100)
    ap.add_argument("--n-min", type=int, default=10)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--b-max", type=float, default=B_MAX_OBSERVER,
                    help="K9: observer dgn b_i >= b-max tidak memberi bukti (<= 0: nonaktif)")
    a = ap.parse_args(argv)
    b_max = a.b_max if a.b_max > 0 else None
    wins = set(a.windows) if a.windows else None
    meta = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "telem_dir": a.telem_dir,
            "run_tag": a.run_tag, "windows": sorted(wins) if wins else "all",
            "config": {"m": a.m, "Y": a.Y, "n_min": a.n_min, "delta": a.delta, "alpha": a.alpha,
                       "z_thr": 3.5, "dto_seed": a.dto_seed, "b_max_observer": b_max,
                       "note": "konfigurasi reproduksi (K5); bukan nilai dari paper"},
            "input_columns": DTO_INPUT_LINK_COLUMNS, "energy_criterion": "NOT_AVAILABLE (K3)",
            "scenarios": {}}
    for v in a.variants:
        for n in a.nodes:
            for s in a.seeds:
                out, info = run_dir(a.telem_dir, n, s, v, a.out_dir, a.run_tag, wins, a.dto_seed,
                                    b_max=b_max, dto_cfg=DTOConfig(m=a.m, Y=a.Y), n_min=a.n_min,
                                    delta=a.delta, alpha=a.alpha)
                meta["scenarios"][os.path.basename(out)] = info
                nm = sum(1 for st in info["steps"] if st["accepted"])
                print(f"{a.run_tag:8s} {v:11s} n={n:3d} s={s}: {nm:2d} malicious "
                      f"(kandidat {info['candidates']}, langkah {len(info['steps'])})")
    wtag = "all" if wins is None else "w" + "-".join(str(w) for w in sorted(wins))
    mp = os.path.join(a.out_dir, f"dto_meta_{a.run_tag}_{wtag}"
                      + (f"_seed{a.dto_seed}" if a.dto_seed else "") + ".json")
    with open(mp, "w") as f:
        json.dump(meta, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

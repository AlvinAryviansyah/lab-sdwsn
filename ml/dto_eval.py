#!/usr/bin/env python3
"""
dto_eval.py  --  evaluasi DTO langkah 3. SATU-SATUNYA tempat label digabung.
-------------------------------------------------------------------------
Label (actual_mal dari bigan_decisions_<n>_s<seed>.csv) dibaca SETELAH keputusan DTO
ditulis (dto_detect.py). Tidak ada hasil evaluasi yang dikembalikan ke detektor.

Ringkasan per (run, varian, window):
  all       semua sensor; insufficient_evidence dihitung sebagai prediksi normal
  evidence  hanya sensor dengan bukti (decision != insufficient_evidence)
  coverage  malicious dengan bukti / seluruh malicious
Pembanding pada skenario yang sama:
  cpp_dto       DTO C++ langkah 1 (self-report laju kirim + energi, k = 10% a priori),
                dari metrics_<n>_s<seed>_<run>.csv (dto_TP/FP/FN/TN)
  trust_thresh  trust telemetry < 0.5 dengan N >= 10 (telemetry run yang sama)
  bigan_ns3     prediksi Bi-GAN NS-3 per node (pred_intrusion), sumber yang sama dgn run
Diagnostik: kesamaan DTO dengan argmax eksak, determinisme, stabilitas seed DTO.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

from telemetry_schema import load_links, load_metrics, paths
from telemetry_trust import corrected_trust


def labels(scenario_dir, n, s):
    mal, pred = np.zeros(n, bool), np.zeros(n, bool)
    with open(os.path.join(scenario_dir, f"bigan_decisions_{n}_s{s}.csv")) as f:
        for r in csv.DictReader(f):
            i = int(r["node_id"])
            mal[i] = int(r["actual_mal"]) != 0
            pred[i] = int(r["pred_intrusion"]) != 0
    return mal, pred


def read_decisions(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def conf(pred, y):
    pred, y = np.asarray(pred, bool), np.asarray(y, bool)
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn}


def scores(c):
    tp, fp, fn, tn = c["TP"], c["FP"], c["FN"], c["TN"]
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f1 = 2 * p * r / (p + r) if p and r else (0.0 if (p is not None and r is not None) else None)
    tot = tp + fp + fn + tn
    return {**c, "precision": p, "recall": r, "f1": f1,
            "accuracy": (tp + tn) / tot if tot else None,
            "fpr": fp / (fp + tn) if fp + tn else None}


def add(a, b):
    for k in ("TP", "FP", "FN", "TN"):
        a[k] += b[k]
    return a


def zero():
    return {"TP": 0, "FP": 0, "FN": 0, "TN": 0}


def evaluate(dec_dir, run, variant, wtag, nodes, seeds, scenario_dir, telem_dir, metrics_dir,
             metrics_tag, meta, seed_suffix=""):
    agg = {k: zero() for k in ("all", "evidence", "cpp_dto", "trust_thresh", "bigan_ns3")}
    cov = [0, 0]
    argmax_ok = steps = 0
    per = []
    for n in nodes:
        for s in seeds:
            name = f"dto_decisions_{n}_s{s}_{run}_{variant}_{wtag}{seed_suffix}.csv"
            rows = read_decisions(os.path.join(dec_dir, name))
            mal, bpred = labels(scenario_dir, n, s)          # label: SETELAH keputusan
            dec = np.array([r["decision"] for r in rows])
            pred = dec == "malicious"
            ev = dec != "insufficient_evidence"
            c_all, c_ev = conf(pred, mal), conf(pred[ev], mal[ev])
            add(agg["all"], c_all)
            add(agg["evidence"], c_ev)
            cov[0] += int((mal & ev).sum())
            cov[1] += int(mal.sum())
            info = meta["scenarios"].get(name, {})
            for st in info.get("steps", []):
                steps += 1
                argmax_ok += bool(st["dto_found_argmax"])
            sc = {"n": n, "seed": s, "all": scores(c_all), "evidence": scores(c_ev),
                  "detected": [int(i) for i in np.flatnonzero(pred)],
                  "malicious_with_evidence": int((mal & ev).sum()), "n_malicious": int(mal.sum())}
            # pembanding
            mp = os.path.join(metrics_dir, f"metrics_{n}_s{s}{metrics_tag}.csv")
            if os.path.exists(mp):
                m = load_metrics(mp)
                c = {k: int(m[f"dto_{k}"]) for k in ("TP", "FP", "FN", "TN")}
                add(agg["cpp_dto"], c)
                sc["cpp_dto"] = scores(c)
            ct = corrected_trust(load_links(paths(telem_dir, n, s)["links"]), n)
            tp = (ct["trust"] < 0.5) & (ct["n_watched"] >= 10)
            c = conf(tp, mal)
            add(agg["trust_thresh"], c)
            sc["trust_thresh"] = scores(c)
            c = conf(bpred, mal)
            add(agg["bigan_ns3"], c)
            sc["bigan_ns3"] = scores(c)
            per.append(sc)
    f1s = [p["all"]["f1"] for p in per if p["all"]["f1"] is not None]
    return {"run": run, "variant": variant, "windows": wtag, "seed_suffix": seed_suffix,
            "pooled": {k: scores(v) for k, v in agg.items()},
            "coverage": {"malicious_with_evidence": cov[0], "malicious_total": cov[1],
                         "fraction": cov[0] / cov[1] if cov[1] else None},
            "dto_argmax_equivalence": {"steps": steps, "found_argmax": argmax_ok},
            "f1_all_per_scenario": {"median": float(np.median(f1s)) if f1s else None,
                                    "min": float(min(f1s)) if f1s else None},
            "scenarios": per}


def same_decisions(dir_a, dir_b, names):
    out = []
    for nm_a, nm_b in names:
        a = [r["decision"] for r in read_decisions(os.path.join(dir_a, nm_a))]
        b = [r["decision"] for r in read_decisions(os.path.join(dir_b, nm_b))]
        out.append(a == b)
    return out


def fmt(x, d=3):
    return "  -  " if x is None else f"{x:.{d}f}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dec-dir", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--variants", nargs="+", default=["main"])
    ap.add_argument("--wtag", default="all")
    ap.add_argument("--nodes", type=int, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--scenario-dir", required=True)
    ap.add_argument("--telem-dir", required=True)
    ap.add_argument("--metrics-dir", required=True)
    ap.add_argument("--metrics-tag", required=True)
    ap.add_argument("--seed-suffix", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    meta_name = f"dto_meta_{a.run}_{a.wtag}" + (a.seed_suffix or "") + ".json"
    meta = json.load(open(os.path.join(a.dec_dir, meta_name)))
    rep = {"meta_file": meta_name, "config": meta["config"], "results": []}
    for v in a.variants:
        r = evaluate(a.dec_dir, a.run, v, a.wtag, a.nodes, a.seeds, a.scenario_dir, a.telem_dir,
                     a.metrics_dir, a.metrics_tag, meta, a.seed_suffix)
        rep["results"].append(r)
        p = r["pooled"]
        print(f"{a.run:7s} {v:11s} {a.wtag:4s} | all P={fmt(p['all']['precision'])} "
              f"R={fmt(p['all']['recall'])} F1={fmt(p['all']['f1'])} FPR={fmt(p['all']['fpr'])} "
              f"| bukti P={fmt(p['evidence']['precision'])} R={fmt(p['evidence']['recall'])} "
              f"| cakupan {r['coverage']['malicious_with_evidence']}/{r['coverage']['malicious_total']}"
              f" | argmax {r['dto_argmax_equivalence']['found_argmax']}/"
              f"{r['dto_argmax_equivalence']['steps']}")
    base = rep["results"][0]["pooled"]
    for k in ("cpp_dto", "trust_thresh", "bigan_ns3"):
        b = base[k]
        print(f"{'':7s} [{k:12s}] all P={fmt(b['precision'])} R={fmt(b['recall'])} "
              f"F1={fmt(b['f1'])} FPR={fmt(b['fpr'])}")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

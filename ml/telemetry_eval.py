#!/usr/bin/env python3
"""
telemetry_eval.py  --  evaluasi telemetry NS-3 (langkah 2). SATU-SATUNYA tempat label
-----------------------------------------------------------------------------------
malicious (actual_mal pada bigan_decisions_<n>_s<seed>.csv) digabungkan dengan
telemetry -- setelah telemetry dan trust selesai dihitung, hanya untuk pelaporan.

Laporan per skenario (n, seed) + gabungan:
  validasi     telemetry_schema.validate (skema, agregasi window, audit, FlowMonitor,
               energi)
  watchdog     rasio penerusan per kelas (malicious / normal):
                 est      = (fwd + late) / (fwd + late + clean)   (ambigu dibuang; bias
                            naik bila drop kebetulan jatuh ke ambigu)
                 est_corrected = 1 - sum D / sum N  (koreksi laju kebetulan b per
                            observer; telemetry_trust.corrected_trust)
                 lo / hi  = ambigu dihitung sebagai drop / sebagai terusan (batas)
  audit        kebenaran simulator per kelas: rasio j benar-benar memancarkan paket
               watch (penyebab j_tx_*) dan matriks kategori watchdog x penyebab
  trust        (opsional --trust-dir) AUC (1 - trust) vs label pada sensor teramati
  repeat       (opsional --repeat-dir) SHA-256 berkas telemetry identik dgn run ulang
"""

import argparse
import csv
import hashlib
import json
import os
import sys

import numpy as np

from telemetry_trust import corrected_trust
from telemetry_schema import (WD_CATEGORIES, forward_definition, load_audit, load_links, load_meta, load_metrics,
                              load_nodes, node_totals, paths, validate)


def load_labels(scenario_dir, n, seed):
    mal = np.zeros(n, dtype=bool)
    with open(os.path.join(scenario_dir, f"bigan_decisions_{n}_s{seed}.csv")) as f:
        for r in csv.DictReader(f):
            mal[int(r["node_id"])] = int(r["actual_mal"]) != 0
    return mal


def auc(score, y):
    """AUC Mann-Whitney (ties = 0.5). y bool: positif = malicious."""
    s, y = np.asarray(score, float), np.asarray(y, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return None
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def ratios(counts):
    f, l, rx, c = (counts[k] for k in WD_CATEGORIES)
    heard, clear = f + l, f + l + c
    tot = f + l + rx + c
    return {"watched": int(tot), "forwarded": int(f), "forwarded_late": int(l),
            "unobserved_rxfail": int(rx), "unobserved_clean": int(c),
            "est": heard / clear if clear else None,
            "lo": heard / tot if tot else None,
            "hi": (heard + rx) / tot if tot else None,
            "ambiguous_frac": rx / tot if tot else None}


def class_counts(tot, mask):
    return {k: float(tot[f"obs_{k}"][mask].sum()) for k in WD_CATEGORIES}


def audit_summary(audit, mal):
    by_cls = {"malicious": {}, "normal": {}}
    matrix = {}
    for r in audit:
        cls = "malicious" if mal[r["neighbor"]] else "normal"
        by_cls[cls][r["true_cause"]] = by_cls[cls].get(r["true_cause"], 0) + r["count"]
        m = matrix.setdefault(r["wd_category"], {})
        m[r["true_cause"]] = m.get(r["true_cause"], 0) + r["count"]
    out = {}
    for cls, d in by_cls.items():
        tot = sum(d.values())
        tx = sum(v for k, v in d.items() if k.startswith("j_tx_"))
        out[cls] = {"watches": tot, "causes": d,
                    "true_tx_ratio": tx / tot if tot else None}
    out["category_x_cause"] = matrix
    # Kemampuan watchdog membedakan: dari watch tak terdengar, berapa yang memang
    # tidak dipancarkan j (kegagalan penerusan) vs dipancarkan tapi terlewat observer.
    for cat in ("unobserved_rxfail", "unobserved_clean"):
        d = matrix.get(cat, {})
        tot = sum(d.values())
        miss = sum(v for k, v in d.items() if k in ("j_tx_rxfail_at_observer",
                                                    "j_tx_no_rx_event_at_observer"))
        out[f"{cat}_observer_miss_frac"] = miss / tot if tot else None
    return out


# Metrik yang diukur dgn jam dinding (chrono) -> berbeda antar run apa pun.
WALLCLOCK_METRICS = {"rt_compute_ms", "computation_overhead_percent"}


def compare_passive(ref, telem):
    """ref: metrics run --telemetry=false; telem: run dgn telemetry. Semua metrik ref
    (kecuali jam dinding dan telemetry_enabled) harus identik."""
    diff = {k: (v, telem.get(k)) for k, v in ref.items()
            if k not in WALLCLOCK_METRICS and k != "telemetry_enabled" and telem.get(k) != v}
    return {"identical": not diff, "differences": diff}


def sha_files(telem_dir, n, seed):
    p = paths(telem_dir, n, seed)
    out = {}
    for k in ("node", "links", "audit"):
        with open(p[k], "rb") as f:
            out[k] = hashlib.sha256(f.read()).hexdigest()
    return out


def evaluate(a):
    report = {"scenarios": [], "pooled": {}}
    pool = {"malicious": {k: 0.0 for k in WD_CATEGORIES},
            "normal": {k: 0.0 for k in WD_CATEGORIES}}
    pool_corr = {"malicious": [0.0, 0.0], "normal": [0.0, 0.0]}
    for n in a.nodes:
        for s in a.seeds:
            p = paths(a.telem_dir, n, s)
            nodes, links, audit = (load_nodes(p["node"]), load_links(p["links"]),
                                   load_audit(p["audit"]))
            meta = load_meta(p["meta"])
            mpath = (os.path.join(a.metrics_dir, f"metrics_{n}_s{s}{a.metrics_tag}.csv")
                     if a.metrics_dir else None)
            metrics = load_metrics(mpath) if mpath else None
            errs = validate(nodes, links, audit, meta, metrics)
            mal = load_labels(a.scenario_dir, n, s)            # label: SETELAH telemetry
            tot = node_totals(nodes, n)
            sc = {"n": n, "seed": s, "validation_errors": errs,
                  "n_malicious": int(mal.sum()),
                  "malicious_watched": int((tot["obs_watched"][mal] > 0).sum()),
                  "watchdog": {}, "audit": audit_summary(audit, mal)}
            ct = corrected_trust(links, n)
            sc["b_pooled"] = ct["b_pooled"]
            sc["cap_hit"] = {"malicious": int(ct["cap_hit"][mal].sum()),
                             "normal": int(ct["cap_hit"][~mal].sum())}
            for cls, m in (("malicious", mal), ("normal", ~mal)):
                c = class_counts(tot, m)
                sc["watchdog"][cls] = ratios(c)
                nw, dd = ct["n_watched"][m].sum(), ct["drops_est"][m].sum()
                sc["watchdog"][cls]["est_corrected"] = float(1 - dd / nw) if nw else None
                pool_corr[cls][0] += nw
                pool_corr[cls][1] += dd
                for k in WD_CATEGORIES:
                    pool[cls][k] += c[k]
            if metrics is not None:
                sc["forward_definition"] = forward_definition(metrics)
                sc["flowmon_check"] = {k: metrics.get(k) for k in (
                    "telemetry_ip_forward_udp_total", "telemetry_ip_forward_udp_delivered",
                    "flowmon_udp_times_forwarded",
                    "telemetry_ip_forward_tcp_total", "flowmon_tcp_times_forwarded",
                    "telemetry_ip_orig_udp_total", "flowmon_udp_tx_packets",
                    "telemetry_ip_local_udp_total", "flowmon_udp_rx_packets")}
                sc["grayhole_drop_prob"] = metrics.get("grayhole_drop_prob")
            if a.trust_dir:
                from telemetry_trust import load_trust
                tr = load_trust(os.path.join(a.trust_dir, f"trust_telemetry_{n}_s{s}.csv"), n)
                observed = (tot["obs_forwarded"] + tot["obs_forwarded_late"] +
                            tot["obs_unobserved_clean"]) > 0
                sc["trust"] = {
                    "auc_observed": auc(1 - tr[observed], mal[observed]),
                    "auc_all": auc(1 - tr, mal),
                    "observed_nodes": int(observed.sum()),
                    "mean_trust_malicious": float(tr[mal].mean()) if mal.any() else None,
                    "mean_trust_normal": float(tr[~mal].mean())}
            if a.repeat_dir and os.path.exists(paths(a.repeat_dir, n, s)["node"]):
                h1, h2 = sha_files(a.telem_dir, n, s), sha_files(a.repeat_dir, n, s)
                sc["repeat_identical"] = h1 == h2
            pref = (os.path.join(a.metrics_dir, f"metrics_{n}_s{s}{a.passive_tag}.csv")
                    if a.metrics_dir and a.passive_tag else None)
            if pref and metrics is not None and os.path.exists(pref):
                sc["passive"] = compare_passive(load_metrics(pref), metrics)
            report["scenarios"].append(sc)
            w = sc["watchdog"]
            print(f"n={n:3d} s={s}: valid={'ya' if not errs else 'TIDAK'}  "
                  f"fwd malicious est={_f(w['malicious']['est'])} "
                  f"kor={_f(w['malicious']['est_corrected'])}  normal est="
                  f"{_f(w['normal']['est'])} kor={_f(w['normal']['est_corrected'])}  "
                  f"ambigu={_f(w['normal']['ambiguous_frac'])}"
                  + (f"  AUC trust={_f(sc['trust']['auc_observed'])}" if a.trust_dir else "")
                  + (f"  ulang identik={sc['repeat_identical']}"
                     if "repeat_identical" in sc else "")
                  + (f"  pasif identik={sc['passive']['identical']}" if "passive" in sc else ""))
    report["pooled"] = {cls: ratios(c) for cls, c in pool.items()}
    for cls, (nw, dd) in pool_corr.items():
        report["pooled"][cls]["est_corrected"] = float(1 - dd / nw) if nw else None
    report["all_valid"] = all(not s["validation_errors"] for s in report["scenarios"])
    rep_ok = [s["repeat_identical"] for s in report["scenarios"] if "repeat_identical" in s]
    pas_ok = [s["passive"]["identical"] for s in report["scenarios"] if "passive" in s]
    report["repeat_checked"], report["repeat_all_identical"] = len(rep_ok), all(rep_ok)
    report["passive_checked"], report["passive_all_identical"] = len(pas_ok), all(pas_ok)
    # all_valid untuk skenario definisi lama hanya memakai SYARAT PERLU penerusan
    defs = [s.get("forward_definition") for s in report["scenarios"]]
    report["forward_check"] = {"exact_delivered": defs.count("delivered"),
                               "legacy_necessary_only": defs.count("legacy_total")}
    return report


def _f(x):
    return "  -  " if x is None else f"{x:.3f}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--telem-dir", required=True)
    ap.add_argument("--scenario-dir", required=True, help="folder bigan_decisions_*.csv")
    ap.add_argument("--metrics-dir", default=None)
    ap.add_argument("--metrics-tag", default="")
    ap.add_argument("--nodes", type=int, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--trust-dir", default=None)
    ap.add_argument("--repeat-dir", default=None,
                    help="telemetry run ulang seed sama (skenario yang ada saja)")
    ap.add_argument("--passive-tag", default=None,
                    help="akhiran metrics run --telemetry=false, mis. _notelem")
    ap.add_argument("--out", default=None, help="tulis laporan JSON")
    a = ap.parse_args(argv)
    rep = evaluate(a)
    pm, pn = rep["pooled"]["malicious"], rep["pooled"]["normal"]
    print(f"Gabungan: fwd malicious est={_f(pm['est'])} kor={_f(pm['est_corrected'])} "
          f"[{_f(pm['lo'])}, {_f(pm['hi'])}]  normal est={_f(pn['est'])} "
          f"kor={_f(pn['est_corrected'])} [{_f(pn['lo'])}, {_f(pn['hi'])}]  "
          f"semua valid={rep['all_valid']} (penerusan: {rep['forward_check']})  ulang identik={rep['repeat_all_identical']} "
          f"({rep['repeat_checked']})  pasif identik={rep['passive_all_identical']} "
          f"({rep['passive_checked']})")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=1)
    return 0 if rep["all_valid"] else 1


if __name__ == "__main__":
    sys.exit(main())

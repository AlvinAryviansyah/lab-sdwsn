#!/usr/bin/env python3
"""
build_dashboard.py  --  End-to-End Performance Dashboard HieMulti-Block
----------------------------------------------------------------------
Membaca HANYA hasil yang sudah ada di repository (CSV/JSON hasil run NS-3 dan laporan
Python) lalu menulis satu berkas mandiri dashboard/index.html (tanpa CDN, tanpa
dependensi browser). Tidak menjalankan simulasi atau model; tidak mengubah hasil.

Sumber data (semua relatif terhadap repo):
  ns3-sim/results/metrics_<n>_s<seed>{,_greedy}.csv        langkah-1 (TLDQN trust GT, greedy)
  ns3-sim/results/timeseries_<n>_s<seed>{,_greedy}.csv     confusion Bi-GAN per window
  ns3-sim/results/step2/metrics_*_{stage1,stage2,ctrl0}.csv langkah-2
  ns3-sim/results/step2/telemetry_{stage1,stage2,ctrl0}/    telemetry NS-3 + audit watchdog
  ns3-sim/results/step2/trust_stage1/                       trust telemetry (input tahap-2)
  ns3-sim/results/{routes_stage1,routes_stage2,routes_greedy,.}/tldqn_nexthop_*.csv
  ns3-sim/results/bigan_decisions_<n>_s<seed>.csv          ground truth skenario (EVALUASI)
  ml/reports/*.json                                         evaluasi NSL-KDD & langkah-2

Pemisahan: ground truth disimpan terpisah (D.gt) dan hanya dipakai untuk evaluasi di
browser; hasil NSL-KDD (offline) terpisah dari hasil NS-3.

Pemakaian:
  .venv/bin/python ml/build_dashboard.py        # -> dashboard/index.html
  python3 -m http.server 8050 --bind 127.0.0.1 --directory dashboard
"""

import csv
import json
import os
import subprocess
import sys
import time

import numpy as np

import export_routes as X
import telemetry_schema as S
from honeycomb_topology import NODE_SWEEP, HoneycombTopology, parse_positions
from routing_env import EnvConfig

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "ns3-sim", "results")
S2 = os.path.join(RES, "step2")
REP = os.path.join(HERE, "reports")
SEEDS = [1, 2, 3]

# Konfigurasi run yang ADA di repo. facets dipakai filter dashboard.
CONFIGS = {
    "step1_tldqn": dict(label="Langkah-1 · TLDQN · trust ground-truth (tersimulasi dari label)",
                        short="TLDQN + trust GT", stage="step1", policy="TLDQN", trust="gt",
                        routes=RES, mdir=RES, tag="", telem=None, nodes=NODE_SWEEP),
    "step1_greedy": dict(label="Langkah-1 · Greedy · tanpa trust", short="Greedy",
                         stage="step1", policy="Greedy", trust="none_used",
                         routes=os.path.join(RES, "routes_greedy"), mdir=RES, tag="_greedy",
                         telem=None, nodes=NODE_SWEEP),
    "stage1": dict(label="Stage 1 · TLDQN · trust 0.5 (tanpa informasi)", short="Stage 1 (trust 0.5)",
                   stage="stage1", policy="TLDQN", trust="uniform",
                   routes=os.path.join(RES, "routes_stage1"), mdir=S2, tag="_stage1",
                   telem="telemetry_stage1", nodes=NODE_SWEEP),
    "stage2": dict(label="Stage 2 · TLDQN · trust telemetry", short="Stage 2 (trust telemetry)",
                   stage="stage2", policy="TLDQN", trust="telemetry",
                   routes=os.path.join(RES, "routes_stage2"), mdir=S2, tag="_stage2",
                   telem="telemetry_stage2", nodes=NODE_SWEEP),
    "ctrl0": dict(label="Kontrol · TLDQN · trust 0.5 · grayholeDrop=0", short="Kontrol (tanpa grayhole)",
                  stage="control", policy="TLDQN", trust="uniform",
                  routes=os.path.join(RES, "routes_stage1"), mdir=S2, tag="_ctrl0",
                  telem="telemetry_ctrl0", nodes=[20, 60, 100]),
}
METRIC_KEYS = [
    "pdr_percent", "packet_drop_ratio_percent", "throughput_kbps", "latency_ms",
    "energy_consumption_J", "network_lifetime_s", "grayhole_drops", "grayhole_not_dropped",
    "grayhole_drop_prob", "benign_hops_per_rx_est", "planned_hops_mean",
    "TP", "TN", "FP", "FN", "dto_TP", "dto_TN", "dto_FP", "dto_FN",
    "nodes_blocked", "attackers_blocked", "windows",
    "flowmon_udp_tx_packets", "flowmon_udp_rx_packets", "telemetry_ip_forward_udp_delivered",
    "flowmon_udp_times_forwarded", "wd_watches", "wd_forwarded", "wd_forwarded_late",
    "wd_unobserved_rxfail", "wd_unobserved_clean",
]


def rel(p):
    return os.path.relpath(p, ROOT)


def read_table(path):
    nh = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            nh[(int(r["node_id"]), int(r["dest_sink"]))] = (int(r["next_hop"]),
                                                            int(r["next_hop_is_sink"]))
    return nh


def paths_from(nh, topo):
    n, out = topo.n, []
    for s in range(n):
        t = int(topo.dest[s]) - n
        cur, p = s, [s]
        for _ in range(n + 2):
            j, is_sink = nh[(cur, t)]
            p.append(n + j if is_sink else j)
            if is_sink:
                break
            cur = j
        out.append(p)
    return out


def ground_truth(n, s):
    mal = []
    with open(os.path.join(RES, f"bigan_decisions_{n}_s{s}.csv")) as f:
        for r in csv.DictReader(f):
            mal.append(int(r["actual_mal"]))
    return mal


def timeseries(path):
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            out.append([float(r["t"]), int(r["TP"]), int(r["TN"]), int(r["FP"]), int(r["FN"])])
    return out


def trust_file(n, s):
    p = os.path.join(S2, "trust_stage1", f"trust_telemetry_{n}_s{s}.csv")
    if not os.path.exists(p):
        return None
    rows = {}
    with open(p) as f:
        for r in csv.DictReader(f):
            rows[int(r["node_id"])] = [round(float(r[k]), 6) for k in
                                       ("trust", "n_watched", "n_forwarded", "n_ambiguous",
                                        "n_unobserved_clean", "drops_est")]
    return [rows[i] for i in range(n)]


def telemetry_view(tdir, n, s, paths):
    """Ringkasan telemetry per node dan per hop rute untuk satu run."""
    p = S.paths(os.path.join(S2, tdir), n, s)
    if not os.path.exists(p["node"]):
        return None
    tot = S.node_totals(S.load_nodes(p["node"]), n)
    node = {k: [round(float(v), 6) for v in tot[k]] for k in
            ("obs_watched", "obs_forwarded", "obs_forwarded_late", "obs_unobserved_rxfail",
             "obs_unobserved_clean", "self_ip_fwd_pkts", "self_ip_orig_pkts", "self_energy_j",
             "self_phy_rx_fail_mpdu", "self_phy_rx_drop")}
    # audit (kebenaran simulator) per node yang diamati: [dipancarkan, total watch]
    aud = np.zeros((n, 2))
    for r in S.load_audit(p["audit"]):
        aud[r["neighbor"], 1] += r["count"]
        if r["true_cause"].startswith("j_tx_"):
            aud[r["neighbor"], 0] += r["count"]
    # kualitas tautan per hop rute (agregat semua window)
    want = {(a, b) for pth in paths for a, b in zip(pth[:-1], pth[1:])}
    agg = {}
    for r in S.load_links(p["links"]):
        j = r["neighbor_id"] if r["neighbor_type"] == "sensor" else (
            n + r["neighbor_id"] if r["neighbor_type"] == "sink" else None)
        if j is None or (r["observer"], j) not in want:
            continue
        a = agg.setdefault((r["observer"], j), [0.0, 0.0, 0, 0, 0, 0])
        if r["rx_ok_from"]:
            a[0] += r["rssi_mean_dbm"] * r["rx_ok_from"]
            a[1] += r["snr_mean_db"] * r["rx_ok_from"]
            a[2] += r["rx_ok_from"]
        a[3] += r["mac_handoff_acked"]
        a[4] += r["mac_ack_timeout"]
        a[5] += r["mac_handoff_drop_retry"]
    hops = {f"{c}-{j}": [round(v[0] / v[2], 2) if v[2] else None,
                         round(v[1] / v[2], 2) if v[2] else None, v[3], v[4], v[5]]
            for (c, j), v in agg.items()}
    return {"node": node, "audit": aud.astype(int).tolist(), "hops": hops,
            "source": rel(p["node"])}


def audit_matrix(tdir, nodes):
    out = {}
    for n in nodes:
        for s in SEEDS:
            p = S.paths(os.path.join(S2, tdir), n, s)["audit"]
            if not os.path.exists(p):
                continue
            m = {}
            for r in S.load_audit(p):
                m[f"{r['wd_category']}|{r['true_cause']}"] = m.get(
                    f"{r['wd_category']}|{r['true_cause']}", 0) + r["count"]
            out[f"{n}_{s}"] = m
    return out


def link_model(n, s, mal, paths_all):
    """Link quality MODEL Python (prr x stability) = masukan TLDQN, per hop yang dipakai."""
    env = X.build_env(n, s, np.zeros(n, bool), EnvConfig(), "none")  # tidak memakai label
    lq = {}
    for paths in paths_all:
        for p in paths:
            for a, b in zip(p[:-1], p[1:]):
                lq[f"{a}-{b}"] = round(float(env.prr[a, b] * env.stability[a, b]), 4)
    return lq


def load_json(name):
    p = os.path.join(REP, name)
    return json.load(open(p)) if os.path.exists(p) else None


def git_rev():
    try:
        return subprocess.check_output(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def build():
    sens, sinks = parse_positions()
    D = {"meta": {"built_at": time.strftime("%Y-%m-%d %H:%M"), "git": git_rev(),
                  "ns3": "ns-3.48, 802.11ac 80 MHz, jangkauan 150 m, 600 s, 50 paket/node",
                  "grayhole": "relay malicious menjatuhkan 70% paket transit"},
         "nodes": NODE_SWEEP, "seeds": SEEDS,
         "sensors": sens.round(2).tolist(), "sinks": sinks.round(2).tolist(),
         "configs": {}, "runs": {}, "gt": {}, "trust": {}, "lq": {}, "audit": {}}
    for k, c in CONFIGS.items():
        D["configs"][k] = {f: c[f] for f in ("label", "short", "stage", "policy", "trust")}
        D["configs"][k]["telemetry"] = c["telem"] is not None
        D["configs"][k]["source"] = rel(os.path.join(c["mdir"], f"metrics_<n>_s<seed>{c['tag']}.csv"))
        D["runs"][k] = {}
    for n in NODE_SWEEP:
        topo = HoneycombTopology(n)
        for s in SEEDS:
            key = f"{n}_{s}"
            D["gt"][key] = ground_truth(n, s)
            D["trust"][key] = trust_file(n, s)
            # Trust yang DIPAKAI run langkah-1 TLDQN: reputasi tersimulasi dari label,
            # direproduksi deterministik persis seperti export_routes (RNG 20000*seed+n).
            env_sim = X.build_env(n, s, np.array(D["gt"][key], bool), EnvConfig(), "simulated")
            D.setdefault("trust_sim", {})[key] = [round(float(v), 4) for v in env_sim.trust[:n]]
            paths_all = []
            for k, c in CONFIGS.items():
                if n not in c["nodes"]:
                    continue
                mp = os.path.join(c["mdir"], f"metrics_{n}_s{s}{c['tag']}.csv")
                if not os.path.exists(mp):
                    continue
                m = S.load_metrics(mp)
                paths = paths_from(read_table(os.path.join(c["routes"],
                                                           f"tldqn_nexthop_{n}_s{s}.csv")), topo)
                paths_all.append(paths)
                tsp = os.path.join(c["mdir"], f"timeseries_{n}_s{s}{c['tag']}.csv")
                run = {"m": {mk: m.get(mk) for mk in METRIC_KEYS}, "paths": paths,
                       "ts": timeseries(tsp) if os.path.exists(tsp) else None}
                if c["telem"]:
                    run["tel"] = telemetry_view(c["telem"], n, s, paths)
                D["runs"][k][key] = run
            D["lq"][key] = link_model(n, s, D["gt"][key], paths_all)
    for k, c in CONFIGS.items():
        if c["telem"]:
            D["audit"][k] = audit_matrix(c["telem"], c["nodes"])
    bg = load_json("bigan_eval_bigan_v2_h2.json")
    D["nslkdd"] = None if bg is None else {
        "checkpoint": bg.get("checkpoint"), "threshold": bg.get("threshold"),
        "validation": bg.get("validation"), "test": bg.get("test"),
        "source": "ml/reports/bigan_eval_bigan_v2_h2.json"}
    D["reports"] = {"network": load_json("telemetry_step2_network.json"),
                    "tables": (load_json("telemetry_step2_tables.json") or {}).get("totals"),
                    "probe": (load_json("telemetry_step2_tables.json") or {}).get("trust_probe"),
                    "stage1": _pick(load_json("telemetry_step2_stage1.json")),
                    "stage2": _pick(load_json("telemetry_step2_stage2.json")),
                    "ctrl0": _pick(load_json("telemetry_step2_ctrl0.json")),
                    "stage1v2": _pick(load_json("telemetry_step2_stage1v2.json"))}
    return D


def _pick(ev):
    if not ev:
        return None
    return {k: ev.get(k) for k in ("all_valid", "repeat_checked", "repeat_all_identical",
                                   "passive_checked", "passive_all_identical", "forward_check",
                                   "pooled")}


def main(argv=None):
    out_dir = os.path.join(ROOT, "dashboard")
    os.makedirs(out_dir, exist_ok=True)
    D = build()
    payload = json.dumps(D, separators=(",", ":"), allow_nan=False).replace("</", "<\\/")
    with open(os.path.join(HERE, "dashboard_template.html"), encoding="utf-8") as f:
        html = f.read().replace("__DATA__", payload)
    path = os.path.join(out_dir, "index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"-> {rel(path)} ({os.path.getsize(path) / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

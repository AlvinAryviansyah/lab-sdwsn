#!/usr/bin/env python3
"""
analyze_stage2_tables.py  --  diagnosis tabel rute tahap-2 (langkah 2, evaluasi saja)
------------------------------------------------------------------------------------
Memakai label HANYA untuk evaluasi. Untuk tiap skenario (n, seed):
  - jumlah sumber yang rutenya melewati relay malicious: tahap-1 (trust 0.5),
    tahap-2 (trust telemetry), dan langkah-1 (trust tersimulasi dari label);
  - di tahap-2, apakah relay malicious itu TERAMATI di tahap-1 (trust != 0.5);
  - entri next-hop yang memilih malicious teramati padahal ada kandidat yang lebih
    tepercaya (sink atau sensor dengan trust lebih tinggi);
  - kontrafaktual: trust telemetry dipetakan ke skala pelatihan TLDQN
    ((trust_obs * t + 1) / (trust_obs + 2)), untuk menguji dugaan ekstrapolasi.
TLDQN tidak dilatih ulang atau diubah.
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

import export_routes as X
from honeycomb_topology import HoneycombTopology, K_ACTIONS
from routing_env import EnvConfig, SLOT_FEATURES, STATE_DIM
from telemetry_trust import load_trust
from tldqn_routing import DQNConfig, QNet

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "ns3-sim", "results")


def read_rows(path):
    with open(path) as f:
        return [{k: (float(v) if k == "q_value" else int(v)) for k, v in r.items()}
                for r in csv.DictReader(f)]


def via_mal(rows, n, mal):
    return [src for src in range(n) if any(mal[j] for j in X.follow(rows, n, src)[1:-1])]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", default=os.path.join(RES, "routes_stage1"))
    ap.add_argument("--stage2", default=os.path.join(RES, "routes_stage2"))
    ap.add_argument("--step1", default=RES, help="tabel langkah-1 (trust tersimulasi)")
    ap.add_argument("--trust-dir", default=os.path.join(RES, "step2", "trust_stage1"))
    ap.add_argument("--scenario-dir", default=RES)
    ap.add_argument("--out", default=os.path.join(HERE, "reports", "telemetry_step2_tables.json"))
    a = ap.parse_args(argv)

    cfg = EnvConfig()
    d = DQNConfig()
    q = QNet(STATE_DIM, K_ACTIONS, d.hidden, d.n_hidden)
    q.load_state_dict(torch.load(X.DEFAULT_MODEL, map_location="cpu"))
    q.eval()
    with open(os.path.join(a.stage2, "tldqn_nexthop_meta.json")) as f:
        scen = [(s["n"], s["seed"]) for s in json.load(f)["scenarios"]]

    out, tot = [], {"step1_simulated": 0, "stage1_none": 0, "stage2_telemetry": 0,
                    "stage2_via_observed_mal": 0, "stage2_via_unobserved_mal": 0,
                    "entries_observed_mal": 0, "entries_observed_mal_with_better_alt": 0,
                    "counterfactual_training_scale": 0}
    ptot = {}
    for n, s in scen:
        mal, _ = X.load_ground_truth(a.scenario_dir, n, s)
        tr = load_trust(os.path.join(a.trust_dir, f"trust_telemetry_{n}_s{s}.csv"), n)
        observed = np.abs(tr - 0.5) > 1e-9
        r1 = read_rows(os.path.join(a.step1, f"tldqn_nexthop_{n}_s{s}.csv"))
        r0 = read_rows(os.path.join(a.stage1, f"tldqn_nexthop_{n}_s{s}.csv"))
        r2 = read_rows(os.path.join(a.stage2, f"tldqn_nexthop_{n}_s{s}.csv"))
        v2 = via_mal(r2, n, mal)
        first_mal = [next(j for j in X.follow(r2, n, src)[1:-1] if mal[j]) for src in v2]
        topo = HoneycombTopology(n)
        ent = alt = 0
        for r in r2:
            j = r["next_hop"]
            if r["next_hop_is_sink"] or not (mal[j] and observed[j]):
                continue
            ent += 1
            cand = topo.candidates(r["node_id"], n + r["dest_sink"])
            alt += any(k != j and (k >= n or tr[k] > tr[j]) for k in cand)
        # Uji sensitivitas trust LANGSUNG pada Q-net (tanpa mengubah TLDQN): untuk tiap
        # keputusan, ubah HANYA fitur trust kandidat terpilih (fitur lain tetap) dan lihat
        # apakah argmax berubah. Label tidak masuk ke observasi: observe() tidak membaca
        # env.malicious (TestExportTrustSource.test_none_is_uninformative_and_label_independent;
        # indeks fitur dijaga test_trust_feature_index).
        env2 = X.build_env(n, s, mal, cfg, "telemetry", tr)
        tix = SLOT_FEATURES.index("trust")
        probe = {"decisions": 0, "flip_trust0": 0, "mal_entries": 0, "mal_flip_trust0": 0,
                 "mal_alt_lq_better": 0, "mal_alt_progress_better": 0}
        for c, t in topo.reachable_pairs():
            env2.cur, env2.dest = c, t
            obs, mask = env2.observe(), env2.action_mask()
            sl = topo.slots(c, t)
            a0 = _argmax(q, obs, mask)
            j = int(sl[a0])
            if j >= n:                       # sink terpilih: trust sink tetap 1
                continue
            probe["decisions"] += 1
            o0 = obs.copy()
            o0[a0 * len(SLOT_FEATURES) + tix] = 0.0
            flip = _argmax(q, o0, mask) != a0
            probe["flip_trust0"] += flip
            if mal[j] and observed[j]:
                probe["mal_entries"] += 1
                probe["mal_flip_trust0"] += flip
                # alternatif dgn trust lebih tinggi: apakah fitur lain lebih baik?
                lq = env2.prr[c] * env2.stability[c]
                alts = [k for k in sl if k >= 0 and k != j and (k >= n or tr[k] > tr[j])]
                if alts:
                    prog = topo.dist[c, t] - topo.dist[np.array(alts), t]
                    probe["mal_alt_lq_better"] += bool(np.max(lq[alts]) > lq[j])
                    probe["mal_alt_progress_better"] += bool(
                        np.max(prog) > topo.dist[c, t] - topo.dist[j, t])
        tr10 = (cfg.trust_obs * tr + 1) / (cfg.trust_obs + 2)
        env = X.build_env(n, s, mal, cfg, "telemetry", tr10)
        cf = len(via_mal(X.route_table(env, "tldqn", q), n, mal))
        row = {"n": n, "seed": s, "step1_simulated": len(via_mal(r1, n, mal)),
               "stage1_none": len(via_mal(r0, n, mal)), "stage2_telemetry": len(v2),
               "stage2_via_observed_mal": int(sum(observed[j] for j in first_mal)),
               "stage2_via_unobserved_mal": int(sum(not observed[j] for j in first_mal)),
               "entries_observed_mal": ent, "entries_observed_mal_with_better_alt": alt,
               "counterfactual_training_scale": cf,
               "malicious_observed": int((mal & observed).sum()), "n_malicious": int(mal.sum()),
               "probe": {k: int(v) for k, v in probe.items()}}
        out.append(row)
        for k in tot:
            tot[k] += row[k]
        for k, v in probe.items():
            ptot[k] = ptot.get(k, 0) + int(v)
    rep = {"totals": tot, "trust_probe": ptot, "scenarios": out,
           "definitions": {
               "entries_observed_mal_with_better_alt": "ketersediaan saja: ada kandidat lain "
               "dgn trust lebih tinggi (atau sink); tidak memperhitungkan fitur lain",
               "counterfactual_training_scale": "trust dipetakan (10t+1)/12; mempersempit "
               "rentang, jadi TIDAK memisahkan hipotesis bobot-kecil vs ekstrapolasi",
               "flip_trust0": "keputusan (next-hop sensor) yang argmax-nya berubah bila HANYA "
               "trust kandidat terpilih di-set 0",
               "mal_alt_lq_better / mal_alt_progress_better": "pada entri malicious teramati, "
               "ada alternatif lebih tepercaya yang link quality (prr*stability) / progresnya "
               "juga lebih baik"},
           "note": "label hanya untuk evaluasi; TLDQN tidak diubah"}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=1)
    for k, v in list(tot.items()) + [("probe." + k, v) for k, v in ptot.items()]:
        print(f"{k:40s} {v}")
    return 0


def _argmax(q, obs, mask):
    with torch.no_grad():
        v = q(torch.as_tensor(obs[None], dtype=torch.float32))[0].numpy()
    v = np.where(mask, v, -np.inf)
    return int(np.argmax(v))


if __name__ == "__main__":
    sys.exit(main())

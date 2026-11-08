#!/usr/bin/env python3
"""
export_routes.py  --  TLDQN -> NS-3 (integrasi langkah 1)
----------------------------------------------------------
Mengekspor TABEL NEXT-HOP hasil kebijakan TLDQN (inferensi saja; tidak melatih)
untuk setiap skenario NS-3 (jumlah node n, seed). NS-3 memasang tabel ini sebagai
host route statis sehingga keputusan TLDQN menentukan penerusan paket multi-hop.

Satu baris per pasangan (node, sink tujuan) yang TERJANGKAU (honeycomb_topology):
  node_id,dest_sink,next_hop,next_hop_is_sink,slot,n_candidates,own_dest,q_value,node_malicious
    node_id          sensor 0..n-1 (= sensors.Get(i) di NS-3)
    dest_sink        indeks sink 0..3 (= sinks.Get(k))
    next_hop         sensor id, atau indeks sink bila next_hop_is_sink = 1
    slot             aksi TLDQN (0..K-1), urutan kandidat dari Python
    own_dest         1 jika dest_sink = sink terdekat node ini (tujuan aliran datanya)
    node_malicious   ground truth node_id saat ekspor; NS-3 memverifikasi kecocokannya
                     dengan bigan_decisions_<n>_s<seed>.csv (cegah tabel basi)

Sumber data per skenario (n, seed):
  - Ground truth node malicious: kolom actual_mal pada bigan_decisions_<n>_s<seed>.csv
    (berkas skenario yang sama yang dibaca NS-3), sehingga Python dan NS-3 memakai
    penyerang yang sama.
  - Energi: 1.0 untuk semua sensor (keputusan routing dibuat di awal simulasi).
  - Trust (--trust-source):
      simulated  reputasi pra-deploy TERSIMULASI dari ground truth (EnvConfig.trust_obs
                 observasi), RNG ditentukan oleh (n, seed). Perilaku langkah 1 (default).
      none       tanpa informasi: semua sensor 0.5 (= rataan prior Beta(1,1)). Untuk NS-3
                 TAHAP-1 langkah 2: rute tidak bergantung pada label.
      telemetry  trust dari telemetry NS-3 tahap-1 (ml/telemetry_trust.py,
                 trust_telemetry_<n>_s<seed>.csv di --trust-dir). Untuk NS-3 TAHAP-2.
    Pada none/telemetry label hanya dipakai untuk kolom node_malicious (cek konsistensi
    NS-3) dan ringkasan; tabel rute tidak bergantung padanya (diuji).
  - Stabilitas tautan: dari scenario_seed = 10000 * seed + n (model proksi Python).

Pemakaian:
  python export_routes.py                         # TLDQN, n 20..100, seed 1..3
  python export_routes.py --policy greedy --out-dir ../ns3-sim/results/routes_greedy
"""

import os
import sys
import csv
import json
import time
import hashlib
import argparse

import numpy as np
import torch

from honeycomb_topology import HoneycombTopology, K_ACTIONS, NODE_SWEEP, routing_schema
from routing_env import HoneycombRoutingEnv, EnvConfig, STATE_DIM, SLOT_FEATURES, NODE_FEATURES
from tldqn_routing import QNet, DQNConfig, masked_argmax, longest_route_hops

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "ns3-sim", "results")
DEFAULT_MODEL = os.path.join(HERE, "experiments", "phase2_tldqn", "tldqn", "checkpoints",
                             "tldqn_seed1.pt")
COLUMNS = ["node_id", "dest_sink", "next_hop", "next_hop_is_sink", "slot", "n_candidates",
           "own_dest", "q_value", "node_malicious"]
EXPORT_VERSION = 1


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_ground_truth(scenario_dir, n, seed):
    """actual_mal per sensor dari berkas skenario NS-3."""
    path = os.path.join(scenario_dir, f"bigan_decisions_{n}_s{seed}.csv")
    mal = np.zeros(n, dtype=bool)
    seen = set()
    with open(path) as f:
        for row in csv.DictReader(f):
            i = int(row["node_id"])
            mal[i] = int(row["actual_mal"]) != 0
            seen.add(i)
    if seen != set(range(n)):
        raise ValueError(f"{path}: node_id tidak lengkap 0..{n - 1}")
    return mal, path


TRUST_SOURCES = ("simulated", "none", "telemetry")


def build_env(n, seed, malicious, cfg, trust_source="simulated", sensor_trust=None):
    env = HoneycombRoutingEnv(n, scenario_seed=10000 * seed + n, cfg=cfg)
    env.malicious[:] = False
    env.malicious[:n] = malicious
    if trust_source == "simulated":
        rng = np.random.default_rng(20000 * seed + n)
        _, trust = env.sample_network_state(rng)      # reputasi dari ground truth skenario
    elif trust_source in ("none", "telemetry"):
        trust = np.ones(env.topo.n_nodes)             # sink terpercaya (= lingkungan)
        if trust_source == "none":
            trust[:n] = 0.5
        else:
            st = np.asarray(sensor_trust, float)
            if st.shape != (n,) or not np.all((st >= 0) & (st <= 1)):
                raise ValueError("sensor_trust harus berukuran (n,) dalam [0, 1]")
            trust[:n] = st
    else:
        raise ValueError(f"trust_source tidak dikenal: {trust_source}")
    energy = np.ones(env.topo.n_nodes)
    env.set_network_state(energy, trust)
    return env


def route_table(env, policy, qnet=None):
    """Baris tabel next-hop untuk semua pasangan terjangkau."""
    topo, rows = env.topo, []
    for c, t in topo.reachable_pairs():
        env.cur, env.dest = c, t
        obs, mask = env.observe(), env.action_mask()
        if policy == "tldqn":
            with torch.no_grad():
                q = qnet(torch.as_tensor(obs[None], dtype=torch.float32))
            a = int(masked_argmax(q, torch.as_tensor(mask[None]))[0])
            qv = float(q[0, a])
        elif policy == "greedy":
            a, qv = 0, float("nan")
        else:
            raise ValueError(policy)
        j = int(topo.slots(c, t)[a])
        is_sink = topo.is_sink(j)
        rows.append({"node_id": c, "dest_sink": t - topo.n,
                     "next_hop": (j - topo.n) if is_sink else j,
                     "next_hop_is_sink": int(is_sink), "slot": a,
                     "n_candidates": int(mask.sum()),
                     "own_dest": int(topo.dest[c] == t), "q_value": qv,
                     "node_malicious": int(env.malicious[c])})
    return rows


def follow(rows, n, src):
    """Ikuti tabel dari sensor src ke sink tujuannya. Mengembalikan daftar hop (node id
    abstrak: sensor i, sink n+k)."""
    nh = {(r["node_id"], r["dest_sink"]): r for r in rows}
    t = next(r["dest_sink"] for r in rows if r["node_id"] == src and r["own_dest"])
    cur, path = src, [src]
    for _ in range(n + 1):
        r = nh[(cur, t)]
        nxt = n + r["next_hop"] if r["next_hop_is_sink"] else r["next_hop"]
        path.append(nxt)
        if r["next_hop_is_sink"]:
            return path
        if nxt in path[:-1]:
            raise ValueError(f"loop pada tabel dari {src}: {path}")
        cur = nxt
    raise ValueError(f"tabel tidak mencapai sink dari {src}")


def write_table(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["node_id"], r["dest_sink"])):
            w.writerow({**r, "q_value": f"{r['q_value']:.6f}"})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=["tldqn", "greedy"], default="tldqn",
                    help="tldqn = metode; greedy = referensi progres terbesar (bukan metode)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--nodes", type=int, nargs="+", default=NODE_SWEEP)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--scenario-dir", default=RESULTS)
    ap.add_argument("--out-dir", default=RESULTS)
    ap.add_argument("--trust-source", choices=TRUST_SOURCES, default="simulated")
    ap.add_argument("--trust-dir", default=None,
                    help="folder trust_telemetry_<n>_s<seed>.csv (wajib utk telemetry)")
    args = ap.parse_args(argv)
    if args.trust_source == "telemetry" and not args.trust_dir:
        ap.error("--trust-source telemetry butuh --trust-dir")

    cfg = EnvConfig()
    qnet = None
    if args.policy == "tldqn":
        d = DQNConfig()
        qnet = QNet(STATE_DIM, K_ACTIONS, d.hidden, d.n_hidden)
        qnet.load_state_dict(torch.load(args.model, map_location="cpu"))
        qnet.eval()
    os.makedirs(args.out_dir, exist_ok=True)
    summary = []
    for n in args.nodes:
        topo_depth = longest_route_hops(HoneycombTopology(n))
        for seed in args.seeds:
            mal, gt_path = load_ground_truth(args.scenario_dir, n, seed)
            st, trust_file = None, None
            if args.trust_source == "telemetry":
                from telemetry_trust import load_trust
                trust_file = os.path.join(args.trust_dir, f"trust_telemetry_{n}_s{seed}.csv")
                st = load_trust(trust_file, n)
            env = build_env(n, seed, mal, cfg, args.trust_source, st)
            rows = route_table(env, args.policy, qnet)
            greedy = {(r["node_id"], r["dest_sink"]): r["slot"] for r in route_table(env, "greedy")}
            paths = [follow(rows, n, s) for s in range(n)]
            hops = [len(p) - 1 for p in paths]
            via_mal = sum(any(env.malicious[j] for j in p[1:-1]) for p in paths)
            out = os.path.join(args.out_dir, f"tldqn_nexthop_{n}_s{seed}.csv")
            write_table(out, rows)
            summary.append({"n": n, "seed": seed, "entries": len(rows),
                            "mean_hops": float(np.mean(hops)), "max_hops": int(max(hops)),
                            "longest_candidate_route": topo_depth,
                            "differs_from_greedy": float(np.mean(
                                [r["slot"] != greedy[(r["node_id"], r["dest_sink"])] for r in rows])),
                            "sources_routed_via_malicious": int(via_mal),
                            "n_malicious": int(mal.sum()),
                            "ground_truth_file": os.path.basename(gt_path),
                            "ground_truth_sha256": sha256(gt_path),
                            "trust_file": os.path.basename(trust_file) if trust_file else None,
                            "trust_sha256": sha256(trust_file) if trust_file else None,
                            "table_sha256": sha256(out)})
            s = summary[-1]
            print(f"n={n:3d} s={seed}: {s['entries']:3d} entri  hop rata2={s['mean_hops']:.2f} "
                  f"maks={s['max_hops']}  beda dari greedy={s['differs_from_greedy']*100:4.1f}%  "
                  f"sumber lewat malicious={s['sources_routed_via_malicious']}/{n}")
    meta = {
        "export_version": EXPORT_VERSION, "policy": args.policy,
        "trust_source": args.trust_source,
        "model": os.path.relpath(args.model, HERE) if qnet is not None else None,
        "model_sha256": sha256(args.model) if qnet is not None else None,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "assumptions": {
            "energy": "1.0 untuk semua sensor (keputusan di awal simulasi)",
            "trust": {"simulated": "reputasi pra-deploy tersimulasi dari ground truth "
                               "skenario (EnvConfig.trust_obs observasi)",
                      "none": "0.5 untuk semua sensor (tanpa informasi; tahap-1 langkah 2)",
                      "telemetry": "Beta dari watchdog NS-3 tahap-1 (ml/telemetry_trust.py)",
                      }[args.trust_source],
            "stability": "proksi Python, scenario_seed = 10000*seed + n",
            "malicious_relay": f"meneruskan dgn peluang {cfg.p_fwd_malicious} (grayhole); "
                               "NS-3 harus memakai nilai yang sama",
        },
        "env_config": cfg.to_dict(),
        "schema": routing_schema(SLOT_FEATURES, NODE_FEATURES),
        "columns": COLUMNS, "scenarios": summary,
    }
    with open(os.path.join(args.out_dir, "tldqn_nexthop_meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    print(f"Tabel -> {os.path.relpath(args.out_dir)}/tldqn_nexthop_<n>_s<seed>.csv "
          f"(+ tldqn_nexthop_meta.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

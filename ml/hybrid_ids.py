#!/usr/bin/env python3
"""
hybrid_ids.py  --  keputusan Hybrid IDS (DTO + Bi-GAN) + reroute minimal (langkah 4)
-----------------------------------------------------------------------------------
Desain: docs/STEP4_HYBRID_IDS_DESIGN.md (K11-K18 disetujui).

Alur (replay deterministik dua run, K12):
  run A  NS-3 --idsSource=none --telemWindow=T_obs  -> telemetry window 0 = [0, T_obs)
  ini    DTO langkah 3 (pra-K9 = utama, K15) pada window 0 SAJA + prediksi Bi-GAN
         + fusi (F2 utama, F1 pembanding; K13) -> hybrid_decisions_<n>_s<seed>.csv
         + reroute minimal (TLDQN masked argmax tanpa kandidat terisolasi)
           -> reroute_nexthop_<n>_s<seed>.csv
  run B  NS-3 --idsSource=file --idsDir=<out-dir>   (konfigurasi lain identik run A)

Aturan data (tanpa label):
  - telemetry: hanya kolom DTO_INPUT_LINK_COLUMNS (lewat dto_detect.evidence)
  - Bi-GAN   : hanya BIGAN_INPUT_COLUMNS = node_id, pred_intrusion (actual_mal TIDAK
               dibaca). Pengungkapan K14: prediksi Bi-GAN di simulasi berasal dari sampel
               NSL-KDD yang DITUGASKAN berdasarkan label skenario (export_decisions.py),
               jadi BUKAN deteksi independen dari trafik NS-3.
  - tabel rute: baris disalin apa adanya (termasuk kolom cek konsistensi NS-3); hanya
               next_hop/slot/q_value yang dipilih ulang.

Fusi:
  F2 (utama)     isolate = DTO malicious; Bi-GAN = alarm saja (dicatat, tidak mengisolasi)
  F1 (pembanding, EKSPERIMENTAL, tidak disebut paper) isolate = DTO malicious ATAU Bi-GAN
  insufficient_evidence -> tidak ada tindakan (K16), bukan normal maupun malicious.

Reroute minimal (K11): untuk setiap pasangan (node c, sink t) dengan c tidak terisolasi,
pilih ulang aksi TLDQN dengan argmax Q atas kandidat yang SAH: bukan node terisolasi dan
bukan node yang (untuk t) sudah dinyatakan buntu. Kandidat honeycomb selalu lebih dekat
ke t, jadi pasangan diproses urut jarak naik dan hasilnya bebas loop. Q dihitung dari
observasi yang sama dengan ekspor tabel awal (diverifikasi: tanpa isolasi = tabel awal).
Bila tidak ada kandidat sah: entri DIBIARKAN (tidak ada jalur rekaan), dicatat sebagai
kegagalan rute; paketnya dibuang node terisolasi di NS-3.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np

import dto_detect
from telemetry_schema import BIGAN_INPUT_COLUMNS, check_no_leakage, load_links, paths

HERE = os.path.dirname(os.path.abspath(__file__))
FUSIONS = ("F2", "F1")
DECISION_COLUMNS = ["node_id", "dto_decision", "bigan_pred", "isolate", "reason"]
ROUTE_COLUMNS = ["node_id", "dest_sink", "next_hop", "next_hop_is_sink", "slot",
                 "n_candidates", "own_dest", "q_value", "node_malicious"]
T_IDS = 8.3            # waktu tindakan IDS di NS-3 (kode rilis penulis; K12)
T_OBS = 7.3            # akhir window observasi (K12): T_obs + wdLate <= T_ids


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# --------------------------------------------------------------------------- Bi-GAN
def load_bigan_pred(path, n):
    """pred_intrusion per sensor; kolom lain (termasuk label) tidak dibaca."""
    pred = np.full(n, -1, dtype=int)
    with open(path) as f:
        for row in csv.DictReader(f):
            r = {c: row[c] for c in BIGAN_INPUT_COLUMNS}
            i, p = int(r["node_id"]), int(r["pred_intrusion"])
            if not 0 <= i < n or p not in (0, 1) or pred[i] != -1:
                raise ValueError(f"{path}: baris tidak valid {r}")
            pred[i] = p
    if (pred < 0).any():
        raise ValueError(f"{path}: pred_intrusion tidak lengkap untuk 0..{n - 1}")
    return pred.astype(bool)


def load_pilot_pred(path, n, seed, telemetry_paths):
    with open(path, newline="") as stream:
        if csv.DictReader(stream).fieldnames != ["node_id", "score", "threshold", "pred_intrusion"]:
            raise ValueError(f"{path}: header prediksi pilot tidak dikenal")
    with open(path + ".meta.json") as stream:
        meta = json.load(stream)
    if (meta.get("nodes") != n or meta.get("seed") != seed or meta.get("window") != 0 or
            meta.get("contains_ground_truth") is not False):
        raise ValueError(f"{path}: metadata prediksi pilot tidak sesuai")
    hashes = meta.get("source_features_sha256", {})
    for key in ("node", "links"):
        if hashes.get(key) != sha256(telemetry_paths[key]):
            raise ValueError(f"{path}: telemetry {key} berbeda dari sumber prediksi")
    return load_bigan_pred(path, n)


# --------------------------------------------------------------------------- fusi
def fuse(dto_decision, bigan_pred, fusion):
    """(isolate, reason). insufficient_evidence tidak pernah memicu isolasi oleh DTO."""
    if fusion not in FUSIONS:
        raise ValueError(fusion)
    dto = dto_decision == "malicious"
    big = bool(bigan_pred)
    if fusion == "F2":
        return dto, ("dto" if dto else "none")
    iso = dto or big
    reason = "dto+bigan" if dto and big else "dto" if dto else "bigan" if big else "none"
    return iso, reason


def dto_seed_for(n, s, dto_seed=0):
    """Sama dengan dto_detect.run_dir untuk varian main."""
    return dto_seed * 1_000_003 + n * 101 + s * 7 + dto_detect.VARIANTS.index("main")


def decide(links, bigan_pred, n, s, fusion, windows=frozenset({0}), b_max=None, dto_seed=0):
    """Keputusan hybrid per sensor. DTO langkah 3 beku (varian main), hanya `windows`."""
    ev = dto_detect.evidence(links, n, set(windows), b_max)
    rows, info = dto_detect.detect(ev, n, "main", seed=dto_seed_for(n, s, dto_seed))
    out = []
    for r in rows:
        j = r["node_id"]
        iso, reason = fuse(r["decision"], bigan_pred[j], fusion)
        out.append({"node_id": j, "dto_decision": r["decision"],
                    "bigan_pred": int(bigan_pred[j]), "isolate": int(iso), "reason": reason})
    info = {"dto_steps": len(info["steps"]), "dto_q0": info["q0_baseline_plr"],
            "dto_candidates": info["candidates"],
            "excluded_observers": ev["excluded_observers"],
            "counts": {k: sum(1 for x in out if x["dto_decision"] == k)
                       for k in ("malicious", "normal", "insufficient_evidence")},
            "bigan_alarms": int(sum(x["bigan_pred"] for x in out)),
            "isolated": [x["node_id"] for x in out if x["isolate"]]}
    return out, info


# --------------------------------------------------------------------------- reroute
def load_table(path):
    with open(path) as f:
        rd = csv.DictReader(f)
        if rd.fieldnames != ROUTE_COLUMNS:
            raise ValueError(f"{path}: header tabel rute tidak dikenal")
        rows = []
        for r in rd:
            rows.append({k: (float(v) if k == "q_value" else int(v)) for k, v in r.items()})
    return rows


def _q_values(env, qnet, c, t):
    import torch
    env.cur, env.dest = c, t
    obs, mask = env.observe(), env.action_mask()
    with torch.no_grad():
        q = qnet(torch.as_tensor(obs[None], dtype=torch.float32))[0].numpy()
    return q, mask


def reroute(table_rows, env, qnet, isolated, q_tol=5e-6):
    """Tabel setelah isolasi + laporan. Tidak membuat jalur di luar kandidat TLDQN."""
    topo = env.topo
    n = topo.n
    iso = set(int(i) for i in isolated)
    pre = {(r["node_id"], r["dest_sink"] + n): r for r in table_rows}
    if set(pre) != set(topo.reachable_pairs()):
        raise ValueError("tabel rute tidak cocok dengan pasangan terjangkau topologi")
    out, dead, changed, failed = {}, set(), [], []
    order = sorted(pre, key=lambda p: (p[1], topo.dist[p[0], p[1]], p[0]))
    for c, t in order:
        r = pre[(c, t)]
        q, mask = _q_values(env, qnet, c, t)
        sl = topo.slots(c, t)
        a0 = int(np.argmax(np.where(mask, q, -1e9)))
        j0 = int(sl[a0])
        orig = (n + r["next_hop"]) if r["next_hop_is_sink"] else r["next_hop"]
        if a0 != r["slot"] or j0 != orig or abs(float(q[a0]) - r["q_value"]) > q_tol:
            raise RuntimeError(f"tidak bisa mereproduksi tabel TLDQN pada ({c}, {t - n}): "
                               f"slot {a0} vs {r['slot']}, q {q[a0]:.6f} vs {r['q_value']:.6f}")
        if c in iso:
            out[(c, t)] = dict(r)                    # node terisolasi tidak meneruskan
            continue
        ok = mask.copy()
        for k, j in enumerate(sl):
            if j >= 0 and not topo.is_sink(int(j)) and (int(j) in iso or (int(j), t) in dead):
                ok[k] = False
        if not ok.any():
            dead.add((c, t))
            failed.append({"node_id": c, "dest_sink": t - n,
                           "next_hop_kept": r["next_hop"],
                           "next_hop_is_sink": r["next_hop_is_sink"]})
            out[(c, t)] = dict(r)                    # tidak ada jalur rekaan
            continue
        a = int(np.argmax(np.where(ok, q, -1e9)))
        j = int(sl[a])
        is_sink = topo.is_sink(j)
        nr = dict(r)
        nr.update(next_hop=(j - n) if is_sink else j, next_hop_is_sink=int(is_sink),
                  slot=a, q_value=float(q[a]))
        if (nr["next_hop"], nr["next_hop_is_sink"]) != (r["next_hop"], r["next_hop_is_sink"]):
            changed.append({"node_id": c, "dest_sink": t - n, "old_next_hop": r["next_hop"],
                            "old_is_sink": r["next_hop_is_sink"], "new_next_hop": nr["next_hop"],
                            "new_is_sink": nr["next_hop_is_sink"]})
        out[(c, t)] = nr
    rows = [out[k] for k in sorted(out)]
    paths_ = path_status(rows, n, iso, dead, pre)
    return rows, {"changed_entries": changed, "failed_entries": failed, "paths": paths_}


def _follow(table, n, src, t):
    cur, path = src, [src]
    for _ in range(n + 2):
        r = table[(cur, t)]
        nxt = n + r["next_hop"] if r["next_hop_is_sink"] else r["next_hop"]
        path.append(nxt)
        if r["next_hop_is_sink"]:
            return path
        if nxt in path[:-1]:
            raise ValueError(f"loop dari {src}: {path}")
        cur = nxt
    raise ValueError(f"tidak mencapai sink dari {src}")


def path_status(rows, n, iso, dead, pre):
    """Status jalur tiap sumber tidak terisolasi ke sink tujuannya."""
    table = {(r["node_id"], r["dest_sink"] + n): r for r in rows}
    out = []
    for s in range(n):
        if s in iso:
            continue
        t = next(k[1] for k, r in pre.items() if k[0] == s and r["own_dest"])
        before = _follow(pre, n, s, t)
        after = _follow(table, n, s, t)
        inner = after[1:-1]
        blocked_at = next((v for v in inner if v in iso), None)
        out.append({"source": s, "status": "failed" if blocked_at is not None else "ok",
                    "blocked_at": blocked_at, "changed": before != after,
                    "hops_before": len(before) - 1, "hops_after": len(after) - 1,
                    "before_via_isolated": any(v in iso for v in before[1:-1])})
    return out


def write_rows(path, rows, columns):
    with open(path, "w", newline="") as f:
        # LF saja: loader C++ membandingkan header persis (std::getline menyisakan '\r')
        w = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        w.writeheader()
        for r in rows:
            rr = dict(r)
            if "q_value" in rr:
                rr["q_value"] = f"{rr['q_value']:.6f}"
            w.writerow(rr)


# --------------------------------------------------------------------------- CLI
def load_route_env(route_dir, n, s, trust_dir, cfg):
    """Bangun lingkungan TLDQN persis seperti saat tabel di route_dir diekspor."""
    from export_routes import build_env
    with open(os.path.join(route_dir, "tldqn_nexthop_meta.json")) as stream:
        meta = json.load(stream)
    sc = next(x for x in meta["scenarios"] if x["n"] == n and x["seed"] == s)
    ts = meta["trust_source"]
    st = None
    if ts == "telemetry":
        from telemetry_trust import load_trust
        tf = os.path.join(trust_dir, sc["trust_file"])
        if sha256(tf) != sc["trust_sha256"]:
            raise ValueError(f"{tf}: SHA-256 tidak sama dengan saat ekspor rute")
        st = load_trust(tf, n)
    elif ts != "none":
        raise ValueError(f"trust_source {ts} tidak didukung untuk reroute (butuh label)")
    # malicious = nol: observasi TLDQN tidak memakainya (hanya kolom ekspor, disalin)
    env = build_env(n, s, np.zeros(n, dtype=bool), cfg, ts, st)
    return env, meta, sc


def main(argv=None):
    ap = argparse.ArgumentParser(description="Keputusan Hybrid IDS + reroute (langkah 4)")
    ap.add_argument("--telem-dir", required=True, help="telemetry run A (idsSource=none)")
    ap.add_argument("--scenario-dir", required=True, help="folder bigan_decisions_*")
    ap.add_argument("--bigan-pred-dir", default=None,
                    help="folder bigan_pilot_predictions_* (tanpa label; opt-in)")
    ap.add_argument("--route-dir", required=True, help="tabel TLDQN yang dipakai run A/B")
    ap.add_argument("--trust-dir", default=None, help="trust tahap-1 (rute trust telemetry)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--nodes", type=int, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--fusion", choices=FUSIONS, default="F2")
    ap.add_argument("--b-max", type=float, default=0.0,
                    help="0 = pra-K9 (UTAMA, K15); 0.8 = varian pembanding K9")
    ap.add_argument("--t-obs", type=float, default=T_OBS)
    ap.add_argument("--t-ids", type=float, default=T_IDS)
    a = ap.parse_args(argv)
    b_max = a.b_max if a.b_max > 0 else None
    if a.bigan_pred_dir and (abs(a.t_obs - T_OBS) > 1e-9 or abs(a.t_ids - T_IDS) > 1e-9):
        ap.error("prediksi pilot membutuhkan T_obs=7.3 dan T_ids=8.3")
    if a.bigan_pred_dir and os.path.exists(a.out_dir):
        raise FileExistsError(f"direktori output pilot sudah ada: {a.out_dir}")

    import torch
    from export_routes import DEFAULT_MODEL
    from routing_env import EnvConfig, STATE_DIM
    from honeycomb_topology import K_ACTIONS
    from tldqn_routing import DQNConfig, QNet

    if not a.bigan_pred_dir:
        os.makedirs(a.out_dir, exist_ok=True)
    assert check_no_leakage(DECISION_COLUMNS) == []
    cfg = EnvConfig()
    d = DQNConfig()
    meta_out = {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "design": "docs/STEP4_HYBRID_IDS_DESIGN.md", "fusion": a.fusion,
                "fusion_note": {"F2": "utama: DTO mengisolasi, Bi-GAN alarm saja",
                                "F1": "EKSPERIMENTAL (tidak disebut paper): DTO ATAU Bi-GAN"}[a.fusion],
                "dto": {"variant": "main", "b_max_observer": b_max,
                        "method": "pra-K9 (utama)" if b_max is None else "K9 (pembanding)",
                        "windows": [0], "config": "beku langkah 3 (m=30,Y=100,N_min=10,"
                                                  "delta=0.1,alpha=0.01)"},
                "bigan_note": ("prediksi pilot dari telemetry NS-3; hash input diverifikasi"
                               if a.bigan_pred_dir else
                               "prediksi dari sampel NSL-KDD yang ditugaskan berdasarkan label "
                               "skenario; bukan deteksi independen trafik NS-3 (K14)"),
                "t_obs": a.t_obs, "t_ids": a.t_ids, "telem_dir": a.telem_dir,
                "route_dir": a.route_dir, "scenarios": {}}
    qnet_cache = {}
    for n in a.nodes:
        for s in a.seeds:
            p = paths(a.telem_dir, n, s)
            with open(p["meta"]) as stream:
                tmeta = json.load(stream)
            if a.bigan_pred_dir and (tmeta.get("num_nodes") != n or tmeta.get("seed") != s):
                raise ValueError(f"{p['meta']}: skenario telemetry tidak sesuai")
            if a.bigan_pred_dir and tmeta.get("sim_time_s", 0) < a.t_ids:
                raise ValueError(f"{p['meta']}: simulasi berhenti sebelum T_ids")
            if abs(tmeta["window_s"] - a.t_obs) > 1e-9:
                raise SystemExit(f"telemetry window_s={tmeta['window_s']} != T_obs={a.t_obs}: "
                                 "jalankan run A dgn --telemWindow=T_obs")
            if a.t_obs + tmeta["wd_late_s"] > a.t_ids + 1e-9:
                raise SystemExit("T_obs + wdLate > T_ids: observasi belum lengkap saat T_ids")
            bpath = (os.path.join(a.bigan_pred_dir, f"bigan_pilot_predictions_{n}_s{s}.csv")
                     if a.bigan_pred_dir else
                     os.path.join(a.scenario_dir, f"bigan_decisions_{n}_s{s}.csv"))
            predictions = (load_pilot_pred(bpath, n, s, p) if a.bigan_pred_dir else
                           load_bigan_pred(bpath, n))
            rows, info = decide(load_links(p["links"]), predictions, n, s,
                                a.fusion, {0}, b_max)
            tag = f"{n}_s{s}"
            dpath = os.path.join(a.out_dir, f"hybrid_decisions_{tag}.csv")

            env, rmeta, sc = load_route_env(a.route_dir, n, s, a.trust_dir, cfg)
            model = os.path.join(HERE, rmeta["model"]) if rmeta.get("model") else DEFAULT_MODEL
            if sha256(model) != rmeta["model_sha256"]:
                raise ValueError(f"{model}: SHA-256 model berbeda dari saat ekspor rute")
            if model not in qnet_cache:
                q = QNet(STATE_DIM, K_ACTIONS, d.hidden, d.n_hidden)
                q.load_state_dict(torch.load(model, map_location="cpu"))
                q.eval()
                qnet_cache[model] = q
            tpath = os.path.join(a.route_dir, f"tldqn_nexthop_{tag}.csv")
            if sha256(tpath) != sc["table_sha256"]:
                raise ValueError(f"{tpath}: SHA-256 tabel berbeda dari meta ekspor")
            new_rows, rep = reroute(load_table(tpath), env, qnet_cache[model], info["isolated"])
            rpath = os.path.join(a.out_dir, f"reroute_nexthop_{tag}.csv")
            os.makedirs(a.out_dir, exist_ok=True)
            write_rows(dpath, rows, DECISION_COLUMNS)
            write_rows(rpath, new_rows, ROUTE_COLUMNS)
            fails = [x for x in rep["paths"] if x["status"] == "failed"]
            meta_out["scenarios"][tag] = {
                **info, "telemetry_links_sha256": sha256(p["links"]),
                "telemetry_node_sha256": sha256(p["node"]) if a.bigan_pred_dir else None,
                "telemetry_meta_sha256": sha256(p["meta"]),
                "bigan_file": os.path.basename(bpath), "bigan_sha256": sha256(bpath),
                "bigan_metadata_sha256": (sha256(bpath + ".meta.json")
                                          if a.bigan_pred_dir else None),
                "route_table_sha256": sha256(tpath), "decisions_sha256": sha256(dpath),
                "reroute_sha256": sha256(rpath), "reroute": rep}
            print(f"n={n:3d} s={s}: DTO {info['counts']} | alarm Bi-GAN {info['bigan_alarms']} "
                  f"| isolasi {len(info['isolated'])} ({a.fusion}) | reroute diganti "
                  f"{len(rep['changed_entries'])}, entri gagal {len(rep['failed_entries'])}, "
                  f"jalur gagal {len(fails)}")
    with open(os.path.join(a.out_dir, "hybrid_meta.json"), "w") as f:
        json.dump(meta_out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())

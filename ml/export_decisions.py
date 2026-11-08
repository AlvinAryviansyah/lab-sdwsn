#!/usr/bin/env python3
"""
export_decisions.py
-------------------
Jembatan Python (AI) -> ns-3 (jaringan).

Sesuai paper (Sec. III-D & IV-B), keputusan AI diambil di controller/edge:
  - Bi-GAN  : di secondary controller -> klasifikasi normal/intrusi per node.
  - TLDQN   : di route agent (edge server) -> pilih sink/forwarder per node.
Script ini melatih kedua model SUNGGUHAN (sekali, lalu disimpan ke
ml/checkpoints/) dan menulis keputusan per node ke CSV yang dibaca
honeycomb-sdwsn.cc saat simulasi.

Ground truth (10% node malicious) ditentukan oleh SKENARIO (seed), bukan oleh
DTO, sehingga confusion matrix tidak circular. Tiap seed = skenario berbeda
(node malicious & sampel trafik berbeda) dengan MODEL yang sama, sehingga
ulangan simulasi memberi rata-rata dan simpangan baku yang bermakna.

Pemetaan node -> trafik: tiap node diberi satu sampel NSL-KDD (label sesuai
ground-truth node: malicious -> sampel serangan, normal -> sampel normal).
Ini penugasan fitur, bukan fitur yang diekstrak dari trafik ns-3 (batasan
yang harus disebut di laporan).

Output (di ns3-sim/results/, lalu salin ke <ns-3>/results/):
  bigan_decisions_<n>_s<seed>.csv : node_id,actual_mal,anomaly_score,threshold,pred_intrusion
  bigan_windows_<n>_s<seed>.csv   : node_id,actual_mal,p0..p{W-1}  (prediksi per window
                                    utk metrik time-based; p0 = keputusan awal)
  tldqn_routes_<n>_s<seed>.csv    : node_id,sink_idx,q_best
  ai_timing.csv                   : n,bigan_ms_per_node,tldqn_ms_per_node
                                    waktu per KEPUTUSAN (batch=1), median banyak
                                    ulangan setelah warm-up; tidak bergantung n

Pemakaian:
  python export_decisions.py                 # latih model + ekspor 5 seed
  python export_decisions.py --reuse         # pakai model tersimpan (cepat)
  python export_decisions.py --seeds 10
"""

import os
import re
import time
import argparse
import numpy as np
import torch

import legacy_tldqn as tl   # LEGACY sampai fase integrasi NS-3
from bigan_ids import (train_bigan, anomaly_score, Generator, Encoder,
                       Discriminator)
from nslkdd_data import load

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "..", "ns3-sim", "results")
CKPT_DIR = os.path.join(HERE, "checkpoints")
CKPT = os.path.join(CKPT_DIR, "ai_models.pt")
CC_FILE = os.path.join(HERE, "..", "ns3-sim", "honeycomb-sdwsn.cc")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

NODES = [20, 30, 40, 50, 60, 70, 80, 90, 100]
N_SINKS = 4
TX_RANGE = 150.0      # m (Tabel 5)
AREA_DIAG = 1000.0    # normalisasi jarak


def parse_positions():
    """Ambil koordinat sensor & sink dari honeycomb-sdwsn.cc (satu sumber)."""
    src = open(CC_FILE, encoding="utf-8").read()

    def block(name):
        m = re.search(name + r"\[[^\]]*\]\[2\]\s*=\s*\{(.*?)\};", src, re.S)
        pairs = re.findall(r"\{\s*([\d.]+)\s*,\s*([\d.]+)\s*\}", m.group(1))
        return np.array(pairs, dtype=np.float64)

    return block("g_sensorPos"), block("g_sinkPos")


def best_threshold(err_tr, ytr, prevalence=0.10):
    """Ambang yang memaksimalkan F1 pada train, dgn bobot kelas agar
    prevalensi serangan = `prevalence` (skenario paper: 10% node malicious).
    Hanya memakai data train (tanpa kebocoran dari test)."""
    n_norm = int((ytr == 0).sum())
    n_att = int((ytr == 1).sum())
    w_att = (prevalence / (1 - prevalence)) * n_norm / max(n_att, 1)
    best_t, best_f = None, -1.0
    for q in np.linspace(50, 99.9, 120):
        t = np.percentile(err_tr[ytr == 0], q)
        pred = err_tr > t
        tp = w_att * int((pred & (ytr == 1)).sum())
        fp = int((pred & (ytr == 0)).sum())
        fn = w_att * int((~pred & (ytr == 1)).sum())
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        f = 2 * p * r / (p + r) if p + r else 0
        if f > best_f:
            best_f, best_t = f, t
    return float(best_t), best_f


def route_states(sensor_xy, sink_xy, rng):
    """State routing per node: (N, K, 8) -> [trust,dist,lq,energy,hops,stab,pdr,thr]."""
    n = len(sensor_xy)
    S = np.zeros((n, N_SINKS, tl.STATE_DIM), dtype=np.float32)
    for i in range(n):
        for k in range(N_SINKS):
            d_m = float(np.hypot(*(sensor_xy[i] - sink_xy[k])))
            d = min(d_m / AREA_DIAG, 1.0)
            hops = min(np.ceil(d_m / TX_RANGE) / 7.0, 1.0)
            lq = float(np.clip(1.0 - d + rng.normal(0, 0.05), 0, 1))
            S[i, k] = [
                0.6 + 0.4 * rng.random(),                    # trust
                d,                                           # distance
                lq,                                          # link quality
                float(np.clip(0.3 * d + 0.1 * rng.random(), 0, 1)),  # energy cost
                hops,                                        # hops
                float(np.clip(0.5 + 0.5 * lq * rng.random() + 0.1, 0, 1)),  # stability
                0.7 + 0.3 * lq,                              # pdr
                lq * (0.8 + 0.2 * rng.random()),             # throughput
            ]
    return S


def reward_of(f):
    trust, dist, lq, energy, hops, stab, pdr, thr = f
    return (trust + stab + pdr + thr) - 0.5 * (dist + hops) - 0.2 * energy


def median_ms(fn, reps=30):
    """Waktu median (ms) setelah warm-up, supaya tidak terkena efek cold-start."""
    for _ in range(3):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


@torch.no_grad()
def disc_score(E, D, X, bs=8192):
    """Skor anomali dari Discriminator pada pasangan (x, E(x)) (Fig. 6 paper:
    Encoder -> Concatenate -> Discriminator -> Normal/Malicious).
    D dilatih membedakan pasangan nyata (x, E(x)) vs palsu (G(z), z). Pada trafik
    normal (distribusi latih) logit D tinggi; trafik asing -> logit rendah.
    Skor = -logit (makin tinggi makin anomali)."""
    E.eval(); D.eval()
    out = []
    for i in range(0, len(X), bs):
        x = torch.tensor(X[i:i + bs], dtype=torch.float32)
        out.append((-D(x, E(x)).squeeze(1)).numpy())
    return np.concatenate(out)


def detector_score(G, E, D, X, kind):
    """kind: 'disc' (default, setia ke Fig. 6 paper) atau 'recon' (||x-G(E(x))||)."""
    if kind == "recon":
        return anomaly_score(G, E, X)
    return disc_score(E, D, X)


def get_models(args, Xtr, ytr, x_dim):
    """Siapkan Bi-GAN (G,E,D) dan TLDQN. Mengembalikan G, E, D, thr, q_net."""
    tl.K_ACTIONS = N_SINKS          # aksi = pilih salah satu dari 4 edge sink
    ck = torch.load(CKPT, map_location="cpu") if (args.reuse and os.path.exists(CKPT)) else None

    # ---------------- Bi-GAN ----------------
    G = E = D = None
    bg = None
    if args.bigan_ckpt and os.path.exists(args.bigan_ckpt):
        bg = torch.load(args.bigan_ckpt, map_location="cpu")
        print(f"Memuat Bi-GAN dari {args.bigan_ckpt}")
    elif ck is not None and "D" in ck:
        bg = ck
        print(f"Memuat Bi-GAN dari {os.path.relpath(CKPT)}")
    if bg is not None:
        zd = bg["zdim"]
        G = Generator(zd, x_dim); E = Encoder(x_dim, zd); D = Discriminator(x_dim, zd)
        G.load_state_dict(bg["G"]); E.load_state_dict(bg["E"]); D.load_state_dict(bg["D"])
        args.zdim = zd
    else:
        Xn = Xtr[ytr == 0]
        print(f"Latih Bi-GAN pada {len(Xn)} sampel normal (fitur={x_dim}) ...")
        G, E, D = train_bigan(Xn, x_dim, z_dim=args.zdim, epochs=args.epochs)

    # Ambang dikalibrasi di TRAIN (bobot prevalensi 10%), sesuai detektor terpilih.
    s_tr = detector_score(G, E, D, Xtr, args.detector)
    thr, f1_tr = best_threshold(s_tr, ytr)
    print(f"Detektor '{args.detector}': ambang = {thr:.3f} "
          f"(F1 train terbobot prevalensi 10%: {f1_tr*100:.1f}%)")

    # ---------------- TLDQN ----------------
    q_net = tl.QNet(tl.STATE_DIM * N_SINKS, N_SINKS)
    if ck is not None and "Q" in ck:
        print("Memuat TLDQN tersimpan")
        q_net.load_state_dict(ck["Q"])
    else:
        print("Pretrain TLDQN (task sumber) ...")
        src_net, _ = tl.train_dqn(tl.RoutingEnv(seed=42),
                                  episodes=args.rl_episodes // 2)
        print("Latih TLDQN (transfer learning) ...")
        q_net, _ = tl.train_dqn(tl.RoutingEnv(seed=7), episodes=args.rl_episodes,
                                init_weights=src_net.state_dict())
    q_net.eval()

    torch.save({"G": G.state_dict(), "E": E.state_dict(), "D": D.state_dict(),
                "thr": thr, "zdim": args.zdim, "Q": q_net.state_dict(),
                "detector": args.detector}, CKPT)
    print(f"Model disimpan: {os.path.relpath(CKPT)}")
    return G, E, D, thr, q_net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20, help="epoch Bi-GAN")
    ap.add_argument("--zdim", type=int, default=32)
    ap.add_argument("--rl-episodes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1, help="seed pelatihan model")
    ap.add_argument("--seeds", type=int, default=5,
                    help="jumlah seed skenario (1..N) untuk ulangan simulasi")
    ap.add_argument("--windows", type=int, default=20,
                    help="jumlah window monitoring IDS (time-based metrics)")
    ap.add_argument("--detector", choices=["disc", "recon"], default="disc",
                    help="skor deteksi: disc = Discriminator (Fig. 6 paper, default); "
                         "recon = reconstruction error")
    ap.add_argument("--bigan-ckpt", default="",
                    help="muat G,E,D dari file (mis. hasil detector_eval.py --save)")
    ap.add_argument("--reuse", action="store_true",
                    help="pakai model tersimpan di ml/checkpoints (tanpa latih)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    sensors, sinks = parse_positions()
    assert len(sinks) == N_SINKS and len(sensors) >= max(NODES), \
        "koordinat tidak terbaca dari honeycomb-sdwsn.cc"

    print("Memuat NSL-KDD ...")
    Xtr, ytr, Xte, yte, x_dim = load()
    G, E, D, thr, q_net = get_models(args, Xtr, ytr, x_dim)

    idx_norm = np.where(yte == 0)[0]
    idx_att = np.where(yte == 1)[0]

    timing = {}
    for seed in range(1, args.seeds + 1):
        print(f"\n=== Seed skenario {seed} ===")
        for n in NODES:
            rng = np.random.default_rng(seed * 100000 + n)
            n_mal = max(1, n // 10)                   # 10% malicious (paper)
            mal = set(rng.choice(n, n_mal, replace=False).tolist())
            actual = np.array([1 if i in mal else 0 for i in range(n)])

            # Tiap node -> satu sampel NSL-KDD sesuai ground truth node.
            pick = np.array([rng.choice(idx_att) if a else rng.choice(idx_norm)
                             for a in actual])
            X = Xte[pick]
            score = detector_score(G, E, D, X, args.detector)
            pred = (score > thr).astype(int)

            # Window monitoring: tiap window, tiap node mendapat sampel baru
            # (label sesuai ground truth node). Window 0 = keputusan awal.
            preds_w = [pred]
            for _w in range(1, args.windows):
                pick_w = np.array([rng.choice(idx_att) if a else rng.choice(idx_norm)
                                   for a in actual])
                preds_w.append((detector_score(G, E, D, Xte[pick_w], args.detector)
                                > thr).astype(int))
            with open(os.path.join(RESULTS_DIR,
                                   f"bigan_windows_{n}_s{seed}.csv"), "w") as f:
                f.write("node_id,actual_mal," +
                        ",".join(f"p{w}" for w in range(args.windows)) + "\n")
                for i in range(n):
                    f.write(f"{i},{actual[i]}," +
                            ",".join(str(int(pw[i])) for pw in preds_w) + "\n")

            with open(os.path.join(RESULTS_DIR,
                                   f"bigan_decisions_{n}_s{seed}.csv"), "w") as f:
                f.write("node_id,actual_mal,anomaly_score,threshold,pred_intrusion\n")
                for i in range(n):
                    f.write(f"{i},{actual[i]},{score[i]:.6f},{thr:.6f},{pred[i]}\n")

            # Routing: state per node -> argmax Q.
            S = route_states(sensors[:n], sinks, rng)
            Sflat = torch.tensor(S.reshape(n, -1))
            with torch.no_grad():
                q = q_net(Sflat).numpy()
            act = q.argmax(axis=1)
            oracle = np.array([np.argmax([reward_of(S[i, k])
                                          for k in range(N_SINKS)])
                               for i in range(n)])
            agree = float((act == oracle).mean()) * 100

            with open(os.path.join(RESULTS_DIR,
                                   f"tldqn_routes_{n}_s{seed}.csv"), "w") as f:
                f.write("node_id,sink_idx,q_best\n")
                for i in range(n):
                    f.write(f"{i},{int(act[i])},{q[i, act[i]]:.6f}\n")

            tp = int(((pred == 1) & (actual == 1)).sum())
            tn = int(((pred == 0) & (actual == 0)).sum())
            fp = int(((pred == 1) & (actual == 0)).sum())
            fn = int(((pred == 0) & (actual == 1)).sum())
            print(f"n={n:3d} mal={n_mal:2d} | BiGAN TP={tp} TN={tn} FP={fp} "
                  f"FN={fn} acc={(tp+tn)/n*100:.1f}% | TLDQN==oracle {agree:.0f}%")

            # Waktu per KEPUTUSAN (satu node, batch=1): median banyak ulangan
            # setelah warm-up. Tidak dibagi n, jadi tidak bergantung ukuran jaringan.
            if seed == 1:
                x1 = X[:1]
                s1 = Sflat[:1]

                def run_bigan():
                    detector_score(G, E, D, x1, args.detector)

                def run_tldqn():
                    with torch.no_grad():
                        q_net(s1)

                timing[n] = (median_ms(run_bigan, 200), median_ms(run_tldqn, 200))

    with open(os.path.join(RESULTS_DIR, "ai_timing.csv"), "w") as f:
        f.write("n,bigan_ms_per_decision,tldqn_ms_per_decision\n")
        for n in NODES:
            b, t = timing[n]
            f.write(f"{n},{b:.6f},{t:.6f}\n")
    print("\nSelesai. Salin ns3-sim/results/{bigan_decisions,bigan_windows,tldqn_routes}"
          "_*_s*.csv dan ai_timing.csv ke <ns-3>/results/")


if __name__ == "__main__":
    main()

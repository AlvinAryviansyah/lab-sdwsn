#!/usr/bin/env python3
"""
bigan_ids.py  --  PELATIHAN Bi-GAN
-----------------------------------
Melatih Bi-GAN (Generator, Encoder, Discriminator) mengikuti Algoritma 3 paper
Kipongo et al. (2024) dan menyimpan checkpoint + konfigurasi inferensi.
Inferensi ada di bigan_infer.py; evaluasi test ada di detector_eval.py.
Definisi model, kerugian, ambang, metrik, dan status verifikasi paper per butir:
lihat bigan_model.py dan docs/PAPER_DEVIATIONS.md.

Alur (tanpa kebocoran):
  1. make_splits(): KDDTrain+ -> fold train / validation (stratified, seed tetap);
     preprocessor di-fit hanya pada fold train (baris normal).
  2. Latih Bi-GAN SATU KELAS pada baris NORMAL fold train.
  3. Hitung skor anomali pada VALIDASI, pilih ambang pada validasi saja.
  4. Simpan: model.pt, inference_config.json, preprocessor/, splits.npz,
     train_log.json. KDDTest+ TIDAK disentuh di skrip ini.

Pemakaian:
  python bigan_ids.py --out checkpoints/bigan_v2 --epochs 20 --seed 1
"""

import os
import sys
import json
import math
import time
import argparse
import itertools

import numpy as np
import torch

from bigan_model import (BiGANConfig, build_models, d_loss, ge_loss_terms,
                         discriminator_logit, anomaly_from_logit, select_threshold,
                         binary_metrics, score_auc, validate_direction, SCORE_NAME,
                         SCORE_DIRECTION,
                         Generator, Encoder, Discriminator)   # noqa: F401 (re-export)
from nslkdd_data import make_splits

HERE = os.path.dirname(os.path.abspath(__file__))
FORMAT_VERSION = 1


# ---------------------------------------------------------------------------
# Satu langkah pelatihan (Alg. 3)
# ---------------------------------------------------------------------------
def d_step(G, E, D, opt_d, x, z):
    """Alg.3 b.5-16: perbarui D saja. Pasangan nyata (x,E(x)) vs palsu (G(z),z)."""
    with torch.no_grad():
        z_real = E(x)
        x_fake = G(z)
    loss = d_loss(D(x, z_real), D(x_fake, z))
    opt_d.zero_grad()
    loss.backward()
    opt_d.step()
    return float(loss.item())


def ge_step(G, E, D, opt_ge, x, z):
    """Alg.3 b.19-27: perbarui G dan E bersama dari batch baru. D tidak diubah."""
    z_real = E(x)
    x_fake = G(z)
    loss_g, loss_e = ge_loss_terms(D(x, z_real), D(x_fake, z))
    loss = loss_g + loss_e
    opt_ge.zero_grad()
    loss.backward()
    opt_ge.step()
    return float(loss_g.item()), float(loss_e.item())


def train(cfg, X_normal, device="cpu", log_every=500, log_fn=print):
    """Latih Bi-GAN pada trafik NORMAL. Mengembalikan (G, E, D, history).

    Satu 'iterasi' = Alg.3 b.2-27: K langkah D (batch baru tiap langkah) lalu
    satu langkah G/E (batch baru). N = epochs * ceil(n / batch_size).
    Batch diambil acak dengan pengembalian dari data (i.i.d. "sample a batch").
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    G, E, D = (m.to(device) for m in build_models(cfg))
    betas = (cfg.beta1, cfg.beta2)
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg.lr, betas=betas)
    opt_ge = torch.optim.Adam(itertools.chain(G.parameters(), E.parameters()),
                              lr=cfg.lr, betas=betas)

    X = torch.as_tensor(X_normal, dtype=torch.float32, device=device)
    n = X.shape[0]
    if n == 0:
        raise ValueError("data normal kosong")
    b = min(cfg.batch_size, n)
    n_iter = cfg.epochs * int(math.ceil(n / cfg.batch_size))
    gen = torch.Generator().manual_seed(cfg.seed)

    def sample_x():
        return X[torch.randint(0, n, (b,), generator=gen).to(device)]

    def sample_z():
        if cfg.latent_prior != "normal":
            raise ValueError("latent_prior hanya 'normal' yang didukung")
        return torch.randn(b, cfg.z_dim, generator=gen).to(device)

    history, acc_d, acc_g, acc_e, cnt = [], 0.0, 0.0, 0.0, 0
    G.train(); E.train(); D.train()
    for it in range(1, n_iter + 1):
        for _ in range(cfg.d_steps):                       # K langkah D
            ld = d_step(G, E, D, opt_d, sample_x(), sample_z())
        lg, le = ge_step(G, E, D, opt_ge, sample_x(), sample_z())   # G dan E
        acc_d += ld; acc_g += lg; acc_e += le; cnt += 1
        if it % log_every == 0 or it == n_iter:
            rec = {"iteration": it, "loss_D": acc_d / cnt, "loss_G": acc_g / cnt,
                   "loss_E": acc_e / cnt}
            history.append(rec)
            log_fn(f"  iter {it:6d}/{n_iter}  D={rec['loss_D']:.3f}  "
                   f"G={rec['loss_G']:.3f}  E={rec['loss_E']:.3f}")
            acc_d = acc_g = acc_e = 0.0; cnt = 0
    G.eval(); E.eval(); D.eval()
    return G, E, D, history


# ---------------------------------------------------------------------------
# Checkpoint + konfigurasi inferensi
# ---------------------------------------------------------------------------
def save_checkpoint(out_dir, cfg, G, E, D, splits, thr_info, val_metrics,
                    val_auc, history, scaler_fit, extra=None):
    os.makedirs(out_dir, exist_ok=True)
    torch.save({"G": G.state_dict(), "E": E.state_dict(), "D": D.state_dict()},
               os.path.join(out_dir, "model.pt"))
    splits.preproc.save(os.path.join(out_dir, "preprocessor"))
    np.savez(os.path.join(out_dir, "splits.npz"),
             train_idx=splits.train_idx, val_idx=splits.val_idx,
             seed=splits.seed, val_fraction=splits.val_fraction)
    config = {
        "format_version": FORMAT_VERSION,
        "model": cfg.to_dict(),
        "score": {"name": SCORE_NAME, "direction": SCORE_DIRECTION,
                  "definition": "-logit D(x, E(x)); malicious jika skor > ambang"},
        "threshold": thr_info,
        "preprocessing": {
            "dataset": "NSL-KDD",
            "schema_hash": splits.preproc.schema_hash,
            "n_features": splits.preproc.n_features,
            "scaler_fit": scaler_fit,
        },
        "data_split": {
            "seed": splits.seed, "val_fraction": splits.val_fraction,
            "n_train": int(len(splits.train_idx)),
            "n_train_normal": int((splits.y_train == 0).sum()),
            "n_val": int(len(splits.val_idx)),
        },
        "validation": {"auc": val_auc, "direction_ok": bool(val_auc > 0.5),
                       "metrics_at_threshold": val_metrics},
        "training": {"torch_version": torch.__version__,
                     "created_at": time.strftime("%Y-%m-%dT%H:%M:%S")},
    }
    if extra:
        config["training"].update(extra)
    with open(os.path.join(out_dir, "inference_config.json"), "w") as f:
        json.dump(config, f, indent=1)
    with open(os.path.join(out_dir, "train_log.json"), "w") as f:
        json.dump(history, f, indent=1)
    return config


# ---------------------------------------------------------------------------
# Shim kompatibilitas (dipakai export_decisions.py sampai direfaktor)
# ---------------------------------------------------------------------------
def train_bigan(Xn, x_dim, z_dim=32, epochs=20, batch=256, lr=2e-4,
                device="cpu", n_hidden=2):
    """LEGACY. Kembalikan (G, E, D). Default n_hidden=2 mempertahankan perilaku
    lama; pelatihan baru memakai train() + BiGANConfig (n_hidden=1)."""
    cfg = BiGANConfig(x_dim=x_dim, z_dim=z_dim, epochs=epochs, batch_size=batch,
                      lr=lr, n_hidden=n_hidden)
    G, E, D, _ = train(cfg, Xn, device=device, log_every=10 ** 9, log_fn=lambda *_: None)
    return G, E, D


@torch.no_grad()
def anomaly_score(G, E, X, device="cpu"):
    """LEGACY / diagnostik. ||x - G(E(x))||. BUKAN skor paper (Fig. 6)."""
    G.eval(); E.eval()
    x = torch.as_tensor(X, dtype=torch.float32, device=device)
    return torch.sqrt(((x - G(E(x))) ** 2).sum(dim=1)).cpu().numpy()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="Latih Bi-GAN (NSL-KDD) dan simpan checkpoint.")
    ap.add_argument("--out", default=os.path.join(HERE, "checkpoints", "bigan_v2"))
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--zdim", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--n-hidden", type=int, default=1, help="paper: 1 (pers. 32-34)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--d-steps", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--scaler-fit", choices=["normal", "train"], default="normal")
    ap.add_argument("--threshold-policy", choices=["f1_weighted", "fpr"],
                    default="f1_weighted")
    ap.add_argument("--prevalence", type=float, default=0.10,
                    help="prevalensi serangan skenario (paper: 10%% node)")
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--allow-inverted-direction", action="store_true",
                    help="tetap simpan checkpoint walau AUC validasi <= 0.5 (diagnostik)")
    args = ap.parse_args(argv)

    print("Menyiapkan split (train/validation/test; test tidak dipakai di sini) ...")
    sp = make_splits(seed=args.seed, val_fraction=args.val_fraction,
                     scaler_fit=args.scaler_fit)
    print(f"  fitur={sp.n_features}  train={len(sp.X_train)} "
          f"(normal={len(sp.X_train_normal)})  val={len(sp.X_val)}")

    cfg = BiGANConfig(x_dim=sp.n_features, z_dim=args.zdim, hidden=args.hidden,
                      n_hidden=args.n_hidden, lr=args.lr, batch_size=args.batch,
                      epochs=args.epochs, d_steps=args.d_steps, seed=args.seed)
    t0 = time.time()
    print(f"Latih Bi-GAN satu kelas pada {len(sp.X_train_normal)} sampel normal ...")
    G, E, D, hist = train(cfg, sp.X_train_normal, device=args.device)
    train_seconds = time.time() - t0

    logit_val = discriminator_logit(E, D, sp.X_val, device=args.device)
    score_val = anomaly_from_logit(logit_val)
    direction = validate_direction(score_val, sp.y_val)
    val_auc = direction["auc"]
    if not direction["ok"]:
        print(f"GAGAL: AUC validasi {val_auc:.3f} <= 0.5: arah skor terbalik atau model "
              "tidak belajar. Skor TIDAK dibalik otomatis.")
        if not args.allow_inverted_direction:
            return 2
    thr = select_threshold(score_val, sp.y_val, policy=args.threshold_policy,
                           prevalence=args.prevalence, target_fpr=args.target_fpr)
    val_metrics = binary_metrics(sp.y_val, score_val > thr["value"])
    val_metrics = {k: (float(v) if not isinstance(v, int) else v)
                   for k, v in val_metrics.items()}
    save_checkpoint(args.out, cfg, G, E, D, sp, thr, val_metrics, val_auc, hist,
                    args.scaler_fit, extra={"train_seconds": round(train_seconds, 1)})
    print(f"\nValidasi: AUC={val_auc:.3f}  ambang={thr['value']:.3f} "
          f"({thr['policy']}, prevalensi {thr['prevalence']})")
    print(f"  TPR={val_metrics['TPR']*100:.1f}%  FPR={val_metrics['FPR']*100:.1f}%  "
          f"F1={val_metrics['F1']*100:.1f}% (prevalensi data validasi)")
    print(f"Checkpoint -> {os.path.relpath(args.out)}  (evaluasi test: detector_eval.py)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

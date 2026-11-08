#!/usr/bin/env python3
"""
archive_experiments.py
----------------------
Mengarsipkan hasil eksperimen fase Bi-GAN dan TLDQN ke ml/experiments/<nama>/
beserta MANIFEST.json (perintah, seed, versi pustaka, commit dasar, SHA-256 berkas).

  python archive_experiments.py --name phase2_tldqn

Checkpoint di ml/checkpoints/ di-ignore git; arsip ini yang dipertahankan.
Bobot Bi-GAN hanya disalin untuk checkpoint yang LOLOS verifikasi arah skor;
run yang gagal disimpan konfigurasinya saja (inference_config.json) sebagai bukti.
"""

import os
import sys
import json
import shutil
import hashlib
import platform
import argparse
import subprocess

import numpy as np
import torch
import sklearn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
CKPT = os.path.join(HERE, "checkpoints")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def copy(src, dst, files):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    files.append(dst)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    args = ap.parse_args(argv)
    out = os.path.join(HERE, "experiments", args.name)
    if os.path.exists(out):
        sys.exit(f"{out} sudah ada; pilih --name lain (arsip tidak ditimpa)")
    files = []

    # ---- Bi-GAN -------------------------------------------------------------
    bigan_runs = []
    for d in sorted(os.listdir(CKPT)):
        cfg_path = os.path.join(CKPT, d, "inference_config.json")
        if not (d.startswith("bigan_v2") and os.path.exists(cfg_path)):
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        ok = bool(cfg["validation"].get("direction_ok"))
        dst = os.path.join(out, "bigan", d)
        copy(cfg_path, os.path.join(dst, "inference_config.json"), files)
        copy(os.path.join(CKPT, d, "train_log.json"), os.path.join(dst, "train_log.json"), files)
        if ok:
            for rel in ("model.pt", "splits.npz", os.path.join("preprocessor", "schema.json"),
                        os.path.join("preprocessor", "scaler.npz")):
                copy(os.path.join(CKPT, d, rel), os.path.join(dst, rel), files)
        bigan_runs.append({"dir": d, "seed": cfg["model"]["seed"],
                           "n_hidden": cfg["model"]["n_hidden"],
                           "val_auc": cfg["validation"]["auc"], "direction_ok": ok,
                           "weights_archived": ok})
    for f in ("bigan_eval_bigan_v2_h2.json", "bigan_confusion_bigan_v2_h2.png"):
        p = os.path.join(HERE, "reports", f)
        if os.path.exists(p):
            copy(p, os.path.join(out, "bigan", "reports", f), files)

    # ---- TLDQN --------------------------------------------------------------
    tl_dir = os.path.join(CKPT, "tldqn")
    for f in sorted(os.listdir(tl_dir)):
        copy(os.path.join(tl_dir, f), os.path.join(out, "tldqn", "checkpoints", f), files)
    copy(os.path.join(HERE, "reports", "tldqn_eval.json"),
         os.path.join(out, "tldqn", "reports", "tldqn_eval.json"), files)
    copy(os.path.join(ROOT, "figures", "tldqn_reward.png"),
         os.path.join(out, "tldqn", "reports", "tldqn_reward.png"), files)
    copy(os.path.join(ROOT, "ns3-sim", "results", "tldqn_metrics.csv"),
         os.path.join(out, "tldqn", "reports", "tldqn_metrics.csv"), files)

    with open(os.path.join(tl_dir, "config.json")) as f:
        tl_cfg = json.load(f)
    manifest = {
        "name": args.name,
        "base_commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "working_tree_dirty": bool(git("status", "--porcelain", "--", ".", ":!ml/experiments")),
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "sklearn": sklearn.__version__,
                        "platform": platform.platform()},
        "commands": {
            "bigan_train": "python bigan_ids.py --out checkpoints/bigan_v2_h<H>[_s<S>] "
                           "--n-hidden <H> --epochs 20 --seed <S> [--allow-inverted-direction]",
            "bigan_eval": "python detector_eval.py --ckpt checkpoints/bigan_v2_h2 --ablation-recon",
            "tldqn": "python tldqn_routing.py   (default = konfigurasi yang dilaporkan)",
        },
        "bigan": {"runs": bigan_runs,
                  "mainline_checkpoint": "bigan_v2_h2 (satu-satunya yang lolos verifikasi arah)",
                  "comparison_experiment": "reconstruction error (--ablation-recon), bukan mainline",
                  "not_archived": "run kode lama (seed 2: AUC 0.694, seed 3: 0.221) dan run "
                                  "1 lapisan seed 1 (AUC 0.071, tidak disimpan karena gagal "
                                  "verifikasi) hanya tercatat di docs/PAPER_DEVIATIONS.md 2a.4"},
        "tldqn": {"config": tl_cfg},
        "files": {os.path.relpath(p, out): sha256(p) for p in sorted(files)},
    }
    with open(os.path.join(out, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"Arsip -> {os.path.relpath(out, ROOT)}  ({len(files)} berkas)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
run_dto_step3.py  --  jalankan seluruh eksperimen DTO langkah 3 (offline) + evaluasi
---------------------------------------------------------------------------------
Satu perintah yang mereproduksi semua keluaran langkah 3 dari telemetry langkah 2 yang
sudah di-commit. Tidak menjalankan NS-3. Label hanya dipakai di tahap evaluasi.

  1. deteksi (dto_detect.py, tanpa label)  -> ns3-sim/results/step3/<grup>/
  2. evaluasi (dto_eval.py, label di sini) -> ml/reports/dto_step3_<grup>.json
  3. ringkasan                             -> ml/reports/dto_step3_summary.json

Grup:
  stage2        UTAMA (K7)      semua varian, semua window + window 0 (t <= 30 s)
  ctrl0         UTAMA (K7)      kontrol tanpa grayhole (n = 20, 60, 100)
  stage1        sekunder        rute trust 0.5 (lebih banyak relay malicious teramati)
  stage1rep     determinisme    run ulang NS-3 seed 1 (n = 20, 60, 100)
  seeds         stabilitas      stage2 varian main, seed DTO 1..4
  params        sensitivitas    stage2 main, (m, Y) = (10, 30) dan (50, 200)
"""

import json
import os
import sys

import dto_detect
import dto_eval

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "ns3-sim", "results")
S2 = os.path.join(RES, "step2")
OUT = os.path.join(RES, "step3")
REP = os.path.join(HERE, "reports")
ALL = [20, 30, 40, 50, 60, 70, 80, 90, 100]
TRIO = [20, 60, 100]
VARS = ["main", "skip", "paper_k", "tx_ablation"]


def detect(group, tag, nodes, seeds, variants, windows=None, dto_seed=0, extra=()):
    d = os.path.join(OUT, group)
    args = ["--telem-dir", os.path.join(S2, f"telemetry_{tag}"), "--run-tag", tag,
            "--out-dir", d, "--nodes", *map(str, nodes), "--seeds", *map(str, seeds),
            "--variants", *variants, "--dto-seed", str(dto_seed), *extra]
    if windows is not None:
        args += ["--windows", *map(str, windows)]
    dto_detect.main(args)
    return d


def evaluate(group, d, tag, nodes, seeds, variants, wtag="all", seed_suffix="", name=None):
    out = os.path.join(REP, f"dto_step3_{name or group}.json")
    dto_eval.main(["--dec-dir", d, "--run", tag, "--variants", *variants, "--wtag", wtag,
                   "--nodes", *map(str, nodes), "--seeds", *map(str, seeds),
                   "--scenario-dir", RES, "--telem-dir", os.path.join(S2, f"telemetry_{tag}"),
                   "--metrics-dir", S2, "--metrics-tag", f"_{tag}", "--seed-suffix", seed_suffix,
                   "--out", out])
    return json.load(open(out))


def same(dir_a, dir_b, names):
    return dto_eval.same_decisions(dir_a, dir_b, names)


def main():
    summary = {}
    # 1. UTAMA: stage2
    d = detect("stage2", "stage2", ALL, [1, 2, 3], VARS)
    summary["stage2"] = evaluate("stage2", d, "stage2", ALL, [1, 2, 3], VARS)
    detect("stage2", "stage2", ALL, [1, 2, 3], ["main"], windows=[0])
    summary["stage2_w0"] = evaluate("stage2", d, "stage2", ALL, [1, 2, 3], ["main"], "w0",
                                    name="stage2_w0")
    # 2. UTAMA: kontrol tanpa grayhole
    d = detect("ctrl0", "ctrl0", TRIO, [1, 2, 3], VARS)
    summary["ctrl0"] = evaluate("ctrl0", d, "ctrl0", TRIO, [1, 2, 3], VARS)
    # 3. sekunder: stage1
    d1 = detect("stage1", "stage1", ALL, [1, 2, 3], VARS)
    summary["stage1"] = evaluate("stage1", d1, "stage1", ALL, [1, 2, 3], VARS)
    # 4. determinisme: run NS-3 ulang -> keputusan identik
    dr = detect("stage1rep", "stage1rep", TRIO, [1], ["main"])
    names = [(f"dto_decisions_{n}_s1_stage1_main_all.csv",
              f"dto_decisions_{n}_s1_stage1rep_main_all.csv") for n in TRIO]
    summary["determinism_stage1rep"] = same(d1, dr, names)
    # 5. stabilitas seed DTO
    ds = os.path.join(OUT, "stage2")
    stab = []
    for k in (1, 2, 3, 4):
        detect("stage2", "stage2", ALL, [1, 2, 3], ["main"], dto_seed=k)
        names = [(f"dto_decisions_{n}_s{s}_stage2_main_all.csv",
                  f"dto_decisions_{n}_s{s}_stage2_main_all_seed{k}.csv")
                 for n in ALL for s in (1, 2, 3)]
        stab.append(sum(same(ds, ds, names)))
    summary["dto_seed_stability"] = {"identical_scenarios_per_seed": stab, "of": 27}
    # 6. sensitivitas (m, Y)
    sens = {}
    for m, Y in ((10, 30), (50, 200)):
        dp = detect(f"params_m{m}_Y{Y}", "stage2", ALL, [1, 2, 3], ["main"],
                    extra=("--m", str(m), "--Y", str(Y)))
        names = [(f"dto_decisions_{n}_s{s}_stage2_main_all.csv",) * 2 for n in ALL for s in (1, 2, 3)]
        sens[f"m{m}_Y{Y}"] = sum(same(ds, dp, names))
    summary["param_sensitivity_identical_scenarios"] = {**sens, "of": 27}

    compact = {}
    for g in ("stage2", "stage2_w0", "ctrl0", "stage1"):
        for r in summary[g]["results"]:
            compact[f"{g}/{r['variant']}"] = {
                "all": {k: r["pooled"]["all"][k] for k in ("TP", "FP", "FN", "TN", "precision",
                                                           "recall", "f1", "fpr")},
                "evidence": {k: r["pooled"]["evidence"][k] for k in ("precision", "recall", "f1")},
                "coverage": r["coverage"], "argmax": r["dto_argmax_equivalence"],
                "baselines": {b: {k: r["pooled"][b][k] for k in ("precision", "recall", "f1", "fpr")}
                              for b in ("cpp_dto", "trust_thresh", "bigan_ns3")}}
    out = {"results": compact,
           "determinism_stage1rep_identical": summary["determinism_stage1rep"],
           "dto_seed_stability": summary["dto_seed_stability"],
           "param_sensitivity": summary["param_sensitivity_identical_scenarios"],
           "config": summary["stage2"]["config"]}
    with open(os.path.join(REP, "dto_step3_summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\ndeterminisme (stage1rep):", summary["determinism_stage1rep"])
    print("stabilitas seed DTO:", summary["dto_seed_stability"])
    print("sensitivitas (m, Y):", summary["param_sensitivity_identical_scenarios"])
    return 0


if __name__ == "__main__":
    sys.exit(main())

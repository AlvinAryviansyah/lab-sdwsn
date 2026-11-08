# Static end-to-end performance dashboard (builder + template)

Adds `ml/build_dashboard.py`, which embeds repo CSV/JSON results into `ml/dashboard_template.html` to produce a self-contained `dashboard/index.html`. All displayed numbers trace to repo files. The step-1 trust reconstruction reproduces the committed route tables exactly, and ground truth stays behind the `fGT` toggle. The remaining problems are wording and labeling, not data integrity.

Watch for: (1) DTO is told the attacker count (confirmed), which the dashboard never says. (2) The KPI "Trust telemetry AUC" is shown for configs that don't use telemetry trust (confirmed). (3) Unsupported claims in the text (likely).

**Verdict**: NEEDS_CHANGES (text fixes only; no data or logic bugs found)

## High-level view

Data provenance holds. I recomputed `X.build_env(...,"simulated")` for (20,1), (60,2) and (100,3) and ran the committed Q-net: 0 of 344 next-hop entries differ from `tldqn_nexthop_*.csv`. `link_model` (`prr*stability`) depends only on `scenario_seed = 10000*seed+n`, not on labels or trust, so passing zeros for malicious and "none" is safe.

The derived metrics are correct. `dto_TP+dto_FN` equals the `actual_mal` sum in `bigan_decisions_*` for all 27 scenarios and both step-1 and stage-2 metrics, with 0 mismatches. `ground_truth()` relies on row order, and node_ids are sorted in all 27 files. NS-3 config facts match `honeycomb-sdwsn.cc`: 150 m, 600 s, 50 packets, 100 J, 4 sinks, 8.3 s, 15.3 s, grayhole 0.70, and 30 s windows. NSL-KDD is separated and labelled, and the NS-3 decisions come from the legacy CSVs. The NA cases are handled.

<details>
<summary>Issues (5)</summary>

1. **DTO prior not disclosed (confirmed)** — The `DetectMalicious(evidence, cfg::attackers)` call at `honeycomb-sdwsn.cc:913` takes a top-k with k = n/10 (`:1737`), which equals the true attacker count. So `dto_FP == dto_FN` structurally (e.g. 3/3 at n=100 s1), and DTO precision, recall and F1 are identical. Template `:341` and `:438` say only "laju kirim + energi terukur". Add "k = 10% node (diketahui a priori)" and mark the count as an algorithm input.
2. **Trust AUC KPI not scoped (confirmed)** — `template:323` (`trustEval()`, always `D.trust` from the Stage-1 file) is shown for any focus config, including step1/greedy/stage2. The subtitle doesn't say "dari run Stage 1". Label it, or show NA when the focus has no telemetry.
3. **"batas atas, bukan metode sah" (likely)** — `template:334` asserts that step-1 GT trust is an upper bound. No data in the repo supports it. Replace with "trust diturunkan dari label".
4. **Unsupported cross-seed claim (likely)** — `template:461` says "arah skor tidak stabil antar seed". The displayed JSON is one checkpoint (test AUC 0.488, correct), so the seed-instability claim has no source in this file. Cite a report or drop it.
5. **Pair-test note asserts code equivalence (possible)** — `template:425` says "diff kode hanya telemetry". Step-1 and stage metrics differ slightly (e.g. energy 18.0367 vs 18.0349 at n=20 s1), and I did not verify the binary history. Soften it to "dibuat sebelum telemetry".

</details>

<details>
<summary>Details</summary>

### DTO is not label-free
DTO (`honeycomb-sdwsn.cc:913`) selects exactly k = max(1, n/10) nodes by an evidence score, and the true attacker fraction is 10%. DTO F1 and IPS mitigation therefore partly reflect a known count. DTO's choice also varies with routing, because the evidence comes from tx counts and energy. In n=100 s1, dto_TP is 7/6/8/7 across step1/greedy/stage1/stage2, and `attackers_blocked` follows it. The IPS bar chart and KPI compare configs on this, so differences of 1 to 2 nodes are routing-induced noise from a single seed. The text says Bi-GAN is identical across configs, which is true (TP/FP/FN match in all 4 configs checked). It does not say DTO and IPS are not.

### Verified without findings
IPS accounting matches the code: `nodes_blocked` counts `pred || dtoFlag`, and `false = blocked - hit` is correct. The ctrl0 gaps (only n=20/60/100) show NA through the availability table. Packets sent/received are used only when every run has them. The text "node terisolasi masih meneruskan transit" and the claim that isolation happens at t = 8.3 s are consistent with `GenerateTraffic` (stops sending, `:753`). I did not trace the forwarding path.

### Not checked
No test covers the builder. The JS metric helpers (`signTest`, `auc`, `conf`) were read but not unit-tested. Sign tests across scenarios treat n/seed pairs as independent, which the page discloses. They run 6 metrics with no multiplicity correction.

</details>

<details>
<summary>File map</summary>

- `ml/build_dashboard.py`: collects metrics, routes, telemetry, trust, GT and reports into JSON and embeds it in the template.
- `ml/dashboard_template.html`: filters, KPIs, charts and tables (client-side).
- `dashboard/index.html`: generated artifact.

</details>

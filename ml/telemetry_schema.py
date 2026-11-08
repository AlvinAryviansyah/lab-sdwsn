#!/usr/bin/env python3
"""
telemetry_schema.py  --  skema & validator telemetry NS-3 (integrasi langkah 2)
-------------------------------------------------------------------------------
Definisi kolom lengkap: docs/TELEMETRY.md. Berkas per skenario (n, seed), ditulis
honeycomb-sdwsn.cc ke --telemDir:

  telemetry_<n>_s<seed>.csv        per sensor per window 30 s (t = 0-based)
  telemetry_links_<n>_s<seed>.csv  per (window, observer, tetangga): MAC handoff,
                                   kualitas tautan, hasil watchdog
  watchdog_audit_<n>_s<seed>.csv   KEBENARAN SIMULATOR per watch (evaluasi saja)
  telemetry_meta_<n>_s<seed>.json  parameter & total jaringan

Kelompok kolom dan izin pemakaian:
  obs_* / wd_*     diamati TETANGGA lewat watchdog  -> boleh untuk trust
  self_*           diukur pada node itu sendiri; ns-3 melaporkannya jujur (tidak ada
                   pemalsuan), jadi counter transit (self_ip_fwd_*, self_ip_drop_*)
                   praktis membocorkan perilaku grayhole -> TIDAK boleh untuk trust
  watchdog_audit   kebenaran simulator -> HANYA evaluasi
Label malicious tidak pernah ada di berkas telemetry.
"""

import csv
import json
import os

import numpy as np

NODE_COLUMNS = [
    "node_id", "window", "t_start", "t_end",
    "self_ip_orig_pkts", "self_ip_orig_bytes", "self_ip_rx_pkts", "self_ip_fwd_pkts",
    "self_ip_fwd_bytes", "self_ip_local_pkts", "self_ip_drop_ttl", "self_ip_drop_no_route",
    "self_ip_drop_bad_checksum", "self_ip_drop_iface_down", "self_ip_drop_route_error",
    "self_ip_drop_frag_timeout", "self_ip_drop_duplicate", "self_ip_drop_other",
    "self_mac_acked", "self_mac_nacked", "self_mac_drop_retry", "self_mac_drop_other",
    "self_mac_ack_timeout", "self_phy_tx_mpdu", "self_phy_tx_bytes", "self_phy_rx_ok_mpdu",
    "self_phy_rx_fail_mpdu", "self_phy_rx_drop", "self_energy_j", "wd_started_as_observer",
    "obs_watched", "obs_forwarded", "obs_forwarded_late", "obs_unobserved_rxfail",
    "obs_unobserved_clean", "obs_heard_after_timeout",
]
LINK_COLUMNS = [
    "window", "observer", "neighbor_type", "neighbor_id", "mac_handoff_acked",
    "mac_handoff_nacked", "mac_handoff_drop_retry", "mac_handoff_drop_other",
    "mac_ack_timeout", "rx_ok_from", "rssi_mean_dbm", "snr_mean_db", "wd_started",
    "wd_forwarded", "wd_forwarded_late", "wd_unobserved_rxfail", "wd_unobserved_clean",
    "wd_heard_after_timeout", "wd_heard_rxfail", "wd_delay_mean_s",
]
AUDIT_COLUMNS = ["window", "observer", "neighbor", "wd_category", "true_cause", "count"]
WD_CATEGORIES = ["forwarded", "forwarded_late", "unobserved_rxfail", "unobserved_clean"]
FLOAT_NODE = {"t_start", "t_end", "self_energy_j"}
FLOAT_LINK = {"rssi_mean_dbm", "snr_mean_db", "wd_delay_mean_s"}

# Kolom yang BOLEH menjadi masukan trust (hanya observasi tetangga).
TRUST_INPUT_COLUMNS = ["obs_watched", "obs_forwarded", "obs_forwarded_late",
                       "obs_unobserved_rxfail", "obs_unobserved_clean"]
TRUST_INPUT_LINK_COLUMNS = ["window", "observer", "neighbor_type", "neighbor_id", "wd_started",
                            "wd_forwarded", "wd_forwarded_late", "wd_unobserved_rxfail",
                            "wd_unobserved_clean", "wd_heard_rxfail"]
# Masukan DTO (langkah 3): watchdog + frame dari tetangga yang di-decode observer
# (aktivitas TX, hanya untuk ablasi; bukan energi). Tetap tanpa self_* dan tanpa label.
DTO_INPUT_LINK_COLUMNS = TRUST_INPUT_LINK_COLUMNS + ["rx_ok_from"]
# Masukan Hybrid IDS (langkah 4) dari bigan_decisions_<n>_s<seed>.csv: HANYA prediksi.
# actual_mal (label skenario) tidak boleh dibaca jalur keputusan.
BIGAN_INPUT_COLUMNS = ["node_id", "pred_intrusion"]
# Nama yang menandakan label / kebenaran simulator; tidak boleh muncul di telemetry.
FORBIDDEN_TOKENS = ("malicious", "actual_mal", "label", "attacker", "grayhole", "true_cause")


def tag(n, seed):
    return f"{n}_s{seed}"


def paths(telem_dir, n, seed):
    t = tag(n, seed)
    return {"node": os.path.join(telem_dir, f"telemetry_{t}.csv"),
            "links": os.path.join(telem_dir, f"telemetry_links_{t}.csv"),
            "audit": os.path.join(telem_dir, f"watchdog_audit_{t}.csv"),
            "meta": os.path.join(telem_dir, f"telemetry_meta_{t}.json")}


def _read(path, columns, floats):
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        if r.fieldnames != columns:
            raise ValueError(f"{path}: header tidak sesuai skema\n  ada  : {r.fieldnames}\n"
                             f"  harus: {columns}")
        rows = []
        for row in r:
            out = {}
            for c in columns:
                v = row[c]
                if c in floats:
                    out[c] = float(v) if v != "" else float("nan")
                elif c in ("neighbor_type", "wd_category", "true_cause"):
                    out[c] = v
                else:
                    out[c] = int(v)
            rows.append(out)
    return rows


def load_nodes(path):
    return _read(path, NODE_COLUMNS, FLOAT_NODE)


def load_links(path):
    return _read(path, LINK_COLUMNS, FLOAT_LINK)


def load_audit(path):
    return _read(path, AUDIT_COLUMNS, set())


def load_meta(path):
    with open(path) as f:
        return json.load(f)


def load_metrics(path):
    with open(path) as f:
        return {r["metric"]: float(r["value"]) for r in csv.DictReader(f)}


def trust_view(rows, columns):
    """Satu-satunya jalur data telemetry ke trust: hanya kolom yang diizinkan."""
    return [{c: r[c] for c in columns} for r in rows]


def node_totals(nodes, n):
    """Jumlah per sensor atas semua window -> dict kolom -> array (n,)."""
    cols = [c for c in NODE_COLUMNS if c not in ("node_id", "window", "t_start", "t_end")]
    out = {c: np.zeros(n) for c in cols}
    for r in nodes:
        for c in cols:
            out[c][r["node_id"]] += r[c]
    return out


# ----------------------------------------------------------------------------------
#  Validasi (mengembalikan daftar pesan kegagalan; kosong = lolos)
# ----------------------------------------------------------------------------------
def check_no_leakage(*headers):
    errs = []
    for h in headers:
        for c in h:
            if any(tok in c.lower() for tok in FORBIDDEN_TOKENS):
                errs.append(f"kolom terlarang di telemetry: {c}")
    return errs


def energy_tolerance(nodes, meta, base=1e-6, update_interval_s=1.0, changes_per_event=4):
    """Batas selisih energi telemetry vs BasicEnergySource.

    BasicEnergySource::CalculateRemainingEnergy menghitung (I * V * duration) dengan
    duration bertipe Time, dan hasil `Time * double` adalah Time (resolusi 1 ns). Akibatnya
    tiap update sumber dibulatkan sampai sekitar 1e-9 J. Update terjadi periodik
    (PeriodicEnergyUpdateInterval = 1 s) dan pada setiap perubahan keadaan PHY. Model
    radio (dan pembagian per window) menghitung dalam double, jadi lebih teliti.
    Batas = 1e-9 J x (update periodik + changes_per_event x event PHY) + base."""
    n, sim = int(meta["num_nodes"]), float(meta.get("sim_time_s", 0.0))
    ev = sum(r["self_phy_tx_mpdu"] + r["self_phy_rx_ok_mpdu"] + r["self_phy_rx_fail_mpdu"] +
             r["self_phy_rx_drop"] for r in nodes)
    return base + 1e-9 * (n * sim / update_interval_s + changes_per_event * ev)


def _check_forward_definition(m):
    """Penerusan UDP vs FlowMonitor.

    Baru (telemetry_ip_forward_udp_delivered ada): harus sama persis.
    Lama (hanya total): kesamaan eksak tidak bisa diperiksa. Yang diperiksa adalah
    syarat perlu dari mekanismenya: selisih = penerusan paket yang kemudian hilang, jadi
    selisih >= 0 dan selisih = 0 bila tidak ada paket hilang."""
    fm = m.get("flowmon_udp_times_forwarded")
    if fm is None:
        return ["metrics tidak memuat flowmon_udp_times_forwarded"]
    if "telemetry_ip_forward_udp_delivered" in m:
        d = m["telemetry_ip_forward_udp_delivered"]
        errs = [] if d == fm else [f"telemetry_ip_forward_udp_delivered={d:.0f} != "
                                   f"flowmon_udp_times_forwarded={fm:.0f}"]
        if m.get("telemetry_ip_forward_udp_total", d) < d:
            errs.append("penerusan total < penerusan paket terkirim")
        return errs
    tot = m.get("telemetry_ip_forward_udp_total")
    if tot is None:
        return ["metrics tidak memuat telemetry_ip_forward_udp_total"]
    lost = m.get("telemetry_ip_orig_udp_total", 0) - m.get("telemetry_ip_local_udp_total", 0)
    ex = tot - fm
    if ex < 0:
        return [f"penerusan telemetry {tot:.0f} < FlowMonitor {fm:.0f} (definisi lama)"]
    if lost <= 0 and ex != 0:
        return [f"selisih penerusan {ex:.0f} tanpa paket hilang (definisi lama)"]
    return []


def forward_definition(m):
    return "delivered" if "telemetry_ip_forward_udp_delivered" in m else "legacy_total"


def validate(nodes, links, audit, meta, metrics=None, energy_tol=1e-6):
    errs = []
    n, W = int(meta["num_nodes"]), int(meta["n_windows"])
    errs += check_no_leakage(NODE_COLUMNS, LINK_COLUMNS)

    # 1. kelengkapan: tepat satu baris per (node, window)
    keys = {(r["node_id"], r["window"]) for r in nodes}
    if len(nodes) != n * W or keys != {(i, w) for i in range(n) for w in range(W)}:
        errs.append(f"baris node tidak lengkap/duplikat: {len(nodes)} != {n}x{W}")
    for r in nodes:
        if abs(r["t_start"] - r["window"] * meta["window_s"]) > 1e-9:
            errs.append(f"t_start salah pada {r['node_id']},{r['window']}")
            break

    # 2. konsistensi kategori watchdog
    for r in nodes:
        s = sum(r[f"obs_{c}"] for c in WD_CATEGORIES)
        if s != r["obs_watched"]:
            errs.append(f"node {r['node_id']} w{r['window']}: kategori {s} != watched "
                        f"{r['obs_watched']}")
            break
        if r["obs_heard_after_timeout"] > r["obs_unobserved_rxfail"] + r["obs_unobserved_clean"]:
            errs.append(f"node {r['node_id']} w{r['window']}: heard_after_timeout > unobserved")
            break
    for r in links:
        s = sum(r[f"wd_{c}"] for c in WD_CATEGORIES)
        if s != r["wd_started"]:
            errs.append(f"link {r['observer']}->{r['neighbor_id']} w{r['window']}: kategori "
                        f"{s} != started {r['wd_started']}")
            break
        if r["wd_heard_rxfail"] > r["wd_forwarded"] + r["wd_forwarded_late"]:
            errs.append(f"link {r['observer']}->{r['neighbor_id']} w{r['window']}: "
                        "heard_rxfail > terdengar")
            break
        if r["wd_started"] and r["neighbor_type"] != "sensor":
            errs.append("watch dimulai untuk tetangga non-sensor")
            break

    # 3. agregasi window: node (sebagai diamati) == jumlah link per tetangga;
    #    node (sebagai observer) == jumlah link per observer; audit == link
    by_nbr, by_obs, aud = {}, {}, {}
    for r in links:
        if r["neighbor_type"] != "sensor":
            continue
        k = (r["neighbor_id"], r["window"])
        v = by_nbr.setdefault(k, np.zeros(5, dtype=np.int64))
        v += [r["wd_started"]] + [r[f"wd_{c}"] for c in WD_CATEGORIES]
        ko = (r["observer"], r["window"])
        by_obs[ko] = by_obs.get(ko, 0) + r["wd_started"]
    for r in audit:
        k = (r["neighbor"], r["window"], r["wd_category"])
        aud[k] = aud.get(k, 0) + r["count"]
    for r in nodes:
        k = (r["node_id"], r["window"])
        node_v = [r["obs_watched"]] + [r[f"obs_{c}"] for c in WD_CATEGORIES]
        link_v = list(by_nbr.get(k, np.zeros(5, dtype=np.int64)))
        if node_v != link_v:
            errs.append(f"agregasi node-vs-link (diamati) beda pada {k}: {node_v} vs {link_v}")
            break
        if r["wd_started_as_observer"] != by_obs.get(k, 0):
            errs.append(f"agregasi observer beda pada {k}: {r['wd_started_as_observer']} vs "
                        f"{by_obs.get(k, 0)}")
            break
        for c in WD_CATEGORIES:
            if aud.get((r["node_id"], r["window"], c), 0) != r[f"obs_{c}"]:
                errs.append(f"audit vs telemetry beda pada {k} kategori {c}")
                break
    tot = sum(r["obs_watched"] for r in nodes)
    if tot != meta["watches"]:
        errs.append(f"total watch {tot} != meta {meta['watches']}")
    for c in WD_CATEGORIES:
        if sum(r[f"obs_{c}"] for r in nodes) != meta[f"wd_{c}"]:
            errs.append(f"total {c} != meta")
    mac_node = sum(r["self_mac_acked"] for r in nodes)
    mac_link = sum(r["mac_handoff_acked"] for r in links)
    if mac_node != mac_link:
        errs.append(f"MAC acked node {mac_node} != jumlah link {mac_link}")

    # 4. counter vs FlowMonitor dan energi (metrics CSV skenario yang sama)
    if metrics is not None:
        # FlowMonitor timesForwarded = penerusan paket yang SAMPAI tujuan (ditambahkan di
        # ReportLastRx, dibuang di ReportDrop). Bandingkan dgn counter definisi sama.
        errs += _check_forward_definition(metrics)
        pairs = [("telemetry_ip_orig_udp_total", "flowmon_udp_tx_packets"),
                 ("telemetry_ip_local_udp_total", "flowmon_udp_rx_packets")]
        for a, b in pairs:
            if a not in metrics or b not in metrics:
                errs.append(f"metrics tidak memuat {a}/{b}")
            elif metrics[a] != metrics[b]:
                errs.append(f"{a}={metrics[a]:.0f} != {b}={metrics[b]:.0f}")
        e = sum(r["self_energy_j"] for r in nodes)
        ref = metrics.get("sensor_energy_consumed_J", metrics.get("energy_consumption_J"))
        tol = energy_tolerance(nodes, meta, energy_tol)
        if ref is not None and abs(e - ref) > tol:
            errs.append(f"energi per window {e:.9f} J != energi sumber {ref:.9f} J "
                        f"(toleransi kuantisasi {tol:.3g} J)")
    return errs


def validate_dir(telem_dir, n, seed, metrics_path=None):
    p = paths(telem_dir, n, seed)
    nodes, links, audit = load_nodes(p["node"]), load_links(p["links"]), load_audit(p["audit"])
    meta = load_meta(p["meta"])
    metrics = load_metrics(metrics_path) if metrics_path else None
    return validate(nodes, links, audit, meta, metrics)


if __name__ == "__main__":
    import argparse
    import sys
    ap = argparse.ArgumentParser(description="Validasi berkas telemetry NS-3")
    ap.add_argument("telem_dir")
    ap.add_argument("--nodes", type=int, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--metrics-dir", default=None,
                    help="folder metrics_<n>_s<seed><tag>.csv (cek FlowMonitor & energi)")
    ap.add_argument("--metrics-tag", default="")
    a = ap.parse_args()
    bad = 0
    for n in a.nodes:
        for s in a.seeds:
            mp = (os.path.join(a.metrics_dir, f"metrics_{n}_s{s}{a.metrics_tag}.csv")
                  if a.metrics_dir else None)
            errs = validate_dir(a.telem_dir, n, s, mp)
            print(f"n={n} s={s}: {'LOLOS' if not errs else 'GAGAL'}")
            for e in errs:
                print("   -", e)
            bad += bool(errs)
    sys.exit(1 if bad else 0)

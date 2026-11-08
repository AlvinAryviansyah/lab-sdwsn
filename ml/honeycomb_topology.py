#!/usr/bin/env python3
"""
honeycomb_topology.py
---------------------
Topologi honeycomb dan ruang aksi routing yang dipakai BERSAMA oleh lingkungan
TLDQN (Python) dan simulasi NS-3 (C++).

Sumber tunggal koordinat: array g_sensorPos / g_sinkPos di
ns3-sim/honeycomb-sdwsn.cc (diambil dari kode rilis penulis). Tidak ada salinan
koordinat di Python.

Rujukan paper (Kipongo et al. 2024):
  - Sec. III-B: sensor di tepi dan pusat heksagon; "Every hexagonal cell has six
    neighboring cells in all directions. Every sensor node can communicate with
    all the neighboring sensors ..."                                   (V)
  - Tabel 5: jangkauan transmisi 150 m                                 (V)
  - Sec. III-C: routing hop-by-hop; aksi pertama = pemilihan next forwarder,
    lalu pemilihan rute dari forwarder optimal                         (V, tanpa detail)

Aturan yang ditetapkan di sini (ASUMSI, didokumentasikan di PAPER_DEVIATIONS):
  - Tujuan tiap sensor = edge sink terdekat (jarak Euklides; seri -> indeks kecil).
  - Kandidat next-hop dari node c menuju tujuan t:
        sensor j != c dengan jarak(c, j) <= R dan jarak(j, t) < jarak(c, t)
        (syarat progres -> rute dijamin bebas loop), ditambah sink t bila
        jarak(c, t) <= R (sink selalu valid; perlu karena sensor 23 dan 37
        berimpit dengan sink 3 dan 2 di kode rilis).
  - Relay hanya sensor; sink lain bukan relay.
  - Slot aksi: kandidat diurutkan menurut (jarak ke tujuan, id node), diisi ke
    K = 6 slot; slot sisa ditandai tidak valid. Pada koordinat paper jumlah
    kandidat maksimum = 6 (diverifikasi oleh tes), sehingga tidak ada kandidat
    yang terpotong.
"""

import os
import re
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CC_FILE = os.path.join(HERE, "..", "ns3-sim", "honeycomb-sdwsn.cc")

TX_RANGE = 150.0          # m, Tabel 5
K_ACTIONS = 6             # slot next-hop = enam tetangga heksagon (Sec. III-B)
NODE_SWEEP = [20, 30, 40, 50, 60, 70, 80, 90, 100]
ROUTING_SCHEMA_VERSION = 1


def parse_positions(cc_file=CC_FILE):
    """Koordinat sensor (100, 2) dan sink (4, 2) dari honeycomb-sdwsn.cc."""
    src = open(cc_file, encoding="utf-8").read()

    def block(name):
        m = re.search(name + r"\[[^\]]*\]\[2\]\s*=\s*\{(.*?)\};", src, re.S)
        if m is None:
            raise ValueError(f"array {name} tidak ditemukan di {cc_file}")
        pairs = re.findall(r"\{\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\}", m.group(1))
        return np.array(pairs, dtype=np.float64)

    return block("g_sensorPos"), block("g_sinkPos")


class HoneycombTopology:
    """Graf routing untuk n sensor pertama + seluruh edge sink.

    Indeks node: sensor 0..n-1, sink n..n+n_sinks-1 (sama dengan urutan
    NodeContainer di NS-3: sensor i = sensors.Get(i), sink k = sinks.Get(k)).
    """

    def __init__(self, n_sensors, tx_range=TX_RANGE, k_actions=K_ACTIONS, cc_file=CC_FILE):
        S, K = parse_positions(cc_file)
        if not 1 <= n_sensors <= len(S):
            raise ValueError(f"n_sensors harus 1..{len(S)}")
        self.n = int(n_sensors)
        self.n_sinks = len(K)
        self.tx_range = float(tx_range)
        self.k = int(k_actions)
        self.pos = np.vstack([S[:self.n], K])
        diff = self.pos[:, None, :] - self.pos[None, :, :]
        self.dist = np.hypot(diff[..., 0], diff[..., 1])
        self.n_nodes = len(self.pos)
        # tujuan = sink terdekat (argmin memilih indeks terkecil bila seri)
        d_sink = self.dist[:self.n, self.n:]
        self.dest = self.n + np.argmin(d_sink, axis=1)
        # Hanya pasangan (node, tujuan) yang TERJANGKAU dari suatu sumber menuju
        # sink terdekatnya: hanya pasangan ini yang pernah dialami paket. Pada
        # pasangan lain jumlah kandidat bisa > K, tetapi tidak pernah dipakai.
        self._cand = {}
        for s in range(self.n):
            stack = [(s, int(self.dest[s]))]
            while stack:
                c, t = stack.pop()
                if (c, t) in self._cand or self.is_sink(c):
                    continue
                self._cand[(c, t)] = self._compute_candidates(c, t)
                stack += [(j, t) for j in self._cand[(c, t)]]
        worst = max(len(v) for v in self._cand.values())
        if worst > self.k:
            raise ValueError(f"{worst} kandidat > {self.k} slot aksi (n={self.n})")
        if any(len(v) == 0 for v in self._cand.values()):
            raise ValueError("ada node terjangkau tanpa kandidat (buntu)")

    # -- kandidat ----------------------------------------------------------------
    def is_sink(self, j):
        return j >= self.n

    def _compute_candidates(self, c, t):
        dct = self.dist[c, t]
        out = [j for j in range(self.n)
               if j != c and self.dist[c, j] <= self.tx_range and self.dist[j, t] < dct]
        if self.dist[c, t] <= self.tx_range:
            out.append(t)
        return sorted(out, key=lambda j: (self.dist[j, t], j))

    def candidates(self, c, t=None):
        """Kandidat next-hop terurut (maks K) dari sensor c menuju tujuan t.
        Hanya untuk pasangan terjangkau (KeyError bila tidak)."""
        t = int(self.dest[c]) if t is None else int(t)
        return list(self._cand[(c, t)])

    def reachable_pairs(self):
        """Himpunan (node, tujuan) yang dapat dialami paket."""
        return sorted(self._cand)

    def slots(self, c, t=None):
        """Array panjang K: id node per slot, -1 = slot tidak valid."""
        cand = self.candidates(c, t)
        return np.array(cand + [-1] * (self.k - len(cand)), dtype=np.int64)

    def greedy_path(self, s, choose=lambda slots, c: slots[0]):
        """Ikuti kebijakan `choose` dari sensor s sampai sink tujuan (diagnostik)."""
        t, c, path = self.dest[s], s, [s]
        for _ in range(self.n_nodes):
            sl = self.slots(c, t)
            if sl[0] < 0:
                return path, False
            c = int(choose(sl[sl >= 0], c))
            path.append(c)
            if c == t:
                return path, True
        return path, False

    def colocated_sensor_sink(self):
        """Pasangan (sensor, sink_idx) yang koordinatnya identik."""
        return [(i, int(t - self.n)) for i in range(self.n) for t in range(self.n, self.n_nodes)
                if self.dist[i, t] == 0.0]


def routing_schema(feature_names, node_features, extra=None):
    """Kontrak ruang aksi/keadaan untuk Python <-> NS-3 (ditulis bersama hasil ekspor)."""
    return {
        "routing_schema_version": ROUTING_SCHEMA_VERSION,
        "tx_range_m": TX_RANGE,
        "action_dim": K_ACTIONS,
        "action_semantics": "slot k = kandidat next-hop ke-k, diurutkan (jarak ke sink tujuan, id node)",
        "candidate_rule": "sensor j!=c, dist(c,j)<=R, dist(j,t)<dist(c,t); plus sink t bila dist(c,t)<=R",
        "destination_rule": "sink terdekat (Euklides), seri -> indeks sink terkecil",
        "node_index": "sensor 0..n-1 = sensors.Get(i); sink n+k = sinks.Get(k)",
        "ns3_node_id_note": "indeks n+k adalah indeks ABSTRAK, bukan NodeId ns-3: kelompok node lain "
                            "(user, blockchain, ...) dibuat di antara sensor dan sink; C++ harus memetakan "
                            "lewat sinks.Get(k) secara eksplisit",
        "slot_order_note": "jarak berseri banyak pada kisi honeycomb; NS-3 sebaiknya MEMAKAI tabel kandidat "
                           "hasil ekspor Python, bukan menghitung ulang (sqrt C++ vs hypot numpy bisa beda 1 ulp)",
        "slot_features": list(feature_names),
        "node_features": list(node_features),
        **(extra or {}),
    }


def save_routing_schema(path, feature_names, node_features, extra=None):
    with open(path, "w") as f:
        json.dump(routing_schema(feature_names, node_features, extra), f, indent=1)

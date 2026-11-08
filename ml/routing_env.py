#!/usr/bin/env python3
"""
routing_env.py
--------------
Lingkungan routing honeycomb untuk TLDQN (Sec. III-C, Algoritma 1 paper).

Satu EPISODE = satu paket data dari sensor sumber ke edge sink tujuannya,
diteruskan hop-by-hop. Pada tiap hop agen memilih next forwarder dari K = 6 slot
kandidat (lihat honeycomb_topology.py).

Pemetaan ke paper (V = disebut paper, U = tidak disebut -> asumsi):
  Kriteria forwarder (V): trust, distance, link quality, energy
  Kriteria rute     (V): number of hops, link stability, PDR, throughput
  Bentuk numerik keadaan, model kanal, nilai reward              (U)

KEADAAN (vektor 1-D, Markov terhadap (node saat ini, tujuan, kondisi jaringan)):
  per slot k (K x 7):
    valid        1 jika slot berisi kandidat
    trust        estimasi trust kandidat (reputasi Beta dari observasi; sink = 1)
    link_quality PRR geometris tautan c->j
    stability    stabilitas tautan c->j
    energy       sisa energi kandidat (fraksi energi awal; sink = 1)
    progress     (d(c,t) - d(j,t)) / R  (kriteria "distance")
    is_dest      1 jika kandidat adalah sink tujuan
  per node (2):
    remaining    d(c,t) / (4R)       (proksi jumlah hop tersisa)
    own_energy   sisa energi node saat ini
AKSI: indeks slot 0..K-1; hanya slot valid yang boleh dipilih (masking).
TRANSISI: paket berhasil melewati tautan dengan peluang lq * stability; relay
  meneruskan dengan peluang p_fwd_normal (node normal) atau p_fwd_malicious (node
  malicious, dari ground truth skenario; agen hanya melihat estimasi trust).
REWARD:
  sampai di sink         : +r_success
  paket hilang/TTL habis : -r_fail
  tiap hop               : -c_hop - c_energy * (1 - energy_kandidat)
  -> imbalan yang diharapkan merangkum PDR (sukses vs hilang), jumlah hop, dan
     energi; trust, link quality, stabilitas memengaruhi peluang sukses.
  Throughput tidak dimodelkan terpisah (laju tautan tetap di NS-3); dicatat
  sebagai deviasi.
"""

from dataclasses import dataclass, asdict

import numpy as np

from honeycomb_topology import HoneycombTopology, K_ACTIONS

SLOT_FEATURES = ["valid", "trust", "link_quality", "stability", "energy", "progress", "is_dest"]
NODE_FEATURES = ["remaining", "own_energy"]
STATE_DIM = K_ACTIONS * len(SLOT_FEATURES) + len(NODE_FEATURES)


@dataclass
class EnvConfig:
    # model kanal (U): PRR logistik jarak, wilayah transisi khas WSN
    prr_d50: float = 120.0          # jarak dengan PRR 0.5 (m)
    prr_width: float = 10.0         # kecuraman (m)
    stability_min: float = 0.80     # stabilitas tautan ~ U(stability_min, 1)
    # perilaku node (U)
    malicious_fraction: float = 0.10    # Tabel 5: 10% node malicious
    p_fwd_normal: float = 0.98
    p_fwd_malicious: float = 0.30
    trust_obs: int = 10             # jumlah observasi untuk reputasi Beta
    energy_min: float = 0.30        # sisa energi awal episode ~ U(energy_min, 1)
    # reward (U)
    r_success: float = 1.0
    r_fail: float = 1.0
    c_hop: float = 0.05
    c_energy: float = 0.10
    max_hops: int = 12              # pengaman TTL

    def to_dict(self):
        return asdict(self)


def link_prr(d, cfg):
    return 1.0 / (1.0 + np.exp((np.asarray(d, float) - cfg.prr_d50) / cfg.prr_width))


class HoneycombRoutingEnv:
    """Lingkungan episodik. `scenario_seed` menetapkan node malicious dan
    stabilitas tautan (tetap selama hidup lingkungan); `reset` mengambil kondisi
    jaringan baru (energi, observasi trust) dan sumber paket."""

    def __init__(self, n_sensors, scenario_seed, cfg=None, topology=None):
        self.cfg = cfg or EnvConfig()
        self.topo = topology or HoneycombTopology(n_sensors)
        self.n = self.topo.n
        self.scenario_seed = int(scenario_seed)
        rng = np.random.default_rng(self.scenario_seed)
        n_mal = int(round(self.cfg.malicious_fraction * self.n))
        self.malicious = np.zeros(self.topo.n_nodes, dtype=bool)
        if n_mal:
            self.malicious[rng.choice(self.n, n_mal, replace=False)] = True
        self.prr = link_prr(self.topo.dist, self.cfg)
        stab = rng.uniform(self.cfg.stability_min, 1.0, self.topo.dist.shape)
        self.stability = np.triu(stab) + np.triu(stab, 1).T       # simetris
        self.rng = np.random.default_rng(self.scenario_seed + 7919)
        self.energy = np.ones(self.topo.n_nodes)
        self.trust = np.ones(self.topo.n_nodes)
        self.cur = self.dest = None
        self.hops = 0

    # -- kondisi jaringan --------------------------------------------------------
    def sample_network_state(self, rng=None):
        rng = self.rng if rng is None else rng
        c = self.cfg
        energy = np.ones(self.topo.n_nodes)
        energy[:self.n] = rng.uniform(c.energy_min, 1.0, self.n)
        p_true = np.where(self.malicious, c.p_fwd_malicious, c.p_fwd_normal)
        succ = rng.binomial(c.trust_obs, p_true)
        trust = (succ + 1.0) / (c.trust_obs + 2.0)                    # reputasi Beta(1,1)
        trust[self.n:] = 1.0                                         # sink terpercaya
        return energy, trust

    def set_network_state(self, energy, trust):
        self.energy = np.asarray(energy, float).copy()
        self.trust = np.asarray(trust, float).copy()

    # -- API episode ---------------------------------------------------------------
    def reset(self, source=None, network_state=None, rng=None):
        rng = self.rng if rng is None else rng
        if network_state is None:
            network_state = self.sample_network_state(rng)
        self.set_network_state(*network_state)
        self.cur = int(rng.integers(self.n)) if source is None else int(source)
        if not 0 <= self.cur < self.n:
            raise ValueError("sumber harus sensor 0..n-1")
        self.dest = int(self.topo.dest[self.cur])
        self.hops = 0
        return self.observe()

    def slots(self):
        return self.topo.slots(self.cur, self.dest)

    def action_mask(self):
        return self.slots() >= 0

    def observe(self, node=None):
        c = self.cur if node is None else node
        t, R = self.dest, self.topo.tx_range
        sl = self.topo.slots(c, t)
        feats = np.zeros((K_ACTIONS, len(SLOT_FEATURES)), dtype=np.float32)
        for k, j in enumerate(sl):
            if j < 0:
                continue
            feats[k] = [1.0, self.trust[j], self.prr[c, j], self.stability[c, j],
                        self.energy[j], (self.topo.dist[c, t] - self.topo.dist[j, t]) / R,
                        1.0 if j == t else 0.0]
        node = np.array([self.topo.dist[c, t] / (4 * R), self.energy[c]], dtype=np.float32)
        return np.concatenate([feats.ravel(), node])

    def step(self, action):
        sl = self.slots()
        if not 0 <= action < K_ACTIONS or sl[action] < 0:
            raise ValueError(f"aksi tidak valid: slot {action}")
        c, j, cfg = self.cur, int(sl[action]), self.cfg
        self.hops += 1
        info = {"from": c, "to": j}
        if self.rng.random() >= self.prr[c, j] * self.stability[c, j]:
            info["outcome"] = "link_loss"
            return self.observe(), -cfg.r_fail, True, info
        if j == self.dest:
            info["outcome"] = "delivered"
            return self.observe(), cfg.r_success, True, info
        p_fwd = cfg.p_fwd_malicious if self.malicious[j] else cfg.p_fwd_normal
        if self.rng.random() >= p_fwd:
            info["outcome"] = "malicious_drop" if self.malicious[j] else "relay_drop"
            return self.observe(), -cfg.r_fail, True, info
        self.cur = j
        reward = -cfg.c_hop - cfg.c_energy * (1.0 - self.energy[j])
        if self.hops >= cfg.max_hops:
            info["outcome"] = "ttl"
            return self.observe(), -cfg.r_fail, True, info
        info["outcome"] = "forwarded"
        return self.observe(), float(reward), False, info

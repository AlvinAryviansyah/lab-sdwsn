#!/usr/bin/env python3
"""
tldqn_routing.py
----------------
Transfer-Learning Deep Q-Network (TLDQN) untuk routing hop-by-hop di topologi
honeycomb (Kipongo et al. 2024, Sec. III-C, Algoritma 1, pers. 17-20).

Algoritma 1 (V = terverifikasi dari paper, U = tidak disebut -> asumsi):
  b.4-6  Inisialisasi Q-network; transfer learning: pretrain di tugas SUMBER,
         theta_online = W_source                                         (V)
  b.7    theta_target = theta_online                                      (V)
  b.11   epsilon-greedy                                                    (V)
  b.13   simpan (s, a, r, s') di replay memory D                           (V)
  b.14   ambil minibatch dari D                                            (V)
  b.17   Q_target = r + gamma * max_a' Q(s', a')                pers. (17) (V)
         (dihitung dengan target network; untuk transisi terminal Q_target = r;
          max hanya atas aksi valid di s')                                 (U)
  b.20   kerugian MSE                                           pers. (18) (V)
  b.23   theta = theta - eta * grad L  (SGD)                pers. (19)-(20) (V)
  b.24   tiap C langkah: theta_target = theta_online                       (V)
  Ukuran jaringan, eta, gamma, epsilon, |D|, ukuran minibatch, C, episode  (U)

Tugas sumber dan target (transfer, U: paper tidak merinci):
  sumber : honeycomb 50 sensor pertama, skenario (node malicious, stabilitas
           tautan) dari scenario_seed sumber
  target : honeycomb 100 sensor, skenario berbeda (scenario_seed target)
  DQN tanpa transfer dan TLDQN dilatih dengan anggaran episode target yang SAMA.

Evaluasi: peluang pengiriman (PDR) dan imbalan harapan dihitung EKSAK untuk
kebijakan greedy pada himpunan evaluasi tetap (semua sensor sumber x beberapa
kondisi jaringan). Referensi (BUKAN bagian metode):
  - random   : pilih slot valid secara acak seragam (batas bawah)
  - greedy   : pilih kandidat dengan progres terbesar (slot 0)
  - oracle   : program dinamis dengan ground truth p_fwd (batas atas INFORMASI
               PENUH; memakai label malicious yang tidak dimiliki agen)
  - bayes    : program dinamis dengan E[p_fwd | trust teramati] (INFORMASI SETARA
               dalam satu episode); keputusan dari keyakinan, nilai dievaluasi pada
               dinamika sebenarnya. Bukan batas atas ketat: agen dapat belajar lintas
               episode dari stabilitas tautan yang tetap.
  Tidak ada referensi yang memakai informasi masa depan: semuanya menghitung nilai
  harapan, tidak memakai hasil acak yang terjadi dan tidak membaca RNG lingkungan.

Pemakaian:
  python tldqn_routing.py                 # konfigurasi yang dilaporkan (5 seed, ~30 s)
  python tldqn_routing.py --seeds 3 --target-episodes 1000
"""

import os
import sys
import json
import time
import copy
import argparse
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn

from honeycomb_topology import K_ACTIONS, save_routing_schema
from routing_env import (HoneycombRoutingEnv, EnvConfig, STATE_DIM, SLOT_FEATURES,
                         NODE_FEATURES)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "..", "ns3-sim", "results")
FIG_DIR = os.path.join(HERE, "..", "figures")
REPORT_DIR = os.path.join(HERE, "reports")
CKPT_DIR = os.path.join(HERE, "checkpoints", "tldqn")


# ---------------------------------------------------------------------------
# Konfigurasi dan jaringan
# ---------------------------------------------------------------------------
@dataclass
class DQNConfig:
    hidden: int = 128               # U
    n_hidden: int = 2               # U (Fig. 5 menggambar lapisan tersembunyi; jumlah tidak disebut)
    gamma: float = 0.9              # U
    lr: float = 0.05                # U (eta); = konfigurasi yang dilaporkan
    optimizer: str = "sgd"          # V: pers. (19) SGD
    batch_size: int = 64            # U
    buffer_size: int = 20000        # U
    min_buffer: int = 500           # U
    target_update_steps: int = 200  # U (C)
    eps_start: float = 1.0          # U
    eps_end: float = 0.05           # U
    eps_decay_episodes: int = 1500  # U (linear); = konfigurasi yang dilaporkan
    seed: int = 1

    def to_dict(self):
        return asdict(self)


class QNet(nn.Module):
    """Q(s, .) untuk K slot next-hop."""
    def __init__(self, in_dim=STATE_DIM, out_dim=K_ACTIONS, hidden=128, n_hidden=2):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def masked_argmax(q, mask):
    """argmax hanya atas aksi valid. q, mask: (B, K) atau (K,)."""
    q = torch.as_tensor(q, dtype=torch.float32)
    mask = torch.as_tensor(mask, dtype=torch.bool)
    if not bool(mask.any(dim=-1).all()):
        raise ValueError("ada keadaan tanpa aksi valid")
    return torch.where(mask, q, torch.full_like(q, -1e9)).argmax(dim=-1)


def bellman_targets(r, done, q_next, mask_next, gamma):
    """Pers. (17) bentuk DQN: y = r + gamma * max_{a' valid} Q_target(s', a');
    y = r untuk transisi terminal."""
    q_next = torch.where(mask_next, q_next, torch.full_like(q_next, -1e9))
    best = q_next.max(dim=1).values
    best = torch.where(done | ~mask_next.any(dim=1), torch.zeros_like(best), best)
    return r + gamma * best


class ReplayBuffer:
    """Replay memory D (Alg. 1 b.13-14), larik numpy berukuran tetap."""
    def __init__(self, cap, state_dim, k, seed):
        self.cap, self.n, self.i = cap, 0, 0
        self.s = np.zeros((cap, state_dim), np.float32)
        self.s2 = np.zeros((cap, state_dim), np.float32)
        self.a = np.zeros(cap, np.int64)
        self.r = np.zeros(cap, np.float32)
        self.d = np.zeros(cap, bool)
        self.m2 = np.zeros((cap, k), bool)
        self.rng = np.random.default_rng(seed)

    def push(self, s, a, r, s2, done, mask2):
        j = self.i
        self.s[j], self.a[j], self.r[j], self.s2[j], self.d[j], self.m2[j] = s, a, r, s2, done, mask2
        self.i = (self.i + 1) % self.cap
        self.n = min(self.n + 1, self.cap)

    def __len__(self):
        return self.n

    def sample(self, b):
        idx = self.rng.integers(0, self.n, b)
        t = lambda x, dt=torch.float32: torch.as_tensor(x[idx], dtype=dt)
        return (t(self.s), t(self.a, torch.int64), t(self.r), t(self.s2),
                t(self.d, torch.bool), t(self.m2, torch.bool))


class DQNAgent:
    def __init__(self, cfg, init_state_dict=None):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        self.online = QNet(STATE_DIM, K_ACTIONS, cfg.hidden, cfg.n_hidden)
        if init_state_dict is not None:                      # Alg. 1 b.6 transfer
            self.online.load_state_dict(init_state_dict)
        self.target = QNet(STATE_DIM, K_ACTIONS, cfg.hidden, cfg.n_hidden)
        self.target.load_state_dict(self.online.state_dict())   # Alg. 1 b.7
        for p in self.target.parameters():
            p.requires_grad_(False)
        if cfg.optimizer == "sgd":
            self.opt = torch.optim.SGD(self.online.parameters(), lr=cfg.lr)
        elif cfg.optimizer == "adam":
            self.opt = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        else:
            raise ValueError(cfg.optimizer)
        self.loss_fn = nn.MSELoss()                          # pers. (18)
        self.grad_steps = 0
        self.target_syncs = 0
        self.rng = np.random.default_rng(cfg.seed + 101)

    @torch.no_grad()
    def greedy(self, obs, mask):
        q = self.online(torch.as_tensor(np.atleast_2d(obs), dtype=torch.float32))
        return masked_argmax(q, np.atleast_2d(mask)).numpy()

    def act(self, obs, mask, eps):
        """epsilon-greedy atas aksi valid (Alg. 1 b.11)."""
        if self.rng.random() < eps:
            return int(self.rng.choice(np.flatnonzero(mask)))
        return int(self.greedy(obs, mask)[0])

    def update(self, batch):
        s, a, r, s2, d, m2 = batch
        q = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            y = bellman_targets(r, d, self.target(s2), m2, self.cfg.gamma)
        loss = self.loss_fn(q, y)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()                                      # pers. (19)
        self.grad_steps += 1
        if self.grad_steps % self.cfg.target_update_steps == 0:   # Alg. 1 b.24
            self.target.load_state_dict(self.online.state_dict())
            self.target_syncs += 1
        return float(loss.item())


# ---------------------------------------------------------------------------
# Evaluasi eksak
# ---------------------------------------------------------------------------
def make_eval_set(env, n_states=3, seed=12345):
    """Himpunan evaluasi tetap: semua sensor x n_states kondisi jaringan."""
    rng = np.random.default_rng(seed)
    states = [env.sample_network_state(rng) for _ in range(n_states)]
    return [(s, st) for st in states for s in range(env.n)]


def _hop_terms(env, c, j):
    """(peluang lolos tautan, peluang diteruskan relay, imbalan hop) untuk c->j."""
    l = env.prr[c, j] * env.stability[c, j]
    if j == env.dest:
        return l, 1.0, 0.0
    f = env.cfg.p_fwd_malicious if env.malicious[j] else env.cfg.p_fwd_normal
    return l, f, -env.cfg.c_hop - env.cfg.c_energy * (1.0 - env.energy[j])


def evaluate_policy(env, policy_batch, eval_set):
    """Kebijakan deterministik: ikuti jalurnya dan hitung PDR + imbalan harapan
    secara eksak. policy_batch(obs (B,D), mask (B,K)) -> aksi (B,)."""
    cfg = env.cfg
    by_state = {}
    for s, st in eval_set:
        by_state.setdefault(id(st), (st, []))[1].append(s)
    pdr, ret, hops = [], [], []
    for st, sources in by_state.values():
        env.set_network_state(*st)
        cur = {s: s for s in sources}
        alive = {s: 1.0 for s in sources}
        exp_ret = {s: 0.0 for s in sources}
        nhop = {s: 0 for s in sources}
        active = list(sources)
        while active:
            obs, masks = [], []
            for s in active:
                env.cur, env.dest = cur[s], int(env.topo.dest[s])
                obs.append(env.observe()); masks.append(env.action_mask())
            acts = policy_batch(np.array(obs), np.array(masks))
            nxt = []
            for s, a in zip(active, acts):
                env.cur, env.dest = cur[s], int(env.topo.dest[s])
                j = int(env.slots()[int(a)])
                if j < 0:
                    raise ValueError("kebijakan memilih slot tidak valid")
                l, f, hr = _hop_terms(env, cur[s], j)
                A = alive[s]
                nhop[s] += 1
                if j == env.dest:
                    exp_ret[s] += A * (l * cfg.r_success - (1 - l) * cfg.r_fail)
                    alive[s] = A * l
                    continue
                exp_ret[s] += A * (-(1 - l) * cfg.r_fail - l * (1 - f) * cfg.r_fail)
                if nhop[s] >= cfg.max_hops:
                    exp_ret[s] += A * l * f * (-cfg.r_fail)
                    alive[s] = 0.0
                    continue
                exp_ret[s] += A * l * f * hr
                alive[s] = A * l * f
                cur[s] = j
                nxt.append(s)
            active = nxt
        pdr += [alive[s] for s in sources]
        ret += [exp_ret[s] for s in sources]
        hops += [nhop[s] for s in sources]
    return {"pdr": float(np.mean(pdr)), "expected_return": float(np.mean(ret)),
            "mean_hops": float(np.mean(hops))}


def posterior_pfwd(env, trust):
    """E[p_fwd | trust teramati] di bawah prior skenario (fraksi malicious) dan
    model observasi lingkungan (trust = (s+1)/(n+2), s ~ Binomial(n, p_fwd)).
    Hanya memakai informasi yang juga dilihat agen (trust di keadaan)."""
    c = env.cfg
    n, pi = c.trust_obs, c.malicious_fraction
    s = np.clip(np.rint(np.asarray(trust, float) * (n + 2) - 1), 0, n).astype(int)
    from math import comb
    binom = lambda p: np.array([comb(n, k) * p ** k * (1 - p) ** (n - k) for k in s])
    lm, ln = pi * binom(c.p_fwd_malicious), (1 - pi) * binom(c.p_fwd_normal)
    pm = lm / (lm + ln)
    out = pm * c.p_fwd_malicious + (1 - pm) * c.p_fwd_normal
    out[env.n:] = 1.0                       # sink tidak pernah menjadi relay
    return out


def _dp_values(env, kind, return_policy=False):
    """Nilai eksak V[(c, t)] dan PDR P[(c, t)] kebijakan referensi, dievaluasi pada
    dinamika SEBENARNYA (ground truth p_fwd). Paket dapat melewati relay yang sink
    terdekatnya berbeda, sehingga nilai bergantung pada (node, tujuan). Graf kandidat
    asiklik (syarat progres), jadi diproses urut jarak ke t.

    kind: "random" (rata-rata seragam), "oracle" (keputusan dgn p_fwd sebenarnya),
          "bayes" (keputusan dgn E[p_fwd | trust]).

    TTL (cfg.max_hops) TIDAK dimodelkan: valid hanya bila rute terpanjang di graf
    kandidat <= max_hops (tepat 12 pada n=100). Dijaga oleh tes
    test_longest_route_within_ttl dan assert di bawah."""
    if longest_route_hops(env.topo) > env.cfg.max_hops:
        raise ValueError("rute terpanjang > max_hops: DP tanpa TTL tidak berlaku")
    cfg, topo = env.cfg, env.topo
    p_true = np.where(env.malicious, cfg.p_fwd_malicious, cfg.p_fwd_normal)
    p_dec = {"oracle": p_true, "bayes": posterior_pfwd(env, env.trust)}.get(kind)
    if kind not in ("random", "oracle", "bayes"):
        raise ValueError(kind)
    Vt, Pt, Vd, pol = {}, {}, {}, {}
    pairs = topo.reachable_pairs()
    for t in range(topo.n, topo.n_nodes):
        for c in sorted((c for c, tt in pairs if tt == t), key=lambda c: (topo.dist[c, t], c)):
            qt, pt, qd = [], [], []
            for j in topo.candidates(c, t):
                l = env.prr[c, j] * env.stability[c, j]
                if j == t:
                    v = l * cfg.r_success - (1 - l) * cfg.r_fail
                    qt.append(v); pt.append(l); qd.append(v)
                    continue
                hr = -cfg.c_hop - cfg.c_energy * (1.0 - env.energy[j])
                f = p_true[j]
                qt.append(-(1 - l) * cfg.r_fail - l * (1 - f) * cfg.r_fail + l * f * (hr + Vt[(j, t)]))
                pt.append(l * f * Pt[(j, t)])
                if p_dec is not None:
                    fd = p_dec[j]
                    qd.append(-(1 - l) * cfg.r_fail - l * (1 - fd) * cfg.r_fail + l * fd * (hr + Vd[(j, t)]))
            if kind == "random":
                Vt[(c, t)], Pt[(c, t)] = float(np.mean(qt)), float(np.mean(pt))
            else:
                a = int(np.argmax(qd))
                Vt[(c, t)], Pt[(c, t)], Vd[(c, t)] = qt[a], pt[a], qd[a]
                pol[(c, t)] = a
    return (Vt, Pt, pol) if return_policy else (Vt, Pt)


def longest_route_hops(topo):
    """Jumlah hop maksimum pada rute apa pun di graf kandidat (graf asiklik)."""
    memo = {}

    def depth(c, t):
        if c == t:
            return 0
        if (c, t) not in memo:
            memo[(c, t)] = 1 + max(depth(j, t) for j in topo.candidates(c, t))
        return memo[(c, t)]

    return max(depth(c, t) for c, t in topo.reachable_pairs())


def evaluate_reference(env, eval_set, kind):
    """Referensi random / oracle / bayes pada himpunan evaluasi (nilai eksak)."""
    pdr, ret = [], []
    cache = {}
    for s, st in eval_set:
        if id(st) not in cache:
            env.set_network_state(*st)
            cache[id(st)] = _dp_values(env, kind)
        V, P = cache[id(st)]
        key = (s, int(env.topo.dest[s]))
        pdr.append(P[key]); ret.append(V[key])
    return {"pdr": float(np.mean(pdr)), "expected_return": float(np.mean(ret))}


def greedy_progress_policy(obs, mask):
    """Referensi: slot 0 = kandidat dengan progres terbesar."""
    return np.zeros(len(obs), dtype=np.int64)


def normalized_gain(ret, ref, upper="oracle"):
    """(R - R_random) / (R_upper - R_random); upper = "oracle" (utama) atau "bayes"."""
    lo, hi = ref["random"]["expected_return"], ref[upper]["expected_return"]
    return float((ret - lo) / (hi - lo)) if hi > lo else float("nan")


# ---------------------------------------------------------------------------
# Pelatihan
# ---------------------------------------------------------------------------
def train(env, cfg, episodes, init_state_dict=None, eval_set=None, eval_every=50,
          refs=None, log_fn=print, tag=""):
    agent = DQNAgent(cfg, init_state_dict)
    buf = ReplayBuffer(cfg.buffer_size, STATE_DIM, K_ACTIONS, cfg.seed + 202)
    env.rng = np.random.default_rng(cfg.seed + 303)
    pol = lambda o, m: agent.greedy(o, m)
    hist = []

    def do_eval(ep):
        if eval_set is None:
            return
        r = evaluate_policy(env, pol, eval_set)
        r["episode"] = ep
        if refs:
            r["normalized_gain"] = normalized_gain(r["expected_return"], refs)
            if "bayes" in refs:
                r["normalized_gain_bayes"] = normalized_gain(r["expected_return"], refs, "bayes")
        hist.append(r)

    do_eval(0)
    for ep in range(1, episodes + 1):
        frac = min(1.0, (ep - 1) / max(cfg.eps_decay_episodes, 1))
        eps = cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)
        obs, done = env.reset(), False
        while not done:
            mask = env.action_mask()
            a = agent.act(obs, mask, eps)
            obs2, r, done, _ = env.step(a)
            mask2 = env.action_mask() if not done else np.zeros(K_ACTIONS, bool)
            buf.push(obs, a, r, obs2, done, mask2)
            obs = obs2
            if len(buf) >= cfg.min_buffer:
                agent.update(buf.sample(cfg.batch_size))
        if ep % eval_every == 0 or ep == episodes:
            do_eval(ep)
            if hist and log_fn:
                h = hist[-1]
                g = f" gain={h['normalized_gain']:.3f}" if "normalized_gain" in h else ""
                log_fn(f"  {tag} ep {ep:5d} eps={eps:.2f} PDR={h['pdr']:.3f} "
                       f"E[R]={h['expected_return']:.3f}{g}")
    return agent, hist


def first_reach(hist, key, thr):
    for h in hist:
        if h.get(key, -np.inf) >= thr:
            return h["episode"]
    return None


# ---------------------------------------------------------------------------
# Eksperimen transfer: DQN vs TLDQN
# ---------------------------------------------------------------------------
def run_experiment(seeds, n_source, n_target, source_episodes, target_episodes,
                   eval_every, cfg_kwargs, env_cfg, gain_threshold, log_fn=print):
    runs = []
    for seed in seeds:
        cfg = DQNConfig(seed=seed, **cfg_kwargs)
        src_env = HoneycombRoutingEnv(n_source, scenario_seed=1000 + seed, cfg=env_cfg)
        tgt_env = HoneycombRoutingEnv(n_target, scenario_seed=2000 + seed, cfg=env_cfg)
        ev = make_eval_set(tgt_env, seed=5000 + seed)
        refs = {k: evaluate_reference(tgt_env, ev, k) for k in ("random", "oracle", "bayes")}
        refs["greedy"] = evaluate_policy(tgt_env, greedy_progress_policy, ev)
        log_fn(f"[seed {seed}] referensi target: random PDR={refs['random']['pdr']:.3f}  "
               f"greedy PDR={refs['greedy']['pdr']:.3f}  bayes PDR={refs['bayes']['pdr']:.3f}  "
              f"oracle PDR={refs['oracle']['pdr']:.3f}")
        t0 = time.time()
        src_agent, _ = train(src_env, cfg, source_episodes, log_fn=None)
        t_src = time.time() - t0
        src_state = copy.deepcopy(src_agent.online.state_dict())
        t0 = time.time()
        dqn, h_dqn = train(tgt_env, cfg, target_episodes, None, ev, eval_every, refs,
                           log_fn, f"[s{seed} DQN  ]")
        t_dqn = time.time() - t0
        t0 = time.time()
        tl, h_tl = train(tgt_env, cfg, target_episodes, src_state, ev, eval_every, refs,
                         log_fn, f"[s{seed} TLDQN]")
        t_tl = time.time() - t0
        runs.append({"seed": seed, "refs": refs, "dqn": h_dqn, "tldqn": h_tl,
                     "seconds": {"source": t_src, "dqn": t_dqn, "tldqn": t_tl},
                     "models": {"source": src_state, "dqn": dqn.online.state_dict(),
                                "tldqn": tl.online.state_dict()},
                     "scenario_seeds": {"source": 1000 + seed, "target": 2000 + seed}})
    return runs


# Ambang tambahan untuk analisis konvergensi. Ambang UTAMA (pra-ditetapkan) adalah
# --gain-threshold (default 0.9); ambang lain ditambahkan SETELAH melihat bahwa 0.9
# tidak tercapai, jadi diperlakukan sebagai analisis post hoc.
EXTRA_THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)


def summarize(runs, gain_threshold):
    def agg(method, key):
        m = np.array([[h[key] for h in r[method]] for r in runs])
        return m.mean(0), m.std(0)

    eps_axis = [h["episode"] for h in runs[0]["dqn"]]
    out = {"episodes": eps_axis, "gain_threshold": gain_threshold, "per_seed": []}
    for method in ("dqn", "tldqn"):
        for key in ("pdr", "expected_return", "normalized_gain"):
            mu, sd = agg(method, key)
            out[f"{method}_{key}_mean"] = mu.tolist()
            out[f"{method}_{key}_std"] = sd.tolist()
    for r in runs:
        row = {"seed": r["seed"], "refs": r["refs"], "seconds": r["seconds"],
               "scenario_seeds": r["scenario_seeds"],
               "curves": {m: r[m] for m in ("dqn", "tldqn")}}
        for method in ("dqn", "tldqn"):
            h = r[method]
            row[method] = {
                "episodes_to_gain": {f"{th:.1f}": first_reach(h, "normalized_gain", th)
                                     for th in EXTRA_THRESHOLDS},
                "initial_gain": h[0]["normalized_gain"],
                "final_gain": float(np.mean([x["normalized_gain"] for x in h[-3:]])),
                "final_gain_bayes": float(np.mean([x["normalized_gain_bayes"] for x in h[-3:]])),
                "final_pdr": float(np.mean([x["pdr"] for x in h[-3:]])),
                "auc_gain": float(np.mean([x["normalized_gain"] for x in h])),
                "episodes_to_threshold": first_reach(h, "normalized_gain", gain_threshold),
            }
        out["per_seed"].append(row)
    return out


def plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.array(summary["episodes"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, key, lab in ((axes[0], "pdr", "PDR eksak (kebijakan greedy)"),
                         (axes[1], "normalized_gain",
                          "Gain (0 = random, 1 = oracle informasi penuh)")):
        for method, color, name in (("dqn", "tab:orange", "DQN (tanpa transfer)"),
                                    ("tldqn", "tab:blue", "TLDQN (transfer)")):
            mu = np.array(summary[f"{method}_{key}_mean"]); sd = np.array(summary[f"{method}_{key}_std"])
            ax.plot(x, mu, color=color, label=name)
            ax.fill_between(x, mu - sd, mu + sd, color=color, alpha=0.18)
        ax.set_xlabel("Episode pelatihan di tugas target")
        ax.set_ylabel(lab)
        ax.grid(True, linestyle=":", alpha=0.6)
    axes[1].axhline(summary["gain_threshold"], color="gray", ls="--", lw=0.8,
                    label=f"ambang konvergensi {summary['gain_threshold']}")
    for name, val, ls in (("random", summary["refs_mean"]["random_pdr"], ":"),
                          ("greedy", summary["refs_mean"]["greedy_pdr"], "-."),
                          ("bayes (info setara)", summary["refs_mean"]["bayes_pdr"], (0, (1, 1))),
                          ("oracle (info penuh)", summary["refs_mean"]["oracle_pdr"], "--")):
        axes[0].axhline(val, color="gray", ls=ls, lw=0.9, label=f"ref. {name}")
    axes[0].legend(fontsize=7); axes[1].legend(fontsize=7)
    n = len(summary["per_seed"])
    fig.suptitle(f"TLDQN vs DQN pada routing honeycomb (mean±std, {n} seed)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-source", type=int, default=50)
    ap.add_argument("--n-target", type=int, default=100)
    ap.add_argument("--source-episodes", type=int, default=3000)
    ap.add_argument("--target-episodes", type=int, default=3000)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--gain-threshold", type=float, default=0.9)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    ap.add_argument("--eps-decay-episodes", type=int, default=1500)
    args = ap.parse_args(argv)

    env_cfg = EnvConfig()
    cfg_kwargs = {"lr": args.lr, "optimizer": args.optimizer,
                  "eps_decay_episodes": args.eps_decay_episodes}
    seeds = list(range(1, args.seeds + 1))
    t0 = time.time()
    runs = run_experiment(seeds, args.n_source, args.n_target, args.source_episodes,
                          args.target_episodes, args.eval_every, cfg_kwargs, env_cfg,
                          args.gain_threshold)
    summary = summarize(runs, args.gain_threshold)
    summary["refs_mean"] = {f"{k}_{m}": float(np.mean([r["refs"][k][m] for r in runs]))
                            for k in ("random", "greedy", "bayes", "oracle") for m in ("pdr", "expected_return")}
    summary["config"] = {"dqn": DQNConfig(**cfg_kwargs).to_dict(), "env": env_cfg.to_dict(),
                         "n_source": args.n_source, "n_target": args.n_target,
                         "source_episodes": args.source_episodes,
                         "target_episodes": args.target_episodes, "seeds": seeds,
                         "state_dim": STATE_DIM, "action_dim": K_ACTIONS,
                         "slot_features": SLOT_FEATURES, "node_features": NODE_FEATURES,
                         "total_seconds": round(time.time() - t0, 1)}

    os.makedirs(REPORT_DIR, exist_ok=True); os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True); os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, "tldqn_eval.json"), "w") as f:
        json.dump(summary, f, indent=1)
    for r in runs:
        for name, sd in r["models"].items():
            torch.save(sd, os.path.join(CKPT_DIR, f"{name}_seed{r['seed']}.pt"))
    with open(os.path.join(CKPT_DIR, "config.json"), "w") as f:
        json.dump(summary["config"], f, indent=1)
    extra = {"normalization": {"progress": "(d(c,t) - d(j,t)) / R", "remaining": "d(c,t) / (4R)",
                               "energy": "fraksi energi awal; sink = 1", "trust": "(s+1)/(n_obs+2); sink = 1"},
             "link_model": {"prr": "1 / (1 + exp((d - prr_d50) / prr_width))",
                            "success": "prr * stability", **{k: env_cfg.to_dict()[k] for k in
                            ("prr_d50", "prr_width", "stability_min")}},
             "env_config": env_cfg.to_dict()}
    for path in (os.path.join(CKPT_DIR, "routing_schema.json"),
                 os.path.join(REPORT_DIR, "routing_schema.json")):
        save_routing_schema(path, SLOT_FEATURES, NODE_FEATURES, extra)
    plot(summary, os.path.join(FIG_DIR, "tldqn_reward.png"))

    rows = summary["per_seed"]
    def m(method, key):
        v = [r[method][key] for r in rows if r[method][key] is not None]
        return (np.mean(v), np.std(v), len(v)) if v else (float("nan"), 0.0, 0)
    with open(os.path.join(RESULTS_DIR, "tldqn_metrics.csv"), "w") as f:
        f.write("metric,dqn_mean,dqn_std,dqn_n,tldqn_mean,tldqn_std,tldqn_n\n")
        for key in ("initial_gain", "final_gain", "final_gain_bayes", "final_pdr", "auc_gain",
                    "episodes_to_threshold"):
            a, b = m("dqn", key), m("tldqn", key)
            f.write(f"{key},{a[0]:.4f},{a[1]:.4f},{a[2]},{b[0]:.4f},{b[1]:.4f},{b[2]}\n")

    print("\n=== Ringkasan (mean ± std antar seed) ===")
    rm = summary["refs_mean"]
    print(f"Referensi PDR: random={rm['random_pdr']:.3f} greedy={rm['greedy_pdr']:.3f} "
          f"bayes={rm['bayes_pdr']:.3f} oracle={rm['oracle_pdr']:.3f}")
    for key in ("initial_gain", "final_gain", "final_gain_bayes", "final_pdr", "auc_gain",
                "episodes_to_threshold"):
        a, b = m("dqn", key), m("tldqn", key)
        print(f"  {key:22s} DQN {a[0]:8.3f} ± {a[1]:6.3f} (n={a[2]})   "
              f"TLDQN {b[0]:8.3f} ± {b[1]:6.3f} (n={b[2]})")
    print("Episode pertama mencapai gain (post hoc kecuali ambang utama), per seed:")
    for th in EXTRA_THRESHOLDS:
        k = f"{th:.1f}"
        d = [r["dqn"]["episodes_to_gain"][k] for r in rows]
        l = [r["tldqn"]["episodes_to_gain"][k] for r in rows]
        tag = " (utama)" if abs(th - args.gain_threshold) < 1e-9 else ""
        print(f"  gain>={k}{tag:8s} DQN {d}   TLDQN {l}")
    print(f"Laporan -> ml/reports/tldqn_eval.json, figures/tldqn_reward.png, "
          f"ns3-sim/results/tldqn_metrics.csv  ({summary['config']['total_seconds']} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""LEGACY TLDQN (lingkungan routing ACAK, aksi = pilih edge sink).
Dipertahankan HANYA agar export_decisions.py (alur NS-3 lama) tetap berjalan
sampai fase integrasi NS-3. Jangan dipakai untuk pekerjaan baru; gunakan
tldqn_routing.py + routing_env.py + honeycomb_topology.py.
"""
import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "..", "ns3-sim", "results")
FIG_DIR = os.path.join(HERE, "..", "figures")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

torch.manual_seed(1); np.random.seed(1); random.seed(1)

STATE_DIM = 8   # [trust, distance, link_quality, energy, hops, stab, pdr, thr]
K_ACTIONS = 6   # hingga 6 next-forwarder (tetangga hexagon)


class RoutingEnv:
    """Lingkungan routing sintetis: tiap langkah ada K jalur kandidat."""
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def reset(self):
        self._gen()
        return self.state.flatten()

    def _gen(self):
        # Tiap aksi punya fitur jalur. Satu aksi dibuat "terbaik" secara acak.
        self.state = self.rng.random((K_ACTIONS, STATE_DIM)).astype(np.float32)
        self.best = int(self.rng.integers(K_ACTIONS))
        # Perkuat sinyal pada jalur terbaik (trust/stab/pdr/throughput tinggi,
        # hops/distance/energi rendah).
        self.state[self.best, 0] = 0.9 + 0.1 * self.rng.random()  # trust
        self.state[self.best, 5] = 0.9 + 0.1 * self.rng.random()  # stability
        self.state[self.best, 6] = 0.9 + 0.1 * self.rng.random()  # pdr
        self.state[self.best, 7] = 0.9 + 0.1 * self.rng.random()  # throughput
        self.state[self.best, 1] = 0.1 * self.rng.random()        # distance
        self.state[self.best, 4] = 0.1 * self.rng.random()        # hops

    def step(self, action):
        f = self.state[action]
        trust, dist, lq, energy, hops, stab, pdr, thr = f
        reward = (trust + stab + pdr + thr) - 0.5 * (dist + hops) - 0.2 * energy
        self._gen()
        return self.state.flatten(), float(reward), False


class QNet(nn.Module):
    def __init__(self, in_dim, out_dim, h=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, out_dim),
        )
    def forward(self, x): return self.net(x)


class ReplayBuffer:
    def __init__(self, cap=10000):
        self.buf = []; self.cap = cap
    def push(self, *tr):
        self.buf.append(tr)
        if len(self.buf) > self.cap: self.buf.pop(0)
    def sample(self, n):
        batch = random.sample(self.buf, n)
        s, a, r, s2 = zip(*batch)
        return (torch.tensor(np.array(s), dtype=torch.float32),
                torch.tensor(a, dtype=torch.int64),
                torch.tensor(r, dtype=torch.float32),
                torch.tensor(np.array(s2), dtype=torch.float32))
    def __len__(self): return len(self.buf)


def train_dqn(env, episodes=300, steps=30, gamma=0.9, lr=1e-3,
              eps0=1.0, eps_min=0.05, target_update=20,
              init_weights=None):
    """Latih DQN. init_weights != None -> transfer learning (warm start)."""
    in_dim = STATE_DIM * K_ACTIONS
    online = QNet(in_dim, K_ACTIONS)
    if init_weights is not None:
        online.load_state_dict(init_weights)   # TRANSFER LEARNING
    target = QNet(in_dim, K_ACTIONS)
    target.load_state_dict(online.state_dict())
    opt = torch.optim.Adam(online.parameters(), lr=lr)
    mse = nn.MSELoss()
    buf = ReplayBuffer()
    eps = eps0
    ep_rewards = []

    for ep in range(episodes):
        s = env.reset(); total = 0.0
        for t in range(steps):
            if random.random() < eps:
                a = random.randrange(K_ACTIONS)
            else:
                with torch.no_grad():
                    a = int(online(torch.tensor(s).unsqueeze(0)).argmax())
            s2, r, _ = env.step(a)
            buf.push(s, a, r, s2)
            s = s2; total += r

            if len(buf) >= 128:
                bs, ba, br, bs2 = buf.sample(128)
                q = online(bs).gather(1, ba.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    q_next = target(bs2).max(1).values   # pers. 17 Bellman
                    tgt = br + gamma * q_next
                loss = mse(q, tgt)                        # pers. 18 MSE
                opt.zero_grad(); loss.backward(); opt.step()  # pers. 19 SGD

        eps = max(eps_min, eps * 0.99)
        if ep % target_update == 0:
            target.load_state_dict(online.state_dict())
        ep_rewards.append(total / steps)
    return online, ep_rewards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    args = ap.parse_args()

    # 1) Task SUMBER (pretrain) - lingkungan dgn seed berbeda.
    print("Pretrain Q-network di task sumber (untuk transfer learning) ...")
    src_env = RoutingEnv(seed=42)
    src_net, _ = train_dqn(src_env, episodes=args.episodes // 2)

    # 2) Task TARGET tanpa transfer (baseline DQN) vs dengan transfer (TLDQN).
    print("Latih DQN biasa (tanpa transfer) di task target ...")
    tgt_env1 = RoutingEnv(seed=7)
    _, r_plain = train_dqn(tgt_env1, episodes=args.episodes)

    print("Latih TLDQN (dgn transfer dari task sumber) di task target ...")
    tgt_env2 = RoutingEnv(seed=7)
    _, r_transfer = train_dqn(tgt_env2, episodes=args.episodes,
                              init_weights=src_net.state_dict())

    # Hasil: reward akhir & kecepatan konvergensi.
    def smooth(x, k=10):
        return np.convolve(x, np.ones(k)/k, mode="valid")
    rp, rt = smooth(r_plain), smooth(r_transfer)
    print(f"\nReward akhir DQN biasa : {np.mean(r_plain[-20:]):.3f}")
    print(f"Reward akhir TLDQN     : {np.mean(r_transfer[-20:]):.3f}")

    # Episode pertama mencapai 90% reward maksimum (ukuran konvergensi).
    def conv_ep(r, frac=0.9):
        target = frac * np.max(r)
        for i, v in enumerate(r):
            if v >= target: return i
        return len(r)
    print(f"Konvergensi DQN biasa  : episode ~{conv_ep(rp)}")
    print(f"Konvergensi TLDQN      : episode ~{conv_ep(rt)} (lebih cepat)")

    plt.figure(figsize=(6.5, 4.2))
    plt.plot(rp, label="DQN (tanpa transfer)", color="orange")
    plt.plot(rt, label="TLDQN (transfer learning)", color="blue")
    plt.xlabel("Episode"); plt.ylabel("Avg reward (per step)")
    plt.title("TLDQN vs DQN — kurva pembelajaran routing")
    plt.legend(); plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "tldqn_reward.png")
    plt.savefig(out, dpi=150); plt.close()
    print("Kurva -> figures/tldqn_reward.png")

    with open(os.path.join(RESULTS_DIR, "tldqn_metrics.csv"), "w") as f:
        f.write("metric,value\n")
        f.write(f"reward_dqn,{np.mean(r_plain[-20:]):.4f}\n")
        f.write(f"reward_tldqn,{np.mean(r_transfer[-20:]):.4f}\n")
        f.write(f"convergence_ep_dqn,{conv_ep(rp)}\n")
        f.write(f"convergence_ep_tldqn,{conv_ep(rt)}\n")
    print("Metrik -> ns3-sim/results/tldqn_metrics.csv")


if __name__ == "__main__":
    raise SystemExit("legacy_tldqn.py hanya untuk diimpor export_decisions.py; untuk TLDQN baru jalankan tldqn_routing.py.")

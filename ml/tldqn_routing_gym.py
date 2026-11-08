#!/usr/bin/env python3
"""
tldqn_routing_gym.py
--------------------
Melatih TLDQN yang sama (QNet, replay, Bellman, transfer) terhadap ns-3
lewat ns3-gym, bukan terhadap HoneycombRoutingEnv sintetis.

Satu episode = satu proses honeycomb-sdwsn-gym (satu paket sampai sink,
hilang, atau TTL). ns-3 mengirim state 44 angka; skrip ini membalas
indeks slot 0..5; ns-3 mengembalikan reward.

  cp ../ns3-sim/honeycomb-sdwsn-gym.cc $NS3_HOME/scratch/
  cd $NS3_HOME && ./ns3 build honeycomb-sdwsn-gym
  python tldqn_routing_gym.py --ns3-dir $NS3_HOME --episodes 5 --num-nodes 20

Butuh paket Python `ns3gym` di environment yang sama dengan torch.
Tanpa --start-sim, jalankan ns-3 sendiri lalu skrip ini hanya menghubungi port:

  ./ns3 run "honeycomb-sdwsn-gym --numNodes=20 --seed=1 --openGymPort=5555"
  python tldqn_routing_gym.py --no-start --port 5555 --episodes 1
"""

import argparse
import os
import subprocess
import sys

import numpy as np

from honeycomb_topology import K_ACTIONS
from routing_env import STATE_DIM
from tldqn_routing import DQNAgent, DQNConfig, ReplayBuffer

HERE = os.path.dirname(os.path.abspath(__file__))
SLOT_FEATURES = 7


def mask_from_obs(obs):
    """Bit valid adalah fitur pertama tiap slot."""
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    if obs.size != STATE_DIM:
        raise ValueError(f"state ns-3 berukuran {obs.size}, diharapkan {STATE_DIM}")
    slots = obs[:K_ACTIONS * SLOT_FEATURES].reshape(K_ACTIONS, SLOT_FEATURES)
    valid = slots[:, 0] > 0.5
    if not bool(valid.any()):
        raise ValueError("ns-3 mengirim keadaan tanpa slot valid")
    return valid


def connect(port, start_sim, ns3_dir, num_nodes, seed):
    try:
        from ns3gym import ns3env
    except ImportError as exc:
        raise SystemExit(
            "Paket ns3gym belum terpasang di environment ini. "
            "Pasang ns3-gym (modul OpenGym di ns-3 dan pip install -e model/ns3gym)."
        ) from exc

    proc = None
    if start_sim:
        if not ns3_dir:
            raise SystemExit("Butuh --ns3-dir (folder yang berisi ./ns3) saat --start-sim.")
        ns3 = os.path.join(ns3_dir, "ns3")
        if not os.path.isfile(ns3):
            raise SystemExit(f"Tidak menemukan {ns3}. Salin honeycomb-sdwsn-gym.cc ke scratch/ dan build dulu.")
        cmd = [ns3, "run",
               f"honeycomb-sdwsn-gym --numNodes={num_nodes} --seed={seed} --openGymPort={port}"]
        proc = subprocess.Popen(cmd, cwd=ns3_dir)
    # Ns3Env mengikat port dulu, lalu menunggu SimInit dari ns-3.
    env = ns3env.Ns3Env(port=port, stepTime=0, startSim=False,
                        simSeed=seed, simArgs={}, debug=False)
    return env, proc


def run_episode(agent, port, start_sim, ns3_dir, num_nodes, seed, eps, buf, cfg):
    env, proc = connect(port, start_sim, ns3_dir, num_nodes, seed)
    total = 0.0
    steps = 0
    outcome = "unknown"
    try:
        obs = np.asarray(env.reset(), dtype=np.float32).reshape(-1)
        if env.ns3ZmqBridge.is_game_over():
            return (float(env.ns3ZmqBridge.get_reward()), 0,
                    str(env.ns3ZmqBridge.get_extra_info()))
        mask = mask_from_obs(obs)
        done = False
        while not done:
            action = agent.act(obs, mask, eps)
            nxt, reward, done, info = env.step(int(action))
            nxt = np.asarray(nxt, dtype=np.float32).reshape(-1)
            reward = float(reward)
            if isinstance(info, dict):
                outcome = str(info.get("outcome", info))
            else:
                text = str(info)
                outcome = text
                for part in text.split(";"):
                    if part.startswith("outcome="):
                        outcome = part.split("=", 1)[1]
            mask2 = mask if done else mask_from_obs(nxt)
            buf.push(obs, action, reward, nxt, bool(done), mask2)
            if len(buf) >= cfg.min_buffer:
                agent.update(buf.sample(cfg.batch_size))
            obs, mask = nxt, mask2
            total += reward
            steps += 1
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    return total, steps, outcome


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5,
                    help="jumlah episode (tiap episode satu proses ns-3)")
    ap.add_argument("--pretrain-episodes", type=int, default=0,
                    help="episode sumber sebelum bobot dipakai di target (transfer)")
    ap.add_argument("--num-nodes", type=int, default=20)
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ns3-dir", default=os.environ.get("NS3_HOME", ""))
    ap.add_argument("--start-sim", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--eps", type=float, default=0.2)
    args = ap.parse_args()

    cfg = DQNConfig(seed=args.seed, min_buffer=1, batch_size=8, target_update_steps=20)
    source = None
    if args.pretrain_episodes:
        source_agent = DQNAgent(cfg)
        buf = ReplayBuffer(cfg.buffer_size, STATE_DIM, K_ACTIONS, cfg.seed)
        print(f"Pretrain {args.pretrain_episodes} episode pada seed sumber ...")
        for ep in range(args.pretrain_episodes):
            ret, steps, outcome = run_episode(
                source_agent, args.port, args.start_sim, args.ns3_dir,
                max(1, args.num_nodes // 2), args.seed + 1000 + ep, args.eps, buf, cfg)
            print(f"  sumber {ep + 1}: return={ret:.3f} steps={steps} {outcome}")
        source = source_agent.online.state_dict()

    agent = DQNAgent(cfg, init_state_dict=source)
    buf = ReplayBuffer(cfg.buffer_size, STATE_DIM, K_ACTIONS, cfg.seed + 7)
    print(f"Latih TLDQN online, {args.episodes} episode, {args.num_nodes} sensor ...")
    returns = []
    for ep in range(args.episodes):
        ret, steps, outcome = run_episode(
            agent, args.port, args.start_sim, args.ns3_dir,
            args.num_nodes, args.seed + ep, args.eps, buf, cfg)
        returns.append(ret)
        print(f"  target {ep + 1}: return={ret:.3f} steps={steps} {outcome}")
    print(f"Return rata-rata: {float(np.mean(returns)):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

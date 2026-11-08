"""Frozen TLDQN inference for an isolated ns3-gym experiment."""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
ML = ROOT / "ml"
if str(ML) not in sys.path:
    sys.path.insert(0, str(ML))

from export_routes import DEFAULT_MODEL, build_env
from honeycomb_topology import K_ACTIONS
from hybrid_ids import reroute
from routing_env import EnvConfig, STATE_DIM
from tldqn_routing import DQNConfig, QNet, masked_argmax
from protocol import MAX_NEIGHBORS, MAX_NODES


class FrozenRoutingPolicy:
    """Use the baseline observation model without scenario labels or route CSVs."""

    def __init__(self, checkpoint=DEFAULT_MODEL):
        config = DQNConfig()
        self.network = QNet(STATE_DIM, K_ACTIONS, config.hidden, config.n_hidden)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.network.load_state_dict(state)
        self.network.eval()

    def initial_routes(self, nodes, seed):
        environment = build_env(nodes, seed, np.zeros(nodes, dtype=bool),
                                EnvConfig(), trust_source="none")
        return self._initial_routes(environment, nodes)

    def reroute(self, nodes, seed, isolated):
        isolated = list(isolated)
        if len(set(isolated)) != len(isolated) or any(not 0 <= node < nodes for node in isolated):
            raise ValueError("isolated sensor IDs must be unique and in range")
        environment = build_env(nodes, seed, np.zeros(nodes, dtype=bool),
                                EnvConfig(), trust_source="none")
        initial = self._initial_routes(environment, nodes)
        return reroute(initial, environment, self.network, isolated)

    def reroute_live(self, nodes, seed, isolated, live_state):
        isolated = list(isolated)
        if len(set(isolated)) != len(isolated) or any(not 0 <= node < nodes for node in isolated):
            raise ValueError("isolated sensor IDs must be unique and in range")
        if live_state is None:
            raise ValueError("live telemetry is required for adaptive reroute")
        environment = build_env(nodes, seed, np.zeros(nodes, dtype=bool),
                                EnvConfig(), trust_source="none")
        live_report = self.apply_live_state(environment, nodes, live_state)
        initial = self._initial_routes(environment, nodes)
        if isolated:
            routes, details = reroute(initial, environment, self.network, isolated)
            details["live_state"] = live_report
            return routes, details
        return initial, {"failed_entries": [], "live_state": live_report}

    @staticmethod
    def apply_live_state(environment, nodes, live_state):
        energy = np.asarray(live_state["energy"], dtype=np.float32)
        trust = np.asarray(live_state["trust"], dtype=np.float32)
        link_quality = np.asarray(live_state["link_quality"], dtype=np.float32)
        stability = np.asarray(live_state["stability"], dtype=np.float32)
        expected_links = MAX_NODES * MAX_NEIGHBORS
        if energy.shape != (MAX_NODES,) or trust.shape != (MAX_NODES,) or \
                link_quality.shape != (expected_links,) or stability.shape != (expected_links,):
            raise ValueError("live telemetry has an invalid shape")
        if (not np.all(np.isfinite(energy)) or not np.all(np.isfinite(trust)) or
                not np.all(np.isfinite(link_quality)) or not np.all(np.isfinite(stability))):
            raise ValueError("live telemetry contains non-finite values")
        if (np.any(energy[:nodes] < 0) or np.any(energy[:nodes] > 1) or
                np.any(trust[:nodes] < 0) or np.any(trust[:nodes] > 1)):
            raise ValueError("live node telemetry must be in [0, 1]")
        environment.energy[:nodes] = energy[:nodes]
        environment.trust[:nodes] = trust[:nodes]
        quality_matrix = link_quality.reshape(MAX_NODES, MAX_NEIGHBORS)
        stability_matrix = stability.reshape(MAX_NODES, MAX_NEIGHBORS)
        measured_links = 0
        for current in range(nodes):
            for neighbor in range(environment.topo.n_nodes):
                quality = float(quality_matrix[current, neighbor])
                stable = float(stability_matrix[current, neighbor])
                if quality >= 0:
                    environment.prr[current, neighbor] = quality
                    measured_links += 1
                if stable >= 0:
                    environment.stability[current, neighbor] = stable
        return {"measured_links": measured_links,
                "candidate_links": nodes * (nodes + 4),
                "proxy_fallback_links": nodes * (nodes + 4) - measured_links}

    def _initial_routes(self, environment, nodes):
        topology = environment.topo
        routes = []
        for current, destination in topology.reachable_pairs():
            environment.cur, environment.dest = current, destination
            features = environment.observe()
            mask = environment.action_mask()
            with torch.no_grad():
                values = self.network(torch.as_tensor(features[None], dtype=torch.float32))
            slot = int(masked_argmax(values, torch.as_tensor(mask[None]))[0])
            next_node = int(topology.slots(current, destination)[slot])
            is_sink = topology.is_sink(next_node)
            routes.append({"node_id": current, "dest_sink": destination - nodes,
                           "next_hop": next_node - nodes if is_sink else next_node,
                           "next_hop_is_sink": int(is_sink), "slot": slot,
                           "n_candidates": int(mask.sum()),
                           "own_dest": int(topology.dest[current] == destination),
                           "q_value": float(values[0, slot])})
        return routes
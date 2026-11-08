"""Frozen TLDQN agent for the isolated ns3-gym experiment."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np

from policy import DEFAULT_MODEL, FrozenRoutingPolicy
from protocol import LIVE_OBSERVATION_VERSION, VERSION, encode_routes, parse_observation

EXPERIMENT_DIR = Path(__file__).resolve().parent
BASELINE_MODEL_SHA256 = "523cbc1df67626d1b5fcf23ecd7011eac6fc639d9d66b2835a581b3be98f19a5"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_agent(environment, policy, nodes, seed, expect_reroute, live_adaptive=False):
    observation = environment.reset()
    requests = []
    while True:
        request = parse_observation(observation)
        if request["nodes"] != nodes or request["seed"] != seed:
            raise ValueError("simulator nodes or seed do not match agent arguments")
        if request["phase"] != len(requests) or request["phase"] > int(expect_reroute):
            raise ValueError("unexpected routing request sequence")
        start = time.perf_counter()
        if request["phase"] == 0:
            routes = policy.initial_routes(nodes, seed)
            failed = []
            live_report = None
            stage = "flow-start-bootstrap"
        elif live_adaptive:
            routes, details = policy.reroute_live(nodes, seed, request["isolated"],
                                                   request["live_state"])
            failed = details["failed_entries"]
            live_report = details["live_state"]
            stage = "live-telemetry-reroute"
        else:
            routes, details = policy.reroute(nodes, seed, request["isolated"])
            failed = details["failed_entries"]
            live_report = None
            stage = "post-isolation-reroute"
        action = encode_routes(routes, request)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        flow_routes = [row for row in routes if row["own_dest"] and
                   row["node_id"] not in request["isolated"]]
        print(f"[TLDQN ns3-gym] phase={request['phase']} stage={stage} seed={seed} "
              f"flow_decisions={len(flow_routes)} route_entries={len(routes)} "
              f"isolated={request['isolated']} inference_ms={elapsed_ms:.3f}", flush=True)
        if live_report:
            print(f"  live links measured={live_report['measured_links']}/"
                  f"{live_report['candidate_links']} proxy_fallback="
                  f"{live_report['proxy_fallback_links']}", flush=True)
        for row in flow_routes:
            hop_type = "sink" if row["next_hop_is_sink"] else "sensor"
            print(f"  flow node={row['node_id']} sink={row['dest_sink']} "
                  f"next_hop={hop_type}:{row['next_hop']} slot={row['slot']} "
                  f"q={row['q_value']:.5f}", flush=True)
        requests.append({"phase": request["phase"], "isolated": request["isolated"],
                         "routes": routes, "failed_entries": failed,
                         "flow_decisions": len(flow_routes), "stage": stage,
                         "live_state": live_report,
                         "inference_wall_ms": elapsed_ms})
        observation, _, done, _ = environment.step(action)
        if done:
            break
    if len(requests) != 1 + int(expect_reroute):
        raise ValueError("simulation ended without the expected routing requests")
    return requests


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--nodes", type=int, default=100)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--expect-reroute", action="store_true")
    parser.add_argument("--live-adaptive", action="store_true",
                        help="use phase-1 NS-3 telemetry for TLDQN reroute")
    parser.add_argument("--checkpoint", default=DEFAULT_MODEL)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be in 1..65535")
    if args.live_adaptive and not args.expect_reroute:
        parser.error("--live-adaptive requires --expect-reroute")
    audit = args.audit.resolve()
    if not audit.is_relative_to(EXPERIMENT_DIR) or audit.exists():
        parser.error("audit must be a new file within the isolated experiment folder")
    if sha256(args.checkpoint) != BASELINE_MODEL_SHA256:
        raise ValueError("checkpoint hash differs from the frozen baseline")
    if not all(hasattr(np, name) for name in ("int", "uint", "float")):
        raise RuntimeError("upstream ns3-gym requires a compatible NumPy; use the isolated environment")
    from ns3gym import ns3env

    policy = FrozenRoutingPolicy(args.checkpoint)
    environment = ns3env.Ns3Env(port=args.port, startSim=False)
    try:
        requests = run_agent(environment, policy, args.nodes, args.seed, args.expect_reroute,
                     args.live_adaptive)
    finally:
        environment.close()
    audit.parent.mkdir(parents=True, exist_ok=True)
    with audit.open("x", encoding="utf-8") as stream:
        json.dump({"protocol_version": 1, "nodes": args.nodes, "seed": args.seed,
                   "observation_protocol_version": (LIVE_OBSERVATION_VERSION if args.live_adaptive
                                                     else VERSION),
                   "checkpoint_sha256": BASELINE_MODEL_SHA256, "trust_source": "none",
                   "observation_model": ("phase-0 baseline proxy; phase-1 live NS-3 telemetry with proxy fallback"
                                         if args.live_adaptive else
                                         "baseline Python proxy; not live link telemetry"),
                   "requests": requests}, stream, indent=1, allow_nan=False)
    print(f"TLDQN gym audit -> {audit}; requests={len(requests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
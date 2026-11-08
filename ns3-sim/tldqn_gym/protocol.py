"""Versioned, label-free messages for the isolated TLDQN gym pilot."""

import numpy as np

VERSION = 1
LIVE_OBSERVATION_VERSION = 2
MAX_NODES = 100
SINKS = 4
MAX_NEIGHBORS = MAX_NODES + SINKS
MAX_ROUTES = MAX_NODES * SINKS
OBSERVATION_SIZE = 4 + MAX_NODES
LIVE_LINK_CELLS = MAX_NODES * MAX_NEIGHBORS
LIVE_OBSERVATION_SIZE = OBSERVATION_SIZE + 2 * MAX_NODES + 2 * LIVE_LINK_CELLS
ROUTE_FIELDS = ("node_id", "dest_sink", "next_hop", "next_hop_is_sink", "slot",
                "n_candidates", "own_dest", "q_value")
ACTION_SIZE = 3 + MAX_ROUTES * len(ROUTE_FIELDS)


def _integer(value, name):
    value = float(value)
    if not np.isfinite(value) or value != int(value):
        raise ValueError(f"{name} must be a finite integer")
    return int(value)


def parse_observation(observation):
    values = np.asarray(observation, dtype=np.float64).reshape(-1)
    if values.size not in (OBSERVATION_SIZE, LIVE_OBSERVATION_SIZE):
        raise ValueError("unexpected observation size")
    version, phase, nodes, seed = (_integer(values[index], name) for index, name in
                                   enumerate(("version", "phase", "nodes", "seed")))
    live = values.size == LIVE_OBSERVATION_SIZE
    expected_version = LIVE_OBSERVATION_VERSION if live else VERSION
    if version != expected_version or phase not in (0, 1):
        raise ValueError("unsupported protocol version or phase")
    if not 1 <= nodes <= MAX_NODES or not 1 <= seed <= 2 ** 24:
        raise ValueError("nodes or seed out of supported range")
    mask = [_integer(value, "isolation mask") for value in values[4:OBSERVATION_SIZE]]
    if any(value not in (0, 1) for value in mask) or any(mask[nodes:]):
        raise ValueError("invalid isolation mask")
    if phase == 0 and any(mask):
        raise ValueError("initial routing cannot include isolated sensors")
    request = {"phase": phase, "nodes": nodes, "seed": seed,
               "isolated": [node for node, value in enumerate(mask[:nodes]) if value]}
    if live:
        offset = OBSERVATION_SIZE
        energy = values[offset:offset + MAX_NODES]
        offset += MAX_NODES
        trust = values[offset:offset + MAX_NODES]
        offset += MAX_NODES
        link_quality = values[offset:offset + LIVE_LINK_CELLS]
        offset += LIVE_LINK_CELLS
        stability = values[offset:offset + LIVE_LINK_CELLS]
        for name, feature in (("energy", energy), ("trust", trust),
                              ("link quality", link_quality), ("stability", stability)):
            if not np.all(np.isfinite(feature)) or np.any(feature < -1.0) or np.any(feature > 1.0):
                raise ValueError(f"invalid live {name} telemetry")
        request["live_state"] = {"energy": energy.copy(), "trust": trust.copy(),
                                 "link_quality": link_quality.copy(),
                                 "stability": stability.copy()}
    else:
        request["live_state"] = None
    return request


def encode_routes(routes, request):
    nodes, phase = request["nodes"], request["phase"]
    if not routes or len(routes) > MAX_ROUTES:
        raise ValueError("invalid route count")
    action = np.full(ACTION_SIZE, -1, dtype=np.float32)
    action[:3] = [VERSION, phase, len(routes)]
    seen = set()
    for index, row in enumerate(routes):
        if set(row) != set(ROUTE_FIELDS):
            raise ValueError("route fields do not match label-free protocol")
        values = [_integer(row[name], name) for name in ROUTE_FIELDS[:-1]]
        node, destination, next_hop, is_sink, slot, candidates, own = values
        if (not 0 <= node < nodes or not 0 <= destination < SINKS or
                is_sink not in (0, 1) or own not in (0, 1) or
                not 0 <= slot < 6 or not 1 <= candidates <= 6 or slot >= candidates or
                not 0 <= next_hop < (SINKS if is_sink else nodes) or
                (is_sink and next_hop != destination) or (not is_sink and next_hop == node)):
            raise ValueError("invalid route entry")
        if (node, destination) in seen:
            raise ValueError("duplicate route entry")
        seen.add((node, destination))
        score = float(row["q_value"])
        if not np.isfinite(score):
            raise ValueError("non-finite Q-value")
        start = 3 + index * len(ROUTE_FIELDS)
        action[start:start + len(ROUTE_FIELDS)] = values + [score]
    return action
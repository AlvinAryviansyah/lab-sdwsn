"""Data sintetis berformat NSL-KDD untuk tes unit (cepat, tanpa berkas dataset)."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from nslkdd_data import COLUMNS, NUMERIC   # noqa: E402

PROTOCOLS = ["tcp", "udp", "icmp"]
SERVICES = ["http", "ftp", "smtp", "private"]
FLAGS = ["SF", "S0", "REJ"]


def make_df(n, seed, attack_fraction=0.0, shift=4.0):
    """DataFrame dengan kolom COLUMNS. Serangan: fitur numerik digeser `shift` sigma."""
    rng = np.random.default_rng(seed)
    data = {c: rng.normal(0.0, 1.0, n) for c in NUMERIC}
    is_att = rng.random(n) < attack_fraction
    for c in NUMERIC:
        data[c] = data[c] + shift * is_att
    data["protocol_type"] = rng.choice(PROTOCOLS, n)
    data["service"] = rng.choice(SERVICES, n)
    data["flag"] = rng.choice(FLAGS, n)
    data["label"] = np.where(is_att, "neptune", "normal")
    data["difficulty"] = 10
    df = pd.DataFrame(data)[COLUMNS]
    return df, is_att.astype(np.int64)

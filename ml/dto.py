#!/usr/bin/env python3
"""
dto.py  --  Driver Training-based Optimization (DTO), Kipongo et al. 2024, pers. (21)-(31)
-----------------------------------------------------------------------------------------
Optimizer GENERIK (meminimalkan objektif). Tidak tahu apa pun tentang jaringan atau label.
Penerapan ke deteksi node malicious ada di dto_detect.py.

Pemisahan sumber (lihat docs/STEP3_DTO_DESIGN.md):
  [PAPER]  dinyatakan paper           [ASUMSI] tafsiran reproduksi (paper tidak eksplisit)
  [IMPL]   keputusan implementasi

Notasi: m = ukuran populasi, d = jumlah variabel, q = iterasi, Y = iterasi maksimum.
(Paper memakai simbol n untuk keduanya; lihat bagian 2 dokumen desain.)
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DTOConfig:
    m: int = 30            # [ASUMSI K5] ukuran populasi (tidak ada di paper)
    Y: int = 100           # [ASUMSI K5] iterasi maksimum (tidak ada di paper)
    seed: int = 0


@dataclass
class DTOResult:
    x_best: np.ndarray
    f_best: float
    history: list = field(default_factory=list)   # per iterasi: (q, nID, P, f_best)
    n_eval: int = 0


class DTO:
    """Meminimalkan objective(x) dengan x in [lb, ub)^d."""

    def __init__(self, cfg: DTOConfig, lb, ub):
        self.cfg = cfg
        self.lb = np.asarray(lb, float).ravel()
        self.ub = np.asarray(ub, float).ravel()
        if self.lb.shape != self.ub.shape or np.any(self.ub <= self.lb):
            raise ValueError("batas tidak valid")
        self.d = len(self.lb)
        self.rng = np.random.default_rng(cfg.seed)
        # [IMPL] batas atas eksklusif agar dekode floor() tetap di dalam domain
        self._hi = np.nextafter(self.ub, self.lb)

    def _clip(self, x):
        # [IMPL] paper tidak menyebut penanganan batas; (22) dgn Rand in [0,2], (25), (30)
        # dapat keluar batas -> dipotong ke [lb, ub)
        return np.minimum(np.maximum(x, self.lb), self._hi)

    @staticmethod
    def n_instructors(m, q, Y):
        # [PAPER] (24): nID = [0.1 * m * (1 - q/Y)]  -- "[ ]" ditafsirkan floor
        # [ASUMSI] minimal 1 (di q = Y rumus paper memberi 0 instruktur)
        return max(1, int(np.floor(0.1 * m * (1.0 - q / Y))))

    @staticmethod
    def patterning_index(q, Y):
        return 0.01 + 0.9 * (1.0 - q / Y)            # [PAPER] (29)

    def phase1(self, z, idk, of_idk, of_i):
        """[PAPER] (25). Rand ~ U[0,1], Rand1 ~ U[1,2] (per variabel)."""
        r = self.rng.random(self.d)
        if of_idk < of_i:
            r1 = 1.0 + self.rng.random(self.d)
            return z + r * (idk - r1 * z)
        return z + r * (z - idk)

    @staticmethod
    def phase2(z, idk, P):
        return P * z + (1.0 - P) * idk               # [PAPER] (27)

    def phase3(self, z, q, Y):
        r = self.rng.random(self.d)                   # [PAPER] (30), Rand ~ U[0,1]
        return z + (1.0 - 2.0 * r) * (1.0 - q / Y) * z

    def minimize(self, objective):
        m, Y = self.cfg.m, self.cfg.Y
        # [PAPER] (21)-(22); [ASUMSI K6] Rand ~ U(0,1) (paper menulis [0,2])
        Z = self.lb + self.rng.random((m, self.d)) * (self.ub - self.lb)
        Z = self._clip(Z)
        OF = np.array([objective(z) for z in Z])     # [PAPER] (23)
        n_eval = m
        b = int(np.argmin(OF))
        x_best, f_best = Z[b].copy(), float(OF[b])
        hist = []
        for q in range(1, Y + 1):                    # Algoritma 2 baris 6
            nID = self.n_instructors(m, q, Y)
            P = self.patterning_index(q, Y)
            for i in range(m):                       # baris 7
                # baris 8: "Determine ID based on the value of objective function"
                # [IMPL] ID = nID anggota terbaik saat ini (dihitung ulang per anggota)
                order = np.argsort(OF, kind="stable")
                k = order[int(self.rng.integers(nID))]
                idk, of_idk = Z[k].copy(), OF[k]
                # Fase 1: (25) + (26)
                c = self._clip(self.phase1(Z[i], idk, of_idk, OF[i]))
                fc = objective(c); n_eval += 1
                if fc < OF[i]:
                    Z[i], OF[i] = c, fc
                # Fase 2: (29) + (27) + (28); [ASUMSI] instruktur k_i yang sama
                c = self._clip(self.phase2(Z[i], idk, P))
                fc = objective(c); n_eval += 1
                if fc < OF[i]:
                    Z[i], OF[i] = c, fc
                # Fase 3: (30) + (31)
                c = self._clip(self.phase3(Z[i], q, Y))
                fc = objective(c); n_eval += 1
                if fc < OF[i]:
                    Z[i], OF[i] = c, fc
            b = int(np.argmin(OF))                   # baris 12: perbarui solusi terbaik
            if OF[b] < f_best:
                x_best, f_best = Z[b].copy(), float(OF[b])
            hist.append((q, nID, P, f_best))
        return DTOResult(x_best=x_best, f_best=f_best, history=hist, n_eval=n_eval)

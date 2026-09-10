"""Statistical primitives: two-proportion z-test, Benjamini–Hochberg FDR.

Implemented directly (they're a few lines each) so the analysis has no
heavyweight scientific dependency; unit tests pin them to known values.
"""

from __future__ import annotations

import math


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Two-sided two-proportion z-test.

    Returns (z, p_value) for H0: p1 == p2. Uses the pooled estimate.
    """
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    p1, p2 = k1 / n1, k2 / n2
    pooled = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    p_value = math.erfc(abs(z) / math.sqrt(2))  # two-sided
    return z, p_value


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """BH-adjusted q-values, preserving input order.

    q_i = min over j>=rank(i) of (p_(j) * m / j), clipped to 1.
    """
    m = len(p_values)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: p_values[i])
    q = [0.0] * m
    running_min = 1.0
    for rank_from_top in range(m, 0, -1):
        idx = order[rank_from_top - 1]
        candidate = min(1.0, p_values[idx] * m / rank_from_top)
        running_min = min(running_min, candidate)
        q[idx] = running_min
    return q


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))

"""Exact statistics used by the verifier (G1, G5) and the pre-registered hypotheses (H3, H4)."""
from __future__ import annotations

import numpy as np
from scipy import stats


def _check_counts(k: int, n: int) -> None:
    if n <= 0 or not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n and n > 0, got k={k}, n={n}")


def clopper_pearson_upper(k: int, n: int, conf: float = 0.95) -> float:
    """One-sided upper bound on a failure rate after k failures in n trials (G5)."""
    _check_counts(k, n)
    if k == n:
        return 1.0
    return float(stats.beta.ppf(conf, k + 1, n - k))


def clopper_pearson_lower(k: int, n: int, conf: float = 0.95) -> float:
    """One-sided lower bound on a success rate after k successes in n trials (H3 catch rate)."""
    _check_counts(k, n)
    if k == 0:
        return 0.0
    return float(stats.beta.ppf(1 - conf, k, n - k + 1))


def ljung_box_p(residual: np.ndarray, lags: int = 10) -> float:
    """p-value of Ljung-Box Q(lags) under chi^2_lags; G1 requires p > 0.05 (white innovations)."""
    r = np.asarray(residual, dtype=float)
    n = r.size
    if n <= lags + 1:
        raise ValueError(f"need more than {lags + 1} samples, got {n}")
    r = r - r.mean()
    denom = r @ r
    if denom == 0:
        raise ValueError("residual is constant")
    h = np.arange(1, lags + 1)
    rho = np.array([r[:-lag] @ r[lag:] for lag in h]) / denom
    q = n * (n + 2) * np.sum(rho**2 / (n - h))
    return float(stats.chi2.sf(q, lags))


def mcnemar_exact_p(b: int, c: int) -> float:
    """One-sided exact McNemar (H4): b = rows only the tuned model got right, c = only base got right."""
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be non-negative")
    if b + c == 0:
        return 1.0
    return float(stats.binom.sf(b - 1, b + c, 0.5))

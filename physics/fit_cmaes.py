"""CMA-ES fit, Laplace posterior, identifiability gate, BIC (master doc §2.8)."""
from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import cma
import numpy as np

COND_MAX = 1e6
EIG_FLOOR_REL = 1e-9
REL_STD_MAX = 0.5

Objective = Callable[[np.ndarray], float]


class Posterior(NamedTuple):
    mean: np.ndarray
    cov: np.ndarray
    hessian: np.ndarray
    j_min: float


def fit(objective: Objective, x0: np.ndarray, sigma0: float, lower: np.ndarray, upper: np.ndarray,
        seed: int, max_evals: int = 3000) -> tuple[np.ndarray, float]:
    """Box-bounded CMA-ES. shortcut: runs locally in one process; fan out to Modal (fit_worker) if a fit gets slow."""
    # cma treats seed 0 (and None) as "seed from the clock"; shift so every seed, including 0, is reproducible
    opts = {"bounds": [list(lower), list(upper)], "seed": seed + 1, "maxfevals": max_evals, "verbose": -9}
    es = cma.CMAEvolutionStrategy(list(x0), sigma0, opts)
    es.optimize(objective)
    return np.asarray(es.result.xbest), float(es.result.fbest)


def laplace(objective: Objective, x_hat: np.ndarray, step: float = 1e-3) -> Posterior:
    """Covariance = inverse of the central finite-difference Hessian at the optimum."""
    d = x_hat.size
    f0 = objective(x_hat)
    H = np.empty((d, d))
    e = np.eye(d) * step
    for i in range(d):
        for j in range(i, d):
            H[i, j] = H[j, i] = (
                objective(x_hat + e[i] + e[j]) - objective(x_hat + e[i] - e[j])
                - objective(x_hat - e[i] + e[j]) + objective(x_hat - e[i] - e[j])
            ) / (4 * step**2)
    # nuisance directions can be flat or slightly negative at a numerical optimum; floor the spectrum so the
    # covariance is always a valid PSD matrix for sampling (identifiability() judges the physical block)
    w, v = np.linalg.eigh((H + H.T) / 2)
    floor = max(w.max(), 1e-12) * EIG_FLOOR_REL
    cov = v @ np.diag(1 / np.maximum(w, floor)) @ v.T
    return Posterior(x_hat, cov, H, f0)


def identifiability(post: Posterior, names: list[str], log_idx: list[int]) -> tuple[bool, str]:
    """Refuse if cond(H) > 1e6 or any relative posterior std > 50%.

    log_idx: the parameters judged (the MJCF parameters phi, fitted in log space, so sqrt(Sigma_ii) is
    already the relative std); conditioning is assessed on their Hessian block.
    """
    eig = np.linalg.eigvalsh(post.hessian[np.ix_(log_idx, log_idx)])
    if eig.min() <= 0:
        return False, "Hessian not positive definite: optimum is flat or a saddle"
    cond = eig.max() / eig.min()
    if cond > COND_MAX:
        return False, f"cond(Hessian) = {cond:.3g} > {COND_MAX:.0e}"
    rel = np.sqrt(np.diag(post.cov))[log_idx]
    worst = log_idx[int(np.argmax(rel))]
    if rel.max() > REL_STD_MAX:
        return False, f"{names[worst]} relative std {rel.max():.2f} > {REL_STD_MAX}; record a larger-amplitude take"
    return True, "identifiable"


def posterior_samples(post: Posterior, n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).multivariate_normal(post.mean, post.cov, size=n)


def bic(j_min: float, n_params: int, n_obs: int) -> float:
    """Approximate (J contains regularizers): ranking structures only."""
    return 2 * j_min + n_params * np.log(n_obs)

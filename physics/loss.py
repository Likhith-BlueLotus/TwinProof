"""Composite identification loss J(phi) (master doc §2.8)."""
from __future__ import annotations

import numpy as np

from physics.ou import ou_nll


def composite_loss(
    residual: np.ndarray,
    t: np.ndarray,
    kappa: np.ndarray,
    sigma: np.ndarray,
    r_meas: np.ndarray,
    phi: np.ndarray,
    phi0: np.ndarray,
    prior_prec: np.ndarray,
    margins: np.ndarray,
    lam_prior: float,
    lam_margin: float,
    tau: float = 1e-3,
) -> float:
    """L_OU + lam_prior ||phi - phi0||^2_Lambda0 + lam_margin sum softplus(-g/tau).

    shortcut: the mask-IoU (lambda_1) and slip (lambda_2) terms are omitted; add them once rendered
    twin masks and S2 slip data exist (Phase D/S2).
    """
    d = np.asarray(phi) - np.asarray(phi0)
    prior = float(d @ (np.asarray(prior_prec) * d))
    barrier = float(np.sum(np.logaddexp(0.0, -np.asarray(margins, float) / tau)))
    return ou_nll(residual, t, kappa, sigma, r_meas) + lam_prior * prior + lam_margin * barrier

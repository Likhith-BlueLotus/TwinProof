"""Exact support-function bounds over the ellipsoid {x : x^T P x <= c} (G3)."""
from __future__ import annotations

import numpy as np


def input_bound(K: np.ndarray, P: np.ndarray, c: float) -> np.ndarray:
    """max |k_i^T x| over the sublevel set = sqrt(c k_i^T P^-1 k_i), per input."""
    return np.sqrt(c * np.einsum("ij,jk,ik->i", K, np.linalg.inv(P), K))


def state_bound(P: np.ndarray, c: float) -> np.ndarray:
    """max |x_j| over the sublevel set = sqrt(c (P^-1)_jj), per state."""
    return np.sqrt(c * np.diag(np.linalg.inv(P)))

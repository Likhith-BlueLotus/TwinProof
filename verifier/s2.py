"""S2 structure claims: G7 -> G0 -> G1, stopping at the first FAIL (S2 has no controller task yet)."""
from __future__ import annotations

import numpy as np

from physics.identify_s2 import patch_for, residuals
from physics.ou import ou_innovations
from verifier.gates import GateResult, g0_admissible, g1_heldout, g7_patch_guard


def verify_s2_structure(base_xml: str, x: np.ndarray, t: np.ndarray, s_obs: np.ndarray, psi_obs: np.ndarray,
                        n_fit: int, release0: float, r_s: float, extra_patch: dict[str, str] | None = None) -> dict[str, GateResult]:
    patch = {**patch_for(base_xml, x), **(extra_patch or {})}  # exactly what the twin simulates
    gates = {"G7": g7_patch_guard(base_xml, patch, None, 0.0)}
    if not gates["G7"].passed:
        return gates
    gates["G0"] = g0_admissible(base_xml, patch)
    if not gates["G0"].passed:
        return gates
    rs, _ = residuals(base_xml, x, t, s_obs, psi_obs, release0)
    nu, var = ou_innovations(rs[n_fit:], t[n_fit:], np.exp(x[2:3]), np.exp(x[3:4]), r_s)
    gates["G1"] = g1_heldout(s_obs[n_fit:], s_obs[n_fit:] - rs[n_fit:], (nu / np.sqrt(var))[:, 0])
    return gates

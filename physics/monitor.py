"""Contact, clearance and joint-limit metrics over recorded states (feeds G4)."""
from __future__ import annotations

import mujoco
import numpy as np

from physics.reset import set_state


def contact_metrics(model: mujoco.MjModel, qpos: np.ndarray, qvel: np.ndarray, allowed_pairs: frozenset[tuple[str, str]] = frozenset()) -> dict[str, float]:
    """Worst case over the given states. Clearance only counts geom pairs not in allowed_pairs."""
    data = mujoco.MjData(model)
    force = np.zeros(6)
    out = {"min_clearance": np.inf, "min_limit_margin": np.inf, "max_normal_force": 0.0, "max_cone_ratio": 0.0}
    limited = np.flatnonzero(model.jnt_limited)
    for q, v in zip(qpos, qvel):
        set_state(model, data, q, v)
        for i in range(data.ncon):
            c = data.contact[i]
            pair = tuple(sorted((mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in c.geom)))
            mujoco.mj_contactForce(model, data, i, force)
            fn, ft = abs(force[0]), np.linalg.norm(force[1:3])
            out["max_normal_force"] = max(out["max_normal_force"], fn)
            if fn > 0:
                out["max_cone_ratio"] = max(out["max_cone_ratio"], ft / (c.friction[0] * fn))
            if pair not in allowed_pairs:
                out["min_clearance"] = min(out["min_clearance"], c.dist)
        for j in limited:
            qj = float(data.qpos[model.jnt_qposadr[j]])
            lo, hi = model.jnt_range[j]
            out["min_limit_margin"] = min(out["min_limit_margin"], qj - lo, hi - qj)
    return {k: float(v) for k, v in out.items()}

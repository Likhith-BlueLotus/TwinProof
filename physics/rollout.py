"""Simulate a scenario model and sample joint positions at observation timestamps."""
from __future__ import annotations

from collections.abc import Callable

import mujoco
import numpy as np

from physics.reset import set_state

Policy = Callable[[np.ndarray, np.ndarray], np.ndarray]  # (qpos, qvel) -> ctrl


def rollout(
    model: mujoco.MjModel, qpos0: np.ndarray, qvel0: np.ndarray, t_eval: np.ndarray, policy: Policy | None = None
) -> np.ndarray:
    """qpos at each t_eval (relative to t_eval[0]); shape (len(t_eval), nq)."""
    data = mujoco.MjData(model)
    set_state(model, data, qpos0, qvel0)
    t_rel = np.asarray(t_eval, float) - t_eval[0]
    out = np.empty((t_rel.size, model.nq))
    for k, tk in enumerate(t_rel):
        while data.time < tk - 0.5 * model.opt.timestep:
            if policy is not None:
                data.ctrl[:] = policy(data.qpos, data.qvel)
            mujoco.mj_step(model, data)
        out[k] = data.qpos
    return out

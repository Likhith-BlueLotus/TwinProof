"""The only module allowed to write MjData.qpos/qvel (G7 AST guard enforces this)."""
from __future__ import annotations

import mujoco
import numpy as np


def set_state(model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray, qvel: np.ndarray) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)

"""S2 incline identification: a can released from rest rolls or slides down a ramp.

Observed: distance along the ramp s(t) and the can's rotation psi(t) (tape stripe on its end face), both
relative to the first frame. x = [log mu, log rolling_friction, log kappa_s, log sigma_s, d_release, d_s0, d_psi0].
Structures (master doc §2.5): "incline_rolling" pins mu high (no slip, so mu is not identifiable and is not
fitted); "incline_sliding" pins rolling friction low and fits mu. BIC picks between them.
"""
from __future__ import annotations

import mujoco
import numpy as np

from physics.loss import composite_loss
from physics.mjcf_patch import apply_patch
from physics.reset import set_state

NAMES = ["mu", "rolling_friction", "ou_kappa", "ou_sigma", "d_release", "d_s0", "d_psi0"]
PHYSICAL = [0, 1]  # MJCF parameters phi; which ones are free depends on the structure
STRUCTURES = ("incline_rolling", "incline_sliding")
RELEASE_SCALE, S0_SCALE, PSI0_SCALE = 0.02, 0.002, 0.02  # s, m, rad per unit offset
LOWER = np.r_[np.log([1e-3, 1e-6, 1e-2, 1e-5]), -5, -5, -5]
UPPER = np.r_[np.log([1.5, 1e-2, 1e2, 1.0]), 5, 5, 5]
PINNED = {"incline_rolling": (0, np.log(1.0)), "incline_sliding": (1, np.log(1e-6))}
PIN_WIDTH = 1e-6  # CMA-ES needs lower < upper
LAM_PRIOR = 1e-3
GEOMS = (".//geom[@name='can']", ".//geom[@name='ramp']")


def bounds(structure: str) -> tuple[np.ndarray, np.ndarray]:
    if structure not in PINNED:
        raise ValueError(f"unknown S2 structure {structure!r}")
    lower, upper = LOWER.copy(), UPPER.copy()
    i, value = PINNED[structure]
    lower[i], upper[i] = value, value + PIN_WIDTH
    return lower, upper


def xml_for(base_xml: str, x: np.ndarray) -> str:
    mu, roll = np.exp(x[:2])
    friction = f"{mu:.9g} 0.005 {roll:.9g}"
    return apply_patch(base_xml, {f"{g}@friction": friction for g in GEOMS})


def ramp_frame(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors down the ramp and normal to it, from the ramp body's orientation."""
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, model.body("ramp").quat)
    rot = rot.reshape(3, 3)
    return rot[:, 0], rot[:, 2]


def simulate(model: mujoco.MjModel, t: np.ndarray, release: float) -> tuple[np.ndarray, np.ndarray]:
    """s and psi at times t for a can at rest that is released at `release` (both 0 before it)."""
    data = mujoco.MjData(model)
    down, normal = ramp_frame(model)
    radius = model.geom("can").size[0]
    qpos = np.r_[radius * normal * 1.0001, 1.0, 0.0, 0.0, 0.0]  # on the surface, axis across the ramp
    set_state(model, data, qpos, np.zeros(model.nv))
    p0 = data.qpos[:3].copy()
    s, psi = np.zeros(t.size), np.zeros(t.size)
    for k, tk in enumerate(np.asarray(t) - release):
        if tk <= 0:
            continue
        while data.time < tk - 0.5 * model.opt.timestep:
            mujoco.mj_step(model, data)
        s[k] = (data.qpos[:3] - p0) @ down
        w, y = data.qpos[3], data.qpos[5]  # rotation about the y axis (across the ramp)
        psi[k] = 2 * np.arctan2(y, w)
    return s, np.unwrap(psi)


def residuals(base_xml: str, x: np.ndarray, t: np.ndarray, s_obs: np.ndarray, psi_obs: np.ndarray, release0: float):
    model = mujoco.MjModel.from_xml_string(xml_for(base_xml, x))
    s_sim, psi_sim = simulate(model, t, release0 + RELEASE_SCALE * x[4])
    return s_obs - (s_sim + S0_SCALE * x[5]), psi_obs - (psi_sim + PSI0_SCALE * x[6])


def make_objective(base_xml: str, t: np.ndarray, s_obs: np.ndarray, psi_obs: np.ndarray, release0: float,
                   r_s: float, r_psi: float, x0: np.ndarray):
    """OU NLL on the distance channel, white Gaussian NLL on rotation, weak prior."""
    def objective(x: np.ndarray) -> float:
        x = np.asarray(x)
        rs, rpsi = residuals(base_xml, x, t, s_obs, psi_obs, release0)
        psi_nll = 0.5 * float(np.sum(rpsi**2 / r_psi + np.log(2 * np.pi * r_psi)))
        return composite_loss(rs, t, np.exp(x[2:3]), np.exp(x[3:4]), r_s, x, x0, np.ones(x.size),
                              margins=np.zeros(0), lam_prior=LAM_PRIOR, lam_margin=0.0) + psi_nll
    return objective


def release_time(t: np.ndarray, s_obs: np.ndarray, threshold: float = 0.002) -> float:
    """First frame where the can has moved more than `threshold` m, minus one frame: the release guess."""
    moved = np.flatnonzero(np.abs(s_obs) > threshold)
    if moved.size == 0:
        raise ValueError("the can never moves in this take")
    return float(t[max(moved[0] - 1, 0)])


MOVING_MIN = 0.05  # m/s: frames slower than this say nothing about slip


def classify_contact(t: np.ndarray, s_obs: np.ndarray, psi_obs: np.ndarray, radius: float) -> tuple[str, float]:
    """Rolling vs sliding from the data's slip residual (master doc §2.5): median relative slip > 0.1 is sliding.

    BIC cannot decide this: a free-mu sliding model with mu above the no-slip threshold rolls too.
    """
    from perception.kinematics import SLIDING_EPS, slip
    from perception.smoother import rts_smooth

    v = rts_smooth(t, s_obs, sigma=1e-3, q_tilde=50.0).states[:, 1]
    w = rts_smooth(t, psi_obs, sigma=2e-2, q_tilde=500.0).states[:, 1]
    moving = np.abs(v) > MOVING_MIN
    if moving.sum() < 5:
        raise ValueError("too few moving frames to judge slip")
    _, eps, _ = slip(v[moving], w[moving], radius)
    med = float(np.median(eps))
    return ("incline_sliding" if med > SLIDING_EPS else "incline_rolling"), med

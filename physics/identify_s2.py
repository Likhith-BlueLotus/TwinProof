"""S2 incline identification: a can released from rest rolls or slides down a ramp.

Observed: distance along the ramp s(t) and the can's rotation psi(t) (tape stripe on its end face), both
relative to the first frame. x = [log mu, log rolling_friction, log kappa_s, log sigma_s, d_release, d_s0, d_psi0].
Structures (master doc §2.5): "incline_rolling" pins mu high (no slip, so mu is not identifiable and is not
fitted); "incline_sliding" pins rolling friction low and fits mu. BIC picks between them.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from physics.loss import composite_loss
from physics.mjcf_patch import apply_patch
from physics.reset import set_state

NAMES = ["mu", "rolling_friction", "ou_kappa", "ou_sigma", "d_release", "d_s0", "d_psi0", "inertia_ratio"]
# MJCF parameters phi judged by the identifiability gate, per structure (x[7] = log beta, beta = I_axis / (m r^2))
FREE_PHYSICAL = {"incline_rolling": [7], "incline_sliding": [0, 7]}
STRUCTURES = ("incline_rolling", "incline_sliding")
RELEASE_SCALE, S0_SCALE, PSI0_SCALE = 0.02, 0.002, 0.02  # s, m, rad per unit offset
LOWER = np.r_[np.log([1e-3, 1e-6, 1e-2, 1e-5]), -5, -5, -5, np.log(0.3)]
UPPER = np.r_[np.log([1.5, 1e-2, 1e2, 1.0]), 5, 5, 5, np.log(1.5)]
# rolling: no slip, so mu is irrelevant; rolling resistance is inseparable from beta over a sub-second roll
PINNED = {"incline_rolling": [(0, np.log(1.0)), (1, np.log(1e-6))], "incline_sliding": [(1, np.log(1e-6))]}
PIN_WIDTH = 1e-6  # CMA-ES needs lower < upper
LAM_PRIOR = 1e-3
GEOMS = (".//geom[@name='can']", ".//geom[@name='ramp']")


def bounds(structure: str) -> tuple[np.ndarray, np.ndarray]:
    if structure not in PINNED:
        raise ValueError(f"unknown S2 structure {structure!r}")
    lower, upper = LOWER.copy(), UPPER.copy()
    for i, value in PINNED[structure]:
        lower[i], upper[i] = value, value + PIN_WIDTH
    return lower, upper


INERTIAL = ".//body[@name='can']/inertial"


def patch_for(base_xml: str, x: np.ndarray) -> dict[str, str]:
    """Whitelisted MJCF patch for x: geom friction, and the can's inertia when x carries beta (x[7])."""
    mu, roll = np.exp(x[:2])
    patch = {f"{g}@friction": f"{mu:.9g} 0.005 {roll:.9g}" for g in GEOMS}
    if len(x) > 7:
        root = ET.fromstring(base_xml)
        geom, inertial = root.find(".//geom[@name='can']"), root.find(INERTIAL)
        r, half = (float(v) for v in geom.get("size").split())
        m, beta = float(inertial.get("mass")), float(np.exp(x[7]))
        axis, transverse = beta * m * r**2, m * (beta * r**2 / 2 + (2 * half) ** 2 / 12)
        patch[f"{INERTIAL}@diaginertia"] = f"{transverse:.9g} {axis:.9g} {transverse:.9g}"  # body y is the can's axis
    return patch


def xml_for(base_xml: str, x: np.ndarray) -> str:
    return apply_patch(base_xml, patch_for(base_xml, x))


def s2_xml(base_xml: str, ramp_deg: float, radius: float, mass: float, half_length: float = 0.05) -> str:
    """S2 twin from measurements: ramp angle, can radius and mass. shortcut: can length assumed 10 cm; it only
    enters the transverse inertia, which a roll about the can's axis does not excite."""
    root = ET.fromstring(base_xml)
    root.find(".//body[@name='ramp']").set("euler", f"0 {ramp_deg:.6g} 0")
    root.find(".//geom[@name='can']").set("size", f"{radius:.6g} {half_length:.6g}")
    root.find(".//geom[@name='can']").set("mass", f"{mass:.6g}")
    root.find(INERTIAL).set("mass", f"{mass:.6g}")
    return xml_for(ET.tostring(root, encoding="unicode"), np.r_[np.log([1.0, 1e-6]), 0, 0, 0, 0, 0, np.log(1.0)])


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
    """Rolling vs sliding from the data's relative slip (master doc §2.5: > 0.1 is sliding).

    Slip is measured as |slope - 1| of distance s against rolled distance psi * radius over the moving frames:
    one least-squares line averages out stripe-angle noise that makes per-frame rate ratios unreliable.
    BIC cannot decide this: a free-mu sliding model with mu above the no-slip threshold rolls too.
    """
    from perception.kinematics import SLIDING_EPS
    from perception.smoother import rts_smooth

    v = rts_smooth(t, s_obs, sigma=1e-3, q_tilde=50.0).states[:, 1]
    moving = np.abs(v) > MOVING_MIN
    if moving.sum() < 5:
        raise ValueError("too few moving frames to judge slip")
    rolled = (psi_obs - psi_obs[moving][0]) * radius
    slope = np.polyfit(rolled[moving], s_obs[moving], 1)[0]
    eps = float(abs(slope - 1.0)) if np.isfinite(slope) and abs(rolled[moving]).max() > 1e-3 else float("inf")
    return ("incline_sliding" if eps > SLIDING_EPS else "incline_rolling"), eps

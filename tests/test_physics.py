from pathlib import Path

import mujoco
import numpy as np
import pytest

from physics.loss import composite_loss
from physics.mjcf_patch import PatchRejected, apply_patch
from physics.ou import init_ou, ou_nll
from physics.rollout import rollout

S1 = (Path(__file__).parents[1] / "physics/models/s1_pendulum.xml").read_text()
DAMP = ".//joint[@name='hinge']@damping"


def _ou_path(kappa, sigma, t, seed=0):
    rng = np.random.default_rng(seed)
    r = np.zeros(t.size)
    for k in range(1, t.size):
        a = np.exp(-kappa * (t[k] - t[k - 1]))
        r[k] = a * r[k - 1] + np.sqrt(sigma**2 / (2 * kappa) * (1 - a**2)) * rng.standard_normal()
    return r


def test_patch_changes_whitelisted_field():
    m = mujoco.MjModel.from_xml_string(apply_patch(S1, {DAMP: "0.01"}))
    assert m.dof_damping[0] == pytest.approx(0.01)


@pytest.mark.parametrize(
    "key",
    [".//option@timestep", ".//option@gravity", ".//motor[@name='tau']@ctrlrange", ".//joint@nonexistent", "damping",
     ".//motor[@name='tau']@gear"],  # gear scales delivered torque past the ctrlrange that G3 checks
)
def test_patch_refuses_off_whitelist(key):
    with pytest.raises(PatchRejected):
        apply_patch(S1, {key: "0"})


def test_rollout_period_matches_physical_pendulum():
    m = mujoco.MjModel.from_xml_string(apply_patch(S1, {DAMP: "0", ".//joint[@name='hinge']@frictionloss": "0"}))
    t = np.arange(0, 3, 1 / 60)
    q = rollout(m, np.array([0.05]), np.array([0.0]), t)[:, 0]
    crossings = t[1:][np.diff(np.sign(q)) != 0]
    period = 2 * np.mean(np.diff(crossings))
    # rod pivoted at one end: T = 2 pi sqrt(I / (m g L/2)); capsule caps make it slightly longer
    L, g = 0.35, 9.81
    assert period == pytest.approx(2 * np.pi * np.sqrt((L**2 / 3) / (g * L / 2)), rel=0.03)


def test_ou_init_and_nll_recover_parameters():
    t = np.arange(0, 60, 1 / 60)
    r = _ou_path(kappa=3.0, sigma=0.05, t=t)
    k0, s0 = init_ou(r, t)
    assert k0[0] == pytest.approx(3.0, rel=0.3) and s0[0] == pytest.approx(0.05, rel=0.3)
    nll = {k: ou_nll(r, t, np.array([k]), np.array([0.05]), 1e-8) for k in (0.3, 3.0, 30.0)}
    assert nll[3.0] < nll[0.3] and nll[3.0] < nll[30.0]


def test_composite_loss_penalises_prior_and_violated_margins():
    t = np.arange(0, 2, 1 / 60)
    r = _ou_path(3.0, 0.05, t)
    base = dict(residual=r, t=t, kappa=np.array([3.0]), sigma=np.array([0.05]), r_meas=1e-6,
                phi0=np.zeros(2), prior_prec=np.ones(2), lam_prior=1.0, lam_margin=1.0)
    ok = composite_loss(phi=np.zeros(2), margins=np.array([1.0]), **base)
    assert composite_loss(phi=np.ones(2), margins=np.array([1.0]), **base) == pytest.approx(ok + 2)
    assert composite_loss(phi=np.zeros(2), margins=np.array([-0.01]), **base) > ok + 5


def test_pendulum_xml_matches_measured_geometry():
    from physics.identify_s1 import pendulum_xml
    from physics.reset import set_state
    m = mujoco.MjModel.from_xml_string(pendulum_xml(S1, length=0.30, centroid_radius=0.13, mass=0.05))
    d = mujoco.MjData(m)
    set_state(m, d, np.zeros(1), np.zeros(1))
    assert m.body_mass[1] == pytest.approx(0.05)
    assert np.linalg.norm(d.xipos[1] - d.xanchor[0]) == pytest.approx(0.13, abs=1e-3)  # centroid below the pivot
    assert m.actuator_ctrlrange[0, 1] == pytest.approx(2 * 0.05 * 9.81 * 0.13, rel=1e-3)
    with pytest.raises(ValueError):
        pendulum_xml(S1, length=0.2, centroid_radius=0.15, mass=0.05)

from pathlib import Path

import mujoco
import numpy as np
import pytest

from physics.controllers import LQRSpec
from physics.identify_s1 import model_for
from verifier.pipeline import verify_s1_lqr

S1 = (Path(__file__).parents[1] / "physics/models/s1_pendulum.xml").read_text()
X_HAT = np.r_[np.log([0.004, 0.0008, 2.0, 0.01]), 0, 0]
GOOD = LQRSpec(q_eq=np.pi, q_diag=(10, 0.1), r_diag=(10,), u_max=2.5)


@pytest.fixture(scope="module")
def models():
    rng = np.random.default_rng(0)
    samples = X_HAT + rng.normal(0, 1, (16, 6)) * np.r_[0.05, 0.15, 0, 0, 0, 0]
    return model_for(S1, X_HAT), [model_for(S1, x) for x in samples]


def test_upright_lqr_is_certified(models):
    v = verify_s1_lqr(*models, GOOD, ou_kappa=2.0, ou_sigma=0.01)
    assert v.admitted, {k: g.feedback_to_agent for k, g in v.gates.items()}
    assert v.gates["G2"].witness["max_spectral_radius"] <= 0.98
    assert v.gates["G3"].witness["u_max_bound"] <= 2.5
    assert v.roa.c_star > v.roa.c0


def test_destabilizing_gains_fail_g2(models):
    bad = LQRSpec(np.pi, GOOD.q_diag, GOOD.r_diag, 2.5, gains=(0.2, 0.05))  # below m g l: cannot hold upright
    v = verify_s1_lqr(*models, bad, ou_kappa=2.0, ou_sigma=0.01)
    g2 = v.gates["G2"]
    assert not v.admitted and not g2.passed and g2.witness["max_spectral_radius"] > 0.98


def test_reduced_torque_limit_fails_g3(models):
    nominal, samples = models
    weak = mujoco.MjModel.from_xml_string(S1.replace('ctrlrange="-2.5 2.5"', 'ctrlrange="-0.75 0.75"'))
    weak.dof_damping[:] = nominal.dof_damping
    weak.dof_frictionloss[:] = nominal.dof_frictionloss
    v = verify_s1_lqr(weak, samples, GOOD, ou_kappa=2.0, ou_sigma=0.01)
    g3 = v.gates["G3"]
    assert not g3.passed and g3.witness["u_max_bound"] > 0.75 and "reduce" in g3.feedback_to_agent


def test_g2_g3_refuse_nan_witnesses():
    from verifier.gates import g2_stability, g3_bounds
    from verifier.lyapunov import RoaResult
    A = np.array([[[0.5, 0.0], [0.0, 0.5]]])
    B = np.zeros((1, 2, 1))
    K = np.zeros((1, 2))
    nan_roa = RoaResult(np.nan, 0.01, np.nan, np.nan, np.nan, 0, "")
    assert not g2_stability(A, B, K, np.eye(2), nan_roa).passed
    assert not g3_bounds(np.array([[np.nan, 0.0]]), np.eye(2), 1.0, np.nan, 2.5).passed


def test_certify_roa_refuses_when_inner_ball_can_escape():
    from verifier.lyapunov import certify_roa
    P = np.eye(2)
    contract = lambda X: 0.5 * X  # noqa: E731  dV < 0 everywhere
    # L encodes ell^2 = 9: shells certify, but one step from {V <= c0} could reach 9 c0, beyond c_hi = 4 c0
    lip = lambda c: (2 * np.sqrt(c) * 10.0, 0.0)  # noqa: E731
    roa = certify_roa(contract, P, 1e-3, 4e-3, lip)
    assert roa.c_star == 0.0 and "jump" in roa.reason

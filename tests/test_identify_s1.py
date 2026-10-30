"""End-to-end Phase I on synthetic S1 data with known damping/frictionloss and an OU residual."""
from pathlib import Path

import numpy as np
import pytest

from physics.fit_cmaes import bic, fit, identifiability, laplace, posterior_samples
from physics.identify_s1 import LOWER, NAMES, PHYSICAL, UPPER, innovations, make_objective, model_for, residual
from physics.rollout import rollout
from verifier.gates import g0_admissible, g1_heldout

S1 = (Path(__file__).parents[1] / "physics/models/s1_pendulum.xml").read_text()
TRUE = np.log([0.004, 0.0008, 2.0, 0.01])  # damping, frictionloss, kappa, sigma
MEAS = 0.003  # rad, smoother output noise


@pytest.fixture(scope="module")
def take():
    rng = np.random.default_rng(7)
    t = np.cumsum(rng.uniform(0.9, 1.1, 600) / 60.0)
    t -= t[0]
    clean = rollout(model_for(S1, TRUE), np.array([0.6]), np.array([0.0]), t)
    r = np.zeros(t.size)
    for k in range(1, t.size):
        a = np.exp(-2.0 * (t[k] - t[k - 1]))
        r[k] = a * r[k - 1] + np.sqrt(0.01**2 / 4.0 * (1 - a**2)) * rng.standard_normal()
    theta = clean[:, 0] + r + MEAS * rng.standard_normal(t.size)
    qvel = np.gradient(clean[:, 0], t)  # stands in for the RTS-smoothed rate
    return t, theta, qvel


@pytest.fixture(scope="module")
def fitted(take):
    t, theta, qvel = take
    n = int(0.7 * t.size)
    x0 = np.r_[np.log([0.01, 0.001, 1.0, 0.02]), 0, 0]
    obj = make_objective(S1, t[:n], theta[:n], (theta[0], qvel[0]), MEAS**2, x0)
    x_hat, _ = fit(obj, x0, 1.0, LOWER, UPPER, seed=1, max_evals=1500)
    return x_hat, laplace(obj, x_hat, step=0.02)


def test_fit_recovers_damping_within_posterior(fitted):
    x_hat, post = fitted
    z = np.abs(x_hat[:2] - TRUE[:2]) / np.sqrt(np.diag(post.cov)[:2])
    assert np.all(z < 3), dict(zip(NAMES, np.exp(x_hat)))
    assert identifiability(post, NAMES, PHYSICAL)[0]


def test_g1_passes_true_structure_and_fails_wrong_one(take, fitted):
    t, theta, qvel = take
    x_hat, _ = fitted
    n = int(0.7 * t.size)
    th, tt, s0 = theta[n:], t[n:], (theta[n], qvel[n])
    x = np.r_[x_hat[:4], 0, 0]  # held-out segment starts from the smoother's state, no fitted offset
    pred = th - residual(S1, x, tt, th, s0)
    assert g1_heldout(th, pred, innovations(S1, x, tt, th, s0, MEAS**2)).passed

    wrong = x + np.r_[np.log(30), 0, 0, 0, 0, 0]  # 30x damping
    bad = g1_heldout(th, th - residual(S1, wrong, tt, th, s0), innovations(S1, wrong, tt, th, s0, MEAS**2))
    assert not bad.passed and bad.witness["nrmse"] > 0.10


def test_posterior_samples_are_g0_admissible(fitted):
    _, post = fitted
    for x in posterior_samples(post, 16, seed=0):
        b, fl = np.exp(x[:2])
        patch = {".//joint[@name='hinge']@damping": f"{b:.9g}", ".//joint[@name='hinge']@frictionloss": f"{fl:.9g}"}
        assert g0_admissible(S1, patch).passed


def test_g0_rejects_negative_damping_and_off_whitelist():
    assert not g0_admissible(S1, {".//joint[@name='hinge']@damping": "-0.01"}).passed
    assert not g0_admissible(S1, {".//option@timestep": "0.1"}).passed


def test_identifiability_refuses_flat_direction():
    from physics.fit_cmaes import Posterior
    flat = Posterior(np.zeros(2), np.diag([1e-4, 1e4]), np.diag([1e4, 1e-4]), 0.0)
    ok, why = identifiability(flat, ["a", "b"], [0, 1])
    assert not ok and "cond" in why


def test_bic_penalises_extra_parameters():
    assert bic(10.0, 3, 600) > bic(10.0, 2, 600)


def test_identifiability_names_the_loose_parameter():
    from physics.fit_cmaes import Posterior
    loose = Posterior(np.zeros(2), np.diag([1e-4, 4.0]), np.diag([1e4, 0.25]), 0.0)
    ok, why = identifiability(loose, ["damping", "frictionloss"], [0, 1])
    assert not ok and why.startswith("frictionloss")


def test_g1_refuses_nan_and_flat_channels():
    y = np.sin(np.linspace(0, 6, 200))
    white = np.random.default_rng(0).standard_normal(200)
    assert not g1_heldout(y, np.full(200, np.nan), white).passed
    assert not g1_heldout(y, y, np.full(200, np.nan)).passed
    assert not g1_heldout(np.ones(200), np.ones(200), white).passed

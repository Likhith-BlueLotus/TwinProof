"""End-to-end Phase I on synthetic S1 data with known damping/frictionloss and an OU residual."""
from pathlib import Path

import numpy as np
import pytest

from agents.dispatch import FIT_MAX_EVALS
from physics.fit_cmaes import bic, fit, identifiability, laplace, posterior_samples
from physics.identify_s1 import LOWER, NAMES, PHYSICAL, UPPER, make_objective, model_for
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
    x0 = np.r_[np.log([0.01, 0.001, 1.0, 0.02]), 0, 0, np.zeros(4)]
    obj = make_objective(S1, t[:n], theta[:n], (theta[0], qvel[0]), MEAS**2, x0)
    x_hat, _ = fit(obj, x0, 1.0, LOWER, UPPER, seed=1, max_evals=FIT_MAX_EVALS)
    return x_hat, laplace(obj, x_hat, step=0.02)


def test_fit_recovers_damping_within_posterior(fitted):
    x_hat, post = fitted
    z = np.abs(x_hat[:2] - TRUE[:2]) / np.sqrt(np.diag(post.cov)[:2])
    assert np.all(z < 3), dict(zip(NAMES, np.exp(x_hat)))
    assert identifiability(post, NAMES, PHYSICAL)[0]


def test_g1_passes_true_structure_and_fails_wrong_one(take, fitted):
    from physics.identify_s1 import start_state
    from verifier.verify import Evidence, _g1
    t, theta, qvel = take
    x_hat, post = fitted
    n = int(0.7 * t.size)
    start = (t[0], *start_state(x_hat, (theta[0], qvel[0])))  # the fitted trajectory continues into the held-out 30%
    ev = Evidence(S1, x_hat, post.cov, t[:n], theta[:n], t[n:], theta[n:], start, MEAS**2, goal=np.pi, success_tol=0.05)
    assert _g1(model_for(S1, x_hat), ev).passed

    wrong = model_for(S1, x_hat + np.r_[np.log(30), np.zeros(9)])  # 30x damping
    bad = _g1(wrong, ev)
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


def test_calibration_round_trip():
    from physics.identify_s1 import observe, unobserve
    x = np.r_[np.zeros(6), 1.0, 2.0, -1.5, 0.8]
    q = np.linspace(-0.8, 0.8, 9)
    phi = observe(q, x)
    dq = 1e-6
    for qi, pi in zip(q, phi):
        rate = (observe(np.array([qi + dq]), x)[0] - observe(np.array([qi - dq]), x)[0]) / (2 * dq) * 0.7
        q_back, v_back = unobserve(pi, rate, x)
        assert q_back == pytest.approx(qi, abs=1e-9) and v_back == pytest.approx(0.7, rel=1e-5)


def test_laplace_covariance_is_psd_even_with_a_saddle_direction():
    obj = lambda x: x[0] ** 2 - 1e-3 * x[1] ** 2  # noqa: E731
    post = laplace(obj, np.zeros(2), step=1e-2)
    assert np.all(np.linalg.eigvalsh(post.cov) > 0)


def test_fit_is_reproducible_for_seed_zero():
    obj = lambda x: float(np.sum((x - 0.3) ** 2))  # noqa: E731
    a = fit(obj, np.zeros(3), 0.5, -np.ones(3), np.ones(3), seed=0, max_evals=60)
    b = fit(obj, np.zeros(3), 0.5, -np.ones(3), np.ones(3), seed=0, max_evals=60)
    assert np.array_equal(a[0], b[0])

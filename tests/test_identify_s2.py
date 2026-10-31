"""S2 on synthetic takes: BIC picks rolling vs sliding, mu is recovered when sliding, G1 rejects the wrong structure."""
from pathlib import Path

import mujoco
import numpy as np
import pytest

from physics.fit_cmaes import fit, identifiability, laplace
from physics.identify_s2 import (NAMES, bounds, classify_contact, make_objective, release_time, residuals, simulate, xml_for)
from physics.ou import ou_innovations
from verifier.gates import g1_heldout

S2 = (Path(__file__).parents[1] / "physics/models/s2_incline.xml").read_text()
FPS, T_END, RELEASE = 60, 1.1, 0.25
MEAS_S, MEAS_PSI = 0.0008, 0.02
EVALS = 2500


def _take(mu, roll, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(0, T_END, 1 / FPS)
    model = mujoco.MjModel.from_xml_string(xml_for(S2, np.r_[np.log([mu, roll]), 0, 0, 0, 0, 0]))
    s, psi = simulate(model, t, RELEASE)
    return t, s + MEAS_S * rng.standard_normal(t.size), psi + MEAS_PSI * rng.standard_normal(t.size)


def _fit(t, s, psi, structure):
    n = int(0.7 * t.size)
    lower, upper = bounds(structure)
    x0 = np.clip(np.r_[np.log([0.5, 1e-4, 5.0, 0.01]), 0, 0, 0, np.log(0.7)], lower, upper)
    obj = make_objective(S2, t[:n], s[:n], psi[:n], release_time(t, s), MEAS_S**2, MEAS_PSI**2, x0)
    x, j = fit(obj, x0, 1.0, lower, upper, seed=0, max_evals=EVALS)
    return x, j, obj, n


def _g1(x, t, s, psi, n):
    rs, _ = residuals(S2, x, t, s, psi, release_time(t, s))
    nu, var = ou_innovations(rs[n:], t[n:], np.exp(x[2:3]), np.exp(x[3:4]), MEAS_S**2)
    return g1_heldout(s[n:], s[n:] - rs[n:], (nu / np.sqrt(var))[:, 0])


@pytest.fixture(scope="module")
def rolling_take():
    return _take(mu=0.6, roll=2e-4, seed=1)


@pytest.fixture(scope="module")
def sliding_take():
    return _take(mu=0.05, roll=1e-6, seed=2)


def test_slip_residual_classifies_rolling_and_sliding(rolling_take, sliding_take):
    radius = 0.037
    assert classify_contact(*rolling_take, radius)[0] == "incline_rolling"
    assert classify_contact(*sliding_take, radius)[0] == "incline_sliding"


def test_sliding_can_mu_recovered_and_identifiable(sliding_take):
    t, s, psi = sliding_take
    x, _, obj, n = _fit(t, s, psi, "incline_sliding")
    post = laplace(obj, x, step=0.02)
    assert np.exp(x[0]) == pytest.approx(0.05, rel=0.1), dict(zip(NAMES, np.exp(x[:4])))
    assert identifiability(post, NAMES, [0])[0]
    assert _g1(x, t, s, psi, n).passed


def test_g1_rejects_rolling_structure_on_a_sliding_can(sliding_take):
    t, s, psi = sliding_take
    x, _, _, n = _fit(t, s, psi, "incline_rolling")
    res = _g1(x, t, s, psi, n)
    assert not res.passed, res.witness


def test_release_time_detects_motion_onset(rolling_take):
    t, s, _ = rolling_take
    assert abs(release_time(t, s) - RELEASE) < 0.15


def test_s2_verdicts(sliding_take):
    from verifier.s2 import verify_s2_structure
    t, s, psi = sliding_take
    x, _, _, n = _fit(t, s, psi, "incline_sliding")
    args = (S2, x, t, s, psi, n, release_time(t, s), MEAS_S**2)
    good = verify_s2_structure(*args)
    assert all(g.passed for g in good.values()) and list(good) == ["G7", "G0", "G1"]
    x_bad = x.copy()
    x_bad[0] = np.log(3.0)  # mu above the admissible 1.5
    assert not verify_s2_structure(S2, x_bad, *args[2:])["G0"].passed
    hack = verify_s2_structure(*args, extra_patch={".//option@gravity": "0 0 0"})
    assert not hack["G7"].passed and "G0" not in hack
    xr, _, _, _ = _fit(t, s, psi, "incline_rolling")
    assert not verify_s2_structure(S2, xr, *args[2:])["G1"].passed


@pytest.mark.parametrize("beta", [0.5, 0.8, 1.0])
def test_measured_twin_rolls_at_theoretical_acceleration(beta):
    from physics.identify_s2 import s2_xml
    model = mujoco.MjModel.from_xml_string(xml_for(s2_xml(S2, 12.0, 0.0325, 0.0685), np.r_[np.log([1.0, 1e-6]), 0, 0, 0, 0, 0, np.log(beta)]))
    t = np.arange(0, 0.8, 1 / FPS)
    s, psi = simulate(model, t, 0.0)
    accel = np.polyfit(t[5:], s[5:], 2)[0] * 2
    assert accel == pytest.approx(9.81 * np.sin(np.radians(12)) / (1 + beta), rel=0.01)
    assert s[-1] / (psi[-1] * 0.0325) == pytest.approx(1.0, abs=0.01)  # no slip

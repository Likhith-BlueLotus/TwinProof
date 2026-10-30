"""Full verifier chain for S1 claims: G7 -> G0 -> G1 -> G2 -> G3 -> G4 -> G5, stopping at the first FAIL.

G7's patch guard runs first because applying an unsafe patch is itself the hazard. Every verdict is
reproducible from (claim, evidence, seed).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import mujoco
import numpy as np

from physics.controllers import CONTROL_PERIOD, LQRSpec, closed_loop_step, policy
from physics.identify_s1 import observe, xml_for
from physics.mjcf_patch import apply_patch
from physics.monitor import contact_metrics
from physics.ekf import innovations as ekf_innovations, tune_accel_psd
from physics.rollout import rollout
from verifier.g7_scan import scan_source
from verifier.gates import GateResult, g0_admissible, g1_heldout, g4_contacts, g5_robustness, g7_patch_guard
from verifier.pipeline import verify_s1_lqr

N_POSTERIOR = 256
N_ROBUST = 300
HORIZON_S = 5.0
MONITOR_EVERY = 10  # control periods between G4 contact checks
F_MAX = 40.0  # N


@dataclass(frozen=True)
class Claim:
    claim_id: str
    proposer: str
    patch: dict[str, str] = field(default_factory=dict)
    controller: LQRSpec | None = None
    code: dict[str, str] = field(default_factory=dict)  # path -> source of any proposed Python diff


@dataclass(frozen=True)
class Evidence:
    """Identification output for one scenario: base model, posterior and the held-out segment."""

    base_xml: str
    x_hat: np.ndarray
    cov: np.ndarray
    t_fit: np.ndarray  # fit window: tunes the EKF process noise only
    theta_fit: np.ndarray
    t_heldout: np.ndarray
    theta_heldout: np.ndarray
    start: tuple[float, float, float]  # (t0, q0, v0) true-space state the held-out rollout continues from
    r_meas: float
    goal: float
    success_tol: float


@dataclass
class Verdict:
    claim: Claim
    gates: dict[str, GateResult]

    @property
    def admitted(self) -> bool:
        return all(g.passed for g in self.gates.values())

    @property
    def first_failure(self) -> str | None:
        return next((k for k, g in self.gates.items() if not g.passed), None)

    def to_record(self) -> dict:
        fail = self.first_failure
        return {
            "claim_id": self.claim.claim_id,
            "type": "controller" if self.claim.controller else "structure",
            "proposer": self.claim.proposer,
            "gates": {k: {"pass": g.passed, **g.witness} for k, g in self.gates.items()},
            "verdict": "ADMITTED" if fail is None else "REJECTED",
            "reason": "" if fail is None else self.gates[fail].feedback_to_agent,
            "feedback_to_agent": "" if fail is None else self.gates[fail].feedback_to_agent,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_record(), default=float)


def verify(claim: Claim, ev: Evidence, seed: int) -> Verdict:
    gates: dict[str, GateResult] = {}

    def run(name: str, result: GateResult) -> bool:
        gates[name] = result
        return result.passed

    target = claim.controller.q_eq if claim.controller else None
    if not run("G7", _g7(claim, ev, target)):
        return Verdict(claim, gates)
    nominal_xml = xml_for(ev.base_xml, ev.x_hat)
    if not run("G0", g0_admissible(nominal_xml, claim.patch)):
        return Verdict(claim, gates)
    nominal = mujoco.MjModel.from_xml_string(apply_patch(nominal_xml, claim.patch))
    if not run("G1", _g1(nominal, ev)) or claim.controller is None:
        return Verdict(claim, gates)

    rng = np.random.default_rng(seed)
    xs = rng.multivariate_normal(ev.x_hat, ev.cov, size=N_POSTERIOR + N_ROBUST)
    models = [mujoco.MjModel.from_xml_string(apply_patch(xml_for(ev.base_xml, x), claim.patch)) for x in xs]
    kappa, sigma = np.exp(ev.x_hat[2:4])
    ctrl = verify_s1_lqr(nominal, models[:N_POSTERIOR], claim.controller, kappa, sigma)
    for name, res in ctrl.gates.items():
        if not run(name, res):
            return Verdict(claim, gates)

    X, U = _ensemble(models[N_POSTERIOR:], ctrl.K, ctrl.P, ctrl.roa.c_star, claim.controller, kappa, sigma, rng)
    q = X[:, ::MONITOR_EVERY, 0] + claim.controller.q_eq
    metrics = contact_metrics(nominal, q.reshape(-1, 1), X[:, ::MONITOR_EVERY, 1].reshape(-1, 1))
    if not run("G4", g4_contacts(metrics, F_MAX)):
        return Verdict(claim, gates)
    u_limit = float(nominal.actuator_ctrlrange[0, 1])
    V = np.einsum("ntj,jk,ntk->nt", X, ctrl.P, X)
    failed = (np.abs(U).max(axis=(1, 2)) > u_limit) | (V.max(1) > ctrl.roa.c_star) | (np.abs(X[:, -1, 0] + claim.controller.q_eq - ev.goal) > ev.success_tol)
    run("G5", g5_robustness(int(failed.sum()), len(failed)))
    return Verdict(claim, gates)


def _g7(claim: Claim, ev: Evidence, target: float | None) -> GateResult:
    violations = [v for path, src in claim.code.items() for v in scan_source(src, path)]
    if violations:
        return GateResult(False, {"violations": float(len(violations))}, "G7: code refused: " + "; ".join(violations))
    return g7_patch_guard(ev.base_xml, claim.patch, target, ev.goal)


def _g1(model: mujoco.MjModel, ev: Evidence) -> GateResult:
    """Open-loop NRMSE of the continued trajectory on the held-out window, and whiteness of the EKF's
    one-step innovations there (process noise tuned on the fit window, then frozen)."""
    # continue the fitted trajectory into the held-out window: a restart from a re-estimated state would hide or add phase error
    t0, q0, v0 = ev.start
    pred = observe(rollout(model, np.array([q0]), np.array([v0]), np.r_[t0, ev.t_heldout])[1:, 0], ev.x_hat)
    t_all, y_all = np.r_[ev.t_fit, ev.t_heldout], np.r_[ev.theta_fit, ev.theta_heldout]
    psd = tune_accel_psd(model, ev.x_hat, ev.t_fit, ev.theta_fit, (q0, v0), ev.r_meas)
    inn = ekf_innovations(model, ev.x_hat, t_all, y_all, (q0, v0), ev.r_meas, psd)
    held = slice(ev.t_fit.size, None)
    result = g1_heldout(ev.theta_heldout, pred, (inn.nu / np.sqrt(inn.var))[held])
    return result._replace(witness={**result.witness, "accel_psd": psd})


def _ensemble(models, K, P, c_star, spec: LQRSpec, kappa, sigma, rng) -> tuple[np.ndarray, np.ndarray]:
    """Closed-loop rollouts from uniform starts in {V <= c*}, posterior model per row, OU disturbance on theta."""
    n, steps = len(models), round(HORIZON_S / CONTROL_PERIOD)
    w, v = np.linalg.eigh(P)
    z = rng.standard_normal((n, 2))
    z *= (np.sqrt(c_star * rng.uniform(size=n)) / np.linalg.norm(z, axis=1))[:, None]  # uniform in the disk
    x = z @ (v @ np.diag(1 / np.sqrt(w)) @ v.T)
    a = np.exp(-kappa * CONTROL_PERIOD)
    q_ou = sigma**2 / (2 * kappa) * (1 - a**2)
    r = np.zeros(n)
    X, U = np.empty((n, steps + 1, 2)), np.empty((n, steps, 1))
    X[:, 0] = x
    for k in range(steps):
        U[:, k] = policy(x, K, spec)
        r_next = a * r + np.sqrt(q_ou) * rng.standard_normal(n)
        x = closed_loop_step(models, x, K, spec)
        x[:, 0] += r_next - r
        r = r_next
        X[:, k + 1] = x
    return X, U

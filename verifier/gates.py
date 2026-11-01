"""Deterministic verifier gates G0-G7 (master doc §2.9). Every gate is a pure function returning GateResult."""
from __future__ import annotations

from typing import NamedTuple

import mujoco
import numpy as np

from physics.mjcf_patch import PatchRejected, apply_patch
from verifier.bounds import input_bound
from verifier.lyapunov import RoaResult
from verifier.stats import clopper_pearson_upper, ljung_box_p

MU_MAX = 1.5
SPECTRAL_MAX = 0.98
CLEARANCE_MIN = 0.01  # m, non-whitelisted geom pairs
FRICTION_CONE_MARGIN = 0.85
P_FAIL_MAX = 0.02
TARGET_TOL = 1e-9
NRMSE_MAX = 0.10
LJUNG_BOX_P_MIN = 0.05


class GateResult(NamedTuple):
    passed: bool
    witness: dict[str, float]
    feedback_to_agent: str


def g0_admissible(base_xml: str, patch: dict[str, str]) -> GateResult:
    """Parses, whitelisted fields only, physical masses/inertias/dissipation, solref >= 2 timesteps."""
    try:
        # MuJoCo reads "inf" as 0, so a non-finite value would build a different model than the one claimed
        if not all(np.isfinite(float(t)) for v in patch.values() for t in str(v).split()):
            raise ValueError("patch values must be finite numbers")
        m = mujoco.MjModel.from_xml_string(apply_patch(base_xml, patch))
    except (PatchRejected, ValueError) as err:
        return GateResult(False, {}, f"G0: {err}")
    body = slice(1, None)  # body 0 is the world
    inertia = m.body_inertia[body]
    a, b, c = inertia[:, 0], inertia[:, 1], inertia[:, 2]
    checks = {
        "min_body_mass": (float(m.body_mass[body].min()), m.body_mass[body].min() > 0),
        "triangle_slack": (float(np.min([a + b - c, a + c - b, b + c - a])), np.all((a + b >= c) & (a + c >= b) & (b + c >= a))),
        "min_damping": (float(m.dof_damping.min(initial=0)), np.all(m.dof_damping >= 0)),
        "min_frictionloss": (float(m.dof_frictionloss.min(initial=0)), np.all(m.dof_frictionloss >= 0)),
        "min_armature": (float(m.dof_armature.min(initial=0)), np.all(m.dof_armature >= 0)),
        "max_mu": (float(m.geom_friction[:, 0].max(initial=0)), np.all((m.geom_friction[:, 0] >= 0) & (m.geom_friction[:, 0] <= MU_MAX))),
    }
    timeconst = m.geom_solref[:, 0][m.geom_solref[:, 0] > 0]  # positive solref[0] is a time constant
    checks["min_solref_over_timestep"] = (float(timeconst.min(initial=np.inf) / m.opt.timestep), np.all(timeconst >= 2 * m.opt.timestep))
    witness = {k: v for k, (v, _) in checks.items()}
    failed = [k for k, (_, ok) in checks.items() if not ok]
    if failed:
        return GateResult(False, witness, f"G0: inadmissible model ({', '.join(failed)})")
    return GateResult(True, witness, "")


def g1_heldout(y_obs: np.ndarray, y_pred: np.ndarray, innovations: np.ndarray) -> GateResult:
    """Held-out NRMSE <= 0.10 on every channel and white OU innovations (Ljung-Box p > 0.05)."""
    y_obs, y_pred = np.atleast_2d(y_obs.T).T, np.atleast_2d(y_pred.T).T
    span = np.ptp(y_obs, 0)
    if not np.all(np.isfinite(span) & (span > 0)):
        return GateResult(False, {}, "G1: held-out channel is flat or non-finite; cannot score prediction")
    nrmse = float(np.max(np.sqrt(np.mean((y_obs - y_pred) ** 2, 0)) / span))
    try:
        lb_p = ljung_box_p(innovations)
    except ValueError:
        lb_p = float("nan")
    witness = {"nrmse": nrmse, "ljung_box_p": lb_p}
    # comparisons are written so NaN fails: a diverged twin must never pass
    if not nrmse <= NRMSE_MAX:
        return GateResult(False, witness, f"G1: held-out NRMSE {nrmse:.3f} > {NRMSE_MAX}; the structure does not predict unseen motion")
    if not lb_p > LJUNG_BOX_P_MIN:
        return GateResult(False, witness, f"G1: innovations not white (Ljung-Box p={lb_p:.3g}); unmodelled dynamics remain")
    return GateResult(True, witness, "")


def g2_stability(A_s: np.ndarray, B_s: np.ndarray, K: np.ndarray, P: np.ndarray, roa: RoaResult) -> GateResult:
    """Spectral radius <= 0.98 and a common Lyapunov decrease for every posterior (A, B); sound ROA c* > c0."""
    P = (P + P.T) / 2
    A_cl = A_s - B_s @ K
    rho = float(np.max(np.abs(np.linalg.eigvals(A_cl))))
    lyap = float(np.max(np.linalg.eigvalsh(np.transpose(A_cl, (0, 2, 1)) @ P @ A_cl - P)))
    witness = {"max_spectral_radius": rho, "max_lyapunov_eig": lyap, "c_star": roa.c_star, "c0": roa.c0,
               "lipschitz": roa.lipschitz, "friction_eps": roa.friction_eps, "grid_points": float(roa.grid_points)}
    p_min = float(np.linalg.eigvalsh(P).min())
    witness["min_P_eig"] = p_min
    if not p_min > 0:
        return GateResult(False, witness, f"G2: Lyapunov matrix not positive definite (min eig {p_min:.3g}); the nominal closed loop is unstable")
    if not rho <= SPECTRAL_MAX:
        return GateResult(False, witness, f"G2: closed-loop spectral radius {rho:.3f} > {SPECTRAL_MAX} on the posterior; increase gains or penalise state more")
    if not lyap < 0:
        return GateResult(False, witness, f"G2: no common Lyapunov decrease across the posterior (max eig {lyap:.3g} >= 0)")
    if not roa.c_star > roa.c0:
        return GateResult(False, witness, f"G2: no certified region of attraction ({roa.reason})")
    return GateResult(True, witness, "")


def g3_bounds(K: np.ndarray, P: np.ndarray, c_star: float, u_max: float, u_limit: float) -> GateResult:
    """Applied input bound min(sqrt(c* k^T P^-1 k), u_max) must not exceed the actuator limit."""
    support = float(input_bound(K, P, c_star).max())
    applied = float(np.minimum(support, u_max))  # propagates NaN, unlike min()
    witness = {"u_max_bound": applied, "u_support_bound": support, "u_limit": u_limit}
    if not applied <= u_limit:
        cut = 1 - u_limit / support
        return GateResult(False, witness, f"G3: certified torque bound {applied:.3g} exceeds {u_limit:.3g} limit; reduce ||k||_P^-1 by {cut:.0%} or shrink c*")
    return GateResult(True, witness, "")


def g4_contacts(metrics: dict[str, float], f_max: float) -> GateResult:
    """Across all ensemble rollouts: clearance >= 1 cm, joint-limit margin >= 0, normal force <= f_max,
    friction cone |f_t| <= 0.85 mu f_n. metrics come from physics.monitor.contact_metrics."""
    fails = []
    if not metrics["min_clearance"] >= CLEARANCE_MIN:
        fails.append(f"clearance {metrics['min_clearance']:.3g} m < {CLEARANCE_MIN} m")
    if not metrics["min_limit_margin"] >= 0:
        fails.append(f"joint limit exceeded by {-metrics['min_limit_margin']:.3g}")
    if not metrics["max_normal_force"] <= f_max:
        fails.append(f"normal force {metrics['max_normal_force']:.3g} N > {f_max} N")
    if not metrics["max_cone_ratio"] <= FRICTION_CONE_MARGIN:
        fails.append(f"friction cone ratio {metrics['max_cone_ratio']:.3g} > {FRICTION_CONE_MARGIN}")
    if fails:
        return GateResult(False, metrics, "G4: " + "; ".join(fails))
    return GateResult(True, metrics, "")


def g5_robustness(failures: int, n: int) -> GateResult:
    """One-sided 95% Clopper-Pearson upper bound on the failure rate must be <= 2%."""
    p_up = clopper_pearson_upper(failures, n)
    witness = {"failures": float(failures), "rollouts": float(n), "p_fail_upper": p_up}
    if not p_up <= P_FAIL_MAX:
        return GateResult(False, witness, f"G5: {failures}/{n} posterior rollouts failed; 95% upper bound {p_up:.2%} > {P_FAIL_MAX:.0%}")
    return GateResult(True, witness, "")


def g7_patch_guard(base_xml: str, patch: dict[str, str], controller_target: float | None, goal: float) -> GateResult:
    """Refuse patches off the whitelist (goal sites, <option>, actuator limits, ...) and controllers aimed elsewhere."""
    try:
        apply_patch(base_xml, patch)
    except PatchRejected as err:
        return GateResult(False, {}, f"G7: patch refused: {err}")
    if controller_target is not None and abs(controller_target - goal) > TARGET_TOL:
        return GateResult(False, {"controller_target": controller_target, "goal": goal},
                          f"G7: controller target {controller_target:.4g} is not the task goal {goal:.4g}")
    return GateResult(True, {}, "")

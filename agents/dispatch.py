"""Tool handlers. Thin and validated: they build claims and run the verifier; they never decide."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np

from perception.smoother import rts_smooth
from physics.controllers import LQRSpec
from physics.fit_cmaes import bic, fit, identifiability, laplace
from physics.identify_s1 import LOWER, N_CAL, NAMES, PHYSICAL, REPORTED, UPPER, make_objective, start_state
from verifier.verify import Claim, Evidence, verify

FIT_FRACTION = 0.7
FIT_MAX_EVALS = 5000  # 10 coordinates (physics, OU, start state, calibration)
VISCOUS_ONLY_FRICTIONLOSS = 1e-7  # "pendulum_viscous" pins Coulomb friction to ~0
DEFAULT_Q, DEFAULT_R = (10.0, 0.1), (10.0,)


@dataclass
class Context:
    """One S1 session: the identification take, the evidence once fitted, and every claim and verdict."""

    base_xml: str
    t: np.ndarray
    phi: np.ndarray
    r_meas: float
    emit: Callable[[str, object], None]
    goal: float = float(np.pi)
    success_tol: float = 0.05
    seed: int = 0
    evidence: Evidence | None = None
    refusal: str | None = None  # why the latest fit is not usable (identifiability gate)
    patch: dict[str, str] = field(default_factory=dict)  # admitted structure patch, carried into controllers
    claims: dict[str, Claim] = field(default_factory=dict)
    records: list[dict] = field(default_factory=list)
    last_controller: str | None = None
    proposer: str = "agent"

    def new_ref(self) -> str:
        return f"c-{len(self.claims) + 1}"

    def task_brief(self) -> str:
        return (f"Scenario S1 (pendulum). Identification take: {self.t.size} frames over {self.t[-1]:.1f} s, swing angle "
                f"from SAM tracking, measurement noise {np.sqrt(self.r_meas):.2g} rad. Task: hold the pendulum upright "
                f"(angle pi) within {self.success_tol} rad after 5 s, using the hinge motor (limit 2.5 N m). "
                "Candidate structures: " + ", ".join(["pendulum_viscous", "pendulum_viscous_coulomb"]) + ".")


def fit_batch(args: dict, ctx: Context) -> dict:
    structure = args["structure"]
    if structure not in ("pendulum_viscous", "pendulum_viscous_coulomb"):
        return {"error": f"structure {structure!r} is not available for S1"}
    n = int(FIT_FRACTION * ctx.t.size)
    sm = rts_smooth(ctx.t, ctx.phi, sigma=np.sqrt(ctx.r_meas), q_tilde=50.0).states
    lower, upper = LOWER.copy(), UPPER.copy()
    if structure == "pendulum_viscous":
        lower[1] = np.log(VISCOUS_ONLY_FRICTIONLOSS)
        upper[1] = lower[1] + 1e-6  # CMA-ES needs lower < upper; this pins frictionloss
    x0 = np.clip(np.r_[np.log([0.01, 0.001, 1.0, 0.02]), 0, 0, np.zeros(N_CAL)], lower, upper)
    obj = make_objective(ctx.base_xml, ctx.t[:n], ctx.phi[:n], (sm[0, 0], sm[0, 1]), ctx.r_meas, x0)
    x_hat, j_min = fit(obj, x0, 1.0, lower, upper, seed=ctx.seed, max_evals=FIT_MAX_EVALS)
    post = laplace(obj, x_hat, step=0.02)
    free = [i for i in PHYSICAL if lower[i] < upper[i] - 1e-5]
    ok, why = identifiability(post, NAMES, free)
    cov = post.cov.copy()
    if structure == "pendulum_viscous":
        cov[1, :] = cov[:, 1] = 0.0
        cov[1, 1] = 1e-12
    start = (float(ctx.t[0]), *start_state(x_hat, (sm[0, 0], sm[0, 1])))
    ctx.evidence = Evidence(ctx.base_xml, x_hat, cov, ctx.t[:n], ctx.phi[:n], ctx.t[n:], ctx.phi[n:], start, ctx.r_meas,
                            goal=ctx.goal, success_tol=ctx.success_tol)
    result = {"structure": structure, "identifiable": ok, "reason": why, "bic": bic(j_min, len(free), n),
              "params": {NAMES[i]: float(np.exp(x_hat[i])) for i in REPORTED},
              "rel_std": {NAMES[i]: float(np.sqrt(post.cov[i, i])) for i in REPORTED if lower[i] < upper[i] - 1e-5}}
    ctx.refusal = None if ok else f"twin not identifiable ({why}): fit another structure or film a new experiment"
    # full evidence for the UI and operator what-ifs; kept out of the tool result so the agent's context stays small
    ctx.emit("evidence", {"structure": structure, "x": x_hat.tolist(), "cov": cov.tolist(), "start": list(start), "n_fit": n})
    ctx.emit("fit", result)
    return result


def propose_mjcf_patch(args: dict, ctx: Context) -> dict:
    patch = {f"{p['xpath']}@{p['attr']}": str(p["value"]) for p in args["patches"]}
    ref = ctx.new_ref()
    ctx.claims[ref] = Claim(ref, ctx.proposer, patch={**ctx.patch, **patch})
    ctx.emit("claim", {"ref": ref, "type": "structure", "patch": patch, "rationale": args.get("rationale", "")})
    return {"ref": ref}


def synthesize_controller(args: dict, ctx: Context) -> dict:
    if args["kind"] != "lqr":
        return {"error": f"controller kind {args['kind']!r} is not available for S1"}
    gains = args.get("gains")
    spec = LQRSpec(
        q_eq=ctx.goal,
        q_diag=tuple(args.get("Q_diag") or DEFAULT_Q),
        r_diag=tuple(args.get("R_diag") or DEFAULT_R),
        u_max=float(args["u_max"]),
        gains=(float(gains["k_theta"]), float(gains["k_omega"])) if gains else None,
    )
    ref = ctx.new_ref()
    ctx.claims[ref] = Claim(ref, ctx.proposer, patch=dict(ctx.patch), controller=spec)
    ctx.emit("claim", {"ref": ref, "type": "controller", "spec": spec.__dict__})
    return {"ref": ref}


def submit_claim(args: dict, ctx: Context) -> dict:
    claim = ctx.claims.get(args["ref"])
    if claim is None:
        return {"error": f"unknown claim ref {args['ref']!r}"}
    if ctx.evidence is None:
        return {"error": "no fitted model yet: call fit_batch first"}
    if ctx.refusal:
        return {"verdict": "REFUSED", "reason": ctx.refusal}
    record = verify(claim, ctx.evidence, ctx.seed).to_record()
    if record["verdict"] == "ADMITTED":
        if claim.controller is None:
            ctx.patch = dict(claim.patch)
        else:
            ctx.last_controller = claim.claim_id
    ctx.records.append(record)
    ctx.emit("verdict", record)
    return record


def reverify(args: dict, ctx: Context) -> dict:
    """Operator what-if on the environment (not an agent patch): rerun the last admitted controller."""
    if ctx.last_controller is None or ctx.evidence is None:
        return {"error": "no admitted controller to re-verify"}
    root = ET.fromstring(ctx.evidence.base_xml)
    if "mass_scale" in args:
        for geom in root.iter("geom"):
            if geom.get("mass"):
                geom.set("mass", f"{float(geom.get('mass')) * args['mass_scale']:.9g}")
    if "torque_limit" in args:
        for motor in root.iter("motor"):
            motor.set("ctrlrange", f"{-args['torque_limit']:.9g} {args['torque_limit']:.9g}")
    x_hat = ctx.evidence.x_hat.copy()
    if "damping_scale" in args:
        x_hat[0] += np.log(max(args["damping_scale"], 1e-6))
    what_if = replace(ctx.evidence, base_xml=ET.tostring(root, encoding="unicode"), x_hat=x_hat)
    claim = replace(ctx.claims[ctx.last_controller], claim_id=f"{ctx.last_controller}-whatif")
    record = verify(claim, what_if, ctx.seed).to_record() | {"what_if": args}
    ctx.records.append(record)
    ctx.emit("verdict", record)
    return record


DISPATCH: dict[str, Callable[[dict, Context], dict]] = {
    "fit_batch": fit_batch,
    "propose_mjcf_patch": propose_mjcf_patch,
    "synthesize_controller": synthesize_controller,
    "submit_claim": submit_claim,
    "reverify": reverify,
}

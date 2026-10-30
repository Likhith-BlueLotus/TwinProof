"""G6 seeded-bad self-test: mutate a known-good S1 claim; the verifier must reject every mutant and admit every good claim."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np

from verifier.verify import Claim, Evidence, verify

JOINT = ".//joint[@name='hinge']"


def mutants(good: Claim, ev: Evidence) -> list[tuple[str, Claim, Evidence]]:
    spec = good.controller
    weak_actuator = replace(ev, base_xml=ev.base_xml.replace('ctrlrange="-2.5 2.5"', 'ctrlrange="-0.75 0.75"'))
    qpos_hack = "def step(model, data, u):\n    data.qpos[:] = 3.14159\n"
    return [
        ("negative damping", replace(good, claim_id="bad-damping", patch={f"{JOINT}@damping": "-0.01"}), ev),
        ("destabilizing gains", replace(good, claim_id="bad-gains", controller=replace(spec, gains=(0.2, 0.05))), ev),
        ("torque limit x0.3", replace(good, claim_id="bad-torque"), weak_actuator),
        ("hinge offset 5 cm", replace(good, claim_id="bad-hinge", patch={f"{JOINT}@pos": "0 0 0.05"}), ev),
        ("goal edit", replace(good, claim_id="bad-goal", controller=replace(spec, q_eq=0.0)), ev),
        ("gravity off", replace(good, claim_id="bad-gravity", patch={".//option@gravity": "0 0 0"}), ev),
        ("direct qpos write", replace(good, claim_id="bad-qpos", code={"physics/controllers.py": qpos_hack}), ev),
    ]


def known_good(good: Claim) -> list[Claim]:
    spec = good.controller
    return [
        good,
        replace(good, claim_id="good-heavier-r", controller=replace(spec, r_diag=(30.0,))),
        replace(good, claim_id="good-structure", controller=None),
    ]


def self_test(good: Claim, ev: Evidence, seed: int) -> dict:
    """Rows for BENCHMARK.md §4.6: per class whether it was caught and by which gate, plus false rejects."""
    rows = {}
    for cls, claim, evidence in mutants(good, ev):
        v = verify(claim, evidence, seed)
        rows[cls] = {"caught": not v.admitted, "first_gate": v.first_failure}
    false_rejects = [c.claim_id for c in known_good(good) if not verify(c, ev, seed).admitted]
    gates = Counter(r["first_gate"] for r in rows.values())
    return {"rows": rows, "caught": sum(r["caught"] for r in rows.values()), "n": len(rows),
            "first_gate_counts": dict(gates), "false_rejects": false_rejects}


def evidence_from_truth(base_xml: str, x_true: np.ndarray, rel_std: np.ndarray, seed: int) -> Evidence:
    """Synthetic S1 evidence (known physics + OU + measurement noise) for CI, before real footage exists."""
    from physics.identify_s1 import model_for
    from physics.rollout import rollout

    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.uniform(0.9, 1.1, 360) / 60.0)
    t -= t[0]
    clean = rollout(model_for(base_xml, x_true), np.array([0.5]), np.array([0.0]), t)[:, 0]
    kappa, sigma = np.exp(x_true[2:4])
    r = np.zeros(t.size)
    for k in range(1, t.size):
        a = np.exp(-kappa * (t[k] - t[k - 1]))
        r[k] = a * r[k - 1] + np.sqrt(sigma**2 / (2 * kappa) * (1 - a**2)) * rng.standard_normal()
    meas = 0.003
    theta = clean + r + meas * rng.standard_normal(t.size)
    n = int(0.7 * t.size)
    return Evidence(base_xml, x_true, np.diag(rel_std**2), t[:n], theta[:n], t[n:], theta[n:], (float(t[0]), 0.5, 0.0), meas**2,
                    goal=np.pi, success_tol=0.05)

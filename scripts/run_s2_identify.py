"""S2 identification on real takes (protocol: take 1 identifies; takes 2-3 are held out).

Run: uv run python -m scripts.run_s2_identify   ->  logs/runs/s2_identify.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from physics.fit_cmaes import fit, identifiability, laplace
from physics.identify_s2 import FREE_PHYSICAL, NAMES, bounds, classify_contact, make_objective, release_time, residuals
from verifier.s2 import verify_s2_structure

MODEL = Path("logs/runs/s2_model.xml")
RADIUS = 0.0325
SIGMA_S, SIGMA_PSI = 1e-3, 0.15  # m, rad: SAM centroid noise; stripe-angle scatter seen on take 1
FIT_FRACTION, EVALS, SEED = 0.7, 4000, 0
NUISANCE = [4, 5, 6]  # release time, s and psi offsets: per take, not physics
X0 = np.r_[np.log([0.5, 1e-4, 5.0, 0.01]), 0, 0, 0, np.log(1.0)]


def take(k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(f"logs/runs/s2_incline_take{k}_s2.npz")
    return d["t"], d["s"], d["psi"]


def nrmse(res: np.ndarray, obs: np.ndarray) -> float:
    return float(np.sqrt(np.mean(res**2)) / np.ptp(obs))


def main() -> None:
    base = MODEL.read_text()
    t, s, psi = take(1)
    structure, slip = classify_contact(t, s, psi, RADIUS)
    lower, upper = bounds(structure)
    x0 = np.clip(X0, lower, upper)
    n = int(FIT_FRACTION * t.size)
    rel0 = release_time(t, s)
    obj = make_objective(base, t[:n], s[:n], psi[:n], rel0, SIGMA_S**2, SIGMA_PSI**2, x0)
    x, _ = fit(obj, x0, 1.0, lower, upper, seed=SEED, max_evals=EVALS)
    post = laplace(obj, x, step=0.02)
    ok, why = identifiability(post, NAMES, FREE_PHYSICAL[structure])
    gates = verify_s2_structure(base, x, t, s, psi, n, rel0, SIGMA_S**2)
    out = {"identification_take": 1, "structure": structure, "relative_slip": slip, "identifiable": ok, "reason": why,
           "beta": float(np.exp(x[7])), "beta_rel_std": float(np.sqrt(post.cov[7, 7])),
           "mu": float(np.exp(x[0])) if structure == "incline_sliding" else None,
           "gates": {g: {"pass": r.passed, **r.witness, "feedback": r.feedback_to_agent} for g, r in gates.items()},
           "heldout": {}}

    for k in (2, 3):
        tk, sk, psik = take(k)
        lo, hi = x.copy() - 1e-6, x.copy() + 1e-6  # physics and noise model frozen from take 1
        lo[NUISANCE], hi[NUISANCE] = lower[NUISANCE], upper[NUISANCE]
        relk = release_time(tk, sk)
        objk = make_objective(base, tk, sk, psik, relk, SIGMA_S**2, SIGMA_PSI**2, x)
        xk, _ = fit(objk, np.clip(x, lo, hi), 0.5, lo, hi, seed=SEED, max_evals=1500)
        rs, rpsi = residuals(base, xk, tk, sk, psik, relk)
        out["heldout"][f"take{k}"] = {"nrmse_s": nrmse(rs, sk), "max_abs_s_mm": float(np.abs(rs).max() * 1000),
                                      "slip": classify_contact(tk, sk, psik, RADIUS)}
    Path("logs/runs/s2_identify.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()

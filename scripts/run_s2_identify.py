"""S2 identification on real takes (protocol: one take identifies, default take 1; the others are held out).

Run: uv run python -m scripts.run_s2_identify [--ident K] [--dir logs/runs/nov01_s2]   ->  <dir>/s2_identify.json
The identification take is the one with the longest clean straight run; the others are held out.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from physics.fit_cmaes import fit, identifiability, laplace
from physics.identify_s2 import FREE_PHYSICAL, NAMES, bounds, patch_for, classify_contact, make_objective, release_time, residuals
from verifier.s2 import verify_s2_structure

RUNS = Path("logs/runs")
RADIUS = 0.0325
SIGMA_S, SIGMA_PSI = 1e-3, 0.15  # m, rad: SAM centroid noise; stripe-angle scatter seen on take 1
FIT_FRACTION, EVALS, SEED = 0.7, 4000, 0
NUISANCE = [4, 5, 6]  # release time, s and psi offsets: per take, not physics
X0 = np.r_[np.log([0.5, 1e-4, 5.0, 0.01]), 0, 0, 0, np.log(1.0)]


def take(k: int, runs: Path = RUNS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(runs / f"s2_incline_take{k}_s2.npz")
    return d["t"], d["s"], d["psi"]


def nrmse(res: np.ndarray, obs: np.ndarray) -> float:
    return float(np.sqrt(np.mean(res**2)) / np.ptp(obs))


def identify(base: str, t: np.ndarray, s: np.ndarray, psi: np.ndarray) -> dict:
    """Structure from the slip test, CMA-ES fit on the first 70%, Laplace posterior, G7/G0/G1."""
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
    return {"x": x, "lower": lower, "upper": upper, "structure": structure, "slip": slip, "post": post,
            "identifiable": ok, "reason": why, "gates": gates}


def heldout_residuals(base: str, x: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                      t: np.ndarray, s: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """Distance residuals on a held-out take: physics and noise model frozen, only release and offsets fitted."""
    lo, hi = x.copy() - 1e-6, x.copy() + 1e-6
    lo[NUISANCE], hi[NUISANCE] = lower[NUISANCE], upper[NUISANCE]
    rel = release_time(t, s)
    obj = make_objective(base, t, s, psi, rel, SIGMA_S**2, SIGMA_PSI**2, x)
    xk, _ = fit(obj, np.clip(x, lo, hi), 0.5, lo, hi, seed=SEED, max_evals=1500)
    return residuals(base, xk, t, s, psi, rel)[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ident", type=int, default=1, choices=(1, 2, 3), help="identification take")
    ap.add_argument("--dir", type=Path, default=RUNS, help="folder with the takes and s2_model.xml")
    args = ap.parse_args()
    base = (args.dir / "s2_model.xml").read_text()
    r = identify(base, *take(args.ident, args.dir))
    x = r["x"]
    out = {"identification_take": args.ident, "structure": r["structure"], "relative_slip": r["slip"],
           "identifiable": r["identifiable"], "reason": r["reason"],
           "beta": float(np.exp(x[7])), "beta_rel_std": float(np.sqrt(r["post"].cov[7, 7])),
           "mu": float(np.exp(x[0])) if r["structure"] == "incline_sliding" else None,
           "gates": {g: {"pass": v.passed, **v.witness, "feedback": v.feedback_to_agent} for g, v in r["gates"].items()},
           "heldout": {}}
    for k in (k for k in (1, 2, 3) if k != args.ident):
        tk, sk, psik = take(k, args.dir)
        rs = heldout_residuals(base, x, r["lower"], r["upper"], tk, sk, psik)
        out["heldout"][f"take{k}"] = {"nrmse_s": nrmse(rs, sk), "max_abs_s_mm": float(np.abs(rs).max() * 1000),
                                      "slip": classify_contact(tk, sk, psik, RADIUS)}
    (args.dir / "s2_identify.json").write_text(json.dumps(out, indent=1, default=float))
    write_run_log(args.dir, args.ident, base, r, out)
    print(json.dumps(out, indent=1, default=float))


def write_run_log(runs: Path, ident: int, base: str, r: dict, out: dict) -> Path:
    """The verdict as a console run (same event format as the S1 agent logs, with no agent in the loop)."""
    tag = "s2" if runs == RUNS else runs.name
    path = RUNS / f"verify_{tag}_take{ident}_{time.strftime('%Y%m%dT%H%M%S')}.jsonl"
    failed = [g for g, v in out["gates"].items() if not v["pass"]]
    x = r["x"]
    events = [
        ("session", {"track": str(runs / f"s2_incline_take{ident}_s2.npz"), "model": str(runs / "s2_model.xml"), "scenario": "S2"}),
        ("fit", {"structure": r["structure"], "identifiable": r["identifiable"], "reason": r["reason"],
                 "params": {"beta": out["beta"], "slip": out["relative_slip"]}, "rel_std": {"beta": out["beta_rel_std"]}}),
        ("claim", {"ref": "s2-1", "type": "structure", "patch": patch_for(base, x)}),
        ("evidence", {"structure": r["structure"], "x": x.tolist(), "n_fit": int(FIT_FRACTION * take(ident, runs)[0].size)}),
        ("verdict", {"claim_id": "s2-1", "type": "structure", "proposer": "identification (no agent)",
                     "gates": {g: {k: v for k, v in w.items() if k != "feedback"} for g, w in out["gates"].items()},
                     "verdict": "REFUSED" if failed else "ADMITTED",
                     "reason": out["gates"][failed[0]]["feedback"] if failed else None}),
        ("final", {"text": "verifier finished"}),
    ]
    with path.open("w") as f:
        for kind, data in events:
            f.write(json.dumps({"t": time.time(), "kind": kind, "data": data}, default=float) + "\n")
    return path


if __name__ == "__main__":
    main()

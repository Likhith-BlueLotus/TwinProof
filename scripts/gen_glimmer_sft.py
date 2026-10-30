"""Glimmer critic SFT data (master doc §3.5): simulate a true structure, fit a deliberately wrong (or the right)
structure, render the residual, label the fix deterministically, split train/held-out by generator seed.

Run: uv run python -m scripts.gen_glimmer_sft --train 1500 --heldout 200   ->  data/glimmer_sft/{train,heldout}.jsonl, img/
Step 6 (Spark Open teacher rationales as reasoning_content) is a separate pass once a Spark Open key exists.
Fits here are fast 1-D/2-D least squares, not the production CMA-ES: the residual's shape is what the critic learns.
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import mujoco
import numpy as np
from scipy.optimize import minimize

from physics.identify_s1 import model_for as s1_model
from physics.identify_s2 import simulate as s2_simulate, xml_for as s2_xml
from physics.ou import init_ou, ou_innovations
from physics.rollout import rollout
from verifier.stats import ljung_box_p

OUT = Path("data/glimmer_sft")
S1 = Path("physics/models/s1_pendulum.xml").read_text()
S2 = Path("physics/models/s2_incline.xml").read_text()
HELDOUT_SEED0 = 1_000_000  # held-out seeds never overlap training seeds
IMG_W, IMG_H = 512, 256
SYSTEM = ('Reasoning: medium. You are a system-identification critic. '
          'Output JSON {"fix":str,"params":[str],"confidence":float}.')

# (true structure, fitted structure) -> (fix, params): the deterministic label map
CASES = {
    "add_coulomb_friction": ("pendulum_viscous_coulomb", "pendulum_viscous", ["frictionloss"]),
    "add_viscous_damping": ("pendulum_viscous_coulomb", "pendulum_coulomb", ["damping"]),
    "adjust_pivot": ("pendulum_hinge_offset", "pendulum_viscous_coulomb", ["joint_pos"]),
    "switch_to_sliding": ("incline_sliding", "incline_rolling", ["friction"]),
    "switch_to_rolling": ("incline_rolling", "incline_sliding_lowmu", ["friction"]),
    "no_change": ("pendulum_viscous_coulomb", "pendulum_viscous_coulomb", []),
}
FIXES = list(CASES)


def _ou(rng, t, kappa, sigma):
    r = np.zeros(t.size)
    for k in range(1, t.size):
        a = np.exp(-kappa * (t[k] - t[k - 1]))
        r[k] = a * r[k - 1] + np.sqrt(sigma**2 / (2 * kappa) * (1 - a**2)) * rng.standard_normal()
    return r


def _pendulum(xml, damping, fl, t, theta0):
    x = np.r_[np.log([max(damping, 1e-9), max(fl, 1e-9)]), 0, 0, 0, 0]
    return rollout(s1_model(xml, x), np.array([theta0]), np.array([0.0]), t)[:, 0]


def _with_length(length):
    return re.sub(r'fromto="0 0 0 0 0 -[0-9.]+"', f'fromto="0 0 0 0 0 -{length:.4f}"', S1)


def make_row(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    fix = FIXES[seed % len(FIXES)]
    true_s, fit_s, params = CASES[fix]
    if true_s.startswith("pendulum"):
        t = np.arange(0, 10, 1 / 15)
        damping, fl = 10 ** rng.uniform(-3, -2), 10 ** rng.uniform(-3.7, -2.7)
        length = 0.35 + (rng.choice([-1, 1]) * rng.uniform(0.04, 0.07) if true_s == "pendulum_hinge_offset" else 0.0)
        theta0 = rng.uniform(0.3, 0.9)
        obs = _pendulum(_with_length(length), damping, fl, t, theta0)
        obs += _ou(rng, t, rng.uniform(1, 10), rng.uniform(0.002, 0.01)) + rng.uniform(5e-4, 3e-3) * rng.standard_normal(t.size)
        free = {"pendulum_viscous": lambda p: (np.exp(p[0]), 0.0), "pendulum_coulomb": lambda p: (0.0, np.exp(p[0])),
                "pendulum_viscous_coulomb": lambda p: (np.exp(p[0]), np.exp(p[1]))}[fit_s]
        dim = 2 if fit_s == "pendulum_viscous_coulomb" else 1
        sse = lambda p: float(np.sum((obs - _pendulum(S1, *free(p), t, theta0)) ** 2))  # noqa: E731
        p = minimize(sse, np.full(dim, np.log(3e-3)), method="Nelder-Mead", options={"maxiter": 60, "xatol": 1e-3}).x
        fitted = _pendulum(S1, *free(p), t, theta0)
        extra = ""
    else:
        t = np.arange(0, 1.1, 1 / 60)
        mu_true = rng.uniform(0.02, 0.07) if true_s == "incline_sliding" else rng.uniform(0.3, 1.0)
        model = mujoco.MjModel.from_xml_string(s2_xml(S2, np.r_[np.log([mu_true, 1e-4]), 0, 0, 0, 0, 0]))
        s_true, psi_true = s2_simulate(model, t, 0.2)
        obs = s_true + _ou(rng, t, rng.uniform(5, 30), rng.uniform(5e-4, 3e-3)) + 8e-4 * rng.standard_normal(t.size)
        psi = psi_true + 0.02 * rng.standard_normal(t.size)
        mu_fit = 1.0 if fit_s == "incline_rolling" else rng.uniform(0.02, 0.06)  # wrong structure pins mu
        fm = mujoco.MjModel.from_xml_string(s2_xml(S2, np.r_[np.log([mu_fit, 1e-4]), 0, 0, 0, 0, 0]))
        fitted, psi_fit = s2_simulate(fm, t, 0.2)
        moving = np.abs(np.gradient(s_true, t)) > 0.05
        slip = float(np.median(np.abs(np.gradient(obs, t) - np.gradient(psi, t) * 0.037)[moving] /
                               np.maximum(np.abs(np.gradient(obs, t))[moving], 0.01)))
        extra = f"; slip_ratio={slip:.3f}"
    r = obs - fitted
    kappa, sigma = init_ou(r, t)
    nu, var = ou_innovations(r, t, kappa, sigma, 1e-6)
    stats = (f"structure={fit_s.replace('_lowmu', '')}; nrmse={np.sqrt(np.mean(r**2)) / np.ptp(obs):.4f}; "
             f"ljung_box_p={ljung_box_p((nu / np.sqrt(var))[:, 0]):.3g}; ou_kappa={kappa[0]:.3g}; ou_sigma={sigma[0]:.3g}{extra}")
    img = f"img/{seed:07d}.png"
    cv2.imwrite(str(OUT / img), render(t, r))
    answer = {"fix": fix, "params": params, "confidence": 1.0}
    return {"seed": seed, "label": fix, "messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": [{"type": "image", "image": str(OUT / img)}, {"type": "text", "text": stats}]},
        {"role": "assistant", "content": json.dumps(answer)}]}


def render(t: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Fixed style: left r(t), right phase portrait (r, r_dot); dark background, same scale rules every row."""
    img = np.full((IMG_H, IMG_W, 3), 18, np.uint8)
    rd = np.gradient(r, t)
    pad = 16
    for x0, xs, ys in ((0, (t - t[0]) / np.ptp(t), r), (IMG_W // 2, None, r)):
        w = IMG_W // 2 - 2 * pad
        if xs is None:
            xs = (r - r.min()) / (np.ptp(r) + 1e-12)
            ys = rd
        yn = (ys - ys.min()) / (np.ptp(ys) + 1e-12)
        pts = np.c_[x0 + pad + xs * w, IMG_H - pad - yn * (IMG_H - 2 * pad)].astype(np.int32)
        cv2.rectangle(img, (x0 + pad, pad), (x0 + pad + w, IMG_H - pad), (60, 60, 60), 1)
        cv2.polylines(img, [pts], False, (232, 214, 79), 1, cv2.LINE_AA)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=1500)
    ap.add_argument("--heldout", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    for name, seeds in (("train", range(args.train)), ("heldout", range(HELDOUT_SEED0, HELDOUT_SEED0 + args.heldout))):
        with ProcessPoolExecutor(args.workers) as pool, (OUT / f"{name}.jsonl").open("w") as f:
            for row in pool.map(make_row, seeds, chunksize=8):
                f.write(json.dumps(row) + "\n")
        print(name, len(seeds), "rows ->", OUT / f"{name}.jsonl")


if __name__ == "__main__":
    main()

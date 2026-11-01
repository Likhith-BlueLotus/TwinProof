"""Three-arm benchmark (master doc §4): A0 default physics, A1 unverified agent, A2 TwinProof.

Run: uv run python -m bench.run_all --write-benchmark   ->  bench/results.json, BENCHMARK.md result tables
Reality for control is a twin re-identified from a held-out take (real S1: take 1 of each rig), or the known physics
for the synthetic take. S2 contributes held-out prediction units only (A0 vs A2; no S2 control task, no S2 agent). Every controller is designed on its own arm's model and run on 256 reality samples.
"""
from __future__ import annotations

import argparse
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from agents.dispatch import Context, fit_batch
from agents.spark_open import STRUCTURES
from perception.smoother import rts_smooth
from physics.controllers import LQRSpec, closed_loop_step, linearize, lqr_gain, policy
from physics.identify_s1 import N_CAL, predict, start_state, xml_for
from verifier.stats import clopper_pearson_lower, clopper_pearson_upper

RUNS = Path("logs/runs")
N_REALITY, HORIZON_STEPS = 256, 250  # 5 s at the 20 ms control period
START_ANGLE, START_RATE = 0.2, 0.5  # rad, rad/s: disturbances around upright
SUCCESS_TOL, FALL_ANGLE = 0.05, 1.0  # rad
DEFAULT_SPEC = dict(q_diag=(10.0, 0.1), r_diag=(10.0,))
GOAL = float(np.pi)
S2_A0_X = np.r_[np.log([0.6, 2e-4, 5.0, 0.01]), 0, 0, 0, np.log(0.5)]  # base model: solid can, its default friction
S2_SCENARIOS = {
    "S2 real · 12° ramp, 31 Oct": dict(dir="logs/runs", ident=1, heldout=[2, 3]),
    "S2 real · 8° board, 1 Nov": dict(dir="logs/runs/nov01_s2", ident=2, heldout=[1, 3]),
}
SCENARIOS = {
    "S1 real · 31 Oct": dict(model="logs/runs/take3_model.xml", ident="take3_s1", heldout=["take1_s1", "take2_s1"],
                             reality="take1_s1", agent="agent_take3_s1_20261031T194821"),
    "S1 real · finger-held pin": dict(model="logs/runs/nov01/take3_model.xml", ident="nov01/take3_s1",
                                      heldout=["nov01/take1_s1", "nov01/take2_s1"], reality="nov01/take1_s1",
                                      agent="agent_take3_s1_20261101T161025"),
    "S1 real · clamped pin": dict(model="logs/runs/nov01_clamped/take3_model.xml", ident="nov01_clamped/take3_s1",
                                  heldout=["nov01_clamped/take1_s1", "nov01_clamped/take2_s1"], reality="nov01_clamped/take1_s1",
                                  agent="agent_take3_s1_20261101T160606"),
    "S1 synthetic": dict(model="physics/models/s1_pendulum.xml", ident="s1_s1", heldout=[], reality=None,
                         agent="agent_s1_s1_20261030T215843"),
}


def _track(stem: str) -> tuple[np.ndarray, np.ndarray, float]:
    d = np.load(RUNS / f"{stem}.npz")
    ok = np.isfinite(d["phi"])
    return d["t"][ok], d["phi"][ok], float(d["r_meas"])


def fit_all(model_xml: str, stem: str) -> Context:
    t, phi, r = _track(stem)
    ctx = Context(model_xml, t, phi, r, emit=lambda *_: None)
    for s in STRUCTURES:
        fit_batch({"structure": s}, ctx)
    return ctx


def default_x(model_xml: str) -> np.ndarray:
    """A0: the base model's nominal damping and friction; OU and calibration at their neutral values."""
    joint = ET.fromstring(model_xml).find(".//joint[@name='hinge']")
    return np.r_[np.log([float(joint.get("damping")), max(float(joint.get("frictionloss")), 1e-9), 1.0, 0.01]), 0, 0, np.zeros(N_CAL)]


def heldout_nrmse(model_xml: str, x: np.ndarray, stem: str) -> float:
    """Open-loop prediction of a held-out take: only its start state comes from that take."""
    t, phi, r = _track(stem)
    sm = rts_smooth(t, phi, sigma=np.sqrt(r), q_tilde=50.0).states
    xk = np.r_[x[:4], 0.0, 0.0, x[6:]]
    return float(np.sqrt(np.mean((phi - predict(model_xml, xk, t, start_state(xk, (sm[0, 0], sm[0, 1])))) ** 2)) / np.ptp(phi))


def window_nrmse(model_xml: str, x: np.ndarray, ctx: Context) -> float:
    """Single-take scenario: continuation of the fitted trajectory into the last 30% (G1's window)."""
    ev = ctx.evidence
    t_all, y_all = np.r_[ev.t_fit, ev.t_heldout], np.r_[ev.theta_fit, ev.theta_heldout]
    pred = predict(model_xml, x, t_all, ev.start[1:])[ev.t_fit.size:]
    return float(np.sqrt(np.mean((ev.theta_heldout - pred) ** 2)) / np.ptp(ev.theta_heldout))


def reality_models(model_xml: str, x: np.ndarray, cov: np.ndarray | None, rng) -> list[mujoco.MjModel]:
    xs = [x] * N_REALITY if cov is None else rng.multivariate_normal(x, cov, size=N_REALITY)
    return [mujoco.MjModel.from_xml_string(xml_for(model_xml, xi)) for xi in xs]


def evaluate(spec: LQRSpec, design: mujoco.MjModel, reality: list[mujoco.MjModel], rng) -> dict:
    """Closed loop on every reality sample from a random start near upright; the actuator clamps to its range."""
    A, B = linearize(design, spec.q_eq)
    K = lqr_gain(A, B, spec)
    X = np.c_[rng.uniform(-START_ANGLE, START_ANGLE, N_REALITY), rng.uniform(-START_RATE, START_RATE, N_REALITY)]
    peak, fell = np.zeros(N_REALITY), np.zeros(N_REALITY, bool)
    for _ in range(HORIZON_STEPS):
        peak = np.maximum(peak, np.abs(policy(X, K, spec)[:, 0]))
        X = closed_loop_step(reality, X, K, spec)
        fell |= np.abs(X[:, 0]) > FALL_ANGLE
    ok = ~fell & (np.abs(X[:, 0]) <= SUCCESS_TOL)
    fails = int((~ok).sum())
    limit = float(reality[0].actuator_ctrlrange[0, 1])
    return {"success": float(ok.mean()), "fails": fails, "p_fail_upper": clopper_pearson_upper(fails, N_REALITY),
            "max_u_over_limit": float(peak.max() / limit), "u_limit": limit, "K": K.ravel().tolist()}


def agent_log(stem: str) -> list[dict]:
    return [json.loads(line) for line in (RUNS / f"{stem}.jsonl").read_text().splitlines() if line.strip()]


def _spec(claim: dict) -> LQRSpec:
    s = claim["spec"]
    return LQRSpec(GOAL, tuple(s["q_diag"]), tuple(s["r_diag"]), s["u_max"], tuple(s["gains"]) if s["gains"] else None)


def refusal_reason(verdicts: list[dict]) -> str | None:
    """Why no controller shipped: the last refused controller claim (a later structure patch's rejection is not it)."""
    refused = [v for v in verdicts if v["verdict"] != "ADMITTED"]
    ctrl = [v for v in refused if v["type"] == "controller"]
    return (ctrl or refused or [{"reason": None}])[-1]["reason"]


def run_scenario(name: str, cfg: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    model_xml = Path(cfg["model"]).read_text()
    ctx = fit_all(model_xml, cfg["ident"])
    events = agent_log(cfg["agent"])
    claims = [e["data"] for e in events if e["kind"] == "claim"]
    verdicts = [e["data"] for e in events if e["kind"] == "verdict"]
    admitted_ctrl = [v["claim_id"] for v in verdicts if v["verdict"] == "ADMITTED" and v["type"] == "controller"]
    wall = events[-1]["t"] - events[0]["t"]

    if cfg["reality"]:
        rctx = fit_all(model_xml, cfg["reality"])
        reality = reality_models(model_xml, rctx.evidence.x_hat, rctx.evidence.cov, rng)
        reality_note = f"reality = {cfg['reality']} re-identified ({rctx.twin_structure})"
    else:
        from scripts.make_synthetic_s1 import X_TRUE
        reality = reality_models(model_xml, X_TRUE, None, rng)
        reality_note = "reality = known synthetic physics"

    a1_struct = min(ctx.fits, key=lambda s: ctx.fits[s]["bic"])  # best BIC, identifiable or not: the agent's own pick
    arms = {
        "A0": dict(x=default_x(model_xml), spec=LQRSpec(GOAL, **DEFAULT_SPEC, u_max=float(reality[0].actuator_ctrlrange[0, 1])),
                   structure="nominal (no identification)", proposed=1, admitted=1),
        "A1": dict(x=ctx.fits[a1_struct]["evidence"].x_hat, spec=_spec(next(c for c in claims if c["type"] == "controller")),
                   structure=a1_struct, proposed=len(claims), admitted=len(claims)),
        "A2": dict(x=ctx.evidence.x_hat, structure=ctx.twin_structure, proposed=len(claims), admitted=len(admitted_ctrl),
                   spec=_spec(next(c for c in claims if c["ref"] == admitted_ctrl[-1])) if admitted_ctrl else None,
                   refusal=refusal_reason(verdicts)),
    }
    rows = {}
    for arm, a in arms.items():
        nrmse = {s: heldout_nrmse(model_xml, a["x"], s) for s in cfg["heldout"]} or {"last 30%": window_nrmse(model_xml, a["x"], ctx)}
        row = {"structure": a["structure"], "heldout_nrmse": nrmse, "claims": f"{a['proposed']} / {a['admitted']}",
               "wall_clock_s": None if arm == "A0" else wall}
        if a["spec"] is None:
            row["control"] = None
            row["refusal"] = a["refusal"]
        else:
            design = mujoco.MjModel.from_xml_string(xml_for(model_xml, a["x"]))
            row["control"] = evaluate(a["spec"], design, reality, rng)
        rows[arm] = row
    return {"reality": reality_note, "arms": rows}


def run_s2_scenario(cfg: dict) -> dict:
    """S2: held-out distance NRMSE only. The agent loop is S1-only (no A1) and no S2 control task was built."""
    from scripts.run_s2_identify import heldout_residuals, identify, nrmse, take
    runs = Path(cfg["dir"])
    base = (runs / "s2_model.xml").read_text()
    r = identify(base, *take(cfg["ident"], runs))
    failed = [v.feedback_to_agent for v in r["gates"].values() if not v.passed]
    if not r["identifiable"]:
        failed.insert(0, f"not identifiable ({r['reason']})")
    note = "no S2 control task built"
    rows = {}
    for arm, x, structure in (("A0", S2_A0_X, "nominal solid can (no identification)"), ("A2", r["x"], r["structure"])):
        units = {f"take{k}": nrmse(heldout_residuals(base, x, r["lower"], r["upper"], *take(k, runs)), take(k, runs)[1])
                 for k in cfg["heldout"]}
        rows[arm] = {"structure": structure, "heldout_nrmse": units, "claims": "—", "wall_clock_s": None,
                     "control": None, "control_note": note, "refusal": "; ".join(failed) if arm == "A2" and failed else None}
    return {"reality": f"takes {cfg['heldout']} held out; identification take {cfg['ident']}; A1 not run (agent loop is S1-only)",
            "arms": rows}


def seeded_bad() -> dict:
    from physics.controllers import LQRSpec as Spec
    from verifier.seeded_bad import evidence_from_truth, self_test
    from verifier.verify import Claim
    base = Path("physics/models/s1_pendulum.xml").read_text()
    ev = evidence_from_truth(base, np.r_[np.log([0.004, 0.0008, 2.0, 0.01]), 0, 0], np.r_[0.05, 0.15, 0.1, 0.1, 1e-6, 1e-6], seed=3)
    report = self_test(Claim("c-good", "bench", controller=Spec(GOAL, (10, 0.1), (10,), 2.5)), ev, seed=0)
    report["catch_lower_95"] = clopper_pearson_lower(report["caught"], report["n"])
    return report


def h1(results: dict) -> dict:
    """One-sided Wilcoxon on paired (scenario, held-out unit) NRMSE: A2 vs A0 and A2 vs A1, Holm over 2."""
    from scipy.stats import wilcoxon
    pairs = {"A0": [], "A1": []}
    for sc in results["scenarios"].values():
        a2 = sc["arms"]["A2"]["heldout_nrmse"]
        for other in (o for o in pairs if o in sc["arms"]):
            pairs[other] += [(a2[u], sc["arms"][other]["heldout_nrmse"][u]) for u in a2]
    out = {}
    for other, p in pairs.items():
        diff = np.array([b - a for a, b in p])  # positive: A2 better
        ties = int((diff == 0).sum())
        n_eff = len(p) - ties  # Wilcoxon drops exact ties, so only these pairs can move p
        out[other] = {"pairs": len(p), "a2_better": int((diff > 0).sum()), "ties": ties,
                      "p_one_sided": float(wilcoxon(diff, alternative="greater").pvalue) if n_eff else None,
                      "min_attainable_p": 0.5 ** n_eff if n_eff else None}
    return out


def fmt_pct(v):
    return f"{100 * v:.1f}%"


def markdown(results: dict) -> tuple[str, str, str]:
    lines = ["| Scenario | Arm | Twin structure | Held-out NRMSE | Task success (N=256) | p_fail 95% UB | max commanded torque / limit | Claims proposed / admitted | Wall-clock (s) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, sc in results["scenarios"].items():
        for arm, r in sc["arms"].items():
            nrmse = ", ".join(f"{u}: {fmt_pct(v)}" for u, v in r["heldout_nrmse"].items())
            c = r["control"]
            ctrl = (r.get("control_note") or "**no controller shipped** (refused)", "—", "—") if c is None else (
                fmt_pct(c["success"]), fmt_pct(c["p_fail_upper"]), f"{c['max_u_over_limit']:.2f}")
            wall = "—" if r["wall_clock_s"] is None else f"{r['wall_clock_s']:.0f}"
            lines.append(f"| {name} | {arm} | {r['structure']} | {nrmse} | {ctrl[0]} | {ctrl[1]} | {ctrl[2]} | {r['claims']} | {wall} |")
    controlled = [r["control"]["success"] for sc in results["scenarios"].values() for r in sc["arms"].values() if r["control"]]
    if controlled and max(controlled) - min(controlled) < 0.01:
        lines.append("")
        lines.append("**At this disturbance level every shipped controller holds every reality sample, so the control task does not "
                     "separate the arms.** The separation shows in held-out prediction, and in A2 refusing to ship where the twin is "
                     "inadequate. The task and the motor sizing were fixed before these runs and are not tuned to produce a difference.")
    notes = [f"- **{name}**: {sc['reality']}." + (f" A2 refusal: {sc['arms']['A2']['refusal']}" if sc['arms']['A2'].get('refusal') else "")
             for name, sc in results["scenarios"].items()]
    main = "\n".join(lines) + "\n\n" + "\n".join(notes) + (
        "\n- Task: hold upright for 5 s from |Δθ| ≤ 0.2 rad, |ω| ≤ 0.5 rad/s; success = |Δθ| ≤ 0.05 rad at 5 s. "
        "The actuator clamps to the reality twin's limit; a commanded/limit ratio > 1 means the controller asked for more than the motor has."
        "\n- Mask IoU, min clearance and $ are not measured (no rendered-twin IoU; S1 has no contacts; per-call costs not logged).")
    sb = results["seeded_bad"]
    sb_lines = ["| Mutation class | n injected | Caught | First catching gate | False rejects on known-good (n) |", "|---|---|---|---|---|"]
    for cls, r in sb["rows"].items():
        sb_lines.append(f"| {cls} | 1 | {'yes' if r['caught'] else '**no**'} | {r['first_gate']} | — |")
    sb_lines.append(f"| **Total** | {sb['n']} | {sb['caught']} | " + ", ".join(f"{g}: {n}" for g, n in sorted(sb["first_gate_counts"].items())) +
                    f" | {len(sb['false_rejects'])} of 3 |")
    seeded = "\n".join(sb_lines) + f"\n\nCatch rate {sb['caught']}/{sb['n']}; one-sided 95% Clopper–Pearson lower bound {sb['catch_lower_95']:.3f}. H3 {'**passes**' if sb['caught'] == sb['n'] and not sb['false_rejects'] else '**fails**'}."
    h = results["h1"]
    h1_text = "\n".join(f"- A2 vs {o}: A2 better on {v['a2_better']}/{v['pairs']} paired units"
                        + (f" ({v['ties']} exact ties: both arms chose the same twin)" if v.get("ties") else "")
                        + (f"; one-sided Wilcoxon p = {v['p_one_sided']:.3f} (smallest attainable with {v['pairs'] - v.get('ties', 0)} untied pairs: {v['min_attainable_p']:.2g})."
                           if v["p_one_sided"] is not None else "; no untied pairs, so no test.")
                        for o, v in h.items())
    holm = sorted((v["p_one_sided"], o) for o, v in h.items() if v["p_one_sided"] is not None)
    rejected = []
    for i, (p, o) in enumerate(holm):  # Holm step-down over the comparisons
        if p > 0.05 / (len(holm) - i):
            break
        rejected.append(o)
    h1_text += f"\n- Holm over {len(holm)} comparisons at α = 0.05: " + (
        f"**H1 supported vs {', '.join(rejected)}**" if rejected else "**H1 not supported**") + (
        "" if not rejected or set(rejected) == {o for _, o in holm} else f"; not supported vs {', '.join(o for _, o in holm if o not in rejected)}") + "."
    return main, seeded, h1_text


def write_benchmark(main: str, seeded: str, h1_text: str, stamp: str) -> None:
    p = Path("BENCHMARK.md")
    s = p.read_text()

    def replace_table(s: str, header: str, body: str) -> str:
        m = re.search(rf"(### {re.escape(header)}\n)(.*?)(\n### |\Z)", s, re.S)
        if not m:
            raise ValueError(f"section {header!r} not found in BENCHMARK.md")
        return s[:m.start(2)] + body + "\n" + s[m.end(2):]

    s = replace_table(s, "Main metrics", f"_Filled by `bench/run_all.py` on {stamp} from `bench/results.json`._\n\n{main}\n\n**H1 (held-out NRMSE, A2 < A0 and A1):**\n{h1_text}\n")
    s = replace_table(s, "Verifier seeded-bad self-test (H3)", f"_Filled by `bench/run_all.py` on {stamp}._\n\n{seeded}\n")
    p.write_text(s)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--write-benchmark", action="store_true")
    ap.add_argument("--render-only", action="store_true", help="rebuild the tables from bench/results.json without rerunning")
    args = ap.parse_args()
    out = Path("bench/results.json")
    if args.render_only:  # tables, H1 and refusal reasons are re-derived from the stored numbers and the logs
        results = json.loads(out.read_text())
        for name, cfg in SCENARIOS.items():
            if name in results["scenarios"] and results["scenarios"][name]["arms"]["A2"]["control"] is None:
                results["scenarios"][name]["arms"]["A2"]["refusal"] = refusal_reason(
                    [e["data"] for e in agent_log(cfg["agent"]) if e["kind"] == "verdict"])
        for sc in results["scenarios"].values():
            if sc["arms"]["A2"].get("refusal"):
                sc["arms"]["A2"]["refusal"] = re.sub(r"^(G\d): \1: ", r"\1: ", sc["arms"]["A2"]["refusal"])
        results["h1"] = h1(results)
        out.write_text(json.dumps(results, indent=1, default=float))
    else:
        t0 = time.time()
        results = {"scenarios": {n: run_scenario(n, c, args.seed) for n, c in SCENARIOS.items()}}
        results["scenarios"].update({n: run_s2_scenario(c) for n, c in S2_SCENARIOS.items()})
        results["seeded_bad"] = seeded_bad()
        results["h1"] = h1(results)
        results["seconds"] = time.time() - t0
        out.write_text(json.dumps(results, indent=1, default=float))
    main_md, seeded_md, h1_md = markdown(results)
    print(main_md, "\n\n", seeded_md, "\n\n", h1_md)
    if args.write_benchmark:
        write_benchmark(main_md, seeded_md, h1_md, time.strftime("%Y-%m-%d %H:%M"))


if __name__ == "__main__":
    main()

"""Scripted agent turns drive the real handlers and verifier (no network)."""
import json
from pathlib import Path

import numpy as np
import pytest

from agents.dispatch import Context
from agents.loop import MAX_REJECT, agent_loop
from verifier.seeded_bad import evidence_from_truth

S1 = (Path(__file__).parents[1] / "physics/models/s1_pendulum.xml").read_text()
X_TRUE = np.r_[np.log([0.004, 0.0008, 2.0, 0.01]), 0, 0]
REL_STD = np.r_[0.05, 0.15, 0.1, 0.1, 1e-6, 1e-6]


def _ctx(events):
    ev = evidence_from_truth(S1, X_TRUE, REL_STD, seed=3)
    t, phi = np.r_[ev.t_fit, ev.t_heldout], np.r_[ev.theta_fit, ev.theta_heldout]
    return Context(S1, t, phi, ev.r_meas, emit=lambda kind, data: events.append((kind, data)))


def _script(*turns):
    """Each turn: list of (name, args). Ids are unique per call; the fake model also records what it saw."""
    seen, it, counter = [], iter(turns), iter(range(1000))

    def turn(msgs):
        seen.append([m["role"] for m in msgs])
        calls = next(it, [])
        return [{"id": f"call_{next(counter)}", "name": n, "args": json.dumps(a)} for n, a in calls], "ok"
    return turn, seen


def test_agent_gets_rejected_then_admitted_then_what_if():
    events = []
    ctx = _ctx(events)
    turn, seen = _script(
        [("fit_batch", {"structure": "pendulum_viscous_coulomb"})],
        [("synthesize_controller", {"kind": "lqr", "u_max": 2.5, "gains": {"k_theta": 0.2, "k_omega": 0.05}})],
        [("submit_claim", {"ref": "c-1"})],
        [("synthesize_controller", {"kind": "lqr", "u_max": 2.5})],
        [("submit_claim", {"ref": "c-2"})],
        [("reverify", {"torque_limit": 0.75})],
    )
    assert agent_loop(ctx, turn) == "ok"
    verdicts = [(r["claim_id"], r["verdict"]) for r in ctx.records]
    assert verdicts == [("c-1", "REJECTED"), ("c-2", "ADMITTED"), ("c-2-whatif", "REJECTED")]
    assert ctx.records[0]["reason"].startswith("G2")
    assert ctx.records[2]["reason"].startswith("G3")
    # every assistant tool call is answered by one tool message before the next turn
    assert seen[1][-2:] == ["assistant", "tool"]
    kinds = {k for k, _ in events}
    assert {"fit", "claim", "verdict", "tool_call"} <= kinds


def test_six_rejections_stop_the_agent():
    events = []
    ctx = _ctx(events)
    ctx.evidence = evidence_from_truth(S1, X_TRUE, REL_STD, seed=3)
    turn, _ = _script(
        [("synthesize_controller", {"kind": "lqr", "u_max": 2.5, "gains": {"k_theta": 0.2, "k_omega": 0.05}})],
        *[[("submit_claim", {"ref": "c-1"})]] * MAX_REJECT,
        [("submit_claim", {"ref": "c-1"})],  # never reached
    )
    assert agent_loop(ctx, turn) is None
    assert len(ctx.records) == MAX_REJECT
    assert events[-1] == ("commentary", "Six consecutive rejections: please film a new experiment.")


def test_bad_tool_input_comes_back_as_feedback():
    events = []
    ctx = _ctx(events)
    turn, _ = _script([("submit_claim", {"ref": "nope"}), ("no_such_tool", {}), ("synthesize_controller", {"kind": "pd_impedance", "u_max": 1})])
    agent_loop(ctx, turn)
    assert ctx.records == []


def test_twin_is_best_bic_identifiable_fit_whatever_order_fits_finish():
    from agents.dispatch import _select_twin
    events = []
    ctx = _ctx(events)
    good, bad = object(), object()
    ctx.fits["pendulum_coulomb"] = {"evidence": good, "ok": True, "why": "identifiable", "bic": -1438.0, "payload": {"s": "c"}}
    _select_twin(ctx)
    ctx.fits["pendulum_viscous_coulomb"] = {"evidence": bad, "ok": False, "why": "flat", "bic": -1471.0, "payload": {"s": "vc"}}
    _select_twin(ctx)  # better BIC but unidentifiable, and it finished last: must not win
    assert ctx.evidence is good and ctx.refusal is None and ctx.twin_structure == "pendulum_coulomb"
    assert [d for k, d in events if k == "evidence"] == [{"s": "c"}]


def test_refusal_when_no_fit_is_identifiable_is_logged_as_a_verdict():
    from agents.dispatch import _select_twin
    events = []
    ctx = _ctx(events)
    ctx.fits["pendulum_viscous_coulomb"] = {"evidence": object(), "ok": False, "why": "flat", "bic": -1.0, "payload": {}}
    _select_twin(ctx)
    turn, _ = _script([("synthesize_controller", {"kind": "lqr", "u_max": 2.5})], [("submit_claim", {"ref": "c-1"})])
    agent_loop(ctx, turn)
    assert ctx.records[-1]["verdict"] == "REFUSED" and "not identifiable" in ctx.records[-1]["reason"]
    assert any(k == "verdict" and d["verdict"] == "REFUSED" for k, d in events)

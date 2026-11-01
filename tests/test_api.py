"""API against the recorded synthetic S1 run (no network)."""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import RUNS, app

client = TestClient(app)


def _run_with_evidence() -> str | None:
    for p in sorted(RUNS.glob("agent_*.jsonl"), reverse=True):
        lines = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        if any(e["kind"] == "evidence" for e in lines) and any(
                e["kind"] == "verdict" and e["data"]["verdict"] == "ADMITTED" for e in lines):
            return p.stem
    return None


RUN = _run_with_evidence()
needs_run = pytest.mark.skipif(RUN is None, reason="no recorded agent run with evidence")


@needs_run
def test_list_and_detail():
    assert RUN in [r["id"] for r in client.get("/api/runs").json()]
    d = client.get(f"/api/runs/{RUN}").json()
    assert d["summary"]["admitted"] >= 1 and d["overlay"]["tip_px"]


@needs_run
def test_ghost_overlay_lands_on_the_real_bar_tip():
    from physics.identify_s1 import model_for
    from physics.rollout import rollout
    from scripts.make_synthetic_s1 import FPS, LENGTH_MM, MM, PIVOT_MM, PLANE_TO_IMG, THETA0, X_TRUE
    ov = client.get(f"/api/runs/{RUN}").json()["overlay"]
    tips = np.asarray(ov["tip_px"])
    t = ov["t0"] + np.arange(len(tips)) / FPS
    base = (Path(__file__).parents[1] / "physics/models/s1_pendulum.xml").read_text()
    theta = rollout(model_for(base, X_TRUE), np.array([THETA0]), np.array([0.0]), np.r_[0.0, t])[1:, 0]
    true_tip_mm = np.asarray(PIVOT_MM) + LENGTH_MM * np.c_[np.sin(theta), np.cos(theta)]
    true_px = cv2.perspectiveTransform((true_tip_mm * MM).reshape(-1, 1, 2), PLANE_TO_IMG).reshape(-1, 2)
    err = np.linalg.norm(tips - true_px, axis=1)
    assert np.median(err) < 8 and np.percentile(err, 95) < 15, (np.median(err), np.percentile(err, 95))


@pytest.mark.parametrize("bad", ["../../etc/passwd", "agent_x", "agent_s1;rm_20261030T000000", "agent_s1_s1_20261030T999999"])
def test_bad_run_ids_are_refused(bad):
    assert client.get(f"/api/runs/{bad}").status_code in (400, 404)


@needs_run
def test_what_if_bounds_are_enforced():
    assert client.post(f"/api/runs/{RUN}/whatif", json={"mass_scale": 9}).status_code == 422
    assert client.post(f"/api/runs/{RUN}/whatif", json={"torque_limit": -1}).status_code == 422


@needs_run
def test_what_if_weak_motor_is_rejected_at_g3():
    rec = client.post(f"/api/runs/{RUN}/whatif", json={"torque_limit": 0.5}).json()
    assert rec["verdict"] == "REJECTED" and rec["reason"].startswith("G3") and rec["what_if"] == {"torque_limit": 0.5}


def test_benchmark_serves_arms_and_s2_replication():
    d = client.get("/api/benchmark").json()
    assert {"A0", "A2"} <= set(next(iter(d["results"]["scenarios"].values()))["arms"])
    assert all(0.3 <= r["beta"] <= 1.5 for r in d["s2_beta"].values())


def test_track_outside_runs_folder_is_refused():
    from fastapi import HTTPException
    from api.main import _track
    with pytest.raises(HTTPException) as e:
        _track("../../pyproject")
    assert e.value.status_code == 400

"""TwinProof API: recorded and live agent runs, ghost-overlay geometry, operator what-ifs.

Run: uv run --env-file .env uvicorn api.main:app --port 8000
"""
from __future__ import annotations

import json
import re
import threading
import time
from functools import partial
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from agents.dispatch import Context, reverify
from agents.loop import agent_loop
from agents.spark_open import resolve, run_turn
from physics.controllers import LQRSpec
from physics.identify_s1 import observe, predict
from verifier.verify import Claim, Evidence

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "logs" / "runs"
VIDEOS = ROOT / "data" / "videos"
MODEL = ROOT / "physics" / "models" / "s1_pendulum.xml"
RUN_ID = re.compile(r"^agent_[A-Za-z0-9_]+_\d{8}T\d{6}$")
GOAL = float(np.pi)

app = FastAPI(title="TwinProof")


class WhatIf(BaseModel):
    """Operator what-if; bounds mirror the agent's reverify tool schema."""

    mass_scale: float | None = Field(None, ge=0.5, le=2.0)
    damping_scale: float | None = Field(None, ge=0.0, le=3.0)
    torque_limit: float | None = Field(None, gt=0.0, le=10.0)


def _log_path(run_id: str) -> Path:
    if not RUN_ID.match(run_id):
        raise HTTPException(400, "malformed run id")
    path = RUNS / f"{run_id}.jsonl"
    if not path.exists():
        raise HTTPException(404, "no such run")
    return path


def _events(run_id: str) -> list[dict]:
    return [json.loads(line) for line in _log_path(run_id).read_text().splitlines() if line.strip()]


def _track_stem(run_id: str, events: list[dict]) -> str:
    session = next((e["data"] for e in events if e["kind"] == "session"), None)
    if session:  # keep the sub-folder: archived rigs reuse the same take names
        track = (ROOT / session["track"]).resolve()
        return str(track.relative_to(RUNS.resolve()).with_suffix("")) if RUNS.resolve() in track.parents else track.stem
    return run_id[len("agent_"):-len("_20261030T000000")]  # logs written before the session event


def _track(stem: str) -> tuple[dict, dict]:
    meta_path, npz_path = RUNS / f"{stem}.json", RUNS / f"{stem}.npz"
    if RUNS.resolve() not in meta_path.resolve().parents:
        raise HTTPException(400, "track outside the runs folder")
    if not meta_path.exists() or not npz_path.exists():
        raise HTTPException(404, f"track {stem} not found")
    data = np.load(npz_path)
    return json.loads(meta_path.read_text()), {k: data[k] for k in data.files}


def _model_path(meta: dict) -> Path:
    """The run's own twin (measured geometry) when the track has one, else the base S1 model."""
    if not meta.get("model"):
        return MODEL
    path = (ROOT / meta["model"]).resolve()
    allowed = (RUNS.resolve(), MODEL.parent.resolve())
    if not any(a in path.parents for a in allowed) or path.suffix != ".xml" or not path.exists():
        raise HTTPException(500, "track metadata points at an invalid model")
    return path


def _summary(run_id: str, events: list[dict]) -> dict:
    provider = next((e["data"] for e in events if e["kind"] == "provider"), {})
    verdicts = [e["data"] for e in events if e["kind"] == "verdict"]
    return {"id": run_id, "provider": provider, "started": events[0]["t"] if events else None,
            "claims": len(verdicts), "admitted": sum(v["verdict"] == "ADMITTED" for v in verdicts),
            "done": any(e["kind"] == "final" for e in events), "track": _track_stem(run_id, events)}


BENCH = ROOT / "bench" / "results.json"
S2_RIGS = {"12° ramp · 31 Oct": RUNS / "s2_identify.json", "8° board · 1 Nov": RUNS / "nov01_s2" / "s2_identify.json"}


@app.get("/api/benchmark")
def benchmark() -> dict:
    """Three-arm results (bench/results.json) and the S2 inertia ratio identified on each rig."""
    if not BENCH.exists():
        raise HTTPException(404, "no benchmark results: run python -m bench.run_all")
    rigs = {}
    for name, path in S2_RIGS.items():
        if path.exists():
            d = json.loads(path.read_text())
            rigs[name] = {"beta": d["beta"], "beta_rel_std": d["beta_rel_std"], "structure": d["structure"],
                          "g1": d["gates"].get("G1", {})}
    return {"results": json.loads(BENCH.read_text()), "s2_beta": rigs}


@app.get("/api/runs")
def list_runs() -> list[dict]:
    ids = sorted((p.stem for p in RUNS.glob("agent_*.jsonl") if RUN_ID.match(p.stem)), reverse=True)
    return [_summary(i, _events(i)) for i in ids]


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    events = _events(run_id)
    meta, track = _track(_track_stem(run_id, events))
    out = {"summary": _summary(run_id, events), "events": events, "video": {"size": meta["size"]}}
    evidence = next((e["data"] for e in reversed(events) if e["kind"] == "evidence"), None)
    if evidence:
        out["overlay"] = _overlay(meta, track, evidence)
    return out


def _overlay(meta: dict, track: dict, evidence: dict) -> dict:
    """Ghost twin in image pixels per video frame, and a phase portrait at the identification rate."""
    x, (t0, q0, v0) = np.asarray(evidence["x"]), evidence["start"]
    base = _model_path(meta).read_text()
    fps = 60.0
    t_video = np.arange(t0, t0 + meta["frames"] / fps, 1 / fps)
    phi_twin_video = predict(base, x, t_video, (q0, v0))  # measured-angle space, same frame as the SAM track
    H_inv = np.linalg.inv(np.asarray(meta["H"]))
    pivot = np.asarray(meta["pivot_m"])
    down = np.asarray(meta["down"], float) / np.linalg.norm(meta["down"])
    right = np.array([down[1], -down[0]])  # swing_angle measures phi positive toward this direction
    r = meta["pivot_radius_m"]
    # tip of a uniform bar: (L + 2r)/2 below the pivot; without a measured length assume the pivot is at the end
    length = (meta["bar_length_m"] + 2 * r) / 2 if meta.get("bar_length_m") else 2 * r
    tips = pivot + length * (np.sin(phi_twin_video)[:, None] * right + np.cos(phi_twin_video)[:, None] * down)

    def to_px(pts: np.ndarray) -> list[list[float]]:
        h = np.c_[pts, np.ones(len(pts))] @ H_inv.T
        return (h[:, :2] / h[:, 2:]).round(1).tolist()

    t_id, phi_obs = track["t"], track["phi"]
    phi_twin = predict(base, x, np.r_[t0, t_id[t_id > t0]], (q0, v0))[1:]
    t_ph = t_id[t_id > t0]
    return {
        "t0": float(t0), "fps": fps, "pivot_px": to_px(pivot[None])[0], "tip_px": to_px(tips),
        "phase": {"t": t_ph.round(4).tolist(),
                  "obs": [phi_obs[t_id > t0].round(5).tolist(), np.gradient(phi_obs[t_id > t0], t_ph).round(4).tolist()],
                  "twin": [phi_twin.round(5).tolist(), np.gradient(phi_twin, t_ph).round(4).tolist()]},
        "n_fit": evidence["n_fit"],
    }


@app.get("/api/runs/{run_id}/video")
def get_video(run_id: str) -> FileResponse:
    meta, _ = _track(_track_stem(run_id, _events(run_id)))
    path = (ROOT / meta["video"]).resolve()
    if VIDEOS.resolve() not in path.parents or not path.exists():
        raise HTTPException(404, "video unavailable")
    return FileResponse(path, media_type="video/mp4")


@app.post("/api/runs/{run_id}/whatif")
def what_if(run_id: str, body: WhatIf) -> dict:
    """Re-verify the run's admitted controller under changed physics (operator input, not an agent patch)."""
    events = _events(run_id)
    evidence = next((e["data"] for e in reversed(events) if e["kind"] == "evidence"), None)
    admitted = [e["data"]["claim_id"] for e in events if e["kind"] == "verdict" and e["data"]["verdict"] == "ADMITTED"
                and e["data"]["type"] == "controller"]
    if evidence is None or not admitted:
        raise HTTPException(409, "this run has no admitted controller to re-verify")
    spec = next(e["data"]["spec"] for e in events if e["kind"] == "claim" and e["data"]["ref"] == admitted[-1])
    meta, track = _track(_track_stem(run_id, events))
    n = evidence["n_fit"]
    t, phi = track["t"], track["phi"]
    ev = Evidence(_model_path(meta).read_text(), np.asarray(evidence["x"]), np.asarray(evidence["cov"]), t[:n], phi[:n], t[n:], phi[n:],
                  tuple(evidence["start"]), float(track["r_meas"]), goal=GOAL, success_tol=0.05)
    ctx = Context(ev.base_xml, t, phi, float(track["r_meas"]), emit=lambda *_: None, evidence=ev, last_controller=admitted[-1])
    gains = spec.get("gains")
    ctx.claims[admitted[-1]] = Claim(admitted[-1], "operator", controller=LQRSpec(
        spec["q_eq"], tuple(spec["q_diag"]), tuple(spec["r_diag"]), spec["u_max"], tuple(gains) if gains else None))
    return reverify(body.model_dump(exclude_none=True), ctx)


@app.post("/api/runs/live")
def start_live(track: str = "s1_s1") -> dict:
    """Start a live agent run on a recorded track; poll /api/runs/{id} for events."""
    if not re.fullmatch(r"[A-Za-z0-9_]+", track):
        raise HTTPException(400, "malformed track name")
    meta, data = _track(track)
    model = _model_path(meta)
    api, provider, slug = resolve()
    run_id = f"agent_{track}_{time.strftime('%Y%m%dT%H%M%S')}"
    log = (RUNS / f"{run_id}.jsonl").open("w")
    lock = threading.Lock()

    def emit(kind: str, payload: object) -> None:
        with lock:
            log.write(json.dumps({"t": time.time(), "kind": kind, "data": payload}, default=float) + "\n")
            log.flush()

    emit("session", {"track": f"logs/runs/{track}.npz", "model": str(model.relative_to(ROOT))})
    emit("provider", {"provider": provider, "slug": slug})
    ok = np.isfinite(data["phi"])
    ctx = Context(model.read_text(), data["t"][ok], data["phi"][ok], float(data["r_meas"]), emit, proposer=f"{provider}:{slug}")

    def run() -> None:
        try:
            final = agent_loop(ctx, partial(run_turn, api, slug, emit=emit))
            emit("final", {"text": final, "verdicts": [(r["claim_id"], r["verdict"], r["reason"]) for r in ctx.records]})
        except Exception as err:  # surface failures in the run log instead of losing them in a thread
            emit("final", {"text": None, "error": f"{type(err).__name__}: {err}"})
        finally:
            log.close()

    threading.Thread(target=run, daemon=True).start()
    return {"id": run_id}

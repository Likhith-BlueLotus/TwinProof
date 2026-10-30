# TwinProof

**Film it. Twin it. Prove it.**

Film a mechanism for ~10 seconds with a printed marker in frame. TwinProof turns the video into a MuJoCo twin that
knows how wrong it is. An agent then proposes a controller. A deterministic verifier either issues a
**certificate** or **refuses**, naming the failed gate and the number that failed it.

> Certificates are **model-relative engineering checks**, not safety certifications. Control is evaluated on the
> identified twin; there is no physical actuator.

## What it does

```
phone video ─► SAM 3.1 (box-prompted video tracking) ─► marker homography ─► metres, pivot, swing angle
          ─► grey-box MuJoCo twin + OU residual (CMA-ES fit, Laplace posterior, identifiability gate)
          ─► agent (Muse Spark) proposes structures and controllers via tools
          ─► verifier G7 → G0 → G1 → G2 → G3 → G4 → G5 ─► ADMITTED + witness  |  REJECTED / REFUSED + reason
```

| Gate | Checks |
|---|---|
| G7 | Patch and code guard: only whitelisted MJCF fields; no goal, option or limit edits; AST scan of Python diffs |
| G0 | Admissibility: masses, inertia triangle, dissipation ≥ 0, μ ∈ [0, 1.5], solref ≥ 2 timesteps |
| G1 | Held-out prediction: open-loop NRMSE ≤ 0.10 and white one-step EKF innovations (Ljung–Box p > 0.05) |
| G2 | Stability across the posterior: spectral radius ≤ 0.98, common Lyapunov decrease, a **Lipschitz-sound** region of attraction |
| G3 | Closed-form actuator bound over the certified region ≤ the motor limit |
| G4 | Clearance, joint limits, contact force, friction cone across ensemble rollouts |
| G5 | 300 posterior rollouts with OU disturbance; Clopper–Pearson 95% failure bound ≤ 2% |
| G6 | CI self-test: 7 seeded-bad mutations must all be caught, with 0 false rejects |

Formulas, thresholds and pre-registered hypotheses: [BENCHMARK.md](BENCHMARK.md). Build log: [HACKATHON.md](HACKATHON.md). Plan: [PLAN.md](PLAN.md).

## Status (honest)

Built and tested (126 tests):
- **Verifier G0–G7** and the **G6 seeded-bad self-test**: 7/7 caught, each at the expected gate, 0 false rejects.
- **S1 pendulum:**
  - SAM 3.1 live tracking (pivot within 5.7 mm on a synthetic take)
  - identification (damping and Coulomb friction recovered to <2%)
  - EKF-based G1
  - LQR with certified region of attraction
- **S2 incline:**
  - rolling vs sliding decided from the slip residual
  - μ recovered on sliding takes
  - G7/G0/G1 structure verdicts
- **Agent loop:** live end to end. 3 controllers refused, then 1 admitted with a certificate; see `logs/runs/`.
- **Web console:**
  - ghost twin on the video
  - phase portrait
  - verdict card with gate chips
  - replayable claim stream
  - operator what-ifs

Not done yet:
- **Real footage:** everything so far runs on a rendered synthetic take with known physics.
- **Muse Spark Open:** waits on Baseten/Fireworks keys. Until then, runs use Muse Spark 1.3, and the UI labels it "dev fallback".
- **Glimmer:** the SFT data generator works; the QLoRA training and serving waits on Modal.
- **Remaining pieces:** S3 (door + Panda), voice what-ifs, the 3-arm benchmark.

## Quickstart

```bash
uv sync
cp .env.example .env            # MODEL_API_KEY (Meta), BASETEN_API_KEY or FIREWORKS_API_KEY (Spark Open)
uv run pytest -q                # ~4 min; the pre-commit gate runs the same suite plus the G7 scan
```

Console (recorded runs work offline; live runs need keys):

```bash
uv run --env-file .env uvicorn api.main:app --port 8000
npm --prefix web install && npm --prefix web run dev     # http://localhost:5173
```

Pipeline on your own take (see [data/FILMING.md](data/FILMING.md); print [the marker](data/aruco/aruco_4x4_id0_80mm.pdf) at 100%):

```bash
uv run --env-file .env python -m scripts.run_s1_video data/videos/s1_pendulum/take1.mp4 --side 0.080 --noun ruler
uv run --env-file .env python -m scripts.run_agent_s1 logs/runs/take1_s1.npz
```

## Meta tools used, and where

| Tool | Role |
|---|---|
| SAM 3.1 | Frame-0 concept detection, then one box-prompted video request tracks every frame (`perception/sam_orchestrator.py`, parsed with Meta's `meta_sam_parser`) |
| Muse Spark 1.3 | Scene nouns and candidate model classes from the video (`agents/spark_api.py`); dev-fallback agent model |
| Muse Spark Open | The agent: tools for fit, patch, controller, submit, what-if (`agents/`), via Baseten → Fireworks |
| Muse Glimmer 30B | Residual-plot critic; SFT data in `scripts/gen_glimmer_sft.py`, QLoRA on Modal (pending) |
| Muse Code | Repository commit gate: `.muse/hooks.json` blocks commits when tests or the G7 scan fail |

Muse Image is not used (not available in India).

## Limits

- **G2 soundness:** G2's Lipschitz bound covers the continuous-time model. MuJoCo integrator error at the 2 ms timestep is not bounded.
- **Measurement model:** angle calibration is a static cubic with a ±2% prior, and identification runs at 15 Hz after codec-safe decimation. Both are pre-registered in BENCHMARK.md.
- **Twins, not hardware:** G5 failure bounds are relative to the identified posterior, not the real hardware.

## Clean room

Created 2026-10-30 inside the hackathon window. Third-party code is used as dependencies, not modified. No employer code, data, tools or hardware were used. Details in [HACKATHON.md](HACKATHON.md).

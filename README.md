# TwinProof

**Film it. Twin it. Prove it.**

Film a mechanism for ~10 seconds with a printed marker in frame. TwinProof turns the video into a MuJoCo twin that
knows how wrong it is. An agent then proposes model fixes and controllers. A deterministic verifier either issues a
**certificate** or **refuses**, naming the failed gate and the number that failed it.

> Certificates are **model-relative engineering checks**, not safety certifications. Control is evaluated on the
> identified twin; there is no physical actuator.

## The result in one paragraph

On **five real rigs** filmed with a phone (three pendulums, two ramps), TwinProof's twins predict held-out takes to
**0.8–3.4% NRMSE** on the two well-built pendulum rigs (6–9% with a finger-held pin) and 1.5–8% on the ramps, and the can's inertia ratio **replicates across two different ramps**
(β = 0.78 ± 4.5% on a 12° ramp, 0.74 ± 2.2% on an 8° board). The verifier **refused every real-hardware controller**
anyway: on each rig the one-step prediction errors still carry structure (Ljung–Box p < 0.05), so the twin is not yet
trustworthy enough to certify. On the synthetic rig with known physics it admits a controller with a certified region
of attraction. That is the point: a verifier that only ever says yes is worthless, and this one says no with the number.

**Benchmark** (every logged rig, pre-registered tests; [BENCHMARK.md](BENCHMARK.md)):
- **H3 passes:** the verifier's self-test catches 7/7 seeded bugs, each at the expected gate, with 0 false rejects.
- **H1 is not supported.** TwinProof's twin beats default physics on 6 of 11 held-out units (p = 0.16) and the
  unverified agent on 2 of 7, where 5 tie because both arms chose the same twin.
- **What TwinProof changes is what ships.** The unverified agent shipped a controller on every real rig; TwinProof
  shipped none, because no real twin passed G1.

## What it does

```
phone video ─► SAM 3.1 (box-prompted video tracking) ─► marker homography ─► metres, pivot, swing angle / roll distance
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

Formulas, thresholds and pre-registered hypotheses: [BENCHMARK.md](BENCHMARK.md) (pre-registered in commit `84cf77a`,
before any real footage). Build log and every post-footage change: [HACKATHON.md](HACKATHON.md). Plan: [PLAN.md](PLAN.md).

## Real footage, rig by rig

| Rig | What the pipeline measured | Verdict |
|---|---|---|
| **S1 pendulum, 31 Oct** (33 cm ruler on a pin) | Pivot within 6.6 mm of the tape measure; dry friction 2.76 mN·m; held-out takes at **0.8% / 1.9%** | G1 refuses (whiteness); no controller certified — [S1_REAL.md](logs/runs/S1_REAL.md) |
| S1, finger-held pin, 1 Nov | Held-out 6–9%; the grip changes the dynamics take to take | G1 refuses — [nov01](logs/runs/nov01/README.md) |
| S1, clamped pin, 1 Nov | Three pivots agree within 5 mm; held-out **3.1% / 3.4%**; period still shortens below a 2° swing (pin-in-hole contact) | G1 refuses — [nov01_clamped](logs/runs/nov01_clamped/README.md) |
| **S2 can on a 12° ramp, 31 Oct** | Rolls (slip 0.2%); β = **0.78 ± 4.5%**; held-out 1.5% / 8.0% | G1 refuses narrowly (p = 0.044) — [S2_REAL.md](logs/runs/S2_REAL.md) |
| S2 can on a board, 1 Nov | Stated 13°; the video says **8.0°** (measured against the wall-tile horizon, and confirmed by the acceleration); β = **0.74 ± 2.2%** | G1 refuses (p = 0.003) — [nov01_s2](logs/runs/nov01_s2/README.md) |

Two findings the pipeline made on its own: a 13° ramp claim was contradicted by the motion (a twin built at 13° predicts
7.9 m of travel where the can rolled 0.3 m, and is refused), and a finger-held pivot was visible in the swing period.

## Status (honest)

Built and tested (Python suite + the voice-parser test):
- **Verifier G0–G7** and the **G6 seeded-bad self-test**: 7/7 caught, each at the expected gate, 0 false rejects.
- **S1 pendulum and S2 incline** on real phone footage: SAM 3.1 tracking, marker metric geometry, identification
  with posterior and identifiability, held-out evaluation, refusal with reasons.
- **Agent loop**, live end to end on every rig (Muse Spark 1.3 as the labelled dev fallback); on the synthetic rig
  3 controllers refused, then 1 admitted with a certificate.
- **Benchmark** over every logged rig: three arms (default physics, unverified agent, TwinProof), Holm-corrected H1.
- **Web console:** ghost twin on the video, phase portrait, verdict card with gate chips, replayable claim stream,
  typed and **spoken** what-ifs, and a Benchmark view with the β replication.

Not done:
- **No certified controller on real hardware.** The next rig change is a ball-bearing pivot instead of a ruler hole on a nail.
- **Muse Spark Open:** waits on Baseten/Fireworks keys; until then runs use Muse Spark 1.3; the UI badge names the model, and its tooltip says it stands in for Spark Open.
- **Glimmer:** the SFT data generator works; QLoRA training and serving wait on Modal.
- **S3 (door + Panda):** cut at the go/no-go gate (no footage).

## Quickstart

```bash
uv sync
cp .env.example .env            # MODEL_API_KEY (Meta), BASETEN_API_KEY or FIREWORKS_API_KEY (Spark Open)
uv run pytest -q                # ~4 min; the pre-commit gate runs the same suite plus the G7 scan
node --test web/src/voice.test.js
```

Console (recorded runs work offline; live runs need keys):

```bash
uv run --env-file .env uvicorn api.main:app --port 8000
npm --prefix web install && npm --prefix web run dev     # landing page http://localhost:5173, console /console.html
```

Pipeline on your own take (see [data/FILMING.md](data/FILMING.md); print [the marker](data/aruco/aruco_4x4_id0_80mm.pdf) at 100% and measure it):

```bash
uv run --env-file .env python -m scripts.run_s1_video data/videos/s1_pendulum/take3.mp4 --side 0.07 --noun ruler --mass 0.2 --length 0.33 --start 1.2
uv run --env-file .env python -m scripts.run_agent_s1 logs/runs/take3_s1.npz --model logs/runs/take3_model.xml
uv run python -m bench.run_all --write-benchmark      # ~25 min: refits every rig
```

## Meta tools used, and where

| Tool | Role |
|---|---|
| SAM 3.1 | Frame-0 concept detection, then one box-prompted video request tracks every frame (`perception/sam_orchestrator.py`, parsed with Meta's `meta_sam_parser`) |
| Muse Spark 1.3 | Scene nouns and candidate model classes from the video (`agents/spark_api.py`); dev-fallback agent model |
| Muse Spark Open | The agent: tools for fit, patch, controller, submit, what-if (`agents/`), via Baseten → Fireworks (keys pending) |
| Muse Glimmer 30B | Residual-plot critic; SFT data in `scripts/gen_glimmer_sft.py`, QLoRA on Modal (pending) |
| Muse Code | Repository commit gate: `.muse/hooks.json` blocks commits when tests or the G7 scan fail |

Muse Image is not used (not available in India). Spoken what-ifs use the browser's speech recognition and a
deterministic parser; no model sits between the microphone and the verifier.

## Limits

- **G2 soundness:** G2's Lipschitz bound covers the continuous-time model. MuJoCo integrator error at the 2 ms timestep is not bounded.
- **Measurement model:** angle calibration is a static cubic with a ±2% prior, and identification runs at 15 Hz after codec-safe decimation. Both are pre-registered in BENCHMARK.md.
- **Footage:** every real take is an 848×478 phone export. Full-resolution exports would tighten every number.
- **Twins, not hardware:** G5 failure bounds are relative to the identified posterior, not the real hardware.

## Clean room

Created 2026-10-30 inside the hackathon window. Third-party code is used as dependencies, not modified. No employer code, data, tools or hardware were used. Details in [HACKATHON.md](HACKATHON.md).

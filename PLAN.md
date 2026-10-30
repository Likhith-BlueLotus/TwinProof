# TwinProof — Build Plan A→V

Source of truth for the math and APIs: `docs/MASTER_BUILD.md` (local only, gitignored).
Times are IST. Exit criteria come from §6 of the master doc. A phase is done only when its exit criterion is met.

| Phase | When | Work | Files | Exit criterion | Needs |
|---|---|---|---|---|---|
| **A. Repo & compliance** | Fri 30 · now | Read MBRDI outside-work/IP clause; fresh public repo; HACKATHON.md, LICENSE, NOTICE, `.env.example`, pyproject | root | First commit; IP clause read | You (IP clause, GitHub repo) |
| **B. Day-1 probes** | Fri 14–16 | Spark Open slug resolver (Baseten plain → `-1026` → Fireworks); log 20 raw stream chunks; SAM single image: concept, zero-match, box prompt, wrong-aspect 400 | `agents/spark_open.py`, `perception/sam_orchestrator.py` | Slug + reasoning field recorded; raw SAM output cached | `MODEL_API_KEY`, `BASETEN_API_KEY`, `FIREWORKS_API_KEY` |
| **C. Filming** | Fri 16–17:30 | Print 8 cm ArUco, measure it; 3 takes × S1/S2/S3 at 60 fps; log Z, Δz | `data/videos/`, `data/aruco/` | 9 clips on disk | Phone, tripod, printer |
| **D. SAM tracking** | Fri 17:30–20 | Noun fan-out on frame 0 → one box-prompted video request → cache → parse | `perception/sam_orchestrator.py` | Masks for every S1 frame | B, C |
| **E. Rectify + moments** | Fri 20–21 | ArUco H, rectify-then-moments, θ unwrap, elongation & occlusion flags | `perception/masks.py` | Unit tests pass on synthetic masks | — |
| **F. Smoother** | Fri 21–21:30 | CA Kalman + RTS, variable Δ, SG cross-check | `perception/smoother.py` | Synthetic-trajectory test passes | — |
| **G. Kinematics** | Fri 21:30–22 | ICR, fixed-pivot WLS, Kåsa, prismatic SVD, slip | `perception/kinematics.py` | **Pivot within ±1 cm of tape measure** | D–F |
| **H. Twin model** | Sat 07:30–11 | S1/S2 MJCF, OU discretization + Kalman NLL, composite loss | `physics/models/`, `physics/ou.py`, `physics/loss.py`, `physics/mjcf_patch.py` | NLL decreases on S1 | G |
| **I. Fit + G0/G1** | Sat 11–15 | CMA-ES on 8 Modal workers, Laplace posterior, identifiability gate; G0, G1; **commit BENCHMARK.md pre-registration** | `physics/fit_cmaes.py`, `infra/modal/fit_worker.py`, `verifier/gates.py` | S1 held-out NRMSE computed | Modal account |
| **J. Glimmer data + QLoRA** | Sat 15–19, 23–01 | SFT generator, Spark Open teacher rationales, Axolotl QLoRA overnight on 1×H100 | `data/glimmer_sft/`, `infra/modal/glimmer_train.py` | ≥1,700 rows; training job running | Modal GPU credits |
| **K. Agent loop** | Sat 19–23 | Spark Open orchestrator, tool dispatch, Glimmer critic + escalation, 6-rejection stop | `agents/*.py` | Agent admits the correct S1 structure | B, I |
| **L. Controllers + G2/G3** | Sun 07:30–12 | LQR, swing-up, PD impedance (parametric); spectral, common Lyapunov, Lipschitz-sound annulus, c*; support-function bounds | `physics/controllers.py`, `verifier/lyapunov.py`, `verifier/bounds.py` | S1 LQR passes or fails with a witness | I |
| **M. G4–G7 + self-test** | Sun 12–16 | Collision/contact/friction cone, Clopper–Pearson G5, seeded-bad G6, AST/patch guard G7, Muse Code tests + commit hook | `verifier/*`, `.muse/` | **Seeded-bad catch 100%, 0 false rejects, CI green** | Muse Code |
| **N. S3 door + Panda** | Sun 16–22 | Menagerie Panda (vendored unchanged), door twin, `pd_impedance` | `third_party/`, `physics/models/s3*.xml` | **KILL GATE 22:00: no admitted S3 claim → drop S3** | — |
| **O. Glimmer eval** | Sun 22–01 | Merge + vLLM serve tuned Glimmer; 200-row eval (H4) | `infra/modal/glimmer_serve.py` | H4 numbers logged | J |
| **P. API** | Mon 07:30–09 | FastAPI: upload, run, WS events, certificate JSON | `api/main.py` | WS streams a cached S1 run | K–M |
| **Q. Web UI** | Mon 09–20 | React + Three.js: ghost overlay, phase portrait, claim stream, certificate/refusal card, 3-arm race | `web/` | **End-to-end S1 in the browser** | P |
| **R. Voice what-ifs** | Mon 20–24 | Browser PCM16 → livekit-plugins-meta → `reverify` | `voice/relay.py` | Spoken command triggers re-verify | P |
| **S. Benchmark** | Tue 07:30–14 | Full grid A0/A1/A2 × S1–S3 + ablations, fill tables from logs only | `bench/run_all.py`, `BENCHMARK.md` | Tables filled from logs | All above |
| **T. Docs + freeze** | Tue 14–18 | README, HACKATHON.md build table, clean-clone test, deploy demo | root | **CODE FREEZE 18:00 Tue** | — |
| **U. Demo video** | Tue 18 – Wed 14 | Screen captures + VO; edit 3:00 cut + 45 s vertical (OpenMontage at `~/tools/OpenMontage`) | `demo/` | Final cut ≤ 3:00 | Mic, recordings |
| **V. Submit + launch** | Wed 14–20 | Form (Open-Weight, Agentic, Community, Viral), re-check PhysicsLens, X post 19:30, submit ~20:00 | — | Confirmation email | You |

**Hard deadline:** 07:30 IST Thu Nov 5. **Fallbacks:** master doc §6.1. **Cut order if behind:** S3 → voice → Glimmer QLoRA (ship base Glimmer with honest eval).

Keyless work (E, F, G math, verifier stats, G0/G6/G7) can run ahead of the probes, so it's built first.

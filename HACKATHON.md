# HACKATHON.md

Meta Global AI Developer Hackathon (Cerebral Valley) · Indie Developer Track · Push the Frontier.
Plan: [PLAN.md](PLAN.md).

## Build table

| Date (IST) | Time | Commit range | What was built | Model/tool used | Evidence (log/clip) |
|---|---|---|---|---|---|
| 2026-10-30 | 13:30– | | Repo scaffold; verifier stats; KF/RTS smoother (F) | Claude Code | `pytest` |
| 2026-10-30 | | | Rectify-then-moments + ArUco (E); fixed pivot/ICR/Kåsa/prismatic/slip (G) | Claude Code | `pytest tests/test_masks.py tests/test_kinematics.py` |
| 2026-10-30 | | | S1 MJCF, whitelisted patcher, OU residual, composite loss (H); CMA-ES + Laplace + identifiability, G0/G1 (I) — synthetic data only | Claude Code | `pytest tests/test_physics.py tests/test_identify_s1.py` |
| 2026-10-30 | | | LQR, Lipschitz-sound ROA shells, G2/G3 (L); G4/G5/G7, full verify chain, G6 seeded-bad 7/7 caught, 0 false rejects (M) — synthetic data only | Claude Code | `pytest tests/test_controller_gates.py tests/test_verify.py tests/test_g7_scan.py` |
| 2026-10-30 | | | Muse Code commit hook + git pre-commit (tests + G7); BENCHMARK.md draft; day-1 Meta probes; SAM orchestrator on meta_sam_parser; synthetic S1 video → live SAM tracking → pivot 5.7 mm from truth | Claude Code; SAM 3.1; Muse Spark 1.3 | `logs/probes/FINDINGS.md`, `logs/runs/s1_s1.json` |
| 2026-10-30 | | | Spark 1.3 scene nouns; agent loop (resolver, tools, dispatch, 6-reject stop); G1 measurement model (decimation, calibration, EKF innovations); cma seed-0 reproducibility fix; live agent on SAM-tracked 15 s synthetic take: 3 refused, c-4 admitted | Claude Code; Muse Spark 1.3 (dev fallback); SAM 3.1 | `logs/runs/agent_s1_s1_20261030T214459.jsonl` |
| 2026-10-30 | | | FastAPI backend (runs, ghost-overlay geometry, what-if re-verify, live start); React web UI (design tokens via ui-ux-pro-max, motion via emil-design-eng): video + ghost twin, phase portrait, verdict card with gate chips, replayable claim stream, what-if bar | Claude Code | `tests/test_api.py`; overlay within 8 px median of the true bar tip |
| 2026-10-30 | | | S2 incline: MuJoCo ramp + can (rolling 1.658 vs 1.693 m/s² theory, sliding exact), slip-residual contact classification, μ identification, G7/G0/G1 verdicts; Glimmer SFT generator (6 fix classes, seed-split); README | Claude Code | `tests/test_identify_s2.py`, `tests/test_glimmer_sft.py` |
| 2026-10-31 | | | Real S1 footage: marker consensus filter, cross-take registration, measured-geometry twin; Coulomb-only structure; twin selection by BIC among identifiable fits; per-run model in the API. Held-out takes predicted at 0.78% / 1.89% NRMSE; G1 whiteness refuses certification | Claude Code; SAM 3.1; Muse Spark 1.3 (dev fallback) | `logs/runs/S1_REAL.md` |
| 2026-10-31 | | | Three-arm benchmark (`bench/run_all.py`): held-out NRMSE real S1 A2 0.8/1.9% vs A1 1.5/2.2% vs A0 3.5/4.5%; A2 ships no controller on the real rig (G1); control task does not separate arms (all 100%); H3 7/7, 0 false rejects; H1 underpowered (3 units) | Claude Code | `bench/results.json`, BENCHMARK.md tables |
| 2026-10-31 | | | Real S2 footage: SAM tracking ("silver disc"), stripe rotation, radius-based scale correction, straight-run window; fitted can inertia (β); slip by line fit; take 1 → β 0.78 ± 4.5%, held-out take 2 at 1.5% NRMSE, take 3 at 8.0%; G1 whiteness refuses narrowly (p 0.044) | Claude Code; SAM 3.1 | `logs/runs/S2_REAL.md` |
| 2026-11-01 | | | Re-films: S1 with a finger-held pin, then a clamped pin (pivots agree within 5 mm; held-out 3.1/3.4%; G1 still refuses: period shortens below a 2° swing; Dahl friction, period-matched radius and armature tested and rejected). S2 on an 8° board (measured against the wall-tile horizon, not the stated 13°): β 0.738 ± 2.2% replicates 0.78 ± 4.5%; G1 refuses (p 0.003) | Claude Code; SAM 3.1; Muse Spark 1.3 (dev fallback) | `logs/runs/nov01*/README.md` |
| 2026-11-01 | | | Benchmark over every logged rig (3 real S1, 2 real S2, synthetic) with Holm decision; console Benchmark view (three-arm table, β replication chart); rig-named runs; spoken what-ifs (browser speech → deterministic parser → same bounded re-verify); API track containment check | Claude Code | `bench/results.json`, `tests/test_api.py`, `web/src/voice.test.js` |
| 2026-11-01 | | `69042b7`–`49d27fe` | Full-codebase review (four reviewer agents), fixed test-first: actuator gear/kp off the patch whitelist; G5 counts NaN rollouts as failures; ROA never certifies above c_hi; G0 refuses non-finite patch values; claimed code gets no G7 reset exemption; Wilcoxon floor counts untied pairs (A2 vs A1 floor 0.25, not 0.0078); API refuses non-local origins and runs one heavy job at a time. Console: light theme, run picker, run-switch races fixed | Claude Code | `tests/test_controller_gates.py`, `tests/test_api.py`, `tests/test_bench.py` |
| 2026-11-01 | | `d39b83a` | Landing page at `/` (console moved to `/console.html`); stills drawn from the real takes with the identified twin | Claude Code | `web/index.html`, `scripts/export_stills.py` |
| 2026-11-03 | | | Glimmer: Modal QLoRA, merge and vLLM serving app; teacher-rationale pass (Muse Spark 1.3 stands in for Spark Open: no key; rows it does not support are dropped); H4 eval script (top-1 fix accuracy, JSON validity, exact McNemar). Not trained or evaluated: Modal refuses GPUs without a payment method, and no other Glimmer host has a key. Meta Model API critics zero-shot on the 200 held-out rows: Muse Spark 1.2 91.0%, 1.3 83.5% top-1 | Claude Code; Muse Spark 1.2/1.3 | `logs/glimmer_eval_spark.json`, `infra/modal/glimmer.py`, `scripts/eval_glimmer.py` |

## Pre-registration

BENCHMARK.md pre-registration commit: **`84cf77a`** (2026-10-30 21:49 IST, pushed). The first real footage was written 2026-10-31 19:03, so every threshold, the λ weights and the G1 definition predate real data.

**Changes after seeing real footage:** perception and structure list only; no threshold changed.
- **Added:**
  - the `pendulum_coulomb` structure (dry friction only); the existing two could not represent the rig
  - the marker consensus filter
  - cross-take camera registration (`--marker-from`)
  - rig reuse for poorly conditioned takes (`--rig-from`)
  - a `--start` trim for the hand release
- **Fixed:** twin selection (best-BIC identifiable fit, independent of the order parallel fits finish in).
- **Took measured inputs:** bar length 33 cm, mass 200 g, marker side 70 mm. For S2: can r 3.25 cm, mass 68.5 g, ramp 12° and 35.5 cm.
- **S2 changes (after footage):** fitted can inertia ratio β (whitelisted inertial); slip measured by a line fit instead of a per-frame median.
- **1 Nov re-films (after footage):** an operator box prompt (`--box`) when no SAM concept matches under the release stick; the S2 identification take chosen by the longest clean straight run (`--ident`), decided before fitting; the S2 slope measured against the wall-tile horizon. No threshold changed. Hypotheses tested on the clamped rig and rejected (Dahl friction, period-matched radius, period-matched armature) are logged, not used.
- **Benchmark scope:** every logged real rig is a scenario, including the finger-held pin, so the held-out units are not cherry-picked.
- **Not used as data:** the two AI-generated clips in `data/videos/synthetic/` (Google encoder).

## Clean-room statement

All TwinProof code, data and footage were created between 2026-10-30 and 2026-11-04 IST, after the hackathon start (2026-10-26 12:00 PM ET), in a repository created on 2026-10-30. Third-party code is vendored under `third_party/` and labelled "third-party, unchanged". Commits containing Muse Code–generated code are tagged `[muse-code]`. No code, data, tools or hardware belonging to the author's employer were used. Muse Image was not used. The author reviewed the intellectual-property clause of their employment agreement on 2026-11-03, before submission.

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

## Pre-registration

BENCHMARK.md pre-registration commit: _(fill on Sat before any held-out evaluation)_

## Clean-room statement

All TwinProof code, data and footage were created between 2026-10-30 and 2026-11-04 IST, after the hackathon start (2026-10-26 12:00 PM ET), in a repository created on 2026-10-30. Third-party code is vendored under `third_party/` and labelled "third-party, unchanged". Commits containing Muse Code–generated code are tagged `[muse-code]`. No code, data, tools or hardware belonging to the author's employer were used. Muse Image was not used.

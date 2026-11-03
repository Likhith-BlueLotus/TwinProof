# BENCHMARK.md — pre-registration

> **Status: DRAFT.** Freeze after the S1 take-1 fit and **before any held-out evaluation**. Then record the freezing commit hash in [HACKATHON.md](HACKATHON.md). After the freeze, nothing above "Results" changes; the tables below are filled only from logged runs.

## Protocol
- **Why G1 uses one-step innovations:** an open-loop simulation of a lightly damped oscillator drifts in phase over many swings, even with correct physics (0.1% period error). On a 15 s SAM-tracked synthetic take, open-loop whiteness rejected the true twin (p = 1e-18), while EKF innovations accepted it (p = 0.86) and still rejected 1.5× damping and ±5 cm length errors. Recorded 2026-10-30, before any real footage.
- Each scenario has 3 takes at 60 fps. **Take 1 identifies; takes 2 and 3 are held out.** Within take 1, G1 also fits on the first 70% and predicts the last 30%.
- "Reality" for control evaluation is the twin re-identified from the held-out takes. There is no physical actuator.
- Every verdict is reproducible from `(claim, data_hash, seed)`. Seeds are logged.

## Scenarios
| ID | Mechanism | Identified | Control task (in twin) | Status |
|---|---|---|---|---|
| S1 | Pendulum | damping, frictionloss, OU (κ, σ), initial state | LQR hold at upright (q = π), \|θ−π\| ≤ 0.05 rad at 5 s | Guaranteed |
| S2 | Incline | rolling/sliding, μ_k, α, OU | PD hold on the ramp | Guaranteed |
| S3 | Hinged door + Panda | hinge pivot, damping, frictionloss, OU | Open to target angle, handle force ≤ 40 N | Killed if no admitted claim by Sun 22:00 IST |

## Arms
| Arm | Definition |
|---|---|
| A0 | Controller tuned on default MuJoCo physics (no identification) |
| A1 | Spark Open proposes structure, parameters and controller; accepted on its own say-so |
| A2 | TwinProof: only verifier-admitted claims |

## Frozen constants (must match code)
| Constant | Value | Where |
|---|---|---|
| Measurement preprocessing | uniform resample, Kaiser-FIR (β = 8) decimation 60 → 15 Hz, 3 samples trimmed per end | `perception/smoother.py` `decimate_uniform` |
| Observation model | static cubic calibration φ_obs = q + c₀ + c₁q + c₂q² + c₃q³, prior cₖ ~ N(0, 0.02²) | `physics/identify_s1.py` `observe`, `CAL_PRIOR_SD` |
| G1 held-out NRMSE max (open-loop continuation of the fitted trajectory) | 0.10 | `verifier/gates.py` `NRMSE_MAX` |
| G1 whiteness: Ljung–Box Q(10) p min, on held-out **one-step EKF innovations** (process noise tuned on the fit window only) | 0.05 | `LJUNG_BOX_P_MIN`, `physics/ekf.py` |
| G0 μ range | [0, 1.5] | `MU_MAX` |
| G2 spectral radius max (per 20 ms control period) | 0.98 | `SPECTRAL_MAX`, `physics/controllers.py` `CONTROL_PERIOD` |
| G2 ROA search ceiling | \|θ−θ_eq\| ≤ 1 rad | `verifier/pipeline.py` `THETA_REGION_MAX` |
| G2 ROA models | nominal + 2 worst-ρ posterior samples | `ROA_WORST_SAMPLES` |
| G4 clearance min / force max / cone margin | 0.01 m / 40 N / 0.85 | `CLEARANCE_MIN`, `verifier/verify.py` `F_MAX`, `FRICTION_CONE_MARGIN` |
| G5 rollouts / horizon / Clopper–Pearson 95% UB max | 300 / 5 s / 2% | `N_ROBUST`, `HORIZON_S`, `P_FAIL_MAX` |
| Posterior samples (G2) | 256 | `N_POSTERIOR` |
| Identifiability (on the MJCF parameters φ only; OU κ, σ are noise hyperparameters): cond(H_φφ) max / relative std max | 1e6 / 0.5 | `physics/fit_cmaes.py`, `physics/identify_s1.py` `PHYSICAL` |
| CMA-ES budget / seed handling | 5000 evaluations; seed s → cma seed s+1 (cma treats 0 as clock) | `agents/dispatch.py`, `physics/fit_cmaes.py` |
| Loss λ_prior (λ₃) | 1e-3 | `physics/identify_s1.py` `LAM_PRIOR` |
| Loss λ_IoU (λ₁), λ_slip (λ₂), λ_margin (λ₄) | _set on S1 take 1, then frozen_ | not yet in code |

## Hypotheses
| H | Statement | Test | Decision rule |
|---|---|---|---|
| H1 | A2 held-out NRMSE < A0 and < A1 | One-sided Wilcoxon signed-rank on paired (scenario, held-out take) units, Holm over 2 comparisons | Reject H0 at α = 0.05. With ≤ 6 pairs the minimum p is 1/64, so every pair must favour A2 |
| H2 | G5's failure bound is calibrated; OU beats white noise | Per admitted controller: held-out empirical failure rate vs p⁺ (binomial exceedance); Brier OU vs white (paired) | Holds if no OU-arm controller exceeds its p⁺ at α = 0.05 and Brier(OU) < Brier(white) |
| H3 | Seeded-bad catch = 100%, 0 false rejects | Counts over the mutation library and the known-good set; Clopper–Pearson 95% lower bound on catch rate | Pass iff catch = n/n and false rejects = 0 |
| H4 | QLoRA Glimmer top-1 fix accuracy > base Glimmer | Exact one-sided McNemar on 200 held-out rows | Reject H0 at α = 0.05 |

## Results (fill only from logged runs)

### Main metrics
_Filled by `bench/run_all.py` on 2026-11-01 21:38 from `bench/results.json`._

| Scenario | Arm | Twin structure | Held-out NRMSE | Task success (N=256) | p_fail 95% UB | max commanded torque / limit | Claims proposed / admitted | Wall-clock (s) |
|---|---|---|---|---|---|---|---|---|
| S1 real · 31 Oct | A0 | nominal (no identification) | take1_s1: 3.5%, take2_s1: 4.5% | 100.0% | 1.2% | 0.48 | 1 / 1 | — |
| S1 real · 31 Oct | A1 | pendulum_viscous_coulomb | take1_s1: 1.5%, take2_s1: 2.2% | 100.0% | 1.2% | 0.63 | 7 / 7 | 236 |
| S1 real · 31 Oct | A2 | pendulum_coulomb | take1_s1: 0.8%, take2_s1: 1.9% | **no controller shipped** (refused) | — | — | 7 / 0 | 236 |
| S1 real · finger-held pin | A0 | nominal (no identification) | nov01/take1_s1: 7.3%, nov01/take2_s1: 7.4% | 100.0% | 1.2% | 0.49 | 1 / 1 | — |
| S1 real · finger-held pin | A1 | pendulum_coulomb | nov01/take1_s1: 6.0%, nov01/take2_s1: 8.8% | 100.0% | 1.2% | 0.66 | 7 / 7 | 384 |
| S1 real · finger-held pin | A2 | pendulum_coulomb | nov01/take1_s1: 6.0%, nov01/take2_s1: 8.8% | **no controller shipped** (refused) | — | — | 7 / 0 | 384 |
| S1 real · clamped pin | A0 | nominal (no identification) | nov01_clamped/take1_s1: 4.1%, nov01_clamped/take2_s1: 2.8% | 100.0% | 1.2% | 0.49 | 1 / 1 | — |
| S1 real · clamped pin | A1 | pendulum_coulomb | nov01_clamped/take1_s1: 3.1%, nov01_clamped/take2_s1: 3.4% | 100.0% | 1.2% | 0.67 | 7 / 7 | 258 |
| S1 real · clamped pin | A2 | pendulum_coulomb | nov01_clamped/take1_s1: 3.1%, nov01_clamped/take2_s1: 3.4% | **no controller shipped** (refused) | — | — | 7 / 0 | 258 |
| S1 synthetic | A0 | nominal (no identification) | last 30%: 352.1% | 100.0% | 1.2% | 0.12 | 1 / 1 | — |
| S1 synthetic | A1 | pendulum_viscous_coulomb | last 30%: 4.3% | 100.0% | 1.2% | 0.21 | 6 / 6 | 210 |
| S1 synthetic | A2 | pendulum_viscous_coulomb | last 30%: 4.3% | 100.0% | 1.2% | 0.09 | 6 / 1 | 210 |
| S2 real · 12° ramp, 31 Oct | A0 | nominal solid can (no identification) | take2: 1.9%, take3: 7.9% | no S2 control task built | — | — | — | — |
| S2 real · 12° ramp, 31 Oct | A2 | incline_rolling | take2: 1.5%, take3: 8.0% | no S2 control task built | — | — | — | — |
| S2 real · 8° board, 1 Nov | A0 | nominal solid can (no identification) | take1: 5.3%, take3: 5.0% | no S2 control task built | — | — | — | — |
| S2 real · 8° board, 1 Nov | A2 | incline_rolling | take1: 5.9%, take3: 5.6% | no S2 control task built | — | — | — | — |

**At this disturbance level every shipped controller holds every reality sample, so the control task does not separate the arms.** The separation shows in held-out prediction, and in A2 refusing to ship where the twin is inadequate. The task and the motor sizing were fixed before these runs and are not tuned to produce a difference.

- **S1 real · 31 Oct**: reality = take1_s1 re-identified (pendulum_viscous_coulomb). A2 refusal: G1: innovations not white (Ljung-Box p=7.11e-08); unmodelled dynamics remain
- **S1 real · finger-held pin**: reality = nov01/take1_s1 re-identified (pendulum_coulomb). A2 refusal: G1: held-out NRMSE 0.236 > 0.1; the structure does not predict unseen motion
- **S1 real · clamped pin**: reality = nov01_clamped/take1_s1 re-identified (pendulum_coulomb). A2 refusal: G1: held-out NRMSE 0.160 > 0.1; the structure does not predict unseen motion
- **S1 synthetic**: reality = known synthetic physics.
- **S2 real · 12° ramp, 31 Oct**: takes [2, 3] held out; identification take 1; A1 not run (agent loop is S1-only). A2 refusal: G1: innovations not white (Ljung-Box p=0.0444); unmodelled dynamics remain
- **S2 real · 8° board, 1 Nov**: takes [1, 3] held out; identification take 2; A1 not run (agent loop is S1-only). A2 refusal: G1: innovations not white (Ljung-Box p=0.00309); unmodelled dynamics remain
- Task: hold upright for 5 s from |Δθ| ≤ 0.2 rad, |ω| ≤ 0.5 rad/s; success = |Δθ| ≤ 0.05 rad at 5 s. The actuator clamps to the reality twin's limit; a commanded/limit ratio > 1 means the controller asked for more than the motor has.
- Mask IoU, min clearance and $ are not measured (no rendered-twin IoU; S1 has no contacts; per-call costs not logged).

**H1 (held-out NRMSE, A2 < A0 and A1):**
- A2 vs A0: A2 better on 6/11 paired units; one-sided Wilcoxon p = 0.160 (smallest attainable with 11 untied pairs: 0.00049).
- A2 vs A1: A2 better on 2/7 paired units (5 exact ties: both arms chose the same twin); one-sided Wilcoxon p = 0.250 (smallest attainable with 2 untied pairs: 0.25).
- Holm over 2 comparisons at α = 0.05: **H1 not supported**.


### Ablations
| Variant | Removed | Metric | S1 | S2 | S3 |
|---|---|---|---|---|---|
| −OU (white noise) | Correlated residual | Empirical held-out fail rate − p⁺ | | | |
| −Posterior ensemble | Σ_φ | # admitted that fail on held-out twin | | | |
| −Glimmer critic | Fast proposer | iterations; $/admitted | | | |
| −G1 | Anti-overfit | # admitted with held-out NRMSE > 0.10 | | | |
| Uniform vs posterior-guided falsification | Sampling | failures / 100 rollouts | | | |

### Verifier seeded-bad self-test (H3)
_Filled by `bench/run_all.py` on 2026-11-01 21:38._

| Mutation class | n injected | Caught | First catching gate | False rejects on known-good (n) |
|---|---|---|---|---|
| negative damping | 1 | yes | G0 | — |
| destabilizing gains | 1 | yes | G2 | — |
| torque limit x0.3 | 1 | yes | G3 | — |
| hinge offset 5 cm | 1 | yes | G1 | — |
| goal edit | 1 | yes | G7 | — |
| gravity off | 1 | yes | G7 | — |
| direct qpos write | 1 | yes | G7 | — |
| **Total** | 7 | 7 | G0: 1, G1: 1, G2: 1, G3: 1, G7: 3 | 0 of 3 |

Catch rate 7/7; one-sided 95% Clopper–Pearson lower bound 0.652. H3 **passes**.


### Glimmer eval (H4; 200 held-out rows)
| Model | Top-1 fix accuracy | JSON validity | Median latency (s) | p95 latency (s) | $/1k calls |
|---|---|---|---|---|---|
| Base Glimmer 30B | | | | | |
| QLoRA Glimmer 30B | | | | | |
| Spark Open zero-shot | | | | | |

**Not run (2026-11-03).** No GPU endpoint was available to serve either model: Modal refuses every GPU type until the
account has a payment method, and no Together, DeepInfra, Featherless or Fireworks key exists. H4 is untested, not failed.
The training and serving app (`infra/modal/glimmer.py`) and the eval (`scripts/eval_glimmer.py`) are built and unit-tested.

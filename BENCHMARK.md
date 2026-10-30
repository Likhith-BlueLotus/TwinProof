# BENCHMARK.md — pre-registration

> **Status: DRAFT.** Freeze after the S1 take-1 fit and **before any held-out evaluation**. Then record the freezing commit hash in [HACKATHON.md](HACKATHON.md). After the freeze, nothing above "Results" changes; the tables below are filled only from logged runs.

## Protocol
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
| G1 held-out NRMSE max | 0.10 | `verifier/gates.py` `NRMSE_MAX` |
| G1 Ljung–Box Q(10) p min | 0.05 | `LJUNG_BOX_P_MIN` |
| G0 μ range | [0, 1.5] | `MU_MAX` |
| G2 spectral radius max (per 20 ms control period) | 0.98 | `SPECTRAL_MAX`, `physics/controllers.py` `CONTROL_PERIOD` |
| G2 ROA search ceiling | \|θ−θ_eq\| ≤ 1 rad | `verifier/pipeline.py` `THETA_REGION_MAX` |
| G2 ROA models | nominal + 2 worst-ρ posterior samples | `ROA_WORST_SAMPLES` |
| G4 clearance min / force max / cone margin | 0.01 m / 40 N / 0.85 | `CLEARANCE_MIN`, `verifier/verify.py` `F_MAX`, `FRICTION_CONE_MARGIN` |
| G5 rollouts / horizon / Clopper–Pearson 95% UB max | 300 / 5 s / 2% | `N_ROBUST`, `HORIZON_S`, `P_FAIL_MAX` |
| Posterior samples (G2) | 256 | `N_POSTERIOR` |
| Identifiability: cond(H) max / relative std max | 1e6 / 0.5 | `physics/fit_cmaes.py` |
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
| Scenario | Arm | Held-out NRMSE | Mask IoU | Task success (N=256) | p_fail 95% UB | max torque / limit | min clearance (m) | Claims proposed / admitted | Wall-clock (s) | $ |
|---|---|---|---|---|---|---|---|---|---|---|
| S1 | A0 | | | | | | | | | |
| S1 | A1 | | | | | | | | | |
| S1 | A2 | | | | | | | | | |
| S2 | A0 | | | | | | | | | |
| S2 | A1 | | | | | | | | | |
| S2 | A2 | | | | | | | | | |
| S3 | A0 | | | | | | | | | |
| S3 | A1 | | | | | | | | | |
| S3 | A2 | | | | | | | | | |

### Ablations
| Variant | Removed | Metric | S1 | S2 | S3 |
|---|---|---|---|---|---|
| −OU (white noise) | Correlated residual | Empirical held-out fail rate − p⁺ | | | |
| −Posterior ensemble | Σ_φ | # admitted that fail on held-out twin | | | |
| −Glimmer critic | Fast proposer | iterations; $/admitted | | | |
| −G1 | Anti-overfit | # admitted with held-out NRMSE > 0.10 | | | |
| Uniform vs posterior-guided falsification | Sampling | failures / 100 rollouts | | | |

### Verifier seeded-bad self-test (H3)
| Mutation class | n injected | Caught | First catching gate | False rejects on known-good (n) |
|---|---|---|---|---|
| Negative damping | | | | |
| Destabilizing gains | | | | |
| Torque limit ×0.3 | | | | |
| Hinge offset 5 cm | | | | |
| Goal edit | | | | |
| `disableflags` / gravity off | | | | |
| Direct `qpos` write | | | | |
| **Total** | | | | |

### Glimmer eval (H4; 200 held-out rows)
| Model | Top-1 fix accuracy | JSON validity | Median latency (s) | p95 latency (s) | $/1k calls |
|---|---|---|---|---|---|
| Base Glimmer 30B | | | | | |
| QLoRA Glimmer 30B | | | | | |
| Spark Open zero-shot | | | | | |

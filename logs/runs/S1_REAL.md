# S1 on real footage (2026-10-31)

Rig: 33 cm, 200 g steel ruler on a pin; 70 mm ArUco marker on the board just below the pivot; phone on a stand at Z ≈ 1 m, Δz ≈ 5 mm; three takes at about 59.94 fps, 848×478 (downscaled on export).

## Perception
| | take 3 (identification) | take 1 (held out) | take 2 (held out) |
|---|---|---|---|
| Release angle | 32° | 22° | 13° |
| SAM masks (free swing) | 954/954 | 954/954 | 888/888 |
| Marker | decoded on 6 clean early frames (consensus filter dropped partially occluded decodes) | hidden all take: take 3's homography + camera shift (−0.62, +1.26) px from 42 background features | hidden: take 3's + shift (−0.50, +2.82) px, 36 features |
| Fitted pivot vs tape-measured (37, −74) mm | 6.6 mm ✅ | 6.3 mm ✅ (and within 0.4 mm of take 3) | ill-conditioned at 13°: 25 mm, so take 3's pivot is reused (`--rig-from`) |
| Centroid radius | 15.21 cm | 15.19 cm | 15.14 cm |

## Identification (Coulomb-only twin: dry friction at the pin)
| | take 3 | take 1 | take 2 |
|---|---|---|---|
| frictionloss | 2.761 mN·m ± 0.5% | 2.718 mN·m ± 2.0% | not identifiable (13° swing) |

**Held-out prediction with take 3's twin** (physics and calibration from take 3; only the start state comes from each held-out take; open loop):
- take 1: **0.78%** NRMSE, max error 0.86° over 15.5 s
- take 2: **1.89%** NRMSE, max error 0.91° over 14.4 s

## Verifier
- **Structures:** viscous-only is identifiable but fails G1 (NRMSE 14–23%). Viscous + Coulomb is not identifiable (damping ≈ 0). Coulomb-only is identifiable, with NRMSE 4.4% inside take 3's held-out 30%.
- **Whiteness:** G1 rejects every Coulomb-twin claim (p = 7e-8; best agent patch 4e-4). A small unmodelled effect remains: a finger-held pin, or air drag on the ruler.
- **Agent run:** `agent_take3_s1_20261031T194821.jsonl`. Six claims, all rejected at G1; the agent stopped at the six-rejection rule. **No controller is certified on this rig.** That is the verifier refusing honestly, not a pipeline failure.

## Bugs found on real data (fixed, with tests)
- **Marker hidden by the bar:** ArUco needs the whole marker, so a single-frame homography fails. Now: consensus filter, multi-frame median, and cross-take registration.
- **Racing fits:** parallel fits overwrote the twin in whatever order they finished. Now: best-BIC identifiable fit; refusals logged as verdicts.
- **API model:** the API drew every run with the synthetic 35 cm model. Now: each run's own twin.

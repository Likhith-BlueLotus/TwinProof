# S1 re-filmed with a clamped pin (2026-11-01, afternoon)

The pin is now held by a clamp with no fingers on it; the marker is clear of the bar. Z = 1 m, Δz = 2.5 mm, marker 70 mm, 848×478 at 59.94 fps. Same pre-registered pipeline. Take 3 identifies; takes 1 and 2 are held out.

## Perception
| | take 3 (identification) | take 1 (held out) | take 2 (held out) |
|---|---|---|---|
| Release / free swing from | 35° / 1.15 s (the hand crosses the bar 0.87–1.0 s) | 23° / 1.25 s | 16° / 0.9 s |
| SAM masks | 1059/1059 | 960/960 | 1000/1000 |
| Fitted pivot from marker top-left | (18.8, −5.0) cm, circle rms 0.7% | (18.6, −4.8) cm | (18.3, −4.9) cm |

- **Pivot agreement:** the three pivots agree to within 5 mm. On the finger-held rig the circle rms was 1.8% and one take's pivot was invalid.
- **The reported tape pivot (−5.5, −9.5) cm does not match the video.** The pin is about 18.5 cm right of the marker's top-left corner and 4.9 cm above it. The measurement may have used another corner or swapped axes, so it was not used.
- **Angle check:** the bar's own axis angle (mask orientation) equals the swing angle about the fitted pivot, with gain 0.998. The bar's metric length is constant across angle to within 0.5%, so homography anisotropy is under 2%.

## Twin and verifier (pre-registered pipeline)
| Structure | Identifiable | BIC | G1 window NRMSE | Whiteness p | Held-out take 1 / take 2 |
|---|---|---|---|---|---|
| viscous | yes | −928 | 22% | 4e-58 | 4.8% / 8.5% |
| **Coulomb (twin)** | yes | **−1005** | 16% | 3e-49 | **3.1% / 3.4%** |
| viscous + Coulomb | no (damping ± 96%) | −1000 | 16% | 4e-49 | 3.1% / 3.4% |

- **Coulomb twin:** frictionloss 2.94 mN·m (the 31 Oct rig: 2.76).
- **Verdict:** G1 refuses every twin, so **no controller is certified on this rig.**

## Why G1 still refuses
- The period is 0.954 s at 29°, 0.939 s at 14°, then **0.86–0.89 s below 2°**. A rigid pendulum on an ideal pin cannot shorten its period below its small-swing value (about 0.934 s here).
- The G1 window is the last 30% of take 3, where the swing is under 6°, so that is where the twin fails.
- **Hypotheses tested and rejected** (each made the held-out error worse):
  - Pre-sliding (Dahl) friction at σ₀ = 0.02–0.3 N·m/rad: window NRMSE 23–66%.
  - Period-matched centroid radius (15.99 cm): held-out 5.6–10%.
  - Period-matched armature (2.2e-4 kg·m²): held-out 5.8–6.4%.
- **Likely cause:** the ruler's hole rocking on a thin nail, a rolling contact whose geometry changes with angle.
- **Next rig:** a ball-bearing pivot.

# Second rig (2026-11-01): S1 and S2 re-filmed

Same pipeline and pre-registered protocol as the 31 Oct rig. The verifier refuses every twin from this footage, so the **primary results stay on the 31 Oct rig** (`logs/runs/take*_s1.*`, `s2_*`; videos in `data/videos/*_oct31/`).

## S1 pendulum (finger-held pin)
These videos were later replaced by the clamped re-film: see `../nov01_clamped/README.md`.

| | take 3 (identification) | take 1 (held out) | take 2 (held out) |
|---|---|---|---|
| Release / free swing from | 31° / 1.25 s | 25° / 1.0 s | 19° / 0.85 s |
| Marker | decoded on every take | | |
| SAM masks | 1021/1021 | 1034/1034 | 972/972 |
| Pivot | (18.3, −3.9) cm from marker top-left; take 2's own fit is 5.5 mm away | take 3's | take 3's |

- **Period:** 0.953 s at 30° amplitude, 0.937 s at 15°, then **0.80 s below 3°**. A rigid pendulum's period only shrinks toward its small-swing value (here about 0.93 s), so this points to a compliant pin: it is held by fingers in every take.
- **Twins:** every structure (viscous, Coulomb, both) fails G1, with whiteness p ≈ 0. Held-out NRMSE is 6–9%, against 0.8–1.9% on the 31 Oct rig.
- **What was tried:**
  - A centroid radius from the swing period (15.9 cm, against 14.4 cm from the masks) improves the fit-window BIC by about 350 but not the held-out error.
  - Clamping the angle calibration to its ±2% prior doesn't help either.
  - Cutting the small-amplitude tail would have been a post-hoc protocol change, so it was not done.

## S2 incline (stated: 13°, 69.5 cm)
- **Tracking:** SAM tracked the can on all three takes over 30–36 cm of straight roll.
- **Motion:** acceleration is **0.02 m/s² early and 0.09 m/s² late**, about 3.5 s to cover 30 cm, top speed about 0.2 m/s. A can rolling down 13° accelerates at about 1 m/s².
- **13° twin** (`s2_identify_13deg.json`): β pins at its bound of 1.5. Fit-window NRMSE is 2780% (it predicts 7.9 m of travel); G1 refuses.
- **2.5° twin** (the path angle in the marker frame; `s2_identify_2.5deg.json`): still 250%; G1 refuses.
- **Reading:** the effective slope is under 1°, and it rises along the board (sag or curvature). The stated angle is not the slope the can rolled on. **Catching this is the verifier's job:** a twin built from the wrong number is refused instead of certified.
- **Scale:** SAM's mask includes part of the can's barrel on this rig (the camera sees it from the side). The area-based scale correction then gives s / (ψ r) = 0.89. A minor-axis radius gives 0.97 here but is worse on the 31 Oct rig, so the committed method is unchanged.

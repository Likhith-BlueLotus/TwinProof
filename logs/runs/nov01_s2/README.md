# S2 re-filmed on a tilted board (2026-11-01, afternoon)

**Rig:**
- Board: 69.5 cm, its high end raised on a stack of books.
- Can: 68.5 g, r = 3.25 cm.
- Marker: 70 mm, on the wall behind.
- Footage: 848×478 at 59.94 fps. The can is held by a stick, then released.

## Slope: measured, not stated
- **Stated:** 13°.
- **From the motion:** acceleration 0.53–0.62 m/s² on all three takes. 13° would give about 1.2 m/s².
- **From the video geometry:** the wall's grey tile band is a true horizontal lying in the marker's plane. Mapped through the marker homography it sits at −0.54°, so the marker is slightly rotated. The can's path relative to that horizon is **7.3° (take 1), 7.96° (take 2), 7.2° (take 3)**.
- **The model uses 7.96°**, from take 2's 52 cm path, the most precise of the three.

## Perception
| Take | Concept | Clean straight run | Why it ends | Slip line fit |
|---|---|---|---|---|
| 1 | "can" | 21.7 cm, 0.89 s | can crosses the marker (mask area −20%) | 15.9% |
| 2 | operator box (no concept matched under the release stick) | **52.1 cm, 1.35 s** | can leaves the board | **9.4%, rolling** |
| 3 | "metal cylinder" | 21.5 cm, 0.92 s | can crosses the marker | 13.7% |

- **Slip on takes 1 and 3:** the "sliding" reading is not credible. Rolling on 8° needs μ ≥ 0.06, and the measurement has a known bias here: the camera sees part of the can's barrel, so the area-based radius is too large (3.8–4.0 cm against 3.4 cm on the mask's minor axis).
- **The scale method was not changed.** On the 31 Oct rig the minor axis reads worse, and choosing a method per rig after seeing the data would be tuning.

## Twin (take 2 identifies: the longest clean run, chosen before fitting)
| | 7.96° (horizon-corrected) | 6.94° (marker frame, uncorrected) | 31 Oct rig (12°, take 1) |
|---|---|---|---|
| Contact | rolling | rolling | rolling |
| β = I/(m r²) | **0.738 ± 2.2%** | 0.517 ± 29% | 0.78 ± 4.5% |
| G1 NRMSE | 4.4% ✅ | 5.2% ✅ | 5.3% ✅ |
| G1 whiteness | p = 0.003 ❌ | p = 0.0004 ❌ | p = 0.044 ❌ |
| Held-out NRMSE | 5.9% / 5.6% (18 mm max) | 6.1% / 5.7% | 1.5% / 8.0% |

- **Replication:** the can's inertia ratio replicates across two ramps and two angles: **0.74 vs 0.78**, within 1.6 combined standard deviations.
- **Ramp angle matters:** with the uncorrected angle, β drifts to 0.52 and loses precision. Measuring the slope against a real horizontal is what makes the twin consistent.
- **Verdict:** G1 still refuses both rigs on whiteness, so no S2 claim is admitted. The 31 Oct rig stays primary: its whiteness is closer to passing.

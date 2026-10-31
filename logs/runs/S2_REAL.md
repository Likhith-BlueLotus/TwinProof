# S2 on real footage (2026-10-31)

**Rig:**
- Ramp: a box at 12°, 35.5 cm long.
- Can: hollow, 68.5 g, r = 3.25 cm, with a black stripe on its end face.
- Marker: 70 mm, on the back board.
- Footage: 848×478 at ~60 fps.

## Perception (SAM 3.1: concept "silver disc"; "can" returned no matches at this size)
| Take | Roll frames | Scale correction (from can radius) | Path angle | Distance |
|---|---|---|---|---|
| 1 | 55 | 0.905 | 12.8° | 30.4 cm |
| 2 | 47 | 0.918 | 13.4° | 30.5 cm |
| 3 | 49 | 0.901 | 13.1° | 34.0 cm |

- **Straight run:** the roll window stops when the can leaves the straight line by more than 3 mm. On take 1 that is at 32.5 cm, which equals the ramp length minus one radius (32.25 cm).
- **Scale correction (~0.91):** the can rolls about 9.5 cm in front of the marker plane, so the marker overstates its travel. The measured radius corrects this.
- **Rotation:** ψ is the angle from the face centre to the stripe centroid, searched only inside 80% of the face radius (the rim shadow is dark too). Readings off the local median trend by more than 0.4 rad are dropped.

## Identification (protocol: take 1 identifies)
- **Contact:** incline_rolling. Relative slip 0.2%, measured as the slope of distance against rolled distance.
- **Inertia ratio:** β = I/(m r²) = **0.78 ± 4.5%**. That's hollow-ish; a solid cylinder would be 0.5 and a thin shell 1.0. Identifiable.
- **Verdict:** G7 pass, G0 pass. G1: NRMSE 5.3% passes, but whiteness p = 0.044 < 0.05, so the twin is **refused, narrowly**.

## Held-out takes (take 1's physics frozen; only release time and start offsets fitted per take)
| Take | NRMSE of s(t) | Max error | Slip |
|---|---|---|---|
| 2 | 1.5% | 9 mm | 4.4% (rolling) |
| 3 | 8.0% | 52 mm | 11.6% (borderline sliding) |

Take 3's larger error comes with its borderline slip, consistent with a skid at release.

## Not used as data
`data/videos/synthetic/Cylindrical_can_rolling_down_ramp_…_1080p.mp4` and `Pendulum_experiment_in_physics_…_1080p.mp4` are AI-generated: Google encoder, 24 fps. Their motion is generated, not measured, so they are excluded from identification, the benchmark and the demo.

# Filming checklist (Phase C)

All footage must be filmed by you during the hackathon window (clean-room rule). Raw clips are gitignored; only measurements are committed.

## Kit
- Phone that records **1080p at 60 fps** (iPhone: Settings → Camera → Record Video → 1080p HD at 60 fps).
- Tripod, or a stack of books/phone clamp. The phone must not move during a take.
- Marker: print `data/aruco/aruco_4x4_id0_80mm.pdf` at **100% / Actual size** (not "fit to page"). Glue it flat on stiff card. Measure the black square with a ruler and write the real side length in the take log. It should be 80 mm, and you must record whatever it actually is.
- Tape measure, kitchen scale, protractor (or a phone level app), masking tape.

## Rules for every take
1. Lock focus and exposure (iPhone: long-press on the subject until "AE/AF LOCK" appears). No zoom.
2. The **whole marker and the whole motion** must stay in frame for the entire clip.
3. The marker must lie **in the plane of motion**. Record the out-of-plane gap Δz (marker surface to the moving object's mid-plane) and the camera distance Z. Error ≈ Δz/Z, so keep Δz small and Z ≥ 1 m.
4. Use a plain background, a moving part that contrasts with it, even light, and no strong shadows across the marker.
5. Each clip is **~10–15 s**. Start recording, wait 1 s, release from rest, keep your hands out of frame after release.

## Static calibration (5 s, once per scenario setup)
- Marker and the resting object, camera on the tripod, nothing moving. Gives the measurement noise σ for the smoother.
- Film it right before each scenario's takes, without moving the tripod: `data/videos/static/s1.mp4`, `s2.mp4`, `s3.mp4`.

## S1: Pendulum (guaranteed)
- **Build:** a rigid bar such as a 30 cm wooden or steel ruler with a hole near one end. Use a bolt or nail through the hole as the pivot, fixed to a shelf edge or clamped stick so the bar swings freely. Stick a strip of contrasting tape along the bar.
- **Marker:** on a board **directly behind the bar, parallel to its swing plane**, as close to the bar as it can be without touching. Put it **within ~10 cm of the pivot bolt**: on a synthetic take, an 80 mm marker 28 cm from the pivot cost ~2.5 mm of pivot error from corner noise alone. Fill the frame with the swing and the marker. Don't shoot a wide room.
- **Camera:** facing the board straight on, at about 1–1.5 m.
- **Measure:** bar length, bar mass, hole-centre to bar-end, pivot position relative to the marker's top-left corner (x, y in mm along the marker edges), Z and Δz.
- **Takes:** take 1 released from ~30°, take 2 from ~20°, take 3 from ~40°.

## S2: Incline (guaranteed)
- **Build:** a flat board ramp at 10–20°, with the angle measured. Roll a **can of food** (a cylinder). Stick one strip of contrasting tape across the can's **end face**, like a clock hand, so its rotation is visible.
- **Marker:** on a **vertical** board or box standing right beside the ramp, **parallel to the can's end face**. That vertical plane is the plane of motion seen from the side. Keep Δz (marker surface to the can's end face) small, and place the marker near the middle of the rolling path.
- **Camera:** side view, looking straight at the marker and the can's end face, at about 1 m.
- **Measure:** ramp angle, can radius and mass, ramp length used, Z, Δz.
- **Takes:** 3 releases from rest at the same top position. Let it roll to the bottom without catching it in frame.

## S3: Cabinet door (optional, killed if not working by Sun 22:00)
- **Build:** a small cabinet door that swings freely. Put contrasting tape on its top edge.
- **Marker:** flat **on the cabinet top, level with the door's top edge**, so it lies in the door's swing plane (horizontal).
- **Camera:** overhead, looking straight down at about 1 m.
- **Measure:** hinge axis position relative to the marker, door width, Z, Δz.
- **Takes:** push the door open by hand to ~60°, release, and let it swing or settle. Do 3 takes.

## Demo B-roll (for the 3:00 video, storyboard §7.1)
Film these with a **second phone or a friend**, while the tripod phone records the real takes:
- **Hero shot (0:00–0:05):** over your shoulder, the tripod phone filming the pendulum with the marker visible. One release, about 8 s.
- **Setup wide shot:** the whole rig (bar, bolt, marker board, tripod phone) with no motion, 5 s.
- **Close-ups:** your hand releasing the bar (3 s); the marker held next to a ruler (3 s).
- **Optional:** the same three shots for the incline.
Only footage you film yourself goes in the video.

## Lessons from the first S1 takes (2026-10-31)
- **Marker beside the pivot, not under it.** Below the pivot the bar covers it for the whole swing, and ArUco needs all four corners. Keep it within ~10 cm of the pivot, off to one side.
- **Export at full resolution.** iPhone: AirDrop with Options → "All Photos Data", or copy the file directly. "Most compatible" sharing produced 848×478.
- **Measure the printed marker.** The current print is **70 mm** (printer scaling; kept on purpose), so every run passes `--side 0.07`. Every metric distance scales with this number; it is recorded per take in `take_log.csv`.
- **Fix the pin.** A pin held between fingers adds friction that changes from take to take; clamp the bolt.
- **Swing at least 20°.** A 13° take could not separate the pivot or the friction.

## Lessons from the second takes (2026-11-01)
- **The marker fix worked:** it was decoded cleanly on every S1 take.
- **The pin is still held by fingers, and it shows.** The swing period drops from 0.93 s to about 0.80 s below a 3° amplitude. A rigid pin can't do that; a compliant one can. G1 refuses every S1 twin, and held-out error is 6–9% (the 31 Oct rig gave 0.8–1.9%). Clamp the bolt.
- **Check the ramp angle against the motion.** At the stated 13°, the can would reach about 1 m/s and leave a 69.5 cm board in under 1.5 s. It actually took about 3.5 s to roll 30 cm, at 0.02–0.09 m/s² and speeding up along the board. That is under 1° effective, on a board that sags or curves. Measure the angle with a phone level app on the board itself, and use a stiff board raised on one end.
- **Clamped re-film (same day):** with no fingers on the pin, all three pivots agree to within 5 mm and the held-out error falls to 3.1% / 3.4%. The period still shortens below a 3° swing, so G1 still refuses. Next, a ball-bearing pivot (a skateboard or fidget-spinner bearing pressed onto the bolt) instead of a ruler hole on a thin nail. See `logs/runs/nov01_clamped/README.md`.
- **S2 re-film on a tilted board:** measured against the wall-tile line (a true horizontal), the slope is **8.0°**, not 13°. β = 0.74 ± 2.2% replicates the 31 Oct rig's 0.78 ± 4.5%. Keep the marker **off the can's path**: crossing it shrinks the can's mask and ends the clean window at 21 cm. See `logs/runs/nov01_s2/README.md`.
- Results: `logs/runs/nov01/README.md`. The primary results stay on the 31 Oct rig (videos in `*_oct31/`).

## Where files go
```
data/videos/static/s1.mp4  s2.mp4  s3.mp4
data/videos/s1_pendulum/take1.mp4  take2.mp4  take3.mp4      (latest rig; older rigs live in s1_pendulum_<date>/)
data/videos/s2_incline/take1.mp4   take2.mp4  take3.mp4
data/videos/s3_hinge/take1.mp4     take2.mp4  take3.mp4
data/videos/broll/<anything>.mp4
```
AirDrop the clips to the Mac, drop them into those folders, and fill one row per take in `data/take_log.csv`.

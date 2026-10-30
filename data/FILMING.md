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

## Static calibration (once, 5 s)
- Marker only, camera on the tripod, nothing moving. Gives the measurement noise σ for the smoother.
- Save as `data/videos/static/take1.mp4`.

## S1: Pendulum (guaranteed)
- **Build:** a rigid bar such as a 30 cm wooden or steel ruler with a hole near one end. Use a bolt or nail through the hole as the pivot, fixed to a shelf edge or clamped stick so the bar swings freely. Stick a strip of contrasting tape along the bar.
- **Marker:** on a board **directly behind the bar, parallel to its swing plane**, as close to the bar as it can be without touching. Put it **within ~10 cm of the pivot bolt**: on a synthetic take, an 80 mm marker 28 cm from the pivot cost ~2.5 mm of pivot error from corner noise alone. Fill the frame with the swing and the marker. Don't shoot a wide room.
- **Camera:** facing the board straight on, at about 1–1.5 m.
- **Measure:** bar length, bar mass, hole-centre to bar-end, pivot position relative to the marker's top-left corner (x, y in mm along the marker edges), Z and Δz.
- **Takes:** take 1 released from ~30°, take 2 from ~20°, take 3 from ~40°.

## S2: Incline (guaranteed)
- **Build:** a flat board ramp at 10–20°, with the angle measured. Roll a can or cylinder and a ball. Put one tape stripe across the can's end face so its rotation is visible.
- **Marker:** flat **on the ramp surface**, beside the rolling path.
- **Camera:** perpendicular to the ramp's side, so the can's end face and the marker are both visible.
- **Measure:** ramp angle, can radius and mass, Z, Δz.
- **Takes:** 3 releases from rest at the top, using the same object each time.

## S3: Cabinet door (optional, killed if not working by Sun 22:00)
- **Build:** a small cabinet door that swings freely. Put contrasting tape on its top edge.
- **Marker:** flat **on the cabinet top, level with the door's top edge**, so it lies in the door's swing plane (horizontal).
- **Camera:** overhead, looking straight down at about 1 m.
- **Measure:** hinge axis position relative to the marker, door width, Z, Δz.
- **Takes:** push the door open by hand to ~60°, release, and let it swing or settle. Do 3 takes.

## Where files go
```
data/videos/static/take1.mp4
data/videos/s1_pendulum/take1.mp4  take2.mp4  take3.mp4
data/videos/s2_incline/take1.mp4   take2.mp4  take3.mp4
data/videos/s3_hinge/take1.mp4     take2.mp4  take3.mp4
```
AirDrop the clips to the Mac, drop them into those folders, and fill one row per take in `data/take_log.csv`.

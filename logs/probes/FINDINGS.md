# Day-1 probe findings (2026-10-30, MODEL_API_KEY only)

Raw responses are the `*.json` files next to this one. Script: `scripts/probe_meta.py`.

| Check (master doc §8) | Result |
|---|---|
| Spark 1.3 Responses API, text | Works; `muse-spark-1.3` answered "ok" (13 in, 120 out, of which 110 reasoning tokens) |
| SAM 3.1 concept prompt | Works. "stick" found the bar in the synthetic S1 video; on the cartoon probe image it returned zero matches |
| SAM zero matches | Empty `output_text` with a completed stream, as documented |
| SAM box prompt under own id | Works; returns our id, a tightened box and a mask |
| SAM wrong aspect `w/h` | HTTP 400 "prompt coordinates are given in a 960x960 frame, whose aspect ratio does not match the 960x540 media" |
| Box prompts "always one_bit" | **Not observed**: the box-prompt mask came back `lossless` (`~`). The parser handles both |
| `meta_sam_parser` Python API | On PyPI (0.1.0, SAM License). `build_object_prompt`, `image/video_segmentation_format().create_parser()` with `push` + `finish(CompletedOutcome())`, `decode_mask_to_raster`. Boxes are half-open (x2 = right - 1 in the prompt text) |
| Files API `purpose="user_data"` + video tracking with `propagation_direction=forward` | Works: one box-prompted request tracked a 300-frame video with a mask on every frame |

## End-to-end synthetic S1 (`scripts/run_s1_video.py`)
- Video: `scripts/make_synthetic_s1.py`, 960x540 at 60 fps, known pivot 280 mm right and 40 mm below the marker's top-left corner, under mild perspective.
- Pivot error **11.1 mm** without sub-pixel ArUco refinement: 6.4 mm came from the homography, because corner error grows far from the 80 mm marker.
- With `CORNER_REFINE_SUBPIX`: **5.7 mm**, inside the ±1 cm exit criterion. Fixed pivot and Kåsa agree.
- **Implication for filming:** keep the marker close to the pivot, or use a bigger marker. Record at 1080p.

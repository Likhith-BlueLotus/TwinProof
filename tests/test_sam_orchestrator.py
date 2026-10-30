import json
from pathlib import Path

import numpy as np
import pytest

from perception.sam_orchestrator import box_prompt, parse, sanitize

PROBE = Path(__file__).parents[1] / "logs/probes/sam_box.json"


@pytest.mark.parametrize("noun,clean", [("wooden ruler", "wooden ruler"), ("bar and bolt", "bar"), ("ball, ramp", "ball"),
                                        ("long thin metal bar swinging", "long thin metal"), ("left rod", "rod"), ("Upper Arm", "Arm")])
def test_sanitize_keeps_one_concept(noun, clean):
    assert sanitize(noun) == clean


def test_sanitize_rejects_empty():
    with pytest.raises(ValueError):
        sanitize("and bolt")


def test_box_prompt_uses_our_ids_and_half_open_boxes():
    assert box_prompt({3: (10, 20, 30, 40)}, 960, 540) == "<0f>3<|box;x1=10;y1=20;x2=29;y2=39;w=960;h=540|>"
    with pytest.raises(ValueError):
        box_prompt({}, 960, 540)


@pytest.mark.skipif(not PROBE.exists(), reason="needs the recorded day-1 SAM probe")
def test_recorded_box_probe_decodes_to_the_bar():
    recs = parse(json.loads(PROBE.read_text())["output_text"], video=False)
    mask = next(r for r in recs if r.kind == "mask")
    assert mask.object_id == "1" and (mask.mask.height, mask.mask.width) == (350, 161)

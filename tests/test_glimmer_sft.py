import json

import cv2
import pytest

from scripts.gen_glimmer_sft import CASES, FIXES, HELDOUT_SEED0, OUT, make_row


@pytest.fixture(scope="module")
def rows():
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    return [make_row(seed) for seed in range(len(FIXES))]  # one row per class


def test_one_row_per_class_with_valid_schema(rows):
    assert sorted(r["label"] for r in rows) == sorted(FIXES)
    for r in rows:
        sys_msg, user, assistant = r["messages"]
        assert sys_msg["role"] == "system" and "JSON" in sys_msg["content"]
        image, text = (part[part["type"]] for part in user["content"])
        assert cv2.imread(image).shape == (256, 512, 3)
        assert "nrmse=" in text and "ljung_box_p=" in text
        answer = json.loads(assistant["content"])
        assert answer["fix"] == r["label"] and answer["params"] == CASES[r["label"]][2]


def test_wrong_structures_leave_visible_residuals(rows):
    by = {r["label"]: float(r["messages"][1]["content"][1]["text"].split("nrmse=")[1].split(";")[0]) for r in rows}
    assert by["no_change"] < min(by[k] for k in ("add_coulomb_friction", "adjust_pivot", "switch_to_sliding"))


def test_splits_never_share_seeds():
    assert HELDOUT_SEED0 > 10 * 1500

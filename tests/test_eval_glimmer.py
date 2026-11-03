"""H4 scoring: the critic's JSON is validated here, and McNemar counts only discordant rows."""
import pytest

from scripts.eval_glimmer import parse_fix, summarise

FIX = '{"fix": "add_coulomb_friction", "params": ["frictionloss"], "confidence": 0.8}'


@pytest.mark.parametrize("text", [FIX, f"```json\n{FIX}\n```", f"Answer: {FIX}"])
def test_parse_fix_accepts_the_schema_with_wrapping(text):
    assert parse_fix(text)["fix"] == "add_coulomb_friction"


@pytest.mark.parametrize("text", [None, "", "no json here", '{"fix": "x"}', '{"fix": "x", "params": [], "confidence": 1.5}',
                                  '{"fix": 3, "params": [], "confidence": 0.5}', "{not json}"])
def test_parse_fix_rejects_anything_off_schema(text):
    assert parse_fix(text) is None


def test_summarise_counts_only_discordant_rows():
    fix = lambda f, c=0.9: {"fix": f, "params": [], "confidence": c}  # noqa: E731
    labels = ["a", "b", "c", "d"]
    base = [fix("a"), fix("x"), None, fix("d")]          # right on a, d
    tuned = [fix("a"), fix("b"), fix("c", 0.3), fix("x")]  # right on a, b, c
    out = summarise(labels, base, tuned)
    assert out["base"]["top1"] == 0.5 and out["tuned"]["top1"] == 0.75
    assert out["discordant"] == {"tuned_only": 2, "base_only": 1}
    assert out["base"]["json_valid"] == 0.75 and out["tuned"]["escalated"] == 0.25

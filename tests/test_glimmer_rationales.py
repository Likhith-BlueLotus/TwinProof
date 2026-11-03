"""Teacher rationales: rows whose rationale names another fix are dropped; the merge adds reasoning, nothing else."""
from scripts.add_glimmer_rationales import contradicts, with_rationale


def test_a_rationale_naming_another_fix_contradicts_the_label():
    assert contradicts("This looks like add_viscous_damping, not friction.", "add_coulomb_friction")
    assert contradicts("Consider switch-to-sliding here.", "switch_to_rolling")  # hyphenated spelling too
    assert not contradicts("The residual flips sign with velocity at constant size: dry friction.", "add_coulomb_friction")


def test_with_rationale_adds_reasoning_to_the_answer_only():
    row = {"seed": 7, "label": "no_change", "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": []},
                                                         {"role": "assistant", "content": '{"fix": "no_change"}'}]}
    out = with_rationale(row, "white residuals")
    assert out["messages"][-1] == {"role": "assistant", "content": '{"fix": "no_change"}', "reasoning_content": "white residuals"}
    assert "reasoning_content" not in row["messages"][-1]  # input row untouched
    assert out["messages"][:2] == row["messages"][:2] and out["seed"] == 7

"""bench/run_all.py rendering: tables from saved results, idempotent BENCHMARK.md section replacement."""
import json
from pathlib import Path

import pytest

from bench.run_all import h1, markdown, refusal_reason, write_benchmark

RESULTS = Path(__file__).parents[1] / "bench/results.json"


@pytest.mark.skipif(not RESULTS.exists(), reason="no saved benchmark results")
def test_rewriting_benchmark_twice_is_idempotent(tmp_path, monkeypatch):
    results = json.loads(RESULTS.read_text())
    md = "# B\n\n### Main metrics\n| blank |\n\n### Ablations\n| a |\n\n### Verifier seeded-bad self-test (H3)\n| blank |\n\n### Glimmer eval\n| g |\n"
    (tmp_path / "BENCHMARK.md").write_text(md)
    monkeypatch.chdir(tmp_path)
    write_benchmark(*markdown(results), "t")
    once = (tmp_path / "BENCHMARK.md").read_text()
    write_benchmark(*markdown(results), "t")
    assert (tmp_path / "BENCHMARK.md").read_text() == once
    assert once.count("### Main metrics") == 1 and "| blank |" not in once
    assert "### Ablations\n| a |" in once and "### Glimmer eval\n| g |" in once  # other sections untouched


@pytest.mark.skipif(not RESULTS.exists(), reason="no saved benchmark results")
def test_refused_arm_reports_no_controller():
    main, seeded, h1 = markdown(json.loads(RESULTS.read_text()))
    assert "no controller shipped" in main and "H3 **passes**" in seeded and "Holm over 2 comparisons" in h1


def test_refusal_reason_is_the_last_refused_controller_not_a_later_patch():
    verdicts = [{"type": "controller", "verdict": "REJECTED", "reason": "G1: held-out NRMSE 0.160 > 0.1"},
                {"type": "structure", "verdict": "REJECTED", "reason": "G7: patch refused"}]
    assert refusal_reason(verdicts).startswith("G1")
    assert refusal_reason(verdicts[1:]).startswith("G7")  # no controller claim at all: the last refusal
    assert refusal_reason([]) is None


def test_h1_counts_ties_and_skips_missing_arms():
    arm = lambda v: {"heldout_nrmse": {"u1": v[0], "u2": v[1]}}  # noqa: E731
    results = {"scenarios": {"s1": {"arms": {"A0": arm((0.05, 0.05)), "A1": arm((0.02, 0.03)), "A2": arm((0.02, 0.01))}},
                             "s2": {"arms": {"A0": arm((0.02, 0.02)), "A2": arm((0.03, 0.01))}}}}
    out = h1(results)
    assert out["A0"]["pairs"] == 4 and out["A0"]["a2_better"] == 3
    assert out["A1"]["pairs"] == 2 and out["A1"]["ties"] == 1 and out["A1"]["a2_better"] == 1

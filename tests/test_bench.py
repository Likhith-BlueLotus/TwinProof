"""bench/run_all.py rendering: tables from saved results, idempotent BENCHMARK.md section replacement."""
import json
from pathlib import Path

import pytest

from bench.run_all import markdown, write_benchmark

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
    assert "no controller shipped" in main and "H3 **passes**" in seeded and "cannot reach" in h1

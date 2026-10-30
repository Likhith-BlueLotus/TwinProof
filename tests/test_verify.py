import json
from pathlib import Path

import numpy as np
import pytest

from physics.controllers import LQRSpec
from verifier.seeded_bad import evidence_from_truth, self_test
from verifier.verify import Claim, verify

S1 = (Path(__file__).parents[1] / "physics/models/s1_pendulum.xml").read_text()
X_TRUE = np.r_[np.log([0.004, 0.0008, 2.0, 0.01]), 0, 0]
REL_STD = np.r_[0.05, 0.15, 0.1, 0.1, 1e-6, 1e-6]
GOOD = Claim("c-good", "test", controller=LQRSpec(q_eq=np.pi, q_diag=(10, 0.1), r_diag=(10,), u_max=2.5))


@pytest.fixture(scope="module")
def ev():
    return evidence_from_truth(S1, X_TRUE, REL_STD, seed=3)


def test_good_claim_is_admitted_with_full_record(ev):
    v = verify(GOOD, ev, seed=0)
    rec = json.loads(v.to_json())
    assert v.admitted, rec
    assert list(rec["gates"]) == ["G7", "G0", "G1", "G2", "G3", "G4", "G5"]
    assert rec["gates"]["G5"]["p_fail_upper"] <= 0.02


def test_verdict_is_reproducible_from_seed(ev):
    assert verify(GOOD, ev, seed=5).to_json() == verify(GOOD, ev, seed=5).to_json()


def test_seeded_bad_self_test_catches_all_and_rejects_no_good(ev):
    report = self_test(GOOD, ev, seed=0)
    assert report["caught"] == report["n"] == 7, report["rows"]
    assert report["false_rejects"] == []
    expected = {"negative damping": "G0", "destabilizing gains": "G2", "torque limit x0.3": "G3",
                "hinge offset 5 cm": "G1", "goal edit": "G7", "gravity off": "G7", "direct qpos write": "G7"}
    assert {k: r["first_gate"] for k, r in report["rows"].items()} == expected

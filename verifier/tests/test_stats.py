import numpy as np
import pytest

from verifier.stats import clopper_pearson_lower, clopper_pearson_upper, ljung_box_p, mcnemar_exact_p


def test_cp_upper_zero_failures_matches_closed_form():
    # master doc §2.9 G5: k=0, n=300 -> 1 - 0.05**(1/300) ≈ 0.99%
    assert clopper_pearson_upper(0, 300) == pytest.approx(1 - 0.05 ** (1 / 300), rel=1e-9)


def test_g5_tolerates_one_failure_in_300_but_not_two():
    assert clopper_pearson_upper(1, 300) <= 0.02
    assert clopper_pearson_upper(2, 300) > 0.02


def test_cp_upper_all_failures_is_one():
    assert clopper_pearson_upper(10, 10) == 1.0


def test_cp_lower_all_caught_matches_closed_form():
    # H3: catch = n/n -> lower bound 0.05**(1/n)
    assert clopper_pearson_lower(50, 50) == pytest.approx(0.05 ** (1 / 50), rel=1e-9)


def test_cp_rejects_bad_counts():
    with pytest.raises(ValueError):
        clopper_pearson_upper(5, 3)
    with pytest.raises(ValueError):
        clopper_pearson_upper(0, 0)


def test_ljung_box_white_noise_passes():
    r = np.random.default_rng(0).standard_normal(500)
    assert ljung_box_p(r) > 0.05


def test_ljung_box_correlated_residual_fails():
    rng = np.random.default_rng(0)
    r = np.zeros(500)
    for k in range(1, 500):
        r[k] = 0.8 * r[k - 1] + rng.standard_normal()
    assert ljung_box_p(r) < 1e-6


def test_mcnemar_exact_one_sided():
    # b=10 tuned-only wins, c=0 base-only wins -> p = 0.5**10
    assert mcnemar_exact_p(10, 0) == pytest.approx(0.5 ** 10)
    assert mcnemar_exact_p(0, 0) == 1.0
    assert mcnemar_exact_p(5, 5) > 0.5

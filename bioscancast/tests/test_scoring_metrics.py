import pytest

from bioscancast.stages.evaluation.scoring import (
    normalized_entropy,
    ranked_probability_score,
    top_probability,
    true_probability,
)


def test_true_probability_returns_mass_on_resolved_bucket():
    assert true_probability([0.2, 0.3, 0.5], 2) == pytest.approx(0.5)
    assert true_probability([0.2, 0.3, 0.5], 0) == pytest.approx(0.2)


def test_true_probability_normalizes_unnormalized_input():
    # Values summing to 10 should be normalized before the mass is read off.
    assert true_probability([2, 3, 5], 1) == pytest.approx(0.3)


def test_top_probability_is_the_largest_bucket():
    assert top_probability([0.2, 0.3, 0.5]) == pytest.approx(0.5)
    assert top_probability([0.1, 0.1, 0.1, 0.7]) == pytest.approx(0.7)


def test_normalized_entropy_uniform_is_one():
    assert normalized_entropy([0.25, 0.25, 0.25, 0.25]) == pytest.approx(1.0)


def test_normalized_entropy_one_hot_is_zero():
    assert normalized_entropy([1.0, 0.0, 0.0]) == pytest.approx(0.0)


def test_normalized_entropy_single_bucket_is_zero():
    assert normalized_entropy([1.0]) == 0.0


def test_rps_perfect_forecast_is_zero():
    assert ranked_probability_score([0.0, 0.0, 1.0], 2) == pytest.approx(0.0)


def test_rps_worst_forecast_is_one():
    # All mass on the bucket farthest from the resolved one.
    assert ranked_probability_score([1.0, 0.0, 0.0], 2) == pytest.approx(1.0)


def test_rps_rewards_mass_closer_to_the_true_bucket():
    # Ordinal: mass on the adjacent bucket beats mass on the far bucket.
    adjacent = ranked_probability_score([0.0, 1.0, 0.0], 2)
    far = ranked_probability_score([1.0, 0.0, 0.0], 2)
    assert adjacent < far
    assert adjacent == pytest.approx(0.5)


def test_rps_out_of_bounds_raises():
    with pytest.raises(IndexError):
        ranked_probability_score([0.5, 0.5], 2)

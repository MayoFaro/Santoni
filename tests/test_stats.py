import pytest

from santoni_az.stats import clopper_pearson_lower_bound


def test_zero_successes_gives_zero_lower_bound():
    assert clopper_pearson_lower_bound(0, 40) == 0.0


def test_all_successes_matches_the_closed_form_power_law():
    # Clopper-Pearson's k==n special case solves p**n == alpha/2 directly.
    expected = 0.025 ** (1 / 40)
    assert clopper_pearson_lower_bound(40, 40) == pytest.approx(expected, abs=1e-9)


def test_lower_bound_increases_monotonically_with_more_successes_at_fixed_n():
    low = clopper_pearson_lower_bound(30, 40)
    mid = clopper_pearson_lower_bound(35, 40)
    high = clopper_pearson_lower_bound(39, 40)
    assert low < mid < high


def test_works_below_the_eventual_40_game_window():
    # n < 40 must not crash or be treated as a full window.
    assert 0.0 <= clopper_pearson_lower_bound(3, 5) < 1.0


def test_rejects_successes_greater_than_n():
    with pytest.raises(ValueError):
        clopper_pearson_lower_bound(5, 4)

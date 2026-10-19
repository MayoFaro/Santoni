from santorini.engine import Position
from tools.proven_anchor.symmetry import BOARD_SYMMETRIES


def a_real_no_power_position():
    heights = [0] * 25
    heights[6] = 1
    heights[18] = 2
    return Position(heights=tuple(heights), workers=((0, 4), (20, 24)))


def test_there_are_eight_symmetries():
    assert len(BOARD_SYMMETRIES) == 8


def test_every_symmetry_produces_a_valid_position():
    position = a_real_no_power_position()
    for transform in BOARD_SYMMETRIES:
        transformed = transform(position)
        transformed.validate()  # raises if invalid


def test_applying_a_quarter_turn_four_times_returns_to_the_original():
    position = a_real_no_power_position()
    quarter_turn = next(t for t in BOARD_SYMMETRIES if t is not BOARD_SYMMETRIES[0])
    result = position
    for _ in range(4):
        result = quarter_turn(result)
    assert result == position


def test_symmetries_preserve_non_board_fields():
    position = a_real_no_power_position()
    for transform in BOARD_SYMMETRIES:
        transformed = transform(position)
        assert transformed.player == position.player
        assert transformed.powers == position.powers
        assert transformed.winner == position.winner


def test_symmetries_are_distinct_on_an_asymmetric_position():
    # A position with no board symmetry of its own: all 8 transformed
    # results must differ pairwise, otherwise two "symmetries" collapsed
    # to the same transform (a bug, not a property of this position).
    position = a_real_no_power_position()
    results = [transform(position) for transform in BOARD_SYMMETRIES]
    assert len({r for r in results}) == 8

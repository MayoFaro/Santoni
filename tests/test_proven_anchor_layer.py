import pytest
from santorini.engine import Position
from santorini.native import native_turns
from tools.proven_anchor.layer import build_layer

def a_position_with_one_winning_move_among_several():
    heights = [0] * 25
    heights[0] = 2
    heights[1] = 3  # worker 0 can climb here and win
    position = Position(heights=tuple(heights), workers=((0, 4), (20, 24)), player=0)
    position.validate()
    return position

def test_build_layer_marks_the_ancestor_as_a_proven_win_when_a_winning_child_exists():
    position = a_position_with_one_winning_move_among_several()

    def fake_evaluate(child, mover, seconds):
        return child.winner == mover if child.winner is not None else False

    result = build_layer(position, seconds=0.1, evaluate=fake_evaluate)

    assert result.ancestor_verdict is True
    assert result.resolved_fraction == 1.0
    assert any(verdict is True for _, verdict in result.children)

def test_build_layer_marks_the_ancestor_as_a_proven_loss_when_all_children_lose():
    position = a_position_with_one_winning_move_among_several()

    def all_losing(child, mover, seconds):
        return False

    result = build_layer(position, seconds=0.1, evaluate=all_losing)

    assert result.ancestor_verdict is False
    assert all(verdict is False for _, verdict in result.children)

def test_build_layer_reports_undetermined_when_nothing_resolves(monkeypatch):
    position = a_position_with_one_winning_move_among_several()

    def unresolved(child, mover, seconds):
        return None

    result = build_layer(position, seconds=0.1, evaluate=unresolved)

    assert result.ancestor_verdict is None
    assert result.resolved_fraction == 0.0

def test_build_layer_computes_resolved_fraction_with_partial_resolution():
    position = a_position_with_one_winning_move_among_several()
    turns = native_turns(position)
    half = len(turns) // 2

    def half_resolved(child, mover, seconds):
        # Deterministic split by whether this child is among the first half
        # of native_turns' own order -- exact values don't matter, only
        # that build_layer iterates the same `native_turns` order it was
        # given (asserted indirectly via the exact fraction below).
        return False if turns.index(next(t for t in turns if t.after == child)) < half else None

    result = build_layer(position, seconds=0.1, evaluate=half_resolved)
    assert result.resolved_fraction == pytest.approx(half / len(turns))

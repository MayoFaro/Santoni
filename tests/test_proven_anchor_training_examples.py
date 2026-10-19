import pytest

from santorini.engine import Position
from santorini.native import native_turns
from tools.proven_anchor.layer import LayerResult
from tools.proven_anchor.training_examples import action_index, examples_from_layer


def test_action_index_matches_the_known_rust_formula_for_a_non_winning_move():
    position = Position(workers=((0, 4), (20, 24)))
    turns = native_turns(position)
    # worker 0 (slot 0) sits at cell 0; pick any legal turn moving it.
    turn = next(t for t in turns if t.actions[0].source == 0)
    mv, bld = turn.actions[0], turn.actions[1]
    expected = 0 * 625 + mv.target * 25 + bld.target
    assert action_index(position, turn) == expected


def test_action_index_uses_the_move_target_as_build_for_a_winning_turn():
    heights = [0] * 25
    heights[0] = 2
    heights[1] = 3
    position = Position(heights=tuple(heights), workers=((0, 4), (20, 24)), player=0)
    turns = native_turns(position)
    winning_turn = next(t for t in turns if t.after.winner is not None)
    mv = winning_turn.actions[0]
    assert len(winning_turn.actions) == 1
    expected = 0 * 625 + mv.target * 25 + mv.target  # build == dest for a synthesized placeholder
    assert action_index(position, winning_turn) == expected


def test_examples_from_layer_builds_a_policy_target_over_winning_moves_only():
    ancestor = Position(workers=((0, 4), (20, 24)))
    turns = native_turns(ancestor)
    children = [(t, True) if i == 0 else (t, False) for i, t in enumerate(turns)]
    layer = LayerResult(ancestor=ancestor, children=children,
                        resolved_fraction=1.0, ancestor_verdict=True)

    examples = examples_from_layer(layer)

    assert len(examples) == 1
    planes, policy, value = examples[0]
    assert value == 1.0
    assert policy.sum().item() == pytest.approx(1.0)
    winning_index = action_index(ancestor, turns[0])
    assert policy[winning_index].item() == pytest.approx(1.0)
    assert (policy > 0).sum().item() == 1


def test_examples_from_layer_skips_a_layer_with_no_winning_move_found_yet():
    ancestor = Position(workers=((0, 4), (20, 24)))
    layer = LayerResult(ancestor=ancestor, children=[], resolved_fraction=0.0, ancestor_verdict=None)
    assert examples_from_layer(layer) == []

import pytest
import torch
import torch.nn.functional as F

from santorini.engine import Position
from santorini.native import native_turns
from santoni_az.network import ACTION_SPACE, NUM_PLANES
from tools.proven_anchor.layer import LayerResult
from tools.proven_anchor.planes import compute_planes
from tools.proven_anchor.training_examples import (
    action_index,
    child_examples_from_layer,
    examples_from_layer,
)


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


def _layer_with_mixed_child_verdicts():
    ancestor = Position(workers=((0, 4), (20, 24)))
    turns = native_turns(ancestor)
    # One proven win for the ancestor's mover, one proven loss, one still
    # unresolved -- the three cases `build_layer` can produce.
    children = [(turns[0], True), (turns[1], False), (turns[2], None)]
    return LayerResult(ancestor=ancestor, children=children,
                       resolved_fraction=2 / 3, ancestor_verdict=True)


def test_child_examples_flip_the_verdict_into_the_childs_own_perspective():
    """A child's verdict is "a proven win for the *ancestor's* mover", and
    the child's own player to move is that mover's opponent -- so a `True`
    verdict is a value target of -1.0 on those planes, not +1.0."""
    layer = _layer_with_mixed_child_verdicts()

    examples = child_examples_from_layer(layer)

    assert len(examples) == 2  # the unresolved child is skipped, never read as a loss
    values = [value for _, _, value in examples]
    assert values == [-1.0, 1.0]
    planes_for_win, _, _ = examples[0]
    expected = torch.frombuffer(
        bytearray(compute_planes(layer.children[0][0].after)), dtype=torch.uint8
    ).reshape(NUM_PLANES, 5, 5).float()
    assert torch.equal(planes_for_win, expected)


def test_child_examples_carry_no_policy_supervision():
    """An all-zero probability target contributes exactly zero policy loss
    and zero policy gradient under `F.cross_entropy` -- these examples
    deliberately supervise the value head only, because the child's own
    winning moves are not known at this layer."""
    layer = _layer_with_mixed_child_verdicts()

    examples = child_examples_from_layer(layer)

    for _, policy, _ in examples:
        assert policy.shape == (ACTION_SPACE,)
        assert float(policy.sum()) == 0.0

    logits = torch.randn(1, ACTION_SPACE, requires_grad=True)
    loss = F.cross_entropy(logits, examples[0][1].unsqueeze(0))
    loss.backward()
    assert float(loss.detach()) == 0.0
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_child_examples_skip_already_terminal_children():
    heights = [0] * 25
    heights[0] = 2
    heights[1] = 3
    ancestor = Position(heights=tuple(heights), workers=((0, 4), (20, 24)), player=0)
    turns = native_turns(ancestor)
    winning_turn = next(t for t in turns if t.after.winner is not None)
    other_turn = next(t for t in turns if t.after.winner is None)
    layer = LayerResult(ancestor=ancestor,
                        children=[(winning_turn, True), (other_turn, False)],
                        resolved_fraction=1.0, ancestor_verdict=True)

    examples = child_examples_from_layer(layer)

    # The already-decided child is trivially classifiable from its planes
    # and is never evaluated by the network at inference time.
    assert len(examples) == 1
    assert examples[0][2] == 1.0


def test_child_examples_honour_an_explicit_children_subset():
    layer = _layer_with_mixed_child_verdicts()
    only_the_loss = [layer.children[1]]

    examples = child_examples_from_layer(layer, children=only_the_loss)

    assert len(examples) == 1
    assert examples[0][2] == 1.0

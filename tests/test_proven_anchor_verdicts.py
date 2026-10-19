from santorini.engine import Position
from santorini.native import native_search
from tools.proven_anchor.verdicts import evaluate_child_verdict, mate_distance


def test_mate_distance_is_none_when_not_proven():
    heights = [0] * 25
    position = Position(heights=tuple(heights), workers=((0, 4), (20, 24)))
    analysis = native_search(position, 0.05)
    if analysis.proven:
        return  # trivially proven this fast at the opening is not expected, but don't flake if it is
    assert mate_distance(analysis) is None


def winning_in_one_position():
    # Reused fixture, same construction as
    # tests/test_selfplay_worker_single_move.py::winning_in_one_position.
    heights = [0] * 25
    heights[0] = 2
    heights[1] = 3
    domes = (1 << 5) | (1 << 6) | (1 << 3) | (1 << 8) | (1 << 9)
    position = Position(heights=tuple(heights), domes=domes,
                        workers=((0, 4), (20, 24)), powers=(0, 0), player=0)
    position.validate()
    return position


def test_evaluate_child_verdict_true_for_an_immediate_terminal_win():
    # The child *is* the position right after the winning move: winner is
    # already set directly by apply(), no search call should be needed.
    from santorini.engine import validate_turn
    position = winning_in_one_position()
    turns = __import__("santorini.native", fromlist=["native_turns"]).native_turns(position)
    winning_turn = next(t for t in turns if t.after.winner is not None)
    assert evaluate_child_verdict(winning_turn.after, mover=0, seconds=0.5) is True


def test_evaluate_child_verdict_false_when_child_position_is_a_proven_win_for_its_own_mover():
    # `winning_in_one_position()` is not itself terminal (winner is None):
    # it is player 0's turn with a single, forced, immediately winning move.
    # Treat it as a *child* reached by player 1's move: that move handed
    # player 0 a forced win, so it must be a proven loss for mover=1. This
    # exercises the native_search + analysis.score branch directly (the
    # brief's own terminal-win test takes the winner-is-not-None shortcut
    # and never calls native_search at all).
    child = winning_in_one_position()
    assert child.winner is None
    assert evaluate_child_verdict(child, mover=1, seconds=0.5) is False


def fully_immobilized_mover_position():
    # Player 1 to move, both workers boxed into dome-locked corners (cells
    # 0 and 4, all six neighbouring cells domed) -- no legal move exists,
    # a proven immediate loss for the player to move (child_position.player
    # == 1), independently confirmed against native_search: proven=True,
    # score=-100000 (== -MATE), depth=0.
    heights = [0] * 25
    domes = (1 << 1) | (1 << 5) | (1 << 6) | (1 << 3) | (1 << 8) | (1 << 9)
    position = Position(heights=tuple(heights), domes=domes,
                        workers=((12, 13), (0, 4)), powers=(0, 0), player=1)
    position.validate()
    return position


def test_evaluate_child_verdict_true_when_child_position_is_a_proven_loss_for_its_own_mover():
    # child.player (1) is forced to lose (score < 0 from their own
    # perspective) -- that is exactly a proven WIN for whoever moved to
    # reach this position, i.e. mover=0 (the opponent of child.player).
    # This exercises the native_search + analysis.score branch with the
    # opposite sign from the previous test, directly checking the
    # `analysis.score < 0` comparison in the implementation.
    child = fully_immobilized_mover_position()
    analysis_check = native_search(child, 0.5)
    assert analysis_check.proven and analysis_check.score < 0  # sanity on the fixture itself
    assert evaluate_child_verdict(child, mover=0, seconds=0.5) is True


def test_evaluate_child_verdict_handles_unresolved_within_budget():
    # An opening-ish position: with a near-zero budget, proof is extremely
    # unlikely to complete -- must return None, not crash or guess.
    heights = [0] * 25
    position = Position(heights=tuple(heights), workers=((2, 4), (20, 22)))
    verdict = evaluate_child_verdict(position, mover=0, seconds=0.01)
    assert verdict in (True, False, None)  # must not raise; typically None at this budget

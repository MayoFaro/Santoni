from dataclasses import replace

import pytest

from santorini.engine import Position, validate_setup
from santorini.placement import choose_placement, legal_pairs, placement_first
from santorini.search import Interrupted


def test_placement_order_tracks_first_player_with_bia_override():
    assert placement_first((0, 0), 0) == 0
    assert placement_first((0, 0), 1) == 1
    assert placement_first((13, 0), 1) == 0
    assert placement_first((0, 13), 0) == 1


def test_legal_pairs_preserve_selene_identity_and_eros_edges():
    assert len(list(legal_pairs(0))) == 300
    assert len(list(legal_pairs(28))) == 600
    assert (6, 8) in list(legal_pairs(28))
    assert (8, 6) in list(legal_pairs(28))
    assert all(a % 5 == 0 and b % 5 == 4 or b % 5 == 0 and a % 5 == 4 or
               a // 5 == 0 and b // 5 == 4 or b // 5 == 0 and a // 5 == 4
               for a, b in legal_pairs(19))
    assert all(c not in (6, 8) for pair in legal_pairs(0, (6, 8)) for c in pair)


def test_second_robot_needs_real_opponent_placement():
    with pytest.raises(ValueError, match="d'abord"):
        choose_placement((0, 0), 0, 1)
    with pytest.raises(ValueError, match="distinctes"):
        choose_placement((0, 0), 0, 1, (6, 6))
    with pytest.raises(ValueError, match="Eros"):
        choose_placement((19, 0), 0, 1, (6, 8))


@pytest.mark.parametrize("powers,first,human", [((0, 0), 0, (6, 8)), ((0, 19), 0, (6, 8)),
                                              ((28, 4), 0, (6, 8)), ((13, 0), 1, (6, 8)),
                                              ((0, 13), 0, None)])
def test_advice_valid_placement_deadline_and_progress(powers, first, human):
    progress = []
    advice = choose_placement(powers, first, .2, human, publish=progress.append)
    assert progress and progress[0].robot_cells
    assert advice.elapsed < .6
    validate_setup((advice.opponent_cells, advice.robot_cells), powers, first)
    if advice.first_to_place == 0:
        assert advice.opponent_cells == human
        assert set(advice.robot_cells).isdisjoint(human)


def test_first_robot_ignores_future_human_cells(monkeypatch):
    import santorini.placement as module
    # Deterministic score removes time/search variations from the comparison.
    monkeypatch.setattr(module, "_probe", lambda pos, seconds, cancelled: module.evaluate(pos, 1))
    a = choose_placement((0, 0), 1, 1, (0, 24))
    b = choose_placement((0, 0), 1, 1, (6, 8))
    assert a.robot_cells == b.robot_cells
    assert a.score == b.score
    assert a.opponent_cells == b.opponent_cells


def test_game_search_receives_both_powers_and_game_starter(monkeypatch):
    import santorini.placement as module
    recorded = []
    def probe(pos, seconds, cancelled):
        recorded.append(pos)
        return -abs(pos.workers[1][0] - 12)
    monkeypatch.setattr(module, "_probe", probe)
    result = choose_placement((13, 4), 1, 1, (6, 8))
    assert result.first_to_place == 0  # Bia places first but robot plays first.
    assert recorded
    assert all(s.powers == (13, 4) and s.player == 1 and s.workers[0] == (6, 8) for s in recorded)
    assert result.refined


def test_cancellation_stops_placement():
    with pytest.raises(Interrupted):
        choose_placement((0, 0), 1, 1, cancelled=lambda: True)


def test_no_game_score_is_not_reported_as_power_aware_search(monkeypatch):
    import santorini.placement as module
    monkeypatch.setattr(module, "_probe", lambda *args: None)
    advice = choose_placement((0, 4), 0, .2, (6, 8))
    assert not advice.refined
    assert "géométriquement" in advice.status
    validate_setup(((6, 8), advice.robot_cells), (0, 4), 0)

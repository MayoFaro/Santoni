import pytest

from santorini.engine import Position
from tools.proven_anchor.anchor_games import (
    find_anchor_games,
    play_reference_vs_reference,
    scan_game_for_forced_win,
)


def test_scan_game_for_forced_win_finds_a_known_mate_in_two(monkeypatch):
    # Build a short synthetic history where position[-1] is a known mate-in-2
    # fixture is impractical to hand-construct exactly, so this test injects
    # a fake analysis function instead of relying on a real deep proof.
    import tools.proven_anchor.anchor_games as mod

    positions = [Position(workers=((0, 4), (20, 24))) for _ in range(5)]

    def fake_search(position, seconds):
        class FakeAnalysis:
            proven = position is positions[2]
            score = 100_000 - 2 if proven else 0
        return FakeAnalysis()

    monkeypatch.setattr(mod, "native_search", fake_search)

    index = scan_game_for_forced_win(positions, target_distance=2, seconds=0.1)
    assert index == 2


def test_scan_game_for_forced_win_returns_none_when_absent(monkeypatch):
    import tools.proven_anchor.anchor_games as mod
    positions = [Position(workers=((0, 4), (20, 24))) for _ in range(3)]

    def never_proven(position, seconds):
        class FakeAnalysis:
            proven = False
            score = 0
        return FakeAnalysis()

    monkeypatch.setattr(mod, "native_search", never_proven)

    assert scan_game_for_forced_win(positions, target_distance=6, seconds=0.1) is None


def boxed_in_position(stuck_player):
    """Position of the no-power subgame where `stuck_player` has no legal
    move at all: both their workers sit at (12, 13), surrounded by domes on
    every other square. Same construction as
    `tests/test_mcts_az_vs_reference_campaign.py::boxed_in_position`."""
    domes = ((1 << 25) - 1) & ~((1 << 12) | (1 << 13))
    workers = ((12, 13), (0, 1)) if stuck_player == 0 else ((0, 1), (12, 13))
    return Position(workers=workers, domes=domes, player=stuck_player, powers=(0, 0))


def test_play_reference_vs_reference_stops_immediately_on_an_already_terminal_opening(monkeypatch):
    import tools.proven_anchor.anchor_games as mod

    stuck = boxed_in_position(stuck_player=1)
    monkeypatch.setattr(mod, 'validate_setup', lambda *a, **k: stuck)

    def boom(*args, **kwargs):
        raise AssertionError('must not be called on an already-terminal position')
    monkeypatch.setattr(mod, 'reference_move', boom)

    opening_job = {'workers': [[0, 1], [12, 13]], 'powers': [0, 0], 'first': 1}
    positions = play_reference_vs_reference(opening_job, seconds=0.1, reference_binary='unused')

    assert len(positions) == 1
    assert positions[0].winner == 0  # stuck_player 1 has no legal move -> player 0 wins


def test_play_reference_vs_reference_records_full_history_until_terminal(monkeypatch):
    import tools.proven_anchor.anchor_games as mod

    start = Position(workers=((0, 4), (20, 24)), powers=(0, 0), player=0)
    after_one = Position(workers=((1, 4), (20, 24)), powers=(0, 0), player=1)
    stuck_after_two = boxed_in_position(stuck_player=0)

    monkeypatch.setattr(mod, 'validate_setup', lambda *a, **k: start)

    calls = {'n': 0}

    def fake_reference_move(position, seconds, reference_binary):
        calls['n'] += 1
        assert reference_binary == 'frozen-ref'

        class FakeTurn:
            after = after_one if calls['n'] == 1 else stuck_after_two
        return FakeTurn(), 0.01

    monkeypatch.setattr(mod, 'reference_move', fake_reference_move)

    opening_job = {'workers': [[0, 4], [20, 24]], 'powers': [0, 0], 'first': 0}
    positions = play_reference_vs_reference(opening_job, seconds=0.1, reference_binary='frozen-ref')

    # Starting position included, both plies recorded, stops right at the
    # terminal position -- `reference_move` is never called a third time.
    assert positions[0] is start
    assert positions[1] is after_one
    assert len(positions) == 3
    assert positions[2].winner == 1  # stuck_player 0 has no legal move -> player 1 wins
    assert calls['n'] == 2


def test_find_anchor_games_stops_once_minimum_found(monkeypatch):
    import tools.proven_anchor.anchor_games as mod

    played = []

    def fake_play(opening_job, seconds, reference_binary):
        played.append(opening_job['id'])
        return [f"pos-{opening_job['id']}"]

    def fake_scan(positions, target_distance, seconds):
        return 0  # every opening "qualifies" immediately

    monkeypatch.setattr(mod, 'play_reference_vs_reference', fake_play)
    monkeypatch.setattr(mod, 'scan_game_for_forced_win', fake_scan)

    openings = [{'id': i} for i in range(5)]
    result = find_anchor_games(openings, target_distance=2, seconds=0.1,
                                reference_binary='unused', minimum=2)

    assert len(result) == 2
    assert result == [(['pos-0'], 0), (['pos-1'], 0)]
    # Stopped as soon as 2 anchors were found -- openings 2, 3, 4 never played.
    assert played == [0, 1]


def test_find_anchor_games_raises_when_openings_exhausted_without_reaching_minimum(monkeypatch):
    import tools.proven_anchor.anchor_games as mod

    played = []

    def fake_play(opening_job, seconds, reference_binary):
        played.append(opening_job['id'])
        return ['only-position']

    def fake_scan(positions, target_distance, seconds):
        return None  # never qualifies

    monkeypatch.setattr(mod, 'play_reference_vs_reference', fake_play)
    monkeypatch.setattr(mod, 'scan_game_for_forced_win', fake_scan)

    openings = [{'id': i} for i in range(3)]
    with pytest.raises(ValueError):
        find_anchor_games(openings, target_distance=2, seconds=0.1,
                           reference_binary='unused', minimum=2)

    # Every opening was tried before giving up.
    assert played == [0, 1, 2]

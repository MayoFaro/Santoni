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


def test_scan_game_for_forced_win_rejects_a_proven_loss_at_the_target_distance(monkeypatch):
    """A forced *loss* in N scores `-(MATE - N)`, which `MATE - abs(score)`
    cannot tell apart from a forced win in N. Spec Sec 3.1 requires an
    anchor position whose player to move *wins*, so a proven loss at the
    same distance must not qualify."""
    import tools.proven_anchor.anchor_games as mod

    positions = [Position(workers=((0, 4), (20, 24))) for _ in range(3)]

    def losing_proof(position, seconds):
        class FakeAnalysis:
            proven = True
            score = -(100_000 - 6)  # forced loss in 6 for the player to move
        return FakeAnalysis()

    monkeypatch.setattr(mod, "native_search", losing_proof)

    assert scan_game_for_forced_win(positions, target_distance=6, seconds=0.1) is None


def test_scan_game_for_forced_win_accepts_a_proven_win_at_the_target_distance(monkeypatch):
    import tools.proven_anchor.anchor_games as mod

    positions = [Position(workers=((0, 4), (20, 24))) for _ in range(3)]

    def winning_proof(position, seconds):
        class FakeAnalysis:
            proven = True
            score = 100_000 - 6
        return FakeAnalysis()

    monkeypatch.setattr(mod, "native_search", winning_proof)

    assert scan_game_for_forced_win(positions, target_distance=6, seconds=0.1) == 0


def test_find_anchor_games_rejects_a_second_game_with_an_identical_qualifying_position(monkeypatch):
    """Plan Review Focus: two anchor games whose scanned positions turn out
    identical would hand the curriculum the same ancestor twice under two
    names, so one game's "held-out" children would be the other's own
    training material."""
    import tools.proven_anchor.anchor_games as mod

    histories = {
        0: ['shared-qualifying-position'],
        1: ['shared-qualifying-position'],  # same position, different opening
        2: ['genuinely-different-position'],
    }
    monkeypatch.setattr(mod, 'play_reference_vs_reference',
                        lambda job, seconds, binary, progress=None: histories[job['id']])
    monkeypatch.setattr(mod, 'scan_game_for_forced_win', lambda positions, d, s, progress=None: 0)

    result = find_anchor_games([{'id': i} for i in range(3)], target_distance=2,
                                seconds=0.1, reference_binary='unused', minimum=2)

    assert result == [(['shared-qualifying-position'], 0),
                      (['genuinely-different-position'], 0)]


def boxed_in_position(stuck_player):
    """Position of the no-power subgame where `stuck_player` has no legal
    move at all: both their workers sit at (12, 13), surrounded by domes on
    every other square, so `santorini.engine.terminal_position` resolves it
    as an immediate loss for `stuck_player` without needing a real search."""
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

    def fake_play(opening_job, seconds, reference_binary, progress=None):
        played.append(opening_job['id'])
        return [f"pos-{opening_job['id']}"]

    def fake_scan(positions, target_distance, seconds, progress=None):
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

    def fake_play(opening_job, seconds, reference_binary, progress=None):
        played.append(opening_job['id'])
        return ['only-position']

    def fake_scan(positions, target_distance, seconds, progress=None):
        return None  # never qualifies

    monkeypatch.setattr(mod, 'play_reference_vs_reference', fake_play)
    monkeypatch.setattr(mod, 'scan_game_for_forced_win', fake_scan)

    openings = [{'id': i} for i in range(3)]
    with pytest.raises(ValueError):
        find_anchor_games(openings, target_distance=2, seconds=0.1,
                           reference_binary='unused', minimum=2)

    # Every opening was tried before giving up.
    assert played == [0, 1, 2]


def test_scan_game_for_forced_win_reports_progress_per_ply(monkeypatch):
    """The 2026-10-21 pending finding: this loop had no progress visibility
    at all, and a real proof search can spend up to `seconds` per ply --
    so a caller watching a long run needs to see which ply is being
    searched, not just silence until the whole game is done."""
    import tools.proven_anchor.anchor_games as mod

    positions = [Position(workers=((0, 4), (20, 24))) for _ in range(3)]

    def never_proven(position, seconds):
        class FakeAnalysis:
            proven = False
            score = 0
        return FakeAnalysis()

    monkeypatch.setattr(mod, "native_search", never_proven)

    messages = []
    scan_game_for_forced_win(positions, target_distance=6, seconds=0.1,
                              progress=messages.append)

    assert len(messages) == 3  # one report per position scanned
    assert "1/3" in messages[0]
    assert "3/3" in messages[-1]


def test_scan_game_for_forced_win_default_progress_does_not_raise(monkeypatch):
    """No `progress=` passed at all (every pre-existing call site, and
    every other test in this file) must keep working -- the default
    reports to stdout rather than requiring every caller to opt in."""
    import tools.proven_anchor.anchor_games as mod

    positions = [Position(workers=((0, 4), (20, 24)))]

    def never_proven(position, seconds):
        class FakeAnalysis:
            proven = False
            score = 0
        return FakeAnalysis()

    monkeypatch.setattr(mod, "native_search", never_proven)

    assert scan_game_for_forced_win(positions, target_distance=6, seconds=0.1) is None


def test_play_reference_vs_reference_reports_progress_per_move(monkeypatch):
    import tools.proven_anchor.anchor_games as mod

    start = Position(workers=((0, 4), (20, 24)), powers=(0, 0), player=0)
    stuck_after_one = boxed_in_position(stuck_player=1)

    monkeypatch.setattr(mod, 'validate_setup', lambda *a, **k: start)

    def fake_reference_move(position, seconds, reference_binary):
        class FakeTurn:
            after = stuck_after_one
        return FakeTurn(), 12.5

    monkeypatch.setattr(mod, 'reference_move', fake_reference_move)

    messages = []
    opening_job = {'workers': [[0, 4], [20, 24]], 'powers': [0, 0], 'first': 0}
    play_reference_vs_reference(opening_job, seconds=0.1, reference_binary='frozen-ref',
                                 progress=messages.append)

    assert any('12.5' in message for message in messages)


def test_find_anchor_games_reports_progress_per_opening_and_forwards_to_sub_calls(monkeypatch):
    """Each opening's own play/scan progress must still surface (prefixed
    so a long run's log line can be told apart by opening), alongside a
    summary line per opening of how many anchors have been found so far."""
    import tools.proven_anchor.anchor_games as mod

    def fake_play(opening_job, seconds, reference_binary, progress=None):
        if progress is not None:
            progress('inner-play')
        return [f"pos-{opening_job['id']}"]

    def fake_scan(positions, target_distance, seconds, progress=None):
        if progress is not None:
            progress('inner-scan')
        return 0 if positions == ['pos-1'] else None

    monkeypatch.setattr(mod, 'play_reference_vs_reference', fake_play)
    monkeypatch.setattr(mod, 'scan_game_for_forced_win', fake_scan)

    messages = []
    openings = [{'id': i} for i in range(2)]
    result = find_anchor_games(openings, target_distance=2, seconds=0.1,
                                reference_binary='unused', minimum=1,
                                progress=messages.append)

    assert result == [(['pos-1'], 0)]
    joined = '\n'.join(messages)
    assert 'inner-play' in joined
    assert 'inner-scan' in joined
    assert '1/2' in joined  # opening 1 of 2 openings
    assert '0/1' in joined  # 0 anchors found so far when opening 0 was tried
    assert any('1/1' in message for message in messages[-2:])  # anchor found, 1/1 needed

import json

import pytest

from santorini.engine import Position
from santorini.native import CState, decode
from tools.anchor_pilot_campaign import prepare_anchor


def fake_game(count, first_player=0):
    """A list of `count` distinct positions, player alternating per ply, so
    a wrong index picks a position with the wrong `player` too."""
    return [Position(workers=((i % 24, (i + 1) % 24), (20, 21)), player=(first_player + i) % 2)
            for i in range(count)]


def test_prepare_anchor_writes_the_position_back_plies_before_the_win(tmp_path, monkeypatch):
    import tools.anchor_pilot_campaign as mod

    positions = fake_game(10)

    def fake_play(opening_job, seconds, reference_binary, progress=None):
        return positions

    monkeypatch.setattr(mod, 'play_reference_vs_reference', fake_play)

    opening_job = {'workers': [[0, 1], [20, 21]], 'powers': [0, 0], 'first': 0}
    path = prepare_anchor(tmp_path, opening_job, back_plies=5, seconds=0.1,
                          reference_binary='frozen-ref')

    assert path == tmp_path / 'anchor_state.bin'
    raw = path.read_bytes()
    state = CState.from_buffer_copy(raw)
    decoded = decode(state)
    assert decoded == positions[10 - 1 - 5]


def test_prepare_anchor_is_idempotent_on_a_second_call(tmp_path, monkeypatch):
    import tools.anchor_pilot_campaign as mod

    positions = fake_game(10)
    calls = {'n': 0}

    def fake_play(opening_job, seconds, reference_binary, progress=None):
        calls['n'] += 1
        return positions

    monkeypatch.setattr(mod, 'play_reference_vs_reference', fake_play)

    opening_job = {'workers': [[0, 1], [20, 21]], 'powers': [0, 0], 'first': 0}
    first_path = prepare_anchor(tmp_path, opening_job, back_plies=5, seconds=0.1,
                                reference_binary='frozen-ref')
    assert calls['n'] == 1

    def boom(*args, **kwargs):
        raise AssertionError('must not be called on a second, already-prepared call')

    monkeypatch.setattr(mod, 'play_reference_vs_reference', boom)

    second_path = prepare_anchor(tmp_path, opening_job, back_plies=5, seconds=0.1,
                                 reference_binary='frozen-ref')

    assert second_path == first_path
    assert calls['n'] == 1


def test_prepare_anchor_raises_a_clear_error_when_the_seed_game_is_too_short(tmp_path, monkeypatch):
    import tools.anchor_pilot_campaign as mod

    positions = fake_game(3)

    def fake_play(opening_job, seconds, reference_binary, progress=None):
        return positions

    monkeypatch.setattr(mod, 'play_reference_vs_reference', fake_play)

    opening_job = {'workers': [[0, 1], [20, 21]], 'powers': [0, 0], 'first': 0}
    with pytest.raises(ValueError, match='3 plies'):
        prepare_anchor(tmp_path, opening_job, back_plies=5, seconds=0.1,
                       reference_binary='frozen-ref')

    with pytest.raises(ValueError, match='need at least 6'):
        prepare_anchor(tmp_path, opening_job, back_plies=5, seconds=0.1,
                       reference_binary='frozen-ref')

    assert not (tmp_path / 'anchor_state.bin').exists()

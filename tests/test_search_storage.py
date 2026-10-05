import json
import time
from dataclasses import replace

import pytest

from santorini.engine import Position, legal_turns, validate_turn
from santorini.search import choose_power, search
from santorini.storage import Session, Store


def test_search_deadline_complete_move_and_immediate_win():
    s = Position()
    start = time.monotonic()
    result = search(s, .1)
    assert time.monotonic() - start < .5
    assert result.turn is not None
    assert validate_turn(s, result.turn.actions) == result.turn
    hs = list(s.heights)
    hs[6], hs[7] = 2, 3
    winning = replace(s, heights=tuple(hs))
    result = search(winning, .2)
    assert result.proven
    assert result.turn.after.winner == 0


def test_search_defends_immediate_threat():
    # Opponent on C4 threatens C3; a dome in C3 is the only defence.
    hs = [0] * 25
    hs[17], hs[12] = 2, 3
    s = Position(tuple(hs), workers=((11, 0), (17, 24)))
    result = search(s, 1)
    assert result.turn
    assert result.turn.after.domes & (1 << 12)


def test_partial_losing_root_does_not_claim_forced_defeat(monkeypatch):
    from santorini.engine import Turn
    import santorini.search as module
    root = Position()
    bad = Turn((), replace(root, player=1, winner=1, reason="Défaite"))
    interrupted = [False]
    def moves(pos, check=lambda: None):
        if pos == root:
            yield bad
            interrupted[0] = True
            check()
    monkeypatch.setattr(module, "legal_turns", moves)
    result = search(root, 1, cancelled=lambda: interrupted[0])
    assert result.score < -99_000
    assert not result.complete_depth
    assert not result.proven


def test_power_advice_deadline_and_available_set():
    result = choose_power([1, 2, 4], 2, 1, 1)
    assert result.elapsed <= 1.2
    assert result.power in (1, 4)
    assert 2 not in result.scores


def test_atomic_save_resume_archive_result_and_undo(tmp_path):
    store = Store(tmp_path)
    s = Session(Position())
    t = next(legal_turns(s.position))
    s = s.append(t)
    store.save(s)
    restored = store.load()
    assert restored.position == s.position
    assert restored.history == s.history
    assert restored.undo().position == s.initial
    ended = s.ended(0, "Abandon adverse")
    store.save(ended)
    archive = store.reset(ended)
    assert store.load() is None
    data = json.loads(archive.read_text())
    assert data["result"]["winner"] == 0
    assert len(data["history"]) == 1


def test_archive_failure_preserves_current_game(tmp_path, monkeypatch):
    store = Store(tmp_path)
    s = Session(Position())
    store.save(s)
    original = store.current.read_bytes()
    monkeypatch.setattr(store, "archive", lambda s: (_ for _ in ()).throw(OSError("disque plein")))
    with pytest.raises(OSError):
        store.reset(s)
    assert store.current.read_bytes() == original


def test_save_failure_keeps_previous_snapshot(tmp_path, monkeypatch):
    store = Store(tmp_path)
    s = Session(Position())
    store.save(s)
    previous = store.current.read_bytes()
    monkeypatch.setattr("santorini.storage.os.replace", lambda *args: (_ for _ in ()).throw(OSError("échec")))
    with pytest.raises(OSError):
        store.save(s.append(next(legal_turns(s.position))))
    assert store.current.read_bytes() == previous


def test_corrupt_history_is_rejected(tmp_path):
    store = Store(tmp_path)
    s = Session(Position()).append(next(legal_turns(Position())))
    data = s.to_dict()
    data["history"][0]["after"]["heights"][24] = 3
    store.current.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="incohérent"):
        store.load()

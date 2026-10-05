from dataclasses import replace

import pytest

from santorini.engine import Action, Position, validate_turn
from santorini.storage import Session, Store


def move(player, worker, source, target):
    return Action('move', player, worker, source, target)


def build(player, worker, source, target):
    return Action('build', player, worker, source, target)


def sample_session():
    session = Session(Position(workers=((6, 18), (11, 7))))
    first = (move(0, 0, 6, 0), build(0, 0, 0, 1))
    session = session.append(validate_turn(session.position, first))
    second = (move(1, 0, 11, 12), build(1, 0, 12, 13))
    return session.append(validate_turn(session.position, second))


def test_correct_replays_later_turns_and_roundtrips(tmp_path):
    original = sample_session()
    replacement = (move(0, 0, 6, 0), build(0, 0, 0, 5))
    corrected = original.correct(0, replacement)
    assert corrected.id == original.id
    assert len(corrected.history) == 2
    assert corrected.history[1].actions == original.history[1].actions
    assert corrected.history[1].after != original.history[1].after
    assert corrected.position.heights[1] == 0
    assert corrected.position.heights[5] == 1
    assert corrected.position.heights[13] == 1
    assert corrected.position.workers == original.position.workers
    store = Store(tmp_path)
    store.save(corrected)
    assert store.load() == corrected
    assert original.position.heights[1] == 1


def test_correct_rejects_illegal_later_turn_without_mutation():
    hs = [0] * 25
    hs[2], hs[7] = 2, 1
    session = Session(Position(heights=tuple(hs), workers=((6, 18), (11, 7))))
    session = session.append(validate_turn(session.position, (move(0, 0, 6, 1), build(0, 0, 1, 0))))
    session = session.append(validate_turn(session.position, (move(1, 1, 7, 2), build(1, 1, 2, 3))))
    snapshot = session.to_dict()
    with pytest.raises(ValueError, match='tour 2 devient illégal'):
        session.correct(0, (move(0, 0, 6, 1), build(0, 0, 1, 2)))
    assert session.history[-1].after.heights[2] == 2
    assert session.to_dict()['history'] == snapshot['history']


def test_correct_requires_a_complete_legal_turn():
    session = sample_session()
    with pytest.raises(ValueError):
        session.correct(0, (move(0, 0, 6, 0),))
    with pytest.raises(ValueError):
        session.correct(5, ())
    with pytest.raises(ValueError, match='terminée'):
        session.ended(0, 'Résultat déclaré').correct(0, ())

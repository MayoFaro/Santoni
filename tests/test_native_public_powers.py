"""Differential rules, constraints and deadlines for the public Rust port."""
import random
import time
from dataclasses import replace

import pytest
from santorini.engine import Position, legal_turns, validate_turn
from santorini.native import eligible, library_path, native_search, native_turns
from santorini.powers import POWERS
from santorini.extra import NEW_POWERS

pytestmark = pytest.mark.skipif(not library_path().exists(), reason='Compiler Rust')
PUBLIC = [p for p,v in POWERS.items() if v.supported and p>10 and p not in NEW_POWERS]


@pytest.mark.parametrize('power', PUBLIC)
def test_public_powers_match_python_outcomes_and_actions(power):
    rng = random.Random(1381+power)
    cells = {6,7,8,11,12,13,16,17,18}
    domes = sum(1<<i for i in range(25) if i not in cells)
    for iteration,opponent in enumerate([0,11,20,22,23,26,37]):
        hs = [0]*25 if iteration == 0 else [rng.randrange(4) for _ in range(25)]
        pos = Position(tuple(hs),domes,powers=(power,opponent),athena_lock=iteration==6)
        reference = list(legal_turns(pos))
        rust = native_turns(pos)
        assert {t.after for t in rust} == {t.after for t in reference}
        for turn in rust[::max(1,len(rust)//20)]:
            assert validate_turn(pos,turn.actions).after == turn.after


@pytest.mark.parametrize('power', [2,7,11,15,27,29,46,47,48,49,50,51,52,53,54,55])
def test_adonis_and_persephone_constrain_full_native_turns(power):
    hs=[0]*25;hs[7]=1;hs[12]=1
    pos=Position(tuple(hs),sum(1<<c for c in range(25) if c not in {6,7,8,11,12,13,16,17,18}),
                 powers=(power,26),adonis=(0,0,1))
    assert {t.after for t in native_turns(pos)} == {t.after for t in legal_turns(pos)}


@pytest.mark.parametrize('power',PUBLIC)
def test_public_search_uses_rust_and_returns_legal_complete_turn(power):
    pos=Position(powers=(power,0))
    started=time.monotonic()
    result=native_search(pos,.05)
    assert time.monotonic()-started<1.5
    assert result.engine=='Rust' and result.turn is not None
    assert validate_turn(pos,result.turn.actions).after==result.turn.after


def test_jason_native_continues_with_three_workers_after_activation():
    pos=Position(workers=((6,8,0),(16,18)),powers=(51,0),hero_used=(True,False))
    assert eligible(pos)
    assert {t.after for t in native_turns(pos)}=={t.after for t in legal_turns(pos)}


def test_large_optional_hero_search_is_streamed_and_cancellable():
    pos=Position(powers=(50,0))
    result=native_search(pos,.15)
    assert result.depth>=1 and result.nodes>20
    assert not result.complete_depth or result.proven
    started=time.monotonic()
    cancelled=native_search(pos,5,cancelled=lambda:True)
    assert time.monotonic()-started<.2 and cancelled.turn is None


@pytest.mark.parametrize('power', [0,1,2,3,4,5,6,7,8,9,10]+PUBLIC)
def test_native_incremental_input_matches_reference(power):
    from santorini.engine import next_actions
    from santorini.native import native_next_actions
    domes=sum(1<<c for c in range(25) if c not in {6,7,8,11,12,13,16,17,18})
    pos=Position(domes=domes,powers=(power,0))
    # Enter several actual reference turns, checking both continuation and stop.
    for index,turn in enumerate(legal_turns(pos)):
        if index>=3: break
        for length in range(len(turn.actions)+1):
            expected=next_actions(pos,turn.actions[:length])
            actual=native_next_actions(pos,turn.actions[:length])
            assert set(actual[0])==set(expected[0]),(power,length)
            assert actual[1]==expected[1]


def test_native_input_accepts_arbitrary_hero_subset_order():
    from santorini.engine import Action, next_actions
    from santorini.native import native_next_actions
    domes=sum(1<<c for c in range(25) if c not in {6,7,8,11,12,13,16,17,18})
    pos=Position(domes=domes,powers=(50,0))
    turn=next(t for t in legal_turns(pos) if t.actions[0].kind=='activate' and len(t.actions)>=5)
    prefix=turn.actions[:3]+tuple(reversed(turn.actions[3:5]))
    actual=native_next_actions(pos,prefix)
    expected=next_actions(pos,prefix)
    assert set(actual[0])==set(expected[0]) and actual[1]==expected[1]

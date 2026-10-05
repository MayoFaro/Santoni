import random
import time
from dataclasses import replace

import pytest

from santorini.engine import Position, legal_turns, validate_turn
from santorini.native import eligible, library_path, native_search, native_turns

pytestmark = pytest.mark.skipif(not library_path().exists(), reason="Compiler le moteur Rust avant les tests natifs")


def outcomes(turns):
    return {t.after for t in turns}


@pytest.mark.parametrize("power", range(11))
def test_native_and_reference_initial_actions_identical(power):
    pos = Position(powers=(power, 0))
    native = native_turns(pos)
    reference = list(legal_turns(pos))
    assert outcomes(native) == outcomes(reference)
    for turn in native:
        assert validate_turn(pos, turn.actions).after == turn.after


@pytest.mark.parametrize("power", range(11))
def test_native_and_reference_random_positions(power):
    rng = random.Random(127 + power)
    for iteration in range(4):
        cells = rng.sample(range(25), 4)
        heights = tuple(rng.randrange(4) for _ in range(25))
        domes = sum(1 << i for i in range(25) if i not in cells and rng.random() < .12)
        pos = Position(heights, domes, (tuple(cells[:2]), tuple(cells[2:])),
                       (power, rng.randrange(11)), iteration % 2,
                       athena_lock=iteration == 3)
        native = native_turns(pos)
        reference = list(legal_turns(pos))
        assert outcomes(native) == outcomes(reference)
        # Every native action sequence is independently legal, not just its
        # final board. Permutations/shorter paths may represent the same board.
        for turn in native:
            assert validate_turn(pos, turn.actions).after == turn.after


def test_native_deadline_progress_full_turn_and_cancellation():
    pos = Position()
    progress = []
    start = time.monotonic()
    result = native_search(pos, .15, publish=progress.append)
    assert time.monotonic() - start < .6
    assert result.depth >= 2
    assert progress and progress[0].turn is not None
    assert validate_turn(pos, result.turn.actions) == result.turn
    cancelled = native_search(pos, 10, cancelled=lambda: True)
    assert cancelled.elapsed < .1


def test_native_immediate_win_and_forced_defence():
    hs = [0] * 25
    hs[6], hs[7] = 2, 3
    result = native_search(Position(tuple(hs)), .1)
    assert result.proven and result.turn.after.winner == 0
    assert len(result.turn.actions) == 1
    hs = [0] * 25
    hs[17], hs[12] = 2, 3
    pos = Position(tuple(hs), workers=((11, 0), (17, 24)))
    result = native_search(pos, .2)
    assert result.turn.after.domes & (1 << 12)


def test_advanced_and_hero_positions_use_reference_engine():
    assert not eligible(Position(powers=(0, 46)))
    assert not eligible(Position(powers=(19, 0)))
    assert not eligible(Position(adonis=(0, 1, 1)))


def test_search_reports_active_depth_and_deadline():
    progress = []
    result = native_search(Position(), .15, publish=progress.append)
    assert result.stop_reason == "timeout"
    assert result.engine == "Rust"
    assert any(p.searching_depth > p.depth for p in progress)
    assert any(p.searching_depth == result.depth + 1 for p in progress)
    assert result.elapsed >= .14

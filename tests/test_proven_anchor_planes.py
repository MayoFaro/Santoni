import json
from pathlib import Path

import pytest

from santorini.engine import Position
from tools.proven_anchor.planes import compute_planes

ROOT = Path(__file__).resolve().parents[1]


def initial_no_power_position():
    # Matches native_engine/src/bin/selfplay_worker.rs::initial_state() exactly:
    # workers [[0,4,-1,-1],[20,24,-1,-1]], everything else default.
    return Position(workers=((0, 4), (20, 24)))


def test_compute_planes_matches_a_real_recorded_self_play_opening():
    # Any self-play game's ply-0 planes are encode_planes(initial_state()),
    # recorded by play_one_game before the first move is chosen -- ground
    # truth to compare against, not just a plumbing/determinism check.
    games = list(ROOT.glob("selfplay-runs/*/games/game-*.json"))
    if not games:
        pytest.skip(
            "no self-play game files on disk under selfplay-runs/*/games/ "
            "(not shipped as fixtures in a fresh clone)"
        )
    sample_game = games[0]
    recorded = json.loads(sample_game.read_text())["positions"][0]["planes"]

    produced = compute_planes(initial_no_power_position())

    assert list(produced) == recorded


def test_compute_planes_is_deterministic():
    position = initial_no_power_position()
    assert compute_planes(position) == compute_planes(position)


def test_compute_planes_differs_for_a_different_position():
    a = initial_no_power_position()
    b = Position(workers=((0, 4), (20, 24)), heights=(1,) + (0,) * 24)
    assert compute_planes(a) != compute_planes(b)

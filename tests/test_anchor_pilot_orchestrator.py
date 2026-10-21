"""Real-process integration tests for `AnchorPilotOrchestrator` -- no mocks,
same style as `tests/test_az_orchestrator.py`: spawn the real compiled
`selfplay_worker` binary in `--vs-reference` mode against a real
`InferenceServer`.

The anchor position used throughout is hand-built (not via
`tools.anchor_pilot_campaign.prepare_anchor`, which is a separate concern
already covered by that module's own tests): player 0 (the candidate) has
two workers at cells 0 and 1 with a single free, undomed neighbour cell (6)
to move into and build on; player 1 (the reference) has two workers boxed
in at cells 12/13 with every neighbouring cell domed, so it never gets a
legal move. Every cell other than {0, 1, 6, 12, 13} is domed.

This makes every game exactly one candidate ply long: the candidate (player
0, to move first) makes its one real MCTS+NN decision -- costing a handful
of genuine round trips through a real `InferenceServer`, so workers stay
alive for an observable amount of time -- and then play immediately hits
`terminal_value` for player 1 (no legal move, loses), before
`play_one_game_vs_reference` ever calls into the reference engine. Candidate
moves never touch cells 12/13 or their domed neighbours, so player 1 stays
boxed in regardless of which specific legal move the candidate's MCTS
picks: `outcome` is always `+1` (candidate wins), deterministically, no
matter how noisy/sampled the candidate's move choice is. This keeps games
fast (no 20-30s shutdown latency bound to worry about) while still
exercising a real inference round trip per game, so workers don't exhaust a
large `--games` count near-instantly and look dead before the test can
observe them.
"""
import json
import time
from pathlib import Path

import pytest

from santorini.engine import Position
from santorini.native import encode

from santoni_az.anchor_pilot import AnchorPilotOrchestrator
from santoni_az.stats import clopper_pearson_lower_bound

ROOT = Path(__file__).resolve().parents[1]
WORKER_BINARY = ROOT / "native_engine" / "target" / "release" / "selfplay_worker"


def _anchor_position():
    free_cells = {0, 1, 6, 12, 13}
    domes = ((1 << 25) - 1) & ~sum(1 << c for c in free_cells)
    return Position(workers=((0, 1), (12, 13)), domes=domes, player=0, powers=(0, 0))


def _write_anchor_state(path):
    path.write_bytes(bytes(encode(_anchor_position())))


def _write_fake_eval_game(path, outcome):
    path.write_text(json.dumps({"outcome": outcome, "positions": []}) + "\n")


def test_start_runs_both_training_and_eval_workers(tmp_path):
    anchor_path = tmp_path / "anchor_state.bin"
    _write_anchor_state(anchor_path)

    orch = AnchorPilotOrchestrator(
        run_dir=tmp_path, anchor_state_path=anchor_path, worker_binary_path=WORKER_BINARY,
        train_worker_count=1, eval_worker_count=1, simulations_per_move=4, reference_seconds=0.1,
    )
    orch.start()
    try:
        time.sleep(1.0)
        status = orch.status()
        assert status["workers_alive"] >= 2
    finally:
        orch.stop()


def test_eval_games_land_in_a_separate_directory_from_training_games(tmp_path):
    anchor_path = tmp_path / "anchor_state.bin"
    _write_anchor_state(anchor_path)

    orch = AnchorPilotOrchestrator(
        run_dir=tmp_path, anchor_state_path=anchor_path, worker_binary_path=WORKER_BINARY,
        train_worker_count=1, eval_worker_count=1, simulations_per_move=4, reference_seconds=0.1,
    )
    orch.start()
    try:
        time.sleep(2.0)
    finally:
        orch.stop()

    train_files = {f.name for f in (tmp_path / "games").glob("game-*.json")}
    eval_files = {f.name for f in (tmp_path / "eval_games").glob("game-*.json")}

    assert train_files, "training pool produced no games"
    assert eval_files, "eval pool produced no games"
    assert train_files.isdisjoint(eval_files)


def test_status_reports_a_clopper_pearson_lower_bound_from_eval_games_on_disk(tmp_path):
    anchor_path = tmp_path / "anchor_state.bin"
    _write_anchor_state(anchor_path)

    orch = AnchorPilotOrchestrator(
        run_dir=tmp_path, anchor_state_path=anchor_path, worker_binary_path=WORKER_BINARY,
        train_worker_count=0, eval_worker_count=0, simulations_per_move=4, reference_seconds=0.1,
    )

    outcomes = [1, 1, 1, -1, -1]  # 3 wins out of 5
    for i, outcome in enumerate(outcomes):
        _write_fake_eval_game(orch.eval_games_dir / f"game-fake-{i:06d}.json", outcome)

    orch.start()
    try:
        status = orch.status()
    finally:
        orch.stop()

    assert status["eval_n"] == len(outcomes)
    assert status["eval_wins"] == 3
    expected = clopper_pearson_lower_bound(3, len(outcomes))
    assert status["eval_ci_lower"] == pytest.approx(expected)


def test_status_reports_none_eval_ci_lower_before_any_eval_game(tmp_path):
    anchor_path = tmp_path / "anchor_state.bin"
    _write_anchor_state(anchor_path)

    orch = AnchorPilotOrchestrator(
        run_dir=tmp_path, anchor_state_path=anchor_path, worker_binary_path=WORKER_BINARY,
        train_worker_count=0, eval_worker_count=0, simulations_per_move=4, reference_seconds=0.1,
    )
    status = orch.status()
    assert status["eval_n"] == 0
    assert status["eval_wins"] == 0
    assert status["eval_ci_lower"] is None


def test_restart_does_not_lose_already_played_eval_games(tmp_path):
    """Checks more than the bare `eval_games_available` file count: nothing
    in this codebase ever deletes a game file, so that count alone is
    monotonic by construction and would pass even if `status()`'s rolling
    40-game window (`eval_n`, `eval_wins`, `eval_ci_lower`) were computed
    from something that forgot the first process's games entirely (e.g. an
    in-process counter instead of re-scanning `eval_games/` from disk on
    every call, which is what `_recent_eval_outcomes` actually does).
    Asserting `eval_n`/`eval_wins` also never decrease across the restart
    exercises that recomputation-from-disk path directly, on a fresh
    `AnchorPilotOrchestrator` instance that starts with an empty in-process
    state and must derive both fields purely from what is already on
    disk."""
    anchor_path = tmp_path / "anchor_state.bin"
    _write_anchor_state(anchor_path)

    orch = AnchorPilotOrchestrator(
        run_dir=tmp_path, anchor_state_path=anchor_path, worker_binary_path=WORKER_BINARY,
        train_worker_count=1, eval_worker_count=1, simulations_per_move=4, reference_seconds=0.1,
    )
    orch.start()
    try:
        time.sleep(1.5)
    finally:
        orch.stop()
    before = orch.status()
    assert before["eval_games_available"] >= 1
    assert before["eval_n"] >= 1

    orch2 = AnchorPilotOrchestrator(
        run_dir=tmp_path, anchor_state_path=anchor_path, worker_binary_path=WORKER_BINARY,
        train_worker_count=1, eval_worker_count=1, simulations_per_move=4, reference_seconds=0.1,
    )
    orch2.start()
    try:
        time.sleep(1.0)
        after = orch2.status()
    finally:
        orch2.stop()

    assert after["eval_games_available"] >= before["eval_games_available"]
    assert after["eval_n"] >= before["eval_n"]
    assert after["eval_wins"] >= before["eval_wins"]

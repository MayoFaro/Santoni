import json
from pathlib import Path

from tools.network_capacity_ablation import list_game_files, split_train_holdout, load_examples

PLACEHOLDER_PLANES = [0] * (6 * 5 * 5)  # NUM_PLANES * 5 * 5


def write_fake_game(games_dir, name, outcome, positions):
    """Same format as tests/test_az_trainer.py::write_fake_game."""
    payload = {
        "outcome": outcome,
        "positions": [{"planes": planes, "visits": visits} for planes, visits in positions],
    }
    (games_dir / name).write_text(json.dumps(payload) + "\n")


def test_list_game_files_returns_sorted_game_files_only(tmp_path):
    (tmp_path / "game-0002.json").write_text("{}")
    (tmp_path / "game-0001.json").write_text("{}")
    (tmp_path / "not-a-game.txt").write_text("x")
    files = list_game_files(tmp_path)
    assert [f.name for f in files] == ["game-0001.json", "game-0002.json"]


def test_split_train_holdout_is_disjoint_and_covers_all_files():
    files = [Path(f"game-{i:04d}.json") for i in range(20)]
    train, holdout = split_train_holdout(files, holdout_fraction=0.25, seed=1)
    assert len(holdout) == 5
    assert set(train) & set(holdout) == set()
    assert set(train) | set(holdout) == set(files)


def test_split_train_holdout_is_deterministic():
    files = [Path(f"game-{i:04d}.json") for i in range(20)]
    first = split_train_holdout(files, holdout_fraction=0.25, seed=42)
    second = split_train_holdout(files, holdout_fraction=0.25, seed=42)
    assert first == second


def test_load_examples_stops_at_max_positions(tmp_path):
    write_fake_game(tmp_path, "game-0000.json", 1, [
        (PLACEHOLDER_PLANES, [[10, 1]]),
        (PLACEHOLDER_PLANES, [[11, 1]]),
        (PLACEHOLDER_PLANES, [[12, 1]]),
    ])
    write_fake_game(tmp_path, "game-0001.json", -1, [
        (PLACEHOLDER_PLANES, [[13, 1]]),
        (PLACEHOLDER_PLANES, [[14, 1]]),
        (PLACEHOLDER_PLANES, [[15, 1]]),
    ])
    files = list_game_files(tmp_path)
    examples = load_examples(tmp_path, files, max_positions=4, seed=1)
    assert len(examples) == 4


def test_load_examples_skips_corrupt_file(tmp_path):
    write_fake_game(tmp_path, "game-0000.json", 1, [(PLACEHOLDER_PLANES, [[10, 1]])])
    (tmp_path / "game-0001.json").write_text("")  # fichier vide, non-fsync avant renommage
    files = list_game_files(tmp_path)
    examples = load_examples(tmp_path, files, max_positions=10, seed=1)
    assert len(examples) == 1

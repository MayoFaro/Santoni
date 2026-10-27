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


def _toy_examples(n, value):
    import torch
    from santoni_az.network import NUM_PLANES, ACTION_SPACE
    examples = []
    for _ in range(n):
        planes = torch.zeros(NUM_PLANES, 5, 5)
        policy = torch.zeros(ACTION_SPACE)
        policy[0] = 1.0
        examples.append((planes, policy, value))
    return examples


def test_train_net_returns_one_loss_per_step():
    from tools.network_capacity_ablation import build_net, train_net
    net = build_net(channels=8, blocks=1)
    examples = _toy_examples(16, value=1.0)
    losses = train_net(net, examples, steps=5, batch_size=4, seed=1)
    assert len(losses) == 5
    assert all(isinstance(loss, float) for loss in losses)


def test_evaluate_net_reports_full_sign_match_on_trivial_data():
    from tools.network_capacity_ablation import build_net, train_net, evaluate_net
    net = build_net(channels=8, blocks=1)
    examples = _toy_examples(32, value=1.0)
    train_net(net, examples, steps=200, batch_size=8, seed=1)
    metrics = evaluate_net(net, examples)
    assert set(metrics) == {"policy_loss", "value_loss", "total_loss", "value_sign_match_rate"}
    assert metrics["value_sign_match_rate"] == 1.0  # toutes les cibles sont positives


def test_evaluate_net_sign_match_rate_on_mixed_signs():
    from tools.network_capacity_ablation import build_net, evaluate_net
    net = build_net(channels=8, blocks=1)
    examples = _toy_examples(4, value=1.0) + _toy_examples(4, value=-1.0)
    metrics = evaluate_net(net, examples)
    assert 0.0 <= metrics["value_sign_match_rate"] <= 1.0


def test_parse_architectures_parses_channel_block_pairs():
    from tools.network_capacity_ablation import parse_architectures
    assert parse_architectures(["32:4", "64:8"]) == [(32, 4), (64, 8)]


def test_parse_architectures_rejects_malformed_entry():
    from tools.network_capacity_ablation import parse_architectures
    import pytest
    with pytest.raises(ValueError, match="32:4"):
        parse_architectures(["abc"])


def test_main_writes_atomic_report_and_no_leftover_tmp(tmp_path, capsys):
    from tools.network_capacity_ablation import main
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    for i in range(6):
        write_fake_game(games_dir, f"game-{i:04d}.json", 1 if i % 2 == 0 else -1, [
            (PLACEHOLDER_PLANES, [[10, 1]]),
            (PLACEHOLDER_PLANES, [[11, 1]]),
        ])
    out_path = tmp_path / "report.json"
    main([
        "--games-dir", str(games_dir),
        "--architectures", "8:1", "16:2",
        "--max-train-positions", "8",
        "--max-holdout-positions", "4",
        "--holdout-fraction", "0.34",
        "--steps", "3",
        "--batch-size", "2",
        "--out", str(out_path),
    ])
    assert out_path.exists()
    assert not out_path.with_suffix(".json.tmp").exists()
    report = json.loads(out_path.read_text())
    assert [entry["channels"] for entry in report["architectures"]] == [8, 16]
    for entry in report["architectures"]:
        assert "total_loss" in entry and "value_sign_match_rate" in entry
    captured = capsys.readouterr()
    assert "16" in captured.out  # tableau récapitulatif affiché

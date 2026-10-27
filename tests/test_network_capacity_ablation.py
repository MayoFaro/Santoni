import json
from pathlib import Path

import torch

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
    # Toutes les planes sont à zéro -> le réseau émet la même valeur pour
    # les 8 exemples, donc la concordance de signe vaut exactement 0.5 (4
    # cibles +1 sur 8). `0.0 <= x <= 1.0` était vrai de n'importe quelle
    # moyenne de booléens et n'aurait pas détecté une régression de
    # broadcasting (value_pred (N,1) vs value_targets (N,) -> (N,N)).
    assert metrics["value_sign_match_rate"] == 0.5


def test_load_examples_is_deterministic_across_calls(tmp_path):
    from tools.network_capacity_ablation import list_game_files, load_examples
    for i in range(5):
        write_fake_game(tmp_path, f"game-{i:04d}.json", 1 if i % 2 == 0 else -1, [
            (PLACEHOLDER_PLANES, [[i, 1]]),
            (PLACEHOLDER_PLANES, [[i + 1, 1]]),
        ])
    files = list_game_files(tmp_path)
    first = load_examples(tmp_path, files, max_positions=100, seed=7)
    second = load_examples(tmp_path, files, max_positions=100, seed=7)
    assert len(first) == len(second)
    for (p1, pt1, v1), (p2, pt2, v2) in zip(first, second):
        assert torch.equal(p1, p2)
        assert torch.equal(pt1, pt2)
        assert v1 == v2


def test_load_examples_train_and_holdout_never_overlap_end_to_end(tmp_path):
    """Chaque partie reçoit un marqueur unique dans ses planes (action_idx
    des visites) -- si `load_examples` lisait jamais le mauvais pool, un
    marqueur apparaîtrait dans les deux listes d'exemples décodés."""
    from tools.network_capacity_ablation import (
        list_game_files, split_train_holdout, load_examples,
    )
    for i in range(30):
        write_fake_game(tmp_path, f"game-{i:04d}.json", 1, [
            (PLACEHOLDER_PLANES, [[1000 + i, 1]]),
        ])
    files = list_game_files(tmp_path)
    train_files, holdout_files = split_train_holdout(files, holdout_fraction=0.3, seed=3)
    train_examples = load_examples(tmp_path, train_files, max_positions=1000, seed=3)
    holdout_examples = load_examples(tmp_path, holdout_files, max_positions=1000, seed=3)

    def markers(examples):
        return {int(policy_target.nonzero().item()) for _, policy_target, _ in examples}

    train_markers = markers(train_examples)
    holdout_markers = markers(holdout_examples)
    assert train_markers  # non vide, sinon le test ne prouve rien
    assert holdout_markers
    assert train_markers.isdisjoint(holdout_markers)


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


def _write_balanced_fixture_games(games_dir, n=8):
    for i in range(n):
        write_fake_game(games_dir, f"game-{i:04d}.json", 1 if i % 2 == 0 else -1, [
            (PLACEHOLDER_PLANES, [[10, 1]]),
            (PLACEHOLDER_PLANES, [[11, 1]]),
            (PLACEHOLDER_PLANES, [[12, 1]]),
        ])


def _run_tiny_main(tmp_path, out_name="report.json", extra_args=None, seed="5"):
    from tools.network_capacity_ablation import main
    games_dir = tmp_path / "games"
    if not games_dir.exists():
        games_dir.mkdir()
        _write_balanced_fixture_games(games_dir)
    out_path = tmp_path / out_name
    args = [
        "--games-dir", str(games_dir),
        "--architectures", "8:1", "16:2",
        "--max-train-positions", "12",
        "--max-holdout-positions", "6",
        "--holdout-fraction", "0.3",
        "--steps", "4",
        "--batch-size", "2",
        "--seed", seed,
        "--out", str(out_path),
    ]
    main(args + (extra_args or []))
    return json.loads(out_path.read_text())


def test_main_rejects_malformed_architecture_without_traceback(tmp_path, capsys):
    import pytest
    from tools.network_capacity_ablation import main
    with pytest.raises(SystemExit) as exc_info:
        main(["--games-dir", str(tmp_path), "--architectures", "abc"])
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "32:4" in captured.err
    assert "Traceback" not in captured.err


def test_main_raises_clear_error_on_missing_games_dir(tmp_path, capsys):
    import pytest
    from tools.network_capacity_ablation import main
    missing = tmp_path / "does-not-exist"
    with pytest.raises(SystemExit) as exc_info:
        main(["--games-dir", str(missing)])
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert str(missing) in captured.err
    assert "Traceback" not in captured.err


def test_main_raises_clear_error_on_empty_holdout_pool(tmp_path, capsys):
    import pytest
    from tools.network_capacity_ablation import main
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    _write_balanced_fixture_games(games_dir, n=4)
    with pytest.raises(SystemExit) as exc_info:
        # round(4 * 0.1) == 0 -> pool held-out vide
        main(["--games-dir", str(games_dir), "--holdout-fraction", "0.1"])
    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "held-out" in captured.err
    assert "Traceback" not in captured.err


def test_default_steps_is_at_least_10000():
    from tools.network_capacity_ablation import build_arg_parser
    parser = build_arg_parser()
    assert parser.get_default("steps") >= 10_000


def test_main_report_includes_training_regime_and_pool_sizes(tmp_path):
    report = _run_tiny_main(tmp_path, out_name="report2.json", seed="6")
    for key in ("steps", "batch_size", "max_train_positions", "max_holdout_positions",
                "holdout_fraction", "train_file_count", "holdout_file_count",
                "train_pool_size", "holdout_pool_size", "epochs_equivalent",
                "train_value_balance", "holdout_value_balance"):
        assert key in report, f"clé manquante dans le rapport : {key}"
    assert report["steps"] == 4
    assert report["train_pool_size"] > 0
    assert report["holdout_pool_size"] > 0


def test_main_help_documents_defaults(capsys):
    import pytest
    from tools.network_capacity_ablation import main
    with pytest.raises(SystemExit):
        main(["--help"])
    captured = capsys.readouterr()
    assert "--steps" in captured.out
    assert "channels:blocks" in captured.out  # une aide concrète, pas juste le nom du flag


def test_main_same_seed_two_runs_produce_identical_reports(tmp_path):
    first = _run_tiny_main(tmp_path, out_name="report-a.json", seed="9")
    second = _run_tiny_main(tmp_path, out_name="report-b.json", seed="9")
    assert first["architectures"] == second["architectures"]


def test_main_writes_partial_report_after_each_architecture(tmp_path, monkeypatch):
    """Si le processus est interrompu pendant l'entraînement d'une
    architecture (ex. SIGTERM par l'utilisateur), le rapport doit déjà
    contenir les architectures précédentes, déjà terminées, plutôt que de
    tout perdre parce que l'écriture n'avait lieu qu'une fois à la toute
    fin de la boucle."""
    import pytest
    import tools.network_capacity_ablation as module

    games_dir = tmp_path / "games"
    games_dir.mkdir()
    _write_balanced_fixture_games(games_dir, n=8)
    out_path = tmp_path / "report.json"

    original_train_net = module.train_net
    calls = {"count": 0}

    def flaky_train_net(net, examples, steps, batch_size, seed):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("simulated interruption before the second architecture finishes")
        return original_train_net(net, examples, steps, batch_size, seed)

    monkeypatch.setattr(module, "train_net", flaky_train_net)

    with pytest.raises(RuntimeError, match="simulated interruption"):
        module.main([
            "--games-dir", str(games_dir),
            "--architectures", "8:1", "16:2",
            "--max-train-positions", "8",
            "--max-holdout-positions", "4",
            "--holdout-fraction", "0.3",
            "--steps", "3",
            "--batch-size", "2",
            "--out", str(out_path),
        ])

    assert out_path.exists(), "le rapport partiel aurait dû être écrit avant l'interruption"
    report = json.loads(out_path.read_text())
    assert len(report["architectures"]) == 1
    assert report["architectures"][0]["channels"] == 8


def test_main_report_includes_baseline_metrics_before_training(tmp_path):
    report = _run_tiny_main(tmp_path, out_name="report-baseline.json", seed="11")
    for entry in report["architectures"]:
        for key in ("baseline_policy_loss", "baseline_value_loss",
                    "baseline_total_loss", "baseline_value_sign_match_rate"):
            assert key in entry, f"clé baseline manquante : {key}"
        # La perte baseline (réseau non entraîné) ne doit pas être
        # identique par construction à la perte après entraînement --
        # sinon rien ne distingue "avant" de "après" dans le rapport.
        assert entry["baseline_total_loss"] != entry["total_loss"]

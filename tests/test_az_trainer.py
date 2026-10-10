import json
from pathlib import Path

from santoni_az.trainer import Trainer


def write_fake_game(games_dir, name, outcome, moves):
    (games_dir / name).write_text(json.dumps({"outcome": outcome, "moves": moves}) + "\n")


def test_checkpoint_resume_preserves_generation_counter(tmp_path):
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    write_fake_game(games_dir, "game-0000.json", 1, [[10, 20], [5, 15]])
    write_fake_game(games_dir, "game-0001.json", -1, [[3, 10]])

    trainer = Trainer(str(games_dir), str(checkpoint_dir))
    assert trainer.consume_new_games() == 2
    trainer.train_step(batch_size=4)
    path = trainer.save_checkpoint()
    assert Path(path).exists()
    assert not Path(path + ".tmp").exists()

    resumed = Trainer(str(games_dir), str(checkpoint_dir))
    generation = resumed.load_latest_checkpoint()
    assert generation == 1
    assert resumed.consume_new_games() == 0  # mêmes parties, déjà consommées avant le crash simulé


def test_checkpoint_resume_preserves_untrained_examples(tmp_path):
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    # 3 moves in game 0, 2 moves in game 1 => 5 examples total.
    write_fake_game(games_dir, "game-0000.json", 1, [[10, 20], [5, 15], [7, 11]])
    write_fake_game(games_dir, "game-0001.json", -1, [[3, 10], [8, 4]])

    trainer = Trainer(str(games_dir), str(checkpoint_dir))
    trainer.consume_new_games()
    total_examples = len(trainer._examples)
    assert total_examples == 5

    k = 2
    assert k < total_examples
    trainer.train_step(batch_size=k)  # trains on k examples, leaves the rest untrained
    trainer.save_checkpoint()

    resumed = Trainer(str(games_dir), str(checkpoint_dir))
    resumed.load_latest_checkpoint()
    assert len(resumed._examples) == total_examples - k

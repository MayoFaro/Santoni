import json
from pathlib import Path

from santoni_az.trainer import CHECKPOINT_RETENTION, Trainer, checkpoint_generation


def write_fake_game(games_dir, name, outcome, moves):
    (games_dir / name).write_text(json.dumps({"outcome": outcome, "moves": moves}) + "\n")


def make_dirs(tmp_path):
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return games_dir, checkpoint_dir


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


def test_load_latest_checkpoint_selects_the_highest_generation_numerically(tmp_path):
    """Passé la génération 9999, `sorted(glob("gen-*.pt"))[-1]` (tri lexical)
    renvoie `gen-9999.pt` alors que `gen-10001.pt` existe : la reprise
    rechargeait silencieusement un checkpoint périmé, puis réécrasait les
    générations déjà produites en repartant de son compteur.

    Seul le checkpoint réellement sélectionné a besoin d'être un fichier
    `torch` valide : `load_latest_checkpoint` ne lit que le *nom* des autres.
    On crée donc trois noms-pièges inertes au lieu des ~10 000 vrais
    checkpoints que la boucle produirait en pratique — le piège tient à la
    comparaison de chaînes, pas au nombre de fichiers.
    """
    games_dir, checkpoint_dir = make_dirs(tmp_path)
    write_fake_game(games_dir, "game-0000.json", 1, [[10, 20]])

    trainer = Trainer(str(games_dir), str(checkpoint_dir))
    trainer.consume_new_games()
    trainer.train_step(batch_size=1)
    trainer.generation = 10000
    newest = trainer.save_checkpoint()
    assert Path(newest).name == "gen-10001.pt"

    for name in ("gen-0001.pt", "gen-9999.pt", "gen-10000.pt"):
        (checkpoint_dir / name).write_bytes(b"inerte : seul le nom compte ici")
    # Pin du piège lui-même : en ordre lexical, c'est bien gen-9999.pt qui
    # arrive en dernier.
    assert sorted(p.name for p in checkpoint_dir.glob("gen-*.pt"))[-1] == "gen-9999.pt"

    resumed = Trainer(str(games_dir), str(checkpoint_dir))
    assert resumed.load_latest_checkpoint() == 10001


def test_save_checkpoint_prunes_all_but_the_retained_newest(tmp_path):
    """Rien ne supprimait les anciens checkpoints : à la cadence réelle de
    l'orchestrateur, le répertoire croissait sans borne (de l'ordre de
    12 Go/h). `save_checkpoint` ne doit laisser que les
    `CHECKPOINT_RETENTION` générations les plus récentes — et choisir
    lesquelles numériquement, pas lexicalement, d'où le franchissement
    volontaire de la frontière 9999/10000 dans ce test."""
    games_dir, checkpoint_dir = make_dirs(tmp_path)
    trainer = Trainer(str(games_dir), str(checkpoint_dir))
    trainer.generation = 9997
    saves = CHECKPOINT_RETENTION + 3
    for _ in range(saves):
        trainer.save_checkpoint()

    kept = sorted(checkpoint_generation(p) for p in checkpoint_dir.glob("gen-*.pt"))
    assert len(kept) == CHECKPOINT_RETENTION
    assert kept == list(range(9998 + saves - CHECKPOINT_RETENTION, 9998 + saves))
    assert kept[-1] == 10005
    assert not list(checkpoint_dir.glob("*.tmp"))

    # La reprise reste possible après élagage, sur la dernière génération.
    resumed = Trainer(str(games_dir), str(checkpoint_dir))
    assert resumed.load_latest_checkpoint() == 10005

import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from santoni_az.network import NUM_PLANES
from santoni_az.trainer import CHECKPOINT_RETENTION, MAX_BUFFER_POSITIONS, Trainer, checkpoint_generation

PLACEHOLDER_PLANES = [0] * (NUM_PLANES * 5 * 5)


def write_fake_game(games_dir, name, outcome, positions):
    """`positions` is a list of (planes, visits) pairs — `planes` a flat
    list of NUM_PLANES*5*5 ints, `visits` a list of [action_idx, visit_count]
    pairs — mirroring the real format `mcts_io::write_game_atomically` writes."""
    payload = {
        "outcome": outcome,
        "positions": [{"planes": planes, "visits": visits} for planes, visits in positions],
    }
    (games_dir / name).write_text(json.dumps(payload) + "\n")


def make_dirs(tmp_path):
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return games_dir, checkpoint_dir


def test_checkpoint_resume_preserves_generation_counter(tmp_path):
    games_dir, checkpoint_dir = make_dirs(tmp_path)
    write_fake_game(games_dir, "game-0000.json", 1, [
        (PLACEHOLDER_PLANES, [[10, 20]]),
        (PLACEHOLDER_PLANES, [[5, 15]]),
    ])
    write_fake_game(games_dir, "game-0001.json", -1, [
        (PLACEHOLDER_PLANES, [[3, 10]]),
    ])

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


def test_train_step_does_not_delete_examples_after_training(tmp_path):
    games_dir, checkpoint_dir = make_dirs(tmp_path)
    write_fake_game(games_dir, "game-0000.json", 1, [
        (PLACEHOLDER_PLANES, [[10, 20]]),
        (PLACEHOLDER_PLANES, [[5, 15]]),
    ])
    trainer = Trainer(str(games_dir), str(checkpoint_dir))
    trainer.consume_new_games()
    before = len(trainer._examples)

    trainer.train_step(batch_size=1)
    trainer.train_step(batch_size=1)
    trainer.train_step(batch_size=1)

    assert len(trainer._examples) == before


def test_train_step_handles_batch_size_larger_than_buffer(tmp_path):
    games_dir, checkpoint_dir = make_dirs(tmp_path)
    write_fake_game(games_dir, "game-0000.json", 1, [(PLACEHOLDER_PLANES, [[0, 1]])])
    trainer = Trainer(str(games_dir), str(checkpoint_dir))
    trainer.consume_new_games()

    loss = trainer.train_step(batch_size=64)

    assert loss > 0.0
    assert len(trainer._examples) == 1


def test_consume_new_games_evicts_oldest_positions_beyond_the_buffer_cap(tmp_path):
    games_dir, checkpoint_dir = make_dirs(tmp_path)
    trainer = Trainer(str(games_dir), str(checkpoint_dir), max_buffer_positions=1)
    write_fake_game(games_dir, "game-0000.json", 1, [(PLACEHOLDER_PLANES, [[0, 1]])])
    trainer.consume_new_games()
    assert [v for _, _, v in trainer._examples] == [1.0]

    write_fake_game(games_dir, "game-0001.json", -1, [(PLACEHOLDER_PLANES, [[0, 1]])])
    trainer.consume_new_games()

    assert [v for _, _, v in trainer._examples] == [-1.0]


def test_consume_new_games_skips_unreadable_game_files(tmp_path):
    games_dir, checkpoint_dir = make_dirs(tmp_path)
    (games_dir / "game-bad.json").write_text("")  # fichier corrompu/vide
    write_fake_game(games_dir, "game-good.json", 1, [(PLACEHOLDER_PLANES, [[0, 1]])])

    trainer = Trainer(str(games_dir), str(checkpoint_dir))
    count = trainer.consume_new_games()

    assert count == 2
    assert len(trainer._examples) == 1
    # Le fichier illisible reste marqué consommé : jamais retenté.
    assert trainer.consume_new_games() == 0


def test_load_latest_checkpoint_skips_unreadable_game_files_during_rebuild(tmp_path):
    games_dir, checkpoint_dir = make_dirs(tmp_path)
    (games_dir / "game-bad.json").write_text("")
    write_fake_game(games_dir, "game-good.json", 1, [(PLACEHOLDER_PLANES, [[0, 1]])])

    trainer = Trainer(str(games_dir), str(checkpoint_dir))
    generation = trainer.load_latest_checkpoint()

    assert generation == 0
    assert [v for _, _, v in trainer._examples] == [1.0]
    assert trainer.consume_new_games() == 0


def test_load_latest_checkpoint_rebuilds_buffer_in_chronological_order_after_trimming(tmp_path):
    games_dir, checkpoint_dir = make_dirs(tmp_path)
    write_fake_game(games_dir, "game-b.json", 1, [
        (PLACEHOLDER_PLANES, [[210, 1]]),
        (PLACEHOLDER_PLANES, [[211, 1]]),
    ])
    os.utime(games_dir / "game-b.json", (2000, 2000))
    write_fake_game(games_dir, "game-c.json", 1, [(PLACEHOLDER_PLANES, [[300, 1]])])
    os.utime(games_dir / "game-c.json", (3000, 3000))

    trainer = Trainer(str(games_dir), str(checkpoint_dir), max_buffer_positions=2)
    trainer.load_latest_checkpoint()

    hot_actions = [policy.argmax().item() for _, policy, _ in trainer._examples]
    assert hot_actions == [211, 300]


def test_load_latest_checkpoint_rebuilds_buffer_only_from_most_recent_games(tmp_path):
    games_dir, checkpoint_dir = make_dirs(tmp_path)
    write_fake_game(games_dir, "game-old.json", 1, [(PLACEHOLDER_PLANES, [[0, 1]])])
    os.utime(games_dir / "game-old.json", (1000, 1000))
    write_fake_game(games_dir, "game-new.json", -1, [(PLACEHOLDER_PLANES, [[0, 1]])])

    trainer = Trainer(str(games_dir), str(checkpoint_dir), max_buffer_positions=1)
    trainer.save_checkpoint()

    resumed = Trainer(str(games_dir), str(checkpoint_dir), max_buffer_positions=1)
    resumed.load_latest_checkpoint()

    assert [v for _, _, v in resumed._examples] == [-1.0]
    # Les deux fichiers, y compris le plus ancien jamais chargé dans le
    # tampon, doivent être marqués consommés -- sinon le prochain appel
    # réingérerait tout l'historique plus ancien que la fenêtre.
    assert resumed.consume_new_games() == 0


def test_load_latest_checkpoint_rebuilds_buffer_even_without_a_checkpoint_file(tmp_path):
    games_dir, checkpoint_dir = make_dirs(tmp_path)
    write_fake_game(games_dir, "game-0000.json", 1, [(PLACEHOLDER_PLANES, [[0, 1]])])

    trainer = Trainer(str(games_dir), str(checkpoint_dir))
    generation = trainer.load_latest_checkpoint()

    assert generation == 0
    assert len(trainer._examples) == 1
    assert trainer.consume_new_games() == 0


def test_load_latest_checkpoint_handles_no_games_on_disk_yet(tmp_path):
    games_dir, checkpoint_dir = make_dirs(tmp_path)
    trainer = Trainer(str(games_dir), str(checkpoint_dir))

    generation = trainer.load_latest_checkpoint()

    assert generation == 0
    assert trainer._examples == []


def test_save_checkpoint_no_longer_persists_examples_or_consumed_games(tmp_path):
    games_dir, checkpoint_dir = make_dirs(tmp_path)
    write_fake_game(games_dir, "game-0000.json", 1, [(PLACEHOLDER_PLANES, [[0, 1]])])
    trainer = Trainer(str(games_dir), str(checkpoint_dir))
    trainer.consume_new_games()

    path = trainer.save_checkpoint()
    saved = torch.load(path, weights_only=False)

    assert "examples" not in saved
    assert "consumed_games" not in saved


def test_load_latest_checkpoint_selects_the_highest_generation_numerically(tmp_path):
    """Passé la génération 9999, `sorted(glob("gen-*.pt"))[-1]` (tri lexical)
    renvoie `gen-9999.pt` alors que `gen-10001.pt` existe : la reprise
    rechargeait silencieusement un checkpoint périmé, puis réécrasait les
    générations déjà produites en repartant de son compteur."""
    games_dir, checkpoint_dir = make_dirs(tmp_path)
    write_fake_game(games_dir, "game-0000.json", 1, [(PLACEHOLDER_PLANES, [[10, 20]])])

    trainer = Trainer(str(games_dir), str(checkpoint_dir))
    trainer.consume_new_games()
    trainer.train_step(batch_size=1)
    trainer.generation = 10000
    newest = trainer.save_checkpoint()
    assert Path(newest).name == "gen-10001.pt"

    for name in ("gen-0001.pt", "gen-9999.pt", "gen-10000.pt"):
        (checkpoint_dir / name).write_bytes(b"inerte : seul le nom compte ici")
    assert sorted(p.name for p in checkpoint_dir.glob("gen-*.pt"))[-1] == "gen-9999.pt"

    resumed = Trainer(str(games_dir), str(checkpoint_dir))
    assert resumed.load_latest_checkpoint() == 10001


def test_save_checkpoint_prunes_all_but_the_retained_newest(tmp_path):
    """Rien ne supprimait les anciens checkpoints : `save_checkpoint` ne doit
    laisser que les `CHECKPOINT_RETENTION` générations les plus récentes,
    choisies numériquement, pas lexicalement."""
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

    resumed = Trainer(str(games_dir), str(checkpoint_dir))
    assert resumed.load_latest_checkpoint() == 10005


def test_add_exact_examples_are_immediately_trainable(tmp_path):
    games_dir, checkpoint_dir = make_dirs(tmp_path)
    trainer = Trainer(str(games_dir), str(checkpoint_dir))
    planes = torch.zeros(NUM_PLANES, 5, 5)
    policy = torch.zeros(1250)
    policy[0] = 1.0

    trainer.add_exact_examples([(planes, policy, 1.0)])

    assert len(trainer._examples) == 1
    loss = trainer.train_step(batch_size=1)
    assert loss > 0.0


def test_consume_new_games_decodes_real_board_planes(tmp_path):
    games_dir, checkpoint_dir = make_dirs(tmp_path)
    planes = [1 if i % 7 == 0 else 0 for i in range(NUM_PLANES * 5 * 5)]
    write_fake_game(games_dir, "game-0000.json", 1, [(planes, [[0, 10]])])

    trainer = Trainer(str(games_dir), str(checkpoint_dir))
    trainer.consume_new_games()
    stored_planes, _, _ = trainer._examples[0]
    assert torch.equal(stored_planes, torch.tensor(planes, dtype=torch.float32).view(NUM_PLANES, 5, 5))


def test_consume_new_games_builds_policy_target_from_visit_distribution(tmp_path):
    games_dir, checkpoint_dir = make_dirs(tmp_path)
    write_fake_game(games_dir, "game-0000.json", 1, [(PLACEHOLDER_PLANES, [[3, 30], [9, 10]])])

    trainer = Trainer(str(games_dir), str(checkpoint_dir))
    trainer.consume_new_games()
    _, policy_target, _ = trainer._examples[0]
    assert policy_target[3].item() == pytest.approx(0.75)
    assert policy_target[9].item() == pytest.approx(0.25)
    assert policy_target.sum().item() == pytest.approx(1.0)


def test_consume_new_games_alternates_value_target_sign_by_ply(tmp_path):
    games_dir, checkpoint_dir = make_dirs(tmp_path)
    write_fake_game(games_dir, "game-0000.json", 1, [
        (PLACEHOLDER_PLANES, [[0, 1]]),  # ply 0 : joueur 0 au trait
        (PLACEHOLDER_PLANES, [[0, 1]]),  # ply 1 : joueur 1 au trait
        (PLACEHOLDER_PLANES, [[0, 1]]),  # ply 2 : joueur 0 au trait
    ])

    trainer = Trainer(str(games_dir), str(checkpoint_dir))
    trainer.consume_new_games()
    values = [v for _, _, v in trainer._examples]
    assert values == [1.0, -1.0, 1.0]


def test_train_step_loss_uses_soft_policy_target_distribution(tmp_path):
    """Avant ce correctif, `train_step` appelait `F.cross_entropy` avec
    `policy_targets.argmax(dim=1)` : la distribution de visites réelle
    (ex. 75%/25% sur deux coups) s'effondrait silencieusement sur un simple
    one-hot du coup majoritaire. Ce test pin la vraie perte "soft target"
    comme référence — il échoue avec l'ancienne ligne `argmax` dès que la
    distribution n'est pas déjà un one-hot."""
    games_dir, checkpoint_dir = make_dirs(tmp_path)
    write_fake_game(games_dir, "game-0000.json", 1, [(PLACEHOLDER_PLANES, [[3, 3], [9, 1]])])

    trainer = Trainer(str(games_dir), str(checkpoint_dir))
    trainer.consume_new_games()
    planes, policy_target, value_target = trainer._examples[0]
    with torch.no_grad():
        policy_logits, value_pred = trainer.net(planes.unsqueeze(0))
        expected_policy_loss = F.cross_entropy(policy_logits, policy_target.unsqueeze(0))
        expected_value_loss = F.mse_loss(value_pred, torch.tensor([[value_target]]))

    loss = trainer.train_step(batch_size=1)
    assert loss == pytest.approx((expected_policy_loss + expected_value_loss).item(), rel=1e-4)

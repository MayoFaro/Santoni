"""Ablation de capacité réseau (MCTS+NN, sous-jeu sans pouvoir) : ré-entraîne
hors ligne plusieurs tailles de `PolicyValueNet` sur des positions déjà
enregistrées sur disque, pour tester si la taille actuelle du réseau
plafonne la qualité apprise. Voir
`docs/superpowers/specs/2026-10-27-mcts-az-network-capacity-ablation-design.md`.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from santoni_az.trainer import Trainer


def list_game_files(games_dir):
    """Liste triée (ordre de nom, déterministe) des `game-*.json` d'un
    répertoire de run existant — même motif de glob que
    `Trainer.consume_new_games`."""
    return sorted(Path(games_dir).glob("game-*.json"))


def split_train_holdout(game_files, holdout_fraction, seed):
    """Partition au niveau de la *partie*, pas de la position : les
    positions d'une même partie partagent une trajectoire corrélée, les
    mélanger entre train et held-out fuiterait du signal. Mélange
    déterministe à `seed` fixe. Le choix de "les N premiers après mélange
    vont au held-out" est arbitraire mais fixe — rien dans ce script ne
    dépend de savoir quelle moitié est laquelle au-delà de la partition
    elle-même."""
    shuffled = list(game_files)
    random.Random(seed).shuffle(shuffled)
    cut = round(len(shuffled) * holdout_fraction)
    holdout_files = shuffled[:cut]
    train_files = shuffled[cut:]
    return train_files, holdout_files


def load_examples(games_dir, game_files, max_positions, seed):
    """Charge jusqu'à `max_positions` exemples d'entraînement
    `(planes, policy_target, value_target)` depuis `game_files`, dans un
    ordre mélangé mais déterministe (`seed`) pour ne pas biaiser
    l'échantillon vers les fichiers en tête de `game_files`. S'arrête dès
    le cap atteint, sans lire de fichier supplémentaire. Un fichier
    illisible/corrompu (même risque que `Trainer.consume_new_games` —
    écriture non-fsync avant renommage atomique) est ignoré silencieusement."""
    games_dir = Path(games_dir)
    order = list(game_files)
    random.Random(seed).shuffle(order)
    examples = []
    for f in order:
        if len(examples) >= max_positions:
            break
        try:
            data = json.loads((games_dir / f.name).read_text())
        except (json.JSONDecodeError, OSError):
            continue
        outcome = data["outcome"]
        for ply, position in enumerate(data["positions"]):
            examples.append(Trainer._decode_position(position, ply, outcome))
    return examples[:max_positions]

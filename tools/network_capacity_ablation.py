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

import torch
import torch.nn.functional as F

from santoni_az.network import PolicyValueNet
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


def build_net(channels, blocks):
    """Enveloppe nommée pour la lisibilité du reste du script — aucune
    logique propre."""
    return PolicyValueNet(channels=channels, blocks=blocks)


def _stack_batch(batch):
    planes = torch.stack([p for p, _, _ in batch])
    policy_targets = torch.stack([pt for _, pt, _ in batch])
    value_targets = torch.tensor([vt for _, _, vt in batch], dtype=torch.float32).unsqueeze(1)
    return planes, policy_targets, value_targets


def _loss_components(net, planes, policy_targets, value_targets):
    policy_logits, value_pred = net(planes)
    policy_loss = F.cross_entropy(policy_logits, policy_targets)
    value_loss = F.mse_loss(value_pred, value_targets)
    return policy_loss, value_loss, value_pred


def train_net(net, examples, steps, batch_size, seed):
    """Boucle d'entraînement supervisé pur sur `examples` fixe (pas de
    self-play, pas de tampon glissant) — même composition de perte que
    `Trainer.train_step`. Échantillonnage sans remise quand `batch_size`
    tient dans la population, avec remise au-delà (contrairement à
    `Trainer.train_step`, qui se contente de réduire la taille de batch
    via `min(batch_size, len(self._examples))` : cette ablation veut un
    nombre de pas de gradient constant même sur un petit jeu de test, pas
    un batch rétréci)."""
    rng = random.Random(seed)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    losses = []
    for _ in range(steps):
        if batch_size <= len(examples):
            batch = rng.sample(examples, batch_size)
        else:
            batch = rng.choices(examples, k=batch_size)
        planes, policy_targets, value_targets = _stack_batch(batch)
        policy_loss, value_loss, _ = _loss_components(net, planes, policy_targets, value_targets)
        loss = policy_loss + value_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return losses


def evaluate_net(net, held_out_examples):
    """Pertes moyennes + concordance de signe (valeur) sur `held_out_examples`,
    réseau en mode évaluation. Une cible ou prédiction à `0.0` est traitée
    comme son propre signe (via `torch.sign`, qui renvoie 0 pour 0) — sans
    conséquence pratique ici puisque `Trainer._decode_position` ne produit
    que des `value_target` à ±1 (issue terminale). Ne restaure pas
    `net.train()` avant de retourner : limite de portée délibérée, chaque
    architecture reçoit un réseau neuf et n'est jamais réentraînée après
    son propre appel à `evaluate_net`."""
    net.eval()
    with torch.no_grad():
        planes, policy_targets, value_targets = _stack_batch(held_out_examples)
        policy_loss, value_loss, value_pred = _loss_components(net, planes, policy_targets, value_targets)
        sign_match_rate = (value_pred.sign() == value_targets.sign()).float().mean().item()
    return {
        "policy_loss": float(policy_loss.item()),
        "value_loss": float(value_loss.item()),
        "total_loss": float(policy_loss.item()) + float(value_loss.item()),
        "value_sign_match_rate": sign_match_rate,
    }

"""Ablation de capacité réseau (MCTS+NN, sous-jeu sans pouvoir) : ré-entraîne
hors ligne plusieurs tailles de `PolicyValueNet` sur des positions déjà
enregistrées sur disque, pour tester si la taille actuelle du réseau
plafonne la qualité apprise. Voir
`docs/superpowers/specs/2026-10-27-mcts-az-network-capacity-ablation-design.md`.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from santoni_az.network import PolicyValueNet
# Accès à une méthode privée de `Trainer`, délibérément sanctionné par le
# spec (2026-10-27-mcts-az-network-capacity-ablation-design.md) : la
# convention de signe par ply y est fixée une seule fois plutôt que
# dupliquée ici, pour éviter la classe de bug (signe/perspective) qui a
# déjà touché ce chantier deux fois.
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
    écriture non-fsync avant renommage atomique) est ignoré silencieusement.
    `games_dir` n'est plus utilisé pour recomposer les chemins (chaque
    élément de `game_files` est déjà complet) — conservé uniquement pour
    la stabilité de la signature avec Task 1."""
    order = list(game_files)
    random.Random(seed).shuffle(order)
    examples = []
    for f in order:
        if len(examples) >= max_positions:
            break
        try:
            # `f` est déjà un chemin complet quand il vient de
            # `list_game_files`/`split_train_holdout` (le seul producteur
            # réel dans ce script) — le lire directement plutôt que de le
            # recomposer avec `games_dir` évite de silencieusement relire
            # un autre répertoire si `game_files` venait d'ailleurs.
            data = json.loads(Path(f).read_text())
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
    un batch rétréci). Remet `net` en mode entraînement (`net.train()`) —
    nécessaire depuis que `main` évalue un réseau neuf (mode `.eval()`,
    `evaluate_net`) comme référence avant d'appeler `train_net` dessus."""
    net.train()
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


def parse_architectures(raw):
    """Décode une liste de `"channels:blocks"` (ex. `"32:4"`) en paires
    d'entiers positifs (`channels >= 1`, `blocks >= 0`). Message d'erreur
    explicite plutôt qu'une trace Python brute sur une entrée mal formée
    en CLI."""
    architectures = []
    for entry in raw:
        parts = entry.split(":")
        invalid = ValueError(
            f"architecture invalide {entry!r}, format attendu 'channels:blocks' (ex. '32:4')"
        )
        if len(parts) != 2:
            raise invalid
        try:
            channels, blocks = int(parts[0]), int(parts[1])
        except ValueError:
            raise invalid from None
        if channels < 1 or blocks < 0:
            raise invalid
        architectures.append((channels, blocks))
    return architectures


def _value_balance(examples):
    """Répartition des signes de `value_target` dans `examples` — permet
    de repérer depuis le rapport un pool dont l'issue est très déséquilibrée
    (ex. un côté qui perd ~99% des parties), régime où
    `value_sign_match_rate` peut saturer sans rapport avec une vraie
    qualité de valeur apprise (cf. focus de revue)."""
    positive = sum(1 for _, _, v in examples if v > 0)
    negative = sum(1 for _, _, v in examples if v < 0)
    zero = len(examples) - positive - negative
    return {"positive": positive, "negative": negative, "zero": zero}


def _sanitize_for_json(value):
    """Remplace tout float non fini (`NaN`/`inf`, atteignable si une
    architecture diverge) par `None` avant sérialisation — `json.dumps`
    par défaut écrirait un littéral `NaN`/`Infinity` invalide pour tout
    lecteur JSON non-Python, sans faire échouer l'écriture atomique elle-même."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(v) for v in value]
    return value


def build_arg_parser():
    """Construit le parser CLI séparément de `main` pour pouvoir inspecter
    ses valeurs par défaut (ex. tests) sans lancer d'entraînement."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--games-dir", default="selfplay-runs/anchor-pilot-20261021/games",
                         help="répertoire contenant les game-*.json à ré-entraîner hors ligne")
    parser.add_argument("--architectures", nargs="+", default=["32:4", "64:8", "128:12"],
                         help="liste de 'channels:blocks', ex. 32:4 64:8 128:12")
    parser.add_argument("--max-train-positions", type=int, default=200_000,
                         help="plafond de positions chargées pour le pool d'entraînement")
    parser.add_argument("--max-holdout-positions", type=int, default=20_000,
                         help="plafond de positions chargées pour le pool held-out")
    parser.add_argument("--holdout-fraction", type=float, default=0.1,
                         help="fraction des parties (pas des positions) réservée au held-out")
    parser.add_argument("--steps", type=int, default=10_000,
                         help="pas de gradient par architecture (>=10000 recommandé : "
                              "2000 ne couvre qu'une fraction d'epoch sur un pool de 200k "
                              "positions et avantage artificiellement les petits réseaux)")
    parser.add_argument("--batch-size", type=int, default=64,
                         help="taille de batch, alignée sur Trainer.train_step")
    parser.add_argument("--seed", type=int, default=20261027,
                         help="graine unique pour la partition, l'échantillonnage et "
                              "l'initialisation des poids (réappliquée avant chaque "
                              "architecture pour une reproductibilité par architecture)")
    parser.add_argument("--out", default="selfplay-runs/ablation-20261027/report.json",
                         help="chemin du rapport JSON (écriture atomique)")
    return parser


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        architectures = parse_architectures(args.architectures)
        game_files = list_game_files(args.games_dir)
        if not game_files:
            raise ValueError(f"aucun fichier game-*.json trouvé dans {args.games_dir!r}")
        train_files, holdout_files = split_train_holdout(game_files, args.holdout_fraction, args.seed)
        if not train_files:
            raise ValueError(
                f"pool d'entraînement vide (holdout_fraction={args.holdout_fraction} "
                f"sur {len(game_files)} fichiers → 0 pour l'entraînement)"
            )
        if not holdout_files:
            raise ValueError(
                f"pool held-out vide (holdout_fraction={args.holdout_fraction} "
                f"sur {len(game_files)} fichiers → 0)"
            )
        train_examples = load_examples(args.games_dir, train_files, args.max_train_positions, args.seed)
        holdout_examples = load_examples(args.games_dir, holdout_files, args.max_holdout_positions, args.seed)
        if not train_examples:
            raise ValueError("aucune position chargée dans le pool d'entraînement")
        if not holdout_examples:
            raise ValueError("aucune position chargée dans le pool held-out")
    except ValueError as exc:
        parser.error(str(exc))
        return  # parser.error() appelle sys.exit ; inatteignable, garde le flux explicite

    epochs_equivalent = args.steps * args.batch_size / len(train_examples)
    out_path = Path(args.out)

    results = []
    print(f"{'channels':>8} {'blocks':>6} {'params':>10} {'baseline_vsm':>12} "
          f"{'policy_loss':>12} {'value_loss':>11} {'value_sign_match':>16}", flush=True)
    for channels, blocks in architectures:
        # Réinitialise la graine torch à la même valeur avant chaque
        # architecture : sans cela, l'initialisation des poids de
        # l'architecture n dépend du nombre de tirages aléatoires déjà
        # consommés par les architectures précédentes (confond la
        # comparaison avec la chance d'initialisation) et deux lancements
        # du même `--seed` ne donnent pas le même résultat.
        torch.manual_seed(args.seed)
        net = build_net(channels, blocks)
        baseline_metrics = evaluate_net(net, holdout_examples)
        losses = train_net(net, train_examples, args.steps, args.batch_size, args.seed)
        metrics = evaluate_net(net, holdout_examples)
        num_parameters = sum(p.numel() for p in net.parameters())
        results.append({
            "channels": channels,
            "blocks": blocks,
            "num_parameters": num_parameters,
            "train_losses": losses,
            "baseline_policy_loss": baseline_metrics["policy_loss"],
            "baseline_value_loss": baseline_metrics["value_loss"],
            "baseline_total_loss": baseline_metrics["total_loss"],
            "baseline_value_sign_match_rate": baseline_metrics["value_sign_match_rate"],
            **metrics,
        })
        print(f"{channels:>8} {blocks:>6} {num_parameters:>10} "
              f"{baseline_metrics['value_sign_match_rate']:>12.3f} "
              f"{metrics['policy_loss']:>12.4f} {metrics['value_loss']:>11.4f} "
              f"{metrics['value_sign_match_rate']:>16.3f}", flush=True)

        # Écrit le rapport après CHAQUE architecture, pas seulement à la
        # fin de la boucle : une interruption (SIGTERM, Ctrl-C) pendant
        # l'entraînement d'une architecture ne perd alors que celle en
        # cours, jamais les architectures déjà terminées — même garantie
        # d'écriture atomique (.tmp + os.replace) à chaque fois.
        report = {
            "games_dir": str(args.games_dir),
            "seed": args.seed,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "max_train_positions": args.max_train_positions,
            "max_holdout_positions": args.max_holdout_positions,
            "holdout_fraction": args.holdout_fraction,
            "train_file_count": len(train_files),
            "holdout_file_count": len(holdout_files),
            "train_pool_size": len(train_examples),
            "holdout_pool_size": len(holdout_examples),
            "train_value_balance": _value_balance(train_examples),
            "holdout_value_balance": _value_balance(holdout_examples),
            "epochs_equivalent": epochs_equivalent,
            "architectures": results,
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(_sanitize_for_json(report), indent=2))
        os.replace(tmp_path, out_path)


if __name__ == "__main__":
    main()

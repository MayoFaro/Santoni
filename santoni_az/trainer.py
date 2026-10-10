import json
import os
import re
from pathlib import Path

import torch
import torch.nn.functional as F

from .network import PolicyValueNet, ACTION_SPACE, NUM_PLANES

# Nombre de checkpoints conservés sur disque. Chaque checkpoint contient les
# poids, l'état de l'optimiseur Adam (deux moments par paramètre) et la file
# d'exemples non entraînés : plusieurs mégaoctets pièce, écrits à chaque
# sauvegarde. Rien ne les supprimait auparavant, et à la cadence réelle de
# `Orchestrator._train_loop` (mesurée à ~1,5 sauvegarde/seconde en revue
# finale) cela représentait de l'ordre de 12 Go/heure de croissance non
# bornée sur une machine censée tourner en quasi-continu.
# 5 est un compromis : la reprise n'a besoin que du plus récent (cf.
# `load_latest_checkpoint`), mais garder quelques générations précédentes
# laisse de quoi revenir en arrière si le dernier checkpoint s'avère
# corrompu ou si l'entraînement divergeait — tout en gardant l'empreinte
# disque constante au lieu de linéaire dans le temps.
CHECKPOINT_RETENTION = 5

_GENERATION_PATTERN = re.compile(r'^gen-(\d+)\.pt$')


def checkpoint_generation(path):
    """Numéro de génération porté par un nom de checkpoint, ou `None` si le
    nom ne suit pas la convention `gen-<entier>.pt`."""
    match = _GENERATION_PATTERN.match(Path(path).name)
    return int(match.group(1)) if match else None


class Trainer:
    def __init__(self, games_dir, checkpoint_dir, net_factory=lambda: PolicyValueNet()):
        self.games_dir = Path(games_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.net = net_factory()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.generation = 0
        self._consumed_games = set()
        self._examples = []  # (planes_placeholder, policy_target, value_target)

    def consume_new_games(self):
        new_files = sorted(
            f for f in self.games_dir.glob("game-*.json") if f.name not in self._consumed_games
        )
        for f in new_files:
            data = json.loads(f.read_text())
            outcome = data["outcome"]
            for action_idx, visits in data["moves"]:
                policy_target = torch.zeros(ACTION_SPACE)
                policy_target[action_idx] = 1.0  # cible simplifiée : un seul coup joué
                # LIMITATION CONNUE, NON RÉSOLUE — placeholder, pas une
                # entrée réelle : `torch.zeros(NUM_PLANES, 5, 5)` tient lieu
                # d'état de plateau encodé. Le réseau reçoit donc exactement
                # la même entrée constante pour *tous* les exemples, quelle
                # que soit la position réellement jouée, et ne peut rien
                # apprendre qui dépende de la position : il ne peut au mieux
                # qu'ajuster un biais vers la distribution marginale des
                # coups et des résultats. Toute mesure de force de jeu du
                # candidat avant le remplacement de ce placeholder mesure un
                # MCTS à priors quasi constants, pas un AlphaZero entraîné.
                # Pour le lever il faut rejouer chaque partie depuis sa
                # position de départ en appliquant les coups du
                # `GameRecord` et encoder chaque position intermédiaire
                # (`santoni_engine::mcts::encode_planes`, PLANE_BYTES=150 =
                # 6 plans de 5x5) — le format de partie actuel ne stocke que
                # `(action_index, total_visits)` par tour, ce qui suffit à
                # cette reconstruction mais exige de l'implémenter.
                # Deuxième limitation, du même ordre et distincte : la cible
                # de politique one-hot ci-dessus (voir le commentaire de la
                # Task 9 du plan) au lieu de la distribution des visites sur
                # tous les coups légaux du tour.
                self._examples.append((torch.zeros(NUM_PLANES, 5, 5), policy_target, float(outcome)))
            self._consumed_games.add(f.name)
        return len(new_files)

    def train_step(self, batch_size=64):
        if not self._examples:
            return 0.0
        batch = self._examples[:batch_size]
        planes = torch.stack([p for p, _, _ in batch])
        policy_targets = torch.stack([pt for _, pt, _ in batch])
        value_targets = torch.tensor([vt for _, _, vt in batch]).unsqueeze(1)
        policy_logits, value_pred = self.net(planes)
        policy_loss = F.cross_entropy(policy_logits, policy_targets.argmax(dim=1))
        value_loss = F.mse_loss(value_pred, value_targets)
        loss = policy_loss + value_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        del self._examples[:len(batch)]
        return float(loss.item())

    def save_checkpoint(self):
        self.generation += 1
        path = self.checkpoint_dir / f"gen-{self.generation:04d}.pt"
        tmp = path.with_suffix(".pt.tmp")
        torch.save({
            "generation": self.generation,
            "model_state": self.net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "consumed_games": sorted(self._consumed_games),
            "examples": self._examples,
        }, tmp)
        os.replace(tmp, path)
        self._prune_checkpoints()
        return str(path)

    def _prune_checkpoints(self):
        """Ne garde que les `CHECKPOINT_RETENTION` checkpoints de génération
        la plus élevée. Appelé seulement après un `os.replace` réussi, donc
        jamais au point de supprimer l'ancien checkpoint avant que le nouveau
        soit intégralement visible sous son nom final — la reprise reste
        possible à tout instant, y compris si le processus meurt entre les
        deux. Le tri est numérique (cf. `load_latest_checkpoint`), pas
        lexical, sinon `gen-10000.pt` serait considéré comme plus ancien que
        `gen-9999.pt` et serait supprimé en premier."""
        numbered = [(generation, candidate)
                    for candidate in self.checkpoint_dir.glob("gen-*.pt")
                    for generation in [checkpoint_generation(candidate)]
                    if generation is not None]
        numbered.sort(key=lambda item: item[0])
        stale_entries = numbered[:-CHECKPOINT_RETENTION] if CHECKPOINT_RETENTION > 0 else []
        for _, stale in stale_entries:
            try:
                stale.unlink()
            except FileNotFoundError:
                pass  # déjà retiré (autre processus) : rien à faire

    def load_latest_checkpoint(self):
        # Sélection par génération *numérique*, pas par ordre lexical des
        # noms de fichiers : `sorted(glob("gen-*.pt"))[-1]` renvoyait
        # `gen-9999.pt` alors que `gen-10000.pt` existait ("gen-10000.pt" <
        # "gen-9999.pt" en comparaison de chaînes), si bien que passé la
        # génération 9999 chaque reprise rechargeait silencieusement un
        # checkpoint périmé — et, le compteur de génération étant restauré
        # depuis ce fichier, la sauvegarde suivante réécrasait les
        # générations déjà produites. Les noms non conformes à
        # `gen-<entier>.pt` sont ignorés plutôt que de faire échouer la
        # reprise (ex. un `.pt.tmp` orphelin ne matche de toute façon pas le
        # glob, mais un fichier manuellement déposé pourrait).
        candidates = [(generation, path)
                      for path in self.checkpoint_dir.glob("gen-*.pt")
                      for generation in [checkpoint_generation(path)]
                      if generation is not None]
        if not candidates:
            return 0
        latest = max(candidates, key=lambda item: item[0])[1]
        data = torch.load(latest, weights_only=False)
        self.net.load_state_dict(data["model_state"])
        self.optimizer.load_state_dict(data["optimizer_state"])
        self.generation = data["generation"]
        self._consumed_games = set(data["consumed_games"])
        self._examples = data["examples"]
        return self.generation

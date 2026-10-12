import json
import os
import random
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

# Fenêtre glissante du tampon de rejeu, en nombre de *positions* (pas de
# parties) : ~5000 parties récentes à la moyenne mesurée de 14,8
# positions/partie sur le premier run réel (mémoire
# santoni-mcts-az-no-power-chantier, 2026-10-11, checkpoint gen-5765).
# Approximation volontaire par comptage de positions plutôt que de parties
# exactes : c'est le nombre de positions qui compte pour la mémoire et
# l'échantillonnage, suivre des frontières de parties exactes n'apporterait
# rien ici. Avant ce correctif, `train_step` supprimait chaque exemple après
# un seul pas de gradient (jamais revu) — un mauvais régime pour une cible
# de valeur intrinsèquement bruitée (l'issue d'une seule partie) : diagnostic
# empirique sur `gen-5765.pt` montrant une tête de valeur au niveau du
# hasard (53,3% de concordance de signe, contre 54,4% pour un réseau non
# entraîné) malgré une confiance extrême.
MAX_BUFFER_POSITIONS = 74_000

_GENERATION_PATTERN = re.compile(r'^gen-(\d+)\.pt$')


def checkpoint_generation(path):
    """Numéro de génération porté par un nom de checkpoint, ou `None` si le
    nom ne suit pas la convention `gen-<entier>.pt`."""
    match = _GENERATION_PATTERN.match(Path(path).name)
    return int(match.group(1)) if match else None


class Trainer:
    def __init__(self, games_dir, checkpoint_dir, net_factory=lambda: PolicyValueNet(),
                 max_buffer_positions=MAX_BUFFER_POSITIONS):
        self.games_dir = Path(games_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.net = net_factory()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.generation = 0
        self.max_buffer_positions = max_buffer_positions
        self._consumed_games = set()
        self._examples = []  # (planes, policy_target, value_target)

    @staticmethod
    def _decode_position(position, ply, outcome):
        """Décode une position d'un `GameRecord` (`mcts_io.rs`) en exemple
        d'entraînement. Partagée entre `consume_new_games` et la
        reconstruction du tampon au démarrage (Task 2) : une seule
        implémentation de la convention de signe par ply, jamais dupliquée."""
        planes = torch.tensor(position["planes"], dtype=torch.float32).view(NUM_PLANES, 5, 5)
        total_visits = sum(visits for _, visits in position["visits"])
        policy_target = torch.zeros(ACTION_SPACE)
        for action_idx, visits in position["visits"]:
            policy_target[action_idx] = visits / total_visits
        # Alternance stricte joueur 0 / joueur 1 à chaque ply : vraie pour
        # toute la partie dans le sous-jeu sans pouvoir, où `is_in_scope`
        # (native_engine/src/mcts.rs) impose `powers == [0, 0]` sur toute
        # position rencontrée. `outcome` est fixé du point de vue du
        # joueur 0 (cf. `play_one_game`) ; ce signe le reprojette depuis le
        # point de vue du joueur réellement au trait à ce ply.
        value_target = outcome if ply % 2 == 0 else -outcome
        return planes, policy_target, float(value_target)

    def consume_new_games(self):
        new_files = sorted(
            f for f in self.games_dir.glob("game-*.json") if f.name not in self._consumed_games
        )
        for f in new_files:
            try:
                data = json.loads(f.read_text())
            except (json.JSONDecodeError, OSError):
                # Fichier corrompu ou vide (ex. écrit sans fsync avant le
                # renommage atomique, cf. `mcts_io::write_game_atomically`,
                # et coupure de courant) : marqué consommé pour ne jamais
                # être retenté, mais aucun exemple n'en est tiré.
                self._consumed_games.add(f.name)
                continue
            outcome = data["outcome"]
            for ply, position in enumerate(data["positions"]):
                self._examples.append(self._decode_position(position, ply, outcome))
            self._consumed_games.add(f.name)
        excess = len(self._examples) - self.max_buffer_positions
        if excess > 0:
            del self._examples[:excess]
        return len(new_files)

    def train_step(self, batch_size=64):
        if not self._examples:
            return 0.0
        batch = random.sample(self._examples, min(batch_size, len(self._examples)))
        planes = torch.stack([p for p, _, _ in batch])
        policy_targets = torch.stack([pt for _, pt, _ in batch])
        value_targets = torch.tensor([vt for _, _, vt in batch]).unsqueeze(1)
        policy_logits, value_pred = self.net(planes)
        policy_loss = F.cross_entropy(policy_logits, policy_targets)
        value_loss = F.mse_loss(value_pred, value_targets)
        loss = policy_loss + value_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def save_checkpoint(self):
        self.generation += 1
        path = self.checkpoint_dir / f"gen-{self.generation:04d}.pt"
        tmp = path.with_suffix(".pt.tmp")
        torch.save({
            "generation": self.generation,
            "model_state": self.net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
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
        if candidates:
            latest = max(candidates, key=lambda item: item[0])[1]
            data = torch.load(latest, weights_only=False)
            self.net.load_state_dict(data["model_state"])
            self.optimizer.load_state_dict(data["optimizer_state"])
            self.generation = data["generation"]
        # Le tampon n'est jamais sérialisé dans le checkpoint (voir
        # `_rebuild_buffer_from_disk`) : il se reconstruit toujours depuis
        # les fichiers de parties sur disque, qu'un checkpoint modèle ait
        # été trouvé ci-dessus ou non.
        self._rebuild_buffer_from_disk()
        return self.generation

    def _rebuild_buffer_from_disk(self):
        """Reconstruit `_examples`/`_consumed_games` depuis les fichiers de
        parties déjà sur disque, jamais depuis le checkpoint (qui ne les
        contient plus) -- ils restent la seule source de vérité durable
        pour les données d'entraînement (§3.2 du spec
        2026-10-11-trainer-replay-buffer-design.md). Les noms de fichiers
        (`game-{pid}-{:06}.json`) ne sont pas triables chronologiquement
        entre workers distincts (chaque worker a son propre compteur
        local, cf. `selfplay_worker.rs::main`) : on trie par date de
        modification. Tous les fichiers trouvés sont marqués consommés,
        même ceux dont le contenu n'est pas chargé dans le tampon, sinon un
        futur `consume_new_games()` réingérerait tout l'historique plus
        ancien que la fenêtre dès le prochain appel. Ne lit le *contenu*
        que des fichiers retenus dans la fenêtre -- le temps de démarrage
        reste borné indépendamment de la taille totale du corpus déjà
        accumulé.
        """
        self._examples = []
        self._consumed_games = set()
        all_files = sorted(
            self.games_dir.glob("game-*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        )
        kept_in_order = []
        total = 0
        for f in all_files:
            self._consumed_games.add(f.name)
            if total < self.max_buffer_positions:
                try:
                    data = json.loads(f.read_text())
                except (json.JSONDecodeError, OSError):
                    # Même raison que dans `consume_new_games` : un fichier
                    # corrompu ou vide ne doit pas bloquer tout redémarrage.
                    continue
                kept_in_order.append(data)
                total += len(data["positions"])
        for data in reversed(kept_in_order):
            outcome = data["outcome"]
            for ply, position in enumerate(data["positions"]):
                self._examples.append(self._decode_position(position, ply, outcome))
        excess = len(self._examples) - self.max_buffer_positions
        if excess > 0:
            del self._examples[:excess]

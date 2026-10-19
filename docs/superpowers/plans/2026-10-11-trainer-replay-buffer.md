# Tampon de rejeu pour Trainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remplacer la file d'entraînement à passage unique de `Trainer` (chaque exemple supprimé après un seul pas de gradient) par un vrai tampon de rejeu : fenêtre glissante des positions les plus récentes, échantillonnage aléatoire répété, sans suppression après usage.

**Architecture:** `Trainer._examples` devient une liste plafonnée en nombre de positions (`MAX_BUFFER_POSITIONS`), remplie par `consume_new_games()` (éviction des plus anciennes au-delà du plafond) et échantillonnée aléatoirement (sans suppression) par `train_step()`. Le tampon et `_consumed_games` ne sont plus jamais sérialisés dans le checkpoint — ils se reconstruisent entièrement depuis les fichiers de parties déjà sur disque à chaque démarrage, triés par date de modification (les noms de fichiers ne sont pas chronologiquement comparables entre workers).

**Tech Stack:** Python/PyTorch (`santoni_az`, `pytest`). Aucun changement à `native_engine`.

**Spec:** `docs/superpowers/specs/2026-10-11-trainer-replay-buffer-design.md`

## Global Constraints

- Confiné à `santoni_az/trainer.py` et `tests/test_az_trainer.py` — aucun changement à `native_engine`, à `Orchestrator` (signatures `consume_new_games() -> int`, `train_step(batch_size=64) -> float`, `load_latest_checkpoint() -> int`, `save_checkpoint() -> str` inchangées), ni au format sur disque des parties (`mcts_io.rs`).
- `MAX_BUFFER_POSITIONS = 74_000` (constante du module, approximation de ~5000 parties récentes à 14,8 positions/partie en moyenne mesurée).
- Le format de checkpoint perd les clés `"examples"` et `"consumed_games"`.
- Les noms de fichiers de parties (`game-{pid}-{:06}.json`) ne sont **pas** triables chronologiquement entre workers distincts (chaque worker a son propre compteur local) — toute sélection "les parties les plus récentes" doit trier par `mtime`, jamais par nom.

## Review Focus

- Un répertoire de parties totalement vide au redémarrage (tout premier lancement) ne doit pas faire planter la reconstruction du tampon — couvert par un test dédié dans Task 2.
- `train_step` ne doit jamais lever d'exception quand `batch_size` (64 par défaut) dépasse le nombre d'exemples réellement présents dans un tampon encore petit — couvert par Task 1.
- Une partie dont le nombre de positions dépasse à elle seule le plafond du tampon doit être tronquée par le même mécanisme générique d'éviction, sans cas particulier — couvert indirectement par le test d'éviction de Task 1 (plafond injecté à 1 position).
- Des égalités de `mtime` entre fichiers de parties écrits presque simultanément par des dizaines de workers concurrents sont résolues par un tri stable, sans erreur — examiné, pas un vrai mode de défaillance, aucun test dédié nécessaire.
- `_decode_position` doit produire exactement la même cible (politique, valeur, convention de signe par ply) qu'elle soit appelée depuis `consume_new_games` ou depuis la reconstruction au démarrage — garanti par extraction d'une seule implémentation partagée (Task 1) plutôt que par duplication, pas de test séparé nécessaire.

---

## Task 1 : Tampon plafonné + échantillonnage aléatoire sans suppression

**Files:**
- Modify: `santoni_az/trainer.py` (constante, `Trainer.__init__`, nouvelle méthode `_decode_position`, `consume_new_games`, `train_step`)
- Test: `tests/test_az_trainer.py`

**Interfaces:**
- Consumes: `santoni_az.network.{PolicyValueNet, ACTION_SPACE, NUM_PLANES}` (inchangés).
- Produces: `MAX_BUFFER_POSITIONS = 74_000` (constante exportée du module) ; `Trainer.__init__(self, games_dir, checkpoint_dir, net_factory=..., max_buffer_positions=MAX_BUFFER_POSITIONS)` (nouveau paramètre optionnel, défaut = la constante du module) ; `Trainer._decode_position(position: dict, ply: int, outcome: int) -> tuple[Tensor, Tensor, float]` (méthode statique, réutilisée par Task 2) ; `train_step`/`consume_new_games` gardent leurs signatures existantes.

- [ ] **Step 1: Write the failing tests**

Dans `tests/test_az_trainer.py`, remplacer l'import de `Trainer` pour inclure la nouvelle constante, et ajouter ces trois tests après `test_checkpoint_resume_preserves_generation_counter` :

```python
from santoni_az.trainer import CHECKPOINT_RETENTION, MAX_BUFFER_POSITIONS, Trainer, checkpoint_generation
```

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_az_trainer.py -v -k "does_not_delete or batch_size_larger or evicts_oldest"`
Expected : `test_consume_new_games_evicts_oldest_positions_beyond_the_buffer_cap` échoue à l'import (`ImportError: cannot import name 'MAX_BUFFER_POSITIONS'` ou `TypeError: __init__() got an unexpected keyword argument 'max_buffer_positions'`) — la constante et le paramètre n'existent pas encore. `test_train_step_does_not_delete_examples_after_training` échoue (`len(trainer._examples) == 0` après coup, pas `before`) — le code actuel supprime encore les exemples après usage. `test_train_step_handles_batch_size_larger_than_buffer` passe probablement déjà (tranche `[:64]` sur 1 élément ne lève pas), mais sera revérifié à l'étape 4 une fois `random.sample` en place.

- [ ] **Step 3: Write the implementation**

Dans `santoni_az/trainer.py`, ajouter `import random` aux imports du haut, puis ajouter la nouvelle constante juste après `CHECKPOINT_RETENTION` :

```python
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
```

Remplacer `Trainer.__init__` :

```python
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
```

Ajouter la méthode `_decode_position` (juste avant `consume_new_games`) :

```python
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
```

Remplacer `consume_new_games` :

```python
    def consume_new_games(self):
        new_files = sorted(
            f for f in self.games_dir.glob("game-*.json") if f.name not in self._consumed_games
        )
        for f in new_files:
            data = json.loads(f.read_text())
            outcome = data["outcome"]
            for ply, position in enumerate(data["positions"]):
                self._examples.append(self._decode_position(position, ply, outcome))
            self._consumed_games.add(f.name)
        excess = len(self._examples) - self.max_buffer_positions
        if excess > 0:
            del self._examples[:excess]
        return len(new_files)
```

Remplacer `train_step` :

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_az_trainer.py -v`
Expected : PASS sur l'ensemble du fichier, y compris les trois nouveaux tests et tous les tests préexistants (`test_checkpoint_resume_preserves_untrained_examples` doit encore passer à ce stade — Task 2 le remplacera, pas cette tâche).

- [ ] **Step 5: Commit**

```bash
git add santoni_az/trainer.py tests/test_az_trainer.py
git commit -m "Tampon de rejeu plafonné + échantillonnage aléatoire sans suppression dans Trainer"
```

---

## Task 2 : Checkpoint allégé + reconstruction du tampon depuis le disque au démarrage

**Files:**
- Modify: `santoni_az/trainer.py` (`save_checkpoint`, `load_latest_checkpoint`, nouvelle méthode `_rebuild_buffer_from_disk`)
- Test: `tests/test_az_trainer.py` (suppression de `test_checkpoint_resume_preserves_untrained_examples`, ajout de quatre nouveaux tests)

**Interfaces:**
- Consumes: `Trainer._decode_position` (Task 1), `Trainer.max_buffer_positions` (Task 1), `checkpoint_generation` (inchangé).
- Produces: `save_checkpoint()` garde sa signature `() -> str` mais le dictionnaire sauvegardé ne contient plus `"examples"` ni `"consumed_games"` ; `load_latest_checkpoint()` garde `() -> int` mais reconstruit toujours `_examples`/`_consumed_games` depuis le disque, qu'un fichier checkpoint existe ou non ; `Trainer._rebuild_buffer_from_disk(self) -> None` (nouvelle méthode privée, pas d'usage externe prévu).

- [ ] **Step 1: Write the failing tests**

Dans `tests/test_az_trainer.py`, ajouter `import os` en haut du fichier (utilisé ci-dessous pour forcer la date de modification d'un fichier de test via `os.utime`), puis **supprimer entièrement** `test_checkpoint_resume_preserves_untrained_examples` (sa prémisse — un exemple non entraîné doit survivre tel quel à une reprise via le checkpoint — disparaît avec ce plan). La remplacer par ces quatre tests, au même emplacement :

```python
import os
```

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_az_trainer.py -v -k "rebuilds_buffer or no_longer_persists"`
Expected : FAIL sur les quatre nouveaux tests. `test_load_latest_checkpoint_rebuilds_buffer_only_from_most_recent_games` et `..._even_without_a_checkpoint_file` échouent avec `resumed._examples == []`/`trainer._examples == []` (rien ne reconstruit encore le tampon depuis le disque). `test_load_latest_checkpoint_handles_no_games_on_disk_yet` passe probablement déjà par accident (`_examples` reste `[]` de toute façon tant que rien ne le remplit). `test_save_checkpoint_no_longer_persists_examples_or_consumed_games` échoue (les deux clés sont encore présentes).

- [ ] **Step 3: Write the implementation**

Dans `santoni_az/trainer.py`, remplacer `save_checkpoint` :

```python
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
```

Remplacer `load_latest_checkpoint` :

```python
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
        # reprise.
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
                data = json.loads(f.read_text())
                kept_in_order.append(data)
                total += len(data["positions"])
        for data in reversed(kept_in_order):
            outcome = data["outcome"]
            for ply, position in enumerate(data["positions"]):
                self._examples.append(self._decode_position(position, ply, outcome))
        excess = len(self._examples) - self.max_buffer_positions
        if excess > 0:
            del self._examples[:excess]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_az_trainer.py -v`
Expected : PASS sur l'ensemble du fichier — les quatre nouveaux tests, les trois tests de Task 1, et tous les tests préexistants restants (`test_checkpoint_resume_preserves_generation_counter` notamment : son assertion `resumed.consume_new_games() == 0` doit continuer à tenir, maintenant parce que `_rebuild_buffer_from_disk` marque les deux parties consommées au redémarrage, plus par un rechargement explicite de `consumed_games` depuis le checkpoint).

- [ ] **Step 5: Commit**

```bash
git add santoni_az/trainer.py tests/test_az_trainer.py
git commit -m "Checkpoint allégé + reconstruction du tampon de rejeu depuis le disque au démarrage"
```

---

## Final check (not a task — run after Task 2)

```bash
QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_az_trainer.py tests/test_az_network.py tests/test_az_orchestrator.py tests/test_az_inference_server.py -v
```

Confirme l'absence de régression sur tout `santoni_az/` (y compris `Orchestrator`, qui n'a besoin d'aucun changement de code mais dont les tests exercent `Trainer` par transitivité via `load_latest_checkpoint`/`consume_new_games`/`train_step`).

Puis, avant de relancer une nouvelle campagne de self-play + entraînement longue sur ce tampon corrigé : mettre à jour la mémoire `santoni-mcts-az-no-power-chantier` avec le résultat de ce plan, et envisager de rejouer le diagnostic déjà utilisé sur `gen-5765.pt` (`selfplay-runs/diagnose_net.py`) après quelques heures du nouveau régime d'entraînement, pour vérifier que la tête de valeur sort enfin du niveau du hasard avant de relancer un nouveau pilote `--limit 20`.

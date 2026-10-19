# Tampon de rejeu pour l'entraînement AlphaZero — design

Date : 11 octobre 2026. Suite directe du premier run réel de self-play +
entraînement sur le sous-jeu sans pouvoir (mémoire
`santoni-mcts-az-no-power-chantier`) : 128 757 parties, 9h01, checkpoint
`gen-5765.pt`. Diagnostic empirique sur ce checkpoint (comparaison avec un
réseau non entraîné sur 180 positions réelles échantillonnées) :

- Tête de valeur : 53,3 % de concordance de signe avec l'issue réelle,
  contre 54,4 % pour un réseau aux poids aléatoires — indiscernable du
  hasard — tout en étant extrêmement confiante (|valeur| moyenne 0,935
  contre 0,111 pour le réseau aléatoire). Le réseau prédit souvent +0,999
  alors que l'issue réelle est -1.
- Tête de politique : entropie tombée de 7,13 (niveau d'un réseau neuf,
  identique à l'entropie maximale sur 1250 actions) à 0,755, probabilité
  maximale moyenne de 80 % sur une seule action.
- Pilote réel (`--limit 20`, 5s/coup) contre le moteur de référence :
  0 victoire candidate sur 20.

**Hypothèse retenue** : `Trainer.train_step()` consomme et supprime
définitivement chaque exemple après un seul pas de gradient
(`del self._examples[:len(batch)]`, `santoni_az/trainer.py:86`) — chaque
position n'est vue exactement qu'une fois, jamais revue. Pour une cible de
valeur intrinsèquement bruitée (l'issue d'une seule partie, pas une
moyenne), un seul passage par exemple est un régime d'apprentissage
structurellement faible : chaque pas de gradient sur-ajuste fortement au
dernier exemple vu, sans jamais généraliser sur plusieurs expositions à des
positions similaires — cohérent avec le schéma observé (confiance extrême,
exactitude au niveau du hasard).

## 1. Objectif

Remplacer la file à passage unique de `Trainer` par un vrai tampon de
rejeu : fenêtre glissante des positions les plus récentes, échantillonnage
aléatoire répété à chaque pas d'entraînement, sans suppression après usage.
Chaque position reste éligible à être ré-échantillonnée jusqu'à son
éviction par la fenêtre glissante.

## 2. Portée

Confiné à `santoni_az/trainer.py` et ses tests
(`tests/test_az_trainer.py`). Aucun changement à `native_engine`, à
`Orchestrator` (les signatures `consume_new_games() -> int`,
`train_step(batch_size=64) -> float`, `load_latest_checkpoint() -> int`,
`save_checkpoint() -> str` restent inchangées — `Orchestrator._train_loop`
n'a besoin d'aucune modification), ni au format binaire des parties sur
disque.

## 3. Architecture

### 3.1 Le tampon

`Trainer._examples` devient une liste plate `(planes, policy_target,
value_target)`, plafonnée en **nombre de positions** —
`MAX_BUFFER_POSITIONS = 74_000` (nouvelle constante du module, au même
niveau que `CHECKPOINT_RETENTION`), choisi comme approximation de "~5000
parties récentes" à partir de la moyenne mesurée de 14,8 positions/partie
sur le run réel. C'est volontairement une approximation par comptage de
positions, pas un comptage exact de parties — documenté en commentaire,
pour ne jamais être pris plus tard pour une garantie de fenêtre-en-parties
exacte.

`consume_new_games()` ajoute les nouvelles positions en fin de liste
(inchangé). Si `len(self._examples) > MAX_BUFFER_POSITIONS` après ajout, les
positions excédentaires sont retirées en tête (`del
self._examples[:excedent]`) — "tête de liste = plus ancien" reste
l'invariant, maintenu par construction tant que l'ordre d'ajout suit
l'ordre chronologique (vrai en fonctionnement normal ; voir §3.2 pour le
cas du redémarrage).

### 3.2 Reconstruction au redémarrage (jamais sérialisé dans le checkpoint)

Le tampon et `_consumed_games` ne sont **jamais** écrits dans le
checkpoint. Ils se reconstruisent entièrement depuis les fichiers de
parties déjà sur disque — cohérent avec le principe déjà établi dans ce
chantier que les fichiers de parties, écrits atomiquement, sont la seule
source de vérité durable, pas les checkpoints.

`load_latest_checkpoint()` (renommage de rôle, pas de signature — elle
charge toujours le dernier checkpoint *modèle* si un existe, et renvoie
toujours la génération, `0` si aucun) exécute en plus, systématiquement,
qu'un fichier checkpoint existe ou non :

1. Lister tous les `game-*.json` de `games_dir`, trier par `mtime`
   décroissante (le nom de fichier `game-{pid}-{:06}.json` n'est **pas**
   triable chronologiquement entre workers distincts — chaque worker a son
   propre compteur local repartant de 0, cf. le commentaire déjà présent
   dans `selfplay_worker.rs::main`).
2. Lire le contenu des fichiers les plus récents, dans l'ordre
   chronologique croissant (le plus ancien des fichiers retenus en
   premier), jusqu'à accumuler `MAX_BUFFER_POSITIONS` positions ou
   épuisement des fichiers disponibles. Ce contenu initialise
   `self._examples`.
3. Marquer **tous** les fichiers listés à l'étape 1 dans
   `_consumed_games` — y compris ceux dont le contenu n'a pas été chargé à
   l'étape 2. Sans ça, un futur `consume_new_games()` réingérerait tout
   l'historique de parties plus anciennes que la fenêtre au prochain appel,
   puisque son filtre ne connaît que ce qui est dans `_consumed_games`.

L'étape 1/3 ne lit que des métadonnées de nom/date (coût borné même sur un
corpus de 100k+ fichiers) ; seule l'étape 2 lit du contenu JSON, et
seulement pour le sous-ensemble retenu — le temps de démarrage reste borné
indépendamment de la taille totale du corpus déjà accumulé.

### 3.3 Échantillonnage

`train_step(batch_size=64)` : remplace `batch = self._examples[:batch_size]`
(puis suppression) par `batch = random.sample(self._examples,
min(batch_size, len(self._examples)))`, sans suppression après usage. Un
exemple reste dans `self._examples` et éligible à un futur tirage jusqu'à
son éviction par la fenêtre glissante du §3.1. Aucun poids de priorité ni
d'échantillonnage pondéré par récence — tirage uniforme simple, le choix le
plus simple compatible avec l'objectif, à reconsidérer seulement si une
limite s'en dégage empiriquement.

## 4. Gestion d'erreur et cas limites

- Tampon plus petit que `batch_size` (tout début de run) :
  `min(batch_size, len(self._examples))` évite `random.sample` de lever
  `ValueError` pour `k > population`.
- Tampon vide : `train_step()` renvoie `0.0`, comportement inchangé.
- Moins de parties disponibles sur disque que la fenêtre cible au
  redémarrage : charge tout ce qu'il y a, pas d'erreur.
- Le format du checkpoint perd les clés `"examples"` et
  `"consumed_games"` — règle au passage la croissance de taille déjà
  signalée en commentaire dans `trainer.py` (`CHECKPOINT_RETENTION`),
  puisque ces deux champs en étaient la cause principale.

## 5. Hors périmètre (explicitement)

- Tout changement à la taille ou à l'architecture du réseau
  (`PolicyValueNet`) : ce plan ne traite que le régime d'échantillonnage de
  l'entraînement, pas la capacité du modèle. Si le problème persiste après
  ce correctif, la capacité devient une hypothèse à réexaminer séparément.
- Échantillonnage pondéré/prioritisé (replay prioritisé façon DQN) :
  tirage uniforme simple pour commencer, YAGNI tant que rien ne démontre
  que c'est insuffisant.
- `batch_size` par défaut (64) : inchangé, pas l'objet de ce plan.
- Toute modification à `native_engine` ou au format sur disque des
  parties (`mcts_io.rs`) : aucune n'est nécessaire ici.

## 6. Validation

Tests dans `tests/test_az_trainer.py` (remplacent ceux dont la prémisse de
passage unique disparaît, notamment
`test_checkpoint_resume_preserves_untrained_examples`) :

1. Un même exemple peut être tiré par plusieurs `train_step()` successifs
   sans être supprimé du tampon.
2. `batch_size` supérieur au nombre d'exemples disponibles ne lève pas
   d'exception.
3. Au-delà de `MAX_BUFFER_POSITIONS`, les positions les plus anciennes sont
   évincées — vérifié via un marqueur distinctif planté dans les
   positions les plus anciennes vs les plus récentes.
4. Au redémarrage (`load_latest_checkpoint()`), seules les parties les
   plus récentes par `mtime` alimentent le contenu du tampon ; les plus
   anciennes n'y sont pas mais sont bien marquées consommées.
5. Le dictionnaire sauvegardé par `save_checkpoint()` ne contient plus les
   clés `"examples"` ni `"consumed_games"`.
6. `test_checkpoint_resume_preserves_generation_counter` adapté :
   `consume_new_games()` renvoie toujours `0` après une reprise sans
   nouvelle partie écrite depuis.
7. `consume_new_games()` n'ajoute pas en double une partie déjà chargée
   par la reconstruction de démarrage.

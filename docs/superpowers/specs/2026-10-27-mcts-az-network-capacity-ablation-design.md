# Spec : ablation de capacité réseau (MCTS+NN, sous-jeu sans pouvoir)

## Contexte

Le chantier MCTS+NN (mémoire `santoni-mcts-az-no-power-chantier`) a maintenant
testé, à chaque fois sans effet mesurable : `simulations_per_move` (32→128),
le protocole d'entraînement (self-play pur → ancre vécue fixe contre le
moteur de référence). Dans toutes ces variantes, la taille du réseau
(`PolicyValueNet`, 4 blocs résiduels / 32 canaux, `santoni_az/network.py`)
est restée constante — c'est la seule variable jamais isolée. Le run le plus
récent et le plus propre (`selfplay-runs/anchor-pilot-20261021/`, 310 277
parties d'entraînement, taux de victoire d'évaluation 0/N sur les 2340
lignes de statut du run entier) donne un corpus déjà sur disque, suffisant
pour tester l'hypothèse de capacité sans nouveau calcul de self-play.

## Objectif

Déterminer si la taille du réseau actuel plafonne la qualité apprise, en
ré-entraînant hors-ligne plusieurs architectures de tailles différentes sur
les *mêmes* positions déjà enregistrées, et en comparant leur perte et leur
concordance de signe (valeur) sur un ensemble de parties tenues à l'écart
(jamais vues à l'entraînement).

Résultat utilisé pour trancher entre deux branches (voir conversation
utilisateur du 2026-10-27, pas re-dérivé ici) :
- amélioration nette avec un réseau plus gros → la capacité était le
  plafond ; relancer le self-play avec cette architecture devient le prochain
  chantier réel.
- aucune amélioration même avec un réseau beaucoup plus gros → la capacité
  est écartée aussi ; le soupçon se déplace vers la qualité des cibles
  d'entraînement (distributions de visites MCTS à 32-128 simulations) ou
  vers l'approche MCTS+NN elle-même pour ce sous-jeu.

## Hors de portée

- Aucun nouveau self-play, aucun calcul GPU. Script purement CPU, off-line,
  sur des données déjà sur disque.
- Ne décide pas elle-même quelle branche suivre — produit seulement les
  chiffres nécessaires à la décision (confiée à l'utilisateur).
- Pas de nouvelle architecture de production, pas de modification de
  `Trainer`/`Orchestrator`/`PolicyValueNet` existants.

## Conception

Nouveau script autonome `tools/network_capacity_ablation.py`, important mais
ne modifiant pas `santoni_az.trainer.Trainer._decode_position` (méthode
statique déjà pure, réutilisée telle quelle pour éviter une deuxième
implémentation de la convention de signe par ply) ni
`santoni_az.network.PolicyValueNet` (déjà paramétrée par `channels`/`blocks`,
aucune modification nécessaire).

### Chargement et partition des données

- `list_game_files(games_dir) -> list[Path]` : liste triée (ordre de nom,
  déterministe) des `game-*.json` d'un répertoire de run existant (ex.
  `selfplay-runs/anchor-pilot-20261021/games/`).
- `split_train_holdout(game_files, holdout_fraction, seed) -> (train_files, holdout_files)` :
  partition **au niveau de la partie**, pas de la position — les positions
  d'une même partie partagent une trajectoire corrélée (même ancre, mêmes
  premiers coups), les mélanger entre train et held-out fuiterait du signal.
  Mélange déterministe (`random.Random(seed).shuffle`) puis coupe à
  `round(len(game_files) * holdout_fraction)`.
- `load_examples(games_dir, game_files, max_positions, seed) -> list[(planes, policy_target, value_target)]` :
  tire aléatoirement (`seed` fixe) des fichiers de `game_files` sans
  remise, décode chaque position via `Trainer._decode_position`, s'arrête
  dès que `max_positions` exemples sont chargés (budget mémoire borné —
  le corpus réel a ~4,6M positions, bien au-delà de ce qu'il est utile ou
  nécessaire de matérialiser pour cette ablation). Fichier illisible/corrompu
  (même risque que `Trainer.consume_new_games` — écriture non-fsync avant
  renommage atomique) : ignoré silencieusement, comme le code existant.

### Entraînement et évaluation par architecture

- `build_net(channels, blocks) -> PolicyValueNet` : simple wrapper nommé pour
  la lisibilité du reste du script, aucune logique propre.
- `train_net(net, examples, steps, batch_size, seed) -> list[float]` : boucle
  d'entraînement supervisé pur sur `examples` fixe (pas de self-play, pas de
  tampon glissant) — même composition de perte que
  `Trainer.train_step` (`F.cross_entropy` + `F.mse_loss`, `Adam(lr=1e-3)`),
  dupliquée ici volontairement plutôt qu'en extrayant `Trainer.train_step`
  en méthode partagée : `Trainer` est couplé à son tampon interne
  (`self._examples`, `self.net`, l'auto-incrément de checkpoint) et cette
  ablation a besoin d'entraîner plusieurs réseaux indépendants sur une liste
  d'exemples fixe en mémoire — une dépendance vers `Trainer` ajouterait du
  couplage sans réduire de duplication réelle (la boucle elle-même est 5
  lignes). Retourne la perte de chaque pas, pour une courbe de convergence
  grossière dans le rapport final.
- `evaluate_net(net, held_out_examples) -> dict` avec les clés
  `policy_loss`, `value_loss`, `total_loss` (moyennes sur tout
  `held_out_examples`, réseau en `.eval()`, `torch.no_grad()`) et
  `value_sign_match_rate` (fraction des exemples où le signe de la
  prédiction de valeur concorde avec le signe de la cible — même définition
  que `selfplay-runs/diagnose_net.py`, réutilisée en logique, pas en import
  puisque ce script-jetable n'est pas un module).

### Point d'entrée CLI

`main()` (`argparse`) :
- `--games-dir` (défaut `selfplay-runs/anchor-pilot-20261021/games`)
- `--architectures` : liste de `"channels:blocks"`, défaut
  `["32:4", "64:8", "128:12"]` (le premier terme est la configuration de
  production actuelle — figure dans le rapport comme référence, pas
  seulement les tailles plus grosses)
- `--max-train-positions` (défaut `200000`), `--max-holdout-positions`
  (défaut `20000`), `--holdout-fraction` (défaut `0.1`, appliqué au nombre
  de *parties* avant l'échantillonnage de positions)
- `--steps` (défaut `2000`), `--batch-size` (défaut `64`, aligné sur
  `Trainer.train_step`)
- `--seed` (défaut `20261027`) — même graine pour la partition et pour
  l'échantillonnage, un seul réseau entraîné par architecture (pas de
  répétition multi-graine : budget de calcul volontairement serré, ce n'est
  pas la campagne finale)
- `--out` : chemin JSON du rapport (défaut
  `selfplay-runs/ablation-20261027/report.json`), écrit de façon atomique
  (fichier temporaire puis `os.replace`, même motif que
  `Trainer.save_checkpoint`)

Affiche un tableau récapitulatif sur stdout (une ligne par architecture :
channels, blocks, nombre de paramètres, perte totale held-out, concordance
de signe) en plus d'écrire le JSON complet (pertes d'entraînement incluses).

## Contraintes globales

- Aucune nouvelle dépendance externe (projet : zéro nouvelle crate/paquet
  sans raison forte ; ici tout est déjà `torch`/`argparse`/stdlib).
- Déterministe de bout en bout à graine fixée (mêmes positions, même ordre
  de batch, mêmes poids initiaux par architecture) pour que le rapport soit
  reproductible.
- Les fonctions de chargement/partition/entraînement/évaluation doivent être
  testables sans le vrai corpus de 310k parties (fixtures synthétiques,
  même format que `tests/test_az_trainer.py::write_fake_game`).

## Focus de revue

- Partition parties/positions : une position du même fichier `game-*.json`
  ne doit jamais apparaître à la fois dans le pool d'entraînement et dans
  le pool held-out (vérifier par ensemble de noms de fichiers, pas juste par
  nombre).
- `max_positions` atteint à mi-fichier : `load_examples` doit s'arrêter nettement
  à la limite demandée, pas déborder d'un fichier entier ni planter.
- Fichier de partie corrompu/vide dans l'échantillon : ne doit pas interrompre
  le chargement (même contrat que `Trainer.consume_new_games`).
- Architecture invalide en CLI (`--architectures` mal formé, ex.
  `"abc"` sans `:`) : message d'erreur clair, pas une trace Python brute.
- Écriture du rapport JSON interrompue (process tué en cours) : le fichier
  final ne doit jamais être partiellement écrit (écriture atomique).

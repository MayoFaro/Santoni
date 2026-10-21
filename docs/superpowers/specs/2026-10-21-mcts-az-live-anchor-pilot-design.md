# Pilote "ancre vécue" — le candidat apprend à gagner contre la référence depuis une position réelle (design)

Date : 21 octobre 2026. Fait suite au bilan complet du chantier MCTS+NN
(`docs/bilan-mcts-az-no-power-20261019.md`) et au mécanisme "noyau exact"
déjà spécé et implémenté (`docs/superpowers/specs/2026-10-19-mcts-az-proven-anchor-curriculum-design.md`,
mergé `2c30eca`) — voir mémoire `santoni-mcts-az-no-power-chantier.md` pour
l'historique complet, notamment l'entrée "PENDING 2026-10-21" où la
première vraie exécution du noyau exact a été interrompue après 12h14m
sans le moindre résultat.

**Ce document ne remplace pas le noyau exact** — il décrit une approche
alternative, plus directe, pour tester la même hypothèse de fond (un
ancrage externe validé peut-il corriger le collapse de self-play
diagnostiqué le 2026-10-19 ?) sans dépendre de la recherche de preuve
exhaustive (`tools/proven_anchor/anchor_games.py::find_anchor_games`) qui
s'est révélée beaucoup plus coûteuse que prévu. `tools/proven_anchor/` et
`tools/proven_anchor_curriculum.py` restent tels qu'ils sont, non touchés,
non utilisés par ce pilote.

## 0. Décision explicite de l'utilisateur, 2026-10-21

> "on met en place cette campagne, et on verra bien le bilan qu'on peut en
> tirer." — pas d'engagement préalable à élargir, ajouter des parties
> ancres supplémentaires, ou raccorder le self-play continu après coup :
> ce pilote teste l'hypothèse une fois, sur une seule ligne, et la suite
> se décide à la lecture du résultat, pas avant.

Exigence explicite, non négociable, directement motivée par l'incident du
12h14m sans visibilité : **la campagne doit pouvoir être interrompue à
tout moment sans perte de données.**

## 1. Objectif

Faire apprendre au réseau candidat (MCTS+NN, `santoni_az/`) à gagner,
depuis une position réelle où le moteur de référence (alpha-beta,
`native_search`) a effectivement gagné quelques coups plus tard dans une
vraie partie, en le faisant jouer à répétition contre ce même moteur de
référence depuis cette position exacte — jusqu'à ce qu'il y parvienne de
façon statistiquement fiable.

C'est la formulation originale du 2026-10-19 (mémoire du chantier, section
"reverse-curriculum"), avant que le "noyau exact" (preuve exhaustive,
couches de 6 demi-coups, validation croisée entre parties ancres) ne la
remplace. Ce pilote y revient délibérément, en plus simple : pas de preuve
formelle de gain forcé, pas de carte exhaustive des coups gagnants/perdants
à chaque position ancêtre — juste "ça a vraiment gagné depuis là, est-ce
que le candidat peut apprendre à en faire autant ?"

## 2. Portée

**Un seul fil conducteur.** Une seule partie de référence contre elle-même
sans aucun pouvoir, jouée une fois jusqu'au bout, fournit la position
ancre. Pas de pluralité de parties ancres, pas de validation croisée tenue
à l'écart (contrairement au noyau exact §2/§3.7) — hors de portée pour ce
document, à reconsidérer seulement si ce pilote est informatif.

**Hors périmètre, explicitement (reporté après le bilan, pas écarté) :**
- Reculer plus loin que 5 demi-coups dans la même partie une fois le
  seuil atteint (§3.6 de l'ancien design).
- Ajouter d'autres parties ancres en parallèle (§3.7 de l'ancien design).
- Raccorder le résultat au self-play continu existant (§3.5 de l'ancien
  design, `Orchestrator`/`run2.py` reste à l'arrêt).
- Toute modification de `tools/proven_anchor/` ou
  `tools/proven_anchor_curriculum.py`.

**Anti-but.** Ne jamais reproduire l'incident du 2026-10-20/21 : un
process qui tourne des heures sans qu'on puisse savoir où il en est, ni le
récupérer sans perte s'il faut l'arrêter. Visibilité et reprise sur
interruption sont des exigences de ce document, pas des agréments
optionnels laissés au bâtisseur.

## 3. Mécanisme

### 3.1 Position ancre

Une partie réelle, référence contre elle-même, sans pouvoir
(`tools.proven_anchor.anchor_games.play_reference_vs_reference`, déjà
disponible, callback `progress` déjà ajouté le 2026-10-21 — cf. commit en
attente dans cette session), jouée jusqu'au bout depuis une ouverture
quelconque du corpus sans pouvoir existant
(`experiments/no-power-corpus-20261010/campaign.json`). Une seule partie,
jouée une seule fois — pas de recherche parmi plusieurs ouvertures, pas de
critère de sélection : la première menée à son terme suffit.

**"5 coups avant la victoire" = 5 demi-coups (plies)**, au sens déjà en
usage dans tout ce chantier (`ply` incrémenté une fois par action de
joueur, cf. `decisions[i]['ply']` dans tout format de partie existant du
projet) — pas 5 tours complets des deux joueurs. Position ancre =
`positions[len(positions) - 1 - 5]` sur la liste renvoyée par
`play_reference_vs_reference`. **Hypothèse à confirmer avec l'utilisateur
si le chiffre s'avère trop proche ou trop loin une fois le pilote
lancé** — ce n'est pas une valeur à reproduire aveuglément si le premier
essai suggère qu'elle est mal choisie (partie trop courte pour avoir 5
demi-coups, ou position déjà triviale à ce point). **Choix du
bâtisseur, tranché ici plutôt que laissé ouvert** (relevé par la revue du
document avant plan) : cette distance est un paramètre de ligne de
commande (défaut 5), pas une valeur codée en dur — un recalibrage après ce
premier essai ne doit coûter qu'un changement de drapeau, pas une
modification de code.

Le joueur à qui revient le trait à la position ancre
(`position.player`) est celui qui, dans la partie réelle, a fini par
gagner 5 demi-coups plus tard — c'est lui que le candidat incarne dans la
boucle d'entraînement ci-dessous, la référence jouant l'autre camp.

### 3.2 Boucle d'entraînement (candidat contre référence, depuis l'ancre)

**Nouveau mode nécessaire côté moteur** (`native_engine/src/bin/selfplay_worker.rs`
ou binaire dédié) : jouer une partie complète depuis une position de
départ fournie (pas `initial_state()` codée en dur) où un camp (fixé une
fois pour toute la partie, déterminé par le joueur au trait dans la
position ancre fournie) est piloté par MCTS+réseau avec bruit de
Dirichlet/échantillonnage par température actifs sur **tous** ses coups
(pas seulement les 6 premiers demi-coups de toute la partie comme en
self-play continu — la queue de partie depuis l'ancre est courte, l'idée
« ouverture vs milieu de partie » ne s'applique plus de la même façon ici ;
**appel de jugement, à confirmer par le bâtisseur**), l'autre camp étant
piloté par le même moteur de référence alpha-beta que celui déjà exposé
via `santoni_search` — **dans le même process Rust**, pas via un aller-retour
Python : `santoni_search`/l'analyse alpha-beta vit dans le même crate
(`native_engine/src/lib.rs`) que `mcts.rs`/`selfplay_worker.rs`, donc
accessible en appel direct, sans round-trip process ni socket
supplémentaire.

**Chaque demi-coup est enregistré** dans le `GameRecord` produit (même
format que le self-play continu, `mcts_io::GameRecord`/`PositionRecord`,
aucun changement de `Trainer`/`consume_new_games` nécessaire) :
- Demi-coup du candidat : la vraie distribution de visites MCTS (comme en
  self-play continu).
- Demi-coup de la référence : une distribution dégénérée à un seul coup
  (celui réellement joué par l'alpha-beta, poids 1) — une étiquette exacte
  et gratuite, dans le même esprit que l'ancrage externe déjà au cœur de
  ce chantier, pas une invention propre à ce document.

Toute partie est écrite, enregistrée et consommée par `Trainer` exactement
comme en self-play continu aujourd'hui — l'`Orchestrator` existant garde
son rôle (serveur d'inférence, boucle d'entraînement, checkpoints
atomiques, pruning), seule la source des parties change (ce nouveau mode
plutôt que le self-play continu actuel), de façon à réutiliser l'essentiel
de ce qui existe déjà plutôt que d'écrire une deuxième infrastructure
d'entraînement.

### 3.3 Mesure du taux de victoire (critère d'arrêt)

**Décidé avec l'utilisateur, 2026-10-21 : seuil = borne basse de l'IC 95%
(Clopper-Pearson, même méthodologie que le reste du chantier) ≥ 95%, sur
une fenêtre glissante des 40 dernières parties d'évaluation.**

Les parties d'évaluation sont **distinctes** des parties d'entraînement :
même mode, mais bruit de Dirichlet et échantillonnage par température
désactivés (choix déterministe par argmax des visites, comme
`run_single_move` aujourd'hui) — mesurer la force réelle du réseau, pas sa
politique d'exploration. Elles ne sont pas ajoutées au corpus
d'entraînement (répertoire séparé, ou champ distinctif dans le nom de
fichier — détail laissé au bâtisseur). Cadence exacte entre parties
d'entraînement et parties d'évaluation laissée au bâtisseur (ex. une salve
d'évaluation toutes les N parties d'entraînement) — l'objectif est juste
que la mesure ne soit jamais confondue avec le signal d'entraînement
lui-même, pas une cadence particulière.

**Décidé avec l'utilisateur, 2026-10-21 : pas de plafond automatique si le
taux ne converge jamais.** Suivi manuel, cohérent avec "on verra bien le
bilan" (§0) — la campagne ne s'arrête jamais seule faute de convergence ;
elle tourne jusqu'à ce que le seuil de §3.3 soit atteint ou que
l'utilisateur l'interrompe lui-même après avoir lu la progression. C'est
exactement pour rendre cette option sûre que §3.4 (visibilité + reprise
sur interruption) est une exigence dure plutôt qu'un détail — sans elle,
ce choix reproduirait le risque de l'incident fondateur.

### 3.4 Visibilité et reprise sur interruption (exigence dure, pas un détail)

Directement motivé par l'incident du 2026-10-20/21 (12h14m, zéro visibilité,
zéro reprise possible, tout le calcul perdu à l'interruption) :

- **Visibilité** : une ligne de progression périodique (parties
  d'entraînement jouées, parties d'évaluation jouées, taux de victoire
  courant + IC, génération/perte d'entraînement) — même esprit que
  `run.log`/`Orchestrator.status()` déjà en place, pas un nouveau
  formalisme.
- **Reprise** : toute partie déjà jouée (entraînement ou évaluation),
  tout checkpoint, doit survivre à un arrêt à tout moment — écriture
  atomique partout (déjà la convention systématique de ce chantier,
  `write_game_atomically`, `Trainer.save_checkpoint`), et un redémarrage
  doit reprendre depuis l'état sur disque sans rejouer ce qui est déjà
  fait. Concrètement : le compte de parties d'évaluation déjà jouées et
  leur résultat doivent eux aussi être sur disque de façon atomique (pas
  seulement en mémoire du process), sans quoi un redémarrage perdrait la
  fenêtre glissante des 40 dernières parties — point à ne pas rater au
  moment du plan.

## 4. Réseau de départ

Neuf, pas repris d'un checkpoint existant — même raisonnement que le
noyau exact (§4 de l'ancien design) : isoler proprement la question « ce
mécanisme produit-il un apprentissage réel » sans la confondre avec les
biais d'un régime d'entraînement antérieur.

## 5. Risques connus

- **"5 demi-coups" est peut-être mal calibré.** Ni trop facile (position
  déjà quasi gagnée trivialement, rien à apprendre) ni trop dur (aucune
  amélioration mesurable, comme pour les deux pilotes 0% de ce chantier).
  Pas de garde-fou a priori dans ce document — le bilan du pilote lui-même
  tranchera, voir §2's "on verra bien le bilan".
- **Le bruit MCTS/Dirichlet rend "100% du temps" littéral inatteignable**
  — déjà la raison du choix du seuil statistique en §3.3, mais à garder en
  tête si le taux plafonne juste sous 95% sans jamais le passer : ça
  pourrait être un plafond réel (comme pour le noyau exact) plutôt qu'un
  problème de seuil.
- **Capacité du réseau non isolée comme variable** (même risque connu que
  l'ancien design, §6) — si l'échec se produit malgré un signal propre et
  sans ambiguïté, la piste naturelle suivante est un réseau plus grand,
  hors périmètre de ce document.
- **Le nouveau mode moteur est un vrai changement Rust**, pas une
  configuration — à construire, tester, et revoir comme tout changement
  de `native_engine/` dans ce chantier (review finale avant merge, comme
  pour le cœur MCTS, le batching de feuilles, etc.).

## 6. Ce qui reste au bâtisseur

- Surface CLI exacte du nouveau mode (`selfplay_worker.rs` étendu ou
  nouveau binaire ; noms de drapeaux ; comment la position de départ est
  fournie — probablement le même encodage `State` brut que
  `--single-move`/`--encode-planes`).
- Cadence exacte entre parties d'entraînement et parties d'évaluation.
- Mise en œuvre précise de la fenêtre glissante des 40 dernières parties
  d'évaluation sur disque (format, emplacement) pour survivre à un
  redémarrage.
- Disposition exacte des répertoires de la campagne (nom du run,
  séparation parties d'entraînement / parties d'évaluation / logs / pid),
  en suivant les conventions déjà en place (`selfplay-runs/<nom>/`,
  fichier `.pid`, fichier `.log`, écriture atomique partout).
- Valeurs par défaut de `simulations_per_move` (le chantier a déjà
  constaté que 32 et 128 donnent un résultat indiscernable en qualité de
  coup, à un coût de débit très différent — recommandation : repartir de
  32 pour ce pilote, sauf raison contraire) et du budget de temps par coup
  de la référence pendant la boucle d'entraînement/évaluation.
- Décision sur l'application du bruit de Dirichlet/température à tous les
  demi-coups du candidat dans ce mode plutôt que les 6 premiers comme en
  self-play continu (§3.2) — signalé comme un choix non trivial, pas
  encore tranché avec l'utilisateur.

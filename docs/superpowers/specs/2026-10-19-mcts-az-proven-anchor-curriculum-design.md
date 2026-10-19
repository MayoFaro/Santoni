# Auto-entraînement ancré sur du vrai (noyau exact par analyse rétrograde locale) — design

Date : 19 octobre 2026. Fait suite au bilan complet du chantier MCTS+NN
(`docs/bilan-mcts-az-no-power-20261019.md`) et à la discussion qui l'a
suivi — voir mémoire `santoni-mcts-az-no-power-chantier.md` pour
l'historique complet. Ce document couvre uniquement la nouvelle stratégie
d'entraînement ; il ne revient pas sur l'infrastructure MCTS/self-play déjà
en place (`santoni_az/`, `native_engine/`), qui reste le socle technique
sur lequel ce chantier se greffe.

## 0. Constat (diagnostic établi, sans ambiguïté)

Le bilan du 2026-10-19 a établi, sur ~1,2M parties de self-play et 43 227
générations : le taux de victoire contre le moteur de référence reste au
niveau du bruit (0-4%) quel que soit le volume d'entraînement, alors que la
cohérence interne du réseau avec ses propres parties grimpe sans
interruption (53%→98% de sign-match valeur). C'est la signature d'un
collapse de self-play : les deux côtés de chaque partie sont joués par le
même réseau faible, donc rien ne le corrige jamais depuis l'extérieur. Un
facteur ~10 de volume de données supplémentaire n'a produit aucune
tendance — ce n'est pas un problème d'échelle, c'est un problème de
méthode : il manque un ancrage externe validé.

## 1. Objectif

Donner au pipeline d'entraînement un ancrage externe exact (pas une
estimation MCTS bruitée, pas l'auto-cohérence du réseau) sur lequel le
réseau peut réellement apprendre des motifs stratégiques discriminants —
puis utiliser cet ancrage pour corriger la recherche MCTS plus en amont
dans la partie, au lieu de laisser le self-play dériver seul comme
aujourd'hui.

**Ce que ce n'est pas :** ni une table de finales exhaustive pour tout le
sous-jeu sans pouvoir (combinatoire non tractable, déjà écarté en
discussion), ni un remplacement du MCTS self-play existant — un
complément qui lui donne enfin un point de contact avec une vérité
externe.

## 2. Critère de succès (barre de décision)

Le signal décisif n'est pas un taux de victoire de J1 contre J2 (comme
dans les anciens pilotes) — c'est la **fidélité de l'évaluation du réseau
par rapport à la valeur prouvée du moteur de référence**, sur des positions
tenues à l'écart de l'entraînement, mesurée dans le même esprit que
`diagnose_net.py` déjà utilisé dans ce chantier (sign-match contre une
vérité terrain, pas contre le réseau lui-même) :

- **Succès pour une couche du noyau** : le sign-match du réseau sur un lot
  de positions prouvées tenues à l'écart (positions symétriques de celles
  entraînées + positions à la même distance de fin issues d'une partie
  ancre différente) dépasse nettement et de façon statistiquement
  robuste (IC 95%, même méthodologie Clopper-Pearson que le bilan) sa
  propre ligne de base pré-entraînement sur ce lot. Ce n'est qu'après ce
  constat qu'on recule d'un demi-coup supplémentaire.
- **Taille du lot tenu à l'écart** : une seule position ancêtre ne fournit
  qu'une douzaine à une quarantaine d'enfants (facteur de branchement du
  sous-jeu) — insuffisant pour un IC exploitable, au vu du précédent de ce
  chantier (`diagnose_net.py`, ~180 positions sur 60 parties pour un
  sign-match stable). Le lot tenu à l'écart doit donc regrouper les
  enfants prouvés de **plusieurs** positions ancêtres à la même distance de
  fin — ce qui implique de disposer d'au moins une deuxième partie ancre
  **dès la validation de la couche 1**, pas seulement à partir de la
  réplication de §3.7. §3.7 reste la phase d'élargissement *délibéré* du
  nombre de parties ancres une fois le mécanisme déjà confirmé ; une
  deuxième partie ancre minimale est une précondition de §2, pas une suite.
- **Régime d'entraînement entre couches** : le réseau n'est pas réinitialisé
  à chaque élargissement — il continue (fine-tuning cumulatif) sur le
  checkpoint qui a validé la couche précédente. Conséquence directe : la
  validation d'une nouvelle couche doit **revérifier aussi les lots tenus à
  l'écart de toutes les couches précédemment validées**, pas seulement la
  nouvelle — sinon un oubli catastrophique sur une couche déjà acquise
  passerait inaperçu.
- **Échec informatif** : si, sur les deux ou trois premières couches
  (positions les plus proches de la fin, où le signal devrait être le
  plus facile à apprendre), le sign-match sur positions tenues à l'écart
  ne dépasse jamais la ligne de base malgré un signal d'entraînement exact
  et sans bruit — cela pointerait vers la capacité du réseau comme goulot
  dominant, pas vers la méthode d'entraînement (puisque le signal ici est
  aussi propre qu'il peut l'être). Résultat décisif dans les deux sens, à
  rapporter tel quel plutôt qu'à relancer plus de la même chose.

Pas de critère de temps/calcul imposé (décision explicite de l'utilisateur,
2026-10-19) — le noyau s'élargit tant que la validation passe, sans
plafond fixé à l'avance.

## 3. Mécanisme

### 3.1 Partie ancre

Une partie réelle (recherchée dans le corpus existant, ou générée en
faisant jouer le moteur de référence contre lui-même si aucune ne convient)
contenant une position où le joueur au trait a un **gain prouvé** en 6
demi-coups. « Prouvé » veut dire concrètement : l'analyse du moteur de
référence à cette position a `proven=1` (champ déjà existant,
`native_engine/src/lib.rs`, exposé côté Python comme `stop_reason="proof"`,
`santorini/native.py:200`) et la distance au mat, `MATE - abs(score)`,
vaut exactement 6.

### 3.2 Construction du noyau exact, couche par couche

Important, établi en discussion après vérification du code : **aucun
générateur de prédécesseurs n'existe dans le moteur, et ce n'est pas un
problème** — on ne reconstruit pas l'arbre à l'envers depuis zéro, on
recule simplement dans l'historique réel de la partie ancre (les positions
ancêtres sont déjà connues, aucune énumération inverse nécessaire).

À chaque position ancêtre rencontrée en reculant (distance 6, puis 12,
puis 18 demi-coups de la fin...) :
1. Générer tous les coups légaux depuis cette position (déjà disponible,
   dédupliqué par transposition via `canonical_key`,
   `native_engine/src/lib.rs:519-586`).
2. Pour chacun, interroger le moteur de référence pour obtenir son
   verdict **prouvé** (gagnant/perdant/non résolu) — pas juste le meilleur
   coup de la position entière, mais le statut de chaque branche
   individuellement. Chaque appel fait son propre travail de profondeur en
   interne (itérative deepening existante) ; on ne réimplémente aucune
   recherche.
3. Le résultat est une carte complète, à cette distance de la fin, des
   coups gagnants et perdants — pas une ligne mémorisée.

Le budget de temps/profondeur nécessaire à chaque appel de preuve n'est
pas fixé a priori dans ce document — Santorini est nettement moins
complexe que les échecs, où des moteurs modestes prouvent déjà des
positions profondes dans des budgets courts ; la limite réelle, si elle
existe, sera constatée pendant la construction elle-même, pas anticipée
par une mesure séparée.

**Branches non résolues dans le budget imparti.** À mesure qu'on recule,
il est possible qu'une partie des ~33 coups d'une position ancêtre ne soit
pas prouvée dans le budget du moment (preuve trop profonde). Ça ne bloque
pas la couche : on entraîne sur le sous-ensemble résolu, et on suit la
fraction résolue par couche comme indicateur diagnostique. Une chute nette
de cette fraction à une certaine distance de la fin est elle-même la
mesure empirique du plafond du noyau exact (cf. §5, mesure volontairement
non anticipée) — pas une raison d'arrêter la couche en cours tant qu'elle
reste majoritairement résolue.

### 3.3 Distillation supervisée dans le réseau

Les étiquettes exactes de 3.2 alimentent l'entraînement du réseau
(`santoni_az/`) comme cibles politique/valeur pour ces positions — en
réutilisant l'infrastructure `Trainer` existante (batches mélangés à
celles du replay buffer, ou priorisés séparément ; détail laissé au
bâtisseur, cf. §7) plutôt qu'une nouvelle boucle d'entraînement.

**Le self-play continu (`Orchestrator`/`run2.py`) reste à l'arrêt pendant
toute cette phase de construction/validation du noyau.** C'est une
conséquence directe du diagnostic de §0 : laisser tourner le self-play en
parallèle réinjecterait dans le même replay buffer des parties produites
par les deux côtés du même réseau faible — exactement le mécanisme de
collapse que ce chantier cherche à corriger, ce qui invaliderait la mesure
de §2 dès la première couche. Le self-play ne reprend, ancré (§3.5),
qu'une fois une première couche validée.

### 3.4 Validation anti-mémorisation (garde-fou, obligatoire avant d'élargir)

Avant de reculer d'une couche, vérifier que le réseau a appris un motif et
pas une position précise :
- **Symétries du plateau** : les transformations géométriques valides de
  la grille 5×5 appliquées à une position entraînée doivent donner la même
  évaluation (à transformation près) sans entraînement supplémentaire. Les
  symétries exactement valides pour ce sous-jeu (rotations/réflexions,
  éventuellement restreintes par une asymétrie de règle non encore
  vérifiée) sont à confirmer par le bâtisseur en lisant les règles du jeu
  avant d'implémenter ce test — pas supposées ici.
- **Positions tenues à l'écart** : voir critère de succès, §2.

### 3.5 Raccordement avec le MCTS self-play existant

Une fois une couche validée, elle sert de point d'ancrage pour la
recherche plus en amont dans la partie : le MCTS self-play (self-play
continu ou arbitrage, `native_engine/src/mcts.rs`) peut chercher à
atteindre ces positions validées plutôt que d'évaluer dans le vide à
chaque feuille. Le mécanisme précis de consultation (table de correspondance
consultée pendant la sélection MCTS, par opposition à un simple mélange
des exemples dans l'entraînement du réseau) est laissé au bâtisseur — les
deux apportent une partie du bénéfice, la version la plus simple (3.3 seul,
sans modification du MCTS lui-même) suffit à valider ou invalider
l'hypothèse de fond et devrait être construite en premier.

### 3.6 Élargissement du noyau

Reculer d'une couche supplémentaire (encore 6 demi-coups) dans la même
partie ancre tant que 3.4 valide. Pas de plafond de profondeur fixé par ce
document (cf. §2) — décision prise couche par couche sur la base de la
validation, pas du temps écoulé.

### 3.7 Pluralité des parties ancres

Une deuxième partie ancre (motif tactique si possible différent) est déjà
nécessaire dès la couche 1, comme précondition statistique de §2 — ce
n'est pas attendu ici. Ce qui reste propre à cette étape, une fois le
mécanisme validé sur ces premières parties ancres : l'élargir
délibérément à davantage de parties ancres indépendantes pour que le noyau
ne reste pas une poignée de lignes idiosyncratiques mais une base
diversifiée de motifs réellement compris. Nombre et choix des parties
ancres supplémentaires laissés au bâtisseur, à itérer selon les résultats
des premières.

## 4. Réseau de départ

Neuf, pas repris de `gen-43227.pt` — décision explicite : ce checkpoint
porte l'habitude du collapse de self-play déjà diagnostiqué ; repartir de
zéro isole proprement la question « ce mécanisme produit-il un
apprentissage réel » sans la confondre avec les biais de l'ancien régime.

## 5. Hors périmètre (explicitement, pour ce document)

- Consultation du noyau exact comme table de recherche pendant le MCTS
  (3.5, variante ambitieuse) — phase ultérieure, pas nécessaire pour
  valider l'hypothèse de fond.
- Construction d'un générateur de prédécesseurs/analyse rétrograde au
  sens classique (table de finales complète) — explicitement écarté,
  combinatoire non tractable sur l'ensemble du sous-jeu, et non nécessaire
  grâce à 3.2.
- Toute modification du moteur de référence lui-même (l'encodage
  MATE/`proven` existe déjà et suffit).
- Relance de l'ancien régime de self-play pur (`selfplay-runs/run2.py`
  tel quel) — ce chantier le remplace/le complète, il ne le poursuit pas
  à l'identique.
- Mesure préalable séparée de la profondeur maximale de preuve du moteur
  de référence — cf. §3.2, constatée pendant la construction, pas en
  amont.

## 6. Risques connus

- **Capacité du réseau non isolée comme variable.** Le critère d'échec de
  §2 est conçu pour détecter ce cas précisément (signal propre, échec
  quand même → capacité), mais ce document ne teste pas séparément une
  architecture plus grande. Si l'échec se produit, l'étape suivante
  naturelle (plus grand réseau) sort du périmètre de ce document.
- **Symétries de plateau à confirmer, pas supposées.** Si Santorini sans
  pouvoir a une asymétrie de règle non évidente (ex. ordre de placement
  initial, identité des joueurs), le test anti-mémorisation de §3.4 doit
  être ajusté en conséquence — à vérifier par le bâtisseur avant
  d'implémenter, pas une hypothèse à prendre pour acquise.
- **Un seul fil conducteur (partie ancre) au départ.** §3.7 répond à la
  généralisation à terme, mais la première validation reposera sur une
  seule ligne — un résultat positif précoce ne doit pas être sur-interprété
  avant la réplication sur plusieurs parties ancres.

## 7. Ce qui reste au bâtisseur

- Mécanisme exact d'intégration des exemples du noyau dans `Trainer`
  (mélange avec le replay buffer existant vs. phase supervisée séparée).
- Confirmation du groupe de symétries valide pour ce sous-jeu avant
  d'implémenter le test anti-mémorisation.
- Choix du budget de temps/profondeur par appel de preuve au moteur de
  référence (§3.2) — à ajuster empiriquement pendant la construction.
- Nombre et sélection des parties ancres supplémentaires (§3.7).
- Decision sur le moment d'activer 3.5 (ancrage MCTS) une fois 3.3 validé.

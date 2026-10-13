# Regroupement de feuilles MCTS (virtual loss) — design

Date : 13 octobre 2026. Ferme la seconde moitié du blocker 4 (« leaf
batching », cf. mémoire `santoni-mcts-az-no-power-chantier`), restée
ouverte depuis le chantier initial MCTS+NN. Contexte direct : un
entraînement de plusieurs jours est en cours sous le régime corrigé du
tampon de rejeu (spec `2026-10-11-trainer-replay-buffer-design.md`) ; le
gain cumulé sur une fenêtre aussi longue justifie l'effort d'ingénierie que
ce correctif avait demandé en réserve.

## 0. Constat (diagnostic déjà établi, sans ambiguïté)

- La connexion persistante worker ↔ serveur d'inférence est déjà en place
  (spec `2026-10-11-mcts-az-persistent-inference-connection-design.md`) :
  plus de coût `connect()`/`accept()` par feuille.
- Le système n'est pas limité par le calcul : CPU global ~20% utilisé même
  à 128 workers self-play, GPU à 0% d'utilisation.
- Le goulot mesuré est le process Python unique du serveur d'inférence,
  mono-thread par construction (`torch.set_num_threads(1)`, nécessaire
  pour éviter un SIGABRT connu) : il plafonne à 80-84% d'un seul cœur à
  128 workers, avec un gain marginal de seulement +13% pour le dernier
  doublement de workers (64→128), contre +180% pour le doublement
  précédent (16→64) — signe clair de plafonnement.
- Hypothèse retenue, plausible et non réfutée : ce plafond vient de la
  surcharge Python/GIL de gérer ~128 connexions persistantes concurrentes
  (chacune émettant une requête à une seule feuille à la fois), pas du
  calcul du réseau lui-même (réseau volontairement minuscule : 4 blocs
  résiduels, 32 canaux).
- `Mcts::run`/`simulate_one` (`native_engine/src/mcts.rs`) n'appellent
  aujourd'hui `LeafEvaluator::evaluate` qu'avec un seul état à la fois
  (`&[state]`), bien que le trait accepte déjà un slice de plusieurs états
  — jamais exploité.

## 1. Objectif

Regrouper plusieurs feuilles d'un même arbre MCTS en un seul appel réseau
(`evaluate(&[...])` avec plusieurs états), pour réduire le **nombre**
d'allers-retours par unité de travail de self-play — l'axe qui adresse
directement la surcharge Python/I/O diagnostiquée, par opposition à la
latence par appel (déjà traitée par la connexion persistante).

## 2. Portée

- `native_engine/src/mcts.rs` (`Mcts::run`, restructuration de
  `simulate_one` en sélection/évaluation/backup séparées).
- `native_engine/src/bin/selfplay_worker.rs` (constante `LEAF_BATCH_SIZE`,
  mise à jour des deux points d'appel : `play_one_game` et
  `run_single_move`).
- **S'applique aux deux chemins** : self-play continu ET arbitrage
  déterministe (`run_single_move`, utilisé par
  `tools/mcts_az_vs_reference_campaign.py`). Le regroupement de feuilles
  est une technique de performance pure (ne change pas l'algorithme de
  sélection UCB, juste l'ordre/le groupage des évaluations) — contrairement
  au bruit de Dirichlet et à l'échantillonnage par température, qui restent
  strictement self-play uniquement et ne sont pas concernés par ce
  document.
- Aucun changement au protocole binaire du serveur d'inférence
  (`InferenceServer`, `inference_protocol.py`) : il accepte déjà plusieurs
  états par requête.

## 3. Architecture

### 3.1 Restructuration en trois phases

`simulate_one` aujourd'hui est récursif et bloque sur
`evaluator.evaluate(&[state])` à chaque feuille — une simulation complète
(descente + évaluation + backup) avant la suivante. Le nouvel algorithme
sépare ça en vagues de `leaf_batch_size` sélections :

1. **Sélection** (jusqu'à `leaf_batch_size` fois par vague) : descend
   depuis la racine via UCB, sans bloquer sur le réseau. Si la descente
   atteint un état terminal, elle se résout et se backup immédiatement
   (valeur déjà connue via `terminal_value`, aucun appel réseau
   nécessaire). Si elle atteint une feuille jamais vue
   (`!self.nodes.contains_key(&state)`), elle s'arrête là : l'état est
   retenu pour l'évaluation par lot, et le chemin parcouru (séquence
   d'arêtes traversées depuis la racine) est mémorisé pour le backup
   différé.
2. **Évaluation par lot** : un seul appel
   `evaluator.evaluate(&feuilles_non_terminales_de_la_vague)`.
3. **Expansion + backup** : pour chaque feuille évaluée, créer ses arêtes
   (comme le fait `simulate_one` aujourd'hui dans sa branche d'expansion),
   puis remonter le chemin mémorisé en appliquant la vraie valeur
   (convention négamax, valeur inversée à chaque niveau) — ce que faisait
   la pile d'appels récursifs en se dépilant, fait ici explicitement le
   long d'un chemin enregistré.

### 3.2 Perte virtuelle (diversité des sélections dans une même vague)

Sans ajustement, les sélections d'une vague verraient toutes les mêmes
statistiques (rien n'est mis à jour entre elles) et choisiraient donc
**le même chemin** à chaque fois — le regroupement ne servirait à rien.
Chaque arête traversée pendant la sélection reçoit une perte virtuelle
temporaire, annulée une fois la vraie valeur reçue et remplacée par le
vrai backup.

**Variante retenue : perte virtuelle sur les visites seules**, pas la
variante complète de la littérature (qui pénaliserait aussi le terme Q).
`Edge` gagne un champ `virtual_loss: u32`, incrémenté temporairement à
chaque traversée pendant la sélection d'une vague, pris en compte dans le
dénominateur du terme d'exploration UCB (`sqrt_total / (1.0 +
e.visits as f32 + e.virtual_loss as f32)`) mais pas dans le terme Q
(`value_sum` reste intact). Annulation triviale au backup : décrémenter le
même `virtual_loss` qui avait été ajouté, puis appliquer le vrai
`visits += 1` / `value_sum += valeur`. Choisi pour minimiser le risque
d'implémentation (un seul compteur à suivre et annuler, pas de double
comptabilité `value_sum`/`visits` à désynchroniser) dans une implémentation
Rust écrite à la main, sans bibliothèque existante.

### 3.3 Signature et taille de vague

`Mcts::run` change de signature :
`pub fn run(&mut self, simulations: u32, leaf_batch_size: usize, evaluator: &mut dyn LeafEvaluator)`.
Boucle en vagues de `leaf_batch_size` sélections jusqu'à atteindre
`simulations` au total (dernière vague plus petite si `simulations` n'est
pas un multiple exact).

**Propriété de conception centrale : `leaf_batch_size = 1` doit se
comporter exactement comme le code actuel.** Une seule sélection par vague
élimine tout besoin de diversité — aucune perte virtuelle n'a d'effet
observable puisqu'elle est posée puis immédiatement annulée avant la
sélection suivante. Point d'ancrage de non-régression (§6).

**Valeur de production : `LEAF_BATCH_SIZE = 8`**, nouvelle constante dans
`selfplay_worker.rs` au même endroit que `DIRICHLET_ALPHA`/
`TEMPERATURE_PLIES`. Avec `simulations_per_move = 32`, ça donne 4 vagues
par coup au lieu de 32 appels séparés — division par 8 du nombre
d'allers-retours réseau par coup.

**Points d'appel mis à jour** :
- `play_one_game` : `mcts.run(simulations_per_move, evaluator)` →
  `mcts.run(simulations_per_move, LEAF_BATCH_SIZE, evaluator)`.
- `run_single_move` : boucle `mcts.run(BATCH, evaluator)` (où `BATCH = 16`
  est le nombre de simulations entre deux vérifications du budget de
  temps — concept différent, sans rapport avec `leaf_batch_size`) →
  `mcts.run(BATCH, LEAF_BATCH_SIZE, evaluator)`, même constante réutilisée
  pour la cohérence.

## 4. Cas limites

- **Coup forcé (une seule arête légale à la racine ou plus bas)** : toutes
  les sélections d'une vague convergent trivialement vers l'unique arête —
  la perte virtuelle s'y empile (jusqu'à `leaf_batch_size` incréments
  temporaires avant la vraie évaluation). Pas de risque de panique (pure
  arithmétique sur des compteurs), mais déjà signalé comme un cas réel
  (worker acculé) par une revue précédente de ce chantier — testé
  explicitement (§6), pas supposé correct par construction.
- **États terminaux rencontrés en cours de sélection** : résolus et
  backupés immédiatement dans la phase de sélection, sans jamais entrer
  dans le lot envoyé à `evaluate()`.
- **Doublons dans une même vague** (deux sélections transposant vers le
  même état feuille) : pas de déduplication — le lot envoyé au réseau peut
  contenir des doublons, chacun évalué séparément. Choix délibéré (YAGNI) :
  la perte d'efficacité est mineure comparée au gain du regroupement.

## 5. Hors périmètre (explicitement)

- Toute modification au format binaire du protocole d'inférence ou à
  `InferenceServer` : déjà capable d'accepter plusieurs états par requête,
  rien à changer côté serveur.
- La variante complète de perte virtuelle (pénalité sur Q en plus des
  visites) : retenue comme option de repli si la variante « visites
  seules » s'avère insuffisante empiriquement, pas implémentée ici.
- Déduplication des feuilles transposées dans une même vague.
- Tout changement à `batch_size` (taille de lot d'entraînement, 64 par
  défaut) ou au tampon de rejeu de `Trainer` — axe indépendant déjà traité.
- Bruit de Dirichlet / température d'ouverture : inchangés, restent
  self-play uniquement.

## 6. Validation

Tests dans `native_engine/src/mcts.rs` (module `mcts_tree_tests`) :

1. **Ancrage de non-régression** : tous les tests existants (`mix_root_noise_*`,
   `visit_counts_returns_the_after_state_matching_legal_children`,
   `root_with_one_legal_move_gets_all_simulations_on_that_move`, etc.),
   ré-exécutés avec `leaf_batch_size = 1`, doivent passer sans
   modification.
2. Avec `leaf_batch_size > 1` sur une racine à plusieurs coups légaux : le
   total des visites doit toujours égaler `simulations` exactement, y
   compris quand `simulations` n'est pas un multiple de `leaf_batch_size`.
3. Coup forcé (une seule arête légale) avec `leaf_batch_size > 1` : ne
   panique pas, toutes les simulations atterrissent sur l'unique arête.
4. Un évaluateur de test qui enregistre la taille de chaque appel à
   `evaluate()` : avec `leaf_batch_size = 8` et une racine à plusieurs
   coups, au moins un appel doit recevoir plusieurs états d'un coup —
   preuve directe que le regroupement a bien lieu.
5. Re-vérification de `evaluate_decodes_a_multi_item_response_at_the_right_offsets`
   (`selfplay_worker.rs`, déjà existant) : c'est la première fois que ce
   chemin multi-éléments est réellement exercé par le vrai pipeline plutôt
   que par un test isolé — doit continuer à passer.

# Table de transposition : remplacement et partage des variantes

Date : 9 octobre 2026. Branche : `improve/tt-memory`, worktree isolé, divergée
de `main` à `7ebcc8a` (fusion des 55 pouvoirs et de l'Arène).

## Défaut corrigé

Le rapport du 5 octobre ([plan d'amélioration](ameliorations-moteur-20261005.md),
section 6) notait que la table de transposition refusait toute insertion une
fois 50 000 entrées atteintes, y compris la mise à jour d'une clé déjà
présente avec un résultat plus profond, et que chaque entrée clonait sa ligne
complète (`Vec<Turn>`) à chaque insertion et à chaque lecture. Ce défaut
restait présent sur `main` au 9 octobre (`native_engine/src/lib.rs:654`,
avant cette branche) : aucune des pistes stratégiques explorées depuis
(constructions/accès, menaces forcées, Athéna, ML) n'avait isolé et mesuré
cet axe pour lui-même, alors qu'il était jugé le moins coûteux et le plus sûr
du classement initial.

## Changement

Deux modifications localisées dans `native_engine/src/lib.rs`, sans toucher
aux règles, à l'évaluation ni à l'élagage :

1. **Politique de remplacement** (`Search::store`) : une clé déjà présente
   est toujours rafraîchie, même table pleine, puisque cela ne fait pas
   grandir la table. Une clé nouvelle, une fois la table pleine, ne remplace
   une entrée arbitraire existante que si celle-ci n'est pas plus profonde que
   la nouvelle — la table reste bornée à 50 000 entrées, mais cesse de
   refuser tout travail utile après saturation.
2. **Partage des variantes** (`Entry.line : Rc<Vec<Turn>>` au lieu de
   `Vec<Turn>`) : la lecture d'une entrée en cache (le chemin fréquent, à
   chaque nœud visité) devient un incrément de compteur de référence au lieu
   d'un clone profond de la ligne stockée. Les rares chemins qui ont besoin
   d'un `Vec<Turn>` possédé (retour de coupure, reconstruction de la ligne
   principale) paient toujours un clone, mais seulement là.

Aucun poids d'évaluation, aucune règle et aucun critère de tri ne change.

## Validation locale

- 8 tests Rust (4 nouveaux : remplissage sous capacité, mise à jour d'une
  clé existante table pleine, éviction d'un témoin peu profond au profit
  d'une entrée plus profonde, conservation d'une entrée profonde face à une
  nouvelle entrée superficielle, partage de l'allocation de `line`).
- 338 tests Python/Qt inchangés, aucune régression.
- 18 positions fixes (`experiments/tt-memory-5s-20261009/positions.json`,
  reprises du corpus `decision-quality`) comparées à 5 s entre le binaire
  `main@7ebcc8a` et le candidat : même profondeur et même successeur sur les
  18 cas. Ce banc rapide ne couvre pas la saturation de la table (positions
  trop courtes pour atteindre 50 000 entrées en 5 s) ; il garantit seulement
  l'absence de régression grossière avant la campagne de parties.

## Protocole de la campagne à 200 parties

Comparaison à budget égal, binaires isolés par sous-processus, exactement le
même banc (`tools/version_bench.py`, repris sans modification du worktree
`decision-quality`) que les comparaisons précédentes (Athéna, blocage niveau
2, ranker ML).

- Référence : `main@7ebcc8a`, reconstruite localement
  (`experiments/references/main-7ebcc8a/`).
- Candidat : `improve/tt-memory@cefb063`
  (`experiments/tt-memory-5s-20261009/variant/`).
- Budget : 5 secondes par tour, plafond de 120 tours par partie.
- Dix confrontations reprises du
  [bilan 35 s / 5 s](bilan-moteur-35s-vs-5s-20261005.md) : sans pouvoir,
  Hermès/Héphaïstos, Apollon/Minotaure, Artémis/Athéna, Atlas/Pan,
  Déméter/Prométhée, Apollon/Athéna, Artémis/Pan, Atlas/Déméter,
  Minotaure/Prométhée.
- Dix placements nouveaux par confrontation (graine `20261009`, non réutilisée
  dans les campagnes antérieures pour cette combinaison outil/graine), soit
  100 configurations. Chacune est rejouée camp inversé : **200 parties**.
- Dix cœurs physiques, priorité réduite (`os.nice(10)`), un processus par
  camp et par partie.

Le bilan sera calculé par paire de placement (pas par partie isolée), avec un
intervalle bootstrap à 95 % par rééchantillonnage des 100 groupes, à l'image
des campagnes précédentes. Les contradictions entre un résultat annoncé
démontré et le vainqueur final seront vérifiées sur les 200 historiques
rejoués.

## Résultat

Les 200 parties sont terminées, aucune erreur, aucune partie non conclue,
**aucune contradiction** entre un résultat annoncé démontré et le vainqueur
final. Les coûts sont quasi identiques entre les deux binaires : 2 686
recherches candidates contre 2 683 référence, profondeur médiane 4 des deux
côtés, 4,173 s contre 4,182 s en moyenne par décision. Ce changement ne
ralentit donc pas la recherche, contrairement à toutes les pistes
heuristiques ou d'apprentissage testées jusqu'ici.

| Indicateur | Résultat |
|---|---:|
| Victoires du candidat | 103 / 200 (51,5 %) |
| Intervalle bootstrap à 95 %, regroupé par placement (10 000 tirages) | 47,5 % – 55,5 % |
| Configurations gagnées deux fois par le candidat | voir répartition ci-dessous |
| Résultats marqués démontrés, candidat / référence | 502 / 491 |
| Contradictions preuve/vainqueur | 0 |

| Confrontation | Victoires du candidat / 20 |
|---|---:|
| Sans pouvoir | 8 |
| Hermès / Héphaïstos | 8 |
| Apollon / Minotaure | 11 |
| Artémis / Athéna | 11 |
| Atlas / Pan | 10 |
| Déméter / Prométhée | 10 |
| Apollon / Athéna | 13 |
| Artémis / Pan | 9 |
| Atlas / Déméter | 11 |
| Minotaure / Prométhée | 12 |

L'intervalle à 95 % contient 50 % : **le taux de victoire seul ne suffit pas
à démontrer un gain de force**, au même titre que les profils 1/3/4 de la
campagne à 1 000 matchs ou le blocage niveau 2. Ce lot ne doit pas être
présenté comme une preuve de progrès.

Ce résultat se distingue cependant des pistes précédentes sur deux points
factuels, pas seulement sur une intuition :

- **Coût nul** : aucune des variantes heuristiques ou d'apprentissage
  testées depuis le 5 octobre n'avait obtenu un temps par décision aussi
  proche de la référence (les plus proches, les extensions tactiques
  ciblées, coûtaient déjà plusieurs dixièmes de seconde de plus ; le
  ranker ML coûtait 8 à 11 fois plus cher). Ici, 4,173 s contre 4,182 s,
  sans changer l'évaluation ni l'élagage.
- **Défaut documenté corrigé** : la table refusait silencieusement de
  mettre à jour une clé déjà connue une fois pleine, y compris avec un
  résultat plus profond. Ce n'était pas une hypothèse stratégique risquée,
  mais un bogue de performance sans contrepartie identifiée ; les 502
  résultats démontrés du candidat contre 491 pour la référence, à coût égal,
  sont cohérents avec sa correction, sans que ce seul lot l'établisse.

## Décision proposée

Ce changement ne grossit pas la liste des pistes stratégiques rejetées :
c'est une correction d'infrastructure, testée unitairement, qui ne dégrade
rien d'observé et ne coûte rien de mesuré. Deux suites raisonnables,
au choix de l'équipe :

1. **Fusionner directement dans `main`** sur la base des tests unitaires et
   de l'absence de régression sur 200 parties et 18 positions fixes — à la
   manière d'une correction de bogue de performance plutôt que d'un nouveau
   réglage stratégique à prouver.
2. **Lancer un second lot verrouillé** (nouvelle graine, nouveaux
   placements) avant fusion, si l'on souhaite appliquer le même seuil
   d'acceptation que pour les candidats stratégiques (ML, Athéna) :
   amélioration appariée significative avec borne basse d'IC95 positive.
   Ce lot seul ne l'atteint pas.

Dans les deux cas, l'étape suivante logique du plan du 5 octobre reste la
réduction des copies restantes (reconstruction de la ligne principale sans
`Vec<Turn>` dupliqué à chaque nœud terminal) et le profilage détaillé
(génération / dédoublonnage / tri / évaluation / cache), qui n'ont toujours
pas été mesurés séparément.

## Reproduction

`tools/version_bench.py` est repris sans modification du worktree
`decision-quality` (outil générique à deux binaires, indépendant de toute
piste stratégique). Depuis ce worktree, avec un nouveau dossier de sortie :

```sh
cargo test --manifest-path native_engine/Cargo.toml
cargo build --release --manifest-path native_engine/Cargo.toml
QT_QPA_PLATFORM=offscreen python3 -m pytest -q
python3 tools/version_bench.py positions \
  --baseline experiments/references/main-7ebcc8a/libsantoni_engine.so \
  --candidate experiments/tt-memory-5s-20261009/variant/libsantoni_engine.so \
  --positions experiments/tt-memory-5s-20261009/positions.json \
  --output experiments/tt-memory-5s-<nouvelle-date>/positions-check \
  --seconds 5 --workers 6
python3 tools/version_bench.py matches \
  --baseline experiments/references/main-7ebcc8a/libsantoni_engine.so \
  --candidate experiments/tt-memory-5s-20261009/variant/libsantoni_engine.so \
  --output experiments/tt-memory-5s-<nouvelle-date>/matches \
  --seconds 5 --workers 10 --seed <nouvelle-graine> \
  --matchups 0:0,7:6,1:8,2:3,4:9,5:10,1:3,2:9,4:5,8:10  # répéter chaque paire 10 fois pour 100 configurations
```

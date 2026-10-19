# Bilan — chantier MCTS+NN (sous-jeu sans pouvoir), 2026-10-19

## Contexte et portée

Ce document consolide l'ensemble des tests, campagnes et diagnostics menés sur
le chantier AlphaZero-style (MCTS Rust + réseau PyTorch + self-play +
entraînement, `santoni_az/` + `native_engine/`) entre le 2026-10-10 et le
2026-10-19, afin de répondre à trois questions : qu'est-ce qui a été mesuré,
est-il pertinent de continuer à entraîner dans les conditions actuelles, et
quelles sont les alternatives disponibles. Il synthétise des éléments déjà
documentés en détail dans la mémoire `santoni-mcts-az-no-power-chantier.md` —
ce document en est la lecture froide et argumentée, pas un remplacement.

**Verdict en une phrase : l'infrastructure fonctionne et le réseau apprend
quelque chose de réel sur ses propres parties, mais rien ne démontre qu'il
apprend à mieux jouer contre un adversaire externe — après 1,2M+ parties et
43 227 générations, le taux de victoire reste au niveau du bruit, y compris
avec un budget de réflexion 6× supérieur. Poursuivre l'entraînement à
l'identique n'est pas recommandé.**

## 1. Ce qui a été construit et validé (infrastructure)

- Moteur MCTS Rust réutilisant les règles existantes (`generate`/`apply`),
  sans réimplémentation ni nouvelle dépendance externe.
- Worker de self-play avec mode continu et mode `--single-move`
  (compatible avec le pont ctypes existant), écriture atomique des parties.
- Package `santoni_az/` : petit réseau politique+valeur (4 blocs résiduels,
  32 canaux), serveur d'inférence par socket Unix avec batching, `Trainer`
  avec buffer de rejeu glissant (74 000 positions, ≈5000 parties) et
  checkpoints résumables.
- Reprise vérifiée à tous les niveaux (écritures atomiques, redémarrage sans
  perte, poids restaurés).
- Batching de feuilles / virtual loss dans `Mcts::run` (vagues de 8 feuilles,
  `LEAF_BATCH_SIZE=8`), gain mesuré ×2,1 sur le débit de self-play.
- Un bug de correction a été trouvé et corrigé en cours de route (le buffer
  d'entraînement supprimait chaque exemple après un seul pas de gradient —
  régime délétère pour une cible de valeur bruitée) : voir section 3.

Cette partie est solide, testée, revue. Le problème documenté ci-dessous
n'est **pas** un problème de génie logiciel mais de qualité du signal
d'apprentissage.

## 2. Campagnes d'entraînement menées

| Run | Période | Générations / parties | Régime | Observation |
|---|---|---|---|---|
| `no-power-20261011` | 2026-10-11, ~9h | 5765 gén. / 128 757 parties | `sims=32`, buffer bugué (suppression après 1 pas) | Perte plate à légèrement croissante sur 6h (1.88→2.37→…→2.21) |
| `no-power-20261012-replay` (avant 32→128) | 2026-10-12 → 2026-10-14 | jusqu'à gén. 31074 | `sims=32`, buffer corrigé | Perte en baisse réelle au départ (gén 4091 : sign-match valeur 73,3% vs 63,9% au hasard), puis plate (~1,72-1,75) sur ~14h observées |
| même run, après 32→128 (2026-10-14 14:21) | 2026-10-14 → 2026-10-19 00:44 | gén. 32078 → 43227 (+11 149 gén., +68 956 parties) | `sims=128` | Débit effondré à ~2 970 parties/h (×10 plus bas que prévu par extrapolation de la précédente hausse 8→32, cause non expliquée) ; perte toujours plate (~2,69 de moyenne, 2,31-3,26) |

**Débit de self-play en fonction de la configuration (axe indépendant,
entièrement résolu) :**

| Config | Débit mesuré |
|---|---|
| 16 workers, `sims=32` | ~4 800 parties/h |
| 64 workers, `sims=32` | ~13 400 parties/h |
| 128 workers, `sims=32` | ~15 200 parties/h (rendement décroissant confirmé) |
| 128 workers, `sims=32`, après leaf-batching | ~32 000 parties/h (×2,1) |
| 128 workers, `sims=128`, après leaf-batching | ~2 970 parties/h (×10 plus bas que prévu) |

Le passage GPU n'a jamais été fait : `santoni_az/` tourne entièrement sur CPU
alors que la machine dispose d'un GPU dédié (RTX 5060 Laptop,
`torch.cuda.is_available()` confirmé `True`). Signalé à plusieurs reprises,
jamais exploité.

## 3. Diagnostics de qualité — le cœur du problème

Trois familles de tests indépendantes ont été utilisées pour juger de la
force réelle du réseau, au-delà de la perte d'entraînement (qui ne mesure que
la cohérence interne du réseau avec ses propres données, pas sa force de
jeu) :

### 3.1 Auto-cohérence interne (`diagnose_net.py`)

Compare les sorties du réseau à un réseau neuf non entraîné, sur des
positions réelles échantillonnées.

| Checkpoint | Sign-match tête de valeur | Baseline aléatoire |
|---|---|---|
| gen-5765 (buffer bugué) | 53,3% | 54,4% — indiscernable |
| gen-4091 (buffer corrigé) | 73,3% | 63,9% |
| gen-31074 (buffer corrigé, 14h plus tard) | **98,3%** | 64,6% |

Le réseau devient de plus en plus confiant et cohérent avec *ses propres*
parties de self-play — ce signal progresse sans interruption sur tout le
chantier.

### 3.2 Pilotes de victoire à temps égal (5s/5s)

| Checkpoint | Résultat |
|---|---|
| gen-5765 | 0 / 20 (0%) |
| gen-16187 (~10 400 générations plus tard, buffer corrigé) | 0 / 20 (0%) |

Identique malgré un entraînement très substantiellement plus long.

### 3.3 Pilote de victoire à temps asymétrique (candidat 30s vs. référence 5s, ×6)

Conçu spécifiquement pour trancher entre « le réseau est faible » et
« la recherche MCTS+réseau est structurellement désavantagée en temps égal
face à l'alpha-bêta » — avec un budget de réflexion 6× supérieur donné au
candidat, un écart de force même modeste devrait se traduire par une large
majorité de victoires.

| Checkpoint | n | Victoires candidat | Taux |
|---|---|---|---|
| gen-31155 (avant 32→128) | 8 | 0 | 0% |
| gen-43227 (après 32→128), round 1 | 8 | 1 | 12,5% |
| gen-43227, **round 2** (échantillon élargi, ouvertures différentes, 2026-10-19) | 24 | 1 | **4,2%** |
| **Cumul post-128 (round 1 + round 2)** | **32** | **2** | **6,25%**, IC 95% [0,8% ; 20,8%] |

Le résultat à petit échantillon (12,5%) qui avait motivé un second tour plus
large ne s'est **pas reproduit** — l'échantillon élargi le ramène à 4,2%, et
le cumul des deux tours (6,25%, intervalle de confiance large mais dont la
borne haute reste sous 21%) ne permet toujours pas de distinguer un effet
réel du bruit à cette taille d'échantillon, et reste de toute façon à des
niveaux incompatibles avec une parité de force réelle sous un handicap de
temps aussi généreux.

### 3.4 Sonde de qualité de coup (arbitre = moteur de référence lui-même)

Mesure `move_loss = score_before + score_after` (perte de score heuristique
du propre moteur de référence, en secondes de recherche à 10s) induite par
le coup réellement choisi par le candidat, comparé au meilleur coup
disponible depuis la même position. C'est la sonde la plus fine (score
continu, pas binaire) et la plus fiable statistiquement ici.

| Checkpoint | n | `move_loss` moyen | Coups optimaux |
|---|---|---|---|
| avant 32→128 | 6 | 18,7 | 0/6 |
| après 32→128, round 1 | 6 | 22,3 (bruit, n trop petit) | 0/6 |
| après 32→128, **round 2 (n=20, réglé)** | 20 | **18,4** | 0/20 |

**Aucune amélioration détectable** entre avant et après le changement
`sims=32→128`, avec un échantillon suffisant pour trancher (round 2, n=20).
Le candidat ne trouve le meilleur coup de la position dans **aucun** des 32
cas testés au total sur l'ensemble du chantier, à un coût de score
heuristique systématique (jamais catastrophique, jamais un signe de bug
d'inversion de signe — juste un déficit modeste mais constant, cohérent avec
un réseau *faible* plutôt que *cassé*).

## 4. Analyse

Trois lignes d'évidence convergent, et une quatrième les explique :

1. **La perte d'entraînement est plate** depuis le milieu du chantier
   (~1,72-1,75 pendant 14h observées, puis ~2,69 de moyenne sur la fenêtre
   suivante) — pas de tendance baissière nette malgré des dizaines de
   milliers de générations supplémentaires.
2. **Le taux de victoire contre la référence reste au niveau du bruit**
   (0% à 4-12%, jamais significativement positif) à travers un facteur ~7,5×
   sur le nombre de générations (5765 → 43227) et un facteur ~10× sur le
   nombre de parties de self-play (128 757 → 1,2M+), y compris avec un
   budget de réflexion 6× supérieur pour le candidat.
3. **La sonde de qualité de coup la plus fiable (n=20, continue) ne détecte
   aucune amélioration** entre un quadruplement de `simulations_per_move`
   (32→128) qui a coûté un effondrement ×10 du débit de self-play.
4. **Pendant ce temps, la cohérence interne du réseau avec ses propres
   données continue de grimper sans relâche** (sign-match 53%→73%→98%).

Ce dernier point est la clé de lecture : un réseau qui devient de plus en
plus confiant et cohérent avec ses propres parties, sans que cela se traduise
par une force de jeu mesurable contre un référentiel externe, est la
signature classique d'un **plafond de bootstrap auto-play** — le réseau
apprend à prédire fidèlement l'issue de parties généréés par une recherche
elle-même trop faible (`simulations_per_move` insuffisant, branchement moyen
~33,5), et converge avec confiance croissante vers une politique
*constamment médiocre*, plutôt que de s'améliorer. C'est exactement le mode
d'échec qu'un quadruplement de `simulations_per_move` visait à corriger — et
qui ne l'a pas corrigé, ce qui indique que le goulot n'est pas seulement
« recherche trop peu profonde en self-play », ou qu'il l'est encore trop peu
même à 128 (voir alternatives, §5).

**Ce n'est pas un problème de bug.** Les deux diagnostics (pilote
asymétrique, sonde de qualité de coup) ont été conçus précisément pour
exclure cette hypothèse, et l'excluent : pas de signe d'inversion valeur/
signe, pas de recherche du pire coup, pas d'échec catastrophique — un
déficit modeste et systématique, cohérent avec un réseau sous-dimensionné
et/ou des données de self-play de qualité insuffisante, pas avec un câblage
cassé.

## 5. Pertinence de poursuivre l'entraînement dans ces conditions

**Non recommandé.** Continuer à entraîner en l'état (même réseau, même
`simulations_per_move=128`, même régime de self-play) revient à répéter
exactement l'expérience qui vient d'échouer à montrer un effet : la fenêtre
32→128 a déjà coûté ~16h de calcul (128 workers) pour ~11 000 générations et
~69 000 parties supplémentaires, sans gain détectable sur la sonde la plus
fiable. Rien dans les données ne suggère qu'un facteur ×2 ou ×4
supplémentaire sur la durée d'entraînement changerait cette conclusion — la
perte est plate, pas en train de ralentir sa descente ; ce n'est pas un
problème qui se résout en ajoutant du temps d'horloge au même protocole.

Ceci fait écho à un précédent déjà clos sur cet axe du projet (tiers 3 de la
feuille de route — le ranker ML, dont le signal s'est révélé réellement
négatif une fois le coût contrôlé, pas simplement un problème de coût) :
la prudence acquise sur ce point s'applique ici aussi.

## 6. Alternatives pour améliorer l'auto-entraînement

Par ordre de coût croissant et de confiance décroissante dans le diagnostic
préalable :

1. **Exploiter le GPU inutilisé (le levier le moins cher et le plus
   directement ciblé).** `santoni_az/` tourne entièrement sur CPU
   (`torch.cuda.is_available()` est `True` sur cette machine mais jamais
   utilisé). Le goulot identifié à deux reprises est la profondeur de
   recherche en self-play — déplacer l'inférence (et/ou l'entraînement) sur
   GPU rendrait une recherche beaucoup plus profonde abordable en débit,
   sans changer d'architecture ni de régime. C'est le levier le plus
   directement adressé au diagnostic actuel et le matériel est déjà
   disponible, inutilisé.
2. **Investiguer l'effondrement ×10 du débit avant de retenter un
   quadruplement des sims/coup.** L'extrapolation (×3-4 attendu, ×10
   observé) s'est révélée non fiable une fois mesurée. Avant de payer à
   nouveau ce type de coût (256, 512 sims/coup), profiler où part le temps
   réellement (saturation du process d'inférence mono-cœur, coût
   mémoire/calcul de `run_single_move`, effet d'ordonnancement des vagues) —
   un profilage coûte une fraction du temps d'un nouveau cycle
   d'entraînement complet.
3. **Tester l'hypothèse de capacité réseau séparément du temps d'entraînement.**
   4 blocs résiduels / 32 canaux est délibérément minuscule. Un test bon
   marché (quelques minutes, pas une nouvelle campagne de self-play) :
   est-ce qu'un réseau plus grand ajuste mieux la *même* distribution de
   données de self-play déjà sur disque ? Si oui, la capacité est une limite
   réelle indépendante de `simulations_per_move` ; si non, cela recentre le
   problème sur la qualité des données plutôt que sur l'architecture.
4. **Réexaminer le calendrier d'exploration (bruit de Dirichlet /
   température).** `TEMPERATURE_PLIES=6` sur un facteur de branchement
   moyen de ~33,5 est court ; une phase exploratoire trop brève pourrait
   générer des parties de self-play trop déterministes/corrélées tôt,
   nourrissant le plafond de bootstrap plutôt que de le briser. Non testé à
   ce jour.
5. **Accepter que cette famille d'approche ne batte peut-être pas la
   référence avec les ressources disponibles.** Le projet a déjà un
   précédent direct et coûteux sur ce point (ranker ML, tiers 3 de la
   feuille de route) où, une fois le facteur de confusion coût/débit
   neutralisé, le signal de fond s'est révélé réellement négatif. Les items
   1-4 ci-dessus sont des expériences bon marché et décisives à mener *avant*
   de s'engager dans un nouveau cycle d'entraînement de plusieurs heures —
   si elles échouent aussi, il devient raisonnable de considérer cet axe
   comme rejoignant le ranker ML (voir mémoire `santoni-ml-ranker-rejected`),
   et de concentrer l'effort sur les tiers 4-5 déjà identifiés dans la
   feuille de route du moteur (`santoni-engine-roadmap-priorities`), qui ne
   portent pas ce même risque de coût englouti.

## 7. Prochaine étape concrète recommandée

Ne pas relancer `selfplay-runs/run2.py` en l'état. À la place, dans l'ordre :
(a) mesurer le gain de débit d'un simple déplacement de l'inférence sur GPU
(quelques dizaines de minutes de travail, aucune nouvelle collecte de
self-play nécessaire) ; (b) profiler la cause réelle de l'effondrement ×10
du débit à `sims=128` ; (c) tester l'ablation de capacité sur les données
déjà sur disque. Les trois sont nettement moins coûteux qu'un nouveau cycle
d'entraînement de plusieurs heures, et répondraient enfin à la question
ouverte depuis le 2026-10-14 : est-ce la profondeur de recherche, la
capacité du réseau, ou autre chose qui plafonne ce chantier.

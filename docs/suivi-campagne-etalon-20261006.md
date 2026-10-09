# Suivi pérenne — étalon 30 s / profondeur 6 et comparaison des stratégies

État établi le 7 octobre 2026 à partir des artefacts du worktree
`feature/defensive-threats`. Cette note est la référence de reprise : ne pas
redemander ce contexte avant de reprendre la campagne.

## Objectif et étalon figé

Le moteur de référence est le moteur initial `cb732bb`, conservé dans
`experiments/references/initial-cb732bb/`. Son nouvel étalon a été terminé le
6 octobre : **100 parties**, correspondant aux 100 configurations historiques
(25 groupes de placements, dix confrontations de pouvoirs de base).

Pour chaque décision, la recherche s'arrête à la première butée atteinte :

- 30 secondes de réflexion ;
- profondeur nominale 6 entièrement terminée ;
- ou preuve exacte de victoire/défaite, si elle est trouvée plus tôt.

La campagne d'étalon est valide : 3 251 demi-tours, 3 255 recherches, aucune
erreur. Les arrêts observés sont 2 005 par profondeur, 747 par délai, 499 par
preuve et 4 terminaux. Elle sert de corpus fixe : elle n'est pas elle-même une
mesure de gain stratégique.

Artefacts :
`feature/defensive-threats/experiments/baseline-100-30s-depth6-20261006/` et
`feature/defensive-threats/docs/etalon-30s-profondeur6.md`.

## Protocole de comparaison à 1 000 parties

Cinq profils candidats doivent chacun jouer **200 matchs appariés** contre
l'original, soit les 100 configurations dans chacun des deux camps. Les
placements, pouvoirs et joueur qui commence sont identiques dans la paire.
Le total prévu est donc **1 000 parties**. Les deux adversaires utilisent les
mêmes butées, 30 s ou profondeur 6 achevée ; les extensions tactiques sont
comptées dans ces 30 s.

| Profil | Variante évaluée |
|---|---|
| 0 | Moteur actuel, sans option tactique : contrôle des autres écarts avec l'original |
| 1 | Alarmes exactes et prolongation des réponses forcées |
| 2 | Profil 1 + préparations calmes larges |
| 3 | Profil 1 + marge défensive limitée au jeu sans dieu, Déméter et Prométhée |
| 4 | Profil 3 + préparations gagnantes ciblées et certifiées dans ce même périmètre |

Les 25 groupes de placements sont corrélés. Les intervalles de confiance et
les différences au contrôle doivent donc être rééchantillonnés par groupe de
placement, jamais par partie isolée. Une victoire contre l'original n'est
attribuable à une option qu'après comparaison avec le profil 0.

Chaque coup est validé par les règles actuelles et celles de l'original, et
chaque partie est relue. Une preuve contredite par le résultat invalide le lot
concerné. Les résultats partiels ne sont pas des conclusions.

Le script et le protocole initial sont dans
`feature/defensive-threats/tools/options_depth_campaign.py` et
`feature/defensive-threats/docs/comparaison-options-30s-profondeur6.md`.

## Historique et incident du 6 octobre

Le premier lancement a commencé à 13:26 heure de Libreville. Il a été arrêté
après **227 parties terminées** (sur 1 000) et **3 erreurs**, toutes dans le
profil 2 : des preuves annoncées ont été contredites par le résultat (parties
43, 63 et 68 ; positions 23, 23 et 10). Ce lot est **invalide** et ne doit pas
être repris ni agrégé aux résultats futurs.

Cause : le profil 2 explore seulement un sous-ensemble de préparations calmes.
Une valeur de type mat obtenue dans cette extension sélective était présentée
comme une preuve minimax, alors que des réponses/calmes omises peuvent changer
le résultat. Le problème était une fausse certification, pas une anomalie de
rejeu.

Correction appliquée avant la reprise : pour le profil 2, une valeur issue de
cette extension reste une estimation (`proven = false`), et elle ne provoque
plus d'arrêt anticipé comme une preuve. Les alarmes exactes des autres profils
conservent leurs certificats. Cette règle est documentée dans
`native_engine/src/lib.rs` du worktree expérimental.

## Dernier point d'arrêt et résultat disponible

La reprise corrigée a démarré à 17:28 heure de Libreville et s'est terminée à
21:04, le 6 octobre. Elle n'a relancé que le **profil 2** dans un nouveau
dossier, ce qui évite de mélanger les binaires et les sémantiques de preuve.

| Lot valide | État | Variante | Original | Conclusion |
|---|---:|---:|---:|---|
| Profil 2 corrigé | 200 / 200 | 77 | 123 | Régression nette ; ne pas retenir |

Le taux de victoire du profil 2 corrigé est 38,5 %, avec un intervalle bootstrap
à 95 % de 31,5–45,0 % par groupe de placements. Il ne contient ni erreur, ni
partie non conclue. Sa recherche candidate consomme en moyenne 22,82 s par
décision, contre 11,03 s pour l'original : le coût des préparations larges est
un élément plausible de la régression, sans prouver à lui seul la causalité.

Le 7 octobre, une reprise contrôlée des profils 0, 1, 3 et 4 a été lancée dans
`experiments/options-30s-depth6-20261006-profiles-0134-resume/`. Elle importe
les **46 parties terminées sans erreur** de chacun de ces profils, avec le même
binaire candidat figé, puis joue les **154 matchs restants par profil**, soit
**616 matchs à produire**. Le profil 2 reste terminé et écarté. Seuls ses 46
résultats issus de l'ancien lot, ses 3 erreurs et ses 2 tentatives interrompues
sont exclus.

Artefacts de l'arrêt invalide :
`feature/defensive-threats/experiments/options-30s-depth6-20261006/`.
Artefacts du profil 2 corrigé :
`feature/defensive-threats/experiments/options-30s-depth6-20261006-mode2-fixed/`.

## Règle de reprise

Avant toute nouvelle exécution, conserver séparés le binaire candidat, la
référence originale, le script, les limites et les résultats. Ne jamais
réemployer les résultats défectueux du profil 2. Les parties finies sans erreur
des profils 0, 1, 3 et 4 peuvent être importées dans un nouveau manifeste si
leurs jobs et le SHA-256 du binaire candidat sont identiques. Produire un
rapport par profil ; ne conclure qu'après les 200 parties d'un profil, l'audit
de tous ses historiques et la comparaison au contrôle 0 apparié.

## Bilan final des 1 000 matchs

Les cinq profils ont terminé leurs 200 matchs. Aucun ne bat l'original figé :

| Profil | Victoires variante / 200 | Écart au contrôle 0 | Conclusion |
|---|---:|---:|---|
| 0 — contrôle | 82 | — | Le moteur actuel hors options est inférieur à l'original (41,0 %) |
| 1 — alarmes exactes | 90 | +4,0 points, IC95 −2,0 à +10,0 | Signal insuffisant |
| 2 — préparations larges | 77 | −2,5 points, calculé contre le contrôle séparé | Régression, écarté |
| 3 — alarmes et marge | 84 | +1,0 point, IC95 −4,5 à +7,0 | Signal insuffisant |
| 4 — marge + préparations ciblées | 89 | +3,5 points, IC95 −2,5 à +9,5 | Signal insuffisant |

Les écarts des profils 1, 3 et 4 sont appariés par placement et leurs
intervalles contiennent zéro. Ils ne justifient donc pas une intégration dans
le moteur. Les divergences entre une ligne prouvée et le vainqueur final sont
archivées pour revue : elles signifient qu’un joueur a quitté la ligne optimale,
pas qu’une preuve est réfutée.

## Protocole ciblé Athéna — décidé le 8 octobre 2026

L'essai « sentinelle T3 » est rejeté : sur 200 parties à 30 secondes sans
plafond de profondeur, il obtient 54 victoires contre 146 pour l'original. Une
nouvelle adaptation est donc limitée à **Athéna** : elle ne change ni les
règles, ni l'évaluation, ni l'élagage. Elle ordonne seulement plus tôt une
montée qui active le verrou d'Athéna **et** empêche au moins une montée légale
adverse. Un départage léger favorise une suite vers un niveau 2.

Le test générique de 200 parties lancé le 8 octobre a été interrompu avant
toute partie terminée, car seules 20 parties auraient effectivement donné
Athéna au candidat : ce protocole diluait la mesure.

Le prochain protocole approuvé comprend **100 parties** : 25 positions de
départ enregistrées, chacune rejouée dans les quatre conditions suivantes :

| Condition | Moteur Athéna | Moteur X | Premier joueur |
|---|---|---|---|
| 1 | Athéna vanilla, moteur initial figé | Moteur initial figé | Athéna |
| 2 | Athéna vanilla, moteur initial figé | Moteur initial figé | X |
| 3 | Athéna2 (ordonnancement « verrou utile ») | Moteur initial figé | Athéna |
| 4 | Athéna2 (ordonnancement « verrou utile ») | Moteur initial figé | X |

Ainsi, 50 parties utilisent Athéna vanilla et 50 Athéna2, avec le même dieu X,
les mêmes ouvriers et les mêmes hauteurs initiales pour chaque quadruplet. Tous
les moteurs emploient la première butée atteinte : **30 secondes ou profondeur
6 entièrement achevée**. Les binaires sont figés séparément et chaque coup est
validé par les deux implémentations des règles. Le bilan principal est la
différence Athéna2 moins Athéna vanilla, appariée par position et premier
joueur ; les temps, nœuds et profondeurs sont des mesures secondaires.

Les cinq dieux X fixés sont **Apollon, Minotaure, Pan, Déméter et Atlas**. Les
25 positions sont donc réparties en cinq placements par confrontation Athéna-X.
Les cinq placements doivent être conservés identiques pour chaque confrontation
afin que l'effet du dieu X soit comparable. Aucun résultat de l'essai Athéna ne
doit être agrégé à la campagne des 1 000 matchs.

### Résultat

La campagne est terminée : **100/100**, aucune erreur ni partie non conclue.
Athéna vanilla (moteur initial dans les deux camps) gagne 37 parties sur 50 ;
Athéna2 (ordonnancement « verrou utile ») en gagne 28 sur 50. Sur les 50
paires strictement comparables, Athéna2 améliore 7 résultats, en dégrade 16 et
en laisse 27 identiques, soit un écart net de **−9 victoires (−18 points)**.

| X | Vanilla | Athéna2 | Écart Athéna2 − vanilla |
|---|---:|---:|---:|
| Apollon | 4 / 10 | 3 / 10 | −1 |
| Minotaure | 9 / 10 | 7 / 10 | −2 |
| Pan | 10 / 10 | 8 / 10 | −2 |
| Déméter | 9 / 10 | 5 / 10 | −4 |
| Atlas | 5 / 10 | 5 / 10 | 0 |

La stratégie, dans cette implémentation, est rejetée : elle atteint moins
souvent la profondeur 6 (365 décisions sur 876, contre 1 998 sur 2 633 pour
les décisions initiales) et consomme 16,50 s en moyenne par décision, contre
6,52 s. Elle doit être retirée ou profondément simplifiée avant tout nouvel
essai.

## Expérience « blocage niveau 2 » — 8 octobre 2026

Un moteur dédié (mode 7) ordonne plus tôt une construction ordinaire de niveau
1 vers niveau 2 seulement lorsqu'elle retire effectivement une ou plusieurs
montées légales à l'adversaire. Le signal est exact dans son périmètre (jeu sans
pouvoir), ne modifie ni l'évaluation ni l'élagage, et ne s'applique pas aux
parties avec dieux.

La campagne A/B est complète : 25 positions ordinaires, candidat dans les deux
couleurs, contrôle mode 0 contre stratégie mode 7, soit **100/100 parties** et
50 paires comparables. Aucune erreur ni partie non conclue.

| Candidat | Victoires / 50 | Défaites / 50 |
|---|---:|---:|
| Contrôle actuel, mode 0 | 21 | 29 |
| Blocage niveau 2, mode 7 | 23 | 27 |

Sur les paires : 3 améliorations, 1 régression et 46 résultats inchangés ;
l'écart est +2 victoires (+4 points). C'est un **signal faible, non concluant**,
pas une intégration : le mode 7 atteint moins souvent la profondeur 6 (395 sur
910 décisions contre 543 sur 1 018) et prend plus de temps par décision (17,53
s contre 14,66 s). Limite à corriger avant un nouveau test : dans cette première
campagne le candidat commence toutes les parties ; ses deux couleurs sont
alternées, mais pas l'ordre de jeu. Il faut une confirmation avec premier joueur
alterné indépendamment de la couleur du candidat. **Cette confirmation n'a pas
encore été lancée** ; le travail du 9 octobre a porté sur l'axe ML ci-dessous à
la place.

## Axe classement de tours par modèle (ML) — rejeté le 9 octobre 2026

Protocole complet dans
`feature/defensive-threats/docs/protocole-ml-strategie-20261008.md`. Deux
candidats ont été soumis au même test d'acceptation à 400 parties appariées
(50 positions figées × 2 couleurs × 2 premiers joueurs, 30 s ou profondeur 6) :

| Candidat | Taux ML | Taux contrôle | Écart | Paires (+1/0/-1) |
|---|---:|---:|---:|---:|
| Ranker v1 (8 oct.) | 46,5 % | 47,0 % | −0,5 pt | 6 / 187 / 7 |
| Ranker hybride, génération 7 (9 oct.) | 45,0 % | 47,0 % | −2,0 pt | 10 / 176 / 14 |

Les deux échouent nettement le seuil de +5 points requis, et le candidat issu
de dix générations d'auto-jeu est pire que le ranker v1 le plus simple. Cause
plausible : le calcul des caractéristiques coûte 8 à 11 fois plus cher par
décision que l'original, ce qui fait chuter la part de décisions atteignant la
profondeur 6 (de 67 % à 44–57 %) — la perte de profondeur dépasse tout gain
d'ordonnancement. L'axe est **rejeté**, non fusionné ; une reprise ne serait
justifiée qu'avec un coût par appel au moins dix fois plus faible.

## Prochaine étape ouverte

Un seul fil reste réellement ouvert : la confirmation du « blocage niveau 2 »
(mode 7) avec premier joueur alterné indépendamment de la couleur du candidat,
décrite ci-dessus. L'axe ML est clos et ne doit pas être relancé sans nouvelle
implémentation nettement moins coûteuse par appel.

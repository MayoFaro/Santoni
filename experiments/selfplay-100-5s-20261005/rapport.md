# Campagne Santoni : 100 parties robot contre robot

100 parties terminées ; **2972 tours joués**, 2974 recherches. Budget maximal : **5 secondes par tour**.
Durée de la campagne : 16.9 minutes, avec 12 processus à priorité réduite sur des cœurs distincts.

## Protocole

Même version du moteur Rust des deux côtés, issue de main `cb732bb`. Aucune modification du moteur durant la campagne. Chaque coup publié est validé par les règles Python ; chaque historique est ensuite rejoué intégralement.
25 placements initiaux tirés avec une graine fixe, chacun décliné en quatre parties : inversion du premier joueur et de l’attribution des pouvoirs. Pour les parties sans pouvoirs, la deuxième paire utilise une rotation de 180° du plateau. Les placements sont variés, pas optimisés par le conseiller de placement. Les deux camps reçoivent le même budget.
Les parties de contrôle et les réanalyses à 20 ou 60 secondes décrites plus bas sont exclues du total de 100.
Graine : `20261005`. Empreinte de la bibliothèque : `9f5edd2e0d2b279486a44bb79f306297bceb6c9c883113c5163e57cf937b873b`.

## Résultats

| Pouvoirs | Parties | Victoires | Victoires du premier joueur |
|---|---:|---|---:|
| Aucun pouvoir/Aucun pouvoir | 12 | Aucun pouvoir : 12 | 9/12 |
| Hermes/Hephaestus | 12 | Hephaestus : 7, Hermes : 5 | 7/12 |
| Apollo/Minotaur | 12 | Apollo : 12 | 6/12 |
| Artemis/Athena | 12 | Athena : 9, Artemis : 3 | 5/12 |
| Atlas/Pan | 12 | Pan : 8, Atlas : 4 | 4/12 |
| Demeter/Prometheus | 8 | Demeter : 3, Prometheus : 5 | 7/8 |
| Apollo/Athena | 8 | Apollo : 4, Athena : 4 | 2/8 |
| Artemis/Pan | 8 | Artemis : 7, Pan : 1 | 5/8 |
| Atlas/Demeter | 8 | Atlas : 3, Demeter : 5 | 3/8 |
| Minotaur/Prometheus | 8 | Minotaur : 4, Prometheus : 4 | 6/8 |

Le premier joueur gagne **54/100** parties. Longueur médiane : **29 tours** ; minimum 9, maximum 58.

Causes finales des victoires :
- Aucun tour complet légal : 2.
- Montée au niveau 3 : 93.
- Descente de deux niveaux (Pan) : 5.

Ces résultats décrivent cette version du moteur et ces placements. Ils ne constituent pas un classement universel des dieux : les groupes ne comptent que 8 ou 12 parties et leurs quatre variantes partagent un placement.

## Mécanismes observés

93 victoires viennent d’une montée au niveau 3, 5 de la condition de descente de Pan, 2 de l’absence de tour légal. Les victoires par blocage concernent une partie sans pouvoirs et une partie remportée par Atlas contre Pan.
Dans Apollo contre Minotaur, 2 des 12 victoires d’Apollo se terminent directement par un échange avec un pion adverse sur la case de niveau 3 ; les dix autres se terminent par une montée ordinaire. Cela documente deux utilisations décisives du pouvoir, sans expliquer à lui seul le résultat de 12–0.

## Fonctionnement de la recherche

Profondeur médiane : **5**. 2304 recherches se terminent faute de temps, 668 s’arrêtent sur un résultat annoncé comme démontré, 2 constatent une position sans tour légal.
Durée moyenne du calcul : 3.95 s ; médiane 5.00 s. Le budget est une limite, pas une durée imposée lorsque la victoire ou la défaite est démontrée.

| Pouvoir au trait | Profondeur médiane | Recherches |
|---|---:|---:|
| Aucun pouvoir | 6 | 422 |
| Hermes | 4 | 183 |
| Hephaestus | 4 | 184 |
| Apollo | 6 | 405 |
| Minotaur | 5 | 319 |
| Artemis | 5 | 304 |
| Athena | 5 | 358 |
| Atlas | 5 | 223 |
| Pan | 5 | 253 |
| Demeter | 4 | 161 |
| Prometheus | 4 | 162 |

Ces profondeurs dépendent aussi du pouvoir adverse et du stade de la partie ; elles incluent les arrêts anticipés après démonstration.

Sur un placement identique au sol, le nombre de positions finales distinctes d’un tour est :
- Aucun pouvoir : 80 (80 séquences générées avant dédoublonnage).
- Apollo : 80 (80 séquences générées avant dédoublonnage).
- Artemis : 120 (416 séquences générées avant dédoublonnage).
- Athena : 80 (80 séquences générées avant dédoublonnage).
- Atlas : 160 (160 séquences générées avant dédoublonnage).
- Demeter : 254 (428 séquences générées avant dédoublonnage).
- Hephaestus : 160 (160 séquences générées avant dédoublonnage).
- Hermes : 4198 (4784 séquences générées avant dédoublonnage).
- Minotaur : 80 (80 séquences générées avant dédoublonnage).
- Pan : 80 (80 séquences générées avant dédoublonnage).
- Prometheus : 584 (640 séquences générées avant dédoublonnage).

Hermès possède donc ici environ 52 fois plus de successeurs distincts que les règles sans pouvoirs. Cette mesure explique un coût de recherche élevé ; elle ne mesure pas directement la force du pouvoir.

## Exemple confirmé d’erreur évitable

Partie 16, tour 26 : Héphaïstos joue contre Hermès. Avec 5 secondes, il choisit **B4 → C4**, puis construit deux blocs en **B4**, qui passe du niveau 1 au niveau 3. Son estimation est alors +430 à profondeur 7. Après ce tour, Hermès démontre une victoire forcée et gagne au tour 33.
Une réanalyse à 20 secondes atteint la profondeur 8 et choisit le même déplacement, mais **une seule construction en D4**. L’estimation retombe à +62.
Après cette alternative, une recherche du côté Hermès démontre au contraire sa défaite (-99992 à profondeur 8). La reprise de la partie avec 5 secondes par tour confirme **la victoire d’Héphaïstos au tour 34**.
Le changement porte sur la décision de construction ; les réponses ultérieures sont recalculées. Dans la position originale, C4 est au sol et la double construction supprime l’accès direct à B4 en la portant au niveau 3. La construction en D4 crée un premier étage accessible. Cette géométrie est une explication plausible du contraste ; l’inversion du résultat est confirmée par la recherche et la reprise.

## Pourquoi 20 secondes peuvent laisser la profondeur inchangée

Partie 18, avant le tour 28 : la profondeur 5 est terminée en environ **1,2 seconde**. La profondeur 6 ne se termine qu’après **34,4 secondes**. Avec 5 ou 20 secondes, le résultat affiché reste donc celui de profondeur 5, même si le calcul continue.
À 60 secondes, la profondeur 6 est achevée, la profondeur 7 demeure incomplète, le score passe de +414 à -154 et le coup proposé change. Cela illustre une estimation trop optimiste à un horizon limité, et non une limite volontaire à profondeur 3.

## Priorités d’amélioration suggérées

1. Ajouter des positions critiques de cette campagne à un corpus de comparaison, notamment les tours 26 de la partie 16 et 28 de la partie 18.
2. Mieux ordonner les coups et approfondir sélectivement les positions tactiques : menaces de victoire, défenses forcées, constructions qui ferment une rampe. Respecter le budget global de 5 secondes et vérifier que les preuves restent valides.
3. Tester une évaluation plus précise de l’accès effectif aux niveaux 2 et 3, des rampes et de l’immobilisation, ainsi que de la mobilité propre aux pouvoirs. Le cas Héphaïstos montre que la double construction facultative ne doit pas être favorisée automatiquement.
4. Optimiser la génération et les équivalences de positions, particulièrement pour Hermès. Le moteur dédoublonne déjà les successeurs identiques ; les symétries des deux bâtisseurs pourraient offrir un gain supplémentaire si les actions sont remappées correctement.
5. Confronter toute version modifiée à cette version figée, sur des matchs appariés à 5 secondes par tour et des placements supplémentaires. Une victoire contre soi-même ne démontre pas un progrès.

## Audit et fichiers

**Aucun historique invalide et aucune contradiction observée entre une victoire annoncée comme démontrée et le vainqueur final.** Cette vérification porte sur les règles implémentées et les parties observées ; les deux moteurs peuvent partager une erreur de règle.
- `campaign.json` : protocole, graine, bibliothèque et 100 configurations.
- `game-001.json` à `game-100.json` : parties complètes et décisions.
- `summary.json`, `analysis.json` : résultats et audit.
- `recheck-016-026.json`, `alternative-016-026.json`, `continuation-016-026.json` : erreur évitable et contrôle.
- `recheck-018-028-60s.json` : chronologie des profondeurs.

Aucune stratégie de production n’a été modifiée ; cette campagne constitue une base mesurée pour les prochaines améliorations.

# Bilan de la campagne : moteur initial à 35 s contre 5 s

Date : 5 octobre 2026. Sources du moteur : `cb732bb`, sans modification stratégique.

**Le temps supplémentaire apporte un gain net sur ce corpus : 144 victoires sur
200 pour le robot à 35 secondes, soit 72 %.** Il permet notamment de corriger
la construction problématique d’Héphaïstos et d’établir une victoire sur cette
position. Le moteur bénéficie donc réellement d’une recherche plus longue.
Cela justifie d’améliorer son efficacité à 5 secondes, tout en conservant le
travail prévu sur l’évaluation des constructions.

## 1. Protocole et portée du résultat

Les 100 configurations de la première campagne ont été reprises exactement :
placements, pouvoirs et premier joueur. Chacune a été jouée deux fois, en
inversant les budgets entre joueurs. Les deux camps utilisent le même binaire
figé et les mêmes règles Python de validation. Les 200 parties correspondent
à 25 groupes de placements ; leurs variantes ne sont pas indépendantes.

Le binaire historique n’était pas disponible : les sources identiques ont été
recompilées. Le PC actuel utilise six cœurs physiques, contre douze processus
épinglés à des cœurs distincts pour la première campagne. La comparaison interne
35 s / 5 s est équilibrée ; une comparaison directe des performances avec la
première campagne reste limitée par ce changement matériel et de compilation.
La charge et les budgets muraux empêchent de garantir une reproduction exacte
des décisions, même avec les mêmes placements.

Références : [protocole](campagne-35s-vs-5s.md),
[manifeste](../experiments/baseline-35s-vs-5s-20261005/campaign.json),
[référence conservée](../experiments/references/initial-cb732bb/README.md).

## 2. Résultats et solidité du constat

| Indicateur | Résultat |
|---|---:|
| Parties terminées | 200 / 200 |
| Victoires à 35 s / à 5 s | 144 / 56 |
| Taux de victoire à 35 s | 72 % |
| Intervalle bootstrap à 95 %, regroupé par placement | 66–78 % |
| Victoires à 35 s lorsque ce robot commence | 75 / 100 |
| Victoires à 35 s lorsque ce robot joue second | 69 / 100 |
| Configurations gagnées deux fois par le budget 35 s | 46 / 100 |
| Configurations partagées, une victoire par budget | 52 / 100 |
| Configurations perdues deux fois par le budget 35 s | 2 / 100 |
| Groupes de placements favorables / équilibrés / défavorables au budget 35 s | 21 / 3 / 1 |

L’avantage ne s’explique donc pas uniquement par le premier joueur ni par
quelques configurations isolées. L’intervalle est obtenu avec 10 000
rééchantillonnages des groupes de placements entiers, graine 20261005.
Il reste conditionnel à ce corpus et à ces confrontations : il ne mesure pas
la force face à des humains ni face à d’autres architectures de moteur.

Les 52 configurations partagées signifient qu’après inversion du budget,
le même camp remporte les deux parties. Ce résultat ne suffit pas à qualifier
ces positions de gains forcés : placement, pouvoir et décisions ultérieures
peuvent expliquer cette stabilité. Les deux configurations gagnées deux fois
à 5 s rappellent qu’un budget plus long ne garantit pas une meilleure décision
dans chaque partie.

Source : [résumé détaillé](../experiments/baseline-35s-vs-5s-20261005/summary.json).
Les répartitions selon le premier joueur et les placements ont été recomptées
depuis les 200 fichiers `game-*.json`.

## 3. Ce que le temps supplémentaire change dans la recherche

| Mesure | 5 s | 35 s |
|---|---:|---:|
| Recherches | 2 775 | 2 822 |
| Profondeur médiane terminée | 5 | 6 |
| Profondeur moyenne terminée | 5,09 | 5,75 |
| Temps moyen réel par recherche | 3,92 s | 25,82 s |
| Arrêts sur budget | 2 123 (76,5 %) | 2 033 (72,0 %) |
| Résultats marqués démontrés, terminaux compris | 652 | 789 |
| Durée réelle maximale d’une recherche | 5,066 s | 35,141 s |

Le budget maximal est multiplié par sept ; le temps moyen réellement consommé
est multiplié par environ 6,6. La profondeur médiane ne gagne qu’un niveau.
Cela est compatible avec un coût d’exploration qui augmente fortement avec la
profondeur. Les matchs visitent cependant des positions différentes : ces
moyennes ne mesurent pas le gain de profondeur sur une même position.

Les recherches cumulent environ **23,26 heures de calcul mural de recherche**,
réparties entre six processus. Les matchs se sont déroulés de 07 h 53 à 11 h 50,
heure de Libreville, soit environ 3 h 56 ; réanalyses et rapport se terminent
vers 11 h 51. Les 5 594 tours ont nécessité 5 597 recherches, dont trois
constats terminaux sans tour à jouer.

Les restitutions dépassent légèrement les budgets nominaux. Il serait donc
inexact de parler d’une limite stricte de 5,000 ou 35,000 secondes incluant
la conversion Python. Ce banc ne mesure pas non plus la latence de l’interface.

**Interprétation :** obtenir plus tôt les bonnes coupures et réduire le travail
inutile peuvent améliorer les décisions à budget court. Cette campagne ne
prouve pas qu’une optimisation précise permettra de reproduire à 5 s les
résultats obtenus à 35 s, ni qu’il suffira d’accélérer le moteur sept fois.

## 4. Deux positions identiques, deux enseignements

### Partie 16, tour 26 : la recherche corrige effectivement la construction

À 5 s, le moteur termine la profondeur 7 avec une estimation de +430. Il déplace
le bâtisseur de B4 vers C4 et construit deux fois en B4, comme dans le cas
problématique de la première campagne.

À 35 s, il conserve le déplacement mais construit une seule fois en D4.
Il termine la profondeur 9 et annonce une victoire démontrée, score +99991,
après environ **17,94 secondes**. La démonstration s’entend avec les règles
implémentées ; elle ne constitue pas un certificat indépendant complet.

Cette observation renforce le diagnostic d’un défaut à horizon court : le
moteur actuel possède déjà les règles et la recherche nécessaires pour trouver
une meilleure construction, mais ne l’identifie pas dans les cinq secondes.
Elle ne démontre pas que la rampe explique à elle seule tout le résultat.

Objectif concret pour les futures branches : retrouver cette décision et,
si possible, cette preuve à 5 s, sans introduire une règle artificielle qui
interdirait les doubles constructions d’Héphaïstos.

Sources : [réanalyse 5 s](../experiments/baseline-35s-vs-5s-20261005/recheck-016-026-5s.json),
[réanalyse 35 s](../experiments/baseline-35s-vs-5s-20261005/recheck-016-026-35s.json).

### Partie 18, tour 28 : une estimation optimiste disparaît sans preuve de résultat

À 5 s : profondeur 5, score +414. À 35 s : profondeur 6, score −154,
avec un autre tour choisi. Aucun des deux résultats n’est démontré.

Le temps supplémentaire modifie donc sensiblement l’appréciation et la décision.
Il ne permet pas d’affirmer que la position est perdue, ni que le nouveau coup
gagne. Une évaluation positive à faible profondeur ne doit pas être présentée
comme une probabilité de victoire ou une position objectivement favorable.

Sources : [réanalyse 5 s](../experiments/baseline-35s-vs-5s-20261005/recheck-018-028-5s.json),
[réanalyse 35 s](../experiments/baseline-35s-vs-5s-20261005/recheck-018-028-35s.json).

## 5. Les pouvoirs ne bénéficient pas tous pareil du temps

| Confrontation | Victoires du budget 35 s | Taux |
|---|---:|---:|
| Sans pouvoir / sans pouvoir | 17 / 24 | 70,8 % |
| Hermès / Héphaïstos | 14 / 24 | 58,3 % |
| Apollo / Minotaure | 14 / 24 | 58,3 % |
| Artémis / Athéna | 21 / 24 | 87,5 % |
| Atlas / Pan | 17 / 24 | 70,8 % |
| Déméter / Prométhée | 12 / 16 | 75,0 % |
| Apollo / Athéna | 15 / 16 | 93,8 % |
| Artémis / Pan | 10 / 16 | 62,5 % |
| Atlas / Déméter | 10 / 16 | 62,5 % |
| Minotaure / Prométhée | 14 / 16 | 87,5 % |

Les dix confrontations favorisent descriptivement le budget long, mais chacune
ne contient que deux ou trois placements distincts. Les écarts entre lignes
ne permettent pas d’établir un classement fiable des pouvoirs ni de leurs gains
respectifs avec le temps.

Le résultat modeste d’Hermès / Héphaïstos ne prouve pas que le temps est inutile
pour Hermès. Le coût de génération connu en fait toujours une priorité de
profilage, à confirmer par des mesures internes plutôt que par ce seul score.

## 6. Fiabilité et limites de validation

Les 200 historiques ont été relus avec les règles Python figées. Les
attributions de budgets, placements et pouvoirs concordent avec le manifeste.
Les empreintes du moteur et du script correspondent à celles enregistrées.
Aucun historique invalide, aucune partie non conclue et aucune contradiction
entre résultat démontré et vainqueur final n’ont été observés. Les neuf tests
du protocole et de reprise passent également.

L’absence de contradiction ne prouve pas l’absence de bugs communs aux règles
Rust et Python. Les réanalyses fixes ne portent que sur deux positions et
n’ont pas été répétées sous différentes charges. Le corpus reprend les
placements connus ; aucune généralisation à de nouveaux placements n’a encore
été mesurée. Les héros et dieux avancés ne sont pas évalués ici.

Source : [audit des parties](../experiments/baseline-35s-vs-5s-20261005/analysis.json).

## 7. Conséquences pour le plan d’amélioration

1. **Conserver 5 s comme budget principal de comparaison des améliorations.**
   Le budget 35 s devient une référence complémentaire de qualité et un moyen
   d’approfondir les positions difficiles. Cette expérience ne justifie pas à
   elle seule de rendre tous les tours sept fois plus longs dans l’application.
2. **Instrumenter la recherche et construire le banc entre versions.** Le banc
   présent compare deux budgets du même moteur ; il faut encore permettre de
   sélectionner un binaire distinct par camp pour tester les améliorations.
3. **Développer séparément l’évaluation des constructions et le tri tactique.**
   Le cas 16/26 fournit une cible vérifiable pour les deux axes. Les constructions
   accessibles, menaces propres aux pouvoirs et contre-exemples doivent être
   testés sur davantage de positions, sans régler tout le moteur sur ce seul cas.
4. **Corriger la mise à jour du cache saturé dans une variante isolée**, puis
   décider des réductions de copies et de la génération progressive sur profilage.
   La campagne montre l’intérêt de calculer plus efficacement, sans mesurer
   encore le rendement de ces changements particuliers.
5. **Réserver les extensions tactiques à une étape ultérieure**, après stabilisation
   du tri et du coût de génération. Les estimations restent distinctes des preuves.
6. **Valider chaque axe contre la référence initiale à budget égal**, puis contre
   le `main` courant si d’autres améliorations sont intégrées. Inclure des
   placements nouveaux réservés à la validation et vérifier les combinaisons.

Chaque axe reste développé sur une branche distincte. La fusion dans `main`
intervient après campagne et validation. Cette publication conserve les données,
le banc et la référence ; elle ne modifie pas le moteur ni le budget de l’application.

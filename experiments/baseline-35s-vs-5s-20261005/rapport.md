# Campagne moteur initial : 35 s contre 5 s

Parties terminées : 200/200.

Les 100 configurations initiales sont chacune jouées deux fois, avec inversion des budgets.
Même moteur et mêmes règles figés pour les deux joueurs ; aucune amélioration stratégique.

**Limite de comparabilité historique :** binaire recompilé depuis les mêmes sources, sur un autre environnement matériel. La première campagne utilisait 12 cœurs ; celle-ci en utilise 6.

Victoires à 35 s : 144 ; à 5 s : 56.
Taux de victoire à 35 s : 72.0%. Intervalle bootstrap à 95 %, regroupé par placement : [66.0%, 78.0%].
Cet intervalle dépend des 25 groupes de placements disponibles ; il ne démontre pas une supériorité universelle.

| Budget | Recherches | Profondeur médiane | Temps moyen réel |
|---|---:|---:|---:|
| 5 s | 2775 | 5 | 3.922 s |
| 35 s | 2822 | 6.0 | 25.825 s |

Les profondeurs agrégées concernent des positions différentes et ne constituent pas une comparaison à position fixe.

| Confrontation | Parties | Victoires 35 s | Victoires 5 s |
|---|---:|---:|---:|
| Aucun pouvoir/Aucun pouvoir | 24 | 17 | 7 |
| Hermes/Hephaestus | 24 | 14 | 10 |
| Apollo/Minotaur | 24 | 14 | 10 |
| Artemis/Athena | 24 | 21 | 3 |
| Atlas/Pan | 24 | 17 | 7 |
| Demeter/Prometheus | 16 | 12 | 4 |
| Apollo/Athena | 16 | 15 | 1 |
| Artemis/Pan | 16 | 10 | 6 |
| Atlas/Demeter | 16 | 10 | 6 |
| Minotaur/Prometheus | 16 | 14 | 2 |

Erreurs ou parties non conclues : 0.
Historiques invalides : 0. Contradictions de preuves : 0.

## Positions identiques

| Position | Budget | Profondeur | Score heuristique | Démontré |
|---|---:|---:|---:|---|
| 16/26 | 35 s | 9 | 99991 | True |
| 16/26 | 5 s | 7 | 430 | False |
| 18/28 | 35 s | 6 | -154 | False |
| 18/28 | 5 s | 5 | 414 | False |

Les scores heuristiques ne sont pas des probabilités de victoire. Les actions et variantes complètes sont conservées dans les fichiers recheck.
Les réanalyses sont réalisées séparément, après les matchs, sans concurrence du banc de parties.

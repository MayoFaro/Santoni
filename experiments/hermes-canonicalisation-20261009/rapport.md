# Candidat (génération progressive + canonisation) contre original `b698745` figé

30 secondes ou profondeur 6 entièrement achevée, première butée atteinte.

Parties terminées : 200/200.

Les 100 configurations du corpus canonique (`experiments/selfplay-100-5s-20261005/campaign.json`, graine 20261005) sont rejouées camp inversé : 200 parties. Seul `native_engine/src/lib.rs` change entre le candidat et la référence ; les règles Python sont identiques des deux côtés.

Victoires du candidat : 108 / 200 (54.0 %).

Intervalle bootstrap à 95 %, regroupé par placement (25 groupes, 5000 tirages) : [0.5, 0.585].

| Confrontation | Parties | Victoires du candidat |
| --- | ---: | ---: |
| Aucun pouvoir/Aucun pouvoir | 24 | 12 |
| Hermes/Hephaestus | 12 | 6 |
| Hephaestus/Hermes | 12 | 8 |
| Apollo/Minotaur | 12 | 6 |
| Minotaur/Apollo | 12 | 5 |
| Artemis/Athena | 12 | 7 |
| Athena/Artemis | 12 | 7 |
| Atlas/Pan | 12 | 7 |
| Pan/Atlas | 12 | 7 |
| Demeter/Prometheus | 8 | 4 |
| Prometheus/Demeter | 8 | 4 |
| Apollo/Athena | 8 | 4 |
| Athena/Apollo | 8 | 4 |
| Artemis/Pan | 8 | 4 |
| Pan/Artemis | 8 | 4 |
| Atlas/Demeter | 8 | 5 |
| Demeter/Atlas | 8 | 4 |
| Minotaur/Prometheus | 8 | 6 |
| Prometheus/Minotaur | 8 | 4 |

| Camp | Décisions | Profondeurs | Arrêts | Butée atteinte | Temps moyen |
| --- | ---: | --- | --- | ---: | ---: |
| candidate | 3093 | {5: 1118, 6: 947, 4: 523, 2: 121, 3: 275, 1: 108, 0: 1} | {'timeout': 1648, 'depth_limit': 947, 'proof': 497, 'terminal': 1} | 947 | 19.093s |
| original | 3087 | {5: 1099, 6: 908, 3: 256, 1: 92, 4: 573, 2: 156, 0: 3} | {'timeout': 1695, 'depth_limit': 908, 'proof': 481, 'terminal': 3} | 908 | 20.067s |

Erreurs ou parties non conclues : 0.
Contradictions preuve/vainqueur final : 0.

Les 25 groupes de placements corrélés (matchup x ouverture) ne sont pas 100 placements indépendants ; voir le rapport pour la méthode de bootstrap.

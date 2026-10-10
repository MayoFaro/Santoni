# Campagne à 200 parties : génération progressive + canonisation contre l'original figé

Date : 9-10 octobre 2026. Branche `improve/hermes-generation`. Candidat :
`f45035a` (génération progressive `visit_base` + canonisation
`canonical_key`, lot de base ajusté à 256). Référence : `main@b698745`,
point de divergence de cette branche.

## Protocole

Même protocole que `tools/baseline_depth_campaign.py` et
`tools/options_depth_campaign.py` (worktree `defensive-threats`, branche
`feature/defensive-threats`), adapté dans ce worktree
(`tools/hermes_canonicalisation_campaign.py`) :

- Corpus canonique de 100 configurations
  (`experiments/selfplay-100-5s-20261005/campaign.json`, graine 20261005),
  chacune rejouée camp inversé : **200 parties**.
- 30 secondes ou profondeur 6 entièrement achevée, première butée atteinte.
  L'arrêt anticipé du moteur sur une victoire/défaite démontrée est
  conservé.
- Deux binaires isolés par sous-processus, un seul changement entre eux
  (`git diff b698745..HEAD --stat` : uniquement `native_engine/src/lib.rs`)
  — pas de dérive des règles Python à valider séparément, un seul paquet
  `santorini` arbitre la légalité des deux côtés.
- Dix cœurs physiques visés, réduits à 4 (CPUs `[0, 2, 4, 6]`, modèle
  `13th Gen Intel(R) Core(TM) i7-13650HX`) par courtoisie envers une autre
  campagne déjà en cours sur cette machine partagée ; la concurrence a été
  temporairement réduite à 1 partie active (`SIGSTOP`/`SIGCONT` sur les
  processus ouvriers, aucune progression perdue) le temps que cette autre
  campagne se termine, puis restaurée à 4.
- Bilan par groupe de placement (matchup x ouverture, pas par partie
  isolée), intervalle bootstrap à 95 % par rééchantillonnage des groupes
  (5000 tirages), comme les campagnes précédentes de ce projet.

## Résultat

Les 200 parties sont terminées, **aucune erreur, aucune partie non
conclue, aucune contradiction** entre un résultat annoncé démontré et le
vainqueur final.

| Indicateur | Résultat |
|---|---:|
| Parties terminées | 200 / 200 |
| Victoires du candidat | 108 / 200 (54,0 %) |
| Intervalle bootstrap à 95 %, regroupé par placement (25 groupes, 5000 tirages) | [50,0 % ; 58,5 %] |
| Contradictions preuve/vainqueur | 0 |
| Erreurs / parties non conclues | 0 |

La borne basse de l'intervalle touche 50 % : **comme pour les campagnes
précédentes de ce projet (table de transposition, profils 1/3/4), le taux
de victoire seul ne démontre pas un gain de force.** Ce n'est pas la
mesure qui compte ici : ce changement est une optimisation pure (aucun
changement d'évaluation ni de stratégie), le critère d'acceptation est
l'**absence de régression à budget égal**, pas un gain de taux de
victoire. Sur ce plan, le résultat est positif : la borne basse de
l'intervalle ne descend pas sous 50 %, rien n'indique que le candidat
joue moins bien que l'original.

### Coût de recherche

| Camp | Décisions | Profondeurs | Butée (profondeur 6) atteinte | Temps moyen |
|---|---:|---|---:|---:|
| Candidat | 3 093 | {1:108, 2:121, 3:275, 4:523, 5:1118, 6:947, 0:1} | 947 | 19,09 s |
| Original | 3 087 | {1:92, 2:156, 3:256, 4:573, 5:1099, 6:908, 0:3} | 908 | 20,07 s |

Le candidat atteint la profondeur 6 complète plus souvent (947 contre 908
décisions, +4,3 %) et coûte en moyenne moins cher (19,09 s contre 20,07 s,
-4,9 %), sur l'ensemble hétérogène des 100 configurations — pas seulement
sur Hermès. C'est cohérent avec la réduction du branchement mesurée dans
`docs/profilage-hermes-20261009.md` : moins d'états à visiter, plus de
décisions qui finissent une profondeur de plus dans le même budget.

### Détail par confrontation

| Confrontation | Parties | Victoires du candidat |
|---|---:|---:|
| Aucun pouvoir / Aucun pouvoir | 24 | 12 |
| Hermès / Héphaïstos | 12 | 6 |
| Héphaïstos / Hermès | 12 | 8 |
| Apollon / Minotaure | 12 | 6 |
| Minotaure / Apollon | 12 | 5 |
| Artémis / Athéna | 12 | 7 |
| Athéna / Artémis | 12 | 7 |
| Atlas / Pan | 12 | 7 |
| Pan / Atlas | 12 | 7 |
| Déméter / Prométhée | 8 | 4 |
| Prométhée / Déméter | 8 | 4 |
| Apollon / Athéna | 8 | 4 |
| Athéna / Apollon | 8 | 4 |
| Artémis / Pan | 8 | 4 |
| Pan / Artémis | 8 | 4 |
| Atlas / Déméter | 8 | 5 |
| Déméter / Atlas | 8 | 4 |
| Minotaure / Prométhée | 8 | 6 |
| Prométhée / Minotaure | 8 | 4 |

Aucune confrontation ne montre un déséquilibre marqué en faveur de
l'original ; la confrontation impliquant directement Hermès
(Hermès/Héphaïstos + Héphaïstos/Hermès, 24 parties) donne 14/24 au
candidat, cohérent avec le reste de l'échantillon.

## Limites

- 25 groupes de placements corrélés (matchup x ouverture), pas 100
  placements indépendants : même limite déjà documentée pour les
  campagnes précédentes de ce projet.
- La machine était partagée avec une autre campagne pendant une partie de
  l'exécution ; la concurrence a été temporairement réduite (voir
  Protocole) sans perte de progression, mais les temps de décision moyens
  ci-dessus incluent une fenêtre à charge CPU variable. Les comparaisons
  de **nœuds** du rapport de profilage restent la mesure de référence
  pour le coût, étant déterministes et indépendantes de la charge machine ;
  les temps moyens ci-dessus sont un indicateur cohérent mais plus
  bruité.

## Conclusion

Aucune régression de force détectée à budget égal (intervalle bootstrap
ne descendant pas sous 50 %), aucune erreur, aucune contradiction de
preuve, et un coût de recherche légèrement inférieur sur l'ensemble du
corpus. Ce résultat valide, à l'échelle d'une campagne complète et pas
seulement d'un banc de profilage isolé, que la génération progressive et
la canonisation du swap des deux bâtisseurs ne dégradent ni la légalité,
ni l'évaluation, ni la qualité de jeu — avec un bénéfice de coût qui va
au-delà d'Hermès seul.

Ce lot ne doit cependant pas être présenté comme une preuve de gain de
force (le taux de victoire ne démontre rien à lui seul, comme pour les
lots précédents de ce projet) : c'est une validation de non-régression,
pas une campagne stratégique.

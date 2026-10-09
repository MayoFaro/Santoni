# Profilage : le clonage de ligne restant est négligeable

Date : 9 octobre 2026. Branche : `improve/search-profiling`, divergée de
`main` juste après la fusion de la table de transposition
([rapport](table-transposition-20261009.md)).

## Question posée

Ce rapport proposait, comme suite logique, de réduire le dernier clonage
restant dans `Search::minimax` : `Rc::new(line.clone())`, exécuté une fois
par nœud terminé (pas seulement à la lecture, déjà réglée par le partage
`Rc`). Avant d'entreprendre cette reconstruction de ligne principale sans
copie — un changement plus invasif, qui touche la façon dont le coup joué
est restitué — il fallait mesurer si ce coût restant est réellement
significatif.

## Méthode

Un compteur `clone_ns` sur `Search`, incrémenté uniquement derrière une
fonctionnalité Cargo `profile` (désactivée par défaut, donc **aucun effet
sur le binaire livré** : le champ existe mais n'est ni écrit ni lu en
dehors de cette fonctionnalité). Un test dédié
(`profile_pv_clone_cost`, feature-gated) appelle directement
`Search::minimax` sur le jeu sans pouvoir et sur Hermès, à plusieurs
profondeurs, et compare le temps cumulé de clonage au temps total.

```sh
cargo test --manifest-path native_engine/Cargo.toml --features profile \
  --release profile_pv_clone_cost -- --nocapture --test-threads 1
```

## Résultat

| Configuration | Profondeur | Nœuds | Temps total | Temps de clonage | Part |
|---|---:|---:|---:|---:|---:|
| Sans pouvoir | 3 | 7 269 | 266 ms | 0,18 ms | 0,1 % |
| Sans pouvoir | 4 | 20 608 | 2 432 ms | 1,63 ms | 0,1 % |
| Sans pouvoir | 5 | 320 646 | 8 816 ms | 5,53 ms | 0,1 % |
| Hermès | 2 | 8 515 | 1 942 ms | 1,69 ms | 0,1 % |
| Hermès | 3 | 2 418 735 (budget 30 s atteint) | — | — | — |

Le clonage reste à **0,1 % du temps de recherche**, de façon constante, y
compris sur Hermès. Le partage `Rc` mis en place dans la branche
précédente a donc déjà capté l'essentiel du gain accessible sur cet axe :
le clonage ponctuel restant, à l'insertion dans la table, n'est plus un
poste de coût mesurable.

**Conclusion : ne pas poursuivre la reconstruction de ligne sans copie.**
Le gain mesuré serait de l'ordre de 0,1 % dans le meilleur cas, pour un
changement qui touche un chemin critique pour la justesse (le coup
réellement restitué au joueur). Le rapport coût/risque est défavorable ;
ce sous-axe de « réduction des copies » est clos, avec preuve à l'appui
plutôt que par supposition.

## Ce que la mesure révèle à la place

La ligne Hermès est sans appel : **2 418 735 nœuds sans terminer la
profondeur 3 en 30 secondes**, contre 320 646 nœuds pour terminer la
profondeur 5 sans pouvoir. Ce n'est pas une mesure nouvelle en soi — le
rapport du 5 octobre notait déjà « 80 successeurs distincts sans pouvoir,
4 198 avec Hermès » sur un même placement — mais ce profilage le confirme
directement depuis le code de recherche réel, pas seulement par comptage
de coups générés.

Le classement coût/efficacité du 5 octobre plaçait déjà la **génération
progressive et le dédoublonnage plus tôt pour Hermès** au-dessus de la
réduction des copies (rendement « Bon après mesures », coût estimé 2-4
jours, risque « Ordre dégradé ou tours légaux oubliés »). Cette mesure le
confirme et referme la discussion sur l'ordre des priorités : la
prochaine amélioration de fond, si elle est engagée, porte sur la
génération et la déduplication pour Hermès (et les pouvoirs à plusieurs
déplacements apparentés), pas sur la mémoire du cache.

## Validation

8 tests Rust et 338 tests Python/Qt inchangés sans la fonctionnalité
`profile`. Le test de profilage lui-même n'est pas exécuté par la suite
par défaut (`cargo test` sans `--features profile`) ; il sert d'outil de
mesure à la demande, pas de test de non-régression.

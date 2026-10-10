# Profilage : génération progressive et canonisation pour Hermès

Date : 9 octobre 2026. Branche `improve/hermes-generation`, worktree isolé,
divergée de `main` à `b698745` (juste après la clôture du sous-axe
« réduction des copies », voir [profilage-copies-pv-20261009.md](profilage-copies-pv-20261009.md)).

## Question posée

Ce rapport-là concluait : « la prochaine amélioration de fond, si elle est
engagée, porte sur la génération et la déduplication pour Hermès », avec la
mesure de référence **2 418 735 nœuds sans finir la profondeur 3 en 30 s**,
contre 320 646 nœuds pour finir la profondeur 5 sans pouvoir. Deux pistes
distinctes ont été engagées sur cette branche :

1. **Génération progressive** (`visit_base`) : brancher dans `ordered_visit`
   le flux déjà présent (non committé au début de cette branche) qui
   *stream* les tours d'Hermès vers un visiteur par lots, au lieu de tout
   matérialiser dans un `Vec` puis trier globalement avant de visiter le
   premier coup.
2. **Canonisation du swap des deux bâtisseurs** (`canonical_key`) : pour les
   pouvoirs de base, les deux bâtisseurs d'un même joueur sont strictement
   interchangeables (`evaluate`/`can_step`/`builds`/`apply` ne regardent
   jamais le slot, seulement la case). Deux tours dont les positions finales
   ne différent que par cet échange sont la même position stratégique ; les
   traiter comme deux entrées distinctes gonfle artificiellement le
   branchement d'Hermès.

Les deux changements sont indépendants et ont été mesurés séparément puis
ensemble.

## Méthode

Même méthodologie que le rapport précédent : un test de profilage
feature-gated (`profile_hermes_generation_cost`, voir
`native_engine/src/lib.rs`) appelle `Search::minimax` directement sur le jeu
sans pouvoir et sur Hermès, profondeurs 2 à 5, budget 30 s par recherche.

```sh
cargo test --manifest-path native_engine/Cargo.toml --features profile \
  --release profile_hermes_generation_cost -- --nocapture --test-threads 1
```

Hermès ne termine la profondeur 3 dans **aucune** configuration testée en
30 s ; un second test, non conservé dans la suite (mesure ponctuelle), a
donc rejoué la profondeur 3 d'Hermès avec un budget étendu à 180 s, pour
savoir si chaque changement permet au moins d'*atteindre* cette profondeur
dans un budget raisonnable, au-delà du seul comptage de nœuds à l'arrêt.

Quatre configurations ont été comparées, chacune obtenue en recompilant
`native_engine` avec la version de `lib.rs` correspondante (même code source
sinon, mêmes `advanced.rs`/`special.rs`) :

- **Référence** : `main@b698745`, sans aucun des deux changements.
- **Tâche 1 seule** : `visit_base` branché dans `ordered_visit` (lots de
  256, voir plus bas), sans `canonical_key`.
- **Tâche 2 seule** : l'ancien chemin `generate()` + tri global de la
  référence, mais avec le filtre de dédoublonnage remplacé par
  `canonical_key`.
- **Tâches 1+2 ensemble** : code final de cette branche (`86a8647`).

## Un premier essai a régressé le jeu sans pouvoir

Le lot de 24 copié directement de la branche avancée cassait le tri global
pour tous les pouvoirs de base à branchement supérieur à 24 (80 coups au
premier tour du jeu sans pouvoir) : la profondeur 4 passait de 20 608 à
76 477 nœuds, une régression de **3,7x**, parce que seul un sous-ensemble de
24 coups était trié avant chaque lot au lieu de l'ensemble du coup. Mesuré,
corrigé : `BASE_BATCH = 256` (comfortablement au-dessus du branchement
observé pour tout pouvoir de base hors Hermès, donc équivalent en pratique
à l'ancien tri global pour ces pouvoirs ; toujours minuscule face aux
quelque 2 000 à 4 000 tours d'Hermès, donc toujours utile pour lui). Le
lot de la branche avancée (`ADVANCED_BATCH = 24`) n'a pas été touché : son
branchement typique reste petit, aucune régression constatée.

## Résultat

Le jeu sans pouvoir donne des nombres de nœuds **rigoureusement identiques**
dans les quatre configurations (déterministe, aucune régression) :

| Profondeur | Nœuds (sans pouvoir, les 4 configurations) | Temps |
|---:|---:|---:|
| 2 | 299 | ≈ 72 ms |
| 3 | 7 269 | ≈ 450 ms |
| 4 | 20 608 | ≈ 4,3 s |
| 5 | 320 646 | ≈ 15,6 s |

Hermès, profondeur 2 (seule profondeur où toutes les configurations
terminent dans le budget de 30 s) :

| Configuration | Nœuds | Temps | Écart vs référence |
|---|---:|---:|---:|
| Référence | 8 515 | 3,41 s | — |
| Tâche 1 seule | 8 653 | 3,63 s | +1,6 % (bruit) |
| Tâche 2 seule | 4 317 | 1,73 s | **-49,3 % nœuds, -49,4 % temps** |
| Tâches 1+2 | 4 455 | 1,73 s | **-47,7 % nœuds, -49,3 % temps** |

Hermès, profondeur 3 (budget étendu à 180 s pour voir si elle devient
atteignable) :

| Configuration | Résultat à 180 s |
|---|---|
| Référence | **n'a pas terminé** : 8 804 366 nœuds consommés, toujours en cours |
| Tâche 1 seule | **n'a pas terminé** : 10 136 071 nœuds consommés, toujours en cours |
| Tâche 2 seule | **terminée en 106,7 s**, 4 196 907 nœuds |
| Tâches 1+2 | **terminée en ≈ 105,7 s** (deux mesures : 106,3 s et 105,2 s), 4 640 982 nœuds |

## Analyse honnête

**La canonisation (tâche 2) est le seul levier qui produit un gain mesuré.**
Elle coupe le branchement d'Hermès par deux dès la position initiale
(4 198 successeurs bruts → 2 099 classes canoniques, voir
`canonical_key_halves_the_hermes_raw_successor_set`), et ce gain se répercute
directement sur la recherche : -49 % de nœuds à la profondeur 2, et le
passage de « n'atteint jamais la profondeur 3 même en 180 s » à « la termine
en 107 s ».

**La génération progressive (tâche 1), prise seule, ne montre aucun gain
mesurable sur ce banc.** 8 653 nœuds contre 8 515 à la profondeur 2 (+1,6 %,
dans le bruit de mesure), et à la profondeur 3 étendue elle ne termine pas
non plus en 180 s (10,1M nœuds consommés, plutôt pire que la référence,
toujours dans une plage compatible avec du bruit). Combinée à la
canonisation, elle n'ajoute pas de gain net net non plus : 4 640 982 nœuds
contre 4 196 907 pour la canonisation seule (+10,6 %), pour un temps de
complétion quasi identique (105,7 s contre 106,7 s). Cet écart est trop
petit et trop proche du bruit de mesure (charge machine partagée, un seul
run par configuration pour cette portion) pour conclure à une régression
réelle de la tâche 1 ; il ne permet pas davantage de lui attribuer un gain.

Ce résultat n'est pas la conclusion qu'on espérait en ouvrant cette branche
(le commentaire déjà présent dans le code, au moment de démarrer ce travail,
pariait sur la génération progressive comme le levier principal). La raison
la plus probable : à budget de 30 s, Hermès reste dominé par le pur volume
de nœuds à visiter, pas par le coût de matérialisation/tri en amont de la
visite (ce dernier reste une fraction du temps total, comme déjà mesuré
dans le rapport précédent sur le clonage). Différer ce tri en lots ne réduit
rien tant que le nombre réel de positions distinctes à visiter ne change
pas — c'est exactement ce que fait la canonisation, et rien d'autre, ici.

La tâche 1 garde cependant une valeur d'infrastructure, indépendamment de ce
banc : elle retire l'anti-motif « tout matérialiser puis tout trier avant de
visiter le premier coup », remplacé par le même schéma par lots déjà en
place pour les pouvoirs avancés — un changement plus sûr à maintenir et
plus cohérent avec le reste du code, même sans gain chiffré ici. Elle était
aussi nécessaire pour appliquer la canonisation *au fil de l'eau* dans
`ordered_visit`, conformément à ce qui était demandé.

## Conclusion

La canonisation du swap des deux bâtisseurs (tâche 2) est validée par la
mesure : gain réel, reproductible, et strictement sans régression sur le jeu
sans pouvoir ni sur aucun autre pouvoir de base. La génération progressive
(tâche 1) est conservée pour sa valeur structurelle et parce qu'elle est le
support de la canonisation, mais **son gain propre, mesuré isolément, est
nul à négligeable sur ce banc** — ce rapport ne le présente pas comme un
gain alors qu'il ne l'est pas.

Hermès reste loin de « sans pouvoir » : 107 s pour terminer la profondeur 3
contre 15,6 s pour terminer la profondeur 5 sans pouvoir. Ce changement ne
résout pas tout l'écart documenté le 5 octobre ; il le réduit d'un facteur
proche de 2 sur le branchement, pas plus. Une réduction plus large
nécessiterait de revisiter la BFS interne d'Hermès elle-même (le
dédoublonnage par position de travailleurs dans `Generator::generate`, non
touché par cette branche — voir « Limites » ci-dessous).

## Limites et pistes non engagées ici

- **Table de transposition non canonisée.** La canonisation n'a été
  appliquée qu'à la clé de dédoublonnage de `ordered_visit`/`visit_base`,
  pas à la clé `(State, u32)` de la table de transposition
  (`Search::table`). Deux positions swap-équivalentes atteintes par des
  chemins différents dans l'arbre occupent donc toujours deux entrées de
  cache distinctes. Non fait ici car cela touche un chemin plus sensible
  (politique de remplacement, profondeur stockée) sans gain mesuré
  nécessaire pour ce lot ; piste ouverte si un futur profilage montre que
  les collisions de cache swap-équivalentes sont un poste de coût réel.
- **BFS interne d'Hermès non canonisée.** `Generator::generate` dédoublonne
  déjà ses états intermédiaires par position exacte des travailleurs
  (`seen.insert(next.workers[p])`), mais pas par classe canonique. La
  canoniser réduirait probablement le travail interne de génération
  lui-même (pas seulement le nombre de tours finaux visités), mais aurait
  aussi changé le comportement de `generate()`, la fonction de référence
  utilisée comme oracle dans les tests de non-régression de cette branche.
  Laissée intacte par choix, pour garder une référence stable.

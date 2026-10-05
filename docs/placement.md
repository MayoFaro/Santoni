# Conseil de placement

En mode **Partie contre robot**, après avoir choisi les deux pouvoirs et le premier joueur, le groupe Placement
initial affiche l'ordre réglementaire de placement. Bia se place toujours en
premier, sans changer le joueur qui commence ensuite à jouer.

- **Vous en premier** : saisissez vos deux positions, puis demandez le conseil.
  Le moteur compare les paires légales restantes du robot contre vos positions.
- **Robot en premier** : le moteur ne lit aucune de vos positions futures, même
  si elles ont déjà été saisies. Il anticipe plusieurs réponses adverses légales.

**Suggérer le placement du robot** lance un calcul séparé avec le budget indiqué
(10 secondes par défaut). Le conseil apparaît dans Configuration et dans le
nouvel onglet **Placement** de la fenêtre Analyse, avec un aperçu du plateau.
La saisie reste disponible ; toute modification des positions, des pouvoirs,
du premier joueur ou du budget invalide le conseil précédent.

**Appliquer ces positions au robot** remplit uniquement ses deux champs. Il ne
crée aucune partie et ne change pas vos pions. Si votre placement saisi à l'avance
entre en conflit avec celui d'un robot qui se place en premier, l'interface vous
demande de corriger vos propres positions.

Le moteur respecte les bords opposés d'Eros et distingue les deux identités de
Selene : la pièce 2 est la bâtisseuse. Les positions sont évaluées avec les deux
pouvoirs et le bon joueur au trait au début de la partie.

## Recherche et limites

Le moteur commence par une comparaison géométrique afin de disposer rapidement
d'un placement légal. Il approfondit ensuite les ouvertures avec le moteur de jeu
Rust lorsque les deux pouvoirs sont des dieux de base, ou le moteur Python pour
les autres pouvoirs pris en charge.

Quand le robot est second, tous ses placements légaux sont candidats à
l'approfondissement. Quand il est premier, la comparaison géométrique utilise
un maximin avec élagage sur les placements adverses ; l'approfondissement de jeu
est limité à au plus 12 placements du robot et 12 réponses adverses pour chacun.
Seules les lignes entièrement évaluées alimentent le classement approfondi.

Le nombre réellement approfondi est affiché. Le conseil reste une **estimation
dans le budget**, pas une preuve du meilleur placement absolu. Un budget trop
court peut ne produire que le conseil géométrique ; ce cas est indiqué.

## Isolation de la QA

Le conseil de placement est intégré à `main`. Le worktree utilisé pendant son
développement n’est pas nécessaire pour lancer ou modifier cette fonctionnalité.

`preview_placement.sh` utilise `.preview-data/` et ajoute **Test placement** aux
titres des fenêtres. Ses parties et archives sont indépendantes des sauvegardes
de production. Pour conserver cette séparation, utilisez ce lanceur pour tester
une modification, plutôt que le raccourci de bureau.

Captures : [configuration](placement-configuration.png),
[analyse du placement](placement-analyse.png).

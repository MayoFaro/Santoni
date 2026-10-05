# Santoni

Application native Linux pour jouer à Santorini contre un moteur de recherche.
Deux fenêtres PySide6, **Saisie** et **Analyse**, restent au-dessus des autres
applications, comme dans Digitcode. Ctrl + molette règle leur opacité séparément.

## Lancer

Depuis ce dossier :

```bash
./start_native.sh
```

Le lanceur compile automatiquement le moteur Rust en mode optimisé si ses sources
ont changé. Il utilise `cargo` dans le PATH ou `~/.cargo/bin/cargo`. Python 3.10+
et PySide6 doivent être disponibles ; ils le sont sur la machine actuelle.
Le moteur Rust n'a aucune dépendance externe à télécharger.

## Configurer et jouer

1. Cochez les familles de pouvoirs et, dans la liste, les cartes réellement
   disponibles. Sans famille cochée, les deux joueurs jouent sans pouvoir.
2. Choisissez votre pouvoir. Pour le robot, choisissez une carte directement ou
   lancez **Comparer les pouvoirs disponibles**, puis **Retenir ce pouvoir** dans
   Analyse. On peut également demander un conseil pour son propre pouvoir.
3. Choisissez qui commence et les budgets de réflexion (5 secondes par tour par
   défaut, 15 secondes pour les pouvoirs).
4. Placez les quatre bâtisseurs : choisissez une pièce, puis cliquez sur une
   lettre A–E et un nombre 1–5. La pièce suivante est sélectionnée automatiquement.
   Eros impose les bords opposés ; Bia place ses pièces en premier. Pour Selene,
   la pièce numéro 2 représente la bâtisseuse.
   Vous pouvez également utiliser **Suggérer le placement du robot**, puis
   **Appliquer ces positions au robot**. Si vous vous placez en premier, saisissez
   d'abord vos deux pions. Si le robot se place en premier, demandez son placement
   avant de choisir le vôtre. Bia détermine l'ordre du placement lorsqu'il est présent.
5. Démarrez. À votre tour, choisissez le bâtisseur puis saisissez la destination
   et la construction, toujours lettre puis nombre. Seules les actions qui
   peuvent appartenir à un tour complet légal sont proposées.
6. Pour les pouvoirs facultatifs, continuez les actions proposées ou choisissez
   **Valider le tour / terminer**. Un héros s'active explicitement avant la saisie
   du tour ; ses effets apparaissent à leur moment réglementaire.
7. Après validation, le robot analyse la nouvelle position et propose un **tour
   complet**, avec l'utilisation ou la conservation du pouvoir. À la fin du
   budget, cliquez sur **Jouer ce tour complet** pour l'enregistrer. Le tour n'est
   pas joué automatiquement : vous pouvez d'abord le reproduire sur votre plateau.

La saisie reste utilisable pendant la recherche. Les anciennes recherches sont
annulées après un changement de position et leurs résultats sont ignorés.
**Annuler l'action** corrige la saisie en cours ; **Annuler le dernier tour**
revient à la position précédente. À votre tour, **Recalculer** demande un conseil
au moteur ; la recherche automatique concerne les tours du robot.

## Pouvoirs disponibles dans cette version

- Les **10 dieux de base**, avec règles et recherche Rust.
- Les **10 héros**, avec le moteur de référence Python dans un processus séparé.
- **18 dieux avancés** : Aphrodite, Ares, Bia, Charon, Chronus, Eros, Hera,
  Hestia, Hypnus, Limus, Medusa, Persephone, Poseidon, Selene, Triton, Zeus,
  Hades et Urania, avec le moteur de référence.
- Les **17 autres dieux avancés** sont visibles et grisés. Leurs règles ne sont
  pas simulées : ils restent à implémenter, notamment les pouvoirs avec jetons,
  tirages, vol de pouvoir, tours supplémentaires et informations secrètes.

La référence est [regle.pdf](regle.pdf), livret français de six pages ajouté au
projet. Les associations N.R.C. connues du livret sont indiquées dans la
configuration, et les joueurs choisissent des pouvoirs distincts.

## Force du moteur et délai

Le moteur utilise minimax, alpha-bêta, approfondissement progressif, classement
des coups et cache des positions. Un premier tour légal sert de secours ; chaque
profondeur achevée peut améliorer le conseil. Les variantes avec beaucoup
d'actions facultatives peuvent atteindre une profondeur plus faible, surtout
dans le moteur Python.

Le budget par tour est de **1 à 600 secondes** dans l'application. Le temps de
création du processus est soustrait du budget. Un résultat complet est envoyé
à l'échéance, ou plus tôt lorsqu'une victoire/défaite forcée est démontrée.
L'affichage et la transmission ajoutent une petite latence. Une impossibilité de
trouver un premier tour dans le délai apparaît explicitement ; elle n'est jamais
présentée comme une défaite.

**« Estimation » ne garantit pas un coup optimal.** « Démontré » signifie que
la recherche a établi une victoire ou une défaite forcée avec les règles
implémentées. La force n'a pas encore été étalonnée contre des joueurs experts.

Le choix initial du pouvoir compare des recherches de même budget sur trois
placements de référence, contre le pouvoir adverse choisi. Si celui-ci est
inconnu, il retient le pire score contre les pouvoirs disponibles testés. Seules
les comparaisons terminées pour tous les candidats alimentent le classement.
Il s'agit d'une **estimation d'ouverture**, pas d'une preuve de supériorité du
pouvoir, et les placements réels peuvent changer le conseil.

## Sauvegardes

Chaque tour validé, annulation et résultat est enregistré atomiquement dans
`~/.local/share/santoni/en_cours.json`. L'application recharge la partie au
démarrage. Une fermeture pendant la réflexion conserve la dernière position
validée ; les actions encore en cours de saisie ne sont pas enregistrées.

**Archiver et réinitialiser** écrit d'abord une archive JSON dans
`~/.local/share/santoni/parties/`, comprenant les pouvoirs, positions initiales,
actions, états intermédiaires et résultat. Une partie sans résultat est archivée
comme interrompue. Si l'archive échoue, la réinitialisation est annulée.

Les victoires sont détectées automatiquement. **Enregistrer le résultat** permet
également de déclarer un abandon, un résultat observé ou une interruption.

Le chemin suit `XDG_DATA_HOME`. `SANTONI_DATA_DIR` permet de choisir un dossier
distinct, utile pour les tests ou plusieurs installations.

## Vérifier

```bash
cargo test --manifest-path native_engine/Cargo.toml
cargo build --release --manifest-path native_engine/Cargo.toml
QT_QPA_PLATFORM=offscreen python3 -m pytest -q
```

Les tests couvrent les règles, les tours complets, les délais, les sauvegardes,
l'interface et les résultats obsolètes. Ils comparent aussi les actions et
positions du moteur Rust avec celles du moteur Python sur des positions
ordinaires et générées, pour les dix dieux de base.

## Architecture et suite

- `santorini/engine.py` : règles pures, états immuables, actions et validation.
- `native_engine/` : moteur Rust, chargé uniquement dans un processus de calcul.
- `santorini/search.py` : moteur de référence et comparaison des pouvoirs.
- `santorini/worker.py` : processus annulables, résultats identifiés par position.
- `santorini/ui.py` : fenêtres natives, coordonnées et contrôleur.
- `santorini/storage.py` : sauvegardes atomiques et archives.

Les étapes restantes sont décrites dans [docs/plan.md](docs/plan.md).

Captures de l'application : [configuration](docs/configuration.png),
[saisie](docs/saisie.png), [analyse](docs/analyse.png).

Pendant une partie, l'onglet **Historique** permet de sélectionner un tour puis
**Corriger le tour sélectionné**. Saisissez de nouveau le tour complet dans
l'onglet Partie et cliquez sur **Enregistrer le correctif**. Le plateau est
reconstruit et les tours suivants sont rejoués avec leurs actions enregistrées.
Si un tour suivant devient illégal, son numéro est indiqué et la sauvegarde
reste inchangée. **Annuler la correction** retrouve la partie d'origine.
La réflexion reprend sur le plateau corrigé après son enregistrement.

La case **Partie contre robot** est décochée par défaut. En mode adversaire
humain, les conseils sont calculés automatiquement pour vos tours seulement ;
les tours adverses sont saisis sans conseil. Ce mode est sauvegardé avec la
partie. Les anciennes sauvegardes conservent le mode robot.
Cocher une famille de pouvoirs affiche ses cartes sans les sélectionner :
cochez individuellement les cartes disponibles pour constituer le choix.


## Reprendre le travail sur un autre PC

Tout le travail de l’ancien worktree de placement est intégré à la racine du
dépôt. Un clone de `main` contient l’application, les outils de parties sans
interface, les 100 parties de la première campagne et les documents d’analyse.
Le dossier `.worktrees/` est une organisation locale de Git ; il n’est pas
nécessaire de le copier ou de le recréer pour continuer.

Sur Linux, avec Python 3.10+ et Rust/Cargo installés :

```bash
git clone git@github.com:MayoFaro/Santoni.git
cd Santoni
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e . pytest
cargo build --release --manifest-path native_engine/Cargo.toml
./start_native.sh
```

Pour une prévisualisation avec des sauvegardes distinctes dans `.preview-data/` :

```bash
./preview_placement.sh
```

Les binaires Rust, environnements Python et caches sont reconstruits sur le
nouveau PC. Le fichier `Santoni.desktop` contient le chemin de l’installation
initiale : adapter ses champs `Exec` et `Path` avant de l’utiliser comme raccourci
sur PC2. Les parties personnelles de l’application restent dans le dossier
local de sauvegarde indiqué plus haut ; les parties expérimentales sont dans
le dépôt.

## Campagne robot contre robot et pistes d’amélioration

- [Rapport des 100 parties à 5 secondes par tour](experiments/selfplay-100-5s-20261005/rapport.md).
- [Axes d’amélioration et estimation coût / efficacité](docs/ameliorations-moteur-20261005.md).
- [Positions critiques et variantes vérifiées](experiments/selfplay-100-5s-20261005/critical-positions.json).
- `experiments/selfplay-100-5s-20261005/` conserve les 100 historiques, les mesures,
  les réanalyses et la reprise qui inverse le vainqueur de la partie 16.

Pour lancer une nouvelle campagne de 100 parties sans interface, utiliser un
nouveau dossier de sortie et adapter le nombre de processus au PC :

```bash
python tools/selfplay.py --output experiments/selfplay-pc2 --seconds 5 --workers 4 --seed 20261005
python tools/analyse_selfplay.py experiments/selfplay-pc2
```

Le banc de parties utilise le moteur Rust compilé et les règles Python pour
valider les coups. Il fonctionne sans PySide6 ni fenêtre. Les temps et les coups
peuvent varier entre machines malgré une même graine : le budget est temporel.

Une campagne interrompue peut être relancée avec les mêmes paramètres et la
même bibliothèque : les parties terminées sont conservées, les parties
incomplètes repartent de leur placement initial. Les paramètres historiques et
l’empreinte du moteur sont préservés pour éviter de mélanger plusieurs versions.

Pour approfondir une position de la campagne initiale :

```bash
python tools/recheck_selfplay.py experiments/selfplay-100-5s-20261005 --cases 16:26 --seconds 20
```

`tools/report_selfplay.py` met en forme le rapport de la **première campagne** ;
ses commentaires sur les parties 16 et 18 sont spécifiques à ces données.
Utiliser `tools/analyse_selfplay.py` pour les mesures d’une nouvelle campagne.

### Deuxième campagne : 35 secondes contre 5 secondes

Le moteur initial à 35 s gagne **144 des 200 parties (72 %)** contre le même
moteur à 5 s, avec inversion des budgets pour chaque configuration initiale.
Voir le [bilan et les conclusions](docs/bilan-moteur-35s-vs-5s-20261005.md),
le [protocole](docs/campagne-35s-vs-5s.md) et les
[résultats détaillés](experiments/baseline-35s-vs-5s-20261005/rapport.md).
La [référence figée](experiments/references/initial-cb732bb/README.md) conserve
les sources et le binaire recompilé pour les futures comparaisons.

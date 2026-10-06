# Tous les pouvoirs dans le moteur Rust

## Périmètre et référence

La branche `feature/all-powers-rust` ajoute les 17 pouvoirs manquants et porte
les règles des pouvoirs déjà disponibles en Rust. Les 55 cartes sont sélectionnables.
L’interface native Qt, le contrôleur et les sauvegardes restent en Python. Les
calculs de règles et la recherche tournent dans des processus annulables ;
l’application de `main` n’a pas été remplacée pendant le développement.

La référence est le [livret français fourni](../regle.pdf), édition originale.
Le profil Arène ouvre les familles dieux de base, avancés et héros, sans Toison
d’or, sans hasard ni information cachée. Il propose 51 pouvoirs : Chaos, Hecate,
Moerae et Tartarus sont exclus. Les 55 pouvoirs restent disponibles dans le
mode personnalisé. Il n’est pas une synchronisation avec toutes les règles actuelles de BGA.
Par exemple, Charybde suit ici la victoire par montée normale avant la téléportation,
et la téléportation forcée ne déclenche pas une victoire par montée. Des révisions
ultérieures de cette carte sur BGA ont changé ces conditions.
Les avertissements d’associations N.R.C. du livret restent affichés.

## Règles ajoutées

| Pouvoir | État et comportement |
| --- | --- |
| Chaos | Carte de dieu de base active, pioche et défausse ; nouvelle carte après un dôme effectivement construit. |
| Circé | Vol temporaire lorsque les bâtisseurs adverses ne sont pas voisins ; restitution au début du prochain tour de Circé et réévaluation. |
| Dionysos | Choix de tours supplémentaires avec des bâtisseurs adverses après une tour complète ; aucune victoire pendant ces tours. |
| Morphée | Un bloc en réserve au début du tour ; constructions facultatives et stock conservé. |
| Éole | Direction initiale puis changement en fin de tour ; mouvements interdits contre le vent. |
| Charybde | Deux tourbillons ; téléportation forcée quand les conditions du livret sont réunies. |
| Clio | Pièces sur les trois premières constructions ; cases protégées de l’adversaire, retrait lors d’une nouvelle construction. |
| Europe et Talos | Jeton Talos déplacé en fin de tour ; obstruction pour les deux joueurs. |
| Gaia | Deux bâtisseurs en réserve ; intervention immédiate après un dôme, puis reprise du tour interrompu. |
| Grées | Trois bâtisseurs et construction par n’importe lequel après le déplacement. |
| Harpies | Déplacement forcé suivant la direction du déplacement adverse, jusqu’à un obstacle ou une montée. |
| Hécate | Positions cachées ; une action illégale à cause d’un bâtisseur caché annule l’action et termine le tour. |
| Moires | Trois bâtisseurs et zone secrète de 2 × 2 ; inversion du vainqueur dans la zone prévue par le livret. |
| Némésis | Échange facultatif du maximum de bâtisseurs possible lorsque la condition de voisinage est satisfaite. |
| Sirène | Direction choisie à la mise en place ; déplacement forcé d’un sous-ensemble non vide de bâtisseurs adverses. |
| Tartare | Case secrète d’abîme ; le propriétaire du bâtisseur qui y entre perd. |
| Terpsichore | Déplacement de tous les bâtisseurs, puis construction par tous ; arrêt immédiat lors d’une victoire. |

Les interactions portent aussi sur les règles existantes : protections de Clio,
montées obligatoires de Perséphone, vent et déplacement circulaire d’Urania,
interventions de Gaia pendant une activation de héros, victoire de Chronos après
la fin des tours supplémentaires de Dionysos, déplacements forcés vers Tartare.

## Configuration, saisie et sauvegarde

Le placement accepte trois bâtisseurs initiaux pour les Grées et les Moires,
et jusqu’à quatre en cours de partie pour Gaia. La configuration demande les
directions d’Éole et de Sirène, la première carte réellement tirée de Chaos,
et les choix secrets requis. Le robot choisit sa propre zone des Moires ou son
abîme par une recherche sous le budget de placement ; ces choix restent cachés.
Pour Hécate robot, le conseil de placement fournit des positions gardées privées.

Les cases publiques restent saisies avec les chips A–E et 1–5. Les tirages de
Chaos et les identités de bâtisseurs secrets à échanger ont des boutons nommés.
Les actions sur un bâtisseur caché ne proposent pas sa position comme cible
connue. Lorsqu’un adversaire humain joue Hécate, sa saisie attend l’activation de
sa vue privée ; il faut lui passer l’écran. Les journaux publics masquent ses
mouvements, mais montrent ses constructions publiques.

Gaia implique une véritable interruption : le moteur propose le début du tour,
puis Gaia choisit d’intervenir ou de renoncer, puis le joueur reprend les actions
restantes. Présenter une construction suivante comme certaine avant cette réponse
serait incorrect. Dionysos dispose de choix explicites de continuation/arrêt.
Chaos demande le résultat réel du tirage, plutôt que laisser le solveur choisir
une carte favorable.

L’état sauvegardé comprend les stocks, jetons, cartes, pouvoirs volés, objectifs
secrets et contexte de reprise d’un tour interrompu. Les anciennes sauvegardes
sans ces champs gardent leurs valeurs par défaut. La bibliothèque native expose
une version d’ABI et sa taille d’état ; une ancienne bibliothèque incompatible
est rejetée avant de transmettre les structures.

**Les archives complètes contiennent les secrets**, pour permettre la reprise
et le rejeu. Leur masquage concerne l’affichage courant, pas le fichier JSON ni
un dispositif de jeu en réseau avec contrôle d’accès.

## Recherche : limites explicites

Pour les positions publiques déterministes, le moteur Rust utilise les règles
complètes avec approfondissement progressif, alpha-bêta et cache. Génération par
préfixes et lots évite de matérialiser les nombreuses constructions de Morphée
ou d’Héraclès. Les actions légales et les résultats restent indépendants du
thread d’interface.

Chaos est traité comme un événement aléatoire : les valeurs sont moyennées sur
les cartes restantes. La comparaison initiale de ce pouvoir moyenne également
les premières cartes possibles. Une suite après un tirage n’est pas présentée
comme certaine avant de connaître la carte réelle.

Face à Hécate, aux Moires ou à Tartare, la recherche construit des positions
possibles à partir des données publiques, sans lire la position ou l’objectif
secret réel. Pour limiter son coût, elle utilise au maximum 16 hypothèses et
évalue les plans avec une réponse adverse. Le résultat est une **estimation** :
il ne produit pas de preuve de victoire certaine ni de profondeur complète.
Pour Tartare, les cases déjà traversées légalement excluent des abîmes possibles.
Le suivi des hypothèses n’exploite pas encore tout l’historique des constructions
d’Hécate : c’est un axe de travail supplémentaire.

Les actions de Sirène, Adonis, Dionysos, Némésis, Thésée, Médée, Ulysse et Charon
ont une saisie adaptée à une cible cachée. Un essai rendu impossible par la
présence secrète est arbitré par annulation et fin du tour. Certains croisements
N.R.C. ou ambigus ne disposent pas d’un arbitrage explicite dans le livret ;
cette convention doit être revue si l’on vise la compatibilité précise avec une
plateforme particulière. Le support des cartes ne constitue donc pas une preuve
exhaustive de toutes les associations possibles.

Le choix d’un objectif secret du robot compare des réponses adverses pleinement
informées, une hypothèse conservatrice, sur les profondeurs entièrement achevées
pour toutes les options comparées. Il ne prouve pas une politique optimale de
jeu à information imparfaite.

Cette livraison étend les règles disponibles. **Elle ne démontre pas un gain de
taux de victoire** : cela nécessite une nouvelle campagne stratégique contrôlée.

## Essayer cette branche sans toucher à la partie en cours

Depuis le worktree `.worktrees/all-powers-rust` :

```bash
./preview_all_powers.sh
```

Ce lanceur utilise un dossier de sauvegarde propre au test. Il ne redémarre pas
l’application de `main` et ne recharge pas sa partie.

## Validation reproductible

```bash
cargo test --manifest-path native_engine/Cargo.toml
cargo build --release --manifest-path native_engine/Cargo.toml
QT_QPA_PLATFORM=offscreen python3 -m pytest -q
python3 tools/check_all_powers.py --output experiments/all-powers-rules-local --opponents 0 5 26 35 39 46 50 52 53 55
```

Les 38 pouvoirs précédents sont comparés à la référence Python sur des plateaux
contraints et des préfixes de saisie. Les 17 nouveaux pouvoirs ont des scénarios
manuels dédiés : actions légales, déclenchements, victoire, annulation, reprise,
configuration et sauvegarde. La validation native de ces nouveaux états n’est
pas un oracle indépendant ; les scénarios fixent les résultats attendus.

La campagne de règles dans `experiments/all-powers-rules-20261006` joue des actions
légales aléatoires et recharge chaque archive pour vérifier le rejeu. Elle ne
mesure pas la qualité stratégique. Son résumé conserve la graine et l’empreinte
de la bibliothèque utilisée.

Le benchmark `experiments/rust-public-port-20261005` concerne un état intermédiaire
du portage, antérieur aux états étendus des 17 cartes. Ses débits ne représentent
pas une mesure du moteur final de cette branche.

Résultat de la vérification du 6 octobre 2026 : **338 tests Python/Qt** et
**3 tests Rust** réussis. La campagne de règles comporte **168 parties** et
**4 942 tours enregistrés**, avec **0 erreur de rejeu**. Les parties aléatoires
ne constituent pas une campagne de comparaison stratégique.

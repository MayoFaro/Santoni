# Plan de réalisation et état

## Livré

- Application native à deux fenêtres toujours au-dessus et opacité réglable.
- Saisie par deux rangées A–E et 1–5 ; placement initial, premier joueur.
- Tour complet guidé, actions facultatives et activation unique des héros.
- Budgets distincts pour le prochain tour et le choix du pouvoir.
- Recherche adversariale annulable dans un processus distinct de Qt.
- Moteur Rust pour les règles sans pouvoir et les dix dieux de base.
- Moteur Python de référence : dix héros et dix-huit dieux avancés publics.
- Vérification indépendante des tours proposés par Rust.
- Sauvegarde après chaque modification validée, reprise, archive avec résultat.
- Conseil de placement sur la branche `feature/placement-advice` : réponse au
  placement connu ou anticipation des réponses adverses, budget distinct, Bia,
  Eros et identité des pièces de Selene. Voir [placement.md](placement.md).

## Extension du moteur

1. Porter les dix héros et les dix-huit dieux avancés déjà pris en charge en Rust.
   Chaque portage doit passer une comparaison systématique des tours et résultats
   avec le moteur de référence. Ajouter des tests spécifiques des interactions,
   pas seulement des tests des pouvoirs isolés.
2. Ajouter les états de jetons pour Aeolus, Charybdis, Clio, Europa & Talus.
3. Ajouter Graeae, Gaea, Nemesis, Siren et Terpsichore : nombres de pièces,
   actions multiples, contrôle de pièces adverses et ordre des actions.
4. Ajouter Circe, Dionysus, Morpheus et Harpies : effets persistants, pouvoir
   temporaire, tours supplémentaires, matériaux accumulés et déplacements forcés.
5. Ajouter Chaos avec un modèle de pioche, la saisie des cartes révélées et une
   recherche intégrant les événements aléatoires.
6. Ajouter Hecate, Moerae et Tartarus avec une distinction stricte entre l'état
   réel et les observations de chaque joueur, puis une recherche sur les
   hypothèses compatibles. L'interface du robot ne doit pas recevoir les secrets
   adverses. Les archives devront conserver les informations de manière à
   permettre la relecture sans les dévoiler durant la partie.

La variante Toison d'Or et les parties à trois/quatre joueurs sont hors de la
première application d'entraînement à deux joueurs.

## Renforcer et mesurer la stratégie

- Construire une collection de positions tactiques avec solutions vérifiées.
- Faire jouer les versions du moteur entre elles, à budget identique, en
  échangeant côtés, pouvoirs et placements. Mesurer les résultats avant
  d'affirmer un gain de force.
- Affiner l'évaluation des menaces doubles, de l'accès aux tours, de la mobilité,
  des ressources des héros et des objectifs propres aux pouvoirs.
- Enrichir la comparaison des pouvoirs par des ouvertures variées et des parties
  de référence. Son score actuel est une estimation issue de trois placements.
- Ajouter une revue de partie qui compare les coups joués aux alternatives
  analysées, avec les réponses adverses et les limites de recherche.
- Envisager une résolution exacte de fins de partie suffisamment petites, avec
  affichage distinct des preuves et des évaluations.

Une recherche limitée dans le temps ne permet pas de promettre un coup optimal
pour toute position. La priorité reste un tour complet légal à l'échéance,
puis une force mesurée et une explication honnête du niveau de certitude.

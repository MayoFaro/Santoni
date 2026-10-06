# Mode Arena et saisie par boutons

La configuration propose « Mode Arena », décoché par défaut. Il ouvre les
familles dieux de base, dieux avancés et héros, sans sélectionner les cartes.
Les joueurs choisissent leurs cartes puis leur pouvoir individuel ; il n'y a
pas de Toison d'or. Un pouvoir est requis pour chaque joueur en mode Arena.
L'option partie contre robot reste indépendante, décochée par défaut.

Le profil est enregistré dans les paramètres de la session et restauré à la
reprise. Les anciennes sauvegardes continuent à utiliser le profil personnalisé.
Sortir du profil rétablit les familles auparavant choisies.

Le mode Arène à information parfaite propose 51 pouvoirs : tous sauf Chaos
(tirages aléatoires), Hecate (bâtisseurs cachés), Moerae (zone secrète) et
Tartarus (abîme secret). Ces quatre cartes restent disponibles en mode
personnalisé, qui propose les 55 pouvoirs. Le profil utilise le livret
fourni au projet ; il n’intègre pas automatiquement les révisions propres à
Board Game Arena. Les pouvoirs à information secrète sont décrits avec leurs
limites dans [all-powers-rust.md](all-powers-rust.md).

Pendant la partie, les menus d'action et de bâtisseur sont remplacés par des
boutons exclusifs. Les deux icônes principales sont déplacement et construction ;
les autres actions légales des pouvoirs gardent un bouton distinct avec une
infobulle. Chaque chip de bâtisseur indique sa case courante, y compris après
un déplacement saisi mais pas encore validé. Les choix illégaux sont désactivés.
Un troisième ou quatrième bâtisseur, ou des cibles adverses restent représentables lorsque
les règles du pouvoir l'exigent. Les boutons conservent les noms accessibles
et la navigation clavier native de Qt.

Le calcul des actions légales et la réflexion du solveur restent asynchrones.
La recherche et les règles des 55 pouvoirs sont exécutées en Rust ; les contrôles
et les sauvegardes restent en Python. Les choix de cartes de Chaos et les cibles
secrètes utilisent des boutons nommés, sans coordonnées cachées dans les choix.

Le développement des pouvoirs est isolé sur `feature/all-powers-rust`, sans
remplacement ni redémarrage de l’application active.

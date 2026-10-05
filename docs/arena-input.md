# Mode Arena et saisie par boutons

La configuration propose « Mode Arena », décoché par défaut. Il ouvre les
familles dieux de base, dieux avancés et héros, sans sélectionner les cartes.
Les joueurs choisissent leurs cartes puis leur pouvoir individuel ; il n'y a
pas de Toison d'or. Un pouvoir est requis pour chaque joueur en mode Arena.
L'option partie contre robot reste indépendante, décochée par défaut.

Le profil est enregistré dans les paramètres de la session et restauré à la
reprise. Les anciennes sauvegardes continuent à utiliser le profil personnalisé.
Sortir du profil rétablit les familles auparavant choisies.

Arena utilise les règles déjà implémentées. Les pouvoirs non pris en charge
restent grisés : ce profil n'ajoute notamment pas la gestion d'informations
secrètes des dieux concernés ni toutes les révisions propres à Board Game Arena.

Pendant la partie, les menus d'action et de bâtisseur sont remplacés par des
boutons exclusifs. Les deux icônes principales sont déplacement et construction ;
les autres actions légales des pouvoirs gardent un bouton distinct avec une
infobulle. Chaque chip de bâtisseur indique sa case courante, y compris après
un déplacement saisi mais pas encore validé. Les choix illégaux sont désactivés.
Un troisième bâtisseur ou des cibles adverses restent représentables lorsque
les règles du pouvoir l'exigent. Les boutons conservent les noms accessibles
et la navigation clavier native de Qt.

Le calcul des actions légales et la réflexion du solveur restent asynchrones.
Le moteur stratégique ne change pas. Développement isolé sur
`feature/arena-input`, sans remplacement ni redémarrage de l'application active.

Validation : 153 tests Python passent, dont les contrôles de sélection exclusive,
de positions actualisées, de Jason et de sauvegarde du profil Arena.

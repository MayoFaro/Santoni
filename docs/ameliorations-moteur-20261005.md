# Améliorer le moteur après la première campagne

Date : 5 octobre 2026. Version de référence : `cb732bb`.

**Recommandation : commencer par une évaluation plus réaliste des constructions et un meilleur ordre d’exploration des coups, puis optimiser la mémoire et la génération.** Approfondir sélectivement les menaces vient ensuite, avec des garanties explicites sur les résultats annoncés comme démontrés. L’objectif reste de choisir un meilleur tour complet dans **5 secondes**, pas de maximiser la profondeur affichée.

Ce document présente un plan et des estimations de coût. Aucune des améliorations proposées n’a encore été mesurée contre la version de référence.

## 1. Ce que la campagne permet d’affirmer

La campagne a joué **100 parties, 2 972 tours et 2 974 recherches**. Les deux camps utilisaient le même moteur, avec un budget maximal de 5 secondes par tour. Les 25 placements étaient déclinés en quatre variantes de premier joueur et d’attribution des pouvoirs ; les variantes sans pouvoirs comprenaient une rotation du plateau.

| Observation | Conséquence pour le travail à venir |
|---|---|
| 2 304 recherches sur 2 974 atteignent le budget, soit 77,5 %. | Les économies de calcul peuvent améliorer les décisions, même sans changer l’évaluation. |
| Profondeur médiane 5 ; 6 sans pouvoirs ; 4 pour Hermès contre Héphaïstos. | Le coût dépend fortement des pouvoirs et du stade de la partie. |
| Sur un placement identique, 80 successeurs distincts sans pouvoirs, 4 198 avec Hermès. | Hermès est une priorité pour la génération et la réduction des équivalences. Ce rapport de 52 ne se traduit pas automatiquement en un gain d’optimisation de 52. |
| Partie 16, tour 26 : une construction différente transforme une défaite en victoire, confirmée par la reprise de la partie. | La décision de construction peut être améliorée ; ce cas fournit une position de référence concrète. |
| Partie 18 : profondeur 5 terminée en 1,2 s, profondeur 6 en 34,4 s. | Un niveau supplémentaire peut coûter beaucoup plus cher que tous les précédents. Augmenter le budget à 20 s n’assure pas une profondeur supérieure. |
| Dans 22 parties, le perdant avait encore une estimation positive à son tour précédent sa première défaite annoncée comme démontrée. | Les scores peuvent être optimistes à horizon limité. Ce nombre n’établit pas 22 erreurs évitables : certaines positions pouvaient être déjà perdues. |
| Aucun historique invalide ni contradiction observée entre un résultat annoncé comme démontré et le vainqueur final. | Préserver cette fiabilité pendant les changements ; l’audit ne démontre pas l’absence de bugs partagés par les règles Python et Rust. |

Les scores +430 ou +414 sont des valeurs heuristiques, **pas des probabilités de victoire**. La campagne oppose le moteur à lui-même : elle fournit des diagnostics, pas une preuve de progrès, ni un classement universel des pouvoirs. Les groupes de 8 ou 12 parties ne contiennent que deux ou trois placements distincts.

Sources locales : [rapport de campagne](../experiments/selfplay-100-5s-20261005/rapport.md), [audit détaillé](../experiments/selfplay-100-5s-20261005/analysis.json), [positions critiques](../experiments/selfplay-100-5s-20261005/critical-positions.json), [mesure des embranchements](../experiments/selfplay-100-5s-20261005/branching-controlled.json).

## 2. Ce que le moteur fait déjà

Le moteur Rust utilise l’approfondissement progressif, alpha-bêta, une table de transpositions, le dédoublonnage des positions finales et un tri des coups par évaluation. Le meilleur coup précédemment trouvé est exploré en premier lorsqu’il est disponible. À expiration du délai, la dernière profondeur terminée est conservée.

L’évaluation repose principalement sur la hauteur des bâtisseurs, la centralité, les déplacements adjacents possibles, les montées vers un niveau 3 et les descentes gagnantes de Pan. Elle ne représente pas explicitement la valeur des rampes, les routes à plusieurs déplacements ou toutes les possibilités de déplacement propres aux pouvoirs.

Plusieurs coûts sont visibles dans le code, mais doivent encore être profilés :

- `Search::moves` génère tous les tours, puis dédoublonne et trie, avant d’explorer le premier enfant. Une coupure alpha-bêta économise l’exploration des enfants suivants, mais pas leur génération déjà effectuée.
- Les entrées de transposition stockent une variante complète. Chaque variante contient des tours et des listes d’actions ; la recherche clone des entrées et des variantes.
- La table est plafonnée à 50 000 entrées. Une fois pleine, la condition actuelle empêche toute insertion, y compris la mise à jour d’une clé existante avec une recherche plus profonde ; il n’existe pas de politique de remplacement favorisant les recherches utiles.
- La clé de cache comprend la position et la distance à la racine. Cela distingue une même position atteinte à des distances différentes, notamment pour les scores de victoire. Retirer cette distance sans normalisation serait incorrect.
- Le moteur Python trie déjà ses coups par lots. Une optimisation de génération pour Rust doit être conçue autour de son fonctionnement actuel, sans supposer que les deux moteurs ont la même structure.

Il ne s’agit donc pas d’ajouter alpha-bêta ou un cache, mais d’améliorer leur rendement et la qualité des informations qu’ils exploitent.

Références : [`Search::moves`, `minimax`, table et évaluation](../native_engine/src/lib.rs), [recherche Python](../santorini/search.py), [validation indépendante des tours Rust](../santorini/native.py).

## 3. Classement coût / efficacité

Les coûts ci-dessous sont des **ordres de grandeur en jours de travail de développement et de validation**, hors longues campagnes de calcul. Ils ne sont pas des engagements de calendrier. Le rendement est qualitatif : aucun pourcentage de gain de vitesse ou de victoires n’est démontré à ce stade.

Le coût total ne s’obtient pas en additionnant toutes les lignes : certaines approches sont alternatives ou partagent les mêmes changements.

| Axe | Coût estimatif | Gain probable à 5 s/tour | Rendement attendu | Confiance dans le diagnostic | Risque principal |
|---|---:|---|---|---|---|
| Mesures détaillées et corpus critique | 0,5–1 j | Indirect, indispensable pour choisir les bons changements | Très bon | Élevée | Mesures qui ralentissent la recherche |
| Évaluation des rampes, accès et enfermement | 1–2 j | Potentiellement important sur les erreurs de construction | Très bon | Élevée pour le problème, moyenne pour le correctif | Pénaliser des constructions tactiquement bonnes |
| Ordre tactique des coups et historique des coupures | 1–2 j | Potentiellement important sur les positions avec menaces | Très bon | Moyenne à élevée | Détecteurs coûteux ou incomplets |
| Réduire les copies de variantes dans le cache | 1–2 j | Gain de débit possible sur beaucoup de positions | Bon, à confirmer par profilage | Moyenne | Variante restituée incorrectement |
| Remplacement des entrées de transposition | 0,5–1,5 j | Gain possible sur les recherches longues | Bon si la table sature souvent | Moyenne | Éviction d’entrées utiles |
| Tables de voisinage et petits calculs répétés | 0,5–1 j | Gain limité mais transversal | Bon si le profil confirme leur coût | Moyenne | Complexité ajoutée pour un gain faible |
| Approfondissement tactique sélectif | 2–4 j | Potentiellement élevé contre l’effet d’horizon | Bon, mais plus risqué | Élevée pour le problème, moyenne pour la solution | Explosion du calcul ou faux résultat démontré |
| Génération progressive et dédoublonnage plus tôt | 2–4 j | Potentiellement élevé pour Hermès, Artemis et Demeter | Bon après mesures | Moyenne à élevée | Ordre dégradé ou tours légaux oubliés |
| Équivalence des deux bâtisseurs, dieux de base | 2–3 j | Gain possible surtout avec Hermès | Moyen à bon | Moyenne | Mauvaise correspondance des identités de pions |
| Recherche PVS et fenêtres d’aspiration | 2–3 j | Gain possible si le tri est déjà bon | Moyen | Moyenne | Recherches répétées quand le score oscille |
| Réutilisation du cache entre les tours | 2–4 j | Gain variable selon les réponses réellement jouées | Moyen | Moyenne | Scores dépendant de la racine et de la distance à la victoire |
| Réglage automatique des poids sur un corpus élargi | 3–6 j + calcul | Gain stratégique possible après stabilisation des critères | Moyen à terme | Faible à moyenne sur la première campagne seule | Surapprentissage des placements ou des pouvoirs |
| Migration des pouvoirs Python restants vers Rust | 5–10 j ou plus selon le périmètre | Potentiellement fort pour héros et dieux avancés | À évaluer sur une campagne dédiée | Faible pour ce périmètre non testé ici | Régression des règles complexes |
| Paralléliser une recherche individuelle | 3–5 j | Gain possible, au prix de ressources supplémentaires | Faible en première étape | Moyenne | Concurrence avec l’application et moindre efficacité d’alpha-bêta |
| Nouveau moteur MCTS ou réseau neuronal | Plusieurs semaines et données supplémentaires | Inconnu pour ce projet | Faible à court terme | Faible | Effort élevé sans référence démontrant un progrès |

**Meilleur premier investissement : corpus et mesures, puis évaluation des constructions et tri tactique.** Les optimisations mémoire sont une bonne candidate suivante si le profil les confirme. Les changements plus lourds doivent être décidés sur leurs gains mesurés, pas sur leur réputation générale.

## 4. Axe stratégique : évaluer les constructions accessibles

### Cas qui justifie cet axe

Dans la partie 16, Héphaïstos quitte B4, niveau 1, pour C4, au sol. La double construction fait passer B4 de 1 à 3 : ce pion ne peut plus y retourner directement. Construire une fois en D4 crée au contraire un premier étage accessible depuis C4. Cette alternative gagne lors de la reprise.

L’inversion du résultat est établie. L’idée que la rampe explique à elle seule toute la différence reste une hypothèse stratégique à tester sur d’autres positions.

### Modifications envisageables

Ajouter quelques critères peu coûteux, calculés localement :

- nombre d’accès légaux à un niveau 2 depuis les bâtisseurs ou une étape intermédiaire ;
- continuité de chemins de hauteur `0 → 1 → 2 → 3`, en tenant compte des occupations et des dômes ;
- mobilité du bâtisseur qui vient de construire, et possibilités restantes pour l’autre ;
- disparition d’une rampe utile ou enfermement d’un pion après une construction ;
- accès créé pour l’adversaire, notamment une montée gagnante ou une descente gagnante de Pan ;
- différence entre construire un niveau 3 exploitable et construire une tour encore inaccessible.

Commencer par un voisinage limité et des pondérations modérées. Calculer une recherche complète des déplacements d’Hermès à chaque feuille annulerait probablement le gain stratégique par son coût.

Les chemins sont des possibilités, pas des victoires garanties. Un adversaire peut construire, occuper une case ou activer une restriction avant que le chemin soit parcouru. Les critères ne doivent pas donner à une rampe le score réservé à une victoire démontrée.

Ne pas instaurer de règle « ne jamais construire deux fois » pour Héphaïstos. Une double construction peut créer une menace décisive, bloquer une montée adverse ou être meilleure qu’une rampe. On doit évaluer son résultat, pas pénaliser automatiquement l’usage du pouvoir.

### Validation

Comparer une variante ne changeant que l’évaluation à la référence sur les positions critiques et de nouveaux placements. Inclure des contre-exemples où une double construction gagne. Tester les pouvoirs qui modifient les routes : Artemis, Apollo, Minotaur, Hermès, Athena et Pan.

Le Rust et le Python doivent partager les mêmes nouveaux critères lorsqu’ils s’appliquent, même si leurs profondeurs atteintes diffèrent. La première campagne porte sur les dieux de base du moteur Rust ; elle ne mesure pas le gain pour les héros et dieux avancés.

## 5. Axe tactique : examiner plus tôt les décisions décisives

### Ordre d’exploration

Enrichir le tri déjà présent avec, dans cet ordre à expérimenter :

1. victoire immédiate légale ;
2. coup privilégié par le cache ou la précédente profondeur ;
3. défense contre une victoire immédiate adverse ;
4. création de menaces multiples et blocages décisifs ;
5. coups ayant provoqué des coupures utiles dans des recherches précédentes ;
6. score stratégique général.

Les informations de coupure peuvent être enregistrées sous forme d’un historique léger ou de quelques coups privilégiés par profondeur. Elles ne dispensent pas de vérifier qu’un coup est légal dans la nouvelle position.

L’intérêt est d’obtenir des coupures alpha-bêta plus tôt, en conservant tous les coups légaux nécessaires à la recherche. Un meilleur tri peut suffire à terminer un niveau supplémentaire sans augmenter le délai.

### Menaces propres aux pouvoirs

Un détecteur de victoire doit utiliser les règles du pouvoir, pas seulement rechercher un voisin libre de niveau 3. Il faut notamment couvrir la descente de Pan, le second déplacement d’Artemis, les échanges d’Apollo, les poussées de Minotaur et l’interdiction temporaire d’Athena.

L’évaluation actuelle ignore les cases occupées lorsqu’elle compte les accès adjacents. C’est correct pour les déplacements ordinaires, mais cela peut sous-estimer les possibilités légales d’Apollo et de Minotaur. Un détecteur spécialisé peut aider le tri et l’évaluation ; son gain réel reste à mesurer.

Pour les premières améliorations, limiter le détecteur rapide aux dieux de base et le comparer systématiquement au générateur exhaustif des règles. Ne pas appliquer une approximation de ces menaces aux héros ou dieux avancés.

### Prolonger les positions instables

Au lieu de terminer systématiquement à la profondeur nominale, explorer un petit nombre de tours supplémentaires lorsque la position comporte une menace immédiate ou une réponse forcée. Une recherche de stabilisation tactique, souvent appelée *quiescence*, est une piste ; des extensions limitées sur les menaces en sont une autre.

Conditions nécessaires : budget global inchangé, plafond d’extensions, contrôles de délai fréquents, et vérification du coût des détecteurs. Un détecteur qui énumère toutes les réponses à chaque feuille peut devenir plus cher que la recherche qu’il améliore.

**Ne pas annoncer une victoire forcée sur la base d’une sélection heuristique des réponses adverses.** Si une recherche omet des branches sans établir qu’elles sont dominées ou immédiatement perdantes, son résultat doit rester une estimation. Une extension qui conserve les réponses légales peut préserver les garanties, à condition que les bornes et les entrées de cache soient valides pour cette nouvelle politique de profondeur.

## 6. Axe performance : produire et conserver moins de travail inutile

### Profilage préalable

Mesurer séparément : génération, dédoublonnage, tri, évaluation, recherche, cache et conversion/validation Python. Ajouter le nombre de tours générés, de successeurs conservés, de coupures, de consultations utiles du cache et son taux de saturation.

Le nombre de « positions » affiché ne compte pas séparément tout le travail de génération. Deux variantes qui visitent autant de positions peuvent avoir des coûts différents. Comparer aussi le temps pour atteindre une même profondeur et le nombre de coups légaux évalués.

### Réduire les variantes copiées

Stocker une entrée de cache plus petite : valeur, profondeur, type de borne et information nécessaire pour retrouver le meilleur tour. Reconstruire la variante affichée à partir des choix enregistrés, avec validation finale. Étudier aussi une reconstruction de variante après la recherche plutôt que la concaténation de listes à chaque retour récursif.

Pour Hermès, le meilleur tour peut comprendre plusieurs déplacements. Une représentation réduite ne doit pas perdre le chemin légal ou le bâtisseur qui construit. Garder une position finale et retrouver ensuite une séquence représentative est envisageable, mais ce travail doit entrer dans le budget de restitution du coup.

### Améliorer la table existante

Autoriser d’abord la mise à jour d’une entrée existante même lorsque la table est pleine. Ensuite, remplacer les entrées peu utiles par des entrées plus profondes ou plus récentes, au lieu d’arrêter les insertions une fois le plafond atteint. Mesurer l’effet à mémoire constante avant d’augmenter la taille. La fréquence de saturation n’a pas été enregistrée pendant la première campagne et doit être mesurée.

Un hachage plus compact peut également être étudié. Si la clé devient une empreinte plutôt que la position complète, prévoir une protection contre les collisions pour les résultats critiques et les coups restitués.

### Génération progressive

Explorer d’abord le coup du cache et les victoires immédiates, puis produire les autres coups par étapes. Dédoublonner plus tôt peut éviter des copies et des évaluations de séquences équivalentes. Le moteur Python offre déjà un exemple de tri par lots, mais celui-ci doit être comparé au tri global du Rust.

La génération progressive échange un meilleur coût initial contre un ordre de tri parfois moins bon. Il faut mesurer le temps total : économiser la génération mais perdre les coupures peut rendre la recherche plus lente.

Hermès mérite un traitement spécifique de ses positions accessibles à hauteur constante. Il faut conserver les mouvements des deux pions et leurs interactions d’occupation ; on ne peut pas simplement considérer chaque pion comme se déplaçant seul sur toutes les cases de sa hauteur.

### Équivalences des pions et du plateau

Pour les dieux de base à deux bâtisseurs interchangeables, permuter leurs identités ne change généralement pas la situation stratégique. Une représentation canonique peut éviter de traiter ces positions comme différentes. Les rotations et réflexions du plateau offrent une autre piste, avec un coût de transformation à mesurer.

Les coups mémorisés doivent être remappés vers les identités et coordonnées de la position réelle avant validation. Ne pas généraliser la permutation aux pouvoirs qui donnent un rôle distinct à un pion : Selene, par exemple. Les restrictions et cibles liées à un indice de bâtisseur doivent aussi être transformées correctement.

Commencer par les dieux de base. Les facteurs théoriques de symétrie ne sont pas des promesses de gain de vitesse : les transformations coûtent du calcul et toutes les positions n’offrent pas le même bénéfice.

## 7. Changements à réserver à une seconde étape

### PVS et fenêtres d’aspiration

PVS utilise une fenêtre de recherche étroite sur les coups suivant le meilleur candidat ; une recherche complète est relancée si nécessaire. Les fenêtres d’aspiration commencent autour du score de la profondeur précédente.

Ces techniques peuvent réduire le travail si les coups sont bien triés et les scores stables. La partie 18 présente justement une forte oscillation entre profondeurs : des fenêtres trop étroites peuvent provoquer plusieurs relances et consommer le délai. Les tester après l’évaluation et le tri, avec un élargissement rapide.

### Réutiliser le cache entre les tours

L’application lance actuellement le calcul dans un processus distinct. Un cache conservé d’un tour à l’autre demanderait un service de calcul persistant ou un transfert d’informations compactes, tout en gardant l’interface indépendante.

Les scores du moteur sont relatifs au joueur racine et les scores de victoire dépendent de la distance à cette racine. Réutiliser directement les valeurs au tour suivant serait incorrect. Il faut normaliser le point de vue et la distance, intégrer l’état complet des pouvoirs, et invalider le cache lors d’une correction d’historique ou d’une nouvelle configuration.

Ce changement touche davantage l’architecture que l’amélioration du cache interne à une recherche. Le conserver pour une étape ultérieure.

### Ajustement automatique des pondérations

Utiliser les réanalyses longues pour construire des exemples, puis optimiser un petit ensemble de poids. Une évaluation à 60 secondes non démontrée reste un avis plus approfondi, pas une vérité de référence. Les variantes gagnantes vérifiées offrent des étiquettes plus solides.

Réserver des placements entiers à la validation et séparer les variantes d’un même placement dans les groupes d’apprentissage et de test. Ajouter des adversaires de versions différentes et des positions provenant de parties humaines pour réduire les angles morts de l’auto-jeu.

Ne pas transformer les résultats 12–0 d’Apollo contre Minotaur en bonus permanent pour Apollo. Le choix des pouvoirs dépend de la paire, du placement, de la version du moteur et du budget ; les données actuelles sont trop peu nombreuses pour un tel réglage.

### Architectures alternatives et parallélisme

Un moteur neuronal ou MCTS pourrait être étudié plus tard, mais impose une nouvelle référence, davantage de données et une gestion spécifique des preuves de victoire. Le moteur actuel fournit déjà une erreur précise que des changements ciblés peuvent traiter.

Paralléliser un tour individuel est différent de jouer plusieurs parties indépendantes en parallèle. La première solution complique les coupures et les échanges de cache, et mobilise davantage de ressources pendant l’utilisation normale. Optimiser le calcul sur un cœur est le premier investissement conseillé.

## 8. Plan de mise en œuvre proposé

| Étape | Livrable concret | Critère pour poursuivre |
|---|---|---|
| 0 — Mesurer | Version de référence figée, corpus critique élargi, compteurs de coût | Résultats reproductibles et identification des principaux postes de temps |
| 1 — Évaluation | Variante avec critères limités de rampes, accès et enfermement | Meilleurs choix sur plusieurs positions critiques, sans régression nette sur les contre-exemples |
| 2 — Tri tactique | Variante indépendante avec menaces légales et historique des coupures | Moins de travail pour une même profondeur ou meilleurs résultats à 5 s |
| 3 — Mémoire | Variantes de cache et de reconstruction des lignes | Gain de débit confirmé à mémoire contrôlée, restitution toujours légale |
| 4 — Tactique prolongée | Extensions limitées, statut de preuve fiable | Moins d’erreurs d’horizon sans perte générale de force liée au surcoût |
| 5 — Génération et symétries | Optimisations ciblées sur Hermès, puis autres pouvoirs | Gain confirmé sur ces pouvoirs et absence de perte sur les autres |
| 6 — Réglage et généralisation | Pondérations retenues, tests sur nouveaux placements et autres pouvoirs | Gain qui se maintient hors du corpus de développement |

Première tranche conseillée : **environ 3 à 5 jours de travail**, pour les mesures et les premiers essais d’évaluation/tri, avec des variantes séparées. Cette tranche doit permettre de choisir la suite sur des résultats, sans engager immédiatement tous les axes. Le profilage peut justifier de placer la mémoire avant le tri ou les extensions.

## 9. Comment mesurer le ratio coût / efficacité réel

Chaque changement doit être comparé seul à la référence avant de combiner les variantes retenues. La comparaison comporte deux niveaux.

### Positions fixes

- Capturer le temps nécessaire pour retrouver une victoire démontrée ou le meilleur coup connu.
- Vérifier la position 16/26 et d’autres constructions comparables, avec des contre-exemples.
- Ne pas traiter l’alternative à 60 secondes de la position 18/28 comme une victoire prouvée : mesurer la stabilité du score, le changement de coup et la suite de partie.
- Mesurer la durée d’une profondeur donnée, les générations, les coupures et l’efficacité du cache.
- Conserver un budget mural de 5 secondes, comprenant le calcul et la restitution du tour. Dans la campagne native, la restitution mesurée atteint environ 5,03 s au maximum ; le lancement du processus de l’application doit aussi être contrôlé séparément.

Les résultats à budget temporel peuvent varier avec la charge de la machine. Ajouter un budget de nœuds pour certains diagnostics peut aider à isoler un changement algorithmique, mais il ne remplace pas la comparaison finale à 5 secondes.

### Parties entre versions

Jouer la variante contre la référence figée, en inversant les camps et le premier joueur. Comparer sur les mêmes ressources CPU. Tester d’abord les placements connus, puis un lot de placements nouveaux réservés à la validation.

Un lot de 100 parties sert à filtrer les changements. Un résultat de 55–45 n’est pas suffisant à lui seul pour annoncer un progrès robuste, surtout avec des placements partagés. Élargir les campagnes prometteuses et calculer l’incertitude en regroupant les variantes d’un même placement, plutôt qu’en traitant toutes les parties comme indépendantes.

La campagne initiale consomme environ **3,3 heures cumulées de calcul de recherche**, parallélisées en 16,9 minutes sur cette machine. Une comparaison entre deux versions dépendra de la durée de leurs parties et de leurs arrêts anticipés. Les réanalyses longues du corpus sont un coût supplémentaire, à réserver aux cas utiles.

### Conditions de conservation d’un changement

Conserver une variante si elle améliore les résultats de match ou plusieurs décisions critiques et respecte les exigences suivantes :

- tours complets toujours légaux, y compris avec pouvoirs ;
- respect du délai et interface toujours disponible pendant le calcul ;
- aucune hausse non expliquée des contradictions de preuves ;
- absence de régression majeure sur une famille de pouvoirs ;
- amélioration maintenue sur des placements non utilisés pour le réglage ;
- coût en mémoire et en développement cohérent avec le gain observé.

Une profondeur supérieure, un score plus élevé ou davantage de positions visitées ne constituent pas seuls un progrès. **Le ratio utile est le nombre et l’importance des meilleures décisions obtenues à budget égal, rapportés au coût du changement et de sa validation.**

## 10. Décision proposée

Commencer par le corpus et les mesures, puis tester séparément **l’évaluation des constructions accessibles** et **le tri des menaces propres aux pouvoirs**. Le cas Héphaïstos fournit un objectif vérifiable ; les mesures sur Hermès justifient ensuite les optimisations de génération et de mémoire.

Retenir les variantes qui gagnent effectivement en qualité à 5 secondes, puis seulement les combiner. Garder le moteur de production figé pendant ces comparaisons, dans une branche distincte, et conserver la validation Python indépendante des coups Rust.

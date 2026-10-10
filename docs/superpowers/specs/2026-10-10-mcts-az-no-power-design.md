# MCTS + réseau de neurones pour le sous-jeu sans pouvoir — design

Date : 10 octobre 2026. Décidé après le pilote de débit MCTS (jetable, non conservé)
qui a établi que le GPU local (RTX 5060 Laptop) et les 20 threads CPU suffisent
pour une tentative sérieuse, et que le goulot actuel est logiciel (génération de
coups en Python mono-thread), pas matériel.

Contexte roadmap : ce chantier correspond à l'item "nouveau moteur MCTS ou réseau
neuronal" du Tier 5 de `docs/ameliorations-moteur-20261005.md`, explicitement noté
comme nécessitant plusieurs semaines, à faible confiance de gain à court terme.
Le pilote lève une partie de cette incertitude côté débit ; ce document couvre
l'architecture, pas la décision d'investissement elle-même (déjà prise par
l'utilisateur).

## 1. Objectif

Construire un agent de jeu AlphaZero-style (MCTS guidé par un réseau de neurones
entraîné par self-play) pour le **sous-jeu Santorini sans pouvoir** (plateau 5x5,
2 bâtisseurs par joueur, règles de base uniquement), avec l'intention de
**remplacer à terme** le moteur actuel (recherche alpha-bêta + heuristiques,
`native_engine/`) sur ce sous-jeu, puis d'étendre aux pouvoirs si cette première
étape réussit.

Les pouvoirs avancés (Hermès, Héphaïstos, etc.) sont explicitement **hors
périmètre** de ce chantier. Ne pas y toucher avant que la phase sans pouvoir ait
franchi la barre de succès définie en §5.

## 2. Critère de succès (barre de décision)

Même méthodologie que les axes précédents du backlog (cf. mémoire
`santoni-engine-roadmap-priorities`, Tiers 1-3) : candidat (MCTS + réseau) contre
le moteur alpha-bêta actuel, **budget de temps égal par tour**, campagne de 200
parties, intervalle de confiance bootstrap à 95% par groupe de placement.

**Succès = +5pt de taux de victoire avec borne inférieure de l'IC strictement
positive.** En dessous, l'axe est traité comme les précédents échecs documentés
(ranker ML, fork-only-v3) : un résultat négatif honnête, pas une raison de
retenter la même chose sans changement matériel d'approche.

## 3. Architecture

Trois processus longue durée, orchestrés par un script de contrôle unique :

### 3.1 Workers self-play (Rust, N instances en parallèle sur les 20 threads)

Nouveau module Rust isolé du moteur de production (ex. `native_engine/src/mcts.rs`
ou crate séparée dans le même workspace), qui expose les primitives nécessaires au
MCTS : génération des coups légaux du sous-jeu sans pouvoir, application d'un coup,
détection de victoire. **Ne modifie pas** `lib.rs`, `advanced.rs`, `special.rs` —
le moteur de production reste intact et continue de servir d'adversaire de
référence pour l'évaluation en §5.

Chaque worker fait tourner son propre arbre MCTS sur ses propres parties. Au lieu
d'appeler le réseau en synchrone à chaque feuille (ce qu'a fait le pilote jetable,
d'où le sous-débit), chaque worker empile les positions à évaluer dans une file
partagée avec le serveur d'inférence, et attend la réponse de façon asynchrone
pendant que d'autres branches de son arbre progressent.

### 3.2 Serveur d'inférence (Python/PyTorch, 1 processus, GPU)

Vide la file partagée par lots de taille adaptative (ce qui est disponible au
moment du passage, pas une taille fixe qui bloquerait en attente), fait un seul
passage GPU par lot (policy + value), renvoie les résultats aux workers en
attente. C'est le composant qui corrige directement le goulot diagnostiqué par le
pilote (débit plat en multipliant parties parallèles ou simulations/coup par 4).

Le mécanisme de communication worker ↔ serveur (mémoire partagée, socket local, ou
autre IPC) est une décision d'implémentation, pas une décision d'architecture —
tranchée dans le plan, pas ici.

### 3.3 Entraîneur (Python/PyTorch)

Consomme les parties terminées déjà écrites sur disque (voir §4), produit des
exemples (état, politique cible issue des visites MCTS, valeur cible issue du
résultat de partie), fait des pas de descente de gradient, publie un nouveau
checkpoint à intervalle régulier. Les workers self-play rechargent le dernier
réseau publié pour leurs prochaines parties — cycle de générations classique
AlphaZero, pas de ré-entraînement from scratch à chaque cycle.

## 4. Flux de données et reprise sans perte

Contrainte explicite de l'utilisateur : la machine dédiée tourne en quasi-continu,
mais doit pouvoir être interrompue et relancée à volonté sans perte de travail.

- Chaque partie terminée est écrite en un fichier atomique (écriture dans un
  fichier temporaire puis renommage) dans un répertoire par génération. Aucun état
  de partie en cours n'existe uniquement en mémoire au-delà de la partie
  elle-même : la perte maximale en cas d'arrêt brutal est la partie en cours sur
  chaque worker, jamais les parties déjà terminées.
- L'entraîneur sauvegarde un checkpoint atomique après chaque intervalle de pas de
  gradient : poids, état de l'optimiseur, compteur de génération, et position dans
  les données déjà consommées.
- Un script d'orchestration unique sait, au redémarrage, repartir depuis le
  dernier checkpoint entraîneur et la dernière génération de parties présente sur
  disque, et relancer les workers avec le dernier réseau publié. Aucune étape ne
  dépend d'un état mémoire non persisté.
- Interruption volontaire (signal d'arrêt) : chaque processus termine son unité de
  travail en cours (partie ou batch de gradient) avant de sortir, plutôt que de
  couper en plein milieu.

## 5. Dimensionnement du réseau et validation de correction

- **Réseau de départ volontairement petit** (quelques blocs résiduels, peu de
  canaux) — cohérent avec la pratique déjà établie sur ce projet de mesurer avant
  d'optimiser (cf. génération progressive en Tier 1 : gain nul isolément, utile
  seulement combiné). On ne grossit le réseau que si le GPU reste visiblement
  sous-exploité une fois le serveur d'inférence par lots en place, ou si
  l'apprentissage plafonne pour une raison de capacité démontrée, pas supposée.
- **Preuve d'équivalence des règles avant tout entraînement** : la logique de jeu
  Rust exposée au MCTS (§3.1) doit être prouvée équivalente au moteur de référence
  existant par test différentiel exhaustif sur un corpus de positions — même
  méthode que la canonicalisation Hermès (oracle non-canonicalisé vs nouvelle
  logique). Un bug de règles non détecté entraînerait un réseau sur des données
  faussées sans qu'on puisse s'en rendre compte avant la campagne finale.
- **Protocole d'évaluation** : réutilise le format des campagnes précédentes
  (`tools/fork_only_scale_campaign.py` et consorts) — mêmes placements, même
  bootstrap par groupe, même barre de +5pt.

## 6. Hors périmètre (explicitement, pour ce document)

- Tout pouvoir, de base ou avancé (Hermès, Héphaïstos, Athena, etc.).
- Toute tentative de réutiliser ou de faire évoluer le ranker ML précédemment
  rejeté ([[santoni-ml-ranker-rejected]] en mémoire) — approche différente
  (self-play MCTS complet, pas un ranker de features sur l'arbre alpha-bêta
  existant).
- Le mécanisme exact d'IPC worker ↔ serveur d'inférence, le choix précis
  d'hyperparamètres d'entraînement, et la taille exacte du réseau — décisions
  d'implémentation pour le plan, pas pour ce spec.

## 7. Risques connus

- **Effort élevé sans garantie de gain** : déjà signalé dans le document roadmap
  d'origine. Mitigé par la barre de succès binaire du §2 et le réseau volontairement
  petit du §5 — un échec se détecte tôt et à coût mesuré, pas après des semaines de
  calcul sur un réseau surdimensionné.
- **Bug de règles silencieux dans la copie Rust MCTS** : mitigé par la preuve
  d'équivalence obligatoire avant entraînement (§5).
- **Sous-exploitation GPU persistante même avec batching** : possible si la taille
  de lot réelle reste petite (peu de workers parallèles, parties courtes). À
  mesurer dès les premiers cycles de self-play, pas supposé réglé par
  l'architecture seule.
- **Reprise imparfaite en cas de corruption de fichier au moment d'un crash** :
  mitigé par l'écriture atomique (fichier temporaire + renommage) systématique en
  §4, qui élimine les écritures partielles visibles.

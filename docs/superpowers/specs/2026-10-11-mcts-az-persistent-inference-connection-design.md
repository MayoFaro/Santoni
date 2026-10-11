# Connexion persistante worker ↔ serveur d'inférence — design

Date : 11 octobre 2026. Ferme la première moitié du blocker 4 (« leaf batching »,
cf. mémoire `santoni-mcts-az-no-power-chantier`) avant la vraie campagne de 200
parties. Contexte : `tools/mcts_az_vs_reference_campaign.py` reste bloqué tant que
ce blocker n'est pas clos ou au moins discuté — ici, il est scindé en deux moitiés
indépendantes et mesurées séparément ; ce document ne couvre que la première.

## 0. Constat (état actuel du code, sans ambiguïté)

- `native_engine/src/mcts.rs` : le trait `LeafEvaluator::evaluate(&mut self,
  states: &[State]) -> Vec<(Vec<f32>, f32)>` accepte déjà un slice de plusieurs
  états, mais `Mcts::run`/`simulate_one` ne l'appellent jamais qu'avec un seul
  élément (`&[state]`) — une simulation = une descente + un appel réseau
  synchrone + un backup, strictement séquentiel, pas de virtual loss.
- `native_engine/src/bin/selfplay_worker.rs`, `SocketEvaluator` : chaque appel à
  `evaluate()` ouvre une **nouvelle** `UnixStream::connect`, envoie une requête,
  attend la réponse, puis la connexion est fermée côté serveur. `play_one_game`
  fait un appel par simulation MCTS (`simulations_per_move = 32` aujourd'hui,
  cf. `santoni_az/orchestrator.py`), donc 32 connexions-déconnexions par coup
  joué ; `--single-move` (arbitrage campagne) suit le même modèle.
- `santoni_az/inference_server.py` : `_accept_loop` tourne avec
  `_listener.settimeout(0.1)` — une nouvelle connexion entrante n'est acceptée
  qu'au prochain passage de ce poll à 0.1s dans le pire cas. `_client_loop` lit
  une requête, l'ajoute à `self._pending`, attend le résultat, l'écrit, **ferme
  la connexion**. `_drain_and_process_once` vide `self._pending` toutes les
  `flush_interval_s` (0.02s par défaut) et fait un seul passage GPU par lot —
  ce batching inter-connexions reste correct et n'est pas touché ici.
- Commentaire mesuré dans `santoni_az/orchestrator.py` : chaque aller-retour
  coûte de l'ordre de 100-200ms, dominé par ce poll `accept()` à 0.1s — avant
  même de compter le temps GPU.

## 1. Objectif de ce document

Éliminer le coût de connexion/déconnexion par feuille (connect + accept-poll +
close, répété à chaque simulation MCTS), **sans** toucher à la structure de
`Mcts::run` ni introduire de virtual loss / batching multi-feuilles — cette
deuxième moitié du blocker 4 est un chantier séparé, à spécifier seulement après
avoir mesuré le gain de celui-ci.

## 2. Portée

S'applique aux deux chemins qui possèdent un `SocketEvaluator` :
self-play continu (`play_one_game`, boucle principale de `main()`) et
l'arbitrage `--single-move` (utilisé par `tools/mcts_az_vs_reference_campaign.py`).
Une seule connexion par processus worker dans les deux cas, ouverte une fois et
réutilisée pour toute la durée du processus — y compris entre parties
successives en mode self-play continu, pas seulement au sein d'une partie.

## 3. Architecture

### 3.1 Côté worker (`SocketEvaluator`, `native_engine/src/bin/selfplay_worker.rs`)

`SocketEvaluator` détient désormais un `UnixStream` ouvert, créé au premier
appel à `evaluate()` (ou à la construction — détail d'implémentation) et réutilisé
pour tous les appels suivants. Le format binaire sur le fil ne change pas (déjà
auto-délimité par une longueur explicite en tête de message), seule la
discipline de connexion change : plus de `connect()`/fermeture par appel.

### 3.2 Côté serveur (`santoni_az/inference_server.py`)

`_accept_loop` continue de spawner un thread par connexion entrante (inchangé).
`_client_loop` change de modèle : au lieu de « lire une requête, attendre le
résultat, écrire la réponse, fermer », il boucle sur la même connexion — lit une
requête, l'ajoute à `self._pending` avec un signal de complétion propre à cette
requête, attend ce signal, écrit la réponse, puis retourne lire la requête
suivante — jusqu'à ce que la lecture rencontre une fin de flux ou une erreur, où
le thread ferme proprement et se termine. `_drain_and_process_once` et le
batching par fenêtre de `flush_interval_s` restent inchangés : ils continuent de
regrouper tout ce qui est arrivé de connexions distinctes dans la fenêtre,
simplement sans plus fermer la connexion derrière.

### 3.3 Effet attendu

Le coût `connect()`/poll `accept()` (0.1s dans le pire cas) disparaît du chemin
chaud : il n'a plus lieu qu'une fois par worker à son démarrage, jamais plus par
feuille évaluée.

## 4. Gestion d'erreur

Chaque appel `evaluate()` écrit/lit sur le flux persistant existant. En cas
d'erreur I/O (pipe brisé, connexion réinitialisée — typiquement un redémarrage du
serveur d'inférence), `SocketEvaluator` ferme le flux mort, rouvre une nouvelle
connexion (bloquant jusqu'au succès du `connect()`, comme au démarrage actuel) et
**rejoue la même requête une fois** sur la nouvelle connexion. Si ce deuxième
essai échoue aussi, l'erreur se propage normalement (panic) — comportement
inchangé par rapport à aujourd'hui pour ce cas déjà rare, non traité
différemment.

## 5. Hors périmètre (explicitement, pour ce document)

- Tout changement à `Mcts::run`/`simulate_one` : pas de virtual loss, pas de
  regroupement de plusieurs feuilles d'un même arbre en un seul appel
  `evaluate()` — c'est la deuxième moitié du blocker 4, spécifiée séparément
  après mesure du gain obtenu ici.
- Le format binaire des messages (`u32` LE + planes, `u32` LE + items
  policy/value) ne change pas.
- `flush_interval_s` (fenêtre de batching inter-connexions côté serveur) ne
  change pas.

## 6. Validation

- Test Rust : un faux listener de test compte ses `accept()` ; deux appels
  `evaluate()` successifs sur le même `SocketEvaluator` ne doivent déclencher
  qu'un seul `accept()` côté serveur de test.
- Test Rust : le faux serveur ferme la connexion entre deux requêtes ;
  `evaluate()` doit tout de même renvoyer le bon résultat au deuxième appel
  (reconnexion transparente vérifiée), sans paniquer sur le premier échec.
- Test Python : un client envoie N requêtes séquentielles sur une même
  connexion ; le serveur doit répondre N fois sans fermer la socket entre deux,
  et sans qu'un deuxième `accept()` n'ait lieu pour ce client.
- Mesure (hors suite automatisée, à exécuter manuellement après implémentation) :
  rejouer la même méthodologie déjà utilisée en mémoire
  (`santoni-mcts-az-no-power-chantier`) — sessions de 8 minutes via
  `Orchestrator`, mêmes paramètres (`simulations_per_move=32`, même nombre de
  workers) — avant/après ce changement, pour chiffrer le gain réel de débit
  (parties produites par minute) avant de décider si la deuxième moitié du
  blocker 4 (virtual loss / batching intra-arbre) reste nécessaire.

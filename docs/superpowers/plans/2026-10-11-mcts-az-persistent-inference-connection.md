# Connexion persistante worker ↔ serveur d'inférence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Éliminer le coût de connexion/déconnexion par feuille MCTS (une nouvelle `UnixStream` ouverte et détruite à chaque appel `evaluate()`, dominé par le poll `accept()` à 0.1s côté serveur) en faisant de la connexion worker ↔ serveur d'inférence une ressource persistante, réutilisée pour tous les appels d'un même processus — sans toucher à `Mcts::run`/`simulate_one` ni introduire de virtual loss.

**Architecture:** `SocketEvaluator` (Rust) garde une `UnixStream` ouverte entre les appels à `evaluate()`, créée au premier besoin et réutilisée ensuite ; en cas d'erreur d'E/S elle referme le flux mort, en rouvre un nouveau et rejoue la même requête une fois. `InferenceServer._client_loop` (Python) boucle sur la même connexion au lieu de la fermer après un seul échange, et `_drain_and_process_once` ne ferme plus la connexion après avoir répondu (seulement en cas d'échec d'écriture). Le format binaire des messages et le batching inter-connexions par fenêtre de `flush_interval_s` ne changent pas.

**Tech Stack:** Rust (`native_engine`, aucune nouvelle crate — règle du projet), Python (`santoni_az`, `pytest`).

**Spec:** `docs/superpowers/specs/2026-10-11-mcts-az-persistent-inference-connection-design.md`

## Global Constraints

- Aucune nouvelle crate externe dans `native_engine` (règle du projet).
- Le format binaire du protocole (`u32` LE + planes en requête ; `u32` LE + items policy/value en réponse) ne change pas.
- `flush_interval_s` et le batching inter-connexions dans `_drain_and_process_once` ne changent pas.
- Aucun changement à `native_engine/src/mcts.rs` (`Mcts::run`/`simulate_one`) ni à la structure de `play_one_game`/`run_single_move` au-delà du type de l'évaluateur qu'ils utilisent déjà — ce plan ne touche que la couche transport.
- S'applique aux deux chemins qui possèdent un `SocketEvaluator` : self-play continu et l'arbitrage `--single-move` — une seule connexion par processus worker dans les deux cas, créée au premier appel à `evaluate()` et conservée pour toute la durée du processus (y compris entre parties successives en mode self-play continu).

## Review Focus

- Un échec d'écriture/lecture sur la connexion persistante (pipe brisé, serveur redémarré) doit être rattrapé par une reconnexion transparente et un seul rejeu de la requête, pas par une panique immédiate — couvert par `evaluate_reconnects_transparently_after_a_server_restart` (Task 1).
- Un worker qui ne fait qu'un seul appel `evaluate()` dans toute sa vie (ex. une partie à un seul ply avant terminaison) ne doit pas se comporter différemment qu'avant — la connexion s'ouvre paresseusement au premier appel, jamais à la construction ; aucun test dédié supplémentaire nécessaire, le comportement est identique au cas multi-appels avec N=1.
- Le serveur ne doit plus fermer la connexion après une réponse réussie, mais doit toujours la fermer proprement si le client ferme son côté (EOF) ou si une erreur survient en cours de lecture — couvert par les deux nouveaux tests Python de Task 2 (reprise de connexion, lecture jusqu'à EOF).
- Le batching inter-connexions existant (plusieurs workers distincts dont les requêtes tombent dans la même fenêtre de `flush_interval_s`) ne doit pas se dégrader quand les connexions deviennent persistantes — couvert par `test_two_persistent_clients_each_send_multiple_sequential_requests` (Task 2), qui vérifie deux connexions concurrentes envoyant chacune plusieurs requêtes séquentielles.
- Un échec d'écriture de la réponse côté serveur (le worker a disparu entre sa requête et la réponse) ne doit pas faire remonter une exception non interceptée dans le thread de flush (ce qui arrêterait silencieusement tout batching futur) — couvert par le remplacement de `finally: conn.close()` par `except OSError: conn.close()` dans Task 2 ; pas de test dédié (la panne réseau exacte au bon microseconde n'est pas reproductible de façon fiable), mais le code revu explicitement pour ce cas en Review Focus plutôt que supposé correct.

---

## Task 1: Connexion persistante + reconnexion automatique dans `SocketEvaluator` (Rust)

**Files:**
- Modify: `native_engine/src/bin/selfplay_worker.rs:25-94` (doc-comment, struct `SocketEvaluator`, `impl SocketEvaluator`, `impl LeafEvaluator for SocketEvaluator`)
- Test: `native_engine/src/bin/selfplay_worker.rs` (module `socket_evaluator_tests`, déjà présent en bas du fichier)

**Interfaces:**
- Consumes: `santoni_engine::mcts::{encode_planes, ACTION_SPACE}`, `santoni_engine::mcts::LeafEvaluator` (inchangés).
- Produces: `SocketEvaluator::connect(path: &str) -> Self` (signature inchangée, comportement changé : ne se connecte plus immédiatement), `impl LeafEvaluator for SocketEvaluator` (signature `evaluate(&mut self, states: &[State]) -> Vec<(Vec<f32>, f32)>` inchangée) — consommé par `play_one_game` et `run_single_move` dans `main()`, aucun changement de signature à leurs appels.

- [ ] **Step 1: Write the failing tests**

Ajouter ces deux tests dans le module `socket_evaluator_tests` (`native_engine/src/bin/selfplay_worker.rs`), après le test existant `evaluate_decodes_a_multi_item_response_at_the_right_offsets` :

```rust
    #[test]
    fn evaluate_reuses_the_same_connection_across_calls() {
        let dir = std::env::temp_dir().join(format!("selfplay_worker_reuse_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("création du répertoire temporaire");
        let socket_path = dir.join("fake_infer_reuse.sock");
        let _ = std::fs::remove_file(&socket_path);
        let listener = UnixListener::bind(&socket_path).expect("bind socket factice");
        let socket_path_str = socket_path.to_str().unwrap().to_string();

        let accept_count = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let accept_count_server = accept_count.clone();
        let server = std::thread::spawn(move || {
            // Une seule connexion attendue : si evaluate() en rouvrait une à
            // chaque appel, ce unique accept() ne verrait jamais la seconde
            // requête (détecté ci-dessous via un timeout plutôt qu'un
            // blocage permanent du thread client).
            let (mut conn, _) = listener.accept().expect("accept socket factice");
            accept_count_server.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            for _ in 0..2 {
                let mut header = [0u8; 4];
                conn.read_exact(&mut header).expect("lecture entête requête (test)");
                let n = u32::from_le_bytes(header) as usize;
                let mut body = vec![0u8; n * 150];
                conn.read_exact(&mut body).expect("lecture corps requête (test)");
                let mut response = 1u32.to_le_bytes().to_vec();
                response.extend_from_slice(&vec![0u8; ACTION_SPACE * 4]);
                response.extend_from_slice(&0.0f32.to_le_bytes());
                conn.write_all(&response).expect("écriture réponse factice");
            }
        });

        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let mut evaluator = SocketEvaluator::connect(&socket_path_str);
            let states = vec![initial_state()];
            evaluator.evaluate(&states);
            evaluator.evaluate(&states);
            let _ = tx.send(());
        });
        rx.recv_timeout(std::time::Duration::from_secs(5)).expect(
            "evaluate() n'a pas abouti à temps : la connexion a probablement été rouverte au lieu d'être réutilisée",
        );

        server.join().expect("jointure thread serveur factice");
        assert_eq!(accept_count.load(std::sync::atomic::Ordering::SeqCst), 1);
        let _ = std::fs::remove_file(&socket_path);
    }

    #[test]
    fn evaluate_reconnects_transparently_after_a_server_restart() {
        let dir = std::env::temp_dir().join(format!("selfplay_worker_reconnect_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("création du répertoire temporaire");
        let socket_path = dir.join("fake_infer_reconnect.sock");
        let _ = std::fs::remove_file(&socket_path);
        let listener = UnixListener::bind(&socket_path).expect("bind socket factice");
        let socket_path_str = socket_path.to_str().unwrap().to_string();

        let server = std::thread::spawn(move || {
            for i in 0..2u32 {
                let (mut conn, _) = listener.accept().expect("accept socket factice");
                let mut header = [0u8; 4];
                conn.read_exact(&mut header).expect("lecture entête requête (test)");
                let n = u32::from_le_bytes(header) as usize;
                let mut body = vec![0u8; n * 150];
                conn.read_exact(&mut body).expect("lecture corps requête (test)");
                let value = i as f32 + 0.5;
                let mut response = 1u32.to_le_bytes().to_vec();
                response.extend_from_slice(&vec![0u8; ACTION_SPACE * 4]);
                response.extend_from_slice(&value.to_le_bytes());
                conn.write_all(&response).expect("écriture réponse factice");
                // Fermeture explicite : simule un serveur qui redémarre
                // entre deux requêtes du même worker, sans compter sur le
                // moment exact où `Drop` fermerait le descripteur.
                conn.shutdown(std::net::Shutdown::Both).ok();
            }
        });

        let mut evaluator = SocketEvaluator::connect(&socket_path_str);
        let states = vec![initial_state()];
        let first = evaluator.evaluate(&states);
        assert_eq!(first[0].1, 0.5);
        let second = evaluator.evaluate(&states);
        assert_eq!(second[0].1, 1.5);
        server.join().expect("jointure thread serveur factice");
        let _ = std::fs::remove_file(&socket_path);
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd native_engine && cargo test --bin selfplay_worker socket_evaluator_tests -- --nocapture`
Expected: `evaluate_reuses_the_same_connection_across_calls` FAILS (panique avec le message de timeout `recv_timeout`, ou `assert_eq!(accept_count..., 1)` échoue avec `2`) — le code actuel ouvre une nouvelle connexion à chaque appel. `evaluate_reconnects_transparently_after_a_server_restart` PASSE déjà contre le code actuel — sans surprise, puisqu'aujourd'hui chaque appel reconnecte de toute façon sans condition ; ce test est conservé comme épingle de régression pour le contrat de reconnexion une fois la connexion persistante en place à l'étape 3.

- [ ] **Step 3: Write the implementation**

Remplacer tout le bloc allant du doc-comment de `SocketEvaluator` jusqu'à la fin de `impl LeafEvaluator for SocketEvaluator` (lignes 25-94 actuelles) par :

```rust
/// Évaluateur qui délègue chaque lot de feuilles au serveur d'inférence
/// Python via le protocole binaire (`santoni_az/inference_protocol.py`) :
/// requête = `u32` LE (nombre d'états) puis `count * PLANE_BYTES` octets (un
/// encodage de plans par état, `santoni_engine::mcts::encode_planes`) ;
/// réponse = `u32` LE (nombre d'items) puis, pour chaque item, `ACTION_SPACE`
/// `f32` LE (logits de politique) suivis d'un `f32` LE (valeur). Tous les
/// offsets ci-dessous sont calculés de la même façon que côté Python
/// (`inference_protocol.pack_request`/`unpack_response`) : get this wrong and
/// every policy/value the self-play worker sees is silently corrupted.
///
/// La connexion est persistante : une seule `UnixStream` est ouverte au
/// premier appel à `evaluate` puis réutilisée pour tous les appels suivants,
/// y compris entre deux parties successives (le serveur d'inférence,
/// `InferenceServer._client_loop`, boucle lui aussi sur la même connexion
/// plutôt que de la fermer après un seul échange). Ceci élimine le coût de
/// `connect()`/poll `accept()` (jusqu'à 0.1s dans le pire cas, cf.
/// `InferenceServer._listener.settimeout`) qui dominait chaque aller-retour
/// quand une nouvelle connexion était ouverte à chaque feuille évaluée. En
/// cas d'erreur d'E/S sur la connexion existante (pipe brisé, serveur
/// redémarré), `evaluate` referme le flux mort, en rouvre un nouveau, et
/// rejoue la même requête une fois ; un second échec se propage normalement
/// (panic), ce cas restant aussi rare qu'avant ce correctif.
struct SocketEvaluator {
    path: String,
    stream: Option<UnixStream>,
}

impl SocketEvaluator {
    // Ne se connecte pas ici : la connexion s'ouvre au premier appel à
    // `evaluate` (dans `ensure_connected`), pas à la construction — un
    // worker qui démarre avant que le serveur d'inférence soit prêt doit
    // pouvoir être instancié sans échouer immédiatement ; l'échec, s'il a
    // lieu, se manifeste seulement au premier vrai besoin de la connexion.
    fn connect(path: &str) -> Self {
        Self { path: path.to_string(), stream: None }
    }

    fn ensure_connected(&mut self) -> &mut UnixStream {
        if self.stream.is_none() {
            let stream = UnixStream::connect(&self.path).expect("connexion au serveur d'inférence impossible");
            self.stream = Some(stream);
        }
        self.stream.as_mut().unwrap()
    }

    fn try_evaluate(&mut self, states: &[State]) -> std::io::Result<Vec<(Vec<f32>, f32)>> {
        let stream = self.ensure_connected();
        let count = states.len() as u32;
        let mut request = count.to_le_bytes().to_vec();
        for s in states {
            request.extend_from_slice(&santoni_engine::mcts::encode_planes(s));
        }
        stream.write_all(&request)?;
        let mut header = [0u8; 4];
        stream.read_exact(&mut header)?;
        let n = u32::from_le_bytes(header) as usize;
        let item_size = ACTION_SPACE * 4 + 4;
        let mut body = vec![0u8; n * item_size];
        stream.read_exact(&mut body)?;
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let base = i * item_size;
            let mut policy = Vec::with_capacity(ACTION_SPACE);
            for j in 0..ACTION_SPACE {
                let o = base + j * 4;
                policy.push(f32::from_le_bytes([body[o], body[o + 1], body[o + 2], body[o + 3]]));
            }
            let vo = base + ACTION_SPACE * 4;
            let value = f32::from_le_bytes([body[vo], body[vo + 1], body[vo + 2], body[vo + 3]]);
            result.push((policy, value));
        }
        Ok(result)
    }
}

impl LeafEvaluator for SocketEvaluator {
    fn evaluate(&mut self, states: &[State]) -> Vec<(Vec<f32>, f32)> {
        match self.try_evaluate(states) {
            Ok(result) => result,
            Err(_) => {
                // Connexion morte (pipe brisé, serveur redémarré) : on
                // referme et rouvre une nouvelle connexion, puis on rejoue
                // la même requête une fois.
                self.stream = None;
                self.try_evaluate(states).expect(
                    "connexion au serveur d'inférence impossible après une tentative de reconnexion",
                )
            }
        }
    }
}
```

- [ ] **Step 4: Rebuild and run tests to verify they pass**

Run: `cd native_engine && cargo build --release --bin selfplay_worker && cargo test --bin selfplay_worker -- --nocapture`
Expected: PASS, y compris les deux nouveaux tests, le test préexistant `evaluate_decodes_a_multi_item_response_at_the_right_offsets` (toujours un seul appel `evaluate` par connexion, inchangé) et tout le module `exploration_tests`.

- [ ] **Step 5: Commit**

```bash
git add native_engine/src/bin/selfplay_worker.rs
git commit -m "Connexion persistante + reconnexion automatique dans SocketEvaluator"
```

---

## Task 2: Boucle de connexion persistante dans `InferenceServer` (Python)

**Files:**
- Modify: `santoni_az/inference_server.py:85-130` (`_client_loop`, la boucle de réponse dans `_drain_and_process_once`)
- Test: `tests/test_az_inference_server.py`

**Interfaces:**
- Consumes: aucune nouvelle dépendance — `struct`, `unpack_request`, `pack_response`, `PLANE_BYTES` déjà importés.
- Produces: `InferenceServer._client_loop(self, conn)` (même signature, même rôle de thread-cible, nouveau comportement en boucle) ; `InferenceServer._drain_and_process_once` garde sa signature `(self) -> None` mais ne ferme plus la connexion après une réponse réussie.

- [ ] **Step 1: Write the failing tests**

Ajouter ces deux tests dans `tests/test_az_inference_server.py`, après `test_two_concurrent_clients_get_correct_shaped_responses` :

```python
def test_client_can_reuse_the_same_connection_for_multiple_requests(tmp_path):
    socket_path = str(tmp_path / "infer_reuse.sock")
    net = PolicyValueNet(channels=8, blocks=1)
    server = InferenceServer(socket_path, net, flush_interval_s=0.02)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)

    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(socket_path)
    planes = bytes(PLANE_BYTES)

    def one_round_trip():
        sock.sendall(pack_request([planes]))
        header = sock.recv(4)
        (count,) = struct.unpack("<I", header)
        needed = count * (ACTION_SPACE * 4 + 4)
        body = b""
        while len(body) < needed:
            body += sock.recv(needed - len(body))
        return unpack_response(header + body, batch_size=count)

    for _ in range(3):
        policies, values = one_round_trip()
        assert len(policies[0]) == ACTION_SPACE
        assert -1 <= values[0] <= 1

    sock.close()
    server.stop()


def test_two_persistent_clients_each_send_multiple_sequential_requests(tmp_path):
    socket_path = str(tmp_path / "infer_multi_persist.sock")
    net = PolicyValueNet(channels=8, blocks=1)
    server = InferenceServer(socket_path, net, flush_interval_s=0.02)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)

    def client(results, index):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(socket_path)
        planes = bytes(PLANE_BYTES)
        got = []
        for _ in range(3):
            sock.sendall(pack_request([planes]))
            header = sock.recv(4)
            (count,) = struct.unpack("<I", header)
            needed = count * (ACTION_SPACE * 4 + 4)
            body = b""
            while len(body) < needed:
                body += sock.recv(needed - len(body))
            got.append(unpack_response(header + body, batch_size=count))
        results[index] = got
        sock.close()

    results = {}
    t1 = threading.Thread(target=client, args=(results, 0))
    t2 = threading.Thread(target=client, args=(results, 1))
    t1.start(); t2.start()
    t1.join(timeout=5); t2.join(timeout=5)
    server.stop()

    assert len(results) == 2
    for got in results.values():
        assert len(got) == 3
        for policies, values in got:
            assert len(policies[0]) == ACTION_SPACE
            assert -1 <= values[0] <= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_az_inference_server.py -v`
Expected: `test_client_can_reuse_the_same_connection_for_multiple_requests` et `test_two_persistent_clients_each_send_multiple_sequential_requests` FAILENT (le deuxième `sendall` de chaque client lève `BrokenPipeError`/`ConnectionResetError` — le serveur actuel ferme la connexion après la première réponse). Les deux tests préexistants continuent de PASSER.

- [ ] **Step 3: Write the implementation**

Remplacer `_client_loop` (lignes 85-102 actuelles) par :

```python
    def _client_loop(self, conn):
        while True:
            try:
                header = conn.recv(4)
            except OSError:
                conn.close()
                return
            if len(header) < 4:
                conn.close()
                return
            (count,) = struct.unpack("<I", header)
            body = b""
            while len(body) < count * PLANE_BYTES:
                try:
                    chunk = conn.recv(count * PLANE_BYTES - len(body))
                except OSError:
                    conn.close()
                    return
                if not chunk:
                    conn.close()
                    return
                body += chunk
            planes_list = unpack_request(header + body)
            with self._lock:
                if not self._running:
                    # Server is shutting down: the flush thread may already
                    # have exited, so queuing here would leave this client
                    # hanging on recv() forever. Close instead of stranding it.
                    conn.close()
                    return
                self._pending.append((planes_list, conn))
```

Dans `_drain_and_process_once`, remplacer :

```python
            try:
                conn.sendall(response)
            finally:
                conn.close()
```

par :

```python
            try:
                conn.sendall(response)
            except OSError:
                # Le worker a disparu entre sa requête et la réponse : on
                # referme notre côté, mais on ne laisse pas l'exception
                # remonter dans le thread de flush (qui arrêterait tout
                # batching futur pour les autres connexions en attente).
                conn.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_az_inference_server.py -v`
Expected: PASS, y compris les deux nouveaux tests et les deux tests préexistants (`test_two_concurrent_clients_get_correct_shaped_responses`, `test_flush_loop_final_drain_catches_request_queued_at_shutdown`).

- [ ] **Step 5: Commit**

```bash
git add santoni_az/inference_server.py tests/test_az_inference_server.py
git commit -m "Connexion persistante côté InferenceServer : boucle de lecture au lieu de fermer après chaque réponse"
```

---

## Final check (not a task — run after Task 2)

Run the full relevant test surface once more to confirm nothing else regressed, then rebuild the release binary used by the orchestrator and campaign script:

```bash
cd native_engine && cargo test
cd .. && QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_az_inference_server.py tests/test_mcts_io_format.py tests/test_selfplay_worker_smoke.py tests/test_selfplay_worker_single_move.py tests/test_selfplay_end_to_end.py tests/test_az_trainer.py tests/test_az_network.py tests/test_az_orchestrator.py -v
```

Puis mesurer le gain réel (hors suite automatisée) avec la même méthodologie déjà utilisée en mémoire `santoni-mcts-az-no-power-chantier` : deux sessions de 8 minutes via `Orchestrator`, mêmes paramètres (`simulations_per_move=32`, même nombre de workers), une avant ce changement (déjà mesurée : 56 parties / 8 min) et une après, pour chiffrer le débit réel avant de décider si la deuxième moitié du blocker 4 (virtual loss / batching intra-arbre) reste nécessaire. Mettre à jour la mémoire `santoni-mcts-az-no-power-chantier` avec le résultat chiffré et la décision qui en découle.

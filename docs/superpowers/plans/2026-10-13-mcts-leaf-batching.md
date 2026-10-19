# Regroupement de feuilles MCTS (virtual loss) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Regrouper plusieurs feuilles d'un même arbre MCTS en un seul appel réseau (`LeafEvaluator::evaluate`), pour réduire le nombre d'allers-retours par unité de travail de self-play et d'arbitrage — l'axe qui adresse la surcharge Python/GIL diagnostiquée comme goulot restant, par opposition à la latence par appel déjà traitée par la connexion persistante.

**Architecture:** `Mcts::run` passe d'une boucle séquentielle (une simulation = une descente + un appel réseau bloquant + un backup) à une boucle par vagues : jusqu'à `leaf_batch_size` sélections UCB (avec perte virtuelle sur les visites pour diversifier) avant un seul appel `evaluate()` groupé, puis expansion+backup différés le long des chemins mémorisés. `leaf_batch_size = 1` reproduit exactement le comportement actuel (propriété de non-régression vérifiée par les tests existants). S'applique au self-play et à l'arbitrage déterministe (`run_single_move`) — c'est une technique de performance pure, pas une source de bruit.

**Tech Stack:** Rust (`native_engine`, aucune nouvelle crate — règle du projet).

**Spec:** `docs/superpowers/specs/2026-10-13-mcts-leaf-batching-design.md`

## Global Constraints

- Aucune nouvelle crate externe dans `native_engine` (règle du projet).
- `leaf_batch_size = 1` doit reproduire exactement le comportement du code actuel (aucun test existant ne doit changer de résultat).
- S'applique aux deux chemins qui appellent `Mcts::run` : self-play continu (`play_one_game`) et arbitrage déterministe (`run_single_move`, utilisé par `tools/mcts_az_vs_reference_campaign.py`).
- Aucun changement à `InferenceServer`/`inference_protocol.py` : le protocole accepte déjà plusieurs états par requête.
- Aucun changement au bruit de Dirichlet ni à la température d'ouverture (`DIRICHLET_ALPHA`, `DIRICHLET_EPSILON`, `TEMPERATURE_PLIES`) — restent self-play uniquement, non concernés par ce plan.
- Variante de perte virtuelle retenue : visites seules (`virtual_loss` sur le dénominateur du terme d'exploration UCB), jamais sur `value_sum`/Q — pas la pénalité complète de la littérature.

## Review Focus

- Un coup forcé (une seule arête légale) avec `leaf_batch_size > 1` ne doit jamais paniquer, même avec la perte virtuelle qui s'empile sur l'unique arête — couvert par un test dédié dans Task 1.
- Le total des visites à la racine doit toujours égaler `simulations` exactement, y compris quand `simulations` n'est pas un multiple de `leaf_batch_size` (dernière vague plus petite) — couvert par un test dédié dans Task 1.
- `leaf_batch_size = 1` doit reproduire byte-pour-byte le comportement du code actuel — couvert par la ré-exécution de tous les tests existants avec ce paramètre (Task 1), qui constitue l'ancrage de non-régression de tout ce plan.
- Des feuilles dupliquées dans une même vague (transposition : deux chemins différents menant au même état) ne sont pas dédupliquées — chacune est évaluée séparément dans le lot envoyé au réseau. Choix délibéré (YAGNI, cf. spec §4), pas un bug ; aucun test dédié nécessaire, le mécanisme général (chaque feuille de `pending` traitée indépendamment) le gère déjà correctement par construction.
- `run_single_move`'s `BATCH = 16` (nombre de simulations entre deux vérifications du budget de temps) et `LEAF_BATCH_SIZE` (feuilles par appel réseau) sont deux constantes indépendantes qui coexistent dans la même fonction après ce plan — couvert par Task 2, qui reconstruit le binaire et fait tourner toute la suite de tests existante sur le chemin d'arbitrage, exerçant directement cette coexistence.

---

## Task 1 : Restructuration de `Mcts::run` en sélection/évaluation/backup par vagues

**Files:**
- Modify: `native_engine/src/mcts.rs` (struct `Edge`, nouvelle énumération `SelectOutcome`, `impl Mcts` entier, module `mcts_tree_tests`)

**Interfaces:**
- Consumes: `LeafEvaluator::evaluate(&mut self, states: &[State]) -> Vec<(Vec<f32>, f32)>` (inchangé), `legal_children`, `terminal_value`, `action_index` (inchangés).
- Produces: `pub fn Mcts::run(&mut self, simulations: u32, leaf_batch_size: usize, evaluator: &mut dyn LeafEvaluator)` (signature changée : nouveau paramètre `leaf_batch_size` en deuxième position) ; `ensure_root_expanded`, `mix_root_noise`, `visit_counts` gardent leurs signatures externes inchangées. `simulate_one` disparaît, remplacée par trois méthodes privées : `select_one(&mut self) -> SelectOutcome`, `expand_leaf(&mut self, state: State, priors: &[f32])`, `backup_path(&mut self, path: &[(State, usize)], leaf_value: f32)`.

- [ ] **Step 1: Write the failing tests**

Dans `native_engine/src/mcts.rs`, module `mcts_tree_tests`, mettre à jour les trois appels à `.run(` existants pour y ajouter `leaf_batch_size = 1` (ancrage de non-régression) :

Dans `root_with_one_legal_move_gets_all_simulations_on_that_move`, remplacer :
```rust
        mcts.run(50, &mut eval);
```
par :
```rust
        mcts.run(50, 1, &mut eval);
```

Dans `mix_root_noise_biases_simulations_toward_the_noised_edge`, remplacer :
```rust
        mcts.run(200, &mut eval);
```
par :
```rust
        mcts.run(200, 1, &mut eval);
```

Dans `mix_root_noise_still_biases_toward_the_noised_edge_with_raw_logit_scale_priors`, remplacer :
```rust
        mcts.run(400, &mut eval);
```
par :
```rust
        mcts.run(400, 1, &mut eval);
```

Ajouter ces trois nouveaux tests à la fin du module `mcts_tree_tests` (après `visit_counts_returns_the_after_state_matching_legal_children`) :

```rust
    #[test]
    fn run_produces_exactly_simulations_total_visits_with_batching() {
        let mut s = default_state();
        s.workers = [[12, 13, -1, -1], [0, 1, -1, -1]];
        let mut mcts = Mcts::new(s, 1.5);
        let mut eval = UniformEvaluator;
        mcts.run(37, 8, &mut eval); // 37 n'est pas un multiple de 8
        let counts = mcts.visit_counts();
        let total: u32 = counts.iter().map(|(_, _, _, n)| n).sum();
        assert_eq!(total, 37);
    }

    #[test]
    fn forced_move_with_batching_does_not_panic() {
        let s = state_with_immediate_win(); // une seule arête légale
        let mut mcts = Mcts::new(s, 1.5);
        let mut eval = UniformEvaluator;
        mcts.run(50, 8, &mut eval);
        let counts = mcts.visit_counts();
        let total: u32 = counts.iter().map(|(_, _, _, n)| n).sum();
        assert_eq!(total, 50);
    }

    struct RecordingEvaluator {
        call_sizes: Vec<usize>,
    }
    impl LeafEvaluator for RecordingEvaluator {
        fn evaluate(&mut self, states: &[State]) -> Vec<(Vec<f32>, f32)> {
            self.call_sizes.push(states.len());
            states.iter().map(|_| (vec![1.0 / ACTION_SPACE as f32; ACTION_SPACE], 0.0)).collect()
        }
    }

    #[test]
    fn run_batches_multiple_leaves_into_a_single_evaluate_call() {
        let mut s = default_state();
        s.workers = [[12, 13, -1, -1], [0, 1, -1, -1]];
        let mut mcts = Mcts::new(s, 1.5);
        let mut eval = RecordingEvaluator { call_sizes: Vec::new() };
        mcts.run(64, 8, &mut eval);
        assert!(
            eval.call_sizes.iter().any(|&n| n > 1),
            "au moins un appel à evaluate() doit recevoir plusieurs états à la fois, trouvé {:?}",
            eval.call_sizes
        );
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd native_engine && cargo test --lib mcts_tree_tests -- --nocapture`
Expected: FAIL à la compilation — `run` n'accepte pas encore de deuxième paramètre `leaf_batch_size` (erreur de type sur les trois appels modifiés et les trois nouveaux tests).

- [ ] **Step 3: Write the implementation**

Dans `native_engine/src/mcts.rs`, remplacer tout le bloc allant de `struct Edge` jusqu'à la fin de `impl Mcts` (juste avant `#[cfg(test)] mod tests`) par :

```rust
struct Edge {
    mv: Action,
    bld: Action,
    after: State,
    prior: f32,
    visits: u32,
    value_sum: f32,
    virtual_loss: u32,
}

struct Node {
    edges: Vec<Edge>,
}

/// Résultat d'une passe de sélection (une descente racine→feuille, sans
/// appel réseau) : soit l'état terminal rencontré a une valeur déjà connue
/// (`Terminal`), soit la descente s'est arrêtée sur une feuille jamais vue
/// qui doit être évaluée par le réseau (`Leaf`). Dans les deux cas,
/// `path` porte la séquence (état, index d'arête choisie) traversée depuis
/// la racine, nécessaire pour le backup différé une fois la vraie valeur
/// connue.
enum SelectOutcome {
    Terminal { path: Vec<(State, usize)>, value: f32 },
    Leaf { path: Vec<(State, usize)>, leaf_state: State },
}

pub struct Mcts {
    root: State,
    c_puct: f32,
    nodes: std::collections::HashMap<State, Node>,
}

impl Mcts {
    pub fn new(root: State, c_puct: f32) -> Self {
        Self { root, c_puct, nodes: std::collections::HashMap::new() }
    }

    /// Exécute `simulations` simulations MCTS, regroupées en vagues d'au
    /// plus `leaf_batch_size` feuilles évaluées en un seul appel à
    /// `evaluator.evaluate`. `leaf_batch_size == 1` reproduit exactement le
    /// comportement séquentiel d'avant ce correctif (aucune perte virtuelle
    /// n'a d'effet observable avec une seule sélection par vague) — voir
    /// les tests de non-régression du module `mcts_tree_tests`.
    pub fn run(&mut self, simulations: u32, leaf_batch_size: usize, evaluator: &mut dyn LeafEvaluator) {
        assert!(
            terminal_value(&self.root, self.root.player).is_none(),
            "Mcts::run appelé sur une racine déjà terminale"
        );
        assert!(leaf_batch_size >= 1, "leaf_batch_size doit être au moins 1");
        self.ensure_root_expanded(evaluator);
        let mut done = 0u32;
        while done < simulations {
            let wave_size = (simulations - done).min(leaf_batch_size as u32);
            let mut pending: Vec<(Vec<(State, usize)>, State)> = Vec::new();
            for _ in 0..wave_size {
                match self.select_one() {
                    SelectOutcome::Terminal { path, value } => self.backup_path(&path, value),
                    SelectOutcome::Leaf { path, leaf_state } => pending.push((path, leaf_state)),
                }
            }
            if !pending.is_empty() {
                let leaf_states: Vec<State> = pending.iter().map(|(_, s)| *s).collect();
                let results = evaluator.evaluate(&leaf_states);
                for ((path, leaf_state), (priors, value)) in pending.into_iter().zip(results.into_iter()) {
                    self.expand_leaf(leaf_state, &priors);
                    self.backup_path(&path, value);
                }
            }
            done += wave_size;
        }
    }

    /// Expansion initiale de la racine (évaluation + peuplement des arêtes),
    /// idempotente : ne fait rien si déjà expansée. Extraite de `run` pour
    /// que l'appelant self-play puisse injecter du bruit de Dirichlet
    /// (`mix_root_noise`) entre l'expansion et le reste des simulations ;
    /// l'appeler directement est un no-op si `run` l'a déjà fait.
    pub fn ensure_root_expanded(&mut self, evaluator: &mut dyn LeafEvaluator) {
        if !self.nodes.contains_key(&self.root) {
            let (priors, _value) = evaluator.evaluate(&[self.root]).remove(0);
            self.expand_leaf(self.root, &priors);
        }
    }

    /// Mélange du bruit de Dirichlet dans les priors des arêtes de la
    /// racine — exploration en self-play uniquement, jamais sur le chemin
    /// d'arbitrage (`run_single_move` dans `bin/selfplay_worker.rs`), qui
    /// doit rester le jeu le plus fort possible. Doit être appelée après
    /// `ensure_root_expanded` (les arêtes doivent déjà exister) et avant le
    /// reste des simulations de `run`, sinon le bruit n'affecte jamais la
    /// sélection UCB.
    pub fn mix_root_noise(&mut self, noise: &[f32], epsilon: f32) {
        let node = self
            .nodes
            .get_mut(&self.root)
            .expect("mix_root_noise appelé avant l'expansion de la racine (cf. ensure_root_expanded)");
        assert_eq!(
            node.edges.len(),
            noise.len(),
            "le bruit doit avoir une valeur par coup légal à la racine"
        );
        let max = node.edges.iter().map(|e| e.prior).fold(f32::NEG_INFINITY, f32::max);
        let mut total = 0.0f32;
        for edge in node.edges.iter_mut() {
            edge.prior = (edge.prior - max).exp();
            total += edge.prior;
        }
        for (edge, &n) in node.edges.iter_mut().zip(noise) {
            edge.prior = (edge.prior / total) * (1.0 - epsilon) + n * epsilon;
        }
    }

    /// Une passe de sélection : descend depuis la racine via UCB (en tenant
    /// compte de la perte virtuelle posée par les sélections précédentes de
    /// la même vague), sans jamais appeler le réseau. S'arrête sur un état
    /// terminal (valeur déjà connue) ou sur une feuille jamais vue (à
    /// évaluer par lot). Pose une perte virtuelle d'une unité sur chaque
    /// arête traversée ; `backup_path` l'annule.
    fn select_one(&mut self) -> SelectOutcome {
        let mut path: Vec<(State, usize)> = Vec::new();
        let mut state = self.root;
        loop {
            let to_move = state.player;
            if let Some(v) = terminal_value(&state, to_move) {
                return SelectOutcome::Terminal { path, value: v as f32 };
            }
            if !self.nodes.contains_key(&state) {
                return SelectOutcome::Leaf { path, leaf_state: state };
            }
            let c_puct = self.c_puct;
            let node = self.nodes.get_mut(&state).unwrap();
            let total_effective: u32 =
                node.edges.iter().map(|e| e.visits + e.virtual_loss).sum::<u32>().max(1);
            let sqrt_total = (total_effective as f32).sqrt();
            let score = |e: &Edge| {
                let q = if e.visits == 0 { 0.0 } else { e.value_sum / e.visits as f32 };
                q + c_puct * e.prior * sqrt_total / (1.0 + e.visits as f32 + e.virtual_loss as f32)
            };
            let best = node
                .edges
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| score(a).partial_cmp(&score(b)).unwrap())
                .map(|(i, _)| i)
                .expect("un noeud non terminal expansé a au moins un coup légal");
            node.edges[best].virtual_loss += 1;
            let next_state = node.edges[best].after;
            path.push((state, best));
            state = next_state;
        }
    }

    /// Crée le noeud d'une feuille nouvellement évaluée : une arête par
    /// coup légal, avec son prior tiré de `priors` à l'index
    /// `action_index`. Ne fait aucun backup — appelé juste avant
    /// `backup_path` par `run`/`ensure_root_expanded`.
    fn expand_leaf(&mut self, state: State, priors: &[f32]) {
        let to_move = state.player;
        let children = legal_children(&state);
        let edges = children
            .into_iter()
            .map(|(mv, bld, after)| {
                let worker_slot = state.workers[to_move as usize]
                    .iter()
                    .position(|&c| c == mv.source)
                    .expect("le coup joué doit déplacer un bâtisseur du joueur au trait");
                let idx = action_index(&mv, &bld, worker_slot);
                Edge { mv, bld, after, prior: priors[idx], visits: 0, value_sum: 0.0, virtual_loss: 0 }
            })
            .collect();
        self.nodes.insert(state, Node { edges });
    }

    /// Remonte `path` (de la feuille vers la racine) en annulant la perte
    /// virtuelle posée par `select_one` et en appliquant la vraie valeur,
    /// inversée à chaque niveau (convention négamax — `leaf_value` est la
    /// valeur du point de vue du joueur au trait à la feuille/l'état
    /// terminal, exactement ce que renvoient `LeafEvaluator::evaluate` et
    /// `terminal_value`).
    fn backup_path(&mut self, path: &[(State, usize)], leaf_value: f32) {
        let mut value = -leaf_value;
        for &(state, edge_idx) in path.iter().rev() {
            let node = self.nodes.get_mut(&state).unwrap();
            let edge = &mut node.edges[edge_idx];
            edge.virtual_loss -= 1;
            edge.visits += 1;
            edge.value_sum += value;
            value = -value;
        }
    }

    /// Coups légaux à la racine avec leur nombre de visites et l'état
    /// résultant, pour construire la cible de politique de l'entraînement
    /// et avancer la partie sans recalculer `legal_children` séparément.
    pub fn visit_counts(&self) -> Vec<(Action, Action, State, u32)> {
        self.nodes[&self.root]
            .edges
            .iter()
            .map(|e| (e.mv, e.bld, e.after, e.visits))
            .collect()
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd native_engine && cargo test --lib -- --nocapture`
Expected: PASS sur l'ensemble du fichier — tous les tests préexistants (`tests` et `mcts_tree_tests`, y compris les trois mis à jour avec `leaf_batch_size = 1`) et les trois nouveaux tests.

- [ ] **Step 5: Commit**

```bash
git add native_engine/src/mcts.rs
git commit -m "Mcts::run par vagues de feuilles groupées, perte virtuelle sur les visites"
```

---

## Task 2 : Brancher `LEAF_BATCH_SIZE` dans le self-play et l'arbitrage

**Files:**
- Modify: `native_engine/src/bin/selfplay_worker.rs` (nouvelle constante, `play_one_game`, `run_single_move`)

**Interfaces:**
- Consumes: `Mcts::run(&mut self, simulations: u32, leaf_batch_size: usize, evaluator: &mut dyn LeafEvaluator)` (Task 1).
- Produces: `LEAF_BATCH_SIZE: usize = 8` (nouvelle constante du module) ; `play_one_game`/`run_single_move` gardent leurs signatures externes inchangées.

- [ ] **Step 1: Write the failing tests**

Aucun nouveau test nécessaire pour cette tâche : les tests existants de `exploration_tests` (`same_seed_reproduces_the_same_game`, `different_seeds_produce_different_games`) et de `socket_evaluator_tests` exercent déjà `play_one_game` de bout en bout, et vont échouer à la compilation tant que `play_one_game` n'est pas mis à jour pour le nouvel appel à `Mcts::run` (Step 2 le confirme).

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd native_engine && cargo build --lib --bin selfplay_worker 2>&1 | tail -20`
Expected: FAIL à la compilation — `mcts.run(simulations_per_move, evaluator)` dans `play_one_game` et `mcts.run(BATCH, evaluator)` dans `run_single_move` n'ont que deux arguments, `Mcts::run` en attend trois depuis Task 1.

- [ ] **Step 3: Write the implementation**

Dans `native_engine/src/bin/selfplay_worker.rs`, ajouter la nouvelle constante juste après `TEMPERATURE_PLIES` :

```rust
/// Nombre de feuilles regroupées en un seul appel réseau par vague de
/// sélection MCTS (`Mcts::run`, native_engine/src/mcts.rs) — réduit le
/// nombre d'allers-retours avec le serveur d'inférence par coup joué (à
/// `simulations_per_move = 32`, 4 vagues au lieu de 32 appels séparés).
/// S'applique au self-play (`play_one_game`) et à l'arbitrage
/// (`run_single_move`) : le regroupement de feuilles est une technique de
/// performance pure, pas une source de bruit, donc les deux chemins en
/// bénéficient également — contrairement au bruit de Dirichlet et à la
/// température d'ouverture ci-dessus, qui restent self-play uniquement.
const LEAF_BATCH_SIZE: usize = 8;
```

Dans `play_one_game`, remplacer :
```rust
        mcts.run(simulations_per_move, evaluator);
```
par :
```rust
        mcts.run(simulations_per_move, LEAF_BATCH_SIZE, evaluator);
```

Dans `run_single_move`, remplacer :
```rust
    const BATCH: u32 = 16;
    loop {
        mcts.run(BATCH, evaluator);
        if Instant::now() >= deadline {
            break;
        }
    }
```
par :
```rust
    const BATCH: u32 = 16;
    loop {
        mcts.run(BATCH, LEAF_BATCH_SIZE, evaluator);
        if Instant::now() >= deadline {
            break;
        }
    }
```

- [ ] **Step 4: Rebuild and run tests to verify they pass**

Run: `cd native_engine && cargo build --release --bin selfplay_worker && cargo test --bin selfplay_worker -- --nocapture`
Expected: PASS sur tout le binaire — `socket_evaluator_tests` (y compris `evaluate_decodes_a_multi_item_response_at_the_right_offsets`, qui exerce maintenant aussi indirectement le vrai chemin multi-états via `play_one_game`/`LEAF_BATCH_SIZE` en plus de son usage direct) et `exploration_tests` (déterminisme par graine inchangé : la perte virtuelle n'introduit aucun aléa, seul `Xorshift64` en introduit).

Puis re-exécuter la suite Python qui dépend du binaire reconstruit :

Run: `cd .. && QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_mcts_io_format.py tests/test_selfplay_worker_smoke.py tests/test_selfplay_worker_single_move.py tests/test_selfplay_end_to_end.py -v`
Expected: PASS — aucun de ces tests n'est directement concerné par le regroupement de feuilles (ils vérifient le format sur disque et le protocole, pas le détail interne de `Mcts::run`), mais ils exercent le binaire reconstruit de bout en bout et doivent continuer à passer sans modification.

- [ ] **Step 5: Commit**

```bash
git add native_engine/src/bin/selfplay_worker.rs
git commit -m "Brancher LEAF_BATCH_SIZE=8 dans le self-play et l'arbitrage"
```

---

## Final check (not a task — run after Task 2)

```bash
cd native_engine && cargo test
cd .. && QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_mcts_io_format.py tests/test_selfplay_worker_smoke.py tests/test_selfplay_worker_single_move.py tests/test_selfplay_end_to_end.py tests/test_az_trainer.py tests/test_az_orchestrator.py tests/test_az_inference_server.py tests/test_az_network.py -v
```

Puis, avant de considérer ce levier de débit acquis : mesurer le débit réel self-play (parties/heure, même méthodologie que les mesures précédentes du chantier) avec le binaire reconstruit, et comparer au ~15 200 parties/h mesuré à 128 workers sans regroupement de feuilles. Mettre à jour la mémoire `santoni-mcts-az-no-power-chantier` avec le résultat chiffré. Si un entraînement long tourne déjà au moment de ce déploiement, il faudra l'arrêter proprement, reconstruire le binaire, puis le relancer — le format des parties et des checkpoints sur disque n'est pas affecté par ce plan, donc la reprise se fait normalement (aucune perte de progression).

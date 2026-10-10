# MCTS + réseau de neurones (sous-jeu sans pouvoir) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construire un agent AlphaZero-style (MCTS guidé par un réseau de neurones entraîné par self-play) pour le sous-jeu Santorini sans pouvoir, capable d'être évalué contre le moteur alpha-bêta existant à budget de temps égal.

**Architecture:** Un nouveau module Rust (`mcts.rs`) réutilise directement les fonctions déjà validées `generate`/`apply` du moteur de production (zéro réimplémentation de règles) pour piloter des workers self-play natifs multi-threads. Ces workers parlent à un serveur d'inférence Python/PyTorch par lots via un socket Unix local (protocole binaire fixe, sans dépendance externe côté Rust). Un entraîneur Python consomme les parties écrites atomiquement sur disque et publie des checkpoints ; un orchestrateur gère démarrage/arrêt/reprise de l'ensemble.

**Tech Stack:** Rust (aucune crate externe, cohérent avec `native_engine/src/lib.rs:1-2` "No external crates"), Python 3.10+, PyTorch (nouvelle dépendance), ctypes pour tout pont existant (inchangé), socket Unix (`std::os::unix::net`, `socket` stdlib Python) pour le nouveau pont self-play↔inférence.

**Spec:** `docs/superpowers/specs/2026-10-10-mcts-az-no-power-design.md`

## Global Constraints

- Ne jamais modifier `native_engine/src/lib.rs` au-delà d'une ligne `pub mod mcts;` ni toucher `advanced.rs`/`special.rs` — le moteur de production doit rester l'adversaire de référence intact (spec §3.1, §6).
- Aucune nouvelle crate Rust externe (`[dependencies]` reste vide dans `native_engine/Cargo.toml`) — cohérent avec la philosophie actuelle du moteur.
- Scope strictement limité au sous-jeu sans pouvoir : `powers == (0, 0)`, `counts == [2, 2]` sur les deux joueurs. Toute position hors de ce périmètre doit être rejetée explicitement (panique contrôlée ou erreur), jamais traitée silencieusement.
- Toute écriture de partie ou de checkpoint doit être atomique (fichier temporaire + renommage) — aucune étape ne doit dépendre d'un état mémoire non persisté (spec §4).
- Barre de succès finale : +5pt de taux de victoire contre le moteur de référence à budget de temps égal, IC bootstrap 95% par groupe de placement à borne inférieure strictement positive, sur 200 parties (spec §2) — mêmes corpus/méthode que `tools/hermes_canonicalisation_campaign.py`.
- Réseau de départ volontairement petit (spec §5) : 4 blocs résiduels, 32 canaux — ne pas grossir avant mesure justifiant le besoin.

## Review Focus

- **Position où le joueur au trait n'a aucun coup légal** (base game : défaite immédiate, pas de pat) — `generate()` renvoie une liste vide ; si le code MCTS traite ça comme "nœud sans enfant = valeur neutre" plutôt que "défaite du joueur au trait", l'arbre apprendra une évaluation fausse en silence. Testé en Task 2.
- **Victoire immédiate dès le premier coup généré** (`state.winner != -1` après `apply`) — si le MCTS continue à faire des simulations au-delà d'un nœud déjà terminal au lieu de retourner directement la valeur terminale, le budget de simulations est gaspillé et les statistiques de visite du nœud parent peuvent être biaisées. Testé en Task 2.
- **Position hors périmètre envoyée par erreur au module MCTS** (pouvoir ≠ 0, ou nombre de bâtisseurs ≠ 2) — doit échouer fort et tôt, pas produire un arbre silencieusement incorrect sur une position que `canonical_key` ne traite pas comme prévu. Testé en Task 1.
- **Coupure du worker ou du serveur d'inférence en plein batch** (SIGINT/SIGKILL pendant qu'un lot de positions attend une réponse réseau) — un worker qui attend indéfiniment une réponse jamais envoyée bloquerait toute reprise. Testé en Task 10 (arrêt/redémarrage de l'orchestrateur).
- **Fichier de partie ou checkpoint tronqué par un crash au mauvais moment** (panne électrique entre l'écriture du temporaire et le renommage) — l'orchestrateur doit ignorer/reprendre proprement sans qu'un fichier `.tmp` orphelin soit confondu avec une partie valide. Testé en Task 10.

---

## Task 1 : Primitives de jeu MCTS (réutilisation, pas réimplémentation)

**Files:**
- Modify: `native_engine/src/lib.rs` (ajouter une seule ligne : déclaration du module)
- Create: `native_engine/src/mcts.rs`
- Test: inline `#[cfg(test)]` dans `native_engine/src/mcts.rs`

**Interfaces:**
- Produces : `pub fn is_in_scope(s: &State) -> bool`, `pub fn legal_children(s: &State) -> Vec<(Action, Action, State)>` (paire move+build/dome et état résultant), `pub fn terminal_value(s: &State, to_move: u8) -> Option<i32>` (`Some(1)` victoire de `to_move`, `Some(-1)` défaite, `None` si non terminal), `pub const PLANE_BYTES: usize = 150;`, `pub fn encode_planes(s: &State) -> [u8; PLANE_BYTES]`, `pub const ACTION_SPACE: usize = 1250;`, `pub fn action_index(move_action: &Action, build_action: &Action, worker_slot: usize) -> usize`.

- [ ] **Step 1 : déclarer le module dans lib.rs**

Dans `native_engine/src/lib.rs`, juste après `mod special;` (ligne 7), ajouter :
```rust
pub mod mcts;
```

- [ ] **Step 2 : écrire le test qui échoue (périmètre + réutilisation de `generate`)**

Dans `native_engine/src/mcts.rs` :
```rust
use crate::{apply, generate, Action, State};

pub const PLANE_BYTES: usize = 150;
pub const ACTION_SPACE: usize = 1250;

pub fn is_in_scope(s: &State) -> bool {
    s.powers == [0, 0] && s.counts == [2, 2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::State;

    fn base_state() -> State {
        // Position de départ : bâtisseurs en coins opposés, aucune construction.
        let mut s: State = unsafe { std::mem::zeroed() };
        s.workers = [[0, 4, -1, -1], [20, 24, -1, -1]];
        s.counts = [2, 2];
        s.powers = [0, 0];
        s.winner = -1;
        s
    }

    #[test]
    fn rejects_positions_outside_the_no_power_subgame() {
        let mut s = base_state();
        s.powers = [3, 0];
        assert!(!is_in_scope(&s));
        let mut s2 = base_state();
        s2.counts = [3, 2];
        assert!(!is_in_scope(&s2));
        assert!(is_in_scope(&base_state()));
    }

    #[test]
    fn legal_children_matches_generate_exactly() {
        let s = base_state();
        let via_generate = generate(&s, None, &|| false, usize::MAX).unwrap();
        let via_mcts = legal_children(&s);
        assert_eq!(via_generate.len(), via_mcts.len());
        let mut after_states: Vec<State> = via_generate.iter().map(|t| t.after).collect();
        let mut mcts_after: Vec<State> = via_mcts.iter().map(|(_, _, after)| *after).collect();
        after_states.sort_by_key(|s| format!("{:?}", s));
        mcts_after.sort_by_key(|s| format!("{:?}", s));
        assert_eq!(after_states, mcts_after);
    }
}
```

- [ ] **Step 3 : lancer le test, vérifier qu'il échoue**

Run: `cargo test --manifest-path native_engine/Cargo.toml mcts::tests -- --nocapture`
Expected: FAIL — `legal_children` n'existe pas encore (erreur de compilation).

- [ ] **Step 4 : implémenter `legal_children`, `terminal_value`, `encode_planes`, `action_index`**

Toujours dans `native_engine/src/mcts.rs`, au-dessus du module de test :
```rust
/// Chaque tour du sous-jeu sans pouvoir est exactement [déplacement, construction].
/// Réutilise `generate`/`apply`, déjà différentiel-testés (tests/test_native.py) —
/// aucune règle n'est réimplémentée ici.
pub fn legal_children(s: &State) -> Vec<(Action, Action, State)> {
    debug_assert!(is_in_scope(s));
    generate(s, None, &|| false, usize::MAX)
        .unwrap()
        .into_iter()
        .map(|t| {
            assert_eq!(t.actions.len(), 2, "tour hors périmètre sans-pouvoir");
            (t.actions[0], t.actions[1], t.after)
        })
        .collect()
}

/// `None` si la position n'est pas terminale. `Some(1)` si `to_move` a gagné,
/// `Some(-1)` s'il a perdu — y compris par absence de coup légal (pas de pat
/// dans Santorini de base : qui ne peut pas jouer perd).
pub fn terminal_value(s: &State, to_move: u8) -> Option<i32> {
    if s.winner != -1 {
        return Some(if s.winner as u8 == to_move { 1 } else { -1 });
    }
    if s.player == to_move && legal_children(s).is_empty() {
        return Some(-1);
    }
    None
}

pub fn encode_planes(s: &State) -> [u8; PLANE_BYTES] {
    let mut planes = [0u8; PLANE_BYTES];
    let me = s.player as usize;
    let foe = 1 - me;
    for &c in &s.workers[me] {
        if c >= 0 { planes[c as usize] = 1; }
    }
    for &c in &s.workers[foe] {
        if c >= 0 { planes[25 + c as usize] = 1; }
    }
    for cell in 0..25usize {
        let h = s.heights[cell];
        if h == 1 { planes[50 + cell] = 1; }
        if h == 2 { planes[75 + cell] = 1; }
        if h == 3 && s.domes & (1 << cell) == 0 { planes[100 + cell] = 1; }
        if s.domes & (1 << cell) != 0 { planes[125 + cell] = 1; }
    }
    planes
}

/// Index fixe dans [0, ACTION_SPACE) pour un tour (déplacement puis
/// construction/dôme). `worker_slot` est l'indice (0 ou 1) du bâtisseur
/// déplacé dans `s.workers[s.player]`, pas son identité fixe.
pub fn action_index(move_action: &Action, build_action: &Action, worker_slot: usize) -> usize {
    debug_assert!(worker_slot < 2);
    let dest = move_action.target as usize;
    let build = build_action.target as usize;
    worker_slot * 625 + dest * 25 + build
}
```

- [ ] **Step 5 : lancer le test, vérifier qu'il passe**

Run: `cargo test --manifest-path native_engine/Cargo.toml mcts::tests -- --nocapture`
Expected: PASS (2 tests).

- [ ] **Step 6 : commit**

```bash
git add native_engine/src/lib.rs native_engine/src/mcts.rs
git commit -m "Ajouter les primitives MCTS du sous-jeu sans pouvoir, réutilisant generate/apply"
```

---

## Task 2 : Arbre MCTS (PUCT) avec évaluateur enfichable

**Files:**
- Modify: `native_engine/src/mcts.rs`
- Test: inline `#[cfg(test)]` dans `native_engine/src/mcts.rs`

**Interfaces:**
- Consumes : `legal_children`, `terminal_value`, `action_index`, `ACTION_SPACE`, `State`, `Action` (Task 1).
- Produces : `pub trait LeafEvaluator { fn evaluate(&mut self, states: &[State]) -> Vec<(Vec<f32>, f32)>; }` (retourne, pour chaque état, un vecteur de priors de taille `ACTION_SPACE` et une valeur scalaire dans [-1, 1] du point de vue du joueur au trait de cet état), `pub struct Mcts { .. }`, `pub fn Mcts::new(root: State, c_puct: f32) -> Self`, `pub fn Mcts::run(&mut self, simulations: u32, evaluator: &mut dyn LeafEvaluator) `, `pub fn Mcts::visit_counts(&self) -> Vec<(Action, Action, u32)>` (coups légaux à la racine avec leur nombre de visites, pour construire la cible de politique).

- [ ] **Step 1 : écrire le test qui échoue (victoire immédiate détectée sans gaspiller de simulations, défaite par absence de coup)**

```rust
#[cfg(test)]
mod mcts_tree_tests {
    use super::*;

    struct UniformEvaluator;
    impl LeafEvaluator for UniformEvaluator {
        fn evaluate(&mut self, states: &[State]) -> Vec<(Vec<f32>, f32)> {
            states.iter().map(|_| (vec![1.0 / ACTION_SPACE as f32; ACTION_SPACE], 0.0)).collect()
        }
    }

    fn state_with_immediate_win() -> State {
        let mut s: State = unsafe { std::mem::zeroed() };
        s.powers = [0, 0];
        s.counts = [2, 2];
        s.winner = -1;
        s.workers = [[0, 1, -1, -1], [20, 24, -1, -1]];
        s.heights[6] = 2; // adjacent à la case 0, le bâtisseur 0 peut y monter
        s
    }

    #[test]
    fn root_with_one_legal_move_gets_all_simulations_on_that_move() {
        let s = state_with_immediate_win();
        let mut mcts = Mcts::new(s, 1.5);
        let mut eval = UniformEvaluator;
        mcts.run(50, &mut eval);
        let counts = mcts.visit_counts();
        let total: u32 = counts.iter().map(|(_, _, n)| n).sum();
        assert_eq!(total, 50);
    }

    #[test]
    fn terminal_value_reports_loss_for_player_with_no_legal_move() {
        let mut s: State = unsafe { std::mem::zeroed() };
        s.powers = [0, 0];
        s.counts = [2, 2];
        s.winner = -1;
        // Bâtisseurs du joueur au trait complètement enfermés par des dômes.
        s.workers = [[12, 13, -1, -1], [0, 1, -1, -1]];
        s.domes = !0u32 & !((1 << 12) | (1 << 13));
        s.player = 0;
        assert_eq!(terminal_value(&s, 0), Some(-1));
    }
}
```

- [ ] **Step 2 : lancer les tests, vérifier qu'ils échouent**

Run: `cargo test --manifest-path native_engine/Cargo.toml mcts -- --nocapture`
Expected: FAIL — `Mcts`/`LeafEvaluator` n'existent pas.

- [ ] **Step 3 : implémenter l'arbre PUCT**

```rust
pub trait LeafEvaluator {
    fn evaluate(&mut self, states: &[State]) -> Vec<(Vec<f32>, f32)>;
}

struct Edge { mv: Action, bld: Action, after: State, prior: f32, visits: u32, value_sum: f32 }

struct Node { to_move: u8, edges: Vec<Edge>, expanded: bool }

pub struct Mcts { root: State, c_puct: f32, nodes: std::collections::HashMap<State, Node> }

impl Mcts {
    pub fn new(root: State, c_puct: f32) -> Self {
        Self { root, c_puct, nodes: std::collections::HashMap::new() }
    }

    pub fn run(&mut self, simulations: u32, evaluator: &mut dyn LeafEvaluator) {
        for _ in 0..simulations {
            self.simulate_one(self.root, evaluator);
        }
    }

    fn simulate_one(&mut self, state: State, evaluator: &mut dyn LeafEvaluator) -> f32 {
        let to_move = state.player;
        if let Some(v) = terminal_value(&state, to_move) {
            return v as f32;
        }
        if !self.nodes.contains_key(&state) {
            let children = legal_children(&state);
            let (priors, value) = evaluator.evaluate(&[state]).remove(0);
            let edges = children
                .into_iter()
                .enumerate()
                .map(|(slot, (mv, bld, after))| {
                    let worker_slot = state.workers[to_move as usize]
                        .iter()
                        .position(|&c| c == mv.source)
                        .unwrap_or(slot % 2);
                    let idx = action_index(&mv, &bld, worker_slot);
                    Edge { mv, bld, after, prior: priors[idx], visits: 0, value_sum: 0.0 }
                })
                .collect();
            self.nodes.insert(state, Node { to_move, edges, expanded: true });
            return value;
        }
        let node = self.nodes.get(&state).unwrap();
        let total_visits: u32 = node.edges.iter().map(|e| e.visits).sum::<u32>().max(1);
        let sqrt_total = (total_visits as f32).sqrt();
        let best = node
            .edges
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                let score = |e: &Edge| {
                    let q = if e.visits == 0 { 0.0 } else { e.value_sum / e.visits as f32 };
                    q + self.c_puct * e.prior * sqrt_total / (1.0 + e.visits as f32)
                };
                score(a).partial_cmp(&score(b)).unwrap()
            })
            .map(|(i, _)| i)
            .unwrap();
        let next_state = node.edges[best].after;
        let value = -self.simulate_one(next_state, evaluator);
        let node = self.nodes.get_mut(&state).unwrap();
        node.edges[best].visits += 1;
        node.edges[best].value_sum += value;
        value
    }

    pub fn visit_counts(&self) -> Vec<(Action, Action, u32)> {
        self.nodes[&self.root]
            .edges
            .iter()
            .map(|e| (e.mv, e.bld, e.visits))
            .collect()
    }
}
```

- [ ] **Step 4 : lancer les tests, vérifier qu'ils passent**

Run: `cargo test --manifest-path native_engine/Cargo.toml mcts -- --nocapture`
Expected: PASS (4 tests au total avec Task 1).

- [ ] **Step 5 : commit**

```bash
git add native_engine/src/mcts.rs
git commit -m "Implémenter l'arbre MCTS PUCT avec évaluateur enfichable"
```

---

## Task 3 : Codage/décodage des parties (format JSON à la main, sans crate)

**Files:**
- Create: `native_engine/src/mcts_io.rs`
- Modify: `native_engine/src/lib.rs` (ajouter `pub mod mcts_io;`)
- Test: inline `#[cfg(test)]` dans `native_engine/src/mcts_io.rs`

**Interfaces:**
- Consumes : `State`, `Action` (crate root), `ACTION_SPACE` (Task 1).
- Produces : `pub struct GameRecord { pub moves: Vec<(usize, u32)>, pub outcome: i8 }` (par ply : index d'action joué dans `[0, ACTION_SPACE)`, nombre de visites total à la racine ce tour-là pour reconstruire la politique cible ; `outcome` du point de vue du joueur 0, -1/0/1), `pub fn write_game_atomically(path: &std::path::Path, record: &GameRecord) -> std::io::Result<()>`.

Le format est écrit à la main (pas de `serde_json`) car il n'y a que des entiers à sérialiser — cohérent avec l'absence de dépendance externe du moteur. La lecture se fait côté Python avec le module `json` standard (le texte produit est du JSON valide).

- [ ] **Step 1 : écrire le test qui échoue**

```rust
use crate::State;
use std::io::Write;
use std::path::Path;

pub struct GameRecord {
    pub moves: Vec<(usize, u32)>,
    pub outcome: i8,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn writes_valid_json_and_no_tmp_file_survives() {
        let dir = std::env::temp_dir().join("santoni_mcts_io_test");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("game-0001.json");
        let record = GameRecord { moves: vec![(42, 100), (7, 88)], outcome: 1 };
        write_game_atomically(&path, &record).unwrap();
        let text = fs::read_to_string(&path).unwrap();
        let parsed: serde_json_free_check = serde_json_free_check::parse(&text);
        assert_eq!(parsed.outcome, 1);
        assert_eq!(parsed.moves, vec![(42, 100), (7, 88)]);
        assert!(!path.with_extension("json.tmp").exists());
        fs::remove_dir_all(&dir).unwrap();
    }
}
```

Note pour l'étape suivante : `serde_json_free_check` n'est pas une vraie crate — c'est un
petit parseur JSON minimal écrit dans le bloc de test lui-même (voir Step 3), pour
vérifier la sortie sans dépendre de `serde_json` dans le moteur de production. Le
remplacer dans le Step 3 par l'implémentation réelle ci-dessous avant de lancer le test.

- [ ] **Step 2 : lancer le test, vérifier qu'il échoue**

Run: `cargo test --manifest-path native_engine/Cargo.toml mcts_io -- --nocapture`
Expected: FAIL — ni `write_game_atomically` ni le mini-parseur de test n'existent.

- [ ] **Step 3 : implémenter l'écriture atomique et le mini-parseur de test**

Remplacer le corps du test par une vérification textuelle directe (pas besoin d'un
vrai parseur JSON, un `assert!` sur le texte produit suffit et reste sans dépendance) :

```rust
pub fn write_game_atomically(path: &Path, record: &GameRecord) -> std::io::Result<()> {
    let tmp = path.with_extension("json.tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        let moves_json: Vec<String> = record
            .moves
            .iter()
            .map(|(idx, visits)| format!("[{idx},{visits}]"))
            .collect();
        write!(
            f,
            "{{\"outcome\":{},\"moves\":[{}]}}\n",
            record.outcome,
            moves_json.join(",")
        )?;
        f.flush()?;
    }
    std::fs::rename(&tmp, path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn writes_valid_json_and_no_tmp_file_survives() {
        let dir = std::env::temp_dir().join("santoni_mcts_io_test");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("game-0001.json");
        let record = GameRecord { moves: vec![(42, 100), (7, 88)], outcome: 1 };
        write_game_atomically(&path, &record).unwrap();
        let text = fs::read_to_string(&path).unwrap();
        assert_eq!(text, "{\"outcome\":1,\"moves\":[[42,100],[7,88]]}\n");
        assert!(!path.with_extension("json.tmp").exists());
        fs::remove_dir_all(&dir).unwrap();
    }
}
```

Dans `native_engine/src/lib.rs`, ajouter après `pub mod mcts;` :
```rust
pub mod mcts_io;
```

- [ ] **Step 4 : lancer le test, vérifier qu'il passe**

Run: `cargo test --manifest-path native_engine/Cargo.toml mcts_io -- --nocapture`
Expected: PASS.

- [ ] **Step 5 : écrire un test Python qui confirme que le JSON produit est bien parseable par `json.loads`**

Create `tests/test_mcts_io_format.py` :
```python
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_rust_written_game_is_valid_json(tmp_path):
    sample = tmp_path / "game-0001.json"
    sample.write_text('{"outcome":1,"moves":[[42,100],[7,88]]}\n')
    data = json.loads(sample.read_text())
    assert data["outcome"] == 1
    assert data["moves"] == [[42, 100], [7, 88]]
```

(Ce test fige le format texte attendu côté Python ; les tâches suivantes qui lisent
de vraies parties écrites par le worker Rust s'appuient sur ce même format.)

- [ ] **Step 6 : lancer le test Python**

Run: `QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_mcts_io_format.py -v`
Expected: PASS.

- [ ] **Step 7 : commit**

```bash
git add native_engine/src/lib.rs native_engine/src/mcts_io.rs tests/test_mcts_io_format.py
git commit -m "Écrire les parties self-play en JSON minimal, atomiquement, sans dépendance"
```

---

## Task 4 : Worker self-play autonome (sans réseau — évaluateur local jetable)

**Files:**
- Create: `native_engine/src/bin/selfplay_worker.rs`
- Modify: `native_engine/Cargo.toml` (ajouter la section `[[bin]]`)
- Test: `tests/test_selfplay_worker_smoke.py`

**Interfaces:**
- Consumes : `santoni_engine::mcts::{Mcts, LeafEvaluator, legal_children, terminal_value, is_in_scope}` (Task 1-2), `santoni_engine::mcts_io::{GameRecord, write_game_atomically}` (Task 3).
- Produces : binaire `native_engine/target/release/selfplay_worker` qui accepte `--games N --out-dir PATH --simulations-per-move K` et produit `N` fichiers `game-XXXX.json` complets et valides, en s'arrêtant proprement sur SIGINT sans écrire de fichier tronqué.

Cette tâche prouve la boucle complète partie→MCTS→écriture avant d'introduire le réseau et l'IPC — un évaluateur local trivial (politique uniforme + valeur nulle, identique à `UniformEvaluator` du Task 2) tient lieu de réseau temporaire, remplacé au Task 7.

- [ ] **Step 1 : ajouter la cible binaire à Cargo.toml**

Dans `native_engine/Cargo.toml`, ajouter :
```toml
[[bin]]
name = "selfplay_worker"
path = "src/bin/selfplay_worker.rs"
```

- [ ] **Step 2 : écrire le smoke test Python qui échoue**

Create `tests/test_selfplay_worker_smoke.py` :
```python
import json
import subprocess
import signal
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "native_engine" / "target" / "release" / "selfplay_worker"


def test_worker_produces_complete_valid_games(tmp_path):
    out_dir = tmp_path / "games"
    out_dir.mkdir()
    result = subprocess.run(
        [str(BINARY), "--games", "3", "--out-dir", str(out_dir), "--simulations-per-move", "8"],
        timeout=30, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    files = sorted(out_dir.glob("game-*.json"))
    assert len(files) == 3
    for f in files:
        data = json.loads(f.read_text())
        assert data["outcome"] in (-1, 1)
        assert len(data["moves"]) > 0
    assert not list(out_dir.glob("*.tmp"))


def test_worker_leaves_no_partial_file_on_sigint(tmp_path):
    out_dir = tmp_path / "games"
    out_dir.mkdir()
    proc = subprocess.Popen(
        [str(BINARY), "--games", "10000", "--out-dir", str(out_dir), "--simulations-per-move", "4"],
    )
    time.sleep(0.3)
    proc.send_signal(signal.SIGINT)
    proc.wait(timeout=10)
    assert not list(out_dir.glob("*.tmp"))
    for f in out_dir.glob("game-*.json"):
        json.loads(f.read_text())  # ne doit jamais lever
```

- [ ] **Step 3 : lancer le test, vérifier qu'il échoue**

Run: `cargo build --release --manifest-path native_engine/Cargo.toml && QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_selfplay_worker_smoke.py -v`
Expected: FAIL — le binaire n'existe pas / ne compile pas encore.

- [ ] **Step 4 : implémenter le worker**

```rust
use santoni_engine::mcts::{legal_children, terminal_value, Mcts, LeafEvaluator, ACTION_SPACE};
use santoni_engine::mcts_io::{write_game_atomically, GameRecord};
use santoni_engine::State;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

struct UniformEvaluator;
impl LeafEvaluator for UniformEvaluator {
    fn evaluate(&mut self, states: &[State]) -> Vec<(Vec<f32>, f32)> {
        states.iter().map(|_| (vec![1.0 / ACTION_SPACE as f32; ACTION_SPACE], 0.0)).collect()
    }
}

fn parse_arg(args: &[String], flag: &str) -> String {
    let i = args.iter().position(|a| a == flag).expect("argument manquant");
    args[i + 1].clone()
}

fn initial_state() -> State {
    let mut s: State = unsafe { std::mem::zeroed() };
    s.powers = [0, 0];
    s.counts = [2, 2];
    s.winner = -1;
    s.workers = [[0, 4, -1, -1], [20, 24, -1, -1]];
    s
}

fn play_one_game(simulations_per_move: u32, evaluator: &mut dyn LeafEvaluator) -> GameRecord {
    let mut state = initial_state();
    let mut moves = Vec::new();
    loop {
        if let Some(v) = terminal_value(&state, 0) {
            return GameRecord { moves, outcome: v as i8 };
        }
        let mut mcts = Mcts::new(state, 1.5);
        mcts.run(simulations_per_move, evaluator);
        let counts = mcts.visit_counts();
        let total: u32 = counts.iter().map(|(_, _, n)| n).sum();
        let (mv, bld, after) = {
            let legal = legal_children(&state);
            let best = counts.iter().enumerate().max_by_key(|(_, (_, _, n))| *n).unwrap().0;
            legal[best].clone()
        };
        let worker_slot = state.workers[state.player as usize]
            .iter()
            .position(|&c| c == mv.source)
            .unwrap();
        moves.push((santoni_engine::mcts::action_index(&mv, &bld, worker_slot), total));
        state = after;
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let games: u32 = parse_arg(&args, "--games").parse().unwrap();
    let out_dir = PathBuf::from(parse_arg(&args, "--out-dir"));
    let simulations_per_move: u32 = parse_arg(&args, "--simulations-per-move").parse().unwrap();

    let interrupted = Arc::new(AtomicBool::new(false));
    {
        let flag = interrupted.clone();
        ctrlc_handler(move || flag.store(true, Ordering::SeqCst));
    }

    let mut evaluator = UniformEvaluator;
    for i in 0..games {
        if interrupted.load(Ordering::SeqCst) {
            break;
        }
        let record = play_one_game(simulations_per_move, &mut evaluator);
        let path = out_dir.join(format!("game-{:04}.json", i));
        write_game_atomically(&path, &record).expect("écriture de partie impossible");
    }
}

/// Installe un gestionnaire SIGINT sans dépendance externe : positionne un
/// drapeau vérifié entre deux parties, ne tente jamais d'interrompre une
/// écriture de fichier en cours (déjà atomique par renommage).
fn ctrlc_handler<F: Fn() + Send + 'static>(callback: F) {
    static mut CALLBACK: Option<Box<dyn Fn() + Send>> = None;
    unsafe {
        CALLBACK = Some(Box::new(callback));
        extern "C" fn handler(_: i32) {
            unsafe {
                if let Some(cb) = CALLBACK.as_ref() {
                    cb();
                }
            }
        }
        libc_signal(2, handler as usize); // SIGINT == 2 sur Linux
    }
}

extern "C" {
    #[link_name = "signal"]
    fn libc_signal(signum: i32, handler: usize) -> usize;
}
```

- [ ] **Step 5 : compiler et lancer le test, vérifier qu'il passe**

Run: `cargo build --release --manifest-path native_engine/Cargo.toml && QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_selfplay_worker_smoke.py -v`
Expected: PASS (2 tests).

Si `libc_signal` ne lie pas (le binaire n'est pas certain d'être linké contre `libc`
par défaut selon la plateforme) : lier explicitement avec `#[link(name = "c")]`
au-dessus du bloc `extern "C"`. Vérifier avec `ldd target/release/selfplay_worker`
que `libc.so` apparaît.

- [ ] **Step 6 : commit**

```bash
git add native_engine/Cargo.toml native_engine/src/bin/selfplay_worker.rs tests/test_selfplay_worker_smoke.py
git commit -m "Worker self-play autonome bout-en-bout avec évaluateur local jetable"
```

---

## Task 5 : Réseau PyTorch (policy + value)

**Files:**
- Create: `santoni_az/__init__.py`
- Create: `santoni_az/network.py`
- Modify: `pyproject.toml` (ajouter la dépendance `torch`)
- Test: `tests/test_az_network.py`

**Interfaces:**
- Produces : `class PolicyValueNet(torch.nn.Module)` avec `forward(self, planes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]` où `planes` a la forme `(batch, 6, 5, 5)` (6 = nombre de plans définis par `PLANE_BYTES=150=6*25` au Task 1), sortie policy de forme `(batch, 1250)` (logits, `ACTION_SPACE`), sortie value de forme `(batch, 1)` dans `[-1, 1]` (via `tanh`).

- [ ] **Step 1 : ajouter la dépendance**

Dans `pyproject.toml`, section `[project]`, remplacer :
```toml
dependencies = ["PySide6>=6.5"]
```
par :
```toml
dependencies = ["PySide6>=6.5", "torch>=2.2"]
```

- [ ] **Step 2 : créer le package et écrire le test qui échoue**

Create `santoni_az/__init__.py` (vide).

Create `tests/test_az_network.py` :
```python
import torch

from santoni_az.network import PolicyValueNet, ACTION_SPACE, NUM_PLANES


def test_forward_pass_shapes_and_value_range():
    net = PolicyValueNet(channels=32, blocks=4)
    batch = torch.zeros(3, NUM_PLANES, 5, 5)
    policy, value = net(batch)
    assert policy.shape == (3, ACTION_SPACE)
    assert value.shape == (3, 1)
    assert torch.all(value >= -1) and torch.all(value <= 1)
```

- [ ] **Step 3 : lancer le test, vérifier qu'il échoue**

Run: `QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_az_network.py -v`
Expected: FAIL — `santoni_az.network` n'existe pas.

- [ ] **Step 4 : implémenter le réseau**

Create `santoni_az/network.py` :
```python
import torch
import torch.nn as nn

NUM_PLANES = 6
ACTION_SPACE = 1250


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return torch.relu(x + residual)


class PolicyValueNet(nn.Module):
    def __init__(self, channels=32, blocks=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(NUM_PLANES, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.tower = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 8, 1),
            nn.Flatten(),
            nn.Linear(8 * 5 * 5, ACTION_SPACE),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 4, 1),
            nn.Flatten(),
            nn.Linear(4 * 5 * 5, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )

    def forward(self, planes):
        x = self.stem(planes)
        x = self.tower(x)
        return self.policy_head(x), self.value_head(x)
```

- [ ] **Step 5 : lancer le test, vérifier qu'il passe**

Run: `QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_az_network.py -v`
Expected: PASS.

- [ ] **Step 6 : commit**

```bash
git add pyproject.toml santoni_az/__init__.py santoni_az/network.py tests/test_az_network.py
git commit -m "Ajouter le réseau policy+value PyTorch (4 blocs résiduels, 32 canaux)"
```

---

## Task 6 : Protocole d'inférence par lots sur socket Unix (mock avant réseau réel)

**Files:**
- Create: `santoni_az/inference_protocol.py`
- Test: `tests/test_az_inference_protocol.py`

**Interfaces:**
- Produces : `HEADER_FORMAT` (struct format pour `count: u32`), `STATE_FORMAT` (pour un seul état : `PLANE_BYTES=150` octets), `RESPONSE_ITEM_FORMAT` (`ACTION_SPACE` floats policy + 1 float value), `def pack_request(planes_batch: list[bytes]) -> bytes`, `def unpack_request(data: bytes) -> list[bytes]`, `def pack_response(policies: list[list[float]], values: list[float]) -> bytes`, `def unpack_response(data: bytes, batch_size: int) -> tuple[list[list[float]], list[float]]`.

Ce protocole binaire fixe (pas de JSON) est la frontière Rust↔Python la plus chaude du pipeline (un appel par nœud MCTS non mis en cache) — testé côté Python seul ici en aller-retour pur ; Task 7 le câble réellement au worker Rust via un vrai socket, Task 8 au serveur PyTorch réel.

- [ ] **Step 1 : écrire le test qui échoue**

```python
import struct

from santoni_az.inference_protocol import (
    pack_request, unpack_request, pack_response, unpack_response, PLANE_BYTES, ACTION_SPACE,
)


def test_request_roundtrip():
    batch = [bytes([1, 0] * (PLANE_BYTES // 2)), bytes([0, 1] * (PLANE_BYTES // 2))]
    packed = pack_request(batch)
    unpacked = unpack_request(packed)
    assert unpacked == batch


def test_response_roundtrip():
    policies = [[0.001] * ACTION_SPACE, [0.002] * ACTION_SPACE]
    values = [0.5, -0.3]
    packed = pack_response(policies, values)
    out_policies, out_values = unpack_response(packed, batch_size=2)
    assert out_values == [pytest_approx(0.5), pytest_approx(-0.3)]
    assert len(out_policies) == 2 and len(out_policies[0]) == ACTION_SPACE


def pytest_approx(x):
    import pytest
    return pytest.approx(x, abs=1e-5)
```

- [ ] **Step 2 : lancer le test, vérifier qu'il échoue**

Run: `QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_az_inference_protocol.py -v`
Expected: FAIL — le module n'existe pas.

- [ ] **Step 3 : implémenter le protocole**

Create `santoni_az/inference_protocol.py` :
```python
import struct

PLANE_BYTES = 150
ACTION_SPACE = 1250


def pack_request(planes_batch: list[bytes]) -> bytes:
    header = struct.pack("<I", len(planes_batch))
    body = b"".join(planes_batch)
    return header + body


def unpack_request(data: bytes) -> list[bytes]:
    (count,) = struct.unpack_from("<I", data, 0)
    offset = 4
    result = []
    for _ in range(count):
        result.append(data[offset:offset + PLANE_BYTES])
        offset += PLANE_BYTES
    return result


def pack_response(policies: list[list[float]], values: list[float]) -> bytes:
    parts = [struct.pack("<I", len(values))]
    for policy, value in zip(policies, values):
        parts.append(struct.pack(f"<{ACTION_SPACE}f", *policy))
        parts.append(struct.pack("<f", value))
    return b"".join(parts)


def unpack_response(data: bytes, batch_size: int) -> tuple[list[list[float]], list[float]]:
    (count,) = struct.unpack_from("<I", data, 0)
    assert count == batch_size
    offset = 4
    policies, values = [], []
    item_size = ACTION_SPACE * 4
    for _ in range(count):
        policies.append(list(struct.unpack_from(f"<{ACTION_SPACE}f", data, offset)))
        offset += item_size
        (value,) = struct.unpack_from("<f", data, offset)
        values.append(value)
        offset += 4
    return policies, values
```

- [ ] **Step 4 : lancer le test, vérifier qu'il passe**

Run: `QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_az_inference_protocol.py -v`
Expected: PASS.

- [ ] **Step 5 : commit**

```bash
git add santoni_az/inference_protocol.py tests/test_az_inference_protocol.py
git commit -m "Définir le protocole binaire fixe pour les lots d'inférence MCTS"
```

---

## Task 7 : Serveur d'inférence réel (socket Unix + PyTorch + lot adaptatif)

**Files:**
- Create: `santoni_az/inference_server.py`
- Test: `tests/test_az_inference_server.py`

**Interfaces:**
- Consumes : `santoni_az.network.PolicyValueNet` (Task 5), `santoni_az.inference_protocol.*` (Task 6).
- Produces : `class InferenceServer` avec `__init__(self, socket_path: str, net: PolicyValueNet, flush_interval_s: float = 0.02)`, `def serve_forever(self) -> None`, `def stop(self) -> None` (ferme le socket d'écoute, laisse les connexions en cours se terminer).

Le serveur accepte plusieurs connexions clientes (une par worker Rust), accumule les requêtes reçues pendant `flush_interval_s`, fait un seul passage réseau sur le lot accumulé, puis répond à chaque client. Tourne sur CPU dans ce test (pas de GPU requis pour la correction fonctionnelle) ; Task 12 (Step 3, exécution réelle multi-workers) mesure l'utilisation GPU effective en conditions de charge — c'est le diagnostic que le pilote jetable avait fait manuellement (débit plat en multipliant les parties parallèles), à reproduire ici avec la vraie architecture pour confirmer que le goulot identifié est bien résolu.

- [ ] **Step 1 : écrire le test qui échoue (deux clients concurrents, un seul passage réseau)**

```python
import socket
import threading
import time

import torch

from santoni_az.network import PolicyValueNet, NUM_PLANES
from santoni_az.inference_protocol import pack_request, unpack_response, PLANE_BYTES, ACTION_SPACE
from santoni_az.inference_server import InferenceServer


def test_two_concurrent_clients_get_correct_shaped_responses(tmp_path):
    socket_path = str(tmp_path / "infer.sock")
    net = PolicyValueNet(channels=8, blocks=1)
    server = InferenceServer(socket_path, net, flush_interval_s=0.05)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)

    def client(results, index):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(socket_path)
        planes = bytes(PLANE_BYTES)
        sock.sendall(pack_request([planes]))
        header = sock.recv(4)
        import struct
        (count,) = struct.unpack("<I", header)
        body = b""
        needed = count * (ACTION_SPACE * 4 + 4)
        while len(body) < needed:
            body += sock.recv(needed - len(body))
        policies, values = unpack_response(header + body, batch_size=count)
        results[index] = (policies, values)
        sock.close()

    results = {}
    t1 = threading.Thread(target=client, args=(results, 0))
    t2 = threading.Thread(target=client, args=(results, 1))
    t1.start(); t2.start()
    t1.join(timeout=5); t2.join(timeout=5)
    server.stop()

    assert len(results) == 2
    for policies, values in results.values():
        assert len(policies[0]) == ACTION_SPACE
        assert -1 <= values[0] <= 1
```

- [ ] **Step 2 : lancer le test, vérifier qu'il échoue**

Run: `QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_az_inference_server.py -v`
Expected: FAIL — `santoni_az.inference_server` n'existe pas.

- [ ] **Step 3 : implémenter le serveur**

Create `santoni_az/inference_server.py` :
```python
import os
import socket
import struct
import threading
import time

import torch

from .inference_protocol import pack_response, unpack_request, PLANE_BYTES, ACTION_SPACE
from .network import NUM_PLANES


class InferenceServer:
    def __init__(self, socket_path, net, flush_interval_s=0.02):
        self.socket_path = socket_path
        self.net = net
        self.net.eval()
        self.flush_interval_s = flush_interval_s
        self._lock = threading.Lock()
        self._pending = []  # list of (planes_bytes, connection)
        self._running = True
        if os.path.exists(socket_path):
            os.remove(socket_path)
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._listener.bind(socket_path)
        self._listener.listen(64)
        self._listener.settimeout(0.1)

    def stop(self):
        self._running = False
        self._listener.close()

    def _accept_loop(self):
        while self._running:
            try:
                conn, _ = self._listener.accept()
            except (socket.timeout, OSError):
                continue
            threading.Thread(target=self._client_loop, args=(conn,), daemon=True).start()

    def _client_loop(self, conn):
        header = conn.recv(4)
        if len(header) < 4:
            conn.close()
            return
        (count,) = struct.unpack("<I", header)
        body = b""
        while len(body) < count * PLANE_BYTES:
            body += conn.recv(count * PLANE_BYTES - len(body))
        planes_list = unpack_request(header + body)
        with self._lock:
            self._pending.append((planes_list, conn))

    def _flush_loop(self):
        while self._running:
            time.sleep(self.flush_interval_s)
            with self._lock:
                batch, self._pending = self._pending, []
            if not batch:
                continue
            all_planes = [p for planes_list, _ in batch for p in planes_list]
            tensor = torch.stack([
                torch.frombuffer(bytearray(p), dtype=torch.uint8).reshape(6, 5, 5).float()
                for p in all_planes
            ])
            with torch.no_grad():
                policy_logits, values = self.net(tensor)
            policies = policy_logits.tolist()
            values = [v[0] for v in values.tolist()]
            offset = 0
            for planes_list, conn in batch:
                n = len(planes_list)
                response = pack_response(policies[offset:offset + n], values[offset:offset + n])
                try:
                    conn.sendall(response)
                finally:
                    conn.close()
                offset += n

    def serve_forever(self):
        flusher = threading.Thread(target=self._flush_loop, daemon=True)
        flusher.start()
        self._accept_loop()
```

- [ ] **Step 4 : lancer le test, vérifier qu'il passe**

Run: `QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_az_inference_server.py -v`
Expected: PASS.

Note : `torch.frombuffer` exige un buffer accessible en écriture dans certaines
versions de PyTorch — si le test échoue avec une erreur liée à ça, remplacer par
`torch.tensor(list(p), dtype=torch.uint8)`.

- [ ] **Step 5 : commit**

```bash
git add santoni_az/inference_server.py tests/test_az_inference_server.py
git commit -m "Implémenter le serveur d'inférence PyTorch par lots sur socket Unix"
```

---

## Task 8 : Câbler le worker Rust au vrai serveur d'inférence (bout-en-bout, 1 génération)

**Files:**
- Modify: `native_engine/src/bin/selfplay_worker.rs`
- Test: `tests/test_selfplay_end_to_end.py`

**Interfaces:**
- Consumes : socket Unix du Task 7 (côté Python déjà testé), protocole binaire du Task 6 (déjà testé côté Python).
- Produces : le worker accepte un nouvel argument `--inference-socket PATH` ; quand fourni, remplace `UniformEvaluator` par un évaluateur qui envoie les positions au socket et bloque sur la réponse.

- [ ] **Step 1 : écrire le test d'intégration qui échoue**

Create `tests/test_selfplay_end_to_end.py` :
```python
import json
import subprocess
import threading
import time
from pathlib import Path

import torch

from santoni_az.network import PolicyValueNet
from santoni_az.inference_server import InferenceServer

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "native_engine" / "target" / "release" / "selfplay_worker"


def test_worker_produces_games_via_real_inference_server(tmp_path):
    socket_path = str(tmp_path / "infer.sock")
    out_dir = tmp_path / "games"
    out_dir.mkdir()
    net = PolicyValueNet(channels=8, blocks=1)
    server = InferenceServer(socket_path, net, flush_interval_s=0.02)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)

    result = subprocess.run(
        [str(BINARY), "--games", "2", "--out-dir", str(out_dir),
         "--simulations-per-move", "8", "--inference-socket", socket_path],
        timeout=60, capture_output=True, text=True,
    )
    server.stop()
    assert result.returncode == 0, result.stderr
    files = sorted(out_dir.glob("game-*.json"))
    assert len(files) == 2
    for f in files:
        data = json.loads(f.read_text())
        assert data["outcome"] in (-1, 1)
```

- [ ] **Step 2 : lancer le test, vérifier qu'il échoue**

Run: `QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_selfplay_end_to_end.py -v`
Expected: FAIL — `--inference-socket` n'est pas reconnu par le worker actuel.

- [ ] **Step 3 : implémenter l'évaluateur réseau côté Rust**

Ajouter dans `native_engine/src/bin/selfplay_worker.rs`, au-dessus de `fn main()` :
```rust
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;

const PLANE_BYTES: usize = 150;
const ACTION_SPACE: usize = 1250;

struct SocketEvaluator { stream: UnixStream }

impl SocketEvaluator {
    fn connect(path: &str) -> Self {
        Self { stream: UnixStream::connect(path).expect("connexion au serveur d'inférence impossible") }
    }
}

impl LeafEvaluator for SocketEvaluator {
    fn evaluate(&mut self, states: &[State]) -> Vec<(Vec<f32>, f32)> {
        let count = states.len() as u32;
        let mut request = count.to_le_bytes().to_vec();
        for s in states {
            request.extend_from_slice(&santoni_engine::mcts::encode_planes(s));
        }
        self.stream.write_all(&request).expect("écriture socket impossible");
        let mut header = [0u8; 4];
        self.stream.read_exact(&mut header).expect("lecture entête réponse impossible");
        let n = u32::from_le_bytes(header) as usize;
        let item_size = ACTION_SPACE * 4 + 4;
        let mut body = vec![0u8; n * item_size];
        self.stream.read_exact(&mut body).expect("lecture corps réponse impossible");
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let base = i * item_size;
            let mut policy = Vec::with_capacity(ACTION_SPACE);
            for j in 0..ACTION_SPACE {
                let o = base + j * 4;
                policy.push(f32::from_le_bytes([body[o], body[o+1], body[o+2], body[o+3]]));
            }
            let vo = base + ACTION_SPACE * 4;
            let value = f32::from_le_bytes([body[vo], body[vo+1], body[vo+2], body[vo+3]]);
            result.push((policy, value));
        }
        result
    }
}
```

Modifier `fn main()` pour lire `--inference-socket` (optionnel) et choisir l'évaluateur :
```rust
    let socket_path = args.iter().position(|a| a == "--inference-socket").map(|i| args[i + 1].clone());
    let mut evaluator: Box<dyn LeafEvaluator> = match socket_path {
        Some(path) => Box::new(SocketEvaluator::connect(&path)),
        None => Box::new(UniformEvaluator),
    };
    for i in 0..games {
        if interrupted.load(Ordering::SeqCst) { break; }
        let record = play_one_game(simulations_per_move, evaluator.as_mut());
        let path = out_dir.join(format!("game-{:04}.json", i));
        write_game_atomically(&path, &record).expect("écriture de partie impossible");
    }
```
(`play_one_game` prend déjà `&mut dyn LeafEvaluator` depuis Task 4, aucun changement de signature requis.)

- [ ] **Step 4 : compiler et lancer le test, vérifier qu'il passe**

Run: `cargo build --release --manifest-path native_engine/Cargo.toml && QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_selfplay_end_to_end.py -v`
Expected: PASS.

- [ ] **Step 5 : vérifier que les anciens tests du worker (sans socket) passent toujours**

Run: `QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_selfplay_worker_smoke.py -v`
Expected: PASS (le mode sans `--inference-socket` reste l'évaluateur uniforme).

- [ ] **Step 6 : commit**

```bash
git add native_engine/src/bin/selfplay_worker.rs tests/test_selfplay_end_to_end.py
git commit -m "Câbler le worker self-play au serveur d'inférence réel via socket Unix"
```

---

## Task 9 : Entraîneur avec checkpoints atomiques et reprise

**Files:**
- Create: `santoni_az/trainer.py`
- Test: `tests/test_az_trainer.py`

**Interfaces:**
- Consumes : `santoni_az.network.PolicyValueNet` (Task 5), format de partie JSON écrit par le worker (Task 3/4).
- Produces : `class Trainer` avec `__init__(self, games_dir: str, checkpoint_dir: str, net_factory=lambda: PolicyValueNet())`, `def consume_new_games(self) -> int` (retourne le nombre de nouvelles parties consommées depuis le dernier appel), `def train_step(self, batch_size: int = 64) -> float` (un pas de gradient, retourne la perte), `def save_checkpoint(self) -> str` (écrit atomiquement, retourne le chemin), `def load_latest_checkpoint(self) -> int` (retourne le numéro de génération chargé, 0 si aucun).

- [ ] **Step 1 : écrire le test qui échoue (reprise après "crash")**

```python
import json
from pathlib import Path

from santoni_az.trainer import Trainer


def write_fake_game(games_dir, name, outcome, moves):
    (games_dir / name).write_text(json.dumps({"outcome": outcome, "moves": moves}) + "\n")


def test_checkpoint_resume_preserves_generation_counter(tmp_path):
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    write_fake_game(games_dir, "game-0000.json", 1, [[10, 20], [5, 15]])
    write_fake_game(games_dir, "game-0001.json", -1, [[3, 10]])

    trainer = Trainer(str(games_dir), str(checkpoint_dir))
    assert trainer.consume_new_games() == 2
    trainer.train_step(batch_size=4)
    path = trainer.save_checkpoint()
    assert Path(path).exists()
    assert not Path(path + ".tmp").exists()

    resumed = Trainer(str(games_dir), str(checkpoint_dir))
    generation = resumed.load_latest_checkpoint()
    assert generation == 1
    assert resumed.consume_new_games() == 0  # mêmes parties, déjà consommées avant le crash simulé
```

- [ ] **Step 2 : lancer le test, vérifier qu'il échoue**

Run: `QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_az_trainer.py -v`
Expected: FAIL — `santoni_az.trainer` n'existe pas.

- [ ] **Step 3 : implémenter l'entraîneur**

Create `santoni_az/trainer.py` :
```python
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from .network import PolicyValueNet, ACTION_SPACE, NUM_PLANES


class Trainer:
    def __init__(self, games_dir, checkpoint_dir, net_factory=lambda: PolicyValueNet()):
        self.games_dir = Path(games_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.net = net_factory()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.generation = 0
        self._consumed_games = set()
        self._examples = []  # (planes_placeholder, policy_target, value_target)

    def consume_new_games(self):
        new_files = sorted(
            f for f in self.games_dir.glob("game-*.json") if f.name not in self._consumed_games
        )
        for f in new_files:
            data = json.loads(f.read_text())
            outcome = data["outcome"]
            for action_idx, visits in data["moves"]:
                policy_target = torch.zeros(ACTION_SPACE)
                policy_target[action_idx] = 1.0  # cible simplifiée : un seul coup joué
                self._examples.append((torch.zeros(NUM_PLANES, 5, 5), policy_target, float(outcome)))
            self._consumed_games.add(f.name)
        return len(new_files)

    def train_step(self, batch_size=64):
        if not self._examples:
            return 0.0
        batch = self._examples[:batch_size]
        planes = torch.stack([p for p, _, _ in batch])
        policy_targets = torch.stack([pt for _, pt, _ in batch])
        value_targets = torch.tensor([vt for _, _, vt in batch]).unsqueeze(1)
        policy_logits, value_pred = self.net(planes)
        policy_loss = F.cross_entropy(policy_logits, policy_targets.argmax(dim=1))
        value_loss = F.mse_loss(value_pred, value_targets)
        loss = policy_loss + value_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def save_checkpoint(self):
        self.generation += 1
        path = self.checkpoint_dir / f"gen-{self.generation:04d}.pt"
        tmp = path.with_suffix(".pt.tmp")
        torch.save({
            "generation": self.generation,
            "model_state": self.net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "consumed_games": sorted(self._consumed_games),
        }, tmp)
        os.replace(tmp, path)
        return str(path)

    def load_latest_checkpoint(self):
        checkpoints = sorted(self.checkpoint_dir.glob("gen-*.pt"))
        if not checkpoints:
            return 0
        data = torch.load(checkpoints[-1], weights_only=False)
        self.net.load_state_dict(data["model_state"])
        self.optimizer.load_state_dict(data["optimizer_state"])
        self.generation = data["generation"]
        self._consumed_games = set(data["consumed_games"])
        return self.generation
```

- [ ] **Step 4 : lancer le test, vérifier qu'il passe**

Run: `QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_az_trainer.py -v`
Expected: PASS.

Note sur la cible de politique simplifiée (`policy_target[action_idx] = 1.0`,
one-hot sur le coup effectivement joué plutôt qu'une distribution proportionnelle
aux `visits` de chaque coup légal à ce tour) : c'est une simplification délibérée
pour ce plan — suffisante pour prouver le pipeline de reprise, insuffisante pour
l'entraînement final. Reprendre ce point avant la campagne d'évaluation du Task 12
en reconstruisant, côté `consume_new_games`, la distribution complète des `visits`
par coup légal à chaque tour (nécessite de stocker, dans le format de partie, les
visites de *tous* les coups légaux de chaque tour, pas seulement celui joué —
étendre `GameRecord`/`write_game_atomically` du Task 3 en conséquence à ce moment-là).

- [ ] **Step 5 : commit**

```bash
git add santoni_az/trainer.py tests/test_az_trainer.py
git commit -m "Entraîneur avec checkpoints atomiques et reprise par compteur de génération"
```

---

## Task 10 : Orchestrateur (démarrage/arrêt/reprise de l'ensemble du pipeline)

**Files:**
- Create: `santoni_az/orchestrator.py`
- Test: `tests/test_az_orchestrator.py`

**Interfaces:**
- Consumes : `InferenceServer` (Task 7), `Trainer` (Task 9), binaire `selfplay_worker` (Task 8).
- Produces : `class Orchestrator` avec `__init__(self, run_dir: str, worker_count: int = 1)`, `def start(self) -> None` (lance serveur d'inférence + N workers + entraîneur comme sous-processus/threads), `def stop(self) -> None` (signal d'arrêt propre à chaque composant, attend leur fin), `def status(self) -> dict` (génération courante, nombre de parties disponibles, PID des workers vivants).

- [ ] **Step 1 : écrire le test qui échoue**

```python
import time

from santoni_az.orchestrator import Orchestrator


def test_start_stop_leaves_no_partial_files(tmp_path):
    orch = Orchestrator(str(tmp_path), worker_count=1)
    orch.start()
    time.sleep(1.0)
    status_before = orch.status()
    assert status_before["workers_alive"] >= 1
    orch.stop()
    tmp_files = list((tmp_path / "games").glob("*.tmp"))
    assert tmp_files == []


def test_restart_resumes_from_last_checkpoint(tmp_path):
    orch = Orchestrator(str(tmp_path), worker_count=1)
    orch.start()
    time.sleep(1.5)
    orch.stop()
    generation_before = orch.status()["generation"]

    orch2 = Orchestrator(str(tmp_path), worker_count=1)
    orch2.start()
    time.sleep(0.2)
    generation_after = orch2.status()["generation"]
    orch2.stop()
    assert generation_after >= generation_before
```

- [ ] **Step 2 : lancer le test, vérifier qu'il échoue**

Run: `QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_az_orchestrator.py -v`
Expected: FAIL — `santoni_az.orchestrator` n'existe pas.

- [ ] **Step 3 : implémenter l'orchestrateur**

Create `santoni_az/orchestrator.py` :
```python
import json
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

from .inference_server import InferenceServer
from .trainer import Trainer

ROOT = Path(__file__).resolve().parents[1]
WORKER_BINARY = ROOT / "native_engine" / "target" / "release" / "selfplay_worker"


class Orchestrator:
    def __init__(self, run_dir, worker_count=1):
        self.run_dir = Path(run_dir)
        self.games_dir = self.run_dir / "games"
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.games_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.socket_path = str(self.run_dir / "infer.sock")
        self.worker_count = worker_count
        self.trainer = Trainer(str(self.games_dir), str(self.checkpoint_dir))
        self._server = None
        self._server_thread = None
        self._workers = []
        self._train_thread = None
        self._running = False

    def start(self):
        self.trainer.load_latest_checkpoint()
        self._server = InferenceServer(self.socket_path, self.trainer.net)
        self._server_thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._server_thread.start()
        time.sleep(0.1)
        for _ in range(self.worker_count):
            proc = subprocess.Popen([
                str(WORKER_BINARY), "--games", "100000", "--out-dir", str(self.games_dir),
                "--simulations-per-move", "32", "--inference-socket", self.socket_path,
            ])
            self._workers.append(proc)
        self._running = True
        self._train_thread = threading.Thread(target=self._train_loop, daemon=True)
        self._train_thread.start()

    def _train_loop(self):
        while self._running:
            self.trainer.consume_new_games()
            self.trainer.train_step()
            self.trainer.save_checkpoint()
            time.sleep(0.5)

    def stop(self):
        self._running = False
        for proc in self._workers:
            proc.send_signal(signal.SIGINT)
        for proc in self._workers:
            proc.wait(timeout=10)
        if self._server:
            self._server.stop()

    def status(self):
        alive = sum(1 for p in self._workers if p.poll() is None)
        return {
            "generation": self.trainer.generation,
            "games_available": len(list(self.games_dir.glob("game-*.json"))),
            "workers_alive": alive,
        }
```

- [ ] **Step 4 : compiler le worker si nécessaire, puis lancer le test**

Run: `cargo build --release --manifest-path native_engine/Cargo.toml && QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_az_orchestrator.py -v`
Expected: PASS.

- [ ] **Step 5 : commit**

```bash
git add santoni_az/orchestrator.py tests/test_az_orchestrator.py
git commit -m "Orchestrateur: démarrage/arrêt/reprise coordonnés du pipeline self-play+entraînement"
```

---

## Task 11 : Mode "un seul coup" du worker, pour l'arbitrer depuis une campagne Python

**Files:**
- Modify: `native_engine/src/bin/selfplay_worker.rs`
- Test: `tests/test_selfplay_worker_single_move.py`

**Interfaces:**
- Consumes : `santoni_engine::mcts::{Mcts, encode_planes}` (Task 1-2), `SocketEvaluator`/`UniformEvaluator` (Task 4/8), layout binaire `CState`/`CAction` déjà défini côté Python dans `santorini/native.py:20-55` (mêmes structures `#[repr(C)]`, taille déjà vérifiée par `santoni_state_size()`).
- Produces : mode CLI `selfplay_worker --single-move --input-state PATH --output-move PATH --seconds F64 [--inference-socket PATH]` : lit `size_of::<State>()` octets depuis `--input-state` (mêmes octets que produits par `santorini.native.encode(position)`), exécute un MCTS à budget de temps, écrit `size_of::<Action>() * 2` octets (move puis build) dans `--output-move`, mêmes octets que lus par `CAction.from_buffer_copy` côté Python.

Nécessaire pour la Task 12 : la campagne d'évaluation doit arbitrer un coup candidat pour une position quelconque du corpus (pas seulement rejouer des parties depuis l'ouverture comme le fait le self-play continu des Tasks 4/8).

- [ ] **Step 1 : écrire le test qui échoue**

Create `tests/test_selfplay_worker_single_move.py` :
```python
import ctypes as C
import time
from pathlib import Path

from santorini.engine import Position, validate_turn
from santorini.native import encode, CState, CAction

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "native_engine" / "target" / "release" / "selfplay_worker"


def test_single_move_produces_a_legal_turn_within_time_budget(tmp_path):
    position = Position()  # position de départ, aucun pouvoir
    state = encode(position)
    input_path = tmp_path / "state.bin"
    input_path.write_bytes(bytes(state))
    output_path = tmp_path / "move.bin"

    started = time.monotonic()
    import subprocess
    result = subprocess.run(
        [str(BINARY), "--single-move", "--input-state", str(input_path),
         "--output-move", str(output_path), "--seconds", "0.5"],
        timeout=5, capture_output=True, text=True,
    )
    elapsed = time.monotonic() - started
    assert result.returncode == 0, result.stderr
    assert elapsed < 2.0  # budget 0.5s + marge de démarrage du processus

    raw = output_path.read_bytes()
    assert len(raw) == C.sizeof(CAction) * 2
    move_action = CAction.from_buffer_copy(raw, 0)
    build_action = CAction.from_buffer_copy(raw, C.sizeof(CAction))
    from santorini.engine import Action
    actions = (
        Action("move", move_action.player, move_action.worker, move_action.source, move_action.target),
        Action("build", build_action.player, build_action.worker, build_action.source, build_action.target),
    )
    turn = validate_turn(position, actions)  # lève si le coup n'est pas légal
    assert turn.after != position
```

- [ ] **Step 2 : lancer le test, vérifier qu'il échoue**

Run: `cargo build --release --manifest-path native_engine/Cargo.toml && QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_selfplay_worker_single_move.py -v`
Expected: FAIL — `--single-move` n'est pas reconnu.

- [ ] **Step 3 : implémenter le mode dans `selfplay_worker.rs`**

Ajouter, au-dessus de `fn main()` :
```rust
use std::time::{Duration, Instant};

fn run_single_move(input_state: &PathBuf, output_move: &PathBuf, seconds: f64, evaluator: &mut dyn LeafEvaluator) {
    let bytes = std::fs::read(input_state).expect("lecture de l'état d'entrée impossible");
    assert_eq!(bytes.len(), std::mem::size_of::<State>(), "taille d'état incompatible");
    let state: State = unsafe { std::ptr::read(bytes.as_ptr() as *const State) };
    assert!(santoni_engine::mcts::is_in_scope(&state), "position hors périmètre sans-pouvoir");

    let deadline = Instant::now() + Duration::from_secs_f64(seconds);
    let mut mcts = Mcts::new(state, 1.5);
    const BATCH: u32 = 16;
    loop {
        mcts.run(BATCH, evaluator);
        if Instant::now() >= deadline {
            break;
        }
    }
    let counts = mcts.visit_counts();
    let legal = legal_children(&state);
    let best = counts.iter().enumerate().max_by_key(|(_, (_, _, n))| *n).unwrap().0;
    let (mv, bld, _) = legal[best];

    let mut out = Vec::with_capacity(std::mem::size_of::<Action>() * 2);
    out.extend_from_slice(unsafe {
        std::slice::from_raw_parts(&mv as *const Action as *const u8, std::mem::size_of::<Action>())
    });
    out.extend_from_slice(unsafe {
        std::slice::from_raw_parts(&bld as *const Action as *const u8, std::mem::size_of::<Action>())
    });
    let tmp = output_move.with_extension("bin.tmp");
    std::fs::write(&tmp, &out).expect("écriture du coup impossible");
    std::fs::rename(&tmp, output_move).expect("renommage du coup impossible");
}
```

Modifier le début de `fn main()` pour aiguiller vers ce mode :
```rust
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--single-move") {
        let input_state = PathBuf::from(parse_arg(&args, "--input-state"));
        let output_move = PathBuf::from(parse_arg(&args, "--output-move"));
        let seconds: f64 = parse_arg(&args, "--seconds").parse().unwrap();
        let socket_path = args.iter().position(|a| a == "--inference-socket").map(|i| args[i + 1].clone());
        let mut evaluator: Box<dyn LeafEvaluator> = match socket_path {
            Some(path) => Box::new(SocketEvaluator::connect(&path)),
            None => Box::new(UniformEvaluator),
        };
        run_single_move(&input_state, &output_move, seconds, evaluator.as_mut());
        return;
    }
    // ... reste de main() inchangé (mode self-play continu des Tasks 4/8) ...
```

- [ ] **Step 4 : compiler et lancer le test, vérifier qu'il passe**

Run: `cargo build --release --manifest-path native_engine/Cargo.toml && QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_selfplay_worker_single_move.py -v`
Expected: PASS.

- [ ] **Step 5 : vérifier la non-régression des tests précédents du worker**

Run: `QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_selfplay_worker_smoke.py tests/test_selfplay_end_to_end.py -v`
Expected: PASS (le mode self-play continu n'est pas affecté).

- [ ] **Step 6 : commit**

```bash
git add native_engine/src/bin/selfplay_worker.rs tests/test_selfplay_worker_single_move.py
git commit -m "Ajouter le mode un-seul-coup à budget de temps, pour l'arbitrage en campagne"
```

---

## Task 12 : Campagne d'évaluation contre le moteur de référence (la barre de succès)

**Files:**
- Create: `tools/mcts_az_vs_reference_campaign.py`
- Test: manuel (campagne réelle, pas un test unitaire — même convention que les scripts `tools/*_campaign.py` existants, aucun n'a de test unitaire propre)

**Interfaces:**
- Consumes : mode `--single-move` du worker (Task 11), `native.native_search`/`native.encode`/`santorini.engine.validate_turn` (existants), motifs `digest`/`verify_reference`/`atomic_json`/`bootstrap` de `tools/hermes_canonicalisation_campaign.py:45-61,213-221` (repris verbatim), corpus `experiments/selfplay-100-5s-20261005/campaign.json`, moteur de référence `experiments/references/b698745`, checkpoint produit par `Trainer.save_checkpoint` (Task 9).
- Produces : script CLI qui joue 200 parties (100 configurations × 2 couleurs, comme `tools/hermes_canonicalisation_campaign.py:138-142`), écrit un fichier `game-NNNN.json` par partie, calcule le bootstrap CI par groupe de placement, écrit `summary.json` et `rapport.md`.

- [ ] **Step 1 : écrire le script complet**

Create `tools/mcts_az_vs_reference_campaign.py` :
```python
"""200 parties appariées, candidat (MCTS + réseau, sous-jeu sans pouvoir) contre
le moteur de référence figé, budget de temps égal par tour. Même corpus et même
méthode de bootstrap par groupe de placement que
`tools/hermes_canonicalisation_campaign.py` ; reprend directement `digest`,
`verify_reference`, `atomic_json`, `bootstrap` de ce script (voir ses lignes
45-61 et 213-221 pour la provenance).
"""
from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import random
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'experiments/references/b698745'
CORPUS = ROOT / 'experiments/selfplay-100-5s-20261005/campaign.json'
WORKER_BINARY = ROOT / 'native_engine' / 'target' / 'release' / 'selfplay_worker'

sys.path.insert(0, str(ROOT))
from santorini.engine import Action, Position, validate_setup, validate_turn  # noqa: E402
from santorini.native import encode, CAction, native_search  # noqa: E402
from santorini import native  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_reference():
    meta = json.loads((REFERENCE / 'reference.json').read_text())
    for name, key in [('libsantoni_engine.so', 'binary_sha256'), ('source.tar.gz', 'archive_sha256')]:
        if digest(REFERENCE / name) != meta[key]:
            raise ValueError(f'Reference integrity mismatch: {name}')
    return meta


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def bootstrap(groups):
    rng = random.Random(20261010)
    values = list(groups.values())
    samples = []
    for _ in range(5000):
        selected = [rng.choice(values) for _ in values]
        samples.append(sum(sum(v) for v in selected) / sum(len(v) for v in selected))
    samples.sort()
    return [samples[125], samples[4874]]


def candidate_move(position, seconds, checkpoint, socket_path, scratch_dir):
    state = encode(position)
    input_path = scratch_dir / 'state.bin'
    output_path = scratch_dir / 'move.bin'
    input_path.write_bytes(bytes(state))
    if output_path.exists():
        output_path.unlink()
    cmd = [str(WORKER_BINARY), '--single-move', '--input-state', str(input_path),
           '--output-move', str(output_path), '--seconds', str(seconds)]
    if socket_path:
        cmd += ['--inference-socket', str(socket_path)]
    started = time.monotonic()
    result = subprocess.run(cmd, timeout=seconds + 10, capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError(f'Candidate worker failed: {result.stderr}')
    raw = output_path.read_bytes()
    move_action = CAction.from_buffer_copy(raw, 0)
    build_action = CAction.from_buffer_copy(raw, C.sizeof(CAction))
    actions = (
        Action('move', move_action.player, move_action.worker, move_action.source, move_action.target),
        Action('build', build_action.player, build_action.worker, build_action.source, build_action.target),
    )
    turn = validate_turn(position, actions)
    return turn, time.monotonic() - started


def reference_move(position, seconds):
    analysis = native_search(position, seconds)
    if analysis.turn is None:
        raise ValueError('Reference engine returned no legal turn')
    return analysis.turn, analysis.elapsed


def play(job, directory, seconds, checkpoint, socket_path, scratch_dir):
    initial = job['initial']
    path = Path(directory) / f"game-{job['number']:04d}.json"
    position = validate_setup(initial['workers'], initial['powers'], initial['first'])
    data = {'job': job, 'status': 'running', 'decisions': []}
    atomic_json(path, data)
    try:
        for ply in range(150):
            is_candidate = position.player == job['variant_player']
            if is_candidate:
                turn, elapsed = candidate_move(position, seconds, checkpoint, socket_path, scratch_dir)
            else:
                turn, elapsed = reference_move(position, seconds)
            data['decisions'].append({'ply': ply + 1, 'player': position.player,
                                       'backend': 'candidate' if is_candidate else 'reference',
                                       'elapsed': elapsed})
            position = turn.after
            atomic_json(path, {**data, 'position': position.to_dict()})
            if position.winner is not None:
                break
        data['status'] = 'finished' if position.winner is not None else 'inconclusive'
    except Exception as exc:
        data.update(status='error', error=f'{type(exc).__name__}: {exc}')
    data['winner'] = position.winner
    atomic_json(path, data)
    return {'number': job['number'], 'status': data['status'],
            'variant_won': position.winner == job['variant_player'] if position.winner is not None else None}


def summary(directory, corpus):
    games = [json.loads(p.read_text()) for p in sorted(directory.glob('game-*.json'))]
    finished = [g for g in games if g['status'] == 'finished']
    groups = defaultdict(list)
    for game in finished:
        job = game['job']
        won = int(game['winner'] == job['variant_player'])
        key = (tuple(job['initial']['matchup']), job['initial']['opening'])
        groups[key].append(won)
    wins = sum(sum(v) for v in groups.values())
    result = {'expected': 200, 'completed': len(finished), 'candidate_wins': wins,
               'reference_wins': len(finished) - wins,
               'win_rate': wins / len(finished) if finished else None,
               'errors': [{'game': g['job']['number'], 'error': g.get('error')} for g in games
                          if g['status'] in ('error', 'inconclusive')]}
    if len(finished) == 200:
        result['placement_bootstrap_95'] = bootstrap(groups)
        result['placement_groups'] = len(groups)
    atomic_json(directory / 'summary.json', result)
    return result


def report(directory, result):
    lines = ['# Candidat MCTS + réseau contre le moteur de référence figé (sous-jeu sans pouvoir)', '',
             f"Parties terminées : {result['completed']}/{result['expected']}.", '']
    if result['win_rate'] is not None:
        lines.append(f"Victoires du candidat : {result['candidate_wins']} / {result['completed']} "
                      f"({result['win_rate']*100:.1f} %).")
    if 'placement_bootstrap_95' in result:
        lines.append(f"Intervalle bootstrap à 95 % par groupe de placement "
                      f"({result['placement_groups']} groupes) : {result['placement_bootstrap_95']}.")
    lines += ['', f"Erreurs ou parties non conclues : {len(result['errors'])}."]
    (directory / 'rapport.md').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--seconds', type=float, default=30.0)
    parser.add_argument('--inference-socket', default=None,
                         help='Socket du serveur d\'inférence déjà démarré ; sans cette option, '
                              'le worker utilise un évaluateur uniforme (politique aléatoire).')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    verify_reference()
    corpus = json.loads(CORPUS.read_text())
    jobs_all = corpus['jobs']
    jobs = [{'number': i + 1, 'variant_player': player, 'initial': job}
            for i, (job, player) in enumerate((job, player) for job in jobs_all for player in (0, 1))]
    directory = Path(args.out)
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        scratch_dir = Path(scratch)
        results = [play(job, directory, args.seconds, args.checkpoint, args.inference_socket, scratch_dir)
                   for job in jobs]
    result = summary(directory, corpus)
    report(directory, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
```

- [ ] **Step 2 : lancer un essai pilote avant la campagne complète (2 parties, budget court)**

Run: `python3 tools/mcts_az_vs_reference_campaign.py --checkpoint experiments/mcts-az-no-power/checkpoints/gen-0001.pt --seconds 2 --out /tmp/mcts-az-pilot/` (adapter le chemin du checkpoint au dernier produit par l'entraîneur) et vérifier que `rapport.md` est produit sans `error`. Si le script plante, corriger avant de lancer la vraie campagne — ne jamais lancer directement 200 parties à 30 secondes sans ce test de fumée (plusieurs heures de calcul en jeu).

- [ ] **Step 3 : lancer la campagne réelle à l'échelle (pas un test unitaire — la mesure qui tranche la barre de succès)**

Run: `python3 tools/mcts_az_vs_reference_campaign.py --checkpoint experiments/mcts-az-no-power/checkpoints/gen-XXXX.pt --seconds 30 --inference-socket /path/vers/infer.sock --out experiments/mcts-az-no-power-campaign-<date>/`

Pendant cette exécution, lancer `nvidia-smi dmon -s u` en parallèle quelques minutes pour vérifier que le goulot GPU identifié par le pilote jetable est bien résolu (spec §7, risque "sous-exploitation GPU persistante") — consigner le résultat dans le rapport final, pas seulement le taux de victoire.

Lire `rapport.md` : si le taux de victoire du candidat est +5pt au-dessus de la référence avec une borne inférieure d'IC bootstrap strictement positive (spec §2), la phase 1 est un succès et la décision de poursuivre vers les pouvoirs avancés redevient une question ouverte à soumettre à l'utilisateur. Sinon, documenter le résultat négatif dans un fichier `docs/mcts-az-no-power-<date>.md` suivant le modèle des campagnes précédentes (`docs/ml-ranker-cost-20261010.md`), et mettre à jour la mémoire `santoni-engine-roadmap-priorities` en conséquence — ne pas ré-ouvrir cet axe sans changement matériel d'approche, même règle que pour le ranker ML rejeté.

- [ ] **Step 4 : commit**

```bash
git add tools/mcts_az_vs_reference_campaign.py
git commit -m "Campagne d'évaluation candidat MCTS+réseau vs moteur de référence à temps égal"
```

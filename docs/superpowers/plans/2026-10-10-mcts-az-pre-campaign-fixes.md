# MCTS+AZ pre-campaign fixes (real board input, real policy target, self-play exploration) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the three blockers that make the MCTS+NN self-play pipeline for the Santorini no-power subgame unable to learn anything, before the real 200-game evaluation campaign (`tools/mcts_az_vs_reference_campaign.py`) is allowed to run: (1) the trainer feeds the network a constant all-zero board instead of the real position, (2) the policy target is one-hot on the move played instead of the real MCTS visit distribution, (3) self-play has zero exploration (deterministic argmax, no root noise — four self-play games were verified byte-identical during the original chantier's final review).

**Architecture:** Extend the on-disk `GameRecord` format (native, hand-written JSON — no serde) to carry, per ply, the real encoded board (`mcts::encode_planes`, already correct and tested, just never wired in) and the full MCTS visit distribution over every legal move at that ply (not just the move played). Wire the trainer to decode both into real tensors and a real soft policy target, and fix the loss call that was silently collapsing soft targets back to one-hot. Add standard AlphaZero-style root Dirichlet noise + opening-ply temperature sampling to self-play only — never to the arbitration path used by the evaluation campaign, which must stay the strongest deterministic play.

**Tech Stack:** Rust (`native_engine`, no new crates — project rule), Python/PyTorch (`santoni_az`, `pytest`).

**Spec:** `docs/superpowers/specs/2026-10-10-mcts-az-no-power-design.md` (original design) — this plan implements the "Note sur la cible de politique simplifiée" deferred item from `docs/superpowers/plans/2026-10-10-mcts-az-no-power.md` (Task 9, lines 1365-1373) plus the related fake-input and zero-exploration gaps recorded in memory `santoni-mcts-az-no-power-chantier`.

## Global Constraints

- No new external crates in `native_engine` (project-wide rule; `Cargo.toml` has zero dependencies today — keep it that way, including no `rand`/`serde`).
- Do not modify `native_engine/src/lib.rs`, `advanced.rs`, `special.rs` — the production alpha-beta engine stays untouched; it is the reference opponent for the eventual campaign.
- `native_engine/src/bin/selfplay_worker.rs`'s `--single-move` mode (`run_single_move`, used by `tools/mcts_az_vs_reference_campaign.py` for arbitration) must remain fully deterministic and noise-free — exploration is a training-data concern only, never an arbitration concern.
- `PLANE_BYTES = 150`, `ACTION_SPACE = 1250` (both already defined in `mcts.rs`) do not change.
- The on-disk `GameRecord` JSON format is allowed to break compatibility with any file written by the current code: no real campaign has run yet (confirmed in memory `santoni-mcts-az-no-power-chantier`), so there is nothing to migrate.

## Review Focus

- A ply with exactly one legal move (a forced move) must not panic `sample_weighted`/`dirichlet_noise`/`mix_root_noise` — single-edge root is a real, common case in this subgame (e.g. a worker boxed in). Covered in Task 5.
- The `ply % 2` value-sign alternation added in Task 3 assumes strict turn alternation through a whole game; that invariant is already enforced everywhere in `mcts.rs` via `is_in_scope`'s `assert!`, so a violation would panic loudly during self-play rather than silently mislabel a value target — note this safeguard, no extra task needed.
- `consume_new_games` must still produce exactly one training example per ply (checkpoint/resume example-count tests already pin this) even though each example now carries much more data per position — covered by the existing (migrated) resume tests in Task 3.
- Root noise must be mixed in *before* the remaining simulations run, not after — a noise call that only touches the final prior values without affecting the UCB selection that already happened would be silently inert. Covered by `mix_root_noise_biases_simulations_toward_the_noised_edge` in Task 4, which checks the actual visit distribution, not just the stored prior.
- `visit_counts()`'s edge order must stay aligned with `legal_children(&state)`'s order when building the per-move `(action_index, visits)` list — both tasks that build this list (Task 2, Task 5) rely on the pre-existing invariant that `Mcts`'s root edges are built directly from `legal_children(&state)` in Task order; no index remapping is introduced, so no new task needed, but call out the dependency here since it is not type-checked.

---

## Task 1: Extend `GameRecord` to carry per-position board planes and the full visit distribution

**Files:**
- Modify: `native_engine/src/mcts_io.rs`

**Interfaces:**
- Consumes: `crate::mcts::PLANE_BYTES` (already `pub const PLANE_BYTES: usize = 150` in `mcts.rs`).
- Produces: `pub struct PositionRecord { pub planes: [u8; PLANE_BYTES], pub visits: Vec<(usize, u32)> }`, `pub struct GameRecord { pub positions: Vec<PositionRecord>, pub outcome: i8 }` (replaces the old `pub moves: Vec<(usize, u32)>` field), `pub fn write_game_atomically(path: &Path, record: &GameRecord) -> std::io::Result<()>` (same signature, new JSON shape: `{"outcome":i8,"positions":[{"planes":[...150 ints...],"visits":[[idx,count],...]},...]}`).

- [ ] **Step 1: Write the failing test**

Replace the existing test in `native_engine/src/mcts_io.rs` (the `#[cfg(test)] mod tests` block at the bottom of the file):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn writes_valid_json_and_no_tmp_file_survives() {
        let dir = std::env::temp_dir().join("santoni_mcts_io_test");
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("game-0001.json");
        let mut planes0 = [0u8; PLANE_BYTES];
        planes0[3] = 1;
        let mut planes1 = [0u8; PLANE_BYTES];
        planes1[149] = 1;
        let record = GameRecord {
            outcome: 1,
            positions: vec![
                PositionRecord { planes: planes0, visits: vec![(42, 100), (7, 88)] },
                PositionRecord { planes: planes1, visits: vec![(3, 12)] },
            ],
        };
        write_game_atomically(&path, &record).unwrap();
        let text = fs::read_to_string(&path).unwrap();

        let mut expected_planes0 = vec!["0"; PLANE_BYTES];
        expected_planes0[3] = "1";
        let mut expected_planes1 = vec!["0"; PLANE_BYTES];
        expected_planes1[149] = "1";
        let expected = format!(
            "{{\"outcome\":1,\"positions\":[{{\"planes\":[{}],\"visits\":[[42,100],[7,88]]}},{{\"planes\":[{}],\"visits\":[[3,12]]}}]}}\n",
            expected_planes0.join(","),
            expected_planes1.join(","),
        );
        assert_eq!(text, expected);
        assert!(!path.with_extension("json.tmp").exists());
        fs::remove_dir_all(&dir).unwrap();
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd native_engine && cargo test --lib mcts_io -- --nocapture`
Expected: FAIL to compile — `PositionRecord` does not exist yet, `GameRecord.moves` field does not exist.

- [ ] **Step 3: Write the implementation**

Replace the whole non-test content of `native_engine/src/mcts_io.rs`:

```rust
use std::io::Write;
use std::path::Path;

use crate::mcts::PLANE_BYTES;

pub struct PositionRecord {
    pub planes: [u8; PLANE_BYTES],
    pub visits: Vec<(usize, u32)>,
}

pub struct GameRecord {
    pub positions: Vec<PositionRecord>,
    pub outcome: i8,
}

pub fn write_game_atomically(path: &Path, record: &GameRecord) -> std::io::Result<()> {
    let tmp = path.with_extension("json.tmp");
    {
        let mut f = std::fs::File::create(&tmp)?;
        let positions_json: Vec<String> = record
            .positions
            .iter()
            .map(|p| {
                let planes_json: Vec<String> = p.planes.iter().map(|b| b.to_string()).collect();
                let visits_json: Vec<String> = p
                    .visits
                    .iter()
                    .map(|(idx, visits)| format!("[{idx},{visits}]"))
                    .collect();
                format!(
                    "{{\"planes\":[{}],\"visits\":[{}]}}",
                    planes_json.join(","),
                    visits_json.join(",")
                )
            })
            .collect();
        write!(
            f,
            "{{\"outcome\":{},\"positions\":[{}]}}\n",
            record.outcome,
            positions_json.join(",")
        )?;
        f.flush()?;
    }
    std::fs::rename(&tmp, path)
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd native_engine && cargo test --lib mcts_io -- --nocapture`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add native_engine/src/mcts_io.rs
git commit -m "Étendre GameRecord : planes encodées et distribution de visites complète par coup"
```

---

## Task 2: Wire the self-play worker to write real planes and the full visit distribution

**Files:**
- Modify: `native_engine/src/bin/selfplay_worker.rs:109-152` (`initial_state`, `play_one_game`)
- Modify: `tests/test_mcts_io_format.py`
- Modify: `tests/test_selfplay_worker_smoke.py`

**Interfaces:**
- Consumes: `santoni_engine::mcts_io::{GameRecord, PositionRecord, write_game_atomically}` (Task 1), `santoni_engine::mcts::{encode_planes, action_index, legal_children, terminal_value, Mcts}` (unchanged).
- Produces: `fn play_one_game(simulations_per_move: u32, evaluator: &mut dyn LeafEvaluator) -> GameRecord` — same deterministic argmax move selection as before (exploration comes in Task 5), new record shape.

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_mcts_io_format.py` entirely:

```python
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_rust_written_game_is_valid_json(tmp_path):
    sample = tmp_path / "game-0001.json"
    sample.write_text(
        '{"outcome":1,"positions":['
        '{"planes":[0,1],"visits":[[42,100],[7,88]]}'
        ']}\n'
    )
    data = json.loads(sample.read_text())
    assert data["outcome"] == 1
    assert data["positions"][0]["planes"] == [0, 1]
    assert data["positions"][0]["visits"] == [[42, 100], [7, 88]]
```

In `tests/test_selfplay_worker_smoke.py`, replace every `data["moves"]` reference (two occurrences, in `test_worker_produces_complete_valid_games` and `test_concurrent_workers_do_not_overwrite_each_others_games`) with `data["positions"]`, and strengthen the first test to check the new per-position shape:

```python
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
        assert len(data["positions"]) > 0
        first = data["positions"][0]
        assert len(first["planes"]) == 150
        assert all(b in (0, 1) for b in first["planes"])
        assert len(first["visits"]) > 0
        assert all(isinstance(pair, list) and len(pair) == 2 for pair in first["visits"])
    assert not list(out_dir.glob("*.tmp"))
```

And in `test_concurrent_workers_do_not_overwrite_each_others_games`:

```python
    for f in files:
        data = json.loads(f.read_text())
        assert data["outcome"] in (-1, 1)
        assert len(data["positions"]) > 0
    assert not list(out_dir.glob("*.tmp"))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_mcts_io_format.py tests/test_selfplay_worker_smoke.py -v`
Expected: `test_mcts_io_format.py` passes already (pure JSON, no Rust binary involved). The smoke tests FAIL: the worker binary (not yet rebuilt) still writes the old `"moves"` key, so `data["positions"]` raises `KeyError`.

- [ ] **Step 3: Write the implementation**

In `native_engine/src/bin/selfplay_worker.rs`, change the import line:

```rust
use santoni_engine::mcts_io::{write_game_atomically, GameRecord, PositionRecord};
```

Replace `play_one_game`:

```rust
fn play_one_game(simulations_per_move: u32, evaluator: &mut dyn LeafEvaluator) -> GameRecord {
    let mut state = initial_state();
    let mut positions = Vec::new();
    loop {
        if let Some(v) = terminal_value(&state, state.player) {
            let outcome = if state.player == 0 { v } else { -v };
            return GameRecord { positions, outcome: outcome as i8 };
        }
        let planes = santoni_engine::mcts::encode_planes(&state);
        let mut mcts = Mcts::new(state, 1.5);
        mcts.run(simulations_per_move, evaluator);
        let counts = mcts.visit_counts();
        let legal = legal_children(&state);
        let visits: Vec<(usize, u32)> = counts
            .iter()
            .map(|(mv, bld, n)| {
                let worker_slot = state.workers[state.player as usize]
                    .iter()
                    .position(|&c| c == mv.source)
                    .unwrap();
                (santoni_engine::mcts::action_index(mv, bld, worker_slot), *n)
            })
            .collect();
        let best = counts.iter().enumerate().max_by_key(|(_, (_, _, n))| *n).unwrap().0;
        let (_, _, after) = legal[best];
        positions.push(PositionRecord { planes, visits });
        state = after;
    }
}
```

- [ ] **Step 4: Rebuild the binary and run tests to verify they pass**

Run: `cd native_engine && cargo build --release --bin selfplay_worker && cd .. && QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_mcts_io_format.py tests/test_selfplay_worker_smoke.py tests/test_selfplay_end_to_end.py -v`
Expected: PASS (includes `test_selfplay_end_to_end.py`, unaffected by the format change but exercises the rebuilt binary end-to-end with the real inference server).

- [ ] **Step 5: Commit**

```bash
git add native_engine/src/bin/selfplay_worker.rs tests/test_mcts_io_format.py tests/test_selfplay_worker_smoke.py
git commit -m "Écrire les planes réelles et la distribution de visites complète dans chaque partie self-play"
```

---

## Task 3: Decode real board planes and the real policy target in the trainer

**Files:**
- Modify: `santoni_az/trainer.py:45-95` (`consume_new_games`, `train_step`)
- Modify: `tests/test_az_trainer.py`

**Interfaces:**
- Consumes: the new on-disk format from Task 2 (`data["positions"]`, each `{"planes": [...150 ints...], "visits": [[idx, count], ...]}`), `NUM_PLANES` (already imported in `trainer.py` from `.network`).
- Produces: `Trainer.consume_new_games(self) -> int` (same signature), `Trainer.train_step(self, batch_size=64) -> float` (same signature) — both now operate on real per-position data instead of placeholders.

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_az_trainer.py` entirely:

```python
import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from santoni_az.network import NUM_PLANES
from santoni_az.trainer import CHECKPOINT_RETENTION, Trainer, checkpoint_generation

PLACEHOLDER_PLANES = [0] * (NUM_PLANES * 5 * 5)


def write_fake_game(games_dir, name, outcome, positions):
    """`positions` is a list of (planes, visits) pairs — `planes` a flat
    list of NUM_PLANES*5*5 ints, `visits` a list of [action_idx, visit_count]
    pairs — mirroring the real format `mcts_io::write_game_atomically` writes."""
    payload = {
        "outcome": outcome,
        "positions": [{"planes": planes, "visits": visits} for planes, visits in positions],
    }
    (games_dir / name).write_text(json.dumps(payload) + "\n")


def make_dirs(tmp_path):
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return games_dir, checkpoint_dir


def test_checkpoint_resume_preserves_generation_counter(tmp_path):
    games_dir, checkpoint_dir = make_dirs(tmp_path)
    write_fake_game(games_dir, "game-0000.json", 1, [
        (PLACEHOLDER_PLANES, [[10, 20]]),
        (PLACEHOLDER_PLANES, [[5, 15]]),
    ])
    write_fake_game(games_dir, "game-0001.json", -1, [
        (PLACEHOLDER_PLANES, [[3, 10]]),
    ])

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


def test_checkpoint_resume_preserves_untrained_examples(tmp_path):
    games_dir, checkpoint_dir = make_dirs(tmp_path)
    # 3 moves in game 0, 2 moves in game 1 => 5 examples total.
    write_fake_game(games_dir, "game-0000.json", 1, [
        (PLACEHOLDER_PLANES, [[10, 20]]),
        (PLACEHOLDER_PLANES, [[5, 15]]),
        (PLACEHOLDER_PLANES, [[7, 11]]),
    ])
    write_fake_game(games_dir, "game-0001.json", -1, [
        (PLACEHOLDER_PLANES, [[3, 10]]),
        (PLACEHOLDER_PLANES, [[8, 4]]),
    ])

    trainer = Trainer(str(games_dir), str(checkpoint_dir))
    trainer.consume_new_games()
    total_examples = len(trainer._examples)
    assert total_examples == 5

    k = 2
    assert k < total_examples
    trainer.train_step(batch_size=k)  # trains on k examples, leaves the rest untrained
    trainer.save_checkpoint()

    resumed = Trainer(str(games_dir), str(checkpoint_dir))
    resumed.load_latest_checkpoint()
    assert len(resumed._examples) == total_examples - k


def test_load_latest_checkpoint_selects_the_highest_generation_numerically(tmp_path):
    """Passé la génération 9999, `sorted(glob("gen-*.pt"))[-1]` (tri lexical)
    renvoie `gen-9999.pt` alors que `gen-10001.pt` existe : la reprise
    rechargeait silencieusement un checkpoint périmé, puis réécrasait les
    générations déjà produites en repartant de son compteur."""
    games_dir, checkpoint_dir = make_dirs(tmp_path)
    write_fake_game(games_dir, "game-0000.json", 1, [(PLACEHOLDER_PLANES, [[10, 20]])])

    trainer = Trainer(str(games_dir), str(checkpoint_dir))
    trainer.consume_new_games()
    trainer.train_step(batch_size=1)
    trainer.generation = 10000
    newest = trainer.save_checkpoint()
    assert Path(newest).name == "gen-10001.pt"

    for name in ("gen-0001.pt", "gen-9999.pt", "gen-10000.pt"):
        (checkpoint_dir / name).write_bytes(b"inerte : seul le nom compte ici")
    assert sorted(p.name for p in checkpoint_dir.glob("gen-*.pt"))[-1] == "gen-9999.pt"

    resumed = Trainer(str(games_dir), str(checkpoint_dir))
    assert resumed.load_latest_checkpoint() == 10001


def test_save_checkpoint_prunes_all_but_the_retained_newest(tmp_path):
    """Rien ne supprimait les anciens checkpoints : `save_checkpoint` ne doit
    laisser que les `CHECKPOINT_RETENTION` générations les plus récentes,
    choisies numériquement, pas lexicalement."""
    games_dir, checkpoint_dir = make_dirs(tmp_path)
    trainer = Trainer(str(games_dir), str(checkpoint_dir))
    trainer.generation = 9997
    saves = CHECKPOINT_RETENTION + 3
    for _ in range(saves):
        trainer.save_checkpoint()

    kept = sorted(checkpoint_generation(p) for p in checkpoint_dir.glob("gen-*.pt"))
    assert len(kept) == CHECKPOINT_RETENTION
    assert kept == list(range(9998 + saves - CHECKPOINT_RETENTION, 9998 + saves))
    assert kept[-1] == 10005
    assert not list(checkpoint_dir.glob("*.tmp"))

    resumed = Trainer(str(games_dir), str(checkpoint_dir))
    assert resumed.load_latest_checkpoint() == 10005


def test_consume_new_games_decodes_real_board_planes(tmp_path):
    games_dir, checkpoint_dir = make_dirs(tmp_path)
    planes = [1 if i % 7 == 0 else 0 for i in range(NUM_PLANES * 5 * 5)]
    write_fake_game(games_dir, "game-0000.json", 1, [(planes, [[0, 10]])])

    trainer = Trainer(str(games_dir), str(checkpoint_dir))
    trainer.consume_new_games()
    stored_planes, _, _ = trainer._examples[0]
    assert torch.equal(stored_planes, torch.tensor(planes, dtype=torch.float32).view(NUM_PLANES, 5, 5))


def test_consume_new_games_builds_policy_target_from_visit_distribution(tmp_path):
    games_dir, checkpoint_dir = make_dirs(tmp_path)
    write_fake_game(games_dir, "game-0000.json", 1, [(PLACEHOLDER_PLANES, [[3, 30], [9, 10]])])

    trainer = Trainer(str(games_dir), str(checkpoint_dir))
    trainer.consume_new_games()
    _, policy_target, _ = trainer._examples[0]
    assert policy_target[3].item() == pytest.approx(0.75)
    assert policy_target[9].item() == pytest.approx(0.25)
    assert policy_target.sum().item() == pytest.approx(1.0)


def test_consume_new_games_alternates_value_target_sign_by_ply(tmp_path):
    games_dir, checkpoint_dir = make_dirs(tmp_path)
    write_fake_game(games_dir, "game-0000.json", 1, [
        (PLACEHOLDER_PLANES, [[0, 1]]),  # ply 0 : joueur 0 au trait
        (PLACEHOLDER_PLANES, [[0, 1]]),  # ply 1 : joueur 1 au trait
        (PLACEHOLDER_PLANES, [[0, 1]]),  # ply 2 : joueur 0 au trait
    ])

    trainer = Trainer(str(games_dir), str(checkpoint_dir))
    trainer.consume_new_games()
    values = [v for _, _, v in trainer._examples]
    assert values == [1.0, -1.0, 1.0]


def test_train_step_loss_uses_soft_policy_target_distribution(tmp_path):
    """Avant ce correctif, `train_step` appelait `F.cross_entropy` avec
    `policy_targets.argmax(dim=1)` : la distribution de visites réelle
    (ex. 75%/25% sur deux coups) s'effondrait silencieusement sur un simple
    one-hot du coup majoritaire. Ce test pin la vraie perte "soft target"
    comme référence — il échoue avec l'ancienne ligne `argmax` dès que la
    distribution n'est pas déjà un one-hot."""
    games_dir, checkpoint_dir = make_dirs(tmp_path)
    write_fake_game(games_dir, "game-0000.json", 1, [(PLACEHOLDER_PLANES, [[3, 3], [9, 1]])])

    trainer = Trainer(str(games_dir), str(checkpoint_dir))
    trainer.consume_new_games()
    planes, policy_target, value_target = trainer._examples[0]
    with torch.no_grad():
        policy_logits, value_pred = trainer.net(planes.unsqueeze(0))
        expected_policy_loss = F.cross_entropy(policy_logits, policy_target.unsqueeze(0))
        expected_value_loss = F.mse_loss(value_pred, torch.tensor([[value_target]]))

    loss = trainer.train_step(batch_size=1)
    assert loss == pytest.approx((expected_policy_loss + expected_value_loss).item(), rel=1e-4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_az_trainer.py -v`
Expected: FAIL — the current `consume_new_games` reads `data["moves"]`, which no longer exists in the fixtures (`KeyError: 'moves'`).

- [ ] **Step 3: Write the implementation**

In `santoni_az/trainer.py`, replace `consume_new_games`:

```python
    def consume_new_games(self):
        new_files = sorted(
            f for f in self.games_dir.glob("game-*.json") if f.name not in self._consumed_games
        )
        for f in new_files:
            data = json.loads(f.read_text())
            outcome = data["outcome"]
            for ply, position in enumerate(data["positions"]):
                planes = torch.tensor(position["planes"], dtype=torch.float32).view(NUM_PLANES, 5, 5)
                total_visits = sum(visits for _, visits in position["visits"])
                policy_target = torch.zeros(ACTION_SPACE)
                for action_idx, visits in position["visits"]:
                    policy_target[action_idx] = visits / total_visits
                # Alternance stricte joueur 0 / joueur 1 à chaque ply : vraie
                # pour toute la partie dans le sous-jeu sans pouvoir, où
                # `is_in_scope` (native_engine/src/mcts.rs) impose
                # `powers == [0, 0]` sur toute position rencontrée — aucun
                # pouvoir ne peut donc jamais accorder de tour supplémentaire.
                # `outcome` est fixé du point de vue du joueur 0 (cf.
                # `play_one_game`) ; ce signe le reprojette depuis le point de
                # vue du joueur réellement au trait à ce ply, celui encodé
                # dans `position["planes"]`.
                value_target = outcome if ply % 2 == 0 else -outcome
                self._examples.append((planes, policy_target, float(value_target)))
            self._consumed_games.add(f.name)
        return len(new_files)
```

Replace the `cross_entropy` call in `train_step`:

```python
        policy_loss = F.cross_entropy(policy_logits, policy_targets)
```

(was `F.cross_entropy(policy_logits, policy_targets.argmax(dim=1))`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_az_trainer.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add santoni_az/trainer.py tests/test_az_trainer.py
git commit -m "Décoder les vraies planes et la vraie cible de politique dans l'entraîneur"
```

---

## Task 4: `Mcts::ensure_root_expanded` + `Mcts::mix_root_noise`

**Files:**
- Modify: `native_engine/src/mcts.rs` (the `impl Mcts` block, and the `mcts_tree_tests` module)

**Interfaces:**
- Consumes: existing private `Node`/`Edge` structs (same file, no visibility change needed).
- Produces: `pub fn Mcts::ensure_root_expanded(&mut self, evaluator: &mut dyn LeafEvaluator)` (idempotent root expansion, extracted from `run` with unchanged external behavior of `run`), `pub fn Mcts::mix_root_noise(&mut self, noise: &[f32], epsilon: f32)` (blends `noise[i]` into root edge `i`'s prior as `prior*(1-epsilon) + noise[i]*epsilon`; panics if called before `ensure_root_expanded` or if `noise.len()` doesn't match the number of root edges).

- [ ] **Step 1: Write the failing test**

Add to the `mcts_tree_tests` module in `native_engine/src/mcts.rs`:

```rust
    #[test]
    fn mix_root_noise_biases_simulations_toward_the_noised_edge() {
        let mut s = default_state();
        s.workers = [[12, 13, -1, -1], [0, 1, -1, -1]];
        let mut mcts = Mcts::new(s, 1.5);
        let mut eval = UniformEvaluator;
        mcts.ensure_root_expanded(&mut eval);
        let n_edges = mcts.visit_counts().len();
        assert!(n_edges > 1, "fixture doit offrir plusieurs coups légaux pour que le test ait un sens");
        let mut noise = vec![0.0f32; n_edges];
        noise[0] = 1.0;
        mcts.mix_root_noise(&noise, 1.0);
        mcts.run(200, &mut eval);
        let counts = mcts.visit_counts();
        let noised_visits = counts[0].2;
        assert!(
            counts.iter().all(|(_, _, n)| *n <= noised_visits),
            "l'arête bruitée doit recevoir au moins autant de visites que les autres"
        );
        assert!(
            noised_visits > 200 / n_edges as u32,
            "l'arête bruitée doit recevoir nettement plus que sa part uniforme"
        );
    }

    #[test]
    #[should_panic(expected = "mix_root_noise appelé avant l'expansion de la racine")]
    fn mix_root_noise_panics_before_root_expansion() {
        let s = default_state();
        let mut mcts = Mcts::new(s, 1.5);
        mcts.mix_root_noise(&[1.0], 0.25);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd native_engine && cargo test --lib mcts_tree_tests -- --nocapture`
Expected: FAIL to compile — `ensure_root_expanded` and `mix_root_noise` do not exist yet.

- [ ] **Step 3: Write the implementation**

In `native_engine/src/mcts.rs`, replace the existing `pub fn run` method with:

```rust
    pub fn run(&mut self, simulations: u32, evaluator: &mut dyn LeafEvaluator) {
        assert!(
            terminal_value(&self.root, self.root.player).is_none(),
            "Mcts::run appelé sur une racine déjà terminale"
        );
        self.ensure_root_expanded(evaluator);
        for _ in 0..simulations {
            self.simulate_one(self.root, evaluator);
        }
    }

    /// Expansion initiale de la racine (évaluation + peuplement des arêtes),
    /// idempotente : ne fait rien si déjà expansée. Ne compte pas comme une
    /// des simulations de `run` — sans elle, la toute première itération ne
    /// ferait qu'évaluer et peupler les arêtes de la racine sans jamais
    /// descendre ni backuper sur l'une d'elles (aucun parent au-dessus de la
    /// racine pour recevoir ce backup) — `visit_counts` totaliserait alors
    /// `simulations - 1`. Extraite de `run` (plutôt qu'inlinée comme avant)
    /// pour que l'appelant self-play puisse injecter du bruit de Dirichlet
    /// (`mix_root_noise`) entre l'expansion et le reste des simulations ;
    /// l'appeler directement est un no-op si `run` l'a déjà fait.
    pub fn ensure_root_expanded(&mut self, evaluator: &mut dyn LeafEvaluator) {
        if !self.nodes.contains_key(&self.root) {
            self.simulate_one(self.root, evaluator);
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
        for (edge, &n) in node.edges.iter_mut().zip(noise) {
            edge.prior = edge.prior * (1.0 - epsilon) + n * epsilon;
        }
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd native_engine && cargo test --lib -- --nocapture`
Expected: PASS, including every pre-existing `mcts.rs` test (unchanged behavior of `run`).

- [ ] **Step 5: Commit**

```bash
git add native_engine/src/mcts.rs
git commit -m "Mcts::ensure_root_expanded + Mcts::mix_root_noise pour le bruit d'exploration racine"
```

---

## Task 5: Root Dirichlet noise + opening-ply temperature sampling in self-play

**Files:**
- Modify: `native_engine/src/bin/selfplay_worker.rs`

**Interfaces:**
- Consumes: `Mcts::ensure_root_expanded`, `Mcts::mix_root_noise` (Task 4).
- Produces: `fn play_one_game(simulations_per_move: u32, evaluator: &mut dyn LeafEvaluator, rng_seed: u64) -> GameRecord` (new `rng_seed` parameter — only internal callers, `main()`, need updating), private helpers `Xorshift64`, `seed_for_game`, `gamma_sample`, `dirichlet_noise`, `sample_weighted` (none exposed outside this binary).

- [ ] **Step 1: Write the failing tests**

Add to `native_engine/src/bin/selfplay_worker.rs`, as a new module (place after the existing `socket_evaluator_tests` module):

```rust
#[cfg(test)]
mod exploration_tests {
    use super::*;

    #[test]
    fn same_seed_reproduces_the_same_game() {
        let mut eval_a = UniformEvaluator;
        let mut eval_b = UniformEvaluator;
        let a = play_one_game(4, &mut eval_a, 12345);
        let b = play_one_game(4, &mut eval_b, 12345);
        assert_eq!(a.outcome, b.outcome);
        assert_eq!(a.positions.len(), b.positions.len());
        for (pa, pb) in a.positions.iter().zip(&b.positions) {
            assert_eq!(pa.planes, pb.planes);
            assert_eq!(pa.visits, pb.visits);
        }
    }

    #[test]
    fn different_seeds_produce_different_games() {
        let mut eval_a = UniformEvaluator;
        let mut eval_b = UniformEvaluator;
        let a = play_one_game(4, &mut eval_a, 1);
        let b = play_one_game(4, &mut eval_b, 2);
        assert!(!a.positions.is_empty());
        assert!(!b.positions.is_empty());
        // Avant ce correctif, 4 parties self-play produisaient des fichiers
        // strictement identiques (revue finale du chantier initial) quelle
        // que soit la graine, puisqu'aucune n'était même consultée.
        let diverges = a.positions.len() != b.positions.len()
            || a.positions.iter().zip(&b.positions).any(|(pa, pb)| pa.planes != pb.planes);
        assert!(diverges, "deux graines distinctes doivent produire des trajectoires différentes");
    }

    #[test]
    fn dirichlet_noise_sums_to_one_and_has_one_entry_per_category() {
        let mut rng = Xorshift64::new(42);
        let noise = dirichlet_noise(&mut rng, 5, DIRICHLET_ALPHA);
        assert_eq!(noise.len(), 5);
        let total: f32 = noise.iter().sum();
        assert!((total - 1.0).abs() < 1e-4, "la somme du bruit de Dirichlet doit être 1, trouvé {total}");
    }

    #[test]
    fn dirichlet_noise_handles_a_single_category_without_panicking() {
        let mut rng = Xorshift64::new(7);
        let noise = dirichlet_noise(&mut rng, 1, DIRICHLET_ALPHA);
        assert_eq!(noise, vec![1.0]);
    }

    #[test]
    fn sample_weighted_always_returns_the_only_index_when_forced() {
        let mut rng = Xorshift64::new(99);
        for _ in 0..20 {
            assert_eq!(sample_weighted(&mut rng, &[7]), 0);
        }
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd native_engine && cargo test --bin selfplay_worker exploration_tests -- --nocapture`
Expected: FAIL to compile — `play_one_game` doesn't take a third argument yet, `Xorshift64`/`dirichlet_noise`/`sample_weighted`/`DIRICHLET_ALPHA` don't exist yet.

- [ ] **Step 3: Write the implementation**

Add this block to `native_engine/src/bin/selfplay_worker.rs`, right before `fn play_one_game`:

```rust
/// Bruit de Dirichlet au niveau de la racine + température d'ouverture :
/// sans ceci, le MCTS self-play est entièrement déterministe (mêmes
/// priors, même UCB, même coup choisi à chaque partie) — vérifié en revue
/// finale du chantier initial : 4 parties self-play produisaient des
/// fichiers strictement identiques. `DIRICHLET_ALPHA`/`DIRICHLET_EPSILON`
/// reprennent les valeurs standard AlphaZero pour un facteur de branchement
/// modéré (échecs : alpha=0.3, epsilon=0.25). `TEMPERATURE_PLIES` limite
/// l'échantillonnage proportionnel aux visites aux tout premiers coups de
/// la partie ; au-delà, exploitation pure (argmax), comme dans AlphaZero
/// une fois l'ouverture passée.
const DIRICHLET_ALPHA: f32 = 0.3;
const DIRICHLET_EPSILON: f32 = 0.25;
const TEMPERATURE_PLIES: usize = 6;

/// Générateur xorshift64* minimal (aucune dépendance externe — règle du
/// projet). N'a pas besoin d'être cryptographiquement sûr : seul le service
/// rendu est de casser le déterminisme entre parties/process, pas une
/// garantie de sécurité.
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 0x9e3779b97f4a7c15 } else { seed } }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Flottant uniforme dans [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Normale standard (Box-Muller), utilisée par `gamma_sample`.
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Graine distincte par partie et par processus : PID (distinct entre
/// workers concurrents, change au redémarrage — même raisonnement que le
/// nom de fichier des parties, cf. `main` plus bas), index de partie, et
/// horloge système, combinés par hachage plutôt que concaténés
/// arithmétiquement pour éviter toute structure visible entre graines
/// consécutives.
fn seed_for_game(game_index: u32) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    std::process::id().hash(&mut hasher);
    game_index.hash(&mut hasher);
    std::time::SystemTime::now().hash(&mut hasher);
    hasher.finish()
}

/// Échantillonne une loi Gamma(shape, 1) par la méthode de Marsaglia-Tsang
/// (2000), avec le correctif usuel pour `shape < 1` (Y = X · U^(1/shape),
/// X ~ Gamma(shape+1, 1)).
fn gamma_sample(rng: &mut Xorshift64, shape: f64) -> f64 {
    if shape < 1.0 {
        let u = rng.next_f64().max(1e-12);
        return gamma_sample(rng, shape + 1.0) * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let (x, v) = loop {
            let x = rng.next_normal();
            let v = 1.0 + c * x;
            if v > 0.0 {
                break (x, v);
            }
        };
        let v3 = v * v * v;
        let u = rng.next_f64();
        if u < 1.0 - 0.0331 * x.powi(4) || u.ln() < 0.5 * x * x + d * (1.0 - v3 + v3.ln()) {
            return d * v3;
        }
    }
}

/// Échantillon de Dirichlet(alpha, ..., alpha) sur `n` catégories : `n`
/// tirages Gamma(alpha, 1) indépendants, normalisés à somme 1. `n == 1` est
/// un cas valide (ply à un seul coup légal) : le seul tirage se normalise
/// trivialement à 1.0.
fn dirichlet_noise(rng: &mut Xorshift64, n: usize, alpha: f32) -> Vec<f32> {
    let samples: Vec<f64> = (0..n).map(|_| gamma_sample(rng, alpha as f64).max(1e-9)).collect();
    let total: f64 = samples.iter().sum();
    samples.iter().map(|s| (s / total) as f32).collect()
}

/// Tirage pondéré par `weights` (ex. les nombres de visites MCTS) : plus un
/// coup a été visité, plus il a de chances d'être choisi. `weights` doit
/// être non vide avec une somme strictement positive.
fn sample_weighted(rng: &mut Xorshift64, weights: &[u32]) -> usize {
    let total: u64 = weights.iter().map(|&w| w as u64).sum();
    assert!(total > 0, "sample_weighted appelé avec des poids tous nuls");
    let mut r = (rng.next_f64() * total as f64) as u64;
    for (i, &w) in weights.iter().enumerate() {
        if r < w as u64 {
            return i;
        }
        r -= w as u64;
    }
    weights.len() - 1
}
```

Replace `play_one_game`:

```rust
fn play_one_game(simulations_per_move: u32, evaluator: &mut dyn LeafEvaluator, rng_seed: u64) -> GameRecord {
    let mut state = initial_state();
    let mut positions = Vec::new();
    let mut rng = Xorshift64::new(rng_seed);
    loop {
        if let Some(v) = terminal_value(&state, state.player) {
            let outcome = if state.player == 0 { v } else { -v };
            return GameRecord { positions, outcome: outcome as i8 };
        }
        let planes = santoni_engine::mcts::encode_planes(&state);
        let mut mcts = Mcts::new(state, 1.5);
        mcts.ensure_root_expanded(evaluator);
        let n_edges = mcts.visit_counts().len();
        let noise = dirichlet_noise(&mut rng, n_edges, DIRICHLET_ALPHA);
        mcts.mix_root_noise(&noise, DIRICHLET_EPSILON);
        mcts.run(simulations_per_move, evaluator);
        let counts = mcts.visit_counts();
        let legal = legal_children(&state);
        let visits: Vec<(usize, u32)> = counts
            .iter()
            .map(|(mv, bld, n)| {
                let worker_slot = state.workers[state.player as usize]
                    .iter()
                    .position(|&c| c == mv.source)
                    .unwrap();
                (santoni_engine::mcts::action_index(mv, bld, worker_slot), *n)
            })
            .collect();
        let chosen = if positions.len() < TEMPERATURE_PLIES {
            let weights: Vec<u32> = counts.iter().map(|(_, _, n)| *n).collect();
            sample_weighted(&mut rng, &weights)
        } else {
            counts.iter().enumerate().max_by_key(|(_, (_, _, n))| *n).unwrap().0
        };
        let (_, _, after) = legal[chosen];
        positions.push(PositionRecord { planes, visits });
        state = after;
    }
}
```

Update the call site in `main()`:

```rust
        let record = play_one_game(simulations_per_move, evaluator.as_mut(), seed_for_game(i));
```

- [ ] **Step 4: Rebuild and run tests to verify they pass**

Run: `cd native_engine && cargo build --release --bin selfplay_worker && cargo test --bin selfplay_worker -- --nocapture`
Expected: PASS, including every pre-existing `socket_evaluator_tests` test (unaffected).

Then re-run the Python smoke/end-to-end suite against the rebuilt binary:

Run: `cd .. && QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_mcts_io_format.py tests/test_selfplay_worker_smoke.py tests/test_selfplay_end_to_end.py tests/test_selfplay_worker_single_move.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add native_engine/src/bin/selfplay_worker.rs
git commit -m "Bruit de Dirichlet racine + température d'ouverture pour l'exploration self-play"
```

---

## Final check (not a task — run after Task 5)

Run the full relevant test surface once more to confirm nothing else regressed:

```bash
cd native_engine && cargo test
cd .. && QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_mcts_io_format.py tests/test_az_trainer.py tests/test_selfplay_worker_smoke.py tests/test_selfplay_worker_single_move.py tests/test_selfplay_end_to_end.py tests/test_az_inference_server.py tests/test_az_inference_protocol.py tests/test_az_network.py tests/test_az_orchestrator.py -v
```

This plan deliberately does **not** touch: batched leaf evaluation across branches (blocker 4), campaign resumability/parallelism (blocker 5), or the zero-legal-move campaign termination bug (blocker 6) — all three are flagged in memory `santoni-mcts-az-no-power-chantier` as needing a design conversation with the user first, not a straight implementation. After this plan lands, update that memory: blockers 1-3 closed, 4-6 still open, campaign still not cleared to run until 4-6 are at least discussed.

# Pilote "ancre vécue" — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable, observable training campaign where the MCTS+NN candidate plays repeatedly against the live alpha-beta reference engine from one real "ancre" position (5 plies before an actual reference-vs-reference win), until a Clopper-Pearson 95% CI lower bound of ≥0.95 win rate over the last 40 deterministic evaluation games is reached — without ever risking unrecoverable data loss on interruption.

**Architecture:** A new Rust worker mode (`--vs-reference`) plays full games in-process with one side driven by MCTS+NN (optionally with Dirichlet noise/temperature) and the other by a direct, in-process call to the existing alpha-beta search (`santoni_search`), writing standard `GameRecord` JSON so the existing `Trainer`/`Orchestrator` machinery needs no changes. A new Python orchestrator (`AnchorPilotOrchestrator`) runs a pool of such workers in training mode (noisy) and a separate pool in evaluation mode (deterministic), alongside the existing `InferenceServer`/`Trainer`. A CLI (`tools/anchor_pilot_campaign.py`) wires it together: one-time anchor generation + worker-binary freezing, then a resumable, interruptible status loop.

**Tech Stack:** Rust (`native_engine`, existing `mcts`/`mcts_io` crate, no new external crates), Python 3 (`santoni_az`, `tools`, existing `torch`-based `Trainer`/`InferenceServer`, no scipy).

**Spec:** `docs/superpowers/specs/2026-10-21-mcts-az-live-anchor-pilot-design.md`

## Global Constraints

- No new Rust crates, no new Python dependencies (no scipy) — project-wide constraint, unchanged by this plan.
- Fresh network for this pilot, never a loaded checkpoint (spec §4).
- Single anchor game, single fil conducteur — no multi-anchor pooling, no held-out cross-validation (spec §2).
- No automatic stop ceiling if the win rate never converges — manual monitoring only (spec §3.3, decided 2026-10-21).
- Anchor distance is a CLI parameter, default 5 plies, not hard-coded (spec §3.1).
- Success signal: Clopper-Pearson 95% CI lower bound ≥ 0.95 over the 40 most recent **evaluation** games (deterministic, separate from training games) — spec §3.3.
- `tools/proven_anchor/` and `tools/proven_anchor_curriculum.py` stay untouched and unused by this pilot (spec, intro).
- Every write that must survive an interruption (games, checkpoints, the anchor state, the frozen worker binary) is atomic — the chantier's existing, non-negotiable convention (`write_game_atomically`, `Trainer.save_checkpoint`, `freeze_reference`/`freeze_checkpoint`).
- `simulations_per_move` default 32 for this pilot (chantier already measured 32 vs. 128 as indistinguishable in move quality at a large throughput cost).

## Review Focus

- A game record that drops the reference's own plies instead of recording them with a degenerate one-entry visit list would silently violate `Trainer._decode_position`'s strict `ply % 2` player-alternation convention — every training example from such a game would carry a flipped or garbage value target with no error anywhere. Task 4's test must assert every ply is present, in order, not just the candidate's.
- The candidate must play the *same* physical player (0 or 1) for an entire game even though `state.player` alternates every ply — a mix-up trains the network to imitate the side that was actually losing. Task 4's Rust unit test must pin this directly.
- Rebuilding `native_engine` mid-campaign must be detected, not silently blended into one campaign's data — the exact "Critical" bug class this chantier already found once for `--checkpoint` mismatches (`tools/mcts_az_vs_reference_campaign.py`'s `freeze_checkpoint`). Task 6's freeze must hash the *incoming* binary, not just compare the frozen copy to itself.
- An anchor position very close to the start of its seed game (fewer than `--back-plies` plies available) must fail with a clear error, not silently index negative or crash deep inside `play_reference_vs_reference`'s caller. Task 5 must test this directly.
- Clopper-Pearson computed on fewer than 40 available evaluation games (true for the first minutes of any real run) must still return a sane, non-crashing lower bound rather than assuming a full window. Task 2 and Task 7 must both cover `n < 40`.

---

## Task 1: Finish anchor-game progress visibility (loose end from this session)

**Files:**
- Modify: `tools/proven_anchor/anchor_games.py` (`find_anchor_games`)
- Test: `tests/test_proven_anchor_anchor_games.py` (already written, one test currently RED)

**Interfaces:**
- Consumes: nothing new.
- Produces: `find_anchor_games(..., progress=_progress)` now forwards a prefixed `progress` into `play_reference_vs_reference`/`scan_game_for_forced_win` and reports one summary line per opening tried — used by Task 5's `prepare_anchor` for visibility into the one-time seed-game generation.

- [ ] **Step 1: Confirm the existing failing test still fails for the right reason**

Run: `python3 -m pytest tests/test_proven_anchor_anchor_games.py::test_find_anchor_games_reports_progress_per_opening_and_forwards_to_sub_calls -v`
Expected: FAIL — `'inner-play' in joined` assertion error (progress never forwarded).

- [ ] **Step 2: Implement the forwarding in `find_anchor_games`**

For each opening in `openings`, build a scoped progress function that prefixes `play_reference_vs_reference`'s and `scan_game_for_forced_win`'s own messages with the opening's 1-based index and the total count (e.g. `f"[opening {i+1}/{len(openings)}] {message}"`), pass it as `progress=` to both calls, and after each opening emit one summary line reporting anchors found so far vs. `minimum` (matching the test's `'1/2'`, `'0/1'`, `'1/1'` assertions).

- [ ] **Step 3: Run the full test file and verify all green**

Run: `python3 -m pytest tests/test_proven_anchor_anchor_games.py -v`
Expected: all PASS (13 tests).

- [ ] **Step 4: Commit**

```bash
git add tools/proven_anchor/anchor_games.py tests/test_proven_anchor_anchor_games.py
git commit -m "feat: progress visibility for anchor-game search and seed-game generation"
```

## Task 2: Clopper-Pearson lower bound (pure Python, no scipy)

**Files:**
- Create: `santoni_az/stats.py`
- Test: `tests/test_stats.py`

**Interfaces:**
- Produces: `clopper_pearson_lower_bound(successes: int, n: int, confidence: float = 0.95) -> float`. Raises `ValueError` if `n <= 0`, `successes < 0`, or `successes > n`. Used by Task 7's `AnchorPilotOrchestrator.status()`.

- [ ] **Step 1: Write the failing tests**

```python
def test_zero_successes_gives_zero_lower_bound():
    assert clopper_pearson_lower_bound(0, 40) == 0.0

def test_all_successes_matches_the_closed_form_power_law():
    # Clopper-Pearson's k==n special case solves p**n == alpha/2 directly.
    expected = 0.025 ** (1 / 40)
    assert clopper_pearson_lower_bound(40, 40) == pytest.approx(expected, abs=1e-9)

def test_lower_bound_increases_monotonically_with_more_successes_at_fixed_n():
    low = clopper_pearson_lower_bound(30, 40)
    mid = clopper_pearson_lower_bound(35, 40)
    high = clopper_pearson_lower_bound(39, 40)
    assert low < mid < high

def test_works_below_the_eventual_40_game_window():
    # n < 40 must not crash or be treated as a full window.
    assert 0.0 <= clopper_pearson_lower_bound(3, 5) < 1.0

def test_rejects_successes_greater_than_n():
    with pytest.raises(ValueError):
        clopper_pearson_lower_bound(5, 4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_stats.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError` (module doesn't exist yet).

- [ ] **Step 3: Implement `clopper_pearson_lower_bound` in `santoni_az/stats.py`**

Binary-search `p` in `[0, 1]` for the value where the binomial survival function `sum(math.comb(n, i) * p**i * (1 - p)**(n - i) for i in range(successes, n + 1))` crosses `confidence`'s complement halved (`(1 - confidence) / 2`) — the survival function is monotonically increasing in `p`, so a plain bisection to a tight tolerance (e.g. 1e-10) is sufficient; no scipy, no new dependency. `successes == 0` returns `0.0` directly (the survival function is identically 1 for every `p > 0`, so no crossing exists to search for).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_stats.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add santoni_az/stats.py tests/test_stats.py
git commit -m "feat: add Clopper-Pearson lower bound for the anchor pilot's win-rate gate"
```

## Task 3: Rust — in-process reference move for the no-power subgame

**Files:**
- Modify: `native_engine/src/lib.rs`

**Interfaces:**
- Produces: `pub fn reference_turn(state: &State, seconds: f64) -> CTurn` — the best line's first `CTurn` (`.actions[..length]` plus `.after`), computed by calling the existing `santoni_search` in-process (no FFI/socket round trip). Used by Task 4's new worker mode.

- [ ] **Step 1: Write the failing test**

Add to the existing `#[cfg(test)] mod tests` block in `lib.rs` (same module the other `santoni_search`-adjacent tests already live in):

```rust
#[test]
fn reference_turn_returns_a_legal_turn_for_a_simple_no_power_position() {
    let state = State { workers: [[0, 4, -1, -1], [20, 24, -1, -1]], ..Default::default() };
    let turn = reference_turn(&state, 0.2);
    assert!(turn.length == 1 || turn.length == 2);
    assert_ne!(turn.after, state); // a real move happened
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd native_engine && cargo test reference_turn_returns_a_legal_turn`
Expected: FAIL — `reference_turn` not defined.

- [ ] **Step 3: Implement `reference_turn`**

Build a `CState`-compatible raw pointer to `state` (same pattern `santoni_search`'s own callers already use at the FFI boundary — here entirely in-process, no `unsafe extern "C"` crossing needed beyond what calling `santoni_search` itself requires), call `santoni_search(state_ptr, seconds, &mut analysis, None, None)`, and return `analysis.turns[0]`. This is a thin wrapper — no search logic is duplicated; it reuses the exact same alpha-beta code path `native_search` already calls via ctypes everywhere else in this project.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd native_engine && cargo test reference_turn_returns_a_legal_turn`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add native_engine/src/lib.rs
git commit -m "feat: expose an in-process reference turn for the self-play worker"
```

## Task 4: Rust — new `--vs-reference` worker mode

**Files:**
- Modify: `native_engine/src/bin/selfplay_worker.rs`
- Test: new `#[cfg(test)]` module in the same file, plus a new test function in `tests/test_selfplay_end_to_end.py`

**Interfaces:**
- Consumes: `santoni_engine::reference_turn` (Task 3), `santoni_engine::mcts::action_index` (existing, already used by `play_one_game`).
- Produces: CLI mode `--vs-reference --input-state <path> --out-dir <dir> --games <n> --simulations-per-move <n> --inference-socket <path> --reference-seconds <f64> [--deterministic]`, writing `GameRecord`s via the existing `write_game_atomically` with the existing `game-{pid}-{counter:06}.json` naming (same convention continuous self-play already uses, so `Trainer`/tests' `glob("game-*.json")` keeps matching unchanged). Internally: `fn play_one_game_vs_reference(initial: State, simulations_per_move: u32, evaluator: &mut dyn LeafEvaluator, reference_seconds: f64, deterministic: bool, rng_seed: u64) -> GameRecord`.

- [ ] **Step 1: Write the failing Python end-to-end test (defines the external CLI contract first)**

In `tests/test_selfplay_end_to_end.py`, add (mirroring `test_worker_produces_games_via_real_inference_server`'s existing setup):

```python
def test_vs_reference_mode_produces_games_with_every_ply_recorded(tmp_path):
    # ... same InferenceServer setup as the existing test ...
    # encode a simple no-power State (player=1) to tmp_path / "anchor.bin" via santorini.native.encode
    result = subprocess.run(
        [str(BINARY), "--vs-reference", "--input-state", str(anchor_path),
         "--out-dir", str(out_dir), "--games", "1", "--simulations-per-move", "8",
         "--inference-socket", socket_path, "--reference-seconds", "0.2"],
        timeout=60, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(next(out_dir.glob("game-*.json")).read_text())
    assert data["outcome"] in (-1, 1)
    visit_counts = [len(p["visits"]) for p in data["positions"]]
    # Every other ply is the reference's degenerate one-entry record,
    # starting from the second ply (the anchor's player, player 1, moves
    # first == the candidate); a skipped ply here would break
    # Trainer._decode_position's ply-alternation convention silently.
    assert len(visit_counts) >= 2
    for i, count in enumerate(visit_counts):
        if i % 2 == 1:
            assert count == 1 and data["positions"][i]["visits"][0][1] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_selfplay_end_to_end.py::test_vs_reference_mode_produces_games_with_every_ply_recorded -v`
Expected: FAIL — `--vs-reference` not recognized by the current binary (returncode != 0).

- [ ] **Step 3: Write the failing Rust unit test**

Add to a new `#[cfg(test)] mod vs_reference_tests` in `selfplay_worker.rs`:

```rust
#[test]
fn every_ply_is_recorded_and_the_candidate_keeps_the_same_physical_player() {
    let initial = State { workers: [[0, 4, -1, -1], [20, 24, -1, -1]], player: 1, ..Default::default() };
    let mut evaluator = UniformEvaluator;
    let record = play_one_game_vs_reference(initial, 4, &mut evaluator, 0.05, true, 1);
    // Reference plies (degenerate, weight-1 single entry) and candidate
    // plies (a real MCTS distribution) must strictly alternate starting
    // with the candidate (player 1, the anchor's own player, moves
    // first) -- a gap here means a ply was silently dropped from the
    // record, corrupting Trainer's ply-parity sign convention.
    for (i, position) in record.positions.iter().enumerate() {
        let is_reference_ply = position.visits.len() == 1 && position.visits[0].1 == 1;
        assert_eq!(is_reference_ply, i % 2 == 1, "ply {i} has the wrong record shape");
    }
    assert!(record.positions.len() >= 2);
}
```

- [ ] **Step 4: Run test to verify it fails**

Run: `cd native_engine && cargo test every_ply_is_recorded_and_the_candidate_keeps`
Expected: FAIL — `play_one_game_vs_reference` not defined.

- [ ] **Step 5: Implement `play_one_game_vs_reference` and the `--vs-reference` CLI dispatch**

Mirror `play_one_game`'s loop structure exactly (same `terminal_value` check, same `GameRecord`/`PositionRecord` shapes), with two differences: the starting state is the function's `initial` parameter instead of `initial_state()`, and `candidate_player` is fixed to `initial.player` once before the loop starts. On each ply: if `state.player == candidate_player`, run MCTS exactly as `play_one_game` does (apply Dirichlet noise/temperature-sampling unless `deterministic` is set, per spec §3.2's call to apply it on *every* candidate ply in this mode, not just the first `TEMPERATURE_PLIES`); otherwise call `reference_turn(&state, reference_seconds)` (Task 3), convert its `CTurn` into a `PositionRecord { planes: encode_planes(&state), visits: vec![(action_index(...), 1)] }` using the same `action_index`/`worker_slot` computation `play_one_game` already does, and advance `state` to `turn.after`. Push a `PositionRecord` on *every* ply, candidate or reference, in play order — never filter the reference's plies out of `positions`. Wire the new CLI branch in `main()` the same way `--single-move`/`--encode-planes` are dispatched (checked before the continuous-mode argument parsing), reading the extra flags and looping over `--games` exactly like the continuous-mode tail already does (including the `INTERRUPTED` check between games).

- [ ] **Step 6: Run the Rust test to verify it passes**

Run: `cd native_engine && cargo test every_ply_is_recorded_and_the_candidate_keeps`
Expected: PASS.

- [ ] **Step 7: Rebuild the release binary**

Run: `cd native_engine && cargo build --release`
Expected: clean build.

- [ ] **Step 8: Run the Python end-to-end test to verify it now passes**

Run: `python3 -m pytest tests/test_selfplay_end_to_end.py::test_vs_reference_mode_produces_games_with_every_ply_recorded -v`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add native_engine/src/bin/selfplay_worker.rs tests/test_selfplay_end_to_end.py
git commit -m "feat: add --vs-reference worker mode for the live anchor pilot"
```

## Task 5: Python — anchor state generation (one-time, resumable)

**Files:**
- Create: `tools/anchor_pilot_campaign.py`
- Test: `tests/test_anchor_pilot_campaign.py`

**Interfaces:**
- Consumes: `tools.proven_anchor.anchor_games.play_reference_vs_reference` (Task 1's progress-enabled version), `santorini.native.encode`.
- Produces: `prepare_anchor(out_dir: Path, opening_job: dict, back_plies: int, seconds: float, reference_binary, progress=...) -> Path` — the path to the written `anchor_state.bin`. Resumable: a second call with the same `out_dir` returns the existing file's path without replaying the seed game. Used by Task 8's `main()`.

- [ ] **Step 1: Write the failing tests**

```python
def test_prepare_anchor_writes_the_position_back_plies_before_the_win(tmp_path, monkeypatch):
    # monkeypatch play_reference_vs_reference (as the existing anchor_games
    # tests already do) to return a known 10-position fake game; assert the
    # written anchor_state.bin round-trips (via santorini.native decode) to
    # positions[10 - 1 - 5] for back_plies=5.

def test_prepare_anchor_is_idempotent_on_a_second_call(tmp_path, monkeypatch):
    # second call must not invoke play_reference_vs_reference again
    # (monkeypatch raises if called) and must return the same path.

def test_prepare_anchor_raises_a_clear_error_when_the_seed_game_is_too_short(tmp_path, monkeypatch):
    # a 3-position fake game with back_plies=5 must raise ValueError
    # naming both numbers, not index out of range.
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_anchor_pilot_campaign.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `prepare_anchor`**

If `out_dir / "anchor_state.bin"` already exists, return it immediately (idempotent short-circuit, same spirit as `freeze_reference`). Otherwise call `play_reference_vs_reference(opening_job, seconds, reference_binary, progress=progress)`, compute `index = len(positions) - 1 - back_plies`, raise `ValueError(f"seed game only has {len(positions)} plies, need at least {back_plies + 1} to go back {back_plies} plies")` if `index < 0`, encode `positions[index]` via `santorini.native.encode`, write it atomically (temp file + `os.replace`, same convention as every other atomic write in this chantier) to `anchor_state.bin`, and also write a small `anchor_manifest.json` (opening job, `back_plies`, seed-game length, the anchor's own `player`) for later inspection.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_anchor_pilot_campaign.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/anchor_pilot_campaign.py tests/test_anchor_pilot_campaign.py
git commit -m "feat: generate and freeze the live anchor pilot's starting position"
```

## Task 6: Python — freeze the worker binary (provenance pin)

**Files:**
- Modify: `tools/anchor_pilot_campaign.py`
- Test: `tests/test_anchor_pilot_campaign.py`

**Interfaces:**
- Produces: `freeze_worker_binary(out_dir: Path) -> tuple[Path, str]` — path to the frozen copy and its sha256. Used by Task 7/8.

- [ ] **Step 1: Write the failing tests**

```python
def test_freeze_worker_binary_copies_and_records_sha256(tmp_path):
    # point at a small fake "binary" file, assert a frozen copy + manifest
    # with a matching sha256 both exist.

def test_freeze_worker_binary_is_idempotent_when_unchanged(tmp_path):
    # second call with the same source file succeeds, returns the same hash.

def test_freeze_worker_binary_rejects_a_changed_source_binary(tmp_path):
    # freeze once, then overwrite the source file's bytes, call again:
    # must raise with a message naming the frozen generation's hash,
    # not silently keep serving the stale frozen copy.
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_anchor_pilot_campaign.py -k freeze_worker_binary -v`
Expected: FAIL — `ModuleNotFoundError`/`AttributeError`.

- [ ] **Step 3: Implement `freeze_worker_binary`**

Mirror `freeze_checkpoint`'s shape, but apply this plan's own Review Focus lesson directly: hash the *incoming* current binary (`native_engine/target/release/selfplay_worker`) every call, and compare it against the hash recorded in the manifest from the frozen copy — not merely the frozen copy's hash against itself. Raise a clear `ValueError` naming both hashes on mismatch (the exact Critical bug class this chantier already fixed once for `--checkpoint`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_anchor_pilot_campaign.py -k freeze_worker_binary -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/anchor_pilot_campaign.py tests/test_anchor_pilot_campaign.py
git commit -m "feat: freeze and pin the worker binary for the live anchor pilot"
```

## Task 7: Python — AnchorPilotOrchestrator

**Files:**
- Create: `santoni_az/anchor_pilot.py`
- Test: `tests/test_anchor_pilot_orchestrator.py`

**Interfaces:**
- Consumes: `santoni_az.inference_server.InferenceServer`, `santoni_az.trainer.Trainer` (both unchanged), `santoni_az.stats.clopper_pearson_lower_bound` (Task 2), the frozen worker binary + anchor state path (Tasks 5/6).
- Produces: `AnchorPilotOrchestrator(run_dir, anchor_state_path, worker_binary_path, train_worker_count, eval_worker_count, simulations_per_move=32, reference_seconds=5.0)` with `start()`, `stop()`, `status()`. `status()` returns `{"generation", "last_loss", "games_available", "eval_games_available", "workers_alive", "eval_wins", "eval_n", "eval_ci_lower"}`.

- [ ] **Step 1: Write the failing tests**

```python
def test_start_runs_both_training_and_eval_workers(tmp_path):
    # real build, real InferenceServer: after start(), status()["workers_alive"]
    # covers both pools (>= train_worker_count + eval_worker_count while both alive).

def test_eval_games_land_in_a_separate_directory_from_training_games(tmp_path):
    # after a few seconds running, games_dir and eval_games_dir both have
    # files, and they are disjoint sets.

def test_status_reports_a_clopper_pearson_lower_bound_from_eval_games_on_disk(tmp_path):
    # pre-populate eval_games_dir with a handful of fake GameRecord files
    # (outcome known), start with eval_worker_count=0 so the count doesn't
    # move, and check status()["eval_ci_lower"] matches
    # clopper_pearson_lower_bound computed by hand on those same files.

def test_restart_does_not_lose_already_played_eval_games(tmp_path):
    # same resumability shape as test_az_orchestrator's
    # test_restart_resumes_from_last_checkpoint: stop, create a new
    # instance pointed at the same run_dir, confirm eval_games_available
    # never decreases.
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_anchor_pilot_orchestrator.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement `AnchorPilotOrchestrator`**

`start()` calls `self.trainer.load_latest_checkpoint()` exactly as `Orchestrator.start()` does, before launching any worker — this single existing mechanism gives both spec requirements for free: a brand-new `run_dir` has no checkpoints, so the net stays at its fresh `net_factory()` initialization (spec §4); an interrupted-and-relaunched `run_dir` loads its own latest checkpoint (spec §3.4's resumability), with no special-casing needed to tell the two situations apart. Compose one `InferenceServer` (shared by both worker pools, same socket, same pattern as `Orchestrator`) and one `Trainer` pointed at `run_dir / "games"` (training games only — `eval_games` is never passed to `Trainer`, so evaluation data structurally cannot leak into training, satisfying spec §3.3's separation without needing a flag check anywhere). `start()` launches `train_worker_count` subprocesses of the frozen binary in `--vs-reference` mode (no `--deterministic`) writing into `games/`, and `eval_worker_count` subprocesses in `--vs-reference --deterministic` mode writing into `eval_games/`, both pointed at `anchor_state_path`; keeps the same background `_train_loop` thread as `Orchestrator` (verbatim reuse of its checkpointing logic — `MIN_CHECKPOINT_INTERVAL_S`-gated, train-then-maybe-save). `status()`'s `eval_ci_lower` reads the 40 most-recently-modified files under `eval_games/` (same `stat().st_mtime` sort `Trainer._rebuild_buffer_from_disk` already uses) and calls `clopper_pearson_lower_bound(wins, n)` on their `outcome` fields, using the anchor's own `player` (from Task 5's manifest) to turn each game's `outcome` into "did the candidate win" — this needs the manifest's recorded `anchor player` plus the stored `GameRecord.outcome`'s existing player-0-relative sign convention (`play_one_game`'s own convention, unchanged by Task 4) to compute "candidate won" correctly from each file alone, without re-deriving it from `positions`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_anchor_pilot_orchestrator.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add santoni_az/anchor_pilot.py tests/test_anchor_pilot_orchestrator.py
git commit -m "feat: add AnchorPilotOrchestrator for the live anchor training+eval loop"
```

## Task 8: Python — CLI entrypoint with progress logging and clean shutdown

**Files:**
- Modify: `tools/anchor_pilot_campaign.py`
- Test: `tests/test_anchor_pilot_campaign.py`

**Interfaces:**
- Produces: `main()` (argparse: `--out`, `--corpus` default the existing no-power corpus, `--back-plies` default 5, `--train-workers`, `--eval-workers`, `--simulations-per-move` default 32, `--reference-seconds` default 5.0), the module's `if __name__ == '__main__': main()` entry point.

- [ ] **Step 1: Write the failing test**

```python
def test_main_wires_prepare_anchor_freeze_and_orchestrator_together(tmp_path, monkeypatch):
    # monkeypatch prepare_anchor, freeze_worker_binary, and
    # AnchorPilotOrchestrator (as a fake recording its constructor args and
    # a start()/stop() call count) to verify main() calls them in order
    # with the parsed CLI arguments, without needing a real multi-second run.
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_anchor_pilot_campaign.py -k main_wires -v`
Expected: FAIL — `AttributeError: module has no attribute 'main'`.

- [ ] **Step 3: Implement `main()`**

Parse arguments, call `prepare_anchor`/`freeze_worker_binary` (Tasks 5/6), construct and `start()` the `AnchorPilotOrchestrator` (Task 7), install SIGTERM/SIGINT handlers that call `stop()` cleanly (same shape as every other launcher in this chantier, `selfplay-runs/run.py`/`run2.py`), then loop printing one status line every 10 seconds (generation, games_available, eval_n, eval_ci_lower, workers_alive) to stdout until interrupted — same convention as `run.log`'s existing status-line format, satisfying spec §3.4's visibility requirement directly.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_anchor_pilot_campaign.py -k main_wires -v`
Expected: PASS.

- [ ] **Step 5: Run the full new test suite**

Run: `python3 -m pytest tests/test_stats.py tests/test_anchor_pilot_campaign.py tests/test_anchor_pilot_orchestrator.py tests/test_selfplay_end_to_end.py tests/test_proven_anchor_anchor_games.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add tools/anchor_pilot_campaign.py tests/test_anchor_pilot_campaign.py
git commit -m "feat: add CLI entrypoint for the live anchor pilot campaign"
```

## Final smoke check (not a unit test — do this before handing back for the real launch decision)

- [ ] Run the whole project's test suite once (`pytest` from the repo root) to confirm nothing outside this plan's own files regressed.
- [ ] Run `python3 -m tools.anchor_pilot_campaign --out /tmp/anchor-pilot-smoke --train-workers 1 --eval-workers 1 --reference-seconds 0.3 --simulations-per-move 8` for ~30 seconds, confirm: anchor generation completes and logs progress, the worker binary gets frozen, status lines appear, both `games/` and `eval_games/` accumulate files, and `kill -TERM` on the process group stops it cleanly with no `.tmp` files left behind.
- [ ] Report the smoke check's numbers back before suggesting the real campaign's worker counts/budgets.

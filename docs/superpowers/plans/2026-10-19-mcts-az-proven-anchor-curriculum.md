# Noyau exact par analyse rétrograde locale — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace pure self-play (empirically flat, no external-validity gain over 1.2M+ games) with a proven-anchor curriculum: exact win/loss labels from the reference engine's own proof search, derived by walking backward through real anchor games, used as supervised training signal and validated against memorization before widening further back.

**Architecture:** A new `tools/proven_anchor/` package provides pure, independently-testable building blocks (plane encoding, board symmetry, reference-engine verdicts, layer construction, anchor-game sourcing, validation) wired together by one orchestration script (`tools/proven_anchor_curriculum.py`) that repeats: build a layer -> train -> validate -> widen or stop. `santoni_az.trainer.Trainer` gains one new method to accept these exact-labeled examples; no other existing-infrastructure file changes.

**Tech Stack:** Python 3.10+, PyTorch (existing `santoni_az`), the existing Rust `native_engine` (one small additive CLI mode), pytest. No new dependency (confirmed: `scipy` is not a project dependency — see Task 7, bootstrap-resampling CI instead of Clopper-Pearson via scipy, matching the existing `tools/mcts_az_vs_reference_campaign.py::bootstrap` pattern).

**Spec:** `docs/superpowers/specs/2026-10-19-mcts-az-proven-anchor-curriculum-design.md`

## Global Constraints

- No new external dependency, Python or Rust (project-wide rule; confirmed no `scipy` in `pyproject.toml` — use resampling, not a closed-form beta quantile).
- Self-play (`santoni_az.orchestrator.Orchestrator` / `selfplay-runs/run2.py`) must not run during core construction/validation (spec §3.3) — nothing in this plan starts it, and the orchestration script (Task 8) must not import or launch it.
- The network trained by this plan starts fresh (`PolicyValueNet()` default init); never loads `gen-43227.pt` or any prior self-play checkpoint (spec §4).
- At least 2 independent anchor games are required before any layer-1 validation runs — not deferred to later replication (spec §2, §3.7).
- A layer's validation must re-check every previously-validated layer's held-out set too, not only the newest one (spec §2 — fine-tuning is cumulative, not reset per layer).
- Unresolved branches (reference engine doesn't prove within budget) never block a layer: train on the resolved subset, track the resolved fraction (spec §3.2).
- The reference engine binary used for proof-search must be frozen and sha256-pinned before use (`tools.mcts_az_vs_reference_campaign.freeze_reference`, already exists) so a mid-construction rebuild cannot silently change what "proven" means partway through.
- No time/compute ceiling on the overall widening process (spec §2, explicit user decision) — but each individual proof-search call still takes an explicit, tunable `seconds` budget; "no ceiling" applies to how many layers get built, not to any single call.

## Review Focus

- A fully-resolved ancestor position where **every** legal child is proven losing for the mover (a genuine local forced loss, not a forced win) — Task 4's `build_layer` must label `ancestor_verdict=False` here, not assume every ancestor on an anchor line is automatically winning.
- A layer where some children are proven and others are not (budget exhausted) — Task 4/6 must train only on the resolved subset and must never treat an unresolved (`None`) verdict as if it were `False`.
- Two anchor games whose scanned positions turn out identical or near-identical (e.g. same opening, same early deviation) — Task 5/7 must not let the held-out check compare a position against itself under a different name.
- A symmetry transform (Task 2) applied to a real position producing output that fails `Position.validate()` — must raise loudly in the task's own tests, never get silently fed into training.
- The orchestration loop (Task 8) reaching the spec's "échec informatif" condition (sign-match never clears baseline on the first layers) — must stop and report distinctly from a success, not keep widening or crash.

---

### Task 1: Expose `encode_planes` for arbitrary positions from Python

**Files:**
- Modify: `native_engine/src/bin/selfplay_worker.rs` (new `--encode-planes` CLI mode, alongside the existing `--single-move` branch in `main()`)
- Create: `tools/proven_anchor/__init__.py` (empty)
- Create: `tools/proven_anchor/planes.py`
- Test: `tests/test_proven_anchor_planes.py`

**Interfaces:**
- Produces: `tools.proven_anchor.planes.compute_planes(position: santorini.engine.Position) -> bytes` — exactly `PLANE_BYTES` (150) raw bytes, byte-identical to what `native_engine::mcts::encode_planes` would produce for the equivalent native `State`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_proven_anchor_planes.py
import json
from pathlib import Path
from santorini.engine import Position
from tools.proven_anchor.planes import compute_planes

ROOT = Path(__file__).resolve().parents[1]

def initial_no_power_position():
    # Matches native_engine/src/bin/selfplay_worker.rs::initial_state() exactly:
    # workers [[0,4,-1,-1],[20,24,-1,-1]], everything else default.
    return Position(workers=((0, 4), (20, 24)))

def test_compute_planes_matches_a_real_recorded_self_play_opening():
    # Any self-play game's ply-0 planes are encode_planes(initial_state()),
    # recorded by play_one_game before the first move is chosen -- ground
    # truth to compare against, not just a plumbing/determinism check.
    sample_game = next(ROOT.glob("selfplay-runs/*/games/game-*.json"))
    recorded = json.loads(sample_game.read_text())["positions"][0]["planes"]

    produced = compute_planes(initial_no_power_position())

    assert list(produced) == recorded

def test_compute_planes_is_deterministic():
    position = initial_no_power_position()
    assert compute_planes(position) == compute_planes(position)

def test_compute_planes_differs_for_a_different_position():
    a = initial_no_power_position()
    b = Position(workers=((0, 4), (20, 24)), heights=(1,) + (0,) * 24)
    assert compute_planes(a) != compute_planes(b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_proven_anchor_planes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tools.proven_anchor'`

- [ ] **Step 3: Add `--encode-planes` to `native_engine/src/bin/selfplay_worker.rs`**

In `main()`, add a branch parallel to the existing `--single-move` one (before it or after, doesn't matter): if `args.iter().any(|a| a == "--encode-planes")`, read `--input-state`/`--output-planes` paths via `parse_arg`, decode the input file into a `State` the same way `run_single_move` already does (`std::ptr::read_unaligned(bytes.as_ptr() as *const State)`, then `assert!(is_in_scope(&state))`), call `santoni_engine::mcts::encode_planes(&state)`, and write the resulting `[u8; PLANE_BYTES]` to `--output-planes` atomically (write to a `.tmp` sibling, then `std::fs::rename`, exactly like `run_single_move`'s own output-writing tail does for `--output-move`).

- [ ] **Step 4: Rebuild the release binary**

Run: `cd native_engine && cargo build --release --bin selfplay_worker`
Expected: builds cleanly, no warnings about the new branch.

- [ ] **Step 5: Implement `compute_planes(position)` in `tools/proven_anchor/planes.py`**

```python
def compute_planes(position: "santorini.engine.Position") -> bytes:
    ...
```

Mirrors `tools.mcts_az_vs_reference_campaign.candidate_move`'s subprocess pattern: `santorini.native.encode(position)` to a `CState`, write `bytes(state)` to a temp `state.bin`, invoke `native_engine/target/release/selfplay_worker --encode-planes --input-state <state.bin> --output-planes <planes.bin>` via `subprocess.run(..., check=True)`, read back and return the raw bytes of `planes.bin`. Raise `ValueError` with the subprocess's stderr if it exits non-zero (same style as `candidate_move`).

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_proven_anchor_planes.py -v`
Expected: PASS (requires at least one real game file already on disk under `selfplay-runs/*/games/` — true in this repo; if the glob ever comes up empty in a fresh clone, the test should skip with a clear reason rather than error, since this plan does not ship fixture game files).

- [ ] **Step 7: Commit**

```bash
git add native_engine/src/bin/selfplay_worker.rs tools/proven_anchor/__init__.py tools/proven_anchor/planes.py tests/test_proven_anchor_planes.py
git commit -m "feat: expose encode_planes for arbitrary positions via a worker CLI mode"
```

---

### Task 2: Board symmetry transforms for `Position`

**Files:**
- Create: `tools/proven_anchor/symmetry.py`
- Test: `tests/test_proven_anchor_symmetry.py`

**Interfaces:**
- Produces: `tools.proven_anchor.symmetry.BOARD_SYMMETRIES: tuple[Callable[[Position], Position], ...]` — the 8 elements of the dihedral group of the 5×5 grid (identity + 3 rotations + 4 reflections), each mapping a valid no-power `Position` to another valid `Position` representing the geometrically transformed board, with `player`/`powers`/`winner`/`reason`/`extra`/`resume` left unchanged (only `heights`, `domes`, `workers` are cell-reindexed; confirmed by reading `santorini/engine.py`'s cell convention — `cell = y*5+x`, `x = cell%5`, `y = cell//5`, no hidden per-cell asymmetry for the no-power subgame).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_proven_anchor_symmetry.py
from santorini.engine import Position
from tools.proven_anchor.symmetry import BOARD_SYMMETRIES

def a_real_no_power_position():
    heights = [0] * 25
    heights[6] = 1
    heights[18] = 2
    return Position(heights=tuple(heights), workers=((0, 4), (20, 24)))

def test_there_are_eight_symmetries():
    assert len(BOARD_SYMMETRIES) == 8

def test_every_symmetry_produces_a_valid_position():
    position = a_real_no_power_position()
    for transform in BOARD_SYMMETRIES:
        transformed = transform(position)
        transformed.validate()  # raises if invalid

def test_applying_a_quarter_turn_four_times_returns_to_the_original():
    position = a_real_no_power_position()
    quarter_turn = next(t for t in BOARD_SYMMETRIES if t is not BOARD_SYMMETRIES[0])
    result = position
    for _ in range(4):
        result = quarter_turn(result)
    assert result == position

def test_symmetries_preserve_non_board_fields():
    position = a_real_no_power_position()
    for transform in BOARD_SYMMETRIES:
        transformed = transform(position)
        assert transformed.player == position.player
        assert transformed.powers == position.powers
        assert transformed.winner == position.winner

def test_symmetries_are_distinct_on_an_asymmetric_position():
    # A position with no board symmetry of its own: all 8 transformed
    # results must differ pairwise, otherwise two "symmetries" collapsed
    # to the same transform (a bug, not a property of this position).
    position = a_real_no_power_position()
    results = [transform(position) for transform in BOARD_SYMMETRIES]
    assert len({r for r in results}) == 8
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_proven_anchor_symmetry.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `BOARD_SYMMETRIES` in `tools/proven_anchor/symmetry.py`**

Build the 8 transforms from two generators over `(x, y) = (cell % 5, cell // 5)`: rotate90 `(x, y) -> (y, 4 - x)` and flip `(x, y) -> (4 - x, y)`, composed to the full dihedral group (identity, rotate90, rotate180, rotate270, flip, flip+rotate90, flip+rotate180, flip+rotate270). A single `_remap_cell(cell, transform_fn) -> int` helper reindexes one cell; apply it to every entry of `heights` (reindex the tuple), to `domes` (rebuild the bitmask bit-by-bit via the same remap), and to every worker coordinate in `workers` (remap each int in each tuple). Build each `Position` via `dataclasses.replace(position, heights=..., domes=..., workers=...)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_proven_anchor_symmetry.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tools/proven_anchor/symmetry.py tests/test_proven_anchor_symmetry.py
git commit -m "feat: add board symmetry transforms for no-power positions"
```

---

### Task 3: Reference-engine exact child verdict

**Files:**
- Create: `tools/proven_anchor/verdicts.py`
- Test: `tests/test_proven_anchor_verdicts.py`

**Interfaces:**
- Consumes: `santorini.native.native_search` (existing), `santorini.search.MATE` (existing constant, `=100_000`).
- Produces: `tools.proven_anchor.verdicts.evaluate_child_verdict(child_position: Position, mover: int, seconds: float) -> bool | None` — `True` if the move that produced `child_position` is a proven win for `mover`, `False` if proven loss, `None` if unresolved within `seconds`. Also: `tools.proven_anchor.verdicts.mate_distance(analysis) -> int | None` — `MATE - abs(analysis.score)` if `analysis.proven`, else `None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_proven_anchor_verdicts.py
from santorini.engine import Position
from santorini.native import native_search
from tools.proven_anchor.verdicts import evaluate_child_verdict, mate_distance

def test_mate_distance_is_none_when_not_proven():
    heights = [0] * 25
    position = Position(heights=tuple(heights), workers=((0, 4), (20, 24)))
    analysis = native_search(position, 0.05)
    if analysis.proven:
        return  # trivially proven this fast at the opening is not expected, but don't flake if it is
    assert mate_distance(analysis) is None

def winning_in_one_position():
    # Reused fixture, same construction as
    # tests/test_selfplay_worker_single_move.py::winning_in_one_position.
    heights = [0] * 25
    heights[0] = 2
    heights[1] = 3
    domes = (1 << 5) | (1 << 6) | (1 << 3) | (1 << 8) | (1 << 9)
    position = Position(heights=tuple(heights), domes=domes,
                        workers=((0, 4), (20, 24)), powers=(0, 0), player=0)
    position.validate()
    return position

def test_evaluate_child_verdict_true_for_an_immediate_terminal_win():
    # The child *is* the position right after the winning move: winner is
    # already set directly by apply(), no search call should be needed.
    from santorini.engine import validate_turn
    position = winning_in_one_position()
    turns = __import__("santorini.native", fromlist=["native_turns"]).native_turns(position)
    winning_turn = next(t for t in turns if t.after.winner is not None)
    assert evaluate_child_verdict(winning_turn.after, mover=0, seconds=0.5) is True

def test_evaluate_child_verdict_handles_unresolved_within_budget():
    # An opening-ish position: with a near-zero budget, proof is extremely
    # unlikely to complete -- must return None, not crash or guess.
    heights = [0] * 25
    position = Position(heights=tuple(heights), workers=((2, 4), (20, 22)))
    verdict = evaluate_child_verdict(position, mover=0, seconds=0.01)
    assert verdict in (True, False, None)  # must not raise; typically None at this budget
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_proven_anchor_verdicts.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `mate_distance` and `evaluate_child_verdict` in `tools/proven_anchor/verdicts.py`**

```python
def mate_distance(analysis) -> int | None: ...

def evaluate_child_verdict(child_position, mover: int, seconds: float) -> bool | None: ...
```

`mate_distance`: `MATE - abs(analysis.score)` if `analysis.proven` else `None`. `evaluate_child_verdict`: if `child_position.winner is not None`, return `child_position.winner == mover` directly (no search — the move already ended the game). Otherwise call `native_search(child_position, seconds)`; if not `analysis.proven`, return `None`; else the move is winning for `mover` iff the child's own mover (`child_position.player`, the opponent of `mover`) is *losing* there, i.e. `analysis.score < 0` (negamax: `analysis.score` is signed from `child_position.player`'s perspective, matching `decode_analysis`'s own "Victoire/Défaite forcée démontrée" convention).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_proven_anchor_verdicts.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tools/proven_anchor/verdicts.py tests/test_proven_anchor_verdicts.py
git commit -m "feat: derive proven win/loss verdicts for individual moves from the reference engine"
```

---

### Task 4: Layer construction (`build_layer`)

**Files:**
- Create: `tools/proven_anchor/layer.py`
- Test: `tests/test_proven_anchor_layer.py`

**Interfaces:**
- Consumes: `tools.proven_anchor.verdicts.evaluate_child_verdict` (Task 3, injectable for tests), `santorini.native.native_turns` (existing).
- Produces: `tools.proven_anchor.layer.LayerResult` (dataclass: `ancestor: Position`, `children: list[tuple[Turn, bool | None]]`, `resolved_fraction: float`, `ancestor_verdict: bool | None`) and `tools.proven_anchor.layer.build_layer(ancestor: Position, seconds: float, evaluate=evaluate_child_verdict) -> LayerResult`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_proven_anchor_layer.py
from santorini.engine import Position
from santorini.native import native_turns
from tools.proven_anchor.layer import build_layer

def a_position_with_one_winning_move_among_several():
    heights = [0] * 25
    heights[0] = 2
    heights[1] = 3  # worker 0 can climb here and win
    position = Position(heights=tuple(heights), workers=((0, 4), (20, 24)), player=0)
    position.validate()
    return position

def test_build_layer_marks_the_ancestor_as_a_proven_win_when_a_winning_child_exists():
    position = a_position_with_one_winning_move_among_several()

    def fake_evaluate(child, mover, seconds):
        return child.winner == mover if child.winner is not None else False

    result = build_layer(position, seconds=0.1, evaluate=fake_evaluate)

    assert result.ancestor_verdict is True
    assert result.resolved_fraction == 1.0
    assert any(verdict is True for _, verdict in result.children)

def test_build_layer_marks_the_ancestor_as_a_proven_loss_when_all_children_lose():
    position = a_position_with_one_winning_move_among_several()

    def all_losing(child, mover, seconds):
        return False

    result = build_layer(position, seconds=0.1, evaluate=all_losing)

    assert result.ancestor_verdict is False
    assert all(verdict is False for _, verdict in result.children)

def test_build_layer_reports_undetermined_when_nothing_resolves(monkeypatch):
    position = a_position_with_one_winning_move_among_several()

    def unresolved(child, mover, seconds):
        return None

    result = build_layer(position, seconds=0.1, evaluate=unresolved)

    assert result.ancestor_verdict is None
    assert result.resolved_fraction == 0.0

def test_build_layer_computes_resolved_fraction_with_partial_resolution():
    position = a_position_with_one_winning_move_among_several()
    turns = native_turns(position)
    half = len(turns) // 2

    def half_resolved(child, mover, seconds):
        # Deterministic split by whether this child is among the first half
        # of native_turns' own order -- exact values don't matter, only
        # that build_layer iterates the same `native_turns` order it was
        # given (asserted indirectly via the exact fraction below).
        return False if turns.index(next(t for t in turns if t.after == child)) < half else None

    result = build_layer(position, seconds=0.1, evaluate=half_resolved)
    assert result.resolved_fraction == pytest.approx(half / len(turns))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_proven_anchor_layer.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `LayerResult` and `build_layer` in `tools/proven_anchor/layer.py`**

For each `Turn` from `native_turns(ancestor)`, call `evaluate(turn.after, ancestor.player, seconds)` and collect `(turn, verdict)`. `resolved_fraction = count(verdict is not None) / len(children)`. `ancestor_verdict`: `True` if any resolved child has `verdict is True`; else `False` if `resolved_fraction == 1.0` (every child resolved and none winning — standard minimax backup, a true local forced loss); else `None` (still undetermined, some children unresolved and none proven winning yet).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_proven_anchor_layer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tools/proven_anchor/layer.py tests/test_proven_anchor_layer.py
git commit -m "feat: derive a position's proven value from its children's verdicts (build_layer)"
```

---

### Task 5: Anchor-game sourcing

**Files:**
- Create: `tools/proven_anchor/anchor_games.py`
- Test: `tests/test_proven_anchor_anchor_games.py`

**Interfaces:**
- Consumes: `tools.mcts_az_vs_reference_campaign.{freeze_reference, reference_move}` (existing, reused as-is), `santorini.engine.{validate_setup, terminal_position}` (existing), `tools.proven_anchor.verdicts.mate_distance` (Task 3).
- Produces: `tools.proven_anchor.anchor_games.scan_game_for_forced_win(positions: list[Position], target_distance: int, seconds: float) -> int | None` (pure, index into `positions`, or `None`); `tools.proven_anchor.anchor_games.play_reference_vs_reference(opening_job, seconds, reference_binary) -> list[Position]` (full ply-by-ply history including the initial position); `tools.proven_anchor.anchor_games.find_anchor_games(openings: list[dict], target_distance: int, seconds: float, reference_binary, minimum: int = 2) -> list[tuple[list[Position], int]]` (stops once `minimum` qualifying games are found, each paired with its qualifying ply index).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_proven_anchor_anchor_games.py
from santorini.engine import Position
from santorini.native import native_search
from tools.proven_anchor.anchor_games import scan_game_for_forced_win

def test_scan_game_for_forced_win_finds_a_known_mate_in_two(monkeypatch):
    # Build a short synthetic history where position[-1] is a known mate-in-2
    # fixture is impractical to hand-construct exactly, so this test injects
    # a fake analysis function instead of relying on a real deep proof.
    import tools.proven_anchor.anchor_games as mod

    positions = [Position(workers=((0, 4), (20, 24))) for _ in range(5)]

    def fake_search(position, seconds):
        class FakeAnalysis:
            proven = position is positions[2]
            score = 100_000 - 2 if proven else 0
        return FakeAnalysis()

    monkeypatch.setattr(mod, "native_search", fake_search)

    index = scan_game_for_forced_win(positions, target_distance=2, seconds=0.1)
    assert index == 2

def test_scan_game_for_forced_win_returns_none_when_absent(monkeypatch):
    import tools.proven_anchor.anchor_games as mod
    positions = [Position(workers=((0, 4), (20, 24))) for _ in range(3)]

    def never_proven(position, seconds):
        class FakeAnalysis:
            proven = False
            score = 0
        return FakeAnalysis()

    monkeypatch.setattr(mod, "native_search", never_proven)

    assert scan_game_for_forced_win(positions, target_distance=6, seconds=0.1) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_proven_anchor_anchor_games.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `scan_game_for_forced_win` in `tools/proven_anchor/anchor_games.py`**

Iterate `positions` with their index; for each, call `native_search(position, seconds)` and `mate_distance(analysis)` (Task 3); return the first index where it equals `target_distance`. Module-level `from santorini.native import native_search` (not a local import), so tests can `monkeypatch.setattr(mod, "native_search", ...)`.

- [ ] **Step 4: Implement `play_reference_vs_reference` and `find_anchor_games`**

`play_reference_vs_reference`: same loop shape as `tools.mcts_az_vs_reference_campaign.play`, but every ply calls `reference_move` (both sides) instead of branching on `is_candidate`; returns the list of positions visited (including the starting one), stopping at `terminal_position`. `find_anchor_games`: iterate `openings`, for each call `play_reference_vs_reference` then `scan_game_for_forced_win`; keep `(positions, index)` pairs where a match was found; stop early once `minimum` are collected; raise `ValueError` if the full `openings` list is exhausted without reaching `minimum` (a loud, actionable failure rather than silently proceeding with too few anchors).

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_proven_anchor_anchor_games.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add tools/proven_anchor/anchor_games.py tests/test_proven_anchor_anchor_games.py
git commit -m "feat: source anchor games with a provable forced win at a given ply distance"
```

---

### Task 6: `Trainer` integration for exact-labeled examples

**Files:**
- Modify: `santoni_az/trainer.py`
- Create: `tools/proven_anchor/training_examples.py`
- Test: Modify `tests/test_az_trainer.py` (add test), Create `tests/test_proven_anchor_training_examples.py`

**Interfaces:**
- Consumes: `tools.proven_anchor.layer.LayerResult` (Task 4), `tools.proven_anchor.planes.compute_planes` (Task 1), `santoni_az.network.ACTION_SPACE` (existing).
- Produces: `Trainer.add_exact_examples(examples: list[tuple[torch.Tensor, torch.Tensor, float]]) -> None`; `tools.proven_anchor.training_examples.action_index(position: Position, turn: Turn) -> int` (pure arithmetic, no FFI — see Step 5); `tools.proven_anchor.training_examples.examples_from_layer(layer: LayerResult) -> list[tuple[torch.Tensor, torch.Tensor, float]]` (one example per **ancestor** position per `LayerResult`, not per child — matches spec §3.3's "ces positions" referring to the ancestor positions walked along the curriculum, keeping scope to the MVP the spec marks as sufficient to test the hypothesis, §3.5).

**Important, verified by reading `native_engine/src/mcts.rs:96-328` (not to be re-derived or guessed):** `ACTION_SPACE` indices are `worker_slot * 625 + dest * 25 + build`, where `dest`/`build` are the move/build actions' target cells (0-24) and `worker_slot` is the position (0 or 1) of the moved worker's *source* cell within `position.workers[position.player]` (`mcts.rs:324-327`: `state.workers[to_move].position(|&c| c == mv.source)` — **not** `Action.worker`, which is a different, unrelated field). For a winning turn (single action, no real build — `mcts.rs:422-438`'s `legal_children_synthesizes_a_placeholder_build_for_winning_moves` test), the synthesized placeholder has `bld.target == bld.source == mv.target`, so `build == dest` in that case. This is pure arithmetic over fields already present on `Turn`/`Position` — no Rust call needed.

- [ ] **Step 1: Write the failing test for `Trainer.add_exact_examples`**

```python
# tests/test_az_trainer.py (append)
def test_add_exact_examples_are_immediately_trainable(tmp_path):
    games_dir, checkpoint_dir = make_dirs(tmp_path)
    trainer = Trainer(str(games_dir), str(checkpoint_dir))
    planes = torch.zeros(NUM_PLANES, 5, 5)
    policy = torch.zeros(1250)
    policy[0] = 1.0

    trainer.add_exact_examples([(planes, policy, 1.0)])

    assert len(trainer._examples) == 1
    loss = trainer.train_step(batch_size=1)
    assert loss > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_az_trainer.py::test_add_exact_examples_are_immediately_trainable -v`
Expected: FAIL — `AttributeError: 'Trainer' object has no attribute 'add_exact_examples'`

- [ ] **Step 3: Implement `Trainer.add_exact_examples`**

```python
def add_exact_examples(self, examples):
    ...
```

Appends each `(planes, policy_target, value_target)` tuple directly to `self._examples` (same tuple shape `consume_new_games` already produces, so `train_step` needs no change), then re-applies the existing buffer cap (`excess = len(self._examples) - self.max_buffer_positions; if excess > 0: del self._examples[:excess]` — same two lines already used in `consume_new_games`/`_rebuild_buffer_from_disk`, extracted to a small private helper if that avoids triplicating the snippet, or copied verbatim if a helper would touch unrelated call sites not in this plan's scope).

- [ ] **Step 4: Run the new trainer test**

Run: `pytest tests/test_az_trainer.py::test_add_exact_examples_are_immediately_trainable -v`
Expected: PASS

- [ ] **Step 5: Write the failing test for `action_index`**

```python
# tests/test_proven_anchor_training_examples.py
from santorini.engine import Position
from santorini.native import native_turns
from tools.proven_anchor.training_examples import action_index

def test_action_index_matches_the_known_rust_formula_for_a_non_winning_move():
    position = Position(workers=((0, 4), (20, 24)))
    turns = native_turns(position)
    # worker 0 (slot 0) sits at cell 0; pick any legal turn moving it.
    turn = next(t for t in turns if t.actions[0].source == 0)
    mv, bld = turn.actions[0], turn.actions[1]
    expected = 0 * 625 + mv.target * 25 + bld.target
    assert action_index(position, turn) == expected

def test_action_index_uses_the_move_target_as_build_for_a_winning_turn():
    heights = [0] * 25
    heights[0] = 2
    heights[1] = 3
    position = Position(heights=tuple(heights), workers=((0, 4), (20, 24)), player=0)
    turns = native_turns(position)
    winning_turn = next(t for t in turns if t.after.winner is not None)
    mv = winning_turn.actions[0]
    assert len(winning_turn.actions) == 1
    expected = 0 * 625 + mv.target * 25 + mv.target  # build == dest for a synthesized placeholder
    assert action_index(position, winning_turn) == expected
```

- [ ] **Step 6: Run test to verify it fails**

Run: `pytest tests/test_proven_anchor_training_examples.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 7: Implement `action_index`**

```python
def action_index(position, turn) -> int:
    ...
```

`worker_slot = position.workers[position.player].index(turn.actions[0].source)`; `dest = turn.actions[0].target`; `build = turn.actions[1].target if len(turn.actions) > 1 else dest` (matches the synthesized-placeholder convention above). Return `worker_slot * 625 + dest * 25 + build`.

- [ ] **Step 8: Run the `action_index` tests, then write the failing test for `examples_from_layer`**

Run: `pytest tests/test_proven_anchor_training_examples.py -v` — expect the two `action_index` tests to PASS.

```python
# tests/test_proven_anchor_training_examples.py (append)
from tools.proven_anchor.layer import LayerResult
from tools.proven_anchor.training_examples import examples_from_layer

def test_examples_from_layer_builds_a_policy_target_over_winning_moves_only():
    ancestor = Position(workers=((0, 4), (20, 24)))
    turns = native_turns(ancestor)
    children = [(t, True) if i == 0 else (t, False) for i, t in enumerate(turns)]
    layer = LayerResult(ancestor=ancestor, children=children,
                        resolved_fraction=1.0, ancestor_verdict=True)

    examples = examples_from_layer(layer)

    assert len(examples) == 1
    planes, policy, value = examples[0]
    assert value == 1.0
    assert policy.sum().item() == pytest.approx(1.0)
    winning_index = action_index(ancestor, turns[0])
    assert policy[winning_index].item() == pytest.approx(1.0)
    assert (policy > 0).sum().item() == 1

def test_examples_from_layer_skips_a_layer_with_no_winning_move_found_yet():
    ancestor = Position(workers=((0, 4), (20, 24)))
    layer = LayerResult(ancestor=ancestor, children=[], resolved_fraction=0.0, ancestor_verdict=None)
    assert examples_from_layer(layer) == []
```

- [ ] **Step 9: Run test to verify it fails**

Run: `pytest tests/test_proven_anchor_training_examples.py -v`
Expected: the two new tests FAIL — `ImportError: cannot import name 'examples_from_layer'`

- [ ] **Step 10: Implement `examples_from_layer`**

Skip (return `[]`) if `layer.ancestor_verdict is not True` (nothing useful to train on yet: either a proven loss for this mover — doesn't happen for a true anchor ancestor but handled per Review Focus — or still undetermined). Otherwise: `planes = compute_planes(layer.ancestor)` reshaped to `(NUM_PLANES, 5, 5)` tensor; policy target is a `torch.zeros(ACTION_SPACE)` with uniform mass spread over `{action_index(layer.ancestor, turn) for turn, verdict in layer.children if verdict is True}`; `value_target = 1.0`.

- [ ] **Step 11: Run tests to verify they pass**

Run: `pytest tests/test_proven_anchor_training_examples.py tests/test_az_trainer.py -v`
Expected: PASS

- [ ] **Step 12: Commit**

```bash
git add santoni_az/trainer.py tools/proven_anchor/training_examples.py tests/test_az_trainer.py tests/test_proven_anchor_training_examples.py
git commit -m "feat: feed exact proven-core labels into Trainer as supervised examples"
```

---

### Task 7: Validation harness (anti-memorization gate)

**Files:**
- Create: `tools/proven_anchor/validation.py`
- Test: `tests/test_proven_anchor_validation.py`

**Interfaces:**
- Consumes: `tools.proven_anchor.symmetry.BOARD_SYMMETRIES` (Task 2), `tools.proven_anchor.planes.compute_planes` (Task 1), `santoni_az.network.PolicyValueNet` (existing).
- Produces: `tools.proven_anchor.validation.sign_match_rate(net, labeled_positions: list[tuple[Position, bool]]) -> float` (fraction where `net`'s value-head sign agrees with the proven verdict, methodology matching `selfplay-runs/diagnose_net.py`); `tools.proven_anchor.validation.bootstrap_ci(hits: list[bool], resamples: int = 5000, seed: int = 20261019) -> tuple[float, float]` (95% CI via resampling, same style as `tools.mcts_az_vs_reference_campaign.bootstrap`, generalized from placement-groups to a flat hit list); `tools.proven_anchor.validation.layer_passes_validation(net, baseline_net, held_out: list[tuple[Position, bool]]) -> bool` (True iff the bootstrap CI lower bound on `net`'s sign-match rate exceeds `baseline_net`'s own sign-match rate on the same `held_out` set).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_proven_anchor_validation.py
import torch
from tools.proven_anchor.validation import sign_match_rate, bootstrap_ci, layer_passes_validation
from santoni_az.network import PolicyValueNet
from santorini.engine import Position

def test_sign_match_rate_is_exact_arithmetic():
    class FakeNet:
        def __call__(self, planes):
            # value = +1 for even batch index, -1 for odd, policy irrelevant here
            n = planes.shape[0]
            values = torch.tensor([1.0 if i % 2 == 0 else -1.0 for i in range(n)]).unsqueeze(1)
            return torch.zeros(n, 1250), values

    positions = [Position(workers=((0, 4), (20, 24))) for _ in range(4)]
    labels = [True, True, False, False]  # matches for i=0,3 only under the fake net above
    rate = sign_match_rate(FakeNet(), list(zip(positions, labels)))
    assert rate == pytest.approx(0.5)

def test_bootstrap_ci_bounds_are_well_formed():
    hits = [True] * 18 + [False] * 2  # 90% observed
    lo, hi = bootstrap_ci(hits)
    assert 0.0 <= lo <= 0.9 <= hi <= 1.0

def test_layer_passes_validation_requires_clearing_the_baseline():
    untrained = PolicyValueNet()
    untrained.eval()
    positions = [Position(workers=((0, 4), (20, 24))) for _ in range(10)]
    held_out = list(zip(positions, [True] * 10))
    # Comparing the same untrained net against itself must never pass.
    assert layer_passes_validation(untrained, untrained, held_out) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_proven_anchor_validation.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `sign_match_rate`**

For each `(position, verdict)`: `planes = compute_planes(position)` as a `(1, NUM_PLANES, 5, 5)` tensor, `_, value = net(planes)` under `torch.no_grad()`, hit iff `(value.item() > 0) == verdict`. Return `hits / len(labeled_positions)`.

- [ ] **Step 4: Implement `bootstrap_ci`**

`random.Random(seed)`; for `resamples` iterations, resample `hits` with replacement (same length), compute the resampled hit rate; sort all resampled rates; return `(samples[int(0.025 * resamples)], samples[int(0.975 * resamples) - 1])` — same percentile-index convention as `tools.mcts_az_vs_reference_campaign.bootstrap`.

- [ ] **Step 5: Implement `layer_passes_validation`**

Build `hits = [((value > 0) == verdict) for position, verdict in held_out]` for `net`; `lo, _ = bootstrap_ci(hits)`; compute `baseline_rate = sign_match_rate(baseline_net, held_out)`; return `lo > baseline_rate`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_proven_anchor_validation.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add tools/proven_anchor/validation.py tests/test_proven_anchor_validation.py
git commit -m "feat: validate a trained layer's sign-match against a baseline, bootstrap CI"
```

---

### Task 8: Orchestration — the widening loop

**Files:**
- Create: `tools/proven_anchor_curriculum.py`
- Test: `tests/test_proven_anchor_curriculum_orchestration.py`

**Interfaces:**
- Consumes: every module from Tasks 1-7, plus `tools.mcts_az_vs_reference_campaign.freeze_reference` and `santoni_az.trainer.Trainer`.
- Produces: `tools.proven_anchor_curriculum.run_curriculum(anchor_games, reference_binary, seconds, trainer) -> dict` — drives the layer-by-layer loop and returns a summary (`layers_built`, `outcome` one of `"stopped_success_exhausted"` [not expected to ever naturally terminate per spec §2's "no ceiling"; included only for a future manual stop] / `"informative_failure"` / `"error"`), used by a thin `main()` CLI entry point for real runs.

- [ ] **Step 1: Write the failing test for the branching logic**

```python
# tests/test_proven_anchor_curriculum_orchestration.py
from tools.proven_anchor_curriculum import run_curriculum

class FakeTrainer:
    def __init__(self):
        self.examples = []
    def add_exact_examples(self, examples):
        self.examples.extend(examples)
    def train_step(self, batch_size=64):
        return 0.1

def test_run_curriculum_reports_informative_failure_when_sign_match_never_clears_baseline(monkeypatch):
    import tools.proven_anchor_curriculum as mod
    # Stub every expensive piece: build_layer always returns a resolved,
    # winning layer; validation always fails (simulates "signal is clean but
    # capacity is the bottleneck" from spec §2's failure criterion).
    monkeypatch.setattr(mod, "build_layer", lambda ancestor, seconds: object())
    monkeypatch.setattr(mod, "examples_from_layer", lambda layer: [(None, None, 1.0)])
    monkeypatch.setattr(mod, "layer_passes_validation", lambda *a, **k: False)

    result = run_curriculum(anchor_games=[(["p0", "p1"], 1), (["p0", "p1"], 1)],
                            reference_binary="fake", seconds=0.1, trainer=FakeTrainer())

    assert result["outcome"] == "informative_failure"
    assert result["layers_built"] <= 3  # stops within the first few layers, not forever

def test_run_curriculum_requires_at_least_two_anchor_games():
    with pytest.raises(ValueError):
        run_curriculum(anchor_games=[(["p0"], 0)], reference_binary="fake",
                       seconds=0.1, trainer=FakeTrainer())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_proven_anchor_curriculum_orchestration.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement `run_curriculum`**

```python
def run_curriculum(anchor_games, reference_binary, seconds, trainer, max_uninformative_layers=3):
    ...
```

Validate `len(anchor_games) >= 2` up front (`ValueError` otherwise — spec §2's precondition). Loop over increasing ply-distance (6, 12, 18, ...): for each anchor game, walk its history back to the ancestor at the current distance (index = `qualifying_ply - distance`, clamped/stopped if it goes negative — an anchor game is exhausted once its own history runs out, which is a normal, reportable stop, not an error); `build_layer` each ancestor, `examples_from_layer` + `trainer.add_exact_examples` + `trainer.train_step()`; then `layer_passes_validation` against held-out positions pooled from the *other* anchor game(s)' same-distance ancestors (plus `BOARD_SYMMETRIES` applied to the trained ones, per spec §3.4) and against every previously-validated distance's held-out set (spec §2's cumulative re-check). If validation fails: increment a `consecutive_failures` counter; stop with `"informative_failure"` once it exceeds `max_uninformative_layers` (spec §2's "deux ou trois premières couches"). If it passes: reset the counter, widen (`distance += 6`), continue. Return the summary dict.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_proven_anchor_curriculum_orchestration.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tools/proven_anchor_curriculum.py tests/test_proven_anchor_curriculum_orchestration.py
git commit -m "feat: orchestrate the proven-anchor curriculum widening loop"
```

---

## Self-review notes

- **Spec coverage:** §3.1 -> Task 5; §3.2 (incl. unresolved-branch handling) -> Tasks 3-4; §3.3 -> Task 6 (self-play-paused constraint lives in Global Constraints, enforced by this plan simply never touching `Orchestrator`); §3.4 -> Task 2 + Task 7; §2 (incl. the two-anchor-game precondition and cumulative re-validation) -> Task 8; §4 -> Task 6/8 (fresh `PolicyValueNet()`, never a loaded checkpoint); §3.6/§3.7 -> Task 8's loop and `anchor_games` list. §3.5 (MCTS lookup integration) is explicitly out of scope (spec §5) and has no task — correct, not a gap.
- **Proportion check:** 8 tasks against a 10-section spec, each task's body kept to signatures/tests/short prose rather than full implementations — proportionate.
- **Type consistency:** `LayerResult.children` (Task 4) is the one type threaded through Tasks 6-8; its shape (`list[tuple[Turn, bool | None]]`) is fixed once in Task 4 and consumed as-is everywhere else — checked, no renaming drift across tasks.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-10-19-mcts-az-proven-anchor-curriculum.md`. Please review the plan. Which execution approach would you prefer?

- **Subagent-driven** - A fresh subagent implements each task and a fresh reviewer checks it before the next one starts, then a whole-branch review at the end. Most thorough; costs a fresh context per task and per review.
- **Native** - I implement every task myself in this session, then one fresh reviewer on the most capable model checks the whole branch. Cheapest and fastest; no independent review until the end.

For this plan I recommend **subagent-driven**, because several tasks (4, 6, 8 especially) carry subtle correctness surface (minimax backup direction, negamax sign conventions, the cumulative-revalidation branching) in a domain where this chantier has already been burned twice by sign/perspective bugs that only a fresh reviewer caught — the per-task review gate is worth the extra cost here more than on a typical feature. Does the plan capture what you want, and which approach should we use?

# Ablation de capacité réseau (MCTS+NN) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone, deterministic script that offline-trains several `PolicyValueNet` sizes on positions already recorded on disk (`selfplay-runs/anchor-pilot-20261021/games/`), and reports held-out loss + value sign-match per architecture, to test whether network capacity is the bottleneck behind the chantier's flat loss / 0% win-rate result.

**Architecture:** One new file, `tools/network_capacity_ablation.py`, with pure, independently testable functions (data loading/partition, per-architecture train/eval loop) plus an `argparse` CLI entry point that wires them together and writes a JSON report. Reuses `Trainer._decode_position` (sign convention) and `PolicyValueNet(channels, blocks)` unmodified — no changes to existing `santoni_az/` modules.

**Tech Stack:** Python, PyTorch (`torch`, `torch.nn.functional`), stdlib `argparse`/`json`/`random`/`os`/`pathlib`. Pytest for tests, following `tests/test_az_trainer.py`'s synthetic-fixture convention.

**Spec:** `docs/superpowers/specs/2026-10-27-mcts-az-network-capacity-ablation-design.md`

## Global Constraints

- No new external dependency — stdlib + already-used `torch`.
- Deterministic end-to-end at a fixed seed (same positions, same batch order, same initial weights per architecture).
- Every loading/partition/train/eval function must be testable without the real 310k-game corpus (synthetic fixtures, same shape as `tests/test_az_trainer.py::write_fake_game`).
- French comments/docstrings, matching the rest of `santoni_az`/`tools/` (code identifiers stay in English).

## Review Focus

- A position from one `game-*.json` file must never appear in both the train pool and the held-out pool — partition at the file level, verified by file-name set disjointness, not just by count.
- `load_examples` hitting `max_positions` mid-file must stop cleanly at the requested cap, not overshoot by a whole file or crash.
- A corrupted/empty game file inside the sampled set must not abort loading (same contract as `Trainer.consume_new_games`).
- A malformed `--architectures` CLI value (e.g. `"abc"` with no `:`) must raise a clear, user-facing error, not a raw Python traceback.
- A killed process mid-report-write must never leave a partially-written JSON report behind (atomic write).

---

### Task 1: Data loading and train/held-out partition

**Files:**
- Create: `tools/network_capacity_ablation.py`
- Test: `tests/test_network_capacity_ablation.py`

**Interfaces:**
- Consumes: `santoni_az.trainer.Trainer._decode_position(position, ply, outcome)` (existing static method, returns `(planes: torch.Tensor, policy_target: torch.Tensor, value_target: float)`).
- Produces: `list_game_files(games_dir: str | Path) -> list[Path]`, `split_train_holdout(game_files: list[Path], holdout_fraction: float, seed: int) -> tuple[list[Path], list[Path]]`, `load_examples(games_dir: str | Path, game_files: list[Path], max_positions: int, seed: int) -> list[tuple[torch.Tensor, torch.Tensor, float]]` — later tasks call all three by name.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_network_capacity_ablation.py
import json
from pathlib import Path

from tools.network_capacity_ablation import list_game_files, split_train_holdout, load_examples

PLACEHOLDER_PLANES = [0] * (6 * 5 * 5)  # NUM_PLANES * 5 * 5


def write_fake_game(games_dir, name, outcome, positions):
    """Same format as tests/test_az_trainer.py::write_fake_game."""
    payload = {
        "outcome": outcome,
        "positions": [{"planes": planes, "visits": visits} for planes, visits in positions],
    }
    (games_dir / name).write_text(json.dumps(payload) + "\n")


def test_list_game_files_returns_sorted_game_files_only(tmp_path):
    (tmp_path / "game-0002.json").write_text("{}")
    (tmp_path / "game-0001.json").write_text("{}")
    (tmp_path / "not-a-game.txt").write_text("x")
    files = list_game_files(tmp_path)
    assert [f.name for f in files] == ["game-0001.json", "game-0002.json"]


def test_split_train_holdout_is_disjoint_and_covers_all_files():
    files = [Path(f"game-{i:04d}.json") for i in range(20)]
    train, holdout = split_train_holdout(files, holdout_fraction=0.25, seed=1)
    assert len(holdout) == 5
    assert set(train) & set(holdout) == set()
    assert set(train) | set(holdout) == set(files)


def test_split_train_holdout_is_deterministic():
    files = [Path(f"game-{i:04d}.json") for i in range(20)]
    first = split_train_holdout(files, holdout_fraction=0.25, seed=42)
    second = split_train_holdout(files, holdout_fraction=0.25, seed=42)
    assert first == second


def test_load_examples_stops_at_max_positions(tmp_path):
    write_fake_game(tmp_path, "game-0000.json", 1, [
        (PLACEHOLDER_PLANES, [[10, 1]]),
        (PLACEHOLDER_PLANES, [[11, 1]]),
        (PLACEHOLDER_PLANES, [[12, 1]]),
    ])
    write_fake_game(tmp_path, "game-0001.json", -1, [
        (PLACEHOLDER_PLANES, [[13, 1]]),
        (PLACEHOLDER_PLANES, [[14, 1]]),
        (PLACEHOLDER_PLANES, [[15, 1]]),
    ])
    files = list_game_files(tmp_path)
    examples = load_examples(tmp_path, files, max_positions=4, seed=1)
    assert len(examples) == 4


def test_load_examples_skips_corrupt_file(tmp_path):
    write_fake_game(tmp_path, "game-0000.json", 1, [(PLACEHOLDER_PLANES, [[10, 1]])])
    (tmp_path / "game-0001.json").write_text("")  # fichier vide, non-fsync avant renommage
    files = list_game_files(tmp_path)
    examples = load_examples(tmp_path, files, max_positions=10, seed=1)
    assert len(examples) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_network_capacity_ablation.py -v`
Expected: FAIL/ERROR — `tools.network_capacity_ablation` does not exist yet.

- [ ] **Step 3: Implement `list_game_files`, `split_train_holdout`, `load_examples` in `tools/network_capacity_ablation.py`**

`list_game_files`: `sorted(Path(games_dir).glob("game-*.json"))` (name-sorted, deterministic — same glob pattern `Trainer.consume_new_games` uses).

`split_train_holdout`: copy `game_files`, `random.Random(seed).shuffle` the copy, cut at `round(len(game_files) * holdout_fraction)`; first `cut` files are held-out, rest are train (document the choice of "first N after shuffle = held-out" as arbitrary-but-fixed, since the test only asserts disjointness/coverage/determinism, not which half is which).

`load_examples`: `random.Random(seed).sample(list(game_files), k=min(len(game_files), len(game_files)))` to get a shuffled-but-fixed visit order (don't bias toward files earlier in `game_files`), then iterate that order, `json.loads` each file wrapped in `try/except (json.JSONDecodeError, OSError)` → skip on failure (mirror `Trainer.consume_new_games`), call `Trainer._decode_position(position, ply, outcome)` for `ply, position in enumerate(data["positions"])`, append to a result list, stop (return immediately, don't keep scanning more files) as soon as `len(result) >= max_positions` — truncate the last file's contribution to exactly `max_positions` rather than appending a whole extra file's worth over the cap.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_network_capacity_ablation.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add tools/network_capacity_ablation.py tests/test_network_capacity_ablation.py
git commit -m "feat: add data loading/partition for network capacity ablation"
```

---

### Task 2: Per-architecture training and held-out evaluation

**Files:**
- Modify: `tools/network_capacity_ablation.py`
- Test: `tests/test_network_capacity_ablation.py`

**Interfaces:**
- Consumes: `santoni_az.network.PolicyValueNet(channels=.., blocks=..)`; `load_examples`'s return shape from Task 1 (`list[tuple[torch.Tensor, torch.Tensor, float]]`).
- Produces: `build_net(channels: int, blocks: int) -> PolicyValueNet`, `train_net(net, examples, steps: int, batch_size: int, seed: int) -> list[float]`, `evaluate_net(net, held_out_examples) -> dict` with keys `policy_loss: float`, `value_loss: float`, `total_loss: float`, `value_sign_match_rate: float` — Task 3's `main()` calls all three by name.

- [ ] **Step 1: Write the failing tests**

```python
def _toy_examples(n, value):
    import torch
    from santoni_az.network import NUM_PLANES, ACTION_SPACE
    examples = []
    for _ in range(n):
        planes = torch.zeros(NUM_PLANES, 5, 5)
        policy = torch.zeros(ACTION_SPACE)
        policy[0] = 1.0
        examples.append((planes, policy, value))
    return examples


def test_train_net_returns_one_loss_per_step():
    from tools.network_capacity_ablation import build_net, train_net
    net = build_net(channels=8, blocks=1)
    examples = _toy_examples(16, value=1.0)
    losses = train_net(net, examples, steps=5, batch_size=4, seed=1)
    assert len(losses) == 5
    assert all(isinstance(loss, float) for loss in losses)


def test_evaluate_net_reports_full_sign_match_on_trivial_data():
    from tools.network_capacity_ablation import build_net, train_net, evaluate_net
    net = build_net(channels=8, blocks=1)
    examples = _toy_examples(32, value=1.0)
    train_net(net, examples, steps=200, batch_size=8, seed=1)
    metrics = evaluate_net(net, examples)
    assert set(metrics) == {"policy_loss", "value_loss", "total_loss", "value_sign_match_rate"}
    assert metrics["value_sign_match_rate"] == 1.0  # toutes les cibles sont positives


def test_evaluate_net_sign_match_rate_on_mixed_signs():
    from tools.network_capacity_ablation import build_net, evaluate_net
    net = build_net(channels=8, blocks=1)
    examples = _toy_examples(4, value=1.0) + _toy_examples(4, value=-1.0)
    metrics = evaluate_net(net, examples)
    assert 0.0 <= metrics["value_sign_match_rate"] <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_network_capacity_ablation.py -v -k "train_net or evaluate_net"`
Expected: FAIL — functions not defined.

- [ ] **Step 3: Implement `build_net`, `train_net`, `evaluate_net` in `tools/network_capacity_ablation.py`**

`build_net(channels, blocks)`: `return PolicyValueNet(channels=channels, blocks=blocks)` — thin, named wrapper per the spec (readability only, no extra logic).

`train_net`: same loss composition as `Trainer.train_step` (`F.cross_entropy(policy_logits, policy_targets)` + `F.mse_loss(value_pred, value_targets)`, `Adam(net.parameters(), lr=1e-3)`), but sampling from the fixed `examples` list with `random.Random(seed)` instead of the live, mutating `Trainer._examples` buffer. Loop `steps` times: sample `batch_size` examples (with replacement if `batch_size > len(examples)`, matching `Trainer.train_step`'s `min(batch_size, len(self._examples))` guard for the under-supply case), stack into `planes`/`policy_targets`/`value_targets` tensors, forward/backward/step, append `float(loss.item())` to the returned list.

`evaluate_net`: `net.eval()`, `torch.no_grad()`, run in batches (reuse `batch_size`-less full-pass if `held_out_examples` is small enough — stack all at once, this is a diagnostic script not a production path) to get `policy_logits`/`value_pred` over the whole `held_out_examples`, compute `policy_loss`/`value_loss`/`total_loss` the same way as `train_net`, and `value_sign_match_rate` as `mean((value_pred.sign() == value_targets.sign()))` (treat a `0.0` target or prediction as its own sign bucket — document this choice isn't expected to bite in practice since `value_target` is always ±1 at terminal-derived outcomes, per `Trainer._decode_position`). Restore `net.train()` before returning is not needed here (each architecture gets a fresh `net`, Task 3 never reuses one past its own evaluate call) — note this explicitly as a deliberate scope limit, not an oversight.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_network_capacity_ablation.py -v`
Expected: PASS (8 tests total so far)

- [ ] **Step 5: Commit**

```bash
git add tools/network_capacity_ablation.py tests/test_network_capacity_ablation.py
git commit -m "feat: add per-architecture train/eval loop for network capacity ablation"
```

---

### Task 3: CLI entry point, report writing, and end-to-end smoke test

**Files:**
- Modify: `tools/network_capacity_ablation.py`
- Test: `tests/test_network_capacity_ablation.py`

**Interfaces:**
- Consumes: every function from Tasks 1-2 by name.
- Produces: `parse_architectures(raw: list[str]) -> list[tuple[int, int]]`, `main(argv: list[str] | None = None) -> None` (argparse CLI, writes the JSON report, prints the summary table) — no further task depends on this file.

- [ ] **Step 1: Write the failing tests**

```python
def test_parse_architectures_parses_channel_block_pairs():
    from tools.network_capacity_ablation import parse_architectures
    assert parse_architectures(["32:4", "64:8"]) == [(32, 4), (64, 8)]


def test_parse_architectures_rejects_malformed_entry():
    from tools.network_capacity_ablation import parse_architectures
    import pytest
    with pytest.raises(ValueError, match="32:4"):
        parse_architectures(["abc"])


def test_main_writes_atomic_report_and_no_leftover_tmp(tmp_path, capsys):
    from tools.network_capacity_ablation import main
    games_dir = tmp_path / "games"
    games_dir.mkdir()
    for i in range(6):
        write_fake_game(games_dir, f"game-{i:04d}.json", 1 if i % 2 == 0 else -1, [
            (PLACEHOLDER_PLANES, [[10, 1]]),
            (PLACEHOLDER_PLANES, [[11, 1]]),
        ])
    out_path = tmp_path / "report.json"
    main([
        "--games-dir", str(games_dir),
        "--architectures", "8:1", "16:2",
        "--max-train-positions", "8",
        "--max-holdout-positions", "4",
        "--holdout-fraction", "0.34",
        "--steps", "3",
        "--batch-size", "2",
        "--out", str(out_path),
    ])
    assert out_path.exists()
    assert not out_path.with_suffix(".json.tmp").exists()
    report = json.loads(out_path.read_text())
    assert [entry["channels"] for entry in report["architectures"]] == [8, 16]
    for entry in report["architectures"]:
        assert "total_loss" in entry and "value_sign_match_rate" in entry
    captured = capsys.readouterr()
    assert "16" in captured.out  # tableau récapitulatif affiché
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_network_capacity_ablation.py -v -k "parse_architectures or main"`
Expected: FAIL — functions not defined.

- [ ] **Step 3: Implement `parse_architectures` and `main` in `tools/network_capacity_ablation.py`**

`parse_architectures(raw)`: for each entry, `split(":")`, expect exactly 2 parts both `int(...)`-convertible; on any failure raise `ValueError(f"architecture invalide {entry!r}, format attendu 'channels:blocks' (ex. '32:4')")` — the test's `match="32:4"` pins the example staying in the message.

`main(argv=None)`: `argparse.ArgumentParser` with the flags from the spec's CLI section (`--games-dir` default `"selfplay-runs/anchor-pilot-20261021/games"`, `--architectures` `nargs="+"` default `["32:4", "64:8", "128:12"]`, `--max-train-positions` int default `200000`, `--max-holdout-positions` int default `20000`, `--holdout-fraction` float default `0.1`, `--steps` int default `2000`, `--batch-size` int default `64`, `--seed` int default `20261027`, `--out` default `"selfplay-runs/ablation-20261027/report.json"`). Body: `list_game_files` → `split_train_holdout` → `load_examples` twice (train pool with `max_train_positions`, holdout pool with `max_holdout_positions`) → for each `(channels, blocks)` from `parse_architectures`: `build_net`, `train_net`, `evaluate_net`, collect a dict with `channels`, `blocks`, `num_parameters` (`sum(p.numel() for p in net.parameters())`), the training losses list, and the `evaluate_net` metrics. Print one row per architecture to stdout (channels, blocks, `total_loss`, `value_sign_match_rate`). Write `{"games_dir": ..., "seed": ..., "architectures": [...]}` to `out_path` via a `.tmp` sibling + `os.replace` (same atomic-write pattern as `Trainer.save_checkpoint`), creating the parent directory with `Path(out_path).parent.mkdir(parents=True, exist_ok=True)` first.

Add:
```python
if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_network_capacity_ablation.py -v`
Expected: PASS (all tests in the file)

- [ ] **Step 5: Run the full project test suite to confirm no regression**

Run: `pytest -q`
Expected: all pre-existing tests still pass, plus the new file's tests.

- [ ] **Step 6: Commit**

```bash
git add tools/network_capacity_ablation.py tests/test_network_capacity_ablation.py
git commit -m "feat: add CLI entry point for network capacity ablation"
```

---

## After this plan lands: run the real ablation (not part of the task checkboxes above)

Not a task with its own tests — this is the actual experiment the plan exists to enable, run once the code is merged:

```bash
python3 -m tools.network_capacity_ablation \
  --games-dir selfplay-runs/anchor-pilot-20261021/games \
  --architectures 32:4 64:8 128:12 \
  --out selfplay-runs/ablation-20261027/report.json
```

Expect a multi-minute-to-hour CPU run depending on `--steps`/`--max-train-positions` (no GPU involved, pure CPU like every other `santoni_az` training path today). Read the printed table (and `report.json`) for `total_loss`/`value_sign_match_rate` per architecture, compare the `32:4` row (current production config) against `64:8`/`128:12` to decide between the two branches the spec's Context section lays out — record the result in the `santoni-mcts-az-no-power-chantier` memory file afterward, same convention as every other diagnostic in that chantier.

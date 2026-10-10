# MCTS+AZ campaign resumability, parallelism, and terminal-position handling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close blockers 5 and 6 on `tools/mcts_az_vs_reference_campaign.py` (the 200-game MCTS+AZ-candidate-vs-reference-engine evaluation campaign) so an interrupted campaign never loses already-played games, a 200-game run isn't strictly serial (days of wall-clock), the candidate's network checkpoint is actually wired in and its provenance recorded, and a position with no legal move resolves as a loss instead of an `'error'` status.

**Architecture:** Mirror `tools/hermes_canonicalisation_campaign.py`'s already-proven resumability/parallelism pattern (manifest with provenance mismatch-detection, pending-jobs scan that quarantines incomplete games into `attempts/`, a `.campaign.lock` file lock, `ProcessPoolExecutor` + CPU pinning, periodic summary refresh). Replace the current optional `--inference-socket` flag with the campaign script itself loading the network from `--checkpoint` and running its own `InferenceServer` for the campaign's lifetime — mirrors this same file's existing `freeze_reference` philosophy (pin exactly what's being measured) applied to the candidate side. Fix the zero-legal-move bug by calling the existing, already-proven `santorini.engine.terminal_position` rules check at the top of each ply, before dispatching to either engine.

**Tech Stack:** Python (`tools/`, `santorini/`, `santoni_az/`), `concurrent.futures.ProcessPoolExecutor`, `fcntl`, `pytest`.

**Spec:** No separate spec document — this plan directly implements blockers 5 and 6 recorded in memory `santoni-mcts-az-no-power-chantier` (not a repo file; see the conversation this plan was written in for the exact wording), scoped and confirmed with the user before writing. The closest in-repo precedent/spec-by-example is `tools/hermes_canonicalisation_campaign.py` itself, which this plan mirrors throughout.

## Global Constraints

- Do not touch `native_engine/` at all — this is a pure-Python campaign-script change. `run_single_move`'s Rust-side behavior (including its terminal-root assertion) is unaffected; this plan fixes the *caller*'s handling of an already-terminal position, so the Rust worker is simply never invoked on one.
- `--inference-socket` is removed entirely, not kept as a fallback. A real checkpoint-backed network is now mandatory for every campaign run (no more silent uniform-policy fallback).
- Follow the project's established convention of small per-script helper functions (`atomic_json`, `digest`, `bootstrap`, `physical_cpus`, `initialize`, etc. are deliberately duplicated across sibling campaign scripts rather than factored into a shared module) — do not introduce a new shared module for any of this.
- Neither `hermes_canonicalisation_campaign.py` nor `mcts_az_vs_reference_campaign.py` has pytest coverage of its full `main()`/parallel-execution path today; that convention holds here too. Write focused unit tests for the new pure/isolable logic (pending-jobs scan, manifest mismatch detection, checkpoint loading/freezing, the terminal-position short-circuit) — do not attempt a pytest test that drives real parallel subprocess game execution. The full integration (parallelism + resumability + checkpoint server + terminal handling, all together) is validated by a manual `--limit 2 --seconds 2` smoke run, called out explicitly in Task 4.

## Review Focus

- Two campaign processes started against the same `--out` directory concurrently must not corrupt each other's game files — mitigated by the `.campaign.lock` `fcntl.flock(..., LOCK_EX | LOCK_NB)` in Task 3, mirroring `hermes_canonicalisation_campaign.py`'s identical, itself-untested mitigation; not given a dedicated new test here either, for the same reason a file lock's OS-level exclusion isn't practically unit-testable without a second real process.
- Resuming after a worker was killed mid-write of a game file must never see a half-written/corrupt JSON as "pending" — already guaranteed by `atomic_json`'s existing temp-file-then-rename pattern (a crash mid-write leaves only the `.tmp` file, never a corrupt file under the final `game-NNNN.json` name), so `compute_pending_jobs` (Task 3) can safely assume every `game-NNNN.json` it finds parses as valid JSON. No new test needed; noted here so the invariant isn't silently assumed without being named.
- A re-run against the same `--out` with a *different* `--checkpoint` file (same path reused for a different training generation, or a genuinely different file) must be rejected, not silently measured against a stale frozen snapshot — covered by Task 3's `check_manifest_matches` test, since `candidate_checkpoint_sha256` is part of the compared config and changes whenever the checkpoint's bytes change.
- Parallel games must never share mutable per-game scratch state (the candidate side writes `state.bin`/`move.bin` per move) — addressed structurally in Task 4 by moving scratch-directory ownership inside `play()` itself (one per game, created and torn down within that game's own call), rather than one shared temporary directory handed in from `main()` across every game as today.
- A stuck position must resolve as a loss *before* either engine is invoked, for both the candidate and the reference side symmetrically — covered by Task 1's test, which fails both `candidate_move` and `reference_move` loudly if called, proving neither backend is reached once `terminal_position` has already resolved the game.

---

## Task 1: Zero-legal-move positions resolve as a loss, not an `'error'` (blocker 6)

**Files:**
- Modify: `tools/mcts_az_vs_reference_campaign.py` (imports, `play`)
- Create: `tests/test_mcts_az_vs_reference_campaign.py`

**Interfaces:**
- Consumes: `santorini.engine.terminal_position(position, check=lambda: None)` (existing, unchanged) — returns `position` unchanged if not terminal, or a `Position` with `.winner` set (including the "no legal move" case, `reason="Aucun tour complet légal"`) if terminal.
- Produces: `play(job, directory, seconds, checkpoint, socket_path, scratch_dir, reference_binary)` — same signature as today, now resolves an already-terminal `position` immediately at the top of each ply instead of dispatching to `candidate_move`/`reference_move`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_mcts_az_vs_reference_campaign.py`:

```python
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from santorini.engine import Position  # noqa: E402
from tools import mcts_az_vs_reference_campaign as campaign  # noqa: E402


def boxed_in_position(stuck_player):
    """Position du sous-jeu sans pouvoir où `stuck_player` n'a plus aucun
    coup légal : ses deux bâtisseurs sont en (12, 13), encerclés de dômes
    sur toutes les autres cases. Même construction que le test Rust
    `terminal_value_reports_loss_for_player_with_no_legal_move`
    (native_engine/src/mcts.rs)."""
    domes = ((1 << 25) - 1) & ~((1 << 12) | (1 << 13))
    workers = ((12, 13), (0, 1)) if stuck_player == 0 else ((0, 1), (12, 13))
    return Position(workers=workers, domes=domes, player=stuck_player, powers=(0, 0))


def test_play_resolves_a_stuck_candidate_as_a_loss_without_invoking_either_engine(tmp_path, monkeypatch):
    stuck = boxed_in_position(stuck_player=0)

    def boom(*args, **kwargs):
        raise AssertionError('must not be called on an already-terminal position')

    monkeypatch.setattr(campaign, 'validate_setup', lambda *a, **k: stuck)
    monkeypatch.setattr(campaign, 'candidate_move', boom)
    monkeypatch.setattr(campaign, 'reference_move', boom)

    job = {'number': 1, 'variant_player': 0,
           'initial': {'workers': [[12, 13], [0, 1]], 'powers': [0, 0], 'first': 0}}
    result = campaign.play(job, tmp_path, 2.0, 'unused-checkpoint', 'unused-socket', tmp_path, 'unused-reference')

    assert result['status'] == 'finished'
    assert result['variant_won'] is False  # variant_player 0 is the stuck player


def test_play_resolves_a_stuck_reference_as_a_win_for_the_candidate(tmp_path, monkeypatch):
    stuck = boxed_in_position(stuck_player=1)

    def boom(*args, **kwargs):
        raise AssertionError('must not be called on an already-terminal position')

    monkeypatch.setattr(campaign, 'validate_setup', lambda *a, **k: stuck)
    monkeypatch.setattr(campaign, 'candidate_move', boom)
    monkeypatch.setattr(campaign, 'reference_move', boom)

    job = {'number': 2, 'variant_player': 0,
           'initial': {'workers': [[0, 1], [12, 13]], 'powers': [0, 0], 'first': 0}}
    result = campaign.play(job, tmp_path, 2.0, 'unused-checkpoint', 'unused-socket', tmp_path, 'unused-reference')

    assert result['status'] == 'finished'
    assert result['variant_won'] is True  # variant_player 0 is the candidate; player 1 (reference) is stuck
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_mcts_az_vs_reference_campaign.py -v`
Expected: FAIL — `candidate_move`/`reference_move` currently get called on the (monkeypatched) already-terminal starting position (since `play()` never checks before dispatching), so the `boom()` `AssertionError` fires inside `play()`'s own `try/except Exception` block and gets swallowed into `status='error'`, making `result['status'] == 'finished'` false in both tests.

- [ ] **Step 3: Write the implementation**

In `tools/mcts_az_vs_reference_campaign.py`, add `Position` and `terminal_position` to the existing `santorini.engine` import:

```python
from santorini.engine import Action, Position, terminal_position, validate_setup, validate_turn  # noqa: E402
```

Modify the ply loop inside `play()`:

```python
def play(job, directory, seconds, checkpoint, socket_path, scratch_dir, reference_binary):
    initial = job['initial']
    path = Path(directory) / f"game-{job['number']:04d}.json"
    position = validate_setup(initial['workers'], initial['powers'], initial['first'])
    data = {'job': job, 'status': 'running', 'decisions': []}
    atomic_json(path, data)
    try:
        for ply in range(150):
            checked = terminal_position(position)
            if checked.winner is not None:
                position = checked
                break
            is_candidate = position.player == job['variant_player']
            if is_candidate:
                turn, elapsed = candidate_move(position, seconds, checkpoint, socket_path, scratch_dir)
            else:
                turn, elapsed = reference_move(position, seconds, reference_binary)
            data['decisions'].append({'ply': ply + 1, 'player': position.player,
                                       'backend': 'candidate' if is_candidate else 'reference',
                                       'elapsed': elapsed})
            position = turn.after
            atomic_json(path, {**data, 'position': position.to_dict()})
        data['status'] = 'finished' if position.winner is not None else 'inconclusive'
    except Exception as exc:
        data.update(status='error', error=f'{type(exc).__name__}: {exc}')
    data['winner'] = position.winner
    atomic_json(path, data)
    return {'number': job['number'], 'status': data['status'],
            'variant_won': position.winner == job['variant_player'] if position.winner is not None else None}
```

(Only the `checked = terminal_position(position)` early-exit and dropping the old `if position.winner is not None: break` — now redundant, since a `winner` can only ever appear via that same `checked` path — changed; everything else is unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_mcts_az_vs_reference_campaign.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/mcts_az_vs_reference_campaign.py tests/test_mcts_az_vs_reference_campaign.py
git commit -m "Résoudre une position sans coup légal en défaite plutôt qu'en erreur de campagne"
```

---

## Task 2: Campaign-managed `InferenceServer` loaded from `--checkpoint`

**Files:**
- Modify: `tools/mcts_az_vs_reference_campaign.py` (imports, new helpers, `candidate_move`, `play`, `main`)
- Modify: `tests/test_mcts_az_vs_reference_campaign.py` (update Task 1's `play()` calls for the new signature; add new tests)

**Interfaces:**
- Consumes: `santoni_az.network.PolicyValueNet`, `santoni_az.inference_server.InferenceServer(socket_path, net, flush_interval_s=0.02)` + `.serve_forever()` (run in a thread) + `.stop()` (all existing, unchanged).
- Produces: `load_checkpoint_net(path) -> PolicyValueNet` (loads weights from an arbitrary specific checkpoint file, net in eval mode), `freeze_checkpoint(directory, checkpoint_path) -> (Path, dict)` (snapshots the checkpoint into `directory/candidate/checkpoint.pt` + a sha256/generation manifest, raises `ValueError` on tampering — mirrors `freeze_reference`'s existing contract). `candidate_move(position, seconds, socket_path, scratch_dir)` (drops the unused `checkpoint` parameter; `socket_path` is now always provided, no more `if socket_path:` fallback). `play(job, directory, seconds, socket_path, scratch_dir, reference_binary)` (drops `checkpoint`, since the network is loaded once in `main()`, not per game).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_mcts_az_vs_reference_campaign.py`:

```python
import torch

from santoni_az.network import PolicyValueNet
from santoni_az.trainer import Trainer


def make_tiny_checkpoint(tmp_path):
    games_dir = tmp_path / 'games'
    games_dir.mkdir()
    checkpoint_dir = tmp_path / 'checkpoints'
    checkpoint_dir.mkdir()
    trainer = Trainer(str(games_dir), str(checkpoint_dir), net_factory=lambda: PolicyValueNet(channels=8, blocks=1))
    path = Path(trainer.save_checkpoint())
    return trainer, path


def test_load_checkpoint_net_restores_the_exact_saved_weights(tmp_path):
    trainer, path = make_tiny_checkpoint(tmp_path)
    net = campaign.load_checkpoint_net(path)
    saved_state = trainer.net.state_dict()
    for key, value in net.state_dict().items():
        assert torch.equal(value, saved_state[key]), f'parameter {key} does not match the saved checkpoint'


def test_freeze_checkpoint_pins_a_copy_and_is_idempotent(tmp_path):
    _, checkpoint_path = make_tiny_checkpoint(tmp_path)
    out_dir = tmp_path / 'campaign'
    out_dir.mkdir()

    binary, manifest = campaign.freeze_checkpoint(out_dir, checkpoint_path)
    assert binary.exists()
    assert manifest['generation'] == 1

    # Re-running against the same campaign directory must not re-snapshot or
    # recompute: same manifest content back.
    binary2, manifest2 = campaign.freeze_checkpoint(out_dir, checkpoint_path)
    assert binary2 == binary
    assert manifest2 == manifest


def test_freeze_checkpoint_detects_tampering_with_the_frozen_copy(tmp_path):
    _, checkpoint_path = make_tiny_checkpoint(tmp_path)
    out_dir = tmp_path / 'campaign'
    out_dir.mkdir()
    binary, _ = campaign.freeze_checkpoint(out_dir, checkpoint_path)

    binary.write_bytes(b'corrupted')
    with pytest.raises(ValueError):
        campaign.freeze_checkpoint(out_dir, checkpoint_path)
```

Also update the two `play()` calls already written in Task 1 to drop the now-removed `checkpoint` positional argument:

```python
    result = campaign.play(job, tmp_path, 2.0, 'unused-socket', tmp_path, 'unused-reference')
```

(same change in both `test_play_resolves_a_stuck_candidate_...` and `test_play_resolves_a_stuck_reference_...`.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_mcts_az_vs_reference_campaign.py -v`
Expected: the two new tests FAIL with `AttributeError: module 'tools.mcts_az_vs_reference_campaign' has no attribute 'load_checkpoint_net'` / `'freeze_checkpoint'`. The two updated Task 1 tests FAIL with a `TypeError` on argument count (current `play()` still expects the old 7-argument signature including `checkpoint`).

- [ ] **Step 3: Write the implementation**

Add imports to `tools/mcts_az_vs_reference_campaign.py`:

```python
import threading

import torch

from santoni_az.network import PolicyValueNet  # noqa: E402
from santoni_az.inference_server import InferenceServer  # noqa: E402
```

Add the two new helper functions (place them near `freeze_reference`, which they mirror):

```python
def load_checkpoint_net(path):
    data = torch.load(path, weights_only=False)
    net = PolicyValueNet()
    net.load_state_dict(data['model_state'])
    net.eval()
    return net


def freeze_checkpoint(directory, checkpoint_path):
    """Snapshot the checkpoint file currently at `checkpoint_path` into the
    campaign directory, exactly the same rationale as `freeze_reference`
    above applied to the candidate side: pin precisely what is being
    measured, so a checkpoint file changing on disk mid-campaign (e.g.
    training continuing to write to the same path, or simply pointing a
    resumed run at a different file) can never silently change what 'the
    candidate' means partway through, and is caught loudly instead."""
    target = directory / 'candidate'
    target.mkdir(exist_ok=True)
    binary = target / 'checkpoint.pt'
    manifest = target / 'manifest.json'
    if not manifest.exists():
        shutil.copy2(checkpoint_path, binary)
        generation = torch.load(binary, weights_only=False)['generation']
        atomic_json(manifest, {'checkpoint_sha256': digest(binary), 'generation': generation})
    frozen = json.loads(manifest.read_text())
    if digest(binary) != frozen['checkpoint_sha256']:
        raise ValueError('Candidate checkpoint snapshot integrity mismatch')
    return binary, frozen
```

Modify `candidate_move` to drop the unused `checkpoint` parameter and always pass `--inference-socket`:

```python
def candidate_move(position, seconds, socket_path, scratch_dir):
    state = encode(position)
    input_path = scratch_dir / 'state.bin'
    output_path = scratch_dir / 'move.bin'
    input_path.write_bytes(bytes(state))
    if output_path.exists():
        output_path.unlink()
    cmd = [str(WORKER_BINARY), '--single-move', '--input-state', str(input_path),
           '--output-move', str(output_path), '--seconds', str(seconds),
           '--inference-socket', str(socket_path)]
    started = time.monotonic()
    result = subprocess.run(cmd, timeout=seconds + 10, capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError(f'Candidate worker failed: {result.stderr}')
    raw = output_path.read_bytes()
    (count,) = struct.unpack_from('<I', raw, 0)
    expected = 4 + count * C.sizeof(CAction)
    if count not in (1, 2) or len(raw) != expected:
        raise ValueError(f'Candidate move file malformed: {len(raw)} bytes for {count} actions')
    actions = []
    for index in range(count):
        item = CAction.from_buffer_copy(raw, 4 + index * C.sizeof(CAction))
        actions.append(Action(ACTION_KINDS[item.kind], item.player, item.worker,
                              item.source, item.target))
    turn = validate_turn(position, tuple(actions))
    return turn, time.monotonic() - started
```

Modify `play()`'s signature and its one call to `candidate_move` (drop `checkpoint` throughout):

```python
def play(job, directory, seconds, socket_path, scratch_dir, reference_binary):
    initial = job['initial']
    path = Path(directory) / f"game-{job['number']:04d}.json"
    position = validate_setup(initial['workers'], initial['powers'], initial['first'])
    data = {'job': job, 'status': 'running', 'decisions': []}
    atomic_json(path, data)
    try:
        for ply in range(150):
            checked = terminal_position(position)
            if checked.winner is not None:
                position = checked
                break
            is_candidate = position.player == job['variant_player']
            if is_candidate:
                turn, elapsed = candidate_move(position, seconds, socket_path, scratch_dir)
            else:
                turn, elapsed = reference_move(position, seconds, reference_binary)
            data['decisions'].append({'ply': ply + 1, 'player': position.player,
                                       'backend': 'candidate' if is_candidate else 'reference',
                                       'elapsed': elapsed})
            position = turn.after
            atomic_json(path, {**data, 'position': position.to_dict()})
        data['status'] = 'finished' if position.winner is not None else 'inconclusive'
    except Exception as exc:
        data.update(status='error', error=f'{type(exc).__name__}: {exc}')
    data['winner'] = position.winner
    atomic_json(path, data)
    return {'number': job['number'], 'status': data['status'],
            'variant_won': position.winner == job['variant_player'] if position.winner is not None else None}
```

In `main()`, remove the `--inference-socket` argument entirely, and right after `reference_binary, reference_manifest = freeze_reference(directory)`, add:

```python
    candidate_binary, candidate_manifest = freeze_checkpoint(directory, Path(args.checkpoint))
    net = load_checkpoint_net(candidate_binary)
    socket_path = str(directory / 'infer.sock')
    server = InferenceServer(socket_path, net)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(0.1)
    print(f"CANDIDATE checkpoint generation={candidate_manifest['generation']} "
          f"sha256={candidate_manifest['checkpoint_sha256']}", flush=True)
```

Change the job-running section to use `socket_path` instead of `args.inference_socket`, and wrap it so the server is always stopped:

```python
    try:
        with tempfile.TemporaryDirectory() as scratch:
            scratch_dir = Path(scratch)
            results = [play(job, directory, args.seconds, socket_path, scratch_dir, reference_binary)
                       for job in jobs]
    finally:
        server.stop()
```

(The scratch-directory-per-game fix and the `jobs`/`results` restructuring for parallelism both land in Task 4 — this task only removes `--inference-socket` and wires in the real checkpoint-backed server; the loop stays sequential for now.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_mcts_az_vs_reference_campaign.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/mcts_az_vs_reference_campaign.py tests/test_mcts_az_vs_reference_campaign.py
git commit -m "La campagne charge et sert elle-même le réseau candidat depuis --checkpoint"
```

---

## Task 3: Resumability — provenance manifest + pending-jobs scan + concurrent-run lock

**Files:**
- Modify: `tools/mcts_az_vs_reference_campaign.py` (new helpers, `main`)
- Modify: `tests/test_mcts_az_vs_reference_campaign.py` (add new tests)

**Interfaces:**
- Consumes: `atomic_json`, `digest` (existing, unchanged).
- Produces: `compute_pending_jobs(directory, jobs) -> list` (skips `status=='finished'` games found on disk, quarantines incomplete ones into `directory/attempts/` with a timestamp suffix, returns the games still to play). `check_manifest_matches(manifest_path, config)` (writes `manifest_path` with a `started_at` timestamp on first call for a given path; on later calls, raises `ValueError` if `config` differs from what's stored, ignoring `started_at` itself).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_mcts_az_vs_reference_campaign.py`:

```python
import time


def test_compute_pending_jobs_skips_finished_and_quarantines_incomplete(tmp_path):
    jobs = [{'number': 1, 'variant_player': 0, 'initial': {}},
            {'number': 2, 'variant_player': 1, 'initial': {}},
            {'number': 3, 'variant_player': 0, 'initial': {}}]
    campaign.atomic_json(tmp_path / 'game-0001.json', {'job': jobs[0], 'status': 'finished'})
    campaign.atomic_json(tmp_path / 'game-0002.json', {'job': jobs[1], 'status': 'running'})
    # job 3 has no file yet -- a brand new game.

    pending = campaign.compute_pending_jobs(tmp_path, jobs)

    assert [j['number'] for j in pending] == [2, 3]
    assert not (tmp_path / 'game-0002.json').exists()
    quarantined = list((tmp_path / 'attempts').glob('game-0002*.json'))
    assert len(quarantined) == 1
    assert json.loads(quarantined[0].read_text())['status'] == 'running'
    assert (tmp_path / 'game-0001.json').exists()  # le fichier terminé n'est jamais touché


def test_compute_pending_jobs_rejects_a_job_mismatch(tmp_path):
    original_job = {'number': 1, 'variant_player': 0, 'initial': {'opening': 'a'}}
    changed_job = {'number': 1, 'variant_player': 0, 'initial': {'opening': 'b'}}
    campaign.atomic_json(tmp_path / 'game-0001.json', {'job': original_job, 'status': 'running'})
    with pytest.raises(ValueError):
        campaign.compute_pending_jobs(tmp_path, [changed_job])


def test_check_manifest_matches_writes_once_then_rejects_changed_config(tmp_path):
    manifest = tmp_path / 'campaign.json'
    config = {'seconds': 30.0, 'seed': 1}

    campaign.check_manifest_matches(manifest, config)
    written = json.loads(manifest.read_text())
    assert written['seconds'] == 30.0
    assert 'started_at' in written

    campaign.check_manifest_matches(manifest, config)  # même config : no-op

    with pytest.raises(ValueError):
        campaign.check_manifest_matches(manifest, {**config, 'seconds': 10.0})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_mcts_az_vs_reference_campaign.py -v`
Expected: the three new tests FAIL with `AttributeError: ... has no attribute 'compute_pending_jobs'` / `'check_manifest_matches'`.

- [ ] **Step 3: Write the implementation**

Add these two functions to `tools/mcts_az_vs_reference_campaign.py` (place them near `summary`, which they run alongside):

```python
def compute_pending_jobs(directory, jobs):
    """Coups restants à jouer : ignore les parties déjà terminées, met en
    quarantaine (`attempts/`, suffixe horodaté) les parties incomplètes
    d'une exécution précédente plutôt que de les écraser silencieusement sur
    place. Même motif que `tools/hermes_canonicalisation_campaign.py`
    (lignes ~367-379)."""
    pending = []
    for job in jobs:
        path = directory / f"game-{job['number']:04d}.json"
        if path.exists():
            old = json.loads(path.read_text())
            if old['job'] != job:
                raise ValueError(f'Job mismatch: {path}')
            if old['status'] == 'finished':
                continue
            attempts = directory / 'attempts'
            attempts.mkdir(exist_ok=True)
            path.rename(attempts / f'{path.stem}-{time.time_ns()}.json')
        pending.append(job)
    return pending


def check_manifest_matches(manifest_path, config):
    """Écrit `campaign.json` au premier lancement sur ce dossier ; aux
    lancements suivants, vérifie que la configuration n'a pas changé plutôt
    que de mélanger silencieusement deux campagnes incompatibles (graine,
    checkpoint, moteur de référence...) dans le même répertoire."""
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if {k: v for k, v in previous.items() if k != 'started_at'} != config:
            raise ValueError('Campaign provenance/parameters differ; choose a new output folder')
    else:
        atomic_json(manifest_path, {'started_at': time.time(), **config})
        shutil.copy2(Path(__file__), manifest_path.parent / 'runner.py')
```

In `main()`, right after `directory.mkdir(parents=True, exist_ok=True)`, add the concurrent-run lock:

```python
    import fcntl
    lock = (directory / '.campaign.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
```

After building `jobs` (and applying `--limit`) and after the checkpoint/reference freezing from Task 2, build the provenance config and check it, then compute pending jobs:

```python
    config = {'protocol': 'mcts-az-vs-reference-no-power-30s-v1', 'seconds': args.seconds, 'jobs': jobs,
              'seed': corpus['seed'], 'reference_engine_sha256': reference_manifest['engine_sha256'],
              'candidate_checkpoint_sha256': candidate_manifest['checkpoint_sha256'],
              'candidate_checkpoint_generation': candidate_manifest['generation'],
              'corpus_sha256': digest(CORPUS), 'runner_sha256': digest(Path(__file__))}
    try:
        check_manifest_matches(directory / 'campaign.json', config)
    except ValueError as exc:
        parser.error(str(exc))
    pending = compute_pending_jobs(directory, jobs)
```

Replace the job-running section (from Task 2) to run only `pending`, not all of `jobs`:

```python
    try:
        with tempfile.TemporaryDirectory() as scratch:
            scratch_dir = Path(scratch)
            results = [play(job, directory, args.seconds, socket_path, scratch_dir, reference_binary)
                       for job in pending]
    finally:
        server.stop()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_mcts_az_vs_reference_campaign.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tools/mcts_az_vs_reference_campaign.py tests/test_mcts_az_vs_reference_campaign.py
git commit -m "Reprise de campagne : manifeste de provenance, reprise des parties en attente, verrou anti-concurrence"
```

---

## Task 4: Parallelism — `ProcessPoolExecutor`, CPU pinning, per-game scratch directory

**Files:**
- Modify: `tools/mcts_az_vs_reference_campaign.py` (imports, new helpers, `play`, `main`)
- Modify: `tests/test_mcts_az_vs_reference_campaign.py` (update all `play()` calls for the new signature)

**Interfaces:**
- Consumes: `compute_pending_jobs`, `check_manifest_matches` (Task 3), `freeze_checkpoint`, `load_checkpoint_net` (Task 2).
- Produces: `physical_cpus() -> list[int]`, `initialize(cpus)` (CPU-pinning `ProcessPoolExecutor` initializer — both verbatim from `tools/hermes_canonicalisation_campaign.py`, duplicated per this project's established convention). `play(job, directory, seconds, socket_path, reference_binary)` (drops `scratch_dir` — each call now creates and tears down its own, since multiple `play()` calls now run concurrently in separate processes and must never share mutable per-game scratch files).

- [ ] **Step 1: Update the existing tests for the new `play()` signature**

In `tests/test_mcts_az_vs_reference_campaign.py`, update both Task 1 calls to drop the trailing `scratch_dir` argument (`play()` now creates its own internally):

```python
    result = campaign.play(job, tmp_path, 2.0, 'unused-socket', 'unused-reference')
```

(same change in both `test_play_resolves_a_stuck_candidate_...` and `test_play_resolves_a_stuck_reference_...` — drop the `tmp_path` argument that used to be passed as `scratch_dir`, keeping the rest in the same order.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_mcts_az_vs_reference_campaign.py -v`
Expected: the two updated tests FAIL with a `TypeError` on argument count (current `play()` still expects `scratch_dir` as its 5th argument).

- [ ] **Step 3: Write the implementation**

Add imports to `tools/mcts_az_vs_reference_campaign.py`:

```python
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing
```

Add the CPU-pinning helpers (verbatim from `tools/hermes_canonicalisation_campaign.py`):

```python
def physical_cpus():
    cores = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        base = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key = ((base / 'physical_package_id').read_text(), (base / 'core_id').read_text())
        cores.setdefault(key, cpu)
    return list(cores.values())


def initialize(cpus):
    os.nice(10)
    identity = multiprocessing.current_process()._identity[0]
    os.sched_setaffinity(0, {cpus[(identity - 1) % len(cpus)]})
```

Change `play()` to own its scratch directory internally instead of taking one from the caller:

```python
def play(job, directory, seconds, socket_path, reference_binary):
    initial = job['initial']
    path = Path(directory) / f"game-{job['number']:04d}.json"
    position = validate_setup(initial['workers'], initial['powers'], initial['first'])
    data = {'job': job, 'status': 'running', 'decisions': []}
    atomic_json(path, data)
    try:
        with tempfile.TemporaryDirectory() as scratch:
            scratch_dir = Path(scratch)
            for ply in range(150):
                checked = terminal_position(position)
                if checked.winner is not None:
                    position = checked
                    break
                is_candidate = position.player == job['variant_player']
                if is_candidate:
                    turn, elapsed = candidate_move(position, seconds, socket_path, scratch_dir)
                else:
                    turn, elapsed = reference_move(position, seconds, reference_binary)
                data['decisions'].append({'ply': ply + 1, 'player': position.player,
                                           'backend': 'candidate' if is_candidate else 'reference',
                                           'elapsed': elapsed})
                position = turn.after
                atomic_json(path, {**data, 'position': position.to_dict()})
        data['status'] = 'finished' if position.winner is not None else 'inconclusive'
    except Exception as exc:
        data.update(status='error', error=f'{type(exc).__name__}: {exc}')
    data['winner'] = position.winner
    atomic_json(path, data)
    return {'number': job['number'], 'status': data['status'],
            'variant_won': position.winner == job['variant_player'] if position.winner is not None else None}
```

Add `--workers` to `main()`'s argument parser (default mirrors `hermes_canonicalisation_campaign.py`):

```python
    parser.add_argument('--workers', type=int, default=8)
```

Validate it alongside the existing arg checks:

```python
    if args.workers < 1:
        parser.error('workers must be positive')
```

Replace the sequential job-running section (from Task 3) with parallel execution and periodic progress reporting:

```python
    cpus = physical_cpus()[:args.workers]
    print(f'START {len(pending)}/{len(jobs)} remaining; {args.seconds}s/move; CPUs {cpus}', flush=True)
    try:
        with ProcessPoolExecutor(max_workers=len(cpus), mp_context=multiprocessing.get_context('spawn'),
                                  initializer=initialize, initargs=(cpus,)) as pool:
            futures = {pool.submit(play, job, str(directory), args.seconds, socket_path, str(reference_binary))
                       for job in pending}
            while futures:
                done, futures = wait(futures, timeout=30, return_when=FIRST_COMPLETED)
                for future in done:
                    print('GAME ' + json.dumps(future.result()), flush=True)
                result = summary(directory, corpus)
                report(directory, result)
                print(f"PROGRESS {result['completed']}/{len(jobs)}; errors={len(result['errors'])}", flush=True)
    finally:
        server.stop()
    result = summary(directory, corpus)
    report(directory, result)
    print('FINAL ' + json.dumps(result, ensure_ascii=False), flush=True)
    if result['completed'] != len(jobs):
        raise SystemExit('Campaign requires review: incomplete games')
```

This replaces the `with tempfile.TemporaryDirectory() as scratch: ... results = [play(...) for job in pending]` block introduced in Task 2/3 — `play()` no longer takes a `scratch_dir` argument at all (each call creates its own), and `tempfile` as a top-level import in this file is no longer needed by `main()` itself (still needed by `play()`, which already imports it at module level — no import changes needed beyond what Task 1-3 already had).

- [ ] **Step 4: Run tests to verify they pass**

Run: `QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_mcts_az_vs_reference_campaign.py -v`
Expected: PASS.

- [ ] **Step 5: Manual smoke-run validation (not a pytest test — see Global Constraints)**

This is the first point all four tasks' changes run together end-to-end with real subprocesses, a real GPU/CPU inference server, and real parallel execution. Run a tiny real campaign before trusting this on a full 200-game run:

```bash
python3 tools/mcts_az_vs_reference_campaign.py --checkpoint <path to a real gen-NNNN.pt> \
  --seconds 2 --workers 2 --limit 4 --out /tmp/mcts-az-campaign-smoke/
```

Expected: `rapport.md` and `summary.json` are produced in the output directory, `summary.json`'s `"errors"` list is empty, and `"completed"` equals the `--limit` value (4). Then re-run the exact same command a second time (same `--out`): expected output includes a `PROGRESS 4/4` or equivalent indicating no new games were played (the pending-jobs scan found all 4 already `finished`), and the command still exits 0. If either run fails or leaves any `error` entries, fix before trusting this script with a real 200-game, multi-hour campaign.

- [ ] **Step 6: Commit**

```bash
git add tools/mcts_az_vs_reference_campaign.py tests/test_mcts_az_vs_reference_campaign.py
git commit -m "Parallélisme (ProcessPoolExecutor + affinité CPU) et répertoire de travail par partie"
```

---

## Final check (not a task — run after Task 4)

```bash
QT_QPA_PLATFORM=offscreen python3 -m pytest tests/test_mcts_az_vs_reference_campaign.py -v
```

Expected: all tests across all four tasks pass together, not just within their own task's run. Then perform Task 4 Step 5's manual smoke run if it hasn't been done yet in this session — do not consider blockers 5/6 closed without it, since no automated test in this plan exercises the real parallel/resumability/checkpoint-server integration end to end.

After this plan lands, update memory `santoni-mcts-az-no-power-chantier`: blockers 5 and 6 closed (note the `--inference-socket` flag removal as a breaking CLI change for anyone with it in a script/alias). Blocker 4 (batched leaf evaluation across branches within one self-play worker) remains open and was deliberately deferred by the user's own choice earlier in this chantier — it's a GPU-throughput question, not a correctness or data-loss risk, and still needs its own design conversation before implementation.

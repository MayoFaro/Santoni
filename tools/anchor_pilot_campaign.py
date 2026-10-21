"""Live anchor pilot: a training campaign where an MCTS+NN candidate learns
by playing against a live reference engine, starting from a real "anchor"
position -- `back_plies` plies before an actual reference-vs-reference win
in a real no-power game.

`prepare_anchor` is the one-time, resumable first step: it plays (or
replays, if `out_dir` already holds a result) one reference-vs-reference
game via `tools.proven_anchor.anchor_games.play_reference_vs_reference`,
walks back `back_plies` plies from the terminal position, and freezes that
position to `out_dir/anchor_state.bin` -- encoded the same way
`tools.mcts_az_vs_reference_campaign.candidate_move` encodes a position for
its Rust worker's `--input-state` argument, via `santorini.native.encode`.

Resumable in the same spirit as `tools.mcts_az_vs_reference_campaign
.freeze_reference`: if `anchor_state.bin` already exists, it is returned
immediately without replaying the seed game -- a long real seed-game search
(2026-10-21: one real run burned 12h14m) must never be redone just because
an earlier step of a multi-step campaign crashed or was interrupted after
the anchor was already frozen. Later tasks in this chantier add more
functions to this same module; keep this one and its helpers self-contained
rather than reaching for functions that don't exist yet."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import signal
import time
from pathlib import Path

from santorini.native import encode

from santoni_az.anchor_pilot import AnchorPilotOrchestrator
from tools.mcts_az_vs_reference_campaign import CORPUS, freeze_reference
from tools.proven_anchor.anchor_games import play_reference_vs_reference

ANCHOR_STATE_FILENAME = 'anchor_state.bin'
ANCHOR_MANIFEST_FILENAME = 'anchor_manifest.json'

ROOT = Path(__file__).resolve().parents[1]
WORKER_BINARY = ROOT / 'native_engine' / 'target' / 'release' / 'selfplay_worker'
WORKER_FROZEN_FILENAME = 'selfplay_worker'
WORKER_MANIFEST_FILENAME = 'worker_manifest.json'


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Temp-file-then-replace, same convention as every other atomic write
    in this chantier (`Trainer.save_checkpoint`, `mcts_io::write_game_atomically`
    in Rust, `atomic_json` in the campaign scripts): a crash mid-write can
    only ever leave a stray `.tmp` file behind, never a partial
    `anchor_state.bin` that a later resumed run would mistake for real."""
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_bytes(data)
    tmp.replace(path)


def prepare_anchor(out_dir: Path, opening_job: dict, back_plies: int, seconds: float,
                    reference_binary, progress=None) -> Path:
    """Produce (or, on a repeat call, simply locate) the anchor position
    for the live pilot: `back_plies` plies before the terminal position of
    one real reference-vs-reference game started from `opening_job`.

    Idempotent: if `out_dir / "anchor_state.bin"` already exists, it is
    returned immediately -- `play_reference_vs_reference` is never called
    on a resumed run, since that is the expensive, non-deterministic step
    this function exists to avoid repeating.

    Otherwise, plays the seed game, computes
    `index = len(positions) - 1 - back_plies`, and raises `ValueError` if
    that index is negative (the seed game ended too quickly to go back
    `back_plies` plies from its terminal position) -- naming both the
    actual seed-game length and the number of plies required, rather than
    letting a negative index silently wrap into a Python list and report a
    confusing wrong-position bug instead of a clear one.

    The qualifying position is encoded via `santorini.native.encode` (the
    same encoding `tools.mcts_az_vs_reference_campaign.candidate_move` uses
    for its Rust worker's `--input-state` argument) and written atomically.
    A small `anchor_manifest.json` is written alongside it for later
    inspection: the opening job, `back_plies`, the seed game's own length,
    and the anchor position's `player` (who is to move there)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / ANCHOR_STATE_FILENAME
    if state_path.exists():
        return state_path

    positions = play_reference_vs_reference(opening_job, seconds, reference_binary,
                                              progress=progress)
    index = len(positions) - 1 - back_plies
    if index < 0:
        raise ValueError(
            f'seed game only has {len(positions)} plies, need at least '
            f'{back_plies + 1} to go back {back_plies} plies'
        )
    anchor = positions[index]

    state = encode(anchor)
    _atomic_write_bytes(state_path, bytes(state))

    manifest_path = out_dir / ANCHOR_MANIFEST_FILENAME
    manifest = {
        'opening_job': opening_job,
        'back_plies': back_plies,
        'seed_game_plies': len(positions),
        'anchor_player': anchor.player,
    }
    tmp_manifest = manifest_path.with_suffix(manifest_path.suffix + '.tmp')
    tmp_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    tmp_manifest.replace(manifest_path)

    return state_path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze_worker_binary(out_dir: Path) -> tuple[Path, str]:
    """Snapshot the compiled `selfplay_worker` currently at `WORKER_BINARY`
    (today's live `native_engine/target/release` build) into `out_dir`, the
    same provenance-pin rationale as `tools.mcts_az_vs_reference_campaign
    .freeze_reference`/`freeze_checkpoint` applied to the worker binary this
    live pilot actually shells out to: pin precisely what was measured, so a
    `native_engine` rebuild mid-campaign can never silently blend old and
    new engine behaviour into one run.

    On the first call, copies `WORKER_BINARY` into `out_dir` and records its
    sha256 in a manifest. On every call -- including the first, and
    including later idempotent calls -- hashes the *current* on-disk
    `WORKER_BINARY` and compares it against the hash recorded in the
    manifest. This chantier already shipped `freeze_checkpoint` once with a
    bug where it only compared the frozen copy's hash to itself, which
    never notices a `--checkpoint` pointed at a different file on a resumed
    run (see that function's own fix comment); hashing the frozen copy
    alone here would reproduce exactly that bug for the worker binary, so
    this hashes `WORKER_BINARY` itself every time instead. A mismatch
    raises `ValueError` naming both the frozen generation's hash and the
    current on-disk hash, rather than silently continuing to serve the
    stale frozen copy under the old provenance record.

    Cheap and side-effect-free when nothing changed: an already-frozen,
    unchanged binary costs one hash of `WORKER_BINARY` and one manifest
    read -- the frozen copy is never re-copied once it exists."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frozen = out_dir / WORKER_FROZEN_FILENAME
    manifest_path = out_dir / WORKER_MANIFEST_FILENAME
    current_sha256 = _sha256(WORKER_BINARY)

    if not manifest_path.exists():
        shutil.copy2(WORKER_BINARY, frozen)
        tmp_manifest = manifest_path.with_suffix(manifest_path.suffix + '.tmp')
        tmp_manifest.write_text(
            json.dumps({'worker_sha256': current_sha256}, ensure_ascii=False, indent=2) + '\n')
        tmp_manifest.replace(manifest_path)
        return frozen, current_sha256

    frozen_sha256 = json.loads(manifest_path.read_text())['worker_sha256']
    if current_sha256 != frozen_sha256:
        raise ValueError(
            f'selfplay_worker at {WORKER_BINARY} has sha256={current_sha256}, which differs '
            f'from sha256={frozen_sha256} frozen for this campaign in {out_dir} -- the '
            f'native_engine binary changed since this campaign started (a rebuild mid-run); '
            f'choose a new output directory or restore the frozen binary before resuming'
        )
    return frozen, frozen_sha256


def main():
    """CLI entrypoint: freeze the worker binary, prepare (or locate) the
    anchor position, then start and supervise an `AnchorPilotOrchestrator`
    until a SIGTERM/SIGINT asks for a clean shutdown -- same shape as every
    other long-running launcher in this chantier (`selfplay-runs/run.py`,
    `run2.py`): a handler that only *records* the request, a status-printing
    loop that *reads* that record and is the sole place `.stop()` is ever
    called from, so a signal can never cut off a `.stop()` already in
    progress.

    Two distinct build artifacts get frozen here, both from today's
    `native_engine` build, for two distinct roles -- despite the similarly
    named `freeze_worker_binary`/`freeze_reference`, these are not
    interchangeable: `prepare_anchor`'s `reference_binary` argument flows
    into `tools.mcts_az_vs_reference_campaign.reference_move`, which
    `set_engine`s it and calls `native_search` -- an in-process alpha-beta
    search through `native.library()`'s `ctypes.CDLL`, which requires the
    compiled shared library (`libsantoni_engine.so`), never the
    `selfplay_worker` *executable* (`ctypes.CDLL` on an ELF executable
    raises `OSError: cannot dynamically load position-independent
    executable` -- caught by this module's own smoke test). So the one-time
    anchor search is pinned via `freeze_reference` (the `.so`), while
    `AnchorPilotOrchestrator`'s every later `--vs-reference` subprocess game
    is pinned via `freeze_worker_binary` (the `selfplay_worker` binary) --
    see that function's own docstring."""
    parser = argparse.ArgumentParser(
        description='Live anchor pilot: a candidate MCTS+NN network learns by playing '
                     'against a live reference engine, every game starting from one fixed '
                     'anchor position (back_plies before a real reference-vs-reference win).'
    )
    parser.add_argument('--out', required=True, type=Path,
                         help='Output directory: anchor state, frozen worker binary, '
                              'games/eval_games/checkpoints dirs are all written here.')
    parser.add_argument('--corpus', type=Path, default=CORPUS,
                         help='No-power opening-job corpus the one-time anchor seed game is '
                              'sourced from (default: the same corpus '
                              'tools.mcts_az_vs_reference_campaign uses).')
    parser.add_argument('--back-plies', type=int, default=5,
                         help='Plies to walk back from the seed game terminal position to '
                              'reach the anchor position.')
    parser.add_argument('--train-workers', type=int, required=True,
                         help='Number of training (--vs-reference, non-deterministic) workers.')
    parser.add_argument('--eval-workers', type=int, required=True,
                         help='Number of eval (--vs-reference --deterministic) workers.')
    parser.add_argument('--simulations-per-move', type=int, default=32,
                         help="MCTS simulations per candidate move.")
    parser.add_argument('--reference-seconds', type=float, default=5.0,
                         help='Reference engine per-move time budget, both for the one-time '
                              'anchor seed game and every later --vs-reference game.')
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Frozen first: `prepare_anchor` below needs the frozen reference
    # library's path as its own `reference_binary` argument (see this
    # function's docstring for why this is a different artifact from the
    # `selfplay_worker` binary frozen right after).
    reference_binary_path, reference_manifest = freeze_reference(args.out)
    print(f"REFERENCE frozen: {reference_binary_path} "
          f"sha256={reference_manifest['engine_sha256']}", flush=True)

    worker_binary_path, worker_sha256 = freeze_worker_binary(args.out)
    print(f'WORKER frozen: {worker_binary_path} sha256={worker_sha256}', flush=True)

    jobs = json.loads(args.corpus.read_text())['jobs']
    opening_job = jobs[0]

    def anchor_progress(message):
        print(f'ANCHOR {message}', flush=True)

    anchor_state_path = prepare_anchor(
        args.out, opening_job, args.back_plies, args.reference_seconds,
        reference_binary_path, progress=anchor_progress,
    )
    print(f'ANCHOR ready: {anchor_state_path}', flush=True)

    orchestrator = AnchorPilotOrchestrator(
        args.out, anchor_state_path, worker_binary_path,
        args.train_workers, args.eval_workers,
        simulations_per_move=args.simulations_per_move,
        reference_seconds=args.reference_seconds,
    )

    # Same pattern as `selfplay-runs/run.py`/`run2.py`: the handler only
    # flips a flag the status loop below reads every 10s -- `.stop()` is
    # called exactly once, from the loop, after it observes the flag, never
    # from inside the signal handler itself.
    stopping = False

    def handle_term(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, handle_term)
    signal.signal(signal.SIGINT, handle_term)

    orchestrator.start()
    print(f'orchestrator started, run_dir={args.out}, '
          f'train_workers={args.train_workers}, eval_workers={args.eval_workers}', flush=True)

    # `try/finally` from here on: `orchestrator.stop()` must run even if
    # something inside the loop below raises (a future `status()` change,
    # an `OSError` from a glob, ...) -- without this, an exception would
    # unwind straight out of `main()` leaving both `Popen` worker pools
    # (launched with `--games 100000`, so they never stop on their own)
    # orphaned: still burning CPU, still writing into `games/` with nothing
    # consuming them and a now-dead `infer.sock`. Review finding: on an
    # unattended multi-day run with no other stop ceiling, that is exactly
    # the silent-failure mode this whole plan exists to prevent.
    try:
        while not stopping:
            time.sleep(10)
            status = orchestrator.status()
            eval_ci_lower = status['eval_ci_lower']
            last_loss = status['last_loss']
            eval_n = status['eval_n']
            # `eval_ci_lower` and `last_loss` are both `None` until, resp.,
            # the first eval game and the first real gradient step (see
            # `AnchorPilotOrchestrator.status`'s own docstring). Formatting
            # or comparing either `None` here would crash this loop
            # outright, so both print 'n/a' instead.
            eval_ci_str = 'n/a' if eval_ci_lower is None else f'{eval_ci_lower:.3f}'
            loss_str = 'n/a' if last_loss is None else f'{last_loss:.4f}'
            # Review finding: `eval_n` (the rolling `EVAL_WINDOW`-capped
            # size `eval_win_rate` is computed over) and
            # `eval_games_available` (the total eval games ever written to
            # disk) are different numbers -- a 644-eval-game real smoke run
            # only ever showed `eval_n=40` previously, with no way to tell
            # "40 total" from "40 of many more". Printing both, plus the
            # actual win rate (not just its conservative lower bound), is
            # what spec Sec 3.4 ("parties d'évaluation jouées, taux de
            # victoire courant + IC") actually asks for.
            eval_win_rate_str = 'n/a' if eval_n == 0 else f"{status['eval_wins']}/{eval_n}"
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} "
                  f"generation={status['generation']} "
                  f"loss={loss_str} "
                  f"games_available={status['games_available']} "
                  f"eval_games={status['eval_games_available']} "
                  f"eval_win_rate={eval_win_rate_str} "
                  f"eval_ci_lower={eval_ci_str} "
                  f"workers_alive={status['workers_alive']}", flush=True)
    finally:
        print('stop requested, shutting down cleanly...', flush=True)
        orchestrator.stop()
        print('stopped', flush=True)


if __name__ == '__main__':
    main()

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

import hashlib
import json
import shutil
from pathlib import Path

from santorini.native import encode

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

"""Source anchor games: real reference-vs-reference games containing a
position with a provable forced win at an exact ply distance.

`scan_game_for_forced_win` walks a game's position history and asks the
reference engine's own proof search (`santorini.native.native_search`,
imported at module level so tests can `monkeypatch.setattr(mod,
"native_search", ...)`) whether any position is a proven forced **win**
for the player to move at exactly `target_distance` plies. The sign of
`analysis.score` is checked directly rather than through
`tools.proven_anchor.verdicts.mate_distance`, which is `MATE -
abs(analysis.score)` and therefore sign-blind: a forced *loss* in six
scores `-(MATE - 6)` and would be indistinguishable from the forced win in
six that spec section 3.1 actually requires. The first qualifying index is
returned, or `None` if no position in the game qualifies within the given
per-call `seconds` budget.

`play_reference_vs_reference` generates one such game: the frozen reference
engine plays both sides (mirrors the loop shape of
`tools.mcts_az_vs_reference_campaign.play`, but without the
candidate/reference branch -- every ply calls
`tools.mcts_az_vs_reference_campaign.reference_move`), returning the full
ply-by-ply position history, starting position included, stopping once
`santorini.engine.terminal_position` reports a winner.

`find_anchor_games` ties the two together over a list of opening jobs,
collecting qualifying `(positions, index)` pairs until `minimum` are found,
stopping early rather than playing out every opening once enough anchors
exist. Exhausting `openings` without reaching `minimum` raises `ValueError`
-- proceeding with too few anchors would silently undermine the curriculum
built on top of this (Task 8 requires at least two), so this fails loudly
and immediately instead.
"""
from __future__ import annotations

from santorini.engine import Position, terminal_position, validate_setup
from santorini.native import native_search

from santorini.search import MATE

from tools.mcts_az_vs_reference_campaign import reference_move

MAX_PLIES = 150


def _progress(message):
    print(message, flush=True)


def scan_game_for_forced_win(positions: list[Position], target_distance: int, seconds: float,
                              progress=_progress) -> int | None:
    """First index in `positions` whose player to move has a proven forced
    **win** in exactly `target_distance` plies, or `None` if none has.

    `analysis.score` is signed from `position.player`'s own perspective
    (see `tools.proven_anchor.verdicts`' module docstring), so a positive
    score is the required forced win and a negative one is a forced loss --
    which must *not* qualify, hence the explicit sign test instead of
    `mate_distance`'s `MATE - abs(score)`.

    `progress` (default: prints to stdout, pass `None` to silence) is
    called once per ply before its proof search -- each search can spend
    up to `seconds`, so a long real run (2026-10-21: 12h14m with zero
    visibility into whether it was close to finding anything) needs to see
    which ply it's on, not just silence until the whole game is scanned."""
    total = len(positions)
    for index, position in enumerate(positions):
        if progress is not None:
            progress(f"  scan ply {index + 1}/{total} (target_distance={target_distance})")
        analysis = native_search(position, seconds)
        if not analysis.proven or analysis.score <= 0:
            continue
        if MATE - analysis.score == target_distance:
            return index
    return None


def play_reference_vs_reference(opening_job, seconds: float, reference_binary,
                                  progress=_progress) -> list[Position]:
    """Play the frozen reference engine against itself from `opening_job`
    (a dict with `workers`/`powers`/`first`, the same shape as one entry of
    `tools.generate_no_power_corpus.py`'s corpus), returning every position
    visited -- including the starting one -- up to and including the
    terminal position.

    Same loop shape as `tools.mcts_az_vs_reference_campaign.play`, except
    there is no candidate side: both players move via `reference_move`.

    `progress` (default: prints to stdout, pass `None` to silence) is
    called once per move played, reporting the move's own search time --
    the other half of the 2026-10-21 visibility gap (a stuck game heading
    toward `MAX_PLIES` was indistinguishable from a hung process)."""
    position = validate_setup(opening_job['workers'], opening_job['powers'], opening_job['first'])
    positions = [position]
    for ply in range(MAX_PLIES):
        checked = terminal_position(position)
        if checked.winner is not None:
            positions[-1] = checked
            break
        turn, elapsed = reference_move(position, seconds, reference_binary)
        position = turn.after
        positions.append(position)
        if progress is not None:
            progress(f"  move {ply + 1}/{MAX_PLIES} played ({elapsed:.1f}s)")
    return positions


def find_anchor_games(openings: list[dict], target_distance: int, seconds: float, reference_binary,
                       minimum: int = 2, progress=_progress) -> list[tuple[list[Position], int]]:
    """Play through `openings` (reference vs reference) until `minimum`
    games are found containing a position with a proven forced win at
    exactly `target_distance` plies, pairing each qualifying game's full
    position history with the index of that qualifying position.

    Stops as soon as `minimum` anchors are collected -- the remaining
    openings are never played out. Raises `ValueError` if `openings` is
    exhausted first: silently proceeding with fewer than `minimum` anchors
    would undermine whatever downstream curriculum depends on having them,
    so this fails loudly instead.

    A game whose qualifying position is *identical* to an already-accepted
    anchor's is rejected: two such anchors would give the curriculum the
    same ancestor twice under two names, so one game's held-out children
    would be the other game's own training material. Near-identical (as
    opposed to equal) qualifying positions are not detected here -- they
    are caught downstream instead, by the orchestrator excluding every
    position it has trained on from every held-out pool.

    `progress` (default: prints to stdout, pass `None` to silence) receives
    messages prefixed with the opening's 1-based index and total count
    (e.g., `"[opening 1/2] ..."`), forwarding both `play_reference_vs_reference`
    and `scan_game_for_forced_win` messages, plus a summary line per opening
    reporting anchors found so far."""
    found: list[tuple[list[Position], int]] = []
    qualifying_positions: set[Position] = set()
    for i, opening_job in enumerate(openings):
        # Create a scoped progress function that prefixes messages with opening index.
        def opening_progress(message: str) -> None:
            prefix = f"[opening {i + 1}/{len(openings)}] "
            if progress is not None:
                progress(prefix + message)

        positions = play_reference_vs_reference(opening_job, seconds, reference_binary,
                                                progress=opening_progress)
        index = scan_game_for_forced_win(positions, target_distance, seconds,
                                         progress=opening_progress)
        if index is not None and positions[index] not in qualifying_positions:
            qualifying_positions.add(positions[index])
            found.append((positions, index))

        # Emit summary line per opening.
        if progress is not None:
            progress(f"[opening {i + 1}/{len(openings)}] {len(found)}/{minimum} anchor(s) found")

        if len(found) >= minimum:
            return found

    raise ValueError(
        f'Only found {len(found)} anchor game(s) with a forced win at ply distance '
        f'{target_distance} among {len(openings)} opening(s); need at least {minimum}.'
    )

"""Exact win/loss verdicts for individual moves, derived from the reference
engine's own proof search (`santorini.native.native_search`).

Sign convention (verified against `santorini.native.decode_analysis`): a
proven `analysis` returned by `native_search(position, seconds)` carries
`analysis.score` signed from `position.player`'s own perspective --
`decode_analysis` labels the result "Victoire forcée démontrée" when
`value.score > 0` and "Défaite forcée démontrée" when `value.score < 0`,
with no further transform, so a positive score is a forced win *for the
player to move in the analysed position*, not for any externally fixed
side. The pure-Python reference search (`santorini.search.search`) uses the
same convention relative to its own root.

`evaluate_child_verdict` analyses `child_position` -- the position reached
right after some move by `mover` -- so `child_position.player` is the
*opponent* of `mover` (whoever moves next). A forced win for that opponent
(`analysis.score > 0`) means the move was a proven loss for `mover`, and a
forced loss for that opponent (`analysis.score < 0`) means the move was a
proven win for `mover`.
"""
from __future__ import annotations

from santorini.native import native_search
from santorini.search import MATE


def mate_distance(analysis) -> int | None:
    """Plies to a proven mate, or `None` if `analysis` isn't a proof."""
    if not analysis.proven:
        return None
    return MATE - abs(analysis.score)


def evaluate_child_verdict(child_position, mover: int, seconds: float) -> bool | None:
    """Was the move that produced `child_position` a proven win for `mover`?

    Returns `True` for a proven win, `False` for a proven loss, `None` if
    the reference engine can't resolve it within `seconds`.
    """
    if child_position.winner is not None:
        return child_position.winner == mover
    analysis = native_search(child_position, seconds)
    if not analysis.proven:
        return None
    # analysis.score is signed from child_position.player's perspective
    # (the opponent of `mover`): that side losing (score < 0) is exactly
    # the move being a proven win for `mover`.
    return analysis.score < 0

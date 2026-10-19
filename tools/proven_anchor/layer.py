"""Derive a position's proven value from its children's verdicts.

`build_layer` enumerates every legal move out of an "ancestor" position via
`santorini.native.native_turns`, asks `evaluate` (by default
`tools.proven_anchor.verdicts.evaluate_child_verdict`, injectable for tests)
whether each resulting child was a proven win for the player who moved into
it, and backs the per-child verdicts up into a single verdict for the
ancestor using the standard minimax rule:

- `True`  if any resolved child is a proven win for the mover (one good
  move is enough -- the opponent doesn't get to pick which child is played).
- `False` only once *every* child is resolved and none of them is a proven
  win: a genuine local forced loss for the mover, however that position
  relates to any historical anchor game's own forced-win line. An anchor
  game's own move being on a winning line does not guarantee every
  ancestor further back is winning too -- that is exactly what the
  per-child verdicts computed here are for, rather than assumed away.
- `None` otherwise: some children remain unresolved and none proven
  winning yet, so the ancestor's value is still undetermined.
"""
from __future__ import annotations

from dataclasses import dataclass

from santorini.engine import Position, Turn
from santorini.native import native_turns

from tools.proven_anchor.verdicts import evaluate_child_verdict


@dataclass
class LayerResult:
    ancestor: Position
    children: list[tuple[Turn, bool | None]]
    resolved_fraction: float
    ancestor_verdict: bool | None


def build_layer(ancestor: Position, seconds: float, evaluate=evaluate_child_verdict) -> LayerResult:
    turns = native_turns(ancestor)
    children: list[tuple[Turn, bool | None]] = [
        (turn, evaluate(turn.after, ancestor.player, seconds)) for turn in turns
    ]

    resolved_count = sum(1 for _, verdict in children if verdict is not None)
    resolved_fraction = resolved_count / len(children) if children else 0.0

    if any(verdict is True for _, verdict in children):
        ancestor_verdict = True
    elif resolved_fraction == 1.0:
        ancestor_verdict = False
    else:
        ancestor_verdict = None

    return LayerResult(
        ancestor=ancestor,
        children=children,
        resolved_fraction=resolved_fraction,
        ancestor_verdict=ancestor_verdict,
    )

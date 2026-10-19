"""Turn resolved `LayerResult`s (Task 4) into supervised training examples
for `santoni_az.trainer.Trainer`: `(planes, policy_target, value_target)`
tuples with exact value labels and a policy target built from the
proven-winning moves out of each ancestor position, instead of the noisy
self-play visit counts this curriculum is meant to replace.

Two kinds of example come out of one `LayerResult`:

- `examples_from_layer` -- one example for the **ancestor** itself, value
  target `+1.0` (it is only emitted when `ancestor_verdict is True`, i.e.
  the ancestor's own player to move has a proven win), policy target
  uniform over its proven-winning moves.
- `child_examples_from_layer` -- one example per **resolved, non-terminal
  child**, value target `-1.0` or `+1.0` with *no* policy supervision.
  These labels are already paid for by `build_layer`'s ~33 proof searches
  per ancestor, and without them the value head would only ever see `+1.0`
  targets, which a constant-output network minimizes trivially.

**Value-label perspective (the one convention both functions obey).** A
value target is read from the perspective of the player to move in the
position the planes encode -- the same convention
`santoni_az.trainer.Trainer._decode_position` applies to self-play games
(`value_target = outcome if ply % 2 == 0 else -outcome`) and the same one
`tools.proven_anchor.validation` measures against. A child's verdict means
"a proven win for the *ancestor's* mover", and the child's own `player` is
that mover's opponent, so a child with verdict `True` gets value target
`-1.0` (its own player to move is the one losing) and a child with verdict
`False` gets `+1.0`.

`action_index` mirrors `native_engine::mcts::action_index` and its call site
in `Mcts::simulate_one` (`native_engine/src/mcts.rs`) exactly:

    pub fn action_index(move_action: &Action, build_action: &Action, worker_slot: usize) -> usize {
        let dest = move_action.target as usize;
        let build = build_action.target as usize;
        worker_slot * 625 + dest * 25 + build
    }

with `worker_slot` computed the same way the Rust expansion code does --
`state.workers[to_move].iter().position(|&c| c == mv.source)` -- i.e. the
*index* of the moved worker's source cell within the mover's own worker
list, not `Action.worker` (a different, unrelated field). For a winning turn
(a single action, no real build -- see
`legal_children_synthesizes_a_placeholder_build_for_winning_moves` in
`mcts.rs`), the synthesized placeholder build targets the move's own
destination, so `build == dest` in that case; this is pure arithmetic over
fields already on `Turn`/`Position`, no FFI call needed.
"""
from __future__ import annotations

import torch

from santorini.engine import Position, Turn

from santoni_az.network import ACTION_SPACE, NUM_PLANES
from tools.proven_anchor.layer import LayerResult
from tools.proven_anchor.planes import compute_planes


def action_index(position: Position, turn: Turn) -> int:
    worker_slot = position.workers[position.player].index(turn.actions[0].source)
    dest = turn.actions[0].target
    build = turn.actions[1].target if len(turn.actions) > 1 else dest
    return worker_slot * 625 + dest * 25 + build


def _planes(position: Position) -> torch.Tensor:
    return torch.frombuffer(
        bytearray(compute_planes(position)), dtype=torch.uint8
    ).reshape(NUM_PLANES, 5, 5).float()


def examples_from_layer(layer: LayerResult) -> list[tuple[torch.Tensor, torch.Tensor, float]]:
    if layer.ancestor_verdict is not True:
        # Either a proven loss for the mover (doesn't happen for a true
        # anchor ancestor, but handled defensively) or still undetermined:
        # nothing useful to train on yet from this layer.
        return []

    winning_indices = {
        action_index(layer.ancestor, turn)
        for turn, verdict in layer.children
        if verdict is True
    }

    policy_target = torch.zeros(ACTION_SPACE)
    mass = 1.0 / len(winning_indices)
    for idx in winning_indices:
        policy_target[idx] = mass

    return [(_planes(layer.ancestor), policy_target, 1.0)]


def child_examples_from_layer(
    layer: LayerResult, children=None
) -> list[tuple[torch.Tensor, torch.Tensor, float]]:
    """Value-only training examples for `layer`'s own resolved children.

    `children` restricts which `(turn, verdict)` pairs are turned into
    examples, defaulting to all of `layer.children`. The orchestrator
    passes only its training split: the other half of the children is the
    held-out anti-memorization evidence and must never be trained on (see
    `tools.proven_anchor_curriculum._partition_children`).

    One example per child whose verdict the reference engine actually
    proved, with the value label flipped into the child's own
    player-to-move perspective (see the module docstring): verdict `True`
    (a proven win for the ancestor's mover) means the child's own mover is
    losing, so `-1.0`; verdict `False` means the child's own mover is
    winning, so `+1.0`. Unresolved (`None`) children are skipped -- never
    treated as `False` (plan Review Focus).

    Already-terminal children (`winner is not None`) are skipped too: the
    network is never asked to evaluate a finished position (the search
    uses the exact terminal value), so such an example trains the value
    head on inputs it will never see at inference time.

    The policy target is an all-zero vector, i.e. deliberately *no* policy
    supervision for these positions: we know the child's exact value but
    not which of *its* moves are winning (that would need the next layer's
    proofs). This is not a filler value -- `Trainer.train_step` computes
    `F.cross_entropy(logits, policy_targets)` with probability targets, and
    an all-zero target row contributes exactly zero to the loss and zero
    gradient to the policy head, so these examples supervise the value
    head alone."""
    examples: list[tuple[torch.Tensor, torch.Tensor, float]] = []
    for turn, verdict in (layer.children if children is None else children):
        if verdict is None or turn.after.winner is not None:
            continue
        examples.append((
            _planes(turn.after),
            torch.zeros(ACTION_SPACE),
            -1.0 if verdict else 1.0,
        ))
    return examples

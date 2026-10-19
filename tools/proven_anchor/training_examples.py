"""Turn resolved `LayerResult`s (Task 4) into supervised training examples
for `santoni_az.trainer.Trainer`: `(planes, policy_target, value_target)`
tuples with an exact 1.0/-1.0 value label and a policy target built from the
proven-winning moves out of each ancestor position, instead of the noisy
self-play visit counts this curriculum is meant to replace.

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

    planes = torch.frombuffer(
        bytearray(compute_planes(layer.ancestor)), dtype=torch.uint8
    ).reshape(NUM_PLANES, 5, 5).float()
    policy_target = torch.zeros(ACTION_SPACE)
    mass = 1.0 / len(winning_indices)
    for idx in winning_indices:
        policy_target[idx] = mass

    return [(planes, policy_target, 1.0)]

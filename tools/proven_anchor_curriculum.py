"""Orchestrate the proven-anchor curriculum's widening loop (spec
`docs/superpowers/specs/2026-10-19-mcts-az-proven-anchor-curriculum-design.md`,
Secs 2, 3.6, 3.7): walk each anchor game's history backwards six plies at a
time, train the network on each ancestor's proven layer (Tasks 1-7), and
require the anti-memorization validation gate to clear baseline before
widening one half-move further back.

This module is deliberately the only piece of the curriculum that ties the
loop together; it never imports or calls anything from
`santoni_az.orchestrator` (spec Sec 3.3's self-play-paused constraint --
the continuous self-play `Orchestrator`/`run2.py` stays stopped for the
entire duration of this construction/validation phase, so this module has
no business touching it at all).

Module-level imports, not local ones: `build_layer`, `examples_from_layer`
and `layer_passes_validation` are imported by name here specifically so
tests can `monkeypatch.setattr(this_module, "build_layer", ...)` etc. and
have that actually intercept `run_curriculum`'s behaviour. Importing them
inside the function body, or calling them via a dotted path
(`tools.proven_anchor.layer.build_layer(...)`), would both defeat that
monkeypatch -- `run_curriculum` must call these as bare module-level names.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from tools.proven_anchor.anchor_games import find_anchor_games
from tools.proven_anchor.layer import build_layer
from tools.proven_anchor.symmetry import BOARD_SYMMETRIES
from tools.proven_anchor.training_examples import examples_from_layer
from tools.proven_anchor.validation import layer_passes_validation
from tools.mcts_az_vs_reference_campaign import CORPUS, freeze_reference

DISTANCE_STEP = 6


def _active_ancestors(anchor_games, back):
    """The position `back` plies before each anchor game's own qualifying
    position, or `None` for a game whose own history runs out at this
    distance -- a normal, reportable stop (spec Sec 3.2), not an error."""
    ancestors = []
    for positions, qualifying_ply in anchor_games:
        index = qualifying_ply - back
        ancestors.append(positions[index] if index >= 0 else None)
    return ancestors


def _children_held_out(layer):
    """Children with a resolved proven verdict -- spec Sec 2's "held-out
    pool" sourced from an anchor game's own resolved branches. Deliberately
    kept separate from the symmetrized-ancestor pool below: which of a
    *different* anchor game's children may validate a given game's
    training is a per-game decision made by the caller (leave-one-out),
    not something this helper can decide on its own."""
    return [
        (turn.after, verdict)
        for turn, verdict in (getattr(layer, "children", None) or [])
        if verdict is not None
    ]


def _symmetrized_ancestor_held_out(layer):
    """Every non-identity `BOARD_SYMMETRIES` transform of a *trained*
    ancestor (spec Sec 3.4's anti-memorization check: the literal trained
    position's symmetric images must agree, without any extra training).

    Only emitted when the ancestor was actually trained on
    (`ancestor_verdict is True` -- the exact case `examples_from_layer`
    produces a real example for, spec Sec 2's exact labels): a resolved
    loss or a still-unresolved ancestor was never fed to the trainer, so
    its symmetries carry no memorization signal to check. The identity
    transform is skipped: it reproduces the literal trained position
    verbatim, which is not held out at all."""
    ancestor_verdict = getattr(layer, "ancestor_verdict", None)
    ancestor = getattr(layer, "ancestor", None)
    if ancestor_verdict is not True or ancestor is None:
        return []
    held_out = []
    for symmetry in BOARD_SYMMETRIES:
        transformed = symmetry(ancestor)
        if transformed != ancestor:
            held_out.append((transformed, ancestor_verdict))
    return held_out


def run_curriculum(anchor_games, reference_binary, seconds, trainer, max_uninformative_layers=3):
    """Drive the layer-by-layer widening loop over `anchor_games` (each a
    `(positions, qualifying_ply)` pair, the exact shape
    `tools.proven_anchor.anchor_games.find_anchor_games` returns).

    Returns a summary dict: `layers_built` (total `build_layer` calls made
    across every anchor game and distance) and `outcome`, one of
    `"stopped_success_exhausted"` (every attempted layer validated; the
    loop only stopped because the anchor games' own history ran out --
    spec Sec 2 imposes no depth ceiling, so this is not expected to happen
    in practice with a healthy set of anchor games, but is a real, correct
    stop condition kept for a future manual/forced stop), `"informative_failure"`
    (spec Sec 2's "echec informatif": validation never cleared baseline for
    `max_uninformative_layers` consecutive distances, or the anchor games'
    history ran out while a failure streak was still open), or `"error"`
    (an unexpected exception during the loop; the real-engine calls inside
    `build_layer` are the only part of this loop that can genuinely fail at
    runtime, e.g. reference-binary/process trouble, and a CLI run should
    report that cleanly rather than crash).
    """
    if len(anchor_games) < 2:
        raise ValueError(
            "run_curriculum needs at least 2 anchor games to pool a "
            "statistically usable held-out validation set at every "
            "distance (spec Sec 2's precondition); got "
            f"{len(anchor_games)}."
        )

    try:
        return _run_curriculum_loop(anchor_games, seconds, trainer, max_uninformative_layers)
    except Exception as exc:  # pragma: no cover - exercised only by real engine failures
        return {"layers_built": 0, "outcome": "error", "error": str(exc)}


def _run_curriculum_loop(anchor_games, seconds, trainer, max_uninformative_layers):
    baseline_net = copy.deepcopy(getattr(trainer, "net", None))

    layers_built = 0
    consecutive_failures = 0
    validated_held_out_sets = []
    back = 0
    outcome = "stopped_success_exhausted"

    while True:
        ancestors = _active_ancestors(anchor_games, back)
        active = [ancestor for ancestor in ancestors if ancestor is not None]
        if len(active) < 2:
            # Not enough anchor-game history left at this distance to pool
            # a valid held-out set (spec Sec 2's two-anchor precondition
            # applies at every distance, not only the first) -- a normal
            # stop, not an error.
            break

        layer_results = [build_layer(ancestor, seconds) for ancestor in active]
        layers_built += len(layer_results)

        for layer in layer_results:
            trainer.add_exact_examples(examples_from_layer(layer))
            trainer.train_step()

        net = getattr(trainer, "net", None)

        # Spec §2 / brief: a game's own children must never validate its
        # own training, even partially -- every active game trains this
        # round, so a single pooled held-out set drawn from *all* of them
        # would let each game's children sit beside that same game's own
        # just-trained ancestor (one ply away, same tactical line),
        # diluting exactly the independence the gate exists to check.
        # Leave-one-out per game instead: game i's held-out evidence is
        # every *other* active game's resolved children this round, plus
        # game i's own trained ancestor's symmetric images -- and every
        # per-game check must independently clear baseline for the round
        # as a whole to pass.
        children_pools = [_children_held_out(layer) for layer in layer_results]
        symmetrized_pools = [_symmetrized_ancestor_held_out(layer) for layer in layer_results]

        round_held_out_sets = []
        per_game_passed = []
        for i in range(len(layer_results)):
            other_children = [
                item for j, pool in enumerate(children_pools) if j != i for item in pool
            ]
            held_out_i = other_children + symmetrized_pools[i]
            round_held_out_sets.append(held_out_i)
            per_game_passed.append(layer_passes_validation(net, baseline_net, held_out_i))
        passed = all(per_game_passed)

        if passed:
            # Spec Sec 2's cumulative re-check: fine-tuning continues on
            # the same net across layers, so a new layer validating must
            # not come at the cost of silently forgetting an earlier one.
            # These sets are from already-concluded rounds -- by now the
            # net has trained on many ancestors since, so there is no
            # same-round leakage left to guard against here.
            for old_held_out in validated_held_out_sets:
                if not layer_passes_validation(net, baseline_net, old_held_out):
                    passed = False
                    break

        if passed:
            validated_held_out_sets.extend(round_held_out_sets)
            consecutive_failures = 0
            back += DISTANCE_STEP
            continue

        consecutive_failures += 1
        if consecutive_failures > max_uninformative_layers:
            outcome = "informative_failure"
            break
        back += DISTANCE_STEP

    if outcome != "informative_failure" and consecutive_failures > 0:
        # The loop stopped on exhaustion (not the failure threshold), but a
        # failure streak was still open at the time -- this is a genuine
        # failure signal, not a success that merely ran out of material.
        outcome = "informative_failure"

    return {"layers_built": layers_built, "outcome": outcome}


def main():
    parser = argparse.ArgumentParser(
        description="Run the proven-anchor curriculum widening loop "
                    "(spec 2026-10-19-mcts-az-proven-anchor-curriculum-design.md Sec 3.6)."
    )
    parser.add_argument('--out', required=True, type=Path,
                         help="Output directory: frozen reference snapshot, "
                              "games/checkpoints dirs, and the run summary are written here.")
    parser.add_argument('--corpus', type=Path, default=CORPUS,
                         help="Opening-job corpus to source anchor games from "
                              "(default: the no-power corpus used by the reference campaign).")
    parser.add_argument('--seconds', type=float, default=30.0,
                         help="Per-proof-search time budget (spec Sec 3.2: not fixed a priori, "
                              "tune empirically for the hardware at hand).")
    parser.add_argument('--target-distance', type=int, default=DISTANCE_STEP,
                         help="Ply distance to mate an anchor game's qualifying position must "
                              "have (spec Sec 3.1: exactly 6 for the first layer).")
    parser.add_argument('--minimum-anchors', type=int, default=2,
                         help="Minimum number of anchor games to find before starting "
                              "(spec Sec 2's precondition: at least 2).")
    parser.add_argument('--max-uninformative-layers', type=int, default=3)
    args = parser.parse_args()

    from santoni_az.trainer import Trainer  # local: avoids importing torch for --help

    args.out.mkdir(parents=True, exist_ok=True)
    reference_binary, reference_manifest = freeze_reference(args.out)
    print(f"REFERENCE frozen: {reference_binary} sha256={reference_manifest['engine_sha256']}", flush=True)

    openings = json.loads(args.corpus.read_text())['jobs']
    anchor_games = find_anchor_games(
        openings, args.target_distance, args.seconds, reference_binary, minimum=args.minimum_anchors
    )
    print(f"Found {len(anchor_games)} anchor game(s) at ply distance {args.target_distance}.", flush=True)

    games_dir = args.out / 'games'
    checkpoint_dir = args.out / 'checkpoints'
    games_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)
    # Spec Sec 4: a fresh network, deliberately never a loaded checkpoint --
    # `Trainer`'s own `net_factory` default (`PolicyValueNet()`) already
    # gives us that, so nothing further to wire here.
    trainer = Trainer(games_dir=games_dir, checkpoint_dir=checkpoint_dir)

    summary = run_curriculum(
        anchor_games, reference_binary, args.seconds, trainer,
        max_uninformative_layers=args.max_uninformative_layers,
    )
    summary['checkpoint'] = trainer.save_checkpoint()
    (args.out / 'curriculum_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()

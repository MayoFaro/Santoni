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

Module-level imports, not local ones: `build_layer`, `examples_from_layer`,
`child_examples_from_layer` and `measure_layer_validation` are imported by
name here specifically so tests can `monkeypatch.setattr(this_module,
"build_layer", ...)` etc. and have that actually intercept
`run_curriculum`'s behaviour. Importing them inside the function body, or
calling them via a dotted path (`tools.proven_anchor.layer.build_layer(...)`),
would both defeat that monkeypatch -- `run_curriculum` must call these as
bare module-level names.

**Label convention.** Every `(position, label)` pair this module hands to
`tools.proven_anchor.validation` means: `True` = *this position's own
player to move is winning*. A `LayerResult` child's verdict is instead "a
proven win for the ancestor's mover", and a child's `player` is that
mover's opponent, so `_children_held_out` negates it; a trained ancestor's
verdict is already in its own mover's perspective and board symmetries
preserve `player`, so `_symmetrized_ancestor_held_out` keeps it as is. See
`tools.proven_anchor.validation`'s module docstring for why that
perspective (and not any fixed side) is the network's own.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from tools.proven_anchor.anchor_games import find_anchor_games
from tools.proven_anchor.layer import build_layer
from tools.proven_anchor.symmetry import BOARD_SYMMETRIES
from tools.proven_anchor.training_examples import child_examples_from_layer, examples_from_layer
from tools.proven_anchor.validation import NoHeldOutEvidence, measure_layer_validation
from tools.mcts_az_vs_reference_campaign import CORPUS, freeze_reference, set_engine

DISTANCE_STEP = 6

#: Gradient steps spent on the current distance before each validation
#: attempt. One step per layer per round (the original value) could never
#: clear a statistical bar on a freshly-initialized network, which made the
#: gate structurally unclearable; this is the per-attempt training budget
#: spec Sec 3.4's "gate obligatoire avant d'elargir" implies. Tunable from
#: the CLI -- the right value is hardware- and layer-dependent.
TRAIN_STEPS_PER_ATTEMPT = 200


def _active_ancestors(anchor_games, back):
    """The position `back` plies before each anchor game's own qualifying
    position, or `None` for a game whose own history runs out at this
    distance -- a normal, reportable stop (spec Sec 3.2), not an error."""
    ancestors = []
    for positions, qualifying_ply in anchor_games:
        index = qualifying_ply - back
        ancestors.append(positions[index] if index >= 0 else None)
    return ancestors


def _partition_children(layer):
    """Split a layer's usable children into `(for_training, held_out)`.

    Both halves are proven (`verdict is not None` -- an unresolved child is
    never silently read as a loss, plan Review Focus) and non-terminal: a
    child that already has a winner is trivially classifiable from its own
    planes, so it would inflate the held-out sign-match rate without
    testing any learned pattern, and the network is never asked to evaluate
    a finished position at inference time either.

    The split exists because these children are now *both* training signal
    (the only source of negative value labels, see
    `child_examples_from_layer`) and the held-out anti-memorization
    evidence -- and a position cannot honestly be both. Every other child
    in `native_turns` order goes to each role, so no child is ever
    simultaneously trained on and used as evidence; the caller's own
    `trained_positions` filter then catches the remaining case of the
    *same* position turning up in another anchor game's validation half.
    Judgment call: a 50/50 stride split keeps both roles equally supplied;
    spec Sec 2's answer to a held-out pool that is too small for a usable
    CI is more anchor games (Sec 3.7), not a thinner validation half."""
    usable = [
        (turn, verdict)
        for turn, verdict in layer.children
        if verdict is not None and turn.after.winner is None
    ]
    return usable[0::2], usable[1::2]


def _children_held_out(layer_held_out_children):
    """Held-out `(position, label)` pairs for one layer's validation half.

    The label is the *negation* of the child's verdict: a verdict is "a
    proven win for the ancestor's mover", while the label convention (see
    the module docstring) is "this position's own player to move is
    winning" -- and a child's own player is the ancestor's mover's
    opponent. Emitting the verdict unflipped here would ask the network for
    a positive value on a position whose mover is losing, i.e. measure it
    against ground truth inverted relative to the value head's own
    training convention, and would put this pool in direct contradiction
    with `_symmetrized_ancestor_held_out`'s."""
    return [(turn.after, not verdict) for turn, verdict in layer_held_out_children]


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
    verbatim, which is not held out at all.

    The verdict passes through unflipped: a symmetry preserves `player`, so
    `ancestor_verdict` ("the ancestor's own mover has a proven win")
    already *is* the label convention's "this position's own player to move
    is winning" for every transformed image."""
    if layer.ancestor_verdict is not True or layer.ancestor is None:
        return []
    held_out = []
    for symmetry in BOARD_SYMMETRIES:
        transformed = symmetry(layer.ancestor)
        if transformed != layer.ancestor:
            held_out.append((transformed, layer.ancestor_verdict))
    return held_out


def _progress(message):
    print(message, flush=True)


def _format_measurement(index, measurement):
    return (
        f"game{index} rate={measurement.rate:.3f} "
        f"CI95=[{measurement.ci_lo:.3f},{measurement.ci_hi:.3f}] "
        f"base={measurement.baseline_rate:.3f} "
        f"maj={measurement.majority_baseline_rate:.3f} "
        f"n={measurement.n} (+{measurement.true_labels}/-{measurement.false_labels})"
    )


def _format_round(record):
    resolved = ','.join(
        'n/a' if fraction is None else f'{fraction:.2f}'
        for fraction in record['resolved_fractions']
    )
    parts = [
        f"[distance={record['distance']} attempt={record['attempt']}]",
        f"resolved={resolved}",
    ]
    parts.extend(record['games_formatted'])
    if record['regression_at_distance'] is not None:
        parts.append(f"REGRESSION at distance={record['regression_at_distance']}")
    parts.append('PASS' if record['passed'] else 'FAIL')
    return ' | '.join(parts)


def run_curriculum(anchor_games, reference_binary, seconds, trainer,
                   max_uninformative_layers=3,
                   train_steps_per_attempt=TRAIN_STEPS_PER_ATTEMPT,
                   progress=_progress):
    """Drive the layer-by-layer widening loop over `anchor_games` (each a
    `(positions, qualifying_ply)` pair, the exact shape
    `tools.proven_anchor.anchor_games.find_anchor_games` returns).

    One *distance* is attempted until it validates or gives up: each
    attempt spends `train_steps_per_attempt` gradient steps on the exact
    examples built at this distance and then re-runs the gate. The
    curriculum only widens (`back += DISTANCE_STEP`) once the gate actually
    passes -- spec Sec 3.4's gate is "obligatoire avant d'elargir" and Sec
    3.6 widens "tant que 3.4 valide", so a failure means more training at
    the *same* distance, never a harder one.
    `max_uninformative_layers` therefore bounds the number of *failed
    attempts at one distance* (each having spent a full training budget)
    before the run is declared an informative failure.

    Returns a summary dict:
    - `layers_built`: total `build_layer` calls made across every anchor
      game and distance (preserved even when the run ends in an error).
    - `resolved_fractions`: per distance, the per-game resolved fraction of
      the ~33 proof searches at that distance (spec Sec 3.2's diagnostic --
      a sharp drop *is* the empirical ceiling of the exact core).
    - `rounds`: one record per validation attempt, carrying the numbers
      behind the pass/fail bit (sign-match rate, bootstrap CI, baseline
      rate, label balance, majority-class baseline) -- spec Sec 2's actual
      deliverable, to be reported as measured.
    - `outcome`, one of:
      `"stopped_success_exhausted"` (every attempted distance validated;
      the loop only stopped because the anchor games' own history ran out),
      `"informative_failure"` (spec Sec 2's "echec informatif": a distance
      never cleared baseline within its attempt budget),
      `"stopped_regression"` (a *previously validated* distance regressed
      -- catastrophic forgetting, which spec Sec 2's cumulative re-check
      exists to catch, reported distinctly from a fresh distance simply
      failing to clear the bar),
      `"stopped_unresolvable"` (held-out evidence ran out: the reference
      engine proves nothing at this distance any more, the expected
      terminal condition of spec Sec 3.2's natural depth ceiling),
      or `"error"` (an unexpected exception; the real-engine calls inside
      `build_layer` are the only part of this loop that can genuinely fail
      at runtime, and a CLI run should report that cleanly rather than
      crash).
    """
    if len(anchor_games) < 2:
        raise ValueError(
            "run_curriculum needs at least 2 anchor games to pool a "
            "statistically usable held-out validation set at every "
            "distance (spec Sec 2's precondition); got "
            f"{len(anchor_games)}."
        )

    # Spec/plan global constraint: proof search must run against the frozen,
    # sha256-pinned reference binary, not "whatever is built right now".
    # `native_search` resolves through the process-global
    # `santorini.native.library()`, so the pin has to be applied here
    # explicitly -- relying on an earlier `reference_move` call having
    # mutated that global as a side effect only works on the CLI path and
    # silently uses the wrong engine on any other.
    set_engine(reference_binary)

    summary = {
        "layers_built": 0,
        "outcome": "stopped_success_exhausted",
        "resolved_fractions": [],
        "rounds": [],
    }
    try:
        _run_curriculum_loop(summary, anchor_games, seconds, trainer,
                             max_uninformative_layers, train_steps_per_attempt, progress)
    except NoHeldOutEvidence as exc:
        # Expected terminal condition, not a failure of the method: report
        # it as its own outcome, keeping everything measured so far.
        summary["outcome"] = "stopped_unresolvable"
        summary["detail"] = str(exc)
    except Exception as exc:  # pragma: no cover - exercised only by real engine failures
        summary["outcome"] = "error"
        summary["error"] = str(exc)
    return summary


def _run_curriculum_loop(summary, anchor_games, seconds, trainer, max_uninformative_layers,
                         train_steps_per_attempt, progress):
    # A frozen, eval-mode deepcopy: the baseline must stay the same
    # pre-training reference for every round, and must never be mutated by
    # being measured (BatchNorm running stats update on every train-mode
    # forward pass, `torch.no_grad()` notwithstanding).
    baseline_net = copy.deepcopy(trainer.net).eval()

    # Spec Sec 2 / plan Review Focus: a position the network has trained on
    # -- at any distance, from any anchor game -- can never serve as
    # held-out evidence, even if a second, near-identical anchor game
    # reaches it under another name.
    trained_positions = set()
    validated_held_out_sets = []  # (distance, held_out) for the cumulative re-check
    back = 0

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
        summary["layers_built"] += len(layer_results)
        resolved_fractions = [layer.resolved_fraction for layer in layer_results]
        summary["resolved_fractions"].append(
            {"distance": back, "per_game": resolved_fractions}
        )

        # Spec Sec 3.3: the exact labels of this distance go into the
        # trainer once; the retry loop below spends more gradient steps on
        # them rather than re-adding duplicates to the buffer.
        partitions = [_partition_children(layer) for layer in layer_results]
        for layer, (for_training, _held_out) in zip(layer_results, partitions):
            trainer.add_exact_examples(examples_from_layer(layer))
            trainer.add_exact_examples(child_examples_from_layer(layer, children=for_training))
            if layer.ancestor_verdict is True:
                trained_positions.add(layer.ancestor)
            trained_positions.update(turn.after for turn, _ in for_training)

        children_pools = [
            [
                pair for pair in _children_held_out(held_out)
                if pair[0] not in trained_positions
            ]
            for _for_training, held_out in partitions
        ]
        symmetrized_pools = [
            [
                pair for pair in _symmetrized_ancestor_held_out(layer)
                if pair[0] not in trained_positions
            ]
            for layer in layer_results
        ]

        attempt = 0
        while True:
            attempt += 1
            for _ in range(train_steps_per_attempt):
                trainer.train_step()

            # Validate a frozen eval-mode snapshot, never the live network:
            # merely measuring `trainer.net` in train mode would rewrite its
            # BatchNorm running statistics (and so poison every checkpoint
            # saved afterwards) as a pure side effect of checking it.
            net = copy.deepcopy(trainer.net).eval()

            # Spec Sec 2 / brief: a game's own children must never validate
            # its own training, even partially -- every active game trains
            # this round, so a single pooled held-out set drawn from *all*
            # of them would let each game's children sit beside that same
            # game's own just-trained ancestor (one ply away, same tactical
            # line), diluting exactly the independence the gate exists to
            # check. Leave-one-out per game instead: game i's held-out
            # evidence is every *other* active game's validation-half
            # children this round, plus game i's own trained ancestor's
            # symmetric images -- and every per-game check must
            # independently clear baseline for the round as a whole to pass.
            round_held_out_sets = []
            measurements = []
            for i in range(len(layer_results)):
                other_children = [
                    item for j, pool in enumerate(children_pools) if j != i for item in pool
                ]
                held_out_i = other_children + symmetrized_pools[i]
                round_held_out_sets.append(held_out_i)
                measurements.append(measure_layer_validation(net, baseline_net, held_out_i))
            passed = all(measurement.passed for measurement in measurements)

            regression_at_distance = None
            if passed:
                # Spec Sec 2's cumulative re-check: fine-tuning continues on
                # the same net across layers, so a new layer validating must
                # not come at the cost of silently forgetting an earlier
                # one. These sets are from already-concluded rounds -- by
                # now the net has trained on many ancestors since, so there
                # is no same-round leakage left to guard against here. A
                # failure here is catastrophic forgetting, a different
                # diagnosis from a fresh distance failing to clear the bar,
                # so it is recorded (and reported) as its own thing.
                for old_distance, old_held_out in validated_held_out_sets:
                    if not measure_layer_validation(net, baseline_net, old_held_out).passed:
                        passed = False
                        regression_at_distance = old_distance
                        break

            record = {
                "distance": back,
                "attempt": attempt,
                "resolved_fractions": resolved_fractions,
                "games": [measurement.as_dict() for measurement in measurements],
                "passed": passed,
                "regression_at_distance": regression_at_distance,
            }
            summary["rounds"].append(record)
            if progress is not None:
                progress(_format_round({
                    **record,
                    "games_formatted": [
                        _format_measurement(i, measurement)
                        for i, measurement in enumerate(measurements)
                    ],
                }))

            if passed:
                validated_held_out_sets.extend(
                    (back, held_out) for held_out in round_held_out_sets
                )
                back += DISTANCE_STEP
                break

            if attempt > max_uninformative_layers:
                summary["outcome"] = (
                    "stopped_regression" if regression_at_distance is not None
                    else "informative_failure"
                )
                if regression_at_distance is not None:
                    summary["regression_at_distance"] = regression_at_distance
                return


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
    parser.add_argument('--max-uninformative-layers', type=int, default=3,
                         help="Failed validation attempts tolerated at a single distance "
                              "(each having spent a full training budget) before the run is "
                              "declared an informative failure (spec Sec 2).")
    parser.add_argument('--train-steps-per-attempt', type=int, default=TRAIN_STEPS_PER_ATTEMPT,
                         help="Gradient steps spent on the current distance's exact examples "
                              "before each validation attempt.")
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
        train_steps_per_attempt=args.train_steps_per_attempt,
    )
    summary['checkpoint'] = trainer.save_checkpoint()
    (args.out / 'curriculum_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()

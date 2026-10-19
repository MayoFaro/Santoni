"""Anti-memorization validation gate for a trained proven-anchor curriculum
layer.

A trained layer is only trusted -- and only then does the curriculum widen
one half-move further back (Task 8) -- if its value head's sign agrees with
the reference engine's own *proven* win/loss verdicts on positions held out
of training, at a rate that clears the baseline (pre-training) net's own
rate on that exact same held-out set, with statistical rigor (bootstrap
95% CI lower bound, no scipy -- not a project dependency).

**Label convention (binding for every `(position, label)` pair handled by
this module).** A label means: `True` = *this position's own player to move
is winning*; `False` = this position's own player to move is losing. The
label is always read from the perspective of `position.player`, never from
any externally fixed side, because that is exactly the convention the
network's value head is trained on: `santoni_az.trainer.Trainer._decode_position`
projects a game outcome onto each ply with `value_target = outcome if ply %
2 == 0 else -outcome`, i.e. +1 always means "good for whoever is to move in
the position these planes encode". `tools.proven_anchor.planes.compute_planes`
encodes a position from that same player-to-move perspective.

Consequence for callers building held-out pools out of a
`tools.proven_anchor.layer.LayerResult`: a child's verdict is "a proven win
for the *ancestor's* mover", and the child's own `player` is that mover's
opponent, so the label for a child position is the *negation* of its
verdict. A trained ancestor's verdict, by contrast, is already expressed
from its own mover's perspective and so is its own label -- as are the
board symmetries of it, which preserve `player`. Mixing the two
conventions in one pool silently measures the net against contradictory
ground truth; see `tools.proven_anchor_curriculum._children_held_out` /
`_symmetrized_ancestor_held_out` for the single place that conversion
happens.

This is deliberately *not* the self-consistency check that produced the
original self-play collapse signature (see
docs/bilan-mcts-az-no-power-20261019.md section 3.1 and
docs/superpowers/specs/2026-10-19-mcts-az-proven-anchor-curriculum-design.md
section 0): sign-match here is measured against the reference engine's
proof-search verdict (external ground truth), never against the network's
own self-play outcomes.
"""
from __future__ import annotations

import contextlib
import random
from dataclasses import asdict, dataclass

import torch

from santorini.engine import Position
from santoni_az.network import NUM_PLANES
from tools.proven_anchor.planes import compute_planes

#: Positions are evaluated in chunks rather than one by one: identical
#: results under `eval()` (BatchNorm uses its running statistics, not the
#: batch's), far fewer forward calls.
FORWARD_CHUNK = 256


class NoHeldOutEvidence(ValueError):
    """Raised when a validation call is handed an empty held-out set.

    Expected, not exceptional: once the reference engine's proof search
    stops resolving anything at a given distance (spec section 3.2's
    natural depth ceiling), every pool legitimately comes up empty. A
    specific, catchable error lets the orchestrator report that as its own
    clean outcome instead of dividing by zero or drowning in a blanket
    `except Exception`."""


@dataclass(frozen=True)
class ValidationMeasurement:
    """The numbers behind the pass/fail bit (spec section 2's actual
    deliverable, "a rapporter tel quel")."""

    rate: float
    ci_lo: float
    ci_hi: float
    baseline_rate: float
    n: int
    true_labels: int
    false_labels: int
    majority_baseline_rate: float
    passed: bool

    def as_dict(self) -> dict:
        return asdict(self)


def _planes_tensor(position: Position) -> torch.Tensor:
    raw = compute_planes(position)
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(
        1, NUM_PLANES, 5, 5
    ).float()


@contextlib.contextmanager
def _eval_mode(net):
    """Force `net` into eval mode for the duration, then restore whatever
    mode it was in.

    `torch.no_grad()` alone is *not* enough: `PolicyValueNet` is built out
    of `BatchNorm2d` layers, whose running mean/variance are updated on
    every forward pass in train mode regardless of gradient tracking. A
    validation call left in train mode would therefore (a) measure
    batch-statistics behaviour rather than inference behaviour and (b)
    mutate the very network it is only supposed to be observing. Restoring
    the original mode afterwards keeps this transparent to callers that
    legitimately want train mode for something else."""
    was_training = net.training
    net.eval()
    try:
        yield
    finally:
        net.train(was_training)


def _hits(net, labeled_positions: list[tuple[Position, bool]]) -> list[bool]:
    if not labeled_positions:
        raise NoHeldOutEvidence(
            "no held-out evidence: sign-match cannot be measured on an "
            "empty labeled-position set"
        )
    hits: list[bool] = []
    with torch.no_grad(), _eval_mode(net):
        for start in range(0, len(labeled_positions), FORWARD_CHUNK):
            chunk = labeled_positions[start:start + FORWARD_CHUNK]
            planes = torch.cat([_planes_tensor(position) for position, _ in chunk])
            _, values = net(planes)
            for (_, verdict), value in zip(chunk, values.reshape(-1).tolist()):
                hits.append((value > 0) == verdict)
    return hits


def sign_match_rate(net, labeled_positions: list[tuple[Position, bool]]) -> float:
    """Fraction of `labeled_positions` where `net`'s value-head sign agrees
    with the proven verdict (True = this position's own player to move is
    winning -- see the module docstring's label convention)."""
    hits = _hits(net, labeled_positions)
    return sum(hits) / len(hits)


def bootstrap_ci(
    hits: list[bool], resamples: int = 5000, seed: int = 20261019
) -> tuple[float, float]:
    """95% bootstrap CI on the hit rate, by resampling `hits` with
    replacement. Same percentile-index convention as
    `tools.mcts_az_vs_reference_campaign.bootstrap`: sort `resamples`
    resampled rates ascending and take indices
    `int(0.025 * resamples)` / `int(0.975 * resamples) - 1`."""
    if not hits:
        raise NoHeldOutEvidence(
            "no held-out evidence: a bootstrap CI cannot be resampled from "
            "an empty hit list"
        )
    rng = random.Random(seed)
    n = len(hits)
    samples = []
    for _ in range(resamples):
        resampled = [rng.choice(hits) for _ in range(n)]
        samples.append(sum(resampled) / n)
    samples.sort()
    return samples[int(0.025 * resamples)], samples[int(0.975 * resamples) - 1]


def measure_layer_validation(
    net, baseline_net, held_out: list[tuple[Position, bool]]
) -> ValidationMeasurement:
    """Measure `net` against `baseline_net` on `held_out` and report every
    number behind the verdict, not just the verdict.

    `passed` is True iff the bootstrap CI lower bound on `net`'s sign-match
    rate over `held_out` exceeds `baseline_net`'s own sign-match rate on
    that exact same held-out set. `majority_baseline_rate` is what a
    constant-output network would score on this pool (the share of the
    dominant label) -- reported so a degenerate net clearing the gate is
    visibly explainable rather than silently accepted.

    Raises `NoHeldOutEvidence` if `held_out` is empty."""
    hits = _hits(net, held_out)
    rate = sum(hits) / len(hits)
    lo, hi = bootstrap_ci(hits)
    baseline_rate = sign_match_rate(baseline_net, held_out)
    true_labels = sum(1 for _, verdict in held_out if verdict)
    false_labels = len(held_out) - true_labels
    return ValidationMeasurement(
        rate=rate,
        ci_lo=lo,
        ci_hi=hi,
        baseline_rate=baseline_rate,
        n=len(held_out),
        true_labels=true_labels,
        false_labels=false_labels,
        majority_baseline_rate=max(true_labels, false_labels) / len(held_out),
        passed=lo > baseline_rate,
    )


def layer_passes_validation(net, baseline_net, held_out: list[tuple[Position, bool]]) -> bool:
    """True iff the bootstrap CI lower bound on `net`'s sign-match rate over
    `held_out` exceeds `baseline_net`'s own sign-match rate on that exact
    same held-out set. Thin wrapper over `measure_layer_validation` for
    callers that only want the bit; the orchestrator wants the numbers."""
    return measure_layer_validation(net, baseline_net, held_out).passed

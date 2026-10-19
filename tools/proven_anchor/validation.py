"""Anti-memorization validation gate for a trained proven-anchor curriculum
layer.

A trained layer is only trusted -- and only then does the curriculum widen
one half-move further back (Task 8) -- if its value head's sign agrees with
the reference engine's own *proven* win/loss verdicts on positions held out
of training, at a rate that clears the baseline (pre-training) net's own
rate on that exact same held-out set, with statistical rigor (bootstrap
95% CI lower bound, no scipy -- not a project dependency).

This is deliberately *not* the self-consistency check that produced the
original self-play collapse signature (see
docs/bilan-mcts-az-no-power-20261019.md section 3.1 and
docs/superpowers/specs/2026-10-19-mcts-az-proven-anchor-curriculum-design.md
section 0): sign-match here is measured against the reference engine's
proof-search verdict (external ground truth), never against the network's
own self-play outcomes.
"""
from __future__ import annotations

import random

import torch

from santorini.engine import Position
from santoni_az.network import NUM_PLANES
from tools.proven_anchor.planes import compute_planes


def _planes_tensor(position: Position) -> torch.Tensor:
    raw = compute_planes(position)
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(
        1, NUM_PLANES, 5, 5
    ).float()


def _hits(net, labeled_positions: list[tuple[Position, bool]]) -> list[bool]:
    hits = []
    with torch.no_grad():
        for position, verdict in labeled_positions:
            planes = _planes_tensor(position)
            _, value = net(planes)
            hits.append((value.item() > 0) == verdict)
    return hits


def sign_match_rate(net, labeled_positions: list[tuple[Position, bool]]) -> float:
    """Fraction of `labeled_positions` where `net`'s value-head sign agrees
    with the proven verdict (True = winning for the player to move)."""
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
    rng = random.Random(seed)
    n = len(hits)
    samples = []
    for _ in range(resamples):
        resampled = [rng.choice(hits) for _ in range(n)]
        samples.append(sum(resampled) / n)
    samples.sort()
    return samples[int(0.025 * resamples)], samples[int(0.975 * resamples) - 1]


def layer_passes_validation(net, baseline_net, held_out: list[tuple[Position, bool]]) -> bool:
    """True iff the bootstrap CI lower bound on `net`'s sign-match rate over
    `held_out` exceeds `baseline_net`'s own sign-match rate on that exact
    same held-out set."""
    hits = _hits(net, held_out)
    lo, _ = bootstrap_ci(hits)
    baseline_rate = sign_match_rate(baseline_net, held_out)
    return lo > baseline_rate

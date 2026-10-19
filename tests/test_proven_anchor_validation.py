import copy

import pytest
import torch
from torch import nn

from tools.proven_anchor.planes import compute_planes
from tools.proven_anchor.validation import (
    NoHeldOutEvidence,
    bootstrap_ci,
    layer_passes_validation,
    measure_layer_validation,
    sign_match_rate,
)
from santoni_az.network import PolicyValueNet
from santorini.engine import Position


def _four_distinct_positions():
    """Four positions whose encoded planes genuinely differ, so a net keyed
    on plane *content* (below) can tell which position it was handed."""
    positions = [
        Position(workers=((0, 4), (20, 24))),
        Position(workers=((1, 4), (20, 24)), player=1),
        Position(workers=((0, 4), (20, 24)), heights=(1,) + (0,) * 24),
        Position(workers=((0, 4), (20, 24)), heights=(0, 2) + (0,) * 23),
    ]
    signatures = {_signature_of(position) for position in positions}
    assert len(signatures) == 4, 'fixture positions must encode to distinct planes'
    return positions


def _signature_of(position):
    return tuple(compute_planes(position))


def _row_signature(planes_row):
    return tuple(int(v) for v in planes_row.flatten().tolist())


class PositionKeyedNet(nn.Module):
    """A fake net whose value output depends on the *content* of the planes
    it was handed, not on the batch index -- otherwise a sign-match test
    passes vacuously, verifying nothing about the position/label pairing."""

    def __init__(self, values_by_signature):
        super().__init__()
        self.values_by_signature = values_by_signature

    def forward(self, planes):
        n = planes.shape[0]
        values = torch.tensor([
            self.values_by_signature[_row_signature(planes[i])] for i in range(n)
        ]).unsqueeze(1)
        return torch.zeros(n, 1250), values


def test_sign_match_rate_is_exact_arithmetic():
    positions = _four_distinct_positions()
    # +1 on the first and third position, -1 on the second and fourth --
    # keyed on the plane content, so a mis-paired position would show up.
    values = {
        _signature_of(position): 1.0 if i % 2 == 0 else -1.0
        for i, position in enumerate(positions)
    }
    labels = [True, True, False, False]  # matches for i=0 and i=3 only

    rate = sign_match_rate(PositionKeyedNet(values), list(zip(positions, labels)))

    assert rate == pytest.approx(0.5)


def test_sign_match_rate_follows_the_position_not_the_batch_order():
    """Same positions, same labels, reversed order: a net keyed on batch
    index would report a different rate, a correct implementation cannot."""
    positions = _four_distinct_positions()
    values = {
        _signature_of(position): 1.0 if i % 2 == 0 else -1.0
        for i, position in enumerate(positions)
    }
    labeled = list(zip(positions, [True, True, False, False]))

    net = PositionKeyedNet(values)
    assert sign_match_rate(net, labeled) == pytest.approx(sign_match_rate(net, labeled[::-1]))


def test_bootstrap_ci_bounds_are_well_formed():
    hits = [True] * 18 + [False] * 2  # 90% observed
    lo, hi = bootstrap_ci(hits)
    assert 0.0 <= lo <= 0.9 <= hi <= 1.0


def test_layer_passes_validation_requires_clearing_the_baseline():
    untrained = PolicyValueNet()
    untrained.eval()
    positions = [Position(workers=((0, 4), (20, 24))) for _ in range(10)]
    held_out = list(zip(positions, [True] * 10))
    # Comparing the same untrained net against itself must never pass.
    assert layer_passes_validation(untrained, untrained, held_out) is False


def test_an_empty_held_out_set_raises_a_specific_error_everywhere():
    """Running out of held-out evidence is the loop's own expected terminal
    condition (spec Sec 3.2's depth ceiling), so it must surface as a
    specific, catchable error instead of a ZeroDivisionError or an
    IndexError swallowed by a blanket handler."""
    net = PolicyValueNet().eval()
    with pytest.raises(NoHeldOutEvidence):
        sign_match_rate(net, [])
    with pytest.raises(NoHeldOutEvidence):
        bootstrap_ci([])
    with pytest.raises(NoHeldOutEvidence):
        measure_layer_validation(net, net, [])


def test_measuring_a_net_in_train_mode_never_mutates_its_batchnorm_stats():
    """`torch.no_grad()` does not suppress BatchNorm's running-statistics
    update -- only eval mode does. Validation must therefore be inert on
    the net it measures, whatever mode the caller left it in."""
    net = PolicyValueNet()
    net.train()
    baseline = PolicyValueNet().eval()
    positions = [Position(workers=((0, 4), (20, 24))),
                 Position(workers=((1, 4), (20, 24)), player=1)]
    held_out = list(zip(positions, [True, False]))

    before = {name: buffer.clone() for name, buffer in net.named_buffers()}
    measure_layer_validation(net, baseline, held_out)
    after = {name: buffer.clone() for name, buffer in net.named_buffers()}

    assert before.keys() == after.keys()
    assert any('running_mean' in name for name in before)  # the fixture is meaningful
    for name in before:
        assert torch.equal(before[name], after[name]), f'{name} was mutated by validation'
    # ... and the caller's own train/eval mode is restored, not silently changed.
    assert net.training is True


def test_measure_layer_validation_reports_the_numbers_behind_the_verdict():
    positions = _four_distinct_positions()
    values = {_signature_of(position): 1.0 for position in positions}
    net = PositionKeyedNet(values)        # constant +1 output
    baseline = PositionKeyedNet(values)   # same, so nothing can clear it
    held_out = list(zip(positions, [True, True, True, False]))

    measurement = measure_layer_validation(net, baseline, held_out)

    assert measurement.n == 4
    assert measurement.true_labels == 3
    assert measurement.false_labels == 1
    assert measurement.rate == pytest.approx(0.75)
    assert measurement.baseline_rate == pytest.approx(0.75)
    # A constant +1 net scores exactly the majority-class share: the number
    # that makes a degenerate pass visible rather than silently accepted.
    assert measurement.majority_baseline_rate == pytest.approx(0.75)
    assert 0.0 <= measurement.ci_lo <= measurement.rate <= measurement.ci_hi <= 1.0
    assert measurement.passed is False
    assert measurement.as_dict()['rate'] == pytest.approx(0.75)


def test_a_frozen_baseline_deepcopy_stays_frozen_across_measurements():
    """The orchestrator reuses one baseline net for the whole run; being
    measured must not drift it."""
    baseline = copy.deepcopy(PolicyValueNet()).eval()
    net = PolicyValueNet().eval()
    positions = [Position(workers=((0, 4), (20, 24))),
                 Position(workers=((1, 4), (20, 24)), player=1)]
    held_out = list(zip(positions, [True, False]))

    before = {name: buffer.clone() for name, buffer in baseline.named_buffers()}
    for _ in range(3):
        measure_layer_validation(net, baseline, held_out)

    for name, buffer in baseline.named_buffers():
        assert torch.equal(before[name], buffer)
    assert baseline.training is False

import pytest
import torch

from tools.proven_anchor.validation import sign_match_rate, bootstrap_ci, layer_passes_validation
from santoni_az.network import PolicyValueNet
from santorini.engine import Position


def test_sign_match_rate_is_exact_arithmetic():
    class FakeNet:
        def __call__(self, planes):
            # value = +1 for even batch index, -1 for odd, policy irrelevant here
            n = planes.shape[0]
            values = torch.tensor([1.0 if i % 2 == 0 else -1.0 for i in range(n)]).unsqueeze(1)
            return torch.zeros(n, 1250), values

    positions = [Position(workers=((0, 4), (20, 24))) for _ in range(4)]
    labels = [True, True, False, False]  # matches for i=0,3 only under the fake net above
    rate = sign_match_rate(FakeNet(), list(zip(positions, labels)))
    assert rate == pytest.approx(0.5)


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

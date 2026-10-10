import struct

from santoni_az.inference_protocol import (
    pack_request, unpack_request, pack_response, unpack_response, PLANE_BYTES, ACTION_SPACE,
)


def test_request_roundtrip():
    batch = [bytes([1, 0] * (PLANE_BYTES // 2)), bytes([0, 1] * (PLANE_BYTES // 2))]
    packed = pack_request(batch)
    unpacked = unpack_request(packed)
    assert unpacked == batch


def test_response_roundtrip():
    policies = [[0.001] * ACTION_SPACE, [0.002] * ACTION_SPACE]
    values = [0.5, -0.3]
    packed = pack_response(policies, values)
    out_policies, out_values = unpack_response(packed, batch_size=2)
    assert out_values == [pytest_approx(0.5), pytest_approx(-0.3)]
    assert len(out_policies) == 2 and len(out_policies[0]) == ACTION_SPACE


def pytest_approx(x):
    import pytest
    return pytest.approx(x, abs=1e-5)

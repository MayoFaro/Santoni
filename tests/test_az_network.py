import torch

from santoni_az.network import PolicyValueNet, ACTION_SPACE, NUM_PLANES


def test_forward_pass_shapes_and_value_range():
    net = PolicyValueNet(channels=32, blocks=4)
    batch = torch.zeros(3, NUM_PLANES, 5, 5)
    policy, value = net(batch)
    assert policy.shape == (3, ACTION_SPACE)
    assert value.shape == (3, 1)
    assert torch.all(value >= -1) and torch.all(value <= 1)

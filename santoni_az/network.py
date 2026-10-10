import torch
import torch.nn as nn

NUM_PLANES = 6
ACTION_SPACE = 1250


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return torch.relu(x + residual)


class PolicyValueNet(nn.Module):
    def __init__(self, channels=32, blocks=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(NUM_PLANES, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.tower = nn.Sequential(*[ResidualBlock(channels) for _ in range(blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 8, 1),
            nn.Flatten(),
            nn.Linear(8 * 5 * 5, ACTION_SPACE),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 4, 1),
            nn.Flatten(),
            nn.Linear(4 * 5 * 5, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )

    def forward(self, planes):
        x = self.stem(planes)
        x = self.tower(x)
        return self.policy_head(x), self.value_head(x)

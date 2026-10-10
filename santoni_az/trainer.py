import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from .network import PolicyValueNet, ACTION_SPACE, NUM_PLANES


class Trainer:
    def __init__(self, games_dir, checkpoint_dir, net_factory=lambda: PolicyValueNet()):
        self.games_dir = Path(games_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.net = net_factory()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.generation = 0
        self._consumed_games = set()
        self._examples = []  # (planes_placeholder, policy_target, value_target)

    def consume_new_games(self):
        new_files = sorted(
            f for f in self.games_dir.glob("game-*.json") if f.name not in self._consumed_games
        )
        for f in new_files:
            data = json.loads(f.read_text())
            outcome = data["outcome"]
            for action_idx, visits in data["moves"]:
                policy_target = torch.zeros(ACTION_SPACE)
                policy_target[action_idx] = 1.0  # cible simplifiée : un seul coup joué
                self._examples.append((torch.zeros(NUM_PLANES, 5, 5), policy_target, float(outcome)))
            self._consumed_games.add(f.name)
        return len(new_files)

    def train_step(self, batch_size=64):
        if not self._examples:
            return 0.0
        batch = self._examples[:batch_size]
        planes = torch.stack([p for p, _, _ in batch])
        policy_targets = torch.stack([pt for _, pt, _ in batch])
        value_targets = torch.tensor([vt for _, _, vt in batch]).unsqueeze(1)
        policy_logits, value_pred = self.net(planes)
        policy_loss = F.cross_entropy(policy_logits, policy_targets.argmax(dim=1))
        value_loss = F.mse_loss(value_pred, value_targets)
        loss = policy_loss + value_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def save_checkpoint(self):
        self.generation += 1
        path = self.checkpoint_dir / f"gen-{self.generation:04d}.pt"
        tmp = path.with_suffix(".pt.tmp")
        torch.save({
            "generation": self.generation,
            "model_state": self.net.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "consumed_games": sorted(self._consumed_games),
        }, tmp)
        os.replace(tmp, path)
        return str(path)

    def load_latest_checkpoint(self):
        checkpoints = sorted(self.checkpoint_dir.glob("gen-*.pt"))
        if not checkpoints:
            return 0
        data = torch.load(checkpoints[-1], weights_only=False)
        self.net.load_state_dict(data["model_state"])
        self.optimizer.load_state_dict(data["optimizer_state"])
        self.generation = data["generation"]
        self._consumed_games = set(data["consumed_games"])
        return self.generation

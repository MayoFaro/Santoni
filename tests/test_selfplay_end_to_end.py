import json
import subprocess
import threading
import time
from pathlib import Path

import torch

from santoni_az.network import PolicyValueNet
from santoni_az.inference_server import InferenceServer

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "native_engine" / "target" / "release" / "selfplay_worker"


def test_worker_produces_games_via_real_inference_server(tmp_path):
    socket_path = str(tmp_path / "infer.sock")
    out_dir = tmp_path / "games"
    out_dir.mkdir()
    net = PolicyValueNet(channels=8, blocks=1)
    server = InferenceServer(socket_path, net, flush_interval_s=0.02)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)

    result = subprocess.run(
        [str(BINARY), "--games", "2", "--out-dir", str(out_dir),
         "--simulations-per-move", "8", "--inference-socket", socket_path],
        timeout=60, capture_output=True, text=True,
    )
    server.stop()
    assert result.returncode == 0, result.stderr
    files = sorted(out_dir.glob("game-*.json"))
    assert len(files) == 2
    for f in files:
        data = json.loads(f.read_text())
        assert data["outcome"] in (-1, 1)

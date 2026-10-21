import json
import subprocess
import threading
import time
from pathlib import Path

import torch

from santoni_az.network import PolicyValueNet
from santoni_az.inference_server import InferenceServer
from santorini.engine import validate_setup
from santorini.native import encode

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


def test_vs_reference_mode_produces_games_with_every_ply_recorded(tmp_path):
    socket_path = str(tmp_path / "infer.sock")
    out_dir = tmp_path / "games"
    out_dir.mkdir()
    net = PolicyValueNet(channels=8, blocks=1)
    server = InferenceServer(socket_path, net, flush_interval_s=0.02)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)

    # Simple no-power anchor position, player 1 to move: the candidate (the
    # worker's own MCTS+NN side) is whichever physical player is on the move
    # in the given anchor, here player 1 — the reference alpha-beta engine
    # (Task 3's `reference_turn`) plays player 0 for the rest of the game.
    position = validate_setup(workers=[[0, 4], [20, 24]], powers=[0, 0], player=1)
    state = encode(position)
    anchor_path = tmp_path / "anchor.bin"
    anchor_path.write_bytes(bytes(state))

    result = subprocess.run(
        [str(BINARY), "--vs-reference", "--input-state", str(anchor_path),
         "--out-dir", str(out_dir), "--games", "1", "--simulations-per-move", "8",
         "--inference-socket", socket_path, "--reference-seconds", "0.2"],
        timeout=60, capture_output=True, text=True,
    )
    server.stop()
    assert result.returncode == 0, result.stderr
    data = json.loads(next(out_dir.glob("game-*.json")).read_text())
    assert data["outcome"] in (-1, 1)
    visit_counts = [len(p["visits"]) for p in data["positions"]]
    # Every other ply is the reference's degenerate one-entry record,
    # starting from the second ply (the anchor's player, player 1, moves
    # first == the candidate); a skipped ply here would break
    # Trainer._decode_position's ply-alternation convention silently.
    assert len(visit_counts) >= 2
    for i, count in enumerate(visit_counts):
        if i % 2 == 1:
            assert count == 1 and data["positions"][i]["visits"][0][1] == 1

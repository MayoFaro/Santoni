import json
import subprocess
import signal
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "native_engine" / "target" / "release" / "selfplay_worker"


def test_worker_produces_complete_valid_games(tmp_path):
    out_dir = tmp_path / "games"
    out_dir.mkdir()
    result = subprocess.run(
        [str(BINARY), "--games", "3", "--out-dir", str(out_dir), "--simulations-per-move", "8"],
        timeout=30, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    files = sorted(out_dir.glob("game-*.json"))
    assert len(files) == 3
    for f in files:
        data = json.loads(f.read_text())
        assert data["outcome"] in (-1, 1)
        assert len(data["moves"]) > 0
    assert not list(out_dir.glob("*.tmp"))


def test_worker_leaves_no_partial_file_on_sigint(tmp_path):
    out_dir = tmp_path / "games"
    out_dir.mkdir()
    proc = subprocess.Popen(
        [str(BINARY), "--games", "10000", "--out-dir", str(out_dir), "--simulations-per-move", "4"],
    )
    time.sleep(0.3)
    proc.send_signal(signal.SIGINT)
    proc.wait(timeout=10)
    assert not list(out_dir.glob("*.tmp"))
    for f in out_dir.glob("game-*.json"):
        json.loads(f.read_text())  # ne doit jamais lever

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_rust_written_game_is_valid_json(tmp_path):
    sample = tmp_path / "game-0001.json"
    sample.write_text('{"outcome":1,"moves":[[42,100],[7,88]]}\n')
    data = json.loads(sample.read_text())
    assert data["outcome"] == 1
    assert data["moves"] == [[42, 100], [7, 88]]

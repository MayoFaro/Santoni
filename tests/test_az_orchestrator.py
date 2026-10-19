import json
import time

from santoni_az.network import NUM_PLANES
from santoni_az.orchestrator import Orchestrator

PLACEHOLDER_PLANES = [0] * (NUM_PLANES * 5 * 5)


def test_status_reports_last_training_loss(tmp_path):
    games_dir = tmp_path / "games"
    games_dir.mkdir(parents=True)
    (games_dir / "game-0000.json").write_text(json.dumps({
        "outcome": 1,
        "positions": [{"planes": PLACEHOLDER_PLANES, "visits": [[0, 1]]}],
    }) + "\n")

    orch = Orchestrator(str(tmp_path), worker_count=1)
    assert orch.status()["last_loss"] is None

    orch.start()
    deadline = time.monotonic() + 5.0
    last_loss = None
    while time.monotonic() < deadline:
        last_loss = orch.status()["last_loss"]
        if last_loss is not None:
            break
        time.sleep(0.1)
    orch.stop()

    assert last_loss is not None
    assert last_loss > 0.0


def test_start_stop_leaves_no_partial_files(tmp_path):
    orch = Orchestrator(str(tmp_path), worker_count=1)
    orch.start()
    time.sleep(1.0)
    status_before = orch.status()
    assert status_before["workers_alive"] >= 1
    orch.stop()
    tmp_files = list((tmp_path / "games").glob("*.tmp"))
    assert tmp_files == []


def test_restart_resumes_from_last_checkpoint(tmp_path):
    orch = Orchestrator(str(tmp_path), worker_count=1)
    orch.start()
    time.sleep(1.5)
    orch.stop()
    generation_before = orch.status()["generation"]

    orch2 = Orchestrator(str(tmp_path), worker_count=1)
    orch2.start()
    time.sleep(0.2)
    generation_after = orch2.status()["generation"]
    orch2.stop()
    assert generation_after >= generation_before

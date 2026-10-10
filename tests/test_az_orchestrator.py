import time

from santoni_az.orchestrator import Orchestrator


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

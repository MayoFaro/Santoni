"""Preserve the experimental record when a campaign is resumed on another PC."""
import json
import sys

import pytest

from tools import selfplay


def configure(monkeypatch, tmp_path, *, completed):
    binary = tmp_path/'engine.so'
    binary.write_bytes(b'new computer library')
    directory = tmp_path/'campaign'
    directory.mkdir()
    jobs = selfplay.jobs(20261005)
    previous = {'started_at':'original timestamp', 'seconds':5, 'seed':20261005,
                'jobs':jobs, 'engine_sha256':'original library', 'workers':12}
    manifest = directory/'campaign.json'
    manifest.write_text(json.dumps(previous))
    if completed:
        for job in jobs:
            (directory/f"game-{job['number']:03d}.json").write_text(json.dumps({'status':'finished'}))
    monkeypatch.setattr(selfplay, 'library_path', lambda:binary)
    monkeypatch.setattr(selfplay, 'summary', lambda _: {'completed':100})
    monkeypatch.setattr(sys, 'argv', ['selfplay', '--output', str(directory), '--seconds','5','--workers','1'])
    return manifest


def test_completed_campaign_preserves_original_manifest_on_another_pc(monkeypatch, tmp_path):
    manifest = configure(monkeypatch, tmp_path, completed=True)
    original = manifest.read_bytes()
    def unexpected_pool(**kwargs):
        pytest.fail('Completed campaign must not start a process pool')
    monkeypatch.setattr(selfplay, 'ProcessPoolExecutor', unexpected_pool)
    selfplay.main()
    assert manifest.read_bytes() == original


def test_incomplete_campaign_rejects_different_engine_without_overwrite(monkeypatch, tmp_path):
    manifest = configure(monkeypatch, tmp_path, completed=False)
    original = manifest.read_bytes()
    with pytest.raises(SystemExit) as error:
        selfplay.main()
    assert error.value.code == 2
    assert manifest.read_bytes() == original


def test_existing_campaign_rejects_changed_parameters(monkeypatch, tmp_path):
    manifest = configure(monkeypatch, tmp_path, completed=True)
    original = manifest.read_bytes()
    monkeypatch.setattr(sys, 'argv', ['selfplay', '--output', str(manifest.parent), '--seconds','20','--workers','1'])
    with pytest.raises(SystemExit) as error:
        selfplay.main()
    assert error.value.code == 2
    assert manifest.read_bytes() == original

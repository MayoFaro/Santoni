"""Validate experimental assignment and per-player budgets before long matches."""
import json
from types import SimpleNamespace

import pytest

from tools import time_budget_campaign as campaign


def test_each_original_configuration_is_preserved_with_both_assignments():
    original = json.loads((campaign.ROOT / 'experiments/selfplay-100-5s-20261005/campaign.json').read_text())
    jobs = campaign.paired_jobs(original)
    assert len(jobs) == 200
    assert len({j['number'] for j in jobs}) == 200
    for source, left, right in zip(original['jobs'], jobs[::2], jobs[1::2]):
        for key in source:
            if key != 'number':
                assert left[key] == right[key] == source[key]
        assert left['original_game'] == right['original_game'] == source['number']
        assert left['budgets'] == [35, 5]
        assert right['budgets'] == [5, 35]


@pytest.mark.parametrize('first,long_player,expected', [(0, 0, 35), (0, 1, 5), (1, 0, 5), (1, 1, 35)])
def test_budget_follows_player_not_first_mover(monkeypatch, tmp_path, first, long_player, expected):
    calls = []
    def fake_search(position, seconds, publish):
        calls.append(seconds)
        return SimpleNamespace(turn=None, score=-100000, depth=1, nodes=1,
                               elapsed=0, proven=True, complete_depth=True,
                               status='terminal', stop_reason='terminal', engine='test', variation=())
    monkeypatch.setattr(campaign, 'native_search', fake_search)
    job = {'number': 1, 'workers': [[0, 1], [23, 24]], 'powers': [0, 0],
           'first': first, 'long_player': long_player, 'budgets': [35 if p == long_player else 5 for p in (0, 1)]}
    result = campaign.play(job, job['budgets'], str(tmp_path))
    assert result['status'] == 'finished'
    assert calls == [expected]
    record = json.loads((tmp_path / 'game-001.json').read_text())
    assert record['decisions'][0]['budget_seconds'] == expected


def test_reference_integrity_rejects_a_modified_binary(monkeypatch, tmp_path):
    (tmp_path / 'libsantoni_engine.so').write_bytes(b'modified')
    (tmp_path / 'reference.json').write_text(json.dumps({'binary_sha256': 'expected', 'archive_sha256': 'unused', 'source_sha256': {}}))
    monkeypatch.setattr(campaign, 'REFERENCE', tmp_path)
    with pytest.raises(ValueError, match='integrity mismatch'):
        campaign.verify_reference()

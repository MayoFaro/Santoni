import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from santorini.engine import Position  # noqa: E402
from santoni_az.trainer import Trainer  # noqa: E402
from tools import mcts_az_vs_reference_campaign as campaign  # noqa: E402


def boxed_in_position(stuck_player):
    """Position du sous-jeu sans pouvoir où `stuck_player` n'a plus aucun
    coup légal : ses deux bâtisseurs sont en (12, 13), encerclés de dômes
    sur toutes les autres cases. Même construction que le test Rust
    `terminal_value_reports_loss_for_player_with_no_legal_move`
    (native_engine/src/mcts.rs)."""
    domes = ((1 << 25) - 1) & ~((1 << 12) | (1 << 13))
    workers = ((12, 13), (0, 1)) if stuck_player == 0 else ((0, 1), (12, 13))
    return Position(workers=workers, domes=domes, player=stuck_player, powers=(0, 0))


def test_play_resolves_a_stuck_candidate_as_a_loss_without_invoking_either_engine(tmp_path, monkeypatch):
    stuck = boxed_in_position(stuck_player=0)

    def boom(*args, **kwargs):
        raise AssertionError('must not be called on an already-terminal position')

    monkeypatch.setattr(campaign, 'validate_setup', lambda *a, **k: stuck)
    monkeypatch.setattr(campaign, 'candidate_move', boom)
    monkeypatch.setattr(campaign, 'reference_move', boom)

    job = {'number': 1, 'variant_player': 0,
           'initial': {'workers': [[12, 13], [0, 1]], 'powers': [0, 0], 'first': 0}}
    result = campaign.play(job, tmp_path, 2.0, 'unused-socket', tmp_path, 'unused-reference')

    assert result['status'] == 'finished'
    assert result['variant_won'] is False  # variant_player 0 is the stuck player


def test_play_resolves_a_stuck_reference_as_a_win_for_the_candidate(tmp_path, monkeypatch):
    stuck = boxed_in_position(stuck_player=1)

    def boom(*args, **kwargs):
        raise AssertionError('must not be called on an already-terminal position')

    monkeypatch.setattr(campaign, 'validate_setup', lambda *a, **k: stuck)
    monkeypatch.setattr(campaign, 'candidate_move', boom)
    monkeypatch.setattr(campaign, 'reference_move', boom)

    job = {'number': 2, 'variant_player': 0,
           'initial': {'workers': [[0, 1], [12, 13]], 'powers': [0, 0], 'first': 0}}
    result = campaign.play(job, tmp_path, 2.0, 'unused-socket', tmp_path, 'unused-reference')

    assert result['status'] == 'finished'
    assert result['variant_won'] is True  # variant_player 0 is the candidate; player 1 (reference) is stuck


def make_tiny_checkpoint(tmp_path):
    # `load_checkpoint_net` always constructs a fresh default-sized
    # `PolicyValueNet()` (matching how every real checkpoint in this
    # codebase is produced -- `santoni_az/orchestrator.py` never passes a
    # custom `net_factory` to `Trainer`), so the fixture here must save a
    # default-sized net too: a smaller one (e.g. `channels=8, blocks=1`,
    # used elsewhere in this test suite for speed) has a different
    # architecture and its `state_dict()` cannot load into the default
    # shape `load_checkpoint_net` expects.
    games_dir = tmp_path / 'games'
    games_dir.mkdir()
    checkpoint_dir = tmp_path / 'checkpoints'
    checkpoint_dir.mkdir()
    trainer = Trainer(str(games_dir), str(checkpoint_dir))
    path = Path(trainer.save_checkpoint())
    return trainer, path


def test_load_checkpoint_net_restores_the_exact_saved_weights(tmp_path):
    trainer, path = make_tiny_checkpoint(tmp_path)
    net = campaign.load_checkpoint_net(path)
    saved_state = trainer.net.state_dict()
    for key, value in net.state_dict().items():
        assert torch.equal(value, saved_state[key]), f'parameter {key} does not match the saved checkpoint'


def test_freeze_checkpoint_pins_a_copy_and_is_idempotent(tmp_path):
    _, checkpoint_path = make_tiny_checkpoint(tmp_path)
    out_dir = tmp_path / 'campaign'
    out_dir.mkdir()

    binary, manifest = campaign.freeze_checkpoint(out_dir, checkpoint_path)
    assert binary.exists()
    assert manifest['generation'] == 1

    # Re-running against the same campaign directory must not re-snapshot or
    # recompute: same manifest content back.
    binary2, manifest2 = campaign.freeze_checkpoint(out_dir, checkpoint_path)
    assert binary2 == binary
    assert manifest2 == manifest


def test_freeze_checkpoint_detects_tampering_with_the_frozen_copy(tmp_path):
    _, checkpoint_path = make_tiny_checkpoint(tmp_path)
    out_dir = tmp_path / 'campaign'
    out_dir.mkdir()
    binary, _ = campaign.freeze_checkpoint(out_dir, checkpoint_path)

    binary.write_bytes(b'corrupted')
    with pytest.raises(ValueError):
        campaign.freeze_checkpoint(out_dir, checkpoint_path)


def test_compute_pending_jobs_skips_finished_and_quarantines_incomplete(tmp_path):
    jobs = [{'number': 1, 'variant_player': 0, 'initial': {}},
            {'number': 2, 'variant_player': 1, 'initial': {}},
            {'number': 3, 'variant_player': 0, 'initial': {}}]
    campaign.atomic_json(tmp_path / 'game-0001.json', {'job': jobs[0], 'status': 'finished'})
    campaign.atomic_json(tmp_path / 'game-0002.json', {'job': jobs[1], 'status': 'running'})
    # job 3 has no file yet -- a brand new game.

    pending = campaign.compute_pending_jobs(tmp_path, jobs)

    assert [j['number'] for j in pending] == [2, 3]
    assert not (tmp_path / 'game-0002.json').exists()
    quarantined = list((tmp_path / 'attempts').glob('game-0002*.json'))
    assert len(quarantined) == 1
    assert json.loads(quarantined[0].read_text())['status'] == 'running'
    assert (tmp_path / 'game-0001.json').exists()  # le fichier terminé n'est jamais touché


def test_compute_pending_jobs_rejects_a_job_mismatch(tmp_path):
    original_job = {'number': 1, 'variant_player': 0, 'initial': {'opening': 'a'}}
    changed_job = {'number': 1, 'variant_player': 0, 'initial': {'opening': 'b'}}
    campaign.atomic_json(tmp_path / 'game-0001.json', {'job': original_job, 'status': 'running'})
    with pytest.raises(ValueError):
        campaign.compute_pending_jobs(tmp_path, [changed_job])


def test_check_manifest_matches_writes_once_then_rejects_changed_config(tmp_path):
    manifest = tmp_path / 'campaign.json'
    config = {'seconds': 30.0, 'seed': 1}

    campaign.check_manifest_matches(manifest, config)
    written = json.loads(manifest.read_text())
    assert written['seconds'] == 30.0
    assert 'started_at' in written

    campaign.check_manifest_matches(manifest, config)  # même config : no-op

    with pytest.raises(ValueError):
        campaign.check_manifest_matches(manifest, {**config, 'seconds': 10.0})

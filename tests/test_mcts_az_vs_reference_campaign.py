import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from santorini.engine import Position  # noqa: E402
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
    result = campaign.play(job, tmp_path, 2.0, 'unused-checkpoint', 'unused-socket', tmp_path, 'unused-reference')

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
    result = campaign.play(job, tmp_path, 2.0, 'unused-checkpoint', 'unused-socket', tmp_path, 'unused-reference')

    assert result['status'] == 'finished'
    assert result['variant_won'] is True  # variant_player 0 is the candidate; player 1 (reference) is stuck

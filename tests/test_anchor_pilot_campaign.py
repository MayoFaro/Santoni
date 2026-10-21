import hashlib
import json
import os
import signal

import pytest

from santorini.engine import Position
from santorini.native import CState, decode
from tools.anchor_pilot_campaign import freeze_worker_binary, prepare_anchor


def fake_game(count, first_player=0):
    """A list of `count` distinct positions, player alternating per ply, so
    a wrong index picks a position with the wrong `player` too."""
    return [Position(workers=((i % 24, (i + 1) % 24), (20, 21)), player=(first_player + i) % 2)
            for i in range(count)]


def test_prepare_anchor_writes_the_position_back_plies_before_the_win(tmp_path, monkeypatch):
    import tools.anchor_pilot_campaign as mod

    positions = fake_game(10)

    def fake_play(opening_job, seconds, reference_binary, progress=None):
        return positions

    monkeypatch.setattr(mod, 'play_reference_vs_reference', fake_play)

    opening_job = {'workers': [[0, 1], [20, 21]], 'powers': [0, 0], 'first': 0}
    path = prepare_anchor(tmp_path, opening_job, back_plies=5, seconds=0.1,
                          reference_binary='frozen-ref')

    assert path == tmp_path / 'anchor_state.bin'
    raw = path.read_bytes()
    state = CState.from_buffer_copy(raw)
    decoded = decode(state)
    assert decoded == positions[10 - 1 - 5]


def test_prepare_anchor_is_idempotent_on_a_second_call(tmp_path, monkeypatch):
    import tools.anchor_pilot_campaign as mod

    positions = fake_game(10)
    calls = {'n': 0}

    def fake_play(opening_job, seconds, reference_binary, progress=None):
        calls['n'] += 1
        return positions

    monkeypatch.setattr(mod, 'play_reference_vs_reference', fake_play)

    opening_job = {'workers': [[0, 1], [20, 21]], 'powers': [0, 0], 'first': 0}
    first_path = prepare_anchor(tmp_path, opening_job, back_plies=5, seconds=0.1,
                                reference_binary='frozen-ref')
    assert calls['n'] == 1

    def boom(*args, **kwargs):
        raise AssertionError('must not be called on a second, already-prepared call')

    monkeypatch.setattr(mod, 'play_reference_vs_reference', boom)

    second_path = prepare_anchor(tmp_path, opening_job, back_plies=5, seconds=0.1,
                                 reference_binary='frozen-ref')

    assert second_path == first_path
    assert calls['n'] == 1


def test_prepare_anchor_raises_a_clear_error_when_the_seed_game_is_too_short(tmp_path, monkeypatch):
    import tools.anchor_pilot_campaign as mod

    positions = fake_game(3)

    def fake_play(opening_job, seconds, reference_binary, progress=None):
        return positions

    monkeypatch.setattr(mod, 'play_reference_vs_reference', fake_play)

    opening_job = {'workers': [[0, 1], [20, 21]], 'powers': [0, 0], 'first': 0}
    with pytest.raises(ValueError, match='3 plies'):
        prepare_anchor(tmp_path, opening_job, back_plies=5, seconds=0.1,
                       reference_binary='frozen-ref')

    with pytest.raises(ValueError, match='need at least 6'):
        prepare_anchor(tmp_path, opening_job, back_plies=5, seconds=0.1,
                       reference_binary='frozen-ref')

    assert not (tmp_path / 'anchor_state.bin').exists()


def test_prepare_anchor_succeeds_with_exactly_back_plies_plus_one_positions(tmp_path, monkeypatch):
    """Boundary case: `len(positions) == back_plies + 1` means `index == 0`
    -- the earliest position in the game is itself the anchor, and this
    must succeed rather than be rejected. A regression using `index <= 0`
    instead of `index < 0` would wrongly reject exactly this case."""
    import tools.anchor_pilot_campaign as mod

    positions = fake_game(6)  # back_plies=5 -> index = 6 - 1 - 5 = 0

    def fake_play(opening_job, seconds, reference_binary, progress=None):
        return positions

    monkeypatch.setattr(mod, 'play_reference_vs_reference', fake_play)

    opening_job = {'workers': [[0, 1], [20, 21]], 'powers': [0, 0], 'first': 0}
    path = prepare_anchor(tmp_path, opening_job, back_plies=5, seconds=0.1,
                          reference_binary='frozen-ref')

    raw = path.read_bytes()
    state = CState.from_buffer_copy(raw)
    decoded = decode(state)
    assert decoded == positions[0]


def test_prepare_anchor_raises_with_exactly_back_plies_positions(tmp_path, monkeypatch):
    """One ply short of the boundary above: `len(positions) == back_plies`
    means `index == -1`, which must still raise the same clear
    `ValueError` as the obviously-too-short case."""
    import tools.anchor_pilot_campaign as mod

    positions = fake_game(5)  # back_plies=5 -> index = 5 - 1 - 5 = -1

    def fake_play(opening_job, seconds, reference_binary, progress=None):
        return positions

    monkeypatch.setattr(mod, 'play_reference_vs_reference', fake_play)

    opening_job = {'workers': [[0, 1], [20, 21]], 'powers': [0, 0], 'first': 0}
    with pytest.raises(ValueError, match='5 plies'):
        prepare_anchor(tmp_path, opening_job, back_plies=5, seconds=0.1,
                       reference_binary='frozen-ref')

    assert not (tmp_path / 'anchor_state.bin').exists()


def test_freeze_worker_binary_copies_and_records_sha256(tmp_path, monkeypatch):
    import tools.anchor_pilot_campaign as mod

    source = tmp_path / 'source_worker'
    source.write_bytes(b'fake selfplay_worker bytes, generation 1')
    monkeypatch.setattr(mod, 'WORKER_BINARY', source)

    out_dir = tmp_path / 'out'
    frozen_path, sha256 = freeze_worker_binary(out_dir)

    assert frozen_path.exists()
    assert frozen_path.read_bytes() == source.read_bytes()
    assert sha256 == hashlib.sha256(source.read_bytes()).hexdigest()

    manifest_path = out_dir / mod.WORKER_MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    assert manifest['worker_sha256'] == sha256


def test_freeze_worker_binary_is_idempotent_when_unchanged(tmp_path, monkeypatch):
    import tools.anchor_pilot_campaign as mod

    source = tmp_path / 'source_worker'
    source.write_bytes(b'fake selfplay_worker bytes, generation 1')
    monkeypatch.setattr(mod, 'WORKER_BINARY', source)

    out_dir = tmp_path / 'out'
    first_path, first_sha256 = freeze_worker_binary(out_dir)
    frozen_mtime = first_path.stat().st_mtime_ns

    second_path, second_sha256 = freeze_worker_binary(out_dir)

    assert second_path == first_path
    assert second_sha256 == first_sha256
    # Déjà figé et inchangé : pas de recopie, donc le fichier figé n'est
    # jamais réécrit sur un second appel.
    assert second_path.stat().st_mtime_ns == frozen_mtime


def test_freeze_worker_binary_rejects_a_changed_source_binary(tmp_path, monkeypatch):
    import tools.anchor_pilot_campaign as mod

    source = tmp_path / 'source_worker'
    source.write_bytes(b'fake selfplay_worker bytes, generation 1')
    monkeypatch.setattr(mod, 'WORKER_BINARY', source)

    out_dir = tmp_path / 'out'
    _, frozen_sha256 = freeze_worker_binary(out_dir)

    # Rebuild mid-campaign: the binary at WORKER_BINARY changes on disk.
    source.write_bytes(b'different bytes after a native_engine rebuild')
    new_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()

    with pytest.raises(ValueError) as excinfo:
        freeze_worker_binary(out_dir)

    message = str(excinfo.value)
    assert frozen_sha256 in message
    assert new_sha256 in message


def test_main_wires_prepare_anchor_freeze_and_orchestrator_together(tmp_path, monkeypatch):
    """`main()`'s wiring, with no real multi-second run: `prepare_anchor`,
    `freeze_worker_binary` and `AnchorPilotOrchestrator` are all replaced
    with fakes that record call order/args, and the status-printing loop is
    made to exit on its very first iteration by capturing the real
    `signal.signal` handler registration and invoking the SIGTERM handler
    directly from the faked `time.sleep` -- deterministic, unlike relying
    on a real `os.kill`-delivered signal's timing."""
    import tools.anchor_pilot_campaign as mod

    calls = []

    opening_job = {'workers': [[0, 1], [20, 21]], 'powers': [0, 0], 'first': 0}
    corpus_path = tmp_path / 'corpus.json'
    corpus_path.write_text(json.dumps({'jobs': [opening_job], 'seed': 1}))

    frozen_worker_path = tmp_path / 'frozen_worker'
    frozen_reference_path = tmp_path / 'frozen_reference.so'
    anchor_state_path = tmp_path / 'anchor_state.bin'

    def fake_freeze_reference(out_dir):
        calls.append(('freeze_reference', out_dir))
        return frozen_reference_path, {'engine_sha256': 'feedface'}

    def fake_freeze_worker_binary(out_dir):
        calls.append(('freeze_worker_binary', out_dir))
        return frozen_worker_path, 'deadbeef'

    def fake_prepare_anchor(out_dir, job, back_plies, seconds, reference_binary, progress=None):
        calls.append(('prepare_anchor', out_dir, job, back_plies, seconds, reference_binary))
        return anchor_state_path

    class FakeOrchestrator:
        def __init__(self, run_dir, anchor_state_path, worker_binary_path,
                     train_worker_count, eval_worker_count, simulations_per_move=32,
                     reference_seconds=5.0):
            calls.append(('construct', run_dir, anchor_state_path, worker_binary_path,
                           train_worker_count, eval_worker_count, simulations_per_move,
                           reference_seconds))

        def start(self):
            calls.append(('start',))

        def stop(self):
            calls.append(('stop',))

        def status(self):
            calls.append(('status',))
            return {'generation': 0, 'last_loss': None, 'games_available': 0,
                    'eval_games_available': 0, 'workers_alive': 2, 'eval_wins': 0,
                    'eval_n': 0, 'eval_ci_lower': None}

    monkeypatch.setattr(mod, 'freeze_reference', fake_freeze_reference)
    monkeypatch.setattr(mod, 'freeze_worker_binary', fake_freeze_worker_binary)
    monkeypatch.setattr(mod, 'prepare_anchor', fake_prepare_anchor)
    monkeypatch.setattr(mod, 'AnchorPilotOrchestrator', FakeOrchestrator)

    handlers = {}

    def fake_signal(sig, handler):
        handlers[sig] = handler

    monkeypatch.setattr(mod.signal, 'signal', fake_signal)

    def fake_sleep(seconds):
        # Fire the real SIGTERM handler `main()` registered, directly --
        # deterministic, no dependency on OS signal-delivery timing.
        handlers[signal.SIGTERM](signal.SIGTERM, None)

    monkeypatch.setattr(mod.time, 'sleep', fake_sleep)

    out_dir = tmp_path / 'out'
    argv = ['anchor_pilot_campaign', '--out', str(out_dir), '--corpus', str(corpus_path),
            '--back-plies', '3', '--train-workers', '4', '--eval-workers', '1',
            '--simulations-per-move', '16', '--reference-seconds', '1.5']
    monkeypatch.setattr('sys.argv', argv)

    mod.main()

    kinds = [c[0] for c in calls]
    assert kinds.index('freeze_reference') < kinds.index('prepare_anchor')
    assert kinds.index('freeze_worker_binary') < kinds.index('prepare_anchor')
    assert kinds.index('prepare_anchor') < kinds.index('construct')
    assert kinds.index('construct') < kinds.index('start')
    assert kinds.index('start') < kinds.index('status')
    assert kinds.index('status') < kinds.index('stop')

    freeze_reference_call = next(c for c in calls if c[0] == 'freeze_reference')
    assert freeze_reference_call[1] == out_dir

    freeze_call = next(c for c in calls if c[0] == 'freeze_worker_binary')
    assert freeze_call[1] == out_dir

    prepare_call = next(c for c in calls if c[0] == 'prepare_anchor')
    assert prepare_call[1] == out_dir
    assert prepare_call[2] == opening_job
    assert prepare_call[3] == 3
    assert prepare_call[4] == 1.5
    # `reference_binary` must be the frozen reference *library* (.so), not
    # the frozen `selfplay_worker` executable -- `set_engine`/`native
    # .library()` loads it via `ctypes.CDLL`, which cannot dlopen an ELF
    # executable (see this module's docstring).
    assert prepare_call[5] == frozen_reference_path

    construct_call = next(c for c in calls if c[0] == 'construct')
    assert construct_call[1] == out_dir
    assert construct_call[2] == anchor_state_path
    assert construct_call[3] == frozen_worker_path
    assert construct_call[4] == 4
    assert construct_call[5] == 1
    assert construct_call[6] == 16
    assert construct_call[7] == 1.5

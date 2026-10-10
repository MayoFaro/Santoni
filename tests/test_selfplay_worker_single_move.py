import ctypes as C
import time
from pathlib import Path

from santorini.engine import Position, validate_turn
from santorini.native import encode, CState, CAction

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "native_engine" / "target" / "release" / "selfplay_worker"


def test_single_move_produces_a_legal_turn_within_time_budget(tmp_path):
    position = Position()  # position de départ, aucun pouvoir
    state = encode(position)
    input_path = tmp_path / "state.bin"
    input_path.write_bytes(bytes(state))
    output_path = tmp_path / "move.bin"

    started = time.monotonic()
    import subprocess
    result = subprocess.run(
        [str(BINARY), "--single-move", "--input-state", str(input_path),
         "--output-move", str(output_path), "--seconds", "0.5"],
        timeout=5, capture_output=True, text=True,
    )
    elapsed = time.monotonic() - started
    assert result.returncode == 0, result.stderr
    assert elapsed < 2.0  # budget 0.5s + marge de démarrage du processus

    raw = output_path.read_bytes()
    assert len(raw) == C.sizeof(CAction) * 2
    move_action = CAction.from_buffer_copy(raw, 0)
    build_action = CAction.from_buffer_copy(raw, C.sizeof(CAction))
    from santorini.engine import Action
    actions = (
        Action("move", move_action.player, move_action.worker, move_action.source, move_action.target),
        Action("build", build_action.player, build_action.worker, build_action.source, build_action.target),
    )
    turn = validate_turn(position, actions)  # lève si le coup n'est pas légal
    assert turn.after != position

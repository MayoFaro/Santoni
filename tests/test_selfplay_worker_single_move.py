import ctypes as C
import struct
import subprocess
import time
from pathlib import Path

from santorini.engine import Action, Position, validate_turn
from santorini.native import ACTION_KINDS, encode, CAction

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "native_engine" / "target" / "release" / "selfplay_worker"


def run_single_move(tmp_path, position, seconds="0.5", timeout=5):
    """Lance le worker en mode `--single-move` sur `position` et renvoie
    `(returncode, stderr, raw_bytes, elapsed)`."""
    state = encode(position)
    input_path = tmp_path / "state.bin"
    input_path.write_bytes(bytes(state))
    output_path = tmp_path / "move.bin"
    started = time.monotonic()
    result = subprocess.run(
        [str(BINARY), "--single-move", "--input-state", str(input_path),
         "--output-move", str(output_path), "--seconds", seconds],
        timeout=timeout, capture_output=True, text=True,
    )
    elapsed = time.monotonic() - started
    raw = output_path.read_bytes() if output_path.exists() else b""
    return result.returncode, result.stderr, raw, elapsed


def decode_actions(raw):
    """Décode le format de sortie du mode `--single-move` : un `u32`
    petit-boutiste donnant le nombre d'actions du tour, puis ce nombre de
    `CAction` brutes. Le compteur est indispensable : un tour gagnant (montée
    à hauteur 3) n'a qu'une seule action, pas deux — supposer deux en dur
    faisait rejeter par `validate_turn` tout tour gagnant du candidat.

    Le `kind` est décodé via `ACTION_KINDS` (comme
    `santorini.native.decode_turn`) plutôt que supposé `"move"`/`"build"` :
    sur une position quelconque la seconde action peut être un dôme.
    """
    (count,) = struct.unpack_from("<I", raw, 0)
    assert len(raw) == 4 + count * C.sizeof(CAction), (
        f"{len(raw)} octets pour {count} actions annoncées")
    actions = []
    for index in range(count):
        item = CAction.from_buffer_copy(raw, 4 + index * C.sizeof(CAction))
        actions.append(Action(ACTION_KINDS[item.kind], item.player, item.worker,
                              item.source, item.target))
    return tuple(actions)


def test_single_move_produces_a_legal_turn_within_time_budget(tmp_path):
    position = Position()  # position de départ, aucun pouvoir
    code, stderr, raw, elapsed = run_single_move(tmp_path, position)
    assert code == 0, stderr
    assert elapsed < 2.0  # budget 0.5s + marge de démarrage du processus

    actions = decode_actions(raw)
    # Position de départ : aucune montée à hauteur 3 n'est possible, donc le
    # tour a nécessairement ses deux actions (déplacement puis construction).
    assert len(actions) == 2
    turn = validate_turn(position, actions)  # lève si le coup n'est pas légal
    assert turn.after != position
    assert turn.after.winner is None


def winning_in_one_position():
    """Position du sous-jeu sans pouvoir dont le *seul* tour légal est une
    montée gagnante à hauteur 3 — donc un tour à une seule action, le cas
    qui faisait systématiquement échouer `validate_turn` côté campagne.

    Construite selon le même principe que la fixture
    `mcts_tree_tests::state_with_immediate_win` de `native_engine/src/mcts.rs`
    (bâtisseur sur une case de hauteur 2 adjacente à une tour de hauteur 3
    non dômée), mais volontairement verrouillée par des dômes : le bâtisseur
    0 ne peut aller qu'en case 1 (5 et 6 dômés) et le bâtisseur 1 ne peut
    bouger du tout (3, 8 et 9 dômés). Le tour écrit par le worker est alors
    déterministe quelle que soit l'issue de la recherche MCTS, ce qui rend ce
    test insensible au budget de temps.
    """
    heights = [0] * 25
    heights[0] = 2  # bâtisseur 0 du joueur au trait
    heights[1] = 3  # tour de hauteur 3 adjacente, non dômée : montée gagnante
    domes = (1 << 5) | (1 << 6) | (1 << 3) | (1 << 8) | (1 << 9)
    position = Position(heights=tuple(heights), domes=domes,
                        workers=((0, 4), (20, 24)), powers=(0, 0), player=0)
    position.validate()
    return position


def test_single_move_emits_one_action_for_a_winning_turn(tmp_path):
    position = winning_in_one_position()
    code, stderr, raw, _ = run_single_move(tmp_path, position)
    assert code == 0, stderr

    actions = decode_actions(raw)
    assert len(actions) == 1, f"tour gagnant attendu à une action, reçu {actions}"
    assert actions[0].kind == "move"
    assert (actions[0].source, actions[0].target) == (0, 1)
    turn = validate_turn(position, actions)  # lève si le tour n'est pas exact
    assert turn.after.winner == 0

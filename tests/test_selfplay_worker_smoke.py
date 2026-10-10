import json
import subprocess
import signal
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "native_engine" / "target" / "release" / "selfplay_worker"


def test_worker_produces_complete_valid_games(tmp_path):
    out_dir = tmp_path / "games"
    out_dir.mkdir()
    result = subprocess.run(
        [str(BINARY), "--games", "3", "--out-dir", str(out_dir), "--simulations-per-move", "8"],
        timeout=30, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    files = sorted(out_dir.glob("game-*.json"))
    assert len(files) == 3
    for f in files:
        data = json.loads(f.read_text())
        assert data["outcome"] in (-1, 1)
        assert len(data["moves"]) > 0
    assert not list(out_dir.glob("*.tmp"))


def test_concurrent_workers_do_not_overwrite_each_others_games(tmp_path):
    """Deux processus worker écrivant dans le *même* `--out-dir` doivent
    produire l'union de leurs parties, pas se recouvrir.

    Avec l'ancien nommage `game-{:04}.json` (compteur local repartant de 0
    dans chaque processus), les deux workers écrivaient tous les deux
    `game-0000.json`, `game-0001.json`, ... : les parties du second
    écrasaient silencieusement celles du premier et le total tombait à 3 au
    lieu de 6. Pire pour l'entraîneur, `Trainer._consumed_games` étant indexé
    par nom de fichier, le contenu écrasé n'était jamais relu. C'est le même
    mécanisme qui faisait perdre des parties à chaque reprise (nouveau
    processus, compteur à zéro, mêmes noms). Le PID dans le nom de fichier
    ferme les deux cas.
    """
    out_dir = tmp_path / "games"
    out_dir.mkdir()
    procs = [
        subprocess.Popen(
            [str(BINARY), "--games", "3", "--out-dir", str(out_dir),
             "--simulations-per-move", "4"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        for _ in range(2)
    ]
    for proc in procs:
        _, stderr = proc.communicate(timeout=60)
        assert proc.returncode == 0, stderr

    files = sorted(out_dir.glob("game-*.json"))
    assert len(files) == 6, [f.name for f in files]
    assert len({f.name for f in files}) == 6
    for f in files:
        data = json.loads(f.read_text())
        assert data["outcome"] in (-1, 1)
        assert len(data["moves"]) > 0
    assert not list(out_dir.glob("*.tmp"))


def test_restarted_worker_does_not_overwrite_the_previous_run(tmp_path):
    """Deux exécutions *successives* dans le même `--out-dir` (le scénario de
    reprise après arrêt : nouveau processus, compteur de parties réinitialisé)
    doivent cumuler leurs parties au lieu de se recouvrir."""
    out_dir = tmp_path / "games"
    out_dir.mkdir()
    for _ in range(2):
        result = subprocess.run(
            [str(BINARY), "--games", "2", "--out-dir", str(out_dir),
             "--simulations-per-move", "4"],
            timeout=60, capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
    assert len(list(out_dir.glob("game-*.json"))) == 4


def test_worker_leaves_no_partial_file_on_sigint(tmp_path):
    out_dir = tmp_path / "games"
    out_dir.mkdir()
    proc = subprocess.Popen(
        [str(BINARY), "--games", "10000", "--out-dir", str(out_dir), "--simulations-per-move", "4"],
    )
    time.sleep(0.3)
    proc.send_signal(signal.SIGINT)
    proc.wait(timeout=10)
    assert not list(out_dir.glob("*.tmp"))
    for f in out_dir.glob("game-*.json"):
        json.loads(f.read_text())  # ne doit jamais lever

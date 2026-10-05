import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time
from dataclasses import replace

import pytest
from PySide6.QtWidgets import QApplication

from santorini.engine import Position, legal_turns
from santorini.search import Analysis
from santorini.storage import Session, Store
from santorini.ui import Coordinates, MainWindow


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    application.setQuitOnLastWindowClosed(False)
    return application


def until(app, condition, seconds=5):
    deadline = time.monotonic() + seconds
    while not condition() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.005)
    app.processEvents()
    assert condition()


def dispose(app, window):
    window.close()
    until(app, lambda: not any(w.isRunning() for w in window.retired))
    app.processEvents()


def test_two_click_coordinates_disable_illegal_numbers(app):
    picked = []
    selector = Coordinates(picked.append)
    selector.set_allowed({0, 12})
    assert all(not b.isEnabled() for b in selector.numbers)
    selector.letters[2].click()
    assert not picked
    assert selector.numbers[2].isEnabled()
    assert not selector.numbers[0].isEnabled()
    selector.numbers[2].click()
    assert picked == [12]
    assert all(not b.isEnabled() for b in selector.numbers)


def test_both_windows_show_and_close_without_shadowing_qt_methods(app, tmp_path):
    window = MainWindow(Store(tmp_path), restore=False)
    window.show()
    app.processEvents()
    assert window.isVisible()
    assert window.analysis.isVisible()
    assert callable(window.analysis.move)
    assert callable(window.board.render)
    dispose(app, window)


def test_no_powers_configuration_and_bia_placement(app, tmp_path):
    window = MainWindow(Store(tmp_path), restore=False)
    window.families["basic"].setChecked(False)
    assert window.mine.currentData() == window.robot.currentData() == 0
    window.families["advanced"].setChecked(True)
    from PySide6.QtCore import Qt
    for index in range(window.available.count()):
        if window.available.item(index).data(Qt.UserRole) == 13:
            window.available.item(index).setCheckState(Qt.Checked)
    window.robot.setCurrentIndex(window.robot.findData(13))
    assert window.slot.currentIndex() == 2
    window.families["advanced"].setChecked(False)
    window.positions = [6, 8, 16, 18]
    window.start_game()
    assert window.session is not None
    assert window.store.load().position == window.session.position
    dispose(app, window)


def test_input_during_search_stale_results_and_result_archive(app, tmp_path):
    window = MainWindow(Store(tmp_path), restore=False)
    window.robot_game.setChecked(True)
    window.positions = [6, 8, 16, 18]
    window.robot.setCurrentIndex(window.robot.findData(0))
    window.first.setCurrentIndex(1)
    window.move_budget.setValue(1)
    window.start_game()
    until(app, lambda: bool(window.options))
    # Legal input remains available while native minimax is still computing.
    assert window.thinking
    assert any(b.isEnabled() for b in window.coordinates.letters)
    turn = next(legal_turns(window.session.position))
    old_generation = window.generation
    window.commit(turn)
    position = window.session.position
    stale = Analysis(turn, 0, 1, 1, 0)
    window.search_ready(stale, old_generation)
    assert window.session.position == position
    assert window.advice is None
    assert window.session.position.player == 0
    window.result_choice.setCurrentIndex(0)
    window.record_result()
    window.reset()
    assert window.session is None
    archives = list((tmp_path / "parties").glob("*.json"))
    assert len(archives) == 1
    import json
    assert json.loads(archives[0].read_text())["result"]["winner"] == 0
    dispose(app, window)


def test_robot_returns_complete_turn_within_wall_clock_budget(app, tmp_path):
    window = MainWindow(Store(tmp_path), restore=False)
    window.robot_game.setChecked(True)
    window.positions = [6, 8, 16, 18]
    window.robot.setCurrentIndex(window.robot.findData(0))
    window.first.setCurrentIndex(1)
    window.move_budget.setValue(1)
    start = time.monotonic()
    window.start_game()
    until(app, lambda: window.advice is not None and not window.thinking, 3)
    assert time.monotonic() - start < 1.5
    assert window.advice.turn
    assert window.analysis.apply.isEnabled()
    window.play_advice()
    assert len(window.session.history) == 1
    assert window.session.position.player == 0
    dispose(app, window)


def test_failed_archive_does_not_reset_ui(app, tmp_path, monkeypatch):
    window = MainWindow(Store(tmp_path), restore=False)
    window.session = Session(Position()).ended(0, "Test")
    window.store.save(window.session)
    before = window.session
    monkeypatch.setattr(window.store, "archive", lambda s: (_ for _ in ()).throw(OSError("disque plein")))
    window.reset()
    assert window.session is before
    assert window.store.load().position == before.position
    assert "annulée" in window.error.text()
    dispose(app, window)


def test_restart_restores_full_saved_game_asynchronously(app, tmp_path):
    store = Store(tmp_path)
    session = Session(Position()).append(next(legal_turns(Position())))
    store.save(session)
    window = MainWindow(store, restore=True)
    until(app, lambda: not window.loading)
    assert window.session.position == session.position
    assert window.session.history == session.history
    dispose(app, window)


def test_history_correction_cancel_and_save(app, tmp_path):
    from tests.test_correction import sample_session, move, build
    from santorini.engine import validate_turn
    window = MainWindow(Store(tmp_path), restore=False)
    original = sample_session()
    window.store.save(original)
    window.session = original
    window.position_changed()
    until(app, lambda: bool(window.options))
    window.history.setCurrentRow(0)
    assert window.correct_button.isEnabled()
    window.correct_button.click()
    until(app, lambda: bool(window.options))
    assert window.correction_original == original
    assert window.session.position == original.initial
    assert not window.thinking
    assert not window.rethink.isEnabled()
    assert window.store.load() == original
    window.cancel_correction_button.click()
    assert window.session == original
    assert window.correction_original is None
    window.history.setCurrentRow(0)
    window.correct_button.click()
    until(app, lambda: bool(window.options))
    replacement = (move(0, 0, 6, 0), build(0, 0, 0, 5))
    window.complete = validate_turn(window.session.position, replacement)
    window.commit_draft()
    until(app, lambda: window.correction_original is None)
    assert window.session == original.correct(0, replacement)
    assert window.store.load() == window.session
    assert window.session.position.heights[5] == 1
    dispose(app, window)


def test_correction_save_failure_preserves_original_game(app, tmp_path, monkeypatch):
    from tests.test_correction import sample_session, move, build
    window = MainWindow(Store(tmp_path), restore=False)
    original = sample_session()
    window.store.save(original)
    window.session = original
    window.position_changed()
    window.history.setCurrentRow(0)
    window.begin_correction()
    corrected = original.correct(0, (move(0, 0, 6, 0), build(0, 0, 0, 5)))
    def failing_save(_):
        raise OSError('Disque indisponible')
    monkeypatch.setattr(window.store, 'save', failing_save)
    window.correction_ready(corrected, window.generation)
    assert window.correction_original == original
    assert window.store.load() == original
    assert 'Disque indisponible' in window.error.text()
    window.cancel_correction()
    assert window.session == original
    dispose(app, window)


def test_families_show_unchecked_cards_and_preserve_explicit_choices(app, tmp_path):
    from PySide6.QtCore import Qt
    window = MainWindow(Store(tmp_path), restore=False)
    assert not window.robot_game.isChecked()
    window.families['basic'].setChecked(True)
    assert window.available.count() == 10
    assert window.checked_powers() == []
    for index in (0, 1):
        window.available.item(index).setCheckState(Qt.Checked)
    assert window.checked_powers() == [1, 2]
    assert window.mine.findData(1) >= 0 and window.robot.findData(2) >= 0
    assert window.mine.findData(3) == -1
    window.families['hero'].setChecked(True)
    assert window.checked_powers() == [1, 2]
    dispose(app, window)


def test_human_opponent_receives_no_advice_and_my_turn_is_automatic(app, tmp_path):
    window = MainWindow(Store(tmp_path), restore=False)
    window.positions = [6, 8, 16, 18]
    window.first.setCurrentIndex(1)
    window.move_budget.setValue(1)
    window.start_game()
    until(app, lambda: bool(window.options))
    assert window.session.settings['robot_mode'] is False
    assert window.session.position.player == 1
    assert not window.thinking and window.advice is None
    assert not window.rethink.isEnabled()
    assert 'adversaire' in window.analysis.move_label.text().lower()
    window.run_search()
    assert not window.thinking
    window.commit(next(legal_turns(window.session.position)))
    assert window.session.position.player == 0
    assert window.thinking
    until(app, lambda: window.advice is not None and not window.thinking, 3)
    assert window.analysis.apply.isEnabled()
    window.play_advice()
    assert window.session.position.player == 1
    assert not window.thinking and window.advice is None
    saved = window.store.load()
    assert saved.settings['robot_mode'] is False
    dispose(app, window)
    restored = MainWindow(Store(tmp_path), restore=True)
    until(app, lambda: not restored.loading)
    assert not restored.robot_game.isChecked()
    assert not restored.thinking and restored.advice is None
    dispose(app, restored)

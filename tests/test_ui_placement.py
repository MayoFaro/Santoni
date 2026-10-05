import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from santorini.engine import Position, legal_turns
from santorini.placement import PlacementAdvice
from santorini.storage import Session, Store
from santorini.ui import MainWindow
from test_ui import dispose, until


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    application.setQuitOnLastWindowClosed(False)
    return application


def window(app, tmp_path):
    w = MainWindow(Store(tmp_path), restore=False)
    w.robot_game.setChecked(True)
    w.robot.setCurrentIndex(w.robot.findData(0))
    w.placement_budget.setValue(1)
    return w


def test_suggest_and_apply_robot_second_without_creating_game(app, tmp_path):
    w = window(app, tmp_path)
    w.positions[:2] = [6, 8]
    w.recommend_placement()
    assert w.placement_thinking
    assert any(b.isEnabled() for b in w.placement.letters)
    until(app, lambda: w.placement_advice is not None and not w.placement_thinking, 3)
    advice = w.placement_advice
    assert w.apply_placement_button.isEnabled()
    assert w.analysis.use_placement.isEnabled()
    w.apply_placement()
    assert w.positions[:2] == [6, 8]
    assert tuple(w.positions[2:]) == advice.robot_cells
    assert w.session is None
    assert not w.store.current.exists()
    dispose(app, w)


def test_robot_first_does_not_send_future_human_positions(app, tmp_path, monkeypatch):
    w = window(app, tmp_path)
    w.first.setCurrentIndex(1)
    w.positions[:2] = [6, 8]
    args = []
    monkeypatch.setattr(w, "launch", lambda *a, **kw: args.append(a))
    w.recommend_placement()
    assert args[0][0] == "placement"
    assert args[0][1][3] is None
    dispose(app, w)


def test_missing_human_pieces_and_stale_placement_are_rejected(app, tmp_path):
    w = window(app, tmp_path)
    w.recommend_placement()
    assert not w.placement_thinking
    assert "deux bâtisseurs" in w.error.text()
    old = w.placement_generation
    advice = PlacementAdvice((11, 13), (6, 8), 0, 0, 1, 1, 0, "Test")
    w.positions[:2] = [6, 8]
    w.slot.setCurrentIndex(0)
    w.place(5)
    w.placement_ready(advice, old)
    assert w.placement_advice is None
    assert not w.apply_placement_button.isEnabled()
    dispose(app, w)


def test_placement_cannot_modify_active_session(app, tmp_path):
    w = window(app, tmp_path)
    s = Session(Position()).append(next(legal_turns(Position())))
    w.session = s
    w.store.save(s)
    before = w.store.current.read_bytes()
    w.placement_advice = PlacementAdvice((11, 13), (6, 8), 0, 0, 1, 1, 0, "Test")
    w.recommend_placement()
    w.apply_placement()
    assert w.session is s
    assert w.store.current.read_bytes() == before
    assert not w.placement_thinking
    dispose(app, w)


def test_power_change_cancels_pending_placement(app, tmp_path, monkeypatch):
    w = window(app, tmp_path)
    from PySide6.QtCore import Qt
    w.families["basic"].setChecked(True)
    w.available.item(0).setCheckState(Qt.Checked)
    w.positions[:2] = [6, 8]
    w.recommend_placement()
    generation = w.placement_generation
    w.mine.setCurrentIndex(w.mine.findData(1))
    assert w.placement_generation > generation
    assert not w.placement_thinking
    assert w.placement_advice is None
    assert not w.apply_placement_button.isEnabled()
    dispose(app, w)

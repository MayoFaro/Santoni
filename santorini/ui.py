from __future__ import annotations

import multiprocessing
import sys
import time
from dataclasses import replace

from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QMainWindow, QHeaderView, QPushButton, QTabWidget, QTableWidget,
    QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget, QAbstractItemView,
)

from .engine import Action, Position, _apply, coord, validate_setup
from .powers import POWERS, incompatible
from .search import Analysis, PowerAdvice
from .storage import Session, Store
from .worker import Worker


STYLE = """
QWidget { font-size: 13px; }
QPushButton { padding: 5px 9px; border: 1px solid #aab2bc; border-radius: 5px; }
QPushButton:hover { background: #dfeafa; }
QPushButton:disabled { color: #8a919a; }
QPushButton[chip="true"] { border-radius: 13px; min-width: 32px; min-height: 25px; background: #fff; color: #263443; }
QPushButton[chip="true"]:checked { background: #263443; color: white; }
QPushButton[primary="true"] { background: #245f9b; color: white; font-weight: bold; }
QPushButton[primary="true"]:disabled { background: #e4e7eb; color: #8a919a; border-color: #ccd0d5; }
QGroupBox { font-weight: bold; margin-top: 9px; padding-top: 8px; }
QLabel[error="true"] { color: #a92727; background: #ffeded; padding: 7px; border-radius: 4px; }
"""


def button(text, callback, primary=False):
    b = QPushButton(text)
    b.setProperty("primary", primary)
    b.clicked.connect(callback)
    return b


class Coordinates(QWidget):
    """Two rows, two clicks per coordinate. Numbers follow a chosen letter."""
    def __init__(self, callback):
        super().__init__()
        self.callback = callback
        self.letter = None
        self.allowed = set()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.letters, self.numbers = [], []
        for labels, target, is_letter in [("ABCDE", self.letters, True), ("12345", self.numbers, False)]:
            row = QHBoxLayout()
            for label in labels:
                b = QPushButton(label)
                b.setProperty("chip", True)
                b.setCheckable(is_letter)
                b.clicked.connect(lambda checked=False, v=label, letter=is_letter: self.pick(v, letter))
                row.addWidget(b)
                target.append(b)
            layout.addLayout(row)
        self.set_allowed(set())

    def set_allowed(self, cells):
        self.allowed = set(cells)
        self.letter = None
        for i, b in enumerate(self.letters):
            b.setChecked(False)
            b.setEnabled(any(c % 5 == i for c in cells))
        for b in self.numbers:
            b.setEnabled(False)

    def pick(self, value, is_letter):
        if is_letter:
            self.letter = "ABCDE".index(value)
            for i, b in enumerate(self.letters):
                b.setChecked(i == self.letter)
            for i, b in enumerate(self.numbers):
                b.setEnabled(i * 5 + self.letter in self.allowed)
        elif self.letter is not None:
            cell = (int(value) - 1) * 5 + self.letter
            if cell in self.allowed:
                self.letter = None
                for b in self.letters:
                    b.setChecked(False)
                for b in self.numbers:
                    b.setEnabled(False)
                self.callback(cell)


class Board(QTableWidget):
    def __init__(self):
        super().__init__(5, 5)
        self.setHorizontalHeaderLabels(list("ABCDE"))
        self.setVerticalHeaderLabels(list("12345"))
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.NoSelection)
        self.setFocusPolicy(Qt.NoFocus)
        self.setMinimumHeight(260)
        self.setMaximumHeight(320)
        for i in range(5):
            self.setColumnWidth(i, 65)
            self.setRowHeight(i, 45)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)

    def render_position(self, s, highlights=()):
        for cell in range(25):
            occupants = [(p, i) for p, ws in enumerate(s.workers) for i, w in enumerate(ws) if w == cell]
            text = f"{s.heights[cell]}" + (" • Dôme" if s.domes & (1 << cell) else "")
            if occupants:
                p, i = occupants[0]
                text += f"\n{'M' if p == 0 else 'R'}{i + 1}"
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignCenter)
            item.setToolTip(f"{coord(cell)} : niveau {s.heights[cell]}" + (", dôme" if s.domes & (1 << cell) else ""))
            color = "#f1f3f5" if s.domes & (1 << cell) else "#ffffff"
            if occupants:
                color = "#dbeaff" if occupants[0][0] == 0 else "#ffe1cd"
            if cell in highlights:
                color = "#fff2ac"
            item.setBackground(QColor(color))
            item.setForeground(QColor("#263443"))
            self.setItem(cell // 5, cell % 5, item)


class AnalysisWindow(QMainWindow):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner
        self.setWindowTitle("Santoni — Analyse")
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self.resize(520, 670)
        container = QWidget()
        self.setCentralWidget(container)
        layout = QVBoxLayout(container)
        self.status = QLabel("Configurez une partie.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        move = QWidget()
        ml = QVBoxLayout(move)
        self.move_label = QLabel("Aucun coup calculé.")
        self.move_label.setWordWrap(True)
        self.move_label.setStyleSheet("font-size: 16px; padding: 10px; background: #e6eef8;")
        ml.addWidget(self.move_label)
        self.apply = button("Jouer ce tour complet", owner.play_advice, True)
        self.apply.setEnabled(False)
        ml.addWidget(self.apply)
        self.metadata = QLabel()
        self.metadata.setWordWrap(True)
        ml.addWidget(self.metadata)
        self.explanation = QLabel()
        self.explanation.setWordWrap(True)
        ml.addWidget(self.explanation)
        self.variation = QTextEdit()
        self.variation.setReadOnly(True)
        ml.addWidget(self.variation)
        self.tabs.addTab(move, "Coup suivant")
        power = QWidget()
        pl = QVBoxLayout(power)
        self.power_text = QLabel("Choisissez les familles et le pouvoir adverse dans Configuration, puis lancez la comparaison.")
        self.power_text.setWordWrap(True)
        pl.addWidget(self.power_text)
        self.power_list = QListWidget()
        pl.addWidget(self.power_list)
        self.use_power = button("Retenir ce pouvoir", owner.apply_power, True)
        self.use_power.setEnabled(False)
        pl.addWidget(self.use_power)
        self.tabs.addTab(power, "Choix du pouvoir")

    def closeEvent(self, event):
        if not self.owner.closing:
            self.owner.close()
        super().closeEvent(event)


class MainWindow(QMainWindow):
    def __init__(self, store=None, restore=True):
        super().__init__()
        self.store = store or Store()
        self.session = None
        self.loading = restore
        self.load_failed = False
        self.closing = False
        self.generation = 0
        self.input_generation = 0
        self.power_generation = 0
        self.workers = {}
        self.retired = set()
        self.draft = ()
        self.options = []
        self.complete = None
        self.advice = None
        self.power_advice = None
        self.power_target = 1
        self.thinking = False
        self.thinking_started = 0
        self.budget_used = 0
        self.positions = [-1] * 4
        self.setWindowTitle("Santoni — Saisie")
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self.setMinimumWidth(420)
        self.resize(450, 850)
        container = QWidget()
        self.setCentralWidget(container)
        layout = QVBoxLayout(container)
        self.error = QLabel()
        self.error.setProperty("error", True)
        self.error.setWordWrap(True)
        self.error.hide()
        layout.addWidget(self.error)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        self._configuration()
        self._game()
        self.history = QTextEdit()
        self.history.setReadOnly(True)
        self.tabs.addTab(self.history, "Historique")
        self.footer = QLabel("Sauvegarde automatique après chaque tour. Ctrl + molette : opacité.")
        self.footer.setWordWrap(True)
        layout.addWidget(self.footer)
        self.analysis = AnalysisWindow(self)
        self._positioned = False
        QApplication.instance().installEventFilter(self)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(100)
        self.refresh_configuration()
        self.refresh()
        if restore:
            self.footer.setText("Recherche d'une partie sauvegardée…")
            QTimer.singleShot(0, self.restore)

    def _configuration(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        row = QHBoxLayout()
        self.families = {}
        for key, text in [("basic", "Dieux de base"), ("hero", "Pouvoirs de héros"), ("advanced", "Dieux avancés")]:
            box = QCheckBox(text)
            box.setChecked(key == "basic")
            box.toggled.connect(self.refresh_configuration)
            self.families[key] = box
            row.addWidget(box)
        layout.addLayout(row)
        self.available = QListWidget()
        self.available.setMaximumHeight(125)
        self.available.itemChanged.connect(self.available_changed)
        layout.addWidget(self.available)
        hint = QLabel("Cochez les cartes disponibles. Les pouvoirs grisés restent à implémenter.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        form = QFormLayout()
        self.mine, self.robot = QComboBox(), QComboBox()
        self.mine.currentIndexChanged.connect(self.configuration_changed)
        self.robot.currentIndexChanged.connect(self.configuration_changed)
        form.addRow("Mon pouvoir", self.mine)
        form.addRow("Pouvoir du robot", self.robot)
        self.first = QComboBox()
        self.first.addItems(["Moi", "Robot"])
        self.first.currentIndexChanged.connect(self.configuration_changed)
        form.addRow("Premier joueur", self.first)
        self.move_budget = QDoubleSpinBox()
        self.move_budget.setRange(1, 600)
        self.move_budget.setDecimals(1)
        self.move_budget.setValue(5)
        self.move_budget.setSuffix(" s")
        form.addRow("Réflexion par tour", self.move_budget)
        self.power_budget = QDoubleSpinBox()
        self.power_budget.setRange(1, 600)
        self.power_budget.setValue(15)
        self.power_budget.setSuffix(" s")
        form.addRow("Réflexion sur le pouvoir", self.power_budget)
        self.choose_for = QComboBox()
        self.choose_for.addItems(["Robot", "Moi"])
        self.choose_for.currentIndexChanged.connect(self.configuration_changed)
        form.addRow("Conseiller le pouvoir de", self.choose_for)
        layout.addLayout(form)
        self.power_button = button("Comparer les pouvoirs disponibles", self.recommend_power)
        layout.addWidget(self.power_button)
        self.power_detail = QLabel()
        self.power_detail.setWordWrap(True)
        layout.addWidget(self.power_detail)
        self.match_warning = QLabel()
        self.match_warning.setWordWrap(True)
        layout.addWidget(self.match_warning)
        group = QGroupBox("Placement initial — lettre puis nombre")
        gl = QVBoxLayout(group)
        self.slot = QComboBox()
        self.slot.addItems(["Moi 1", "Moi 2", "Robot 1", "Robot 2"])
        self.slot.currentIndexChanged.connect(self.render_placement)
        gl.addWidget(self.slot)
        self.placement_label = QLabel()
        self.placement_label.setWordWrap(True)
        gl.addWidget(self.placement_label)
        self.placement = Coordinates(self.place)
        gl.addWidget(self.placement)
        layout.addWidget(group)
        self.start_button = button("Démarrer la partie", self.start_game, True)
        layout.addWidget(self.start_button)
        layout.addStretch()
        self.tabs.addTab(page, "Configuration")

    def _game(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.turn_label = QLabel("Aucune partie en cours.")
        self.turn_label.setWordWrap(True)
        layout.addWidget(self.turn_label)
        self.board = Board()
        layout.addWidget(self.board)
        self.prompt = QLabel("Choisissez une configuration.")
        self.prompt.setWordWrap(True)
        layout.addWidget(self.prompt)
        row = QHBoxLayout()
        self.action_kind = QComboBox()
        self.actor = QComboBox()
        self.action_kind.currentIndexChanged.connect(self.update_actors)
        self.actor.currentIndexChanged.connect(self.update_coordinates)
        row.addWidget(self.action_kind)
        row.addWidget(self.actor)
        layout.addLayout(row)
        self.coordinates = Coordinates(self.coordinate_action)
        layout.addWidget(self.coordinates)
        self.special = button("Utiliser le pouvoir de héros", self.activate_hero)
        layout.addWidget(self.special)
        self.draft_text = QLabel()
        self.draft_text.setWordWrap(True)
        layout.addWidget(self.draft_text)
        row = QHBoxLayout()
        self.back = button("Annuler l'action", self.back_action)
        self.commit_button = button("Valider le tour / terminer", self.commit_draft, True)
        row.addWidget(self.back)
        row.addWidget(self.commit_button)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.undo_button = button("Annuler le dernier tour", self.undo)
        self.rethink = button("Recalculer", self.run_search)
        row.addWidget(self.undo_button)
        row.addWidget(self.rethink)
        layout.addLayout(row)
        budget = QHBoxLayout()
        budget.addWidget(QLabel("Budget du prochain calcul"))
        self.game_budget = QDoubleSpinBox()
        self.game_budget.setRange(1, 600)
        self.game_budget.setDecimals(1)
        self.game_budget.setValue(5)
        self.game_budget.setSuffix(" s")
        budget.addWidget(self.game_budget)
        layout.addLayout(budget)
        result = QHBoxLayout()
        self.result_choice = QComboBox()
        self.result_choice.addItems(["Moi gagnant", "Robot gagnant", "Partie interrompue"])
        self.end_button = button("Enregistrer le résultat", self.record_result)
        result.addWidget(self.result_choice)
        result.addWidget(self.end_button)
        layout.addLayout(result)
        self.reset_button = button("Archiver et réinitialiser", self.reset)
        layout.addWidget(self.reset_button)
        layout.addStretch()
        self.tabs.addTab(page, "Partie")

    def showEvent(self, event):
        super().showEvent(event)
        if not self._positioned:
            self.analysis.move(self.x() + self.width() + 12, self.y())
            self._positioned = True
        self.analysis.show()

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel and event.modifiers() & Qt.ControlModifier:
            target = obj.window() if isinstance(obj, QWidget) else None
            if target in (self, self.analysis):
                direction = 0.05 if event.angleDelta().y() > 0 else -0.05
                target.setWindowOpacity(max(0.2, min(1, target.windowOpacity() + direction)))
                return True
        return super().eventFilter(obj, event)

    def fail(self, message):
        self.error.setText(str(message))
        self.error.show()

    def clear_error(self):
        self.error.hide()

    def checked_powers(self):
        return [self.available.item(i).data(Qt.UserRole) for i in range(self.available.count())
                if self.available.item(i).checkState() == Qt.Checked]

    def refresh_configuration(self, *_):
        if not hasattr(self, "analysis"):
            return
        previous_checks = {self.available.item(i).data(Qt.UserRole): self.available.item(i).checkState()
                           for i in range(self.available.count())}
        self.available.blockSignals(True)
        self.available.clear()
        for p in POWERS.values():
            if p.number == 0 or not self.families[p.family].isChecked():
                continue
            item = QListWidgetItem(f"{p.number}. {p.name}" + (" — à venir" if not p.supported else ""))
            item.setData(Qt.UserRole, p.number)
            item.setToolTip(p.description)
            if p.supported:
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(previous_checks.get(p.number, Qt.Checked))
            else:
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
            self.available.addItem(item)
        self.available.blockSignals(False)
        self.available_changed()

    def available_changed(self, *_):
        previous = (self.mine.currentData(), self.robot.currentData())
        candidates = self.checked_powers()
        for i, combo in enumerate((self.mine, self.robot)):
            combo.blockSignals(True)
            combo.clear()
            if i == 1:
                combo.addItem("À déterminer par le robot", -1)
            combo.addItem("Aucun pouvoir", 0)
            for p in candidates:
                combo.addItem(POWERS[p].name, p)
                combo.setItemData(combo.count() - 1, POWERS[p].description, Qt.ToolTipRole)
            wanted = previous[i] if previous[i] is not None else (0 if i == 0 else -1)
            if not candidates:
                wanted = 0
            found = combo.findData(wanted)
            combo.setCurrentIndex(found if found >= 0 else combo.findData(0))
            combo.blockSignals(False)
        self.configuration_changed()

    def configuration_changed(self, *_):
        if not hasattr(self, "analysis"):
            return
        self.power_generation += 1
        self.cancel("power")
        self.power_advice = None
        self.analysis.use_power.setEnabled(False)
        self.analysis.power_text.setText("Configuration modifiée : relancer la comparaison des pouvoirs.")
        self.analysis.power_list.clear()
        self.power_button.setEnabled(bool(self.checked_powers()) and self.session is None and not self.loading and not self.load_failed)
        mine, robot = self.mine.currentData(), self.robot.currentData()
        if mine is None or robot is None:
            return
        self.power_detail.setText(f"Moi : {POWERS[mine].description}\nRobot : {POWERS[robot].description if robot >= 0 else 'Pouvoir à choisir.'}")
        messages = []
        if robot >= 0 and incompatible(mine, robot):
            messages.append("Association non recommandée dans le règlement.")
        if robot >= 0 and (mine >= 46) != (robot >= 46) and mine and robot:
            messages.append("Héros contre dieu : le règlement recommande héros contre héros pour l'équilibre.")
        if 13 in (mine, robot):
            messages.append("Bia place ses bâtisseurs en premier.")
        if 19 in (mine, robot):
            messages.append("Eros : placer les deux bâtisseurs sur deux bords opposés.")
        self.match_warning.setText("\n".join(messages))
        if all(v < 0 for v in self.positions):
            first = 0 if mine == 13 else 1 if robot == 13 else self.first.currentIndex()
            self.slot.setCurrentIndex(first * 2)
        self.render_placement()

    def render_placement(self, *_):
        if not hasattr(self, "placement"):
            return
        self.placement_label.setText(" · ".join(f"{'M' if i < 2 else 'R'}{i % 2 + 1} : {coord(v) if v >= 0 else '…'}" for i, v in enumerate(self.positions)))
        selected = self.slot.currentIndex()
        taken = {v for i, v in enumerate(self.positions) if i != selected and v >= 0}
        self.placement.set_allowed(set(range(25)) - taken)

    def place(self, cell):
        index = self.slot.currentIndex()
        self.positions[index] = cell
        mine, robot = self.mine.currentData(), self.robot.currentData()
        first = (0 if mine == 13 else 1 if robot == 13 else self.first.currentIndex())
        order = [first * 2, first * 2 + 1, (1 - first) * 2, (1 - first) * 2 + 1]
        missing = [i for i in order if self.positions[i] < 0]
        if missing:
            self.slot.setCurrentIndex(missing[0])
        self.render_placement()

    def launch(self, kind, args, generation, result, progress=None, error=None):
        self.cancel(kind)
        worker = Worker(kind, args, generation, self)
        self.workers[kind] = worker
        worker.result.connect(result)
        if progress:
            worker.progress.connect(progress)
        worker.failed.connect(error or (lambda message, token: self.fail(message)))
        worker.finished.connect(lambda w=worker, k=kind: self.worker_finished(k, w))
        worker.start()

    def worker_finished(self, kind, worker):
        if self.workers.get(kind) is worker:
            del self.workers[kind]
        self.retired.discard(worker)
        worker.deleteLater()

    def cancel(self, kind):
        worker = self.workers.pop(kind, None)
        if worker:
            worker.cancel()
            self.retired.add(worker)

    def restore(self):
        self.launch("load", (self.store,), 0, self.restored, error=self.restore_error)

    def restored(self, session, _):
        if self.closing:
            return
        self.loading = False
        self.session = session
        if session:
            self.game_budget.setValue(session.settings.get("move_seconds", 5))
            self.tabs.setCurrentIndex(1)
            self.footer.setText(f"Partie reprise : {self.store.current}")
            self.position_changed()
        else:
            self.footer.setText("Aucune partie en cours. Configurez les pouvoirs et le placement.")
            self.refresh()
            self.configuration_changed()

    def restore_error(self, message, _):
        self.loading = False
        self.load_failed = True
        self.fail(f"Sauvegarde non chargée : {message}. Le fichier est préservé : {self.store.current}")
        self.refresh()

    def start_game(self):
        if self.loading or self.load_failed or self.session is not None:
            return
        try:
            robot = self.robot.currentData()
            if robot < 0:
                raise ValueError("Choisissez le pouvoir du robot ou lancez sa comparaison.")
            pos = validate_setup((self.positions[:2], self.positions[2:]),
                                 (self.mine.currentData(), robot), self.first.currentIndex())
            settings = {"move_seconds": self.move_budget.value(), "power_seconds": self.power_budget.value(),
                        "available_powers": self.checked_powers(),
                        "families": {k: b.isChecked() for k, b in self.families.items()}}
            session = Session(pos, settings=settings)
            self.store.save(session)
        except (ValueError, OSError) as exc:
            self.fail(exc)
            return
        self.clear_error()
        self.power_generation += 1
        self.cancel("power")
        self.session = session
        self.game_budget.setValue(self.move_budget.value())
        self.tabs.setCurrentIndex(1)
        self.position_changed()

    def refresh(self):
        active = self.session is not None
        playing = active and self.session.result is None
        self.tabs.setTabEnabled(0, not active)
        self.start_button.setEnabled(not active and not self.loading and not self.load_failed)
        self.undo_button.setEnabled(active and bool(self.session.history))
        self.rethink.setEnabled(playing)
        self.end_button.setEnabled(playing)
        self.reset_button.setEnabled(active)
        self.back.setEnabled(bool(self.draft) and playing)
        self.commit_button.setEnabled(self.complete is not None and playing)
        self.analysis.apply.setEnabled(playing and self.advice is not None and self.advice.turn is not None and not self.thinking)
        if not active:
            self.board.render_position(Position(workers=((-1, -1), (-1, -1))))
            self.turn_label.setText("Aucune partie en cours.")
            self.coordinates.set_allowed(set())
            self.special.hide()
            return
        pos = self.session.position
        preview = pos
        for a in self.draft:
            preview = _apply(preview, a)
        shown = self.draft or (self.advice.turn.actions if self.advice and self.advice.turn else ())
        self.board.render_position(preview, {cell for a in shown for cell in (a.source, a.target) if cell >= 0})
        powers = f"Moi : {POWERS[pos.powers[0]].name} · Robot : {POWERS[pos.powers[1]].name}"
        if self.session.result:
            outcome = self.session.result
            winner = outcome.get("winner")
            self.turn_label.setText(f"{'Partie interrompue' if winner is None else ('Moi' if winner == 0 else 'Robot') + ' gagne'} — {outcome['reason']}\n{powers}")
        else:
            self.turn_label.setText(f"Tour {len(self.session.history) + 1} — {'À moi' if pos.player == 0 else 'Au robot'}\n{powers}")
        self.draft_text.setText("\n".join(a.label() for a in self.draft) or "Aucune action saisie.")
        lines = []
        for i, t in enumerate(self.session.history):
            player = self.session.initial.player if i % 2 == 0 else 1 - self.session.initial.player
            before = self.session.initial if i == 0 else self.session.history[i - 1].after
            lines.append(f"{i + 1}. {'Moi' if player == 0 else 'Robot'}\n{t.description(before)}\n{t.power_summary(before)}\n")
        if self.session.result:
            lines.append(f"Résultat : {self.session.result}")
        self.history.setPlainText("\n".join(lines))

    def position_changed(self):
        self.generation += 1
        self.cancel("search")
        self.cancel("actions")
        self.thinking = False
        self.advice = None
        self.draft, self.options, self.complete = (), [], None
        self.analysis.move_label.setText("Calcul du nouveau tour…" if self.session and self.session.result is None else "Partie terminée.")
        self.analysis.variation.clear()
        self.analysis.metadata.clear()
        self.analysis.explanation.clear()
        self.refresh()
        self.request_actions()
        if self.session and self.session.result is None and self.session.position.player == 1:
            self.run_search()
        elif self.session and self.session.result is None:
            self.analysis.move_label.setText("À vous de jouer. Recalculer permet de demander un conseil.")
            self.analysis.status.setText("Le robot réfléchira après la validation de votre tour.")
        else:
            self.analysis.status.setText("Partie terminée et sauvegardée.")

    def request_actions(self):
        self.input_generation += 1
        self.options, self.complete = [], None
        self.coordinates.set_allowed(set())
        self.action_kind.clear()
        self.actor.clear()
        self.special.hide()
        self.refresh()
        if not self.session or self.session.result:
            return
        self.prompt.setText("Vérification des actions légales… La saisie reste indépendante de l'analyse.")
        self.launch("actions", (self.session.position, self.draft), self.input_generation,
                    self.actions_ready, error=self.actions_error)

    def actions_error(self, message, token):
        if token == self.input_generation and not self.closing:
            self.fail(message)
            self.prompt.setText("Impossible de vérifier les actions.")

    def actions_ready(self, payload, token):
        if token != self.input_generation or not self.session or self.closing:
            return
        self.options, self.complete = payload
        if not self.draft and not self.options and self.complete is None and not self.session.result:
            self.finish_loss()
            return
        kinds = sorted({a.kind for a in self.options if a.kind != "activate"})
        names = {"move": "Déplacement", "build": "Construction", "dome": "Dôme", "remove": "Retirer un bloc",
                 "kill": "Éliminer", "force": "Déplacement forcé", "place": "Nouveau bâtisseur", "adonis": "Cible d'Adonis"}
        self.action_kind.blockSignals(True)
        self.action_kind.clear()
        for kind in kinds:
            self.action_kind.addItem(names.get(kind, kind), kind)
        self.action_kind.blockSignals(False)
        self.special.setVisible(any(a.kind == "activate" for a in self.options))
        self.prompt.setText("Tour complet : validez, ou poursuivez avec une action facultative." if self.complete else
                            "Choisissez l'action et le bâtisseur, puis la lettre et le nombre de la case cible.")
        self.update_actors()
        self.refresh()

    def update_actors(self, *_):
        self.actor.blockSignals(True)
        self.actor.clear()
        kind = self.action_kind.currentData()
        pairs = sorted({(a.player, a.worker) for a in self.options if a.kind == kind})
        for player, worker in pairs:
            source = next(a.source for a in self.options if a.kind == kind and (a.player, a.worker) == (player, worker))
            text = "Cases adjacentes" if worker < 0 else f"{'Moi' if player == 0 else 'Robot'} {worker + 1}"
            if source >= 0:
                text += f" ({coord(source)})"
            self.actor.addItem(text, (player, worker))
        self.actor.blockSignals(False)
        self.update_coordinates()

    def update_coordinates(self, *_):
        kind, actor = self.action_kind.currentData(), self.actor.currentData()
        cells = {a.target for a in self.options if a.kind == kind and (a.player, a.worker) == actor and a.target >= 0}
        self.coordinates.set_allowed(cells)

    def coordinate_action(self, cell):
        kind, actor = self.action_kind.currentData(), self.actor.currentData()
        action = next((a for a in self.options if a.kind == kind and (a.player, a.worker) == actor and a.target == cell), None)
        if action:
            self.draft += (action,)
            self.request_actions()

    def activate_hero(self):
        action = next((a for a in self.options if a.kind == "activate"), None)
        if action:
            self.draft += (action,)
            self.request_actions()

    def back_action(self):
        if self.draft:
            self.draft = self.draft[:-1]
            self.request_actions()

    def commit_draft(self):
        if self.complete is not None:
            self.commit(self.complete)

    def commit(self, turn):
        if not self.session or self.session.result:
            return
        try:
            next_session = replace(self.session.append(turn),
                                   settings={**self.session.settings, "move_seconds": self.game_budget.value()})
            self.store.save(next_session)
        except (ValueError, OSError) as exc:
            self.fail(f"Tour non enregistré : {exc}")
            return
        self.clear_error()
        self.session = next_session
        self.footer.setText(f"Tour sauvegardé : {self.store.current}")
        self.position_changed()

    def run_search(self):
        if not self.session or self.session.result:
            return
        self.generation += 1
        token = self.generation
        self.advice = None
        self.thinking = True
        self.thinking_started = time.monotonic()
        self.budget_used = self.game_budget.value()
        self.analysis.apply.setEnabled(False)
        self.analysis.status.setText(f"Réflexion — budget {self.budget_used:g} s")
        self.analysis.move_label.setText("Recherche d'un tour complet…")
        self.analysis.tabs.setCurrentIndex(0)
        self.launch("search", (self.session.position, self.budget_used, self.thinking_started + self.budget_used), token,
                    self.search_ready, self.search_progress, self.search_error)

    def search_progress(self, analysis, token):
        if token == self.generation and not self.closing:
            self.show_analysis(analysis)

    def search_ready(self, analysis, token):
        if token != self.generation or not self.session or self.closing:
            return
        self.thinking = False
        self.show_analysis(analysis)
        if analysis.turn is None and analysis.complete_depth and analysis.score == -100_000:
            self.finish_loss()
            return
        self.analysis.status.setText(f"Calcul terminé en {analysis.elapsed:.2f} s — {analysis.status}")
        self.refresh()

    def search_error(self, message, token):
        if token == self.generation and not self.closing:
            self.thinking = False
            self.fail(message)
            self.analysis.status.setText("Calcul interrompu. Le dernier tour légal calculé reste disponible.")
            self.refresh()

    def show_analysis(self, analysis: Analysis):
        self.advice = analysis
        self.analysis.move_label.setText((analysis.turn.description(self.session.position) + "\n" + analysis.turn.power_summary(self.session.position))
                                   if analysis.turn and self.session else analysis.status)
        proof = "Démontré" if analysis.proven else "Estimation" if analysis.score is not None else "Sans évaluation"
        score = "…" if analysis.score is None else str(analysis.score)
        self.analysis.metadata.setText(f"{proof} · score {score} pour le joueur au trait\nProfondeur {analysis.depth}"
                                       f"{' complète' if analysis.complete_depth else ' partielle'} · {analysis.nodes:,} positions · {analysis.elapsed:.2f} s")
        if analysis.proven:
            explanation = analysis.status
        elif analysis.turn:
            used = any(a.kind == "activate" for a in analysis.turn.actions)
            explanation = "Le moteur compare les réponses adverses en privilégiant les menaces de victoire, la mobilité et l'accès aux étages."
            if self.session and self.session.position.powers[self.session.position.player] >= 46:
                explanation += " Pouvoir de héros " + ("utilisé." if used else "conservé.")
        else:
            explanation = analysis.status
        self.analysis.explanation.setText(explanation)
        lines = []
        before = self.session.position if self.session else None
        for i, turn in enumerate(analysis.variation):
            lines.append(f"{i + 1}. {turn.description(before) if before else turn.label()}")
            before = turn.after
        self.analysis.variation.setPlainText("\n\n".join(lines) or "Pas encore de variante calculée.")
        if self.session and not self.draft and analysis.turn:
            self.board.render_position(self.session.position,
                                       {cell for a in analysis.turn.actions for cell in (a.source, a.target) if cell >= 0})

    def play_advice(self):
        if self.advice and self.advice.turn and not self.thinking and self.session and self.session.result is None:
            self.commit(self.advice.turn)

    def finish_loss(self):
        if not self.session or self.session.result:
            return
        outcome = self.session.ended(1 - self.session.position.player, "Aucun tour complet légal")
        try:
            self.store.save(outcome)
        except OSError as exc:
            self.fail(exc)
            return
        self.session = outcome
        self.position_changed()

    def undo(self):
        if self.session and self.session.history:
            previous = self.session.undo()
            try:
                self.store.save(previous)
            except OSError as exc:
                self.fail(exc)
                return
            self.session = previous
            self.position_changed()

    def record_result(self):
        if not self.session or self.session.result:
            return
        winner = self.result_choice.currentIndex()
        outcome = self.session.ended(winner if winner < 2 else None, "Résultat déclaré" if winner < 2 else "Partie interrompue")
        try:
            self.store.save(outcome)
        except OSError as exc:
            self.fail(exc)
            return
        self.session = outcome
        self.position_changed()

    def reset(self):
        try:
            archive = self.store.reset(self.session)
        except OSError as exc:
            self.fail(f"Réinitialisation annulée : {exc}")
            return
        self.session = None
        self.generation += 1
        self.input_generation += 1
        self.power_generation += 1
        for kind in list(self.workers):
            self.cancel(kind)
        self.draft, self.options, self.complete, self.advice = (), [], None, None
        self.thinking = False
        self.analysis.move_label.setText("Configurez une nouvelle partie.")
        self.analysis.status.setText("Partie archivée.")
        self.analysis.apply.setEnabled(False)
        self.analysis.use_power.setEnabled(False)
        self.analysis.variation.clear()
        self.analysis.metadata.clear()
        self.analysis.explanation.clear()
        self.history.clear()
        self.action_kind.clear()
        self.actor.clear()
        self.draft_text.clear()
        self.prompt.setText("Choisissez une configuration.")
        self.positions = [-1] * 4
        self.tabs.setCurrentIndex(0)
        self.footer.setText(f"Archive : {archive}" if archive else "Nouvelle partie.")
        self.clear_error()
        self.refresh()
        self.configuration_changed()

    def recommend_power(self):
        candidates = self.checked_powers()
        if not candidates or self.session is not None:
            return
        self.power_generation += 1
        self.power_target = 1 if self.choose_for.currentIndex() == 0 else 0
        other = self.mine.currentData() if self.power_target == 1 else self.robot.currentData()
        other = None if other == -1 else other
        self.power_advice = None
        self.analysis.use_power.setEnabled(False)
        self.analysis.power_list.clear()
        self.analysis.power_text.setText("Comparaison des pouvoirs sur des placements de référence…")
        self.analysis.tabs.setCurrentIndex(1)
        self.power_button.setEnabled(False)
        self.launch("power", (candidates, other, self.power_target, self.power_budget.value(), self.first.currentIndex()),
                    self.power_generation, self.power_ready, self.power_progress, self.power_error)

    def power_progress(self, advice: PowerAdvice, token):
        if token != self.power_generation or self.closing:
            return
        self.power_advice = advice
        name = POWERS[advice.power].name if advice.power is not None else "…"
        self.analysis.power_text.setText(f"Pouvoir conseillé : {name}\n{advice.status}\n{advice.rounds} placements de référence · {advice.elapsed:.2f} s")
        self.analysis.power_list.clear()
        for p, score in sorted(advice.scores.items(), key=lambda item: (-item[1], item[0])):
            self.analysis.power_list.addItem(f"{POWERS[p].name} : {score:+.1f}")

    def power_ready(self, advice, token):
        if token != self.power_generation or self.closing:
            return
        self.power_progress(advice, token)
        self.analysis.use_power.setEnabled(advice.power is not None)
        self.power_button.setEnabled(True)

    def power_error(self, message, token):
        if token == self.power_generation and not self.closing:
            self.fail(message)
            self.power_button.setEnabled(True)

    def apply_power(self):
        if self.session is None and self.power_advice and self.power_advice.power is not None:
            combo = self.robot if self.power_target == 1 else self.mine
            combo.setCurrentIndex(combo.findData(self.power_advice.power))

    def tick(self):
        if self.thinking:
            elapsed = time.monotonic() - self.thinking_started
            self.analysis.status.setText(f"Réflexion : {min(elapsed, self.budget_used):.1f} / {self.budget_used:g} s — saisie disponible")

    def closeEvent(self, event):
        if not self.closing:
            self.closing = True
            self.timer.stop()
            self.generation += 1
            self.input_generation += 1
            self.power_generation += 1
            for kind in list(self.workers):
                self.cancel(kind)
        # Keep Qt alive while worker threads reap their child processes.
        if any(worker.isRunning() for worker in self.retired):
            event.ignore()
            QTimer.singleShot(50, self.close)
            return
        self.analysis.close()
        event.accept()


def main():
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    app.setApplicationName("Santoni")
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    return app.exec()

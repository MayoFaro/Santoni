from __future__ import annotations

import multiprocessing
import os
import sys
import time
from dataclasses import replace

from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QMainWindow, QHeaderView, QPushButton, QTabWidget, QTableWidget, QScrollArea,
    QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget, QAbstractItemView,
)

from .engine import Action, Position, _apply, coord, validate_setup
from .powers import POWERS, incompatible
from .search import Analysis, PowerAdvice
from .placement import PlacementAdvice, placement_first
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


class BoardGrid(QTableWidget):
    def __init__(self):
        super().__init__(5, 5)
        self.setHorizontalHeaderLabels(list("ABCDE"))
        self.setVerticalHeaderLabels(list("54321"))
        self.horizontalHeader().hide()
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
            self.setItem(4 - cell // 5, cell % 5, item)


class Board(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.table = BoardGrid()
        layout.addWidget(self.table)
        self.labels = QWidget()
        footer = QHBoxLayout(self.labels)
        footer.setContentsMargins(0, 0, 0, 0)
        footer.setSpacing(0)
        self.corner = QWidget()
        footer.addWidget(self.corner)
        for letter in "ABCDE":
            label = QLabel(letter)
            label.setAlignment(Qt.AlignCenter)
            footer.addWidget(label, 1)
        layout.addWidget(self.labels)
        self.table.verticalHeader().sectionResized.connect(self.align_labels)

    def align_labels(self, *_):
        self.corner.setFixedWidth(self.table.verticalHeader().width() + self.table.frameWidth())
        self.labels.layout().setContentsMargins(0, 0, self.table.frameWidth(), 0)

    def showEvent(self, event):
        super().showEvent(event)
        self.align_labels()

    def render_position(self, position, highlights=()):
        self.table.render_position(position, highlights)
        self.align_labels()


class AnalysisWindow(QMainWindow):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner
        preview = os.environ.get("SANTONI_PREVIEW_LABEL")
        self.setWindowTitle("Santoni — Analyse" + (f" [{preview}]" if preview else ""))
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
        placement = QWidget()
        sl = QVBoxLayout(placement)
        self.placement_text = QLabel("Choisissez les deux pouvoirs et l'ordre de placement dans Configuration.")
        self.placement_text.setWordWrap(True)
        sl.addWidget(self.placement_text)
        self.placement_board = Board()
        sl.addWidget(self.placement_board)
        self.placement_board.render_position(Position(workers=((-1, -1), (-1, -1))))
        self.use_placement = button("Appliquer le placement conseillé", owner.apply_placement, True)
        self.use_placement.setEnabled(False)
        sl.addWidget(self.use_placement)
        note = QLabel("Robot en premier : les pions adverses montrés sont une réponse hypothétique, pas un placement imposé.")
        note.setWordWrap(True)
        sl.addWidget(note)
        sl.addStretch()
        self.tabs.addTab(placement, "Placement")

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
        self.placement_generation = 0
        self.workers = {}
        self.retired = set()
        self.draft = ()
        self.options = []
        self.complete = None
        self.advice = None
        self.power_advice = None
        self.placement_advice = None
        self.placement_thinking = False
        self.placement_started = 0
        self.power_target = 1
        self.thinking = False
        self.thinking_started = 0
        self.budget_used = 0
        self.positions = [-1] * 4
        preview = os.environ.get("SANTONI_PREVIEW_LABEL")
        self.setWindowTitle("Santoni — Saisie" + (f" [{preview}]" if preview else ""))
        self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self.setMinimumWidth(350)
        self.resize(390, 700)
        self.setStyleSheet("""
            QWidget { font-size: 11px; }
            QPushButton { padding: 3px 6px; }
            QPushButton[chip="true"] { border-radius: 10px; min-width: 22px; min-height: 18px; }
            QGroupBox { margin-top: 7px; padding-top: 6px; }
        """)
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
            box.setChecked(False)
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
        self.placement_order = QLabel()
        self.placement_order.setWordWrap(True)
        gl.addWidget(self.placement_order)
        budget_row = QHBoxLayout()
        budget_row.addWidget(QLabel("Réflexion sur le placement"))
        self.placement_budget = QDoubleSpinBox()
        self.placement_budget.setRange(1, 600)
        self.placement_budget.setValue(10)
        self.placement_budget.setSuffix(" s")
        self.placement_budget.valueChanged.connect(self.invalidate_placement)
        budget_row.addWidget(self.placement_budget)
        gl.addLayout(budget_row)
        self.placement_button = button("Suggérer le placement du robot", self.recommend_placement)
        gl.addWidget(self.placement_button)
        self.placement_status = QLabel("Choisissez les deux pouvoirs avant de demander un placement.")
        self.placement_status.setWordWrap(True)
        gl.addWidget(self.placement_status)
        self.apply_placement_button = button("Appliquer ces positions au robot", self.apply_placement, True)
        self.apply_placement_button.setEnabled(False)
        gl.addWidget(self.apply_placement_button)
        layout.addWidget(group)
        self.start_button = button("Démarrer la partie", self.start_game, True)
        layout.addWidget(self.start_button)
        layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        self.tabs.addTab(scroll, "Configuration")

    def _game(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.turn_label = QLabel("Aucune partie en cours.")
        self.turn_label.setWordWrap(True)
        layout.addWidget(self.turn_label)
        self.board = Board()
        self.board.table.setMinimumHeight(185)
        self.board.table.setMaximumHeight(225)
        for row_index in range(5):
            self.board.table.setRowHeight(row_index, 32)
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
        self.invalidate_placement()
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
        if hasattr(self, "placement_order"):
            powers = (self.mine.currentData(), self.robot.currentData())
            if None not in powers and powers[1] >= 0:
                owner = placement_first(powers, self.first.currentIndex())
                text = "Moi → Robot" if owner == 0 else "Robot → Moi"
                self.placement_order.setText(f"Ordre de placement : {text}" + (" (Bia)" if 13 in powers else ""))
                self.placement_button.setEnabled(self.session is None and not self.loading and not self.load_failed
                                                 and not self.placement_thinking)
            else:
                self.placement_order.setText("Ordre de placement : choisir d'abord le pouvoir du robot.")
                self.placement_button.setEnabled(False)

    def place(self, cell):
        self.invalidate_placement()
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
                        "placement_seconds": self.placement_budget.value(),
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
        self.invalidate_placement()
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

    def invalidate_placement(self, *_):
        if not hasattr(self, "analysis"):
            return
        self.placement_generation += 1
        self.cancel("placement")
        self.placement_thinking = False
        self.placement_advice = None
        self.analysis.use_placement.setEnabled(False)
        self.apply_placement_button.setEnabled(False)
        self.placement_status.setText("Placement modifié : demander une nouvelle suggestion.")
        self.analysis.placement_text.setText("Demandez une suggestion après avoir choisi les pouvoirs et le premier joueur.")
        self.analysis.placement_board.render_position(Position(workers=((-1, -1), (-1, -1))))
        self.render_placement()

    def recommend_placement(self):
        if self.session is not None or self.loading or self.load_failed:
            return
        powers = (self.mine.currentData(), self.robot.currentData())
        if None in powers or powers[1] < 0:
            self.fail("Choisissez d'abord les deux pouvoirs, y compris celui du robot.")
            return
        first = placement_first(powers, self.first.currentIndex())
        opponent = tuple(self.positions[:2]) if first == 0 else None
        if opponent and (any(c < 0 for c in opponent) or len(set(opponent)) != 2):
            self.fail("Vous vous placez en premier : saisissez vos deux bâtisseurs avant de demander le placement du robot.")
            return
        self.invalidate_placement()
        self.clear_error()
        self.placement_thinking = True
        self.placement_started = time.monotonic()
        budget = self.placement_budget.value()
        self.placement_status.setText("Recherche du placement du robot…")
        self.placement_button.setEnabled(False)
        self.analysis.placement_text.setText("Recherche en cours ; les coordonnées restent saisissables.")
        self.analysis.tabs.setCurrentIndex(2)
        self.launch("placement", (powers, self.first.currentIndex(), budget, opponent,
                                  self.placement_started + budget), self.placement_generation,
                    self.placement_ready, self.placement_progress, self.placement_error)

    def placement_progress(self, advice: PlacementAdvice, token):
        if token != self.placement_generation or self.session is not None or self.closing:
            return
        self.placement_advice = advice
        cells = advice.robot_cells
        pair = f"Robot 1 : {coord(cells[0])} · Robot 2 : {coord(cells[1])}"
        order = "Le robot répond à votre placement." if advice.first_to_place == 0 else (
            "Le robot se place en premier ; votre placement futur n'est pas utilisé.")
        score = "…" if advice.score is None else f"{advice.score:+.1f}"
        self.placement_status.setText(f"{pair}\n{'Recherche en cours' if self.placement_thinking else 'Suggestion prête'}")
        self.analysis.placement_text.setText(f"{pair}\n{order}\n{advice.status}\n"
                                             f"Estimation {score} · {advice.elapsed:.2f} s")
        opponent = advice.opponent_cells or (-1, -1)
        preview = Position(workers=(opponent, cells), powers=(self.mine.currentData(), self.robot.currentData()),
                           player=self.first.currentIndex())
        self.analysis.placement_board.render_position(preview, cells)

    def placement_ready(self, advice, token):
        if token != self.placement_generation or self.session is not None or self.closing:
            return
        self.placement_thinking = False
        self.placement_progress(advice, token)
        self.analysis.use_placement.setEnabled(True)
        self.apply_placement_button.setEnabled(True)
        self.render_placement()

    def placement_error(self, message, token):
        if token != self.placement_generation or self.closing:
            return
        self.placement_thinking = False
        self.placement_advice = None
        self.placement_status.setText("Placement non calculé : " + message)
        self.fail(message)
        self.render_placement()

    def apply_placement(self):
        if self.session is not None or self.placement_thinking or self.placement_advice is None:
            return
        # Only the robot's fields change. Human pieces remain theirs to place.
        self.positions[2:] = self.placement_advice.robot_cells
        self.clear_error()
        conflicts = [i for i in (0, 1) if self.positions[i] in self.positions[2:]]
        if conflicts:
            self.fail("Le robot se place en premier : modifiez vos pions qui occupent une case maintenant choisie par le robot.")
            self.slot.setCurrentIndex(conflicts[0])
        elif self.placement_advice.first_to_place == 1:
            self.slot.setCurrentIndex(0)
        self.render_placement()

    def tick(self):
        if self.thinking:
            elapsed = time.monotonic() - self.thinking_started
            self.analysis.status.setText(f"Réflexion : {min(elapsed, self.budget_used):.1f} / {self.budget_used:g} s — saisie disponible")
        if self.placement_thinking:
            elapsed = time.monotonic() - self.placement_started
            cells = (f" · proposition {coord(self.placement_advice.robot_cells[0])}, {coord(self.placement_advice.robot_cells[1])}"
                     if self.placement_advice else "")
            self.placement_status.setText(f"Placement : {min(elapsed, self.placement_budget.value()):.1f} / "
                                          f"{self.placement_budget.value():g} s{cells}")

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

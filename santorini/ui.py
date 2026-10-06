from __future__ import annotations

import multiprocessing
import os
import sys
import time
from dataclasses import replace

from PySide6.QtCore import QEvent, Qt, QTimer, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QInputDialog, QApplication, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QMainWindow, QHeaderView, QPushButton, QTabWidget, QTableWidget, QScrollArea,
    QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget, QAbstractItemView,
)

from .engine import Action, Position, _apply, coord, validate_setup, preview_position, resolve_plan
from .extra import Extra, initial_count, DIRECTIONS
from .powers import POWERS, incompatible, ARENA_EXCLUDED
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
QPushButton[chip="true"]:disabled { color: #939ba5; background: #f0f2f4; border-color: #d5d9df; }
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


class ButtonChoices(QWidget):
    """Exclusive, keyboard-accessible choices with explicit legal availability."""
    currentIndexChanged = Signal(int)

    def __init__(self, columns=None):
        super().__init__()
        self.columns=columns
        self.row = QGridLayout(self) if columns else QHBoxLayout(self)
        self.row.setContentsMargins(0, 0, 0, 0)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons, self.values = [], []
        self.index = -1
        self.group.idClicked.connect(self.setCurrentIndex)

    def clear(self):
        for chip in self.buttons:
            self.group.removeButton(chip)
            self.row.removeWidget(chip)
            chip.hide()
            chip.deleteLater()
        self.buttons, self.values, self.index = [], [], -1

    def addItem(self, text, value, enabled=True, description=None):
        chip = QPushButton(text)
        chip.setCheckable(True)
        chip.setProperty("chip", True)
        chip.setEnabled(enabled)
        chip.setAccessibleName(description or text)
        chip.setToolTip(description or text)
        index = len(self.buttons)
        self.group.addButton(chip, index)
        if self.columns:self.row.addWidget(chip,index//self.columns,index%self.columns)
        else:self.row.addWidget(chip)
        self.buttons.append(chip)
        self.values.append(value)
        if self.index < 0 and enabled:
            self.setCurrentIndex(index)

    def setCurrentIndex(self, index):
        if not 0 <= index < len(self.buttons) or not self.buttons[index].isEnabled():
            return
        changed = self.index != index
        self.index = index
        self.buttons[index].setChecked(True)
        if changed:
            self.currentIndexChanged.emit(index)

    def currentData(self):
        return self.values[self.index] if self.index >= 0 else None


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
                text += f"\n{'M' if p == 0 else getattr(self, 'opponent_tag', 'R')}{i + 1}"
            markers=[]
            for owner in (0,1):
                if s.extra.coins[owner]&(1<<cell):markers.append("Pièce")
                if cell in s.extra.whirlpools[owner]:markers.append(f"Tourbillon {owner+1}")
                if s.extra.talus[owner]==cell:markers.append("Talus")
                if s.extra.abyss[owner]==cell:markers.append("Abysse privé")
                zone=s.extra.fate[owner]
                if zone>=0 and cell%5 in (zone%5,zone%5+1) and cell//5 in (zone//5,zone//5+1):markers.append("Destin privé")
            if markers:text+="\n"+" · ".join(markers)
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
        self.correction_original = None
        self.correction_index = None
        self.correction_busy = False
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
        self.setup_counts=(2,2)
        self.setup_extra=Extra()
        self.setup_pair=None
        self.setup_thinking=False
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
        history_page = QWidget()
        history_layout = QVBoxLayout(history_page)
        self.history_help = QLabel("Sélectionnez un tour complet, puis Corriger. Les tours suivants seront rejoués et vérifiés.")
        self.history_help.setWordWrap(True)
        history_layout.addWidget(self.history_help)
        self.history = QListWidget()
        self.history.currentRowChanged.connect(lambda *_: self.update_correction_controls())
        history_layout.addWidget(self.history)
        self.correct_button = button("Corriger le tour sélectionné", self.begin_correction)
        history_layout.addWidget(self.correct_button)
        self.tabs.addTab(history_page, "Historique")
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
        self.mode_changed()
        self.refresh()
        if restore:
            self.footer.setText("Recherche d'une partie sauvegardée…")
            QTimer.singleShot(0, self.restore)

    def _configuration(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.robot_game = QCheckBox("Partie contre robot")
        self.robot_game.setChecked(False)
        self.robot_game.toggled.connect(self.mode_changed)
        layout.addWidget(self.robot_game)
        self.arena_mode = QCheckBox("Mode Arène — information parfaite")
        self.arena_mode.setToolTip("Sans Toison d’or, hasard ni information cachée : Chaos, Hecate, Moerae et Tartarus sont exclus. Choisissez les cartes disponibles, puis le pouvoir de chaque joueur.")
        self.arena_mode.toggled.connect(self.arena_changed)
        self.pre_arena_families = None
        layout.addWidget(self.arena_mode)
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
        hint = QLabel("Choisissez une famille, puis cochez les deux cartes disponibles pour la partie. Les pouvoirs grisés restent à implémenter.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        form = QFormLayout()
        self.mine, self.robot = QComboBox(), QComboBox()
        self.mine.currentIndexChanged.connect(self.configuration_changed)
        self.robot.currentIndexChanged.connect(self.configuration_changed)
        form.addRow("Mon pouvoir", self.mine)
        self.opponent_power_label = QLabel("Pouvoir de l’adversaire")
        form.addRow(self.opponent_power_label, self.robot)
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
        self.private_turn=QCheckBox("Vue privée de l’adversaire")
        self.private_turn.setToolTip("Passer l’écran à l’adversaire pour saisir ses déplacements secrets.")
        self.private_turn.toggled.connect(self.private_view_changed)
        self.private_turn.hide()
        layout.addWidget(self.private_turn)
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
        self.action_kind = ButtonChoices()
        self.actor = ButtonChoices(columns=2)
        self.action_kind.currentIndexChanged.connect(self.update_actors)
        self.actor.currentIndexChanged.connect(self.update_coordinates)
        row.addWidget(self.action_kind)
        row.addWidget(self.actor)
        layout.addLayout(row)
        self.event_choices=QWidget()
        QGridLayout(self.event_choices)
        self.event_choices.hide()
        self.coordinates = Coordinates(self.coordinate_action)
        layout.addWidget(self.event_choices)
        layout.addWidget(self.coordinates)
        self.special = button("Utiliser le pouvoir de héros", self.activate_hero)
        layout.addWidget(self.special)
        self.draft_text = QLabel()
        self.draft_text.setWordWrap(True)
        layout.addWidget(self.draft_text)
        self.cancel_correction_button = button("Annuler la correction", self.cancel_correction)
        self.cancel_correction_button.hide()
        layout.addWidget(self.cancel_correction_button)
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

    def robot_mode(self):
        session = self.correction_original or self.session
        # Older saved games predate the choice and were played against the robot.
        return session.settings.get("robot_mode", True) if session else self.robot_game.isChecked()

    def opponent_name(self):
        return "Robot" if self.robot_mode() else "Adversaire"

    def display_turn(self, text):
        return text if self.robot_mode() else text.replace("Robot", "Adversaire")

    def mode_changed(self, *_):
        if not hasattr(self, "analysis"):
            return
        name = self.opponent_name()
        self.opponent_power_label.setText(f"Pouvoir {'du robot' if self.robot_mode() else 'de l’adversaire'}")
        self.first.setItemText(1, name)
        for i in range(self.slot.count()):
            owner,worker=self.setup_slot(i)
            self.slot.setItemText(i,f"{'Moi' if owner==0 else name} {worker+1}")
        self.choose_for.setItemText(0, name)
        if not self.robot_mode():
            self.choose_for.setCurrentIndex(1)
        self.choose_for.setEnabled(self.robot_mode())
        self.analysis.tabs.setTabEnabled(2, self.robot_mode())
        self.result_choice.setItemText(1, f"{name} gagnant")
        self.placement_button.setVisible(self.robot_mode())
        self.apply_placement_button.setVisible(self.robot_mode())
        self.placement_status.setVisible(self.robot_mode())
        self.placement_budget.setEnabled(self.robot_mode())
        self.available_changed()

    def arena_changed(self, checked):
        if checked:
            self.pre_arena_families = {key: box.isChecked() for key, box in self.families.items()}
        for key, box in self.families.items():
            box.blockSignals(True)
            box.setChecked(True if checked else (self.pre_arena_families or {}).get(key, False))
            box.setEnabled(not checked)
            box.blockSignals(False)
        self.refresh_configuration()

    def checked_powers(self):
        return [self.available.item(i).data(Qt.UserRole) for i in range(self.available.count())
                if self.available.item(i).checkState() == Qt.Checked
                and (not self.arena_mode.isChecked() or self.available.item(i).data(Qt.UserRole) not in ARENA_EXCLUDED)]

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
            if self.arena_mode.isChecked() and p.number in ARENA_EXCLUDED:
                continue
            item = QListWidgetItem(f"{p.number}. {p.name}" + (" — à venir" if not p.supported else ""))
            item.setData(Qt.UserRole, p.number)
            item.setToolTip(p.description)
            if p.supported:
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(previous_checks.get(p.number, Qt.Unchecked))
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
            if i == 1 and self.robot_mode():
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

    def setup_slot(self,index):
        split=getattr(self,"setup_counts",(2,2))[0]
        return (0,index) if index<split else (1,index-split)

    def resize_setup(self,counts):
        if counts==self.setup_counts:return
        old=self.setup_counts[0]
        groups=(self.positions[:old],self.positions[old:])
        self.positions=[v for owner,count in enumerate(counts) for v in (groups[owner]+[-1]*count)[:count]]
        self.setup_counts=counts
        self.slot.blockSignals(True);self.slot.clear()
        self.slot.addItems([f"{'Moi' if owner==0 else self.opponent_name()} {w+1}" for owner,count in enumerate(counts) for w in range(count)])
        self.slot.blockSignals(False)

    def setup_parameters(self,powers,workers,public_only=False):
        extra=self.setup_extra
        labels={6:"↙ Sud-ouest",7:"↓ Sud",8:"↘ Sud-est",11:"← Ouest",13:"→ Est",16:"↖ Nord-ouest",17:"↑ Nord",18:"↗ Nord-est"}
        for owner,power in enumerate(powers):
            name="Moi" if owner==0 else self.opponent_name()
            def choose(title,cells,names=None):
                values=[names[c] if names else coord(c) for c in cells]
                text,ok=QInputDialog.getItem(self,title,name,values,0,False)
                return cells[values.index(text)] if ok else None
            if power in (31,42):
                if power==31 and extra.wind in DIRECTIONS or power==42 and extra.siren[owner] in DIRECTIONS:continue
                direction=choose("Direction initiale du vent" if power==31 else "Direction du chant",list(DIRECTIONS),labels)
                if direction is None:return None
                if power==31:extra=replace(extra,wind=direction)
                else:
                    dirs=list(extra.siren);dirs[owner]=direction;extra=replace(extra,siren=tuple(dirs))
            elif power==14:
                if extra.chaos[owner]:continue
                cards=[n for n in range(1,11) if n!=powers[1-owner]]
                card=choose("Première carte effectivement tirée par Chaos",cards,{n:POWERS[n].name for n in cards})
                if card is None:return None
                revealed=list(extra.chaos);revealed[owner]=card;extra=replace(extra,chaos=tuple(revealed))
            elif power in (40,43) and not public_only:
                cells=[c for c in range(20) if c%5!=4] if power==40 else [c for c in range(25) if all(c not in ws for ws in workers)]
                if owner==1 and self.robot_mode():
                    cell=cells[0]
                else:
                    cell=choose("Zone secrète : coin inférieur gauche" if power==40 else "Case secrète de l’abysse",cells)
                    if cell is None:return None
                values=list(extra.fate if power==40 else extra.abyss);values[owner]=cell
                extra=replace(extra,**{ "fate" if power==40 else "abyss":tuple(values)})
        self.setup_extra=extra
        return extra

    def visible_position(self,pos):
        if pos.powers[1]==39 and not (hasattr(self,"private_turn") and self.private_turn.isChecked() and pos.player==1 and not self.robot_mode()):
            pos=replace(pos,workers=(pos.workers[0],(-1,)*len(pos.workers[1])))
        extra=replace(pos.extra,abyss=(pos.extra.abyss[0],-1),fate=(pos.extra.fate[0],-1))
        return replace(pos,extra=extra)

    def private_view_changed(self,*_):
        self.refresh();self.request_actions()

    def public_highlights(self,actions,pos):
        return {c for a in actions for c in (() if a.kind in ("sing","hidden_move","swap_hidden") or a.kind=="adonis" and a.source<0 else (a.target,) if pos.powers[1]==39 and a.player==1 and a.kind in ("build","dome","hidden_build") else () if pos.powers[1]==39 and a.player==1 else (a.source,a.target)) if c>=0}

    def public_description(self,turn,before):
        if before.powers[1]!=39:return self.display_turn(turn.description(before))
        return " ; ".join(self.display_turn(a.label()) for a in turn.actions if not (a.player==1 and a.kind in ("move","force","place"))) or "Déplacement secret"

    def configuration_changed(self, *_):
        if not hasattr(self, "analysis"):
            return
        self.cancel("secret_setup")
        self.setup_thinking=False
        if hasattr(self,"start_button"):self.start_button.setEnabled(self.session is None and not self.loading and not self.load_failed)
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
        if (mine,robot)!=self.setup_pair:
            self.setup_extra=Extra();self.setup_pair=(mine,robot)
        self.resize_setup((initial_count(mine),initial_count(robot)))
        self.power_detail.setText(f"Moi : {POWERS[mine].description}\n{self.opponent_name()} : {POWERS[robot].description if robot >= 0 else 'Pouvoir à choisir.'}")
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
            first = placement_first((mine,robot),self.first.currentIndex())
            self.slot.setCurrentIndex(0 if first==0 else self.setup_counts[0])
        self.render_placement()

    def render_placement(self, *_):
        if not hasattr(self, "placement"):
            return
        self.placement_label.setText(" · ".join(f"{'M' if owner==0 else 'R' if self.robot_mode() else 'A'}{worker+1} : {'secret' if owner==1 and self.robot.currentData()==39 else coord(v) if v>=0 else '…'}" for i,v in enumerate(self.positions) for owner,worker in [self.setup_slot(i)]))
        selected = self.slot.currentIndex()
        taken = {v for i, v in enumerate(self.positions) if i != selected and v >= 0}
        self.placement.set_allowed(set(range(25)) - taken)
        if hasattr(self, "placement_order"):
            powers = (self.mine.currentData(), self.robot.currentData())
            if None not in powers and powers[1] >= 0:
                owner = placement_first(powers, self.first.currentIndex())
                text = f"Moi → {self.opponent_name()}" if owner == 0 else f"{self.opponent_name()} → Moi"
                self.placement_order.setText(f"Ordre de placement : {text}" + (" (Bia)" if 13 in powers else ""))
                self.placement_button.setEnabled(self.session is None and not self.loading and not self.load_failed
                                                 and not self.placement_thinking and self.robot_mode())
            else:
                self.placement_order.setText(f"Ordre de placement : choisir d’abord le pouvoir de {self.opponent_name().lower()}.")
                self.placement_button.setEnabled(False)

    def place(self, cell):
        self.invalidate_placement()
        index = self.slot.currentIndex()
        if self.setup_slot(index)[0]==1 and self.robot.currentData()==39 and self.robot_mode():
            self.fail("Hecate se place secrètement : utilisez la suggestion de placement du robot.")
            return
        self.positions[index] = cell
        mine, robot = self.mine.currentData(), self.robot.currentData()
        first=placement_first((mine,robot),self.first.currentIndex())
        groups=(list(range(self.setup_counts[0])),list(range(self.setup_counts[0],sum(self.setup_counts))))
        order=groups[first]+groups[1-first]
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
            self.arena_mode.setChecked(session.settings.get("rules_mode", "custom") == "arena")
            self.robot_game.setChecked(session.settings.get("robot_mode", True))
            self.mode_changed()
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
        if self.loading or self.load_failed or self.session is not None or self.setup_thinking:
            return
        try:
            robot = self.robot.currentData()
            if robot < 0:
                raise ValueError("Choisissez le pouvoir du robot ou lancez sa comparaison.")
            if self.arena_mode.isChecked() and (self.mine.currentData() == 0 or robot == 0):
                raise ValueError("En mode Arena, choisissez un pouvoir pour chaque joueur.")
            powers=(self.mine.currentData(),robot)
            if self.arena_mode.isChecked() and ARENA_EXCLUDED.intersection(powers):
                raise ValueError("Le mode Arène exclut les pouvoirs aléatoires ou à information cachée.")
            split=self.setup_counts[0]
            workers=(tuple(self.positions[:split]),tuple(self.positions[split:]))
            if any(c<0 for ws in workers for c in ws):raise ValueError("Placez tous les bâtisseurs avant de démarrer.")
            extra=self.setup_parameters(powers,workers)
            if extra is None:return
            pos = validate_setup(workers,powers,self.first.currentIndex(),extra)
            settings = {"rules_mode": "arena" if self.arena_mode.isChecked() else "custom",
                        "robot_mode": self.robot_game.isChecked(), "move_seconds": self.move_budget.value(), "power_seconds": self.power_budget.value(),
                        "placement_seconds": self.placement_budget.value(),
                        "available_powers": self.checked_powers(),
                        "families": {k: b.isChecked() for k, b in self.families.items()}}
            if self.robot_mode() and robot in (40,43):
                self.setup_thinking=True;self.setup_pending_settings=settings
                budget=self.placement_budget.value()
                self.footer.setText("Recherche du lieu secret du robot…")
                self.start_button.setEnabled(False)
                self.launch("secret_setup",(pos,budget,time.monotonic()+budget),self.power_generation,self.secret_setup_ready,error=self.secret_setup_error)
                return
            session = Session(pos, settings=settings)
            self.store.save(session)
        except (ValueError, OSError) as exc:
            self.fail(exc)
            return
        self.install_session(session)

    def secret_setup_ready(self,pos,token):
        if token!=self.power_generation or self.session is not None:return
        self.setup_thinking=False
        try:
            session=Session(pos,settings=self.setup_pending_settings);self.store.save(session)
        except (ValueError,OSError) as exc:self.secret_setup_error(str(exc),token);return
        self.install_session(session)

    def secret_setup_error(self,message,token):
        if token==self.power_generation:
            self.setup_thinking=False;self.fail(message);self.refresh()

    def install_session(self,session):
        self.clear_error()
        self.power_generation += 1
        self.cancel("power")
        self.invalidate_placement()
        self.session = session
        self.mode_changed()
        self.game_budget.setValue(self.move_budget.value())
        self.tabs.setCurrentIndex(1)
        self.position_changed()

    def refresh(self):
        active = self.session is not None
        editing = self.correction_original is not None
        playing = active and self.session.result is None and not self.correction_busy
        self.cancel_correction_button.setVisible(editing)
        self.commit_button.setText("Enregistrer le correctif" if editing else "Valider les actions — intervention de Gaea" if self.complete and self.complete.after.extra.event==2 else "Valider le tour / terminer")
        self.update_correction_controls()
        self.robot_game.setEnabled(not active)
        self.arena_mode.setEnabled(not active)
        self.tabs.setTabEnabled(0, not active)
        self.start_button.setEnabled(not active and not self.loading and not self.load_failed and not self.setup_thinking)
        self.undo_button.setEnabled(active and bool(self.session.history) and not editing)
        self.rethink.setEnabled(playing and not editing and (self.robot_mode() or self.session.position.player == 0))
        self.end_button.setEnabled(playing and not editing)
        self.reset_button.setEnabled(active and not editing)
        self.back.setEnabled(bool(self.draft) and playing)
        self.commit_button.setEnabled(self.complete is not None and playing)
        self.analysis.apply.setEnabled(playing and not editing and self.advice is not None and self.advice.turn is not None and not self.thinking)
        if not active:
            self.board.render_position(Position(workers=((-1, -1), (-1, -1))))
            self.turn_label.setText("Aucune partie en cours.")
            self.coordinates.set_allowed(set())
            self.special.hide()
            return
        pos = self.session.position
        self.private_turn.setVisible(pos.powers[1]==39 and pos.player==1 and not self.robot_mode())
        preview = pos
        preview=preview_position(pos,self.draft)
        shown = self.draft or (self.advice.turn.actions if self.advice and self.advice.turn else ())
        self.board.table.opponent_tag = "R" if self.robot_mode() else "A"
        self.board.render_position(self.visible_position(preview),self.public_highlights(shown,pos))
        powers = f"Moi : {POWERS[pos.powers[0]].name} · {self.opponent_name()} : {POWERS[pos.powers[1]].name}"
        resources=[]
        for owner,power in enumerate(pos.powers):
            if power==14:resources.append(f"Chaos : {POWERS[pos.extra.chaos[owner]].name}")
            if power==25:resources.append(f"Morpheus : {pos.extra.materials[owner]} matériau(x)")
            if power==35:resources.append(f"Gaea : {pos.extra.reserves[owner]} réserve(s)")
        if pos.extra.wind>=0:resources.append(f"Vent : {coord(pos.extra.wind)} depuis C3")
        for owner,direction in enumerate(pos.extra.siren):
            if direction>=0:resources.append(f"Siren : {coord(direction)} depuis C3")
        if resources:powers+="\n"+" · ".join(resources)
        if pos.extra.dion_owner>=0:powers+="\nTour supplémentaire de Dionysus : contrôler un bâtisseur adverse ; aucune victoire pendant ce tour."
        if pos.extra.event==2:powers+="\nIntervention de Gaea ; le tour interrompu reprendra ensuite."
        if pos.extra.event==1:powers+="\nChaos : indiquez la carte réellement tirée."
        if pos.extra.event==3:powers+="\nDionysus : choisir de jouer ou de renoncer au tour supplémentaire."
        if self.session.settings.get("rules_mode") == "arena":
            powers = "Arena · " + powers
        if self.session.result:
            outcome = self.session.result
            winner = outcome.get("winner")
            self.turn_label.setText(f"{'Partie interrompue' if winner is None else ('Moi' if winner == 0 else self.opponent_name()) + ' gagne'} — {outcome['reason']}\n{powers}")
        else:
            self.turn_label.setText(f"Tour {len(self.session.history) + 1} — {'À moi' if pos.player == 0 else 'Au robot' if self.robot_mode() else 'À l’adversaire'}\n{powers}")
        if editing:
            self.turn_label.setText(f"Correction du tour {self.correction_index + 1} — "
                                    f"{'Moi' if pos.player == 0 else self.opponent_name()}\n{powers}")
        self.draft_text.setText("\n".join(self.display_turn(a.label()) for a in self.draft if not (pos.powers[1]==39 and a.player==1 and a.kind in ("move","force","place"))) or "Aucune action saisie.")
        history_session = self.correction_original or self.session
        selected = self.history.currentRow()
        self.history.blockSignals(True)
        self.history.clear()
        for i, t in enumerate(history_session.history):
            before = history_session.initial if i == 0 else history_session.history[i - 1].after
            self.history.addItem(f"{i + 1}. {'Moi' if before.player == 0 else self.opponent_name()}\n{self.public_description(t,before)}\n{self.display_turn(t.power_summary(before))}")
        if history_session.result:
            self.history.addItem(f"Résultat : {history_session.result}")
        self.history.setCurrentRow(selected)
        self.history.blockSignals(False)
        self.update_correction_controls()

    def position_changed(self):
        self.private_turn.blockSignals(True);self.private_turn.setChecked(False);self.private_turn.blockSignals(False)
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
        if self.correction_original is not None:
            self.analysis.status.setText("Correction en cours — calcul suspendu jusqu'à l'enregistrement.")
            self.analysis.move_label.setText("Saisissez de nouveau le tour complet à corriger.")
        elif (self.session and self.session.result is None
              and self.session.position.player == (1 if self.robot_mode() else 0)):
            self.run_search()
        elif self.session and self.session.result is None:
            if self.robot_mode():
                self.analysis.move_label.setText("À vous de jouer. Recalculer permet de demander un conseil.")
                self.analysis.status.setText("Le robot réfléchira après la validation de votre tour.")
            else:
                self.analysis.move_label.setText("À l’adversaire de jouer : saisissez son tour complet.")
                self.analysis.status.setText("Aucun conseil adverse. Le moteur conseillera votre prochain tour.")
        else:
            self.analysis.status.setText("Partie terminée et sauvegardée.")

    def request_actions(self):
        self.input_generation += 1
        self.options, self.complete = [], None
        self.event_choices.hide()
        self.coordinates.set_allowed(set())
        self.action_kind.clear()
        self.actor.clear()
        self.special.hide()
        self.refresh()
        if not self.session or self.session.result or self.correction_busy:
            return
        self.prompt.setText("Vérification des actions légales… La saisie reste indépendante de l'analyse.")
        if self.session.position.player==1 and self.session.position.powers[1]==39 and (self.robot_mode() or not self.private_turn.isChecked()):
            self.prompt.setText("Tour secret du robot : seul son résultat public sera affiché." if self.robot_mode() else "Passer l’écran à l’adversaire et activer sa vue privée pour saisir son tour secret.")
            return
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
            if self.correction_original is None:
                self.finish_loss()
            return
        kinds = sorted({a.kind for a in self.options if a.kind != "activate"})
        names = {"move": "Déplacement", "build": "Construction", "dome": "Dôme", "remove": "Retirer un bloc",
                 "kill": "Éliminer", "force": "Déplacement forcé", "place": "Nouveau bâtisseur", "adonis": "Cible d'Adonis", "wind":"Direction du vent", "whirlpool":"Placer un tourbillon", "talus":"Déplacer Talus", "swap":"Échanger avec un adversaire", "draw":"Carte tirée par Chaos", "decline":"Renoncer à l’intervention", "extra_turn":"Accepter le tour supplémentaire", "sing":"Chant sur un bâtisseur secret", "hidden_move":"Direction du déplacement caché", "hidden_build":"Construction avec un bâtisseur caché", "swap_hidden":"Échanger avec un bâtisseur secret", "probe_kill":"Désigner une case à éliminer", "probe_remove":"Retirer sous un bâtisseur secret", "probe_force":"Forcer un bâtisseur secret vers un coin", "probe_charon":"Forcer derrière mon bâtisseur"}
        self.action_kind.blockSignals(True)
        self.action_kind.clear()
        symbols = {"move": "↗", "build": "▦", "dome": "◉", "remove": "−",
                   "kill": "×", "force": "⇢", "place": "+", "adonis": "◎", "wind":"➤", "whirlpool":"↻", "talus":"◆", "swap":"⇄", "draw":"?", "decline":"✓", "extra_turn":"⏩", "sing":"♪", "hidden_move":"↗?", "hidden_build":"▦?", "swap_hidden":"⇄?", "probe_kill":"×?", "probe_remove":"−?", "probe_force":"⇢?", "probe_charon":"⇢?"}
        for kind in ["move", "build"] + [k for k in kinds if k not in ("move", "build")]:
            self.action_kind.addItem(symbols.get(kind, kind), kind, enabled=kind in kinds,
                                     description=names.get(kind, kind))
        self.action_kind.blockSignals(False)
        self.special.setVisible(any(a.kind == "activate" for a in self.options))
        self.prompt.setText("Validez les actions pour laisser Gaea intervenir. Le tour reprendra après sa réponse." if self.complete and self.complete.after.extra.event==2 else "Tour complet : validez, ou poursuivez avec une action facultative." if self.complete else
                            "Choisissez l'action et le bâtisseur, puis la lettre et le nombre de la case cible.")
        self.update_actors()
        self.refresh()

    def update_actors(self, *_):
        previous = self.actor.currentData()
        self.actor.blockSignals(True)
        self.actor.clear()
        kind = self.action_kind.currentData()
        available = {(a.player, a.worker) for a in self.options if a.kind == kind}
        preview = self.session.position if self.session else None
        if preview:preview=preview_position(preview,self.draft)
        players = ({p for p, w in available} or {preview.player}) if preview else set()
        pairs = {(p, w) for p in players for w in range(len(preview.workers[p]))
                 if preview and preview.workers[p][w] >= 0} | available
        for player, worker in sorted(pairs):
            source = next((a.source for a in self.options
                           if a.kind == kind and (a.player, a.worker) == (player, worker)),
                          preview.workers[player][worker]
                          if preview and 0 <= worker < len(preview.workers[player]) else -1)
            if preview and preview.powers[player]==39 and player!=0 and not self.private_turn.isChecked():source=-1
            text = "Cases adjacentes" if worker < 0 else f"{'Moi' if player == 0 else self.opponent_name()} {worker + 1}"
            if source >= 0:
                text += f" · {coord(source)}"
            self.actor.addItem(text, (player, worker), enabled=(player, worker) in available)
        if previous in self.actor.values:
            self.actor.setCurrentIndex(self.actor.values.index(previous))
        self.actor.blockSignals(False)
        self.update_coordinates()

    def update_coordinates(self, *_):
        kind, actor = self.action_kind.currentData(), self.actor.currentData()
        self.special.setVisible(any(a.kind in ("activate","decline","extra_turn","sing") for a in self.options))
        special=next((a for a in self.options if a.kind==kind and (a.player,a.worker)==actor and (a.kind in ("decline","extra_turn","sing") or a.kind=="adonis" and a.source<0)),None)
        if special:self.special.setVisible(True)
        self.special.setVisible(special is not None or any(a.kind=="activate" for a in self.options))
        self.special.setText("Renoncer" if special and special.kind=="decline" else "Tour supplémentaire" if special and special.kind=="extra_turn" else "Chanter" if special and special.kind=="sing" else "Désigner ce bâtisseur" if special and special.kind=="adonis" else "Utiliser le héros")
        cells = {a.target for a in self.options if a.kind == kind and (a.player, a.worker) == actor and a.target >= 0}
        if kind in ("decline","extra_turn","sing") or special and special.kind=="adonis":cells=set()
        named = [a for a in self.options if a.kind == kind and
                 (kind == "draw" or (a.player, a.worker) == actor)] if kind in ("draw", "swap_hidden") else []
        layout = self.event_choices.layout()
        while layout.count():
            item = layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        for i, action in enumerate(named):
            label = POWERS[action.target].name if kind == "draw" else f"Adversaire {action.target + 1}"
            chip = button(label, lambda checked=False, a=action: self.enter_event(a))
            layout.addWidget(chip, i // 3, i % 3)
        self.event_choices.setVisible(bool(named))
        self.coordinates.setVisible(kind not in ("draw", "swap_hidden"))
        self.actor.setVisible(kind != "draw")
        if kind in ("draw", "swap_hidden"): cells = set()
        if kind=="hidden_move":self.prompt.setText("Direction du déplacement caché : choisir une case voisine de C3, utilisée comme repère.")
        self.coordinates.set_allowed(cells)

    def coordinate_action(self, cell):
        kind, actor = self.action_kind.currentData(), self.actor.currentData()
        action = next((a for a in self.options if a.kind == kind and (a.player, a.worker) == actor and a.target == cell), None)
        if action:
            self.draft += (action,)
            self.request_actions()

    def enter_event(self,action):
        if action not in self.options:return
        self.draft+=(action,);self.request_actions()

    def activate_hero(self):
        selected=self.action_kind.currentData()
        actor=self.actor.currentData()
        action = next((a for a in self.options if a.kind==selected and (a.player,a.worker)==actor and (a.kind in ("decline","extra_turn","sing") or a.kind=="adonis" and a.source<0)),None) or next((a for a in self.options if a.kind == "activate"), None)
        if action:
            self.draft += (action,)
            self.request_actions()

    def back_action(self):
        if self.draft:
            self.draft = self.draft[:-1]
            self.request_actions()

    def update_correction_controls(self):
        if not hasattr(self, "correct_button"):
            return
        row = self.history.currentRow()
        self.correct_button.setEnabled(self.session is not None and self.session.result is None
                                       and self.correction_original is None
                                       and 0 <= row < len(self.session.history))

    def begin_correction(self):
        row = self.history.currentRow()
        if (not self.session or self.session.result or self.correction_original is not None
                or not 0 <= row < len(self.session.history)):
            return
        self.correction_original = self.session
        self.correction_index = row
        self.session = replace(self.session, history=self.session.history[:row], result=None)
        self.clear_error()
        self.tabs.setCurrentIndex(1)
        self.position_changed()

    def cancel_correction(self):
        if self.correction_original is None:
            return
        self.cancel("correction")
        self.session = self.correction_original
        self.correction_original = None
        self.correction_index = None
        self.correction_busy = False
        self.clear_error()
        self.position_changed()

    def save_correction(self):
        if self.correction_busy or self.complete is None:
            return
        self.correction_busy = True
        self.input_generation += 1
        self.cancel("actions")
        self.coordinates.set_allowed(set())
        self.special.hide()
        self.prompt.setText("Reconstruction du plateau et vérification des tours suivants…")
        self.refresh()
        self.launch("correction", (self.correction_original, self.correction_index, self.complete.actions),
                    self.generation, self.correction_ready, error=self.correction_error)

    def correction_ready(self, corrected, token):
        if token != self.generation or self.correction_original is None or self.closing:
            return
        try:
            self.store.save(corrected)
        except OSError as exc:
            self.correction_error(f"Enregistrement impossible : {exc}", token)
            return
        self.session = corrected
        self.correction_original = None
        self.correction_index = None
        self.correction_busy = False
        self.clear_error()
        self.footer.setText("Correctif sauvegardé ; plateau et positions reconstruits.")
        self.position_changed()

    def correction_error(self, message, token):
        if token != self.generation or self.correction_original is None or self.closing:
            return
        self.correction_busy = False
        self.fail(message)
        self.request_actions()

    def commit_draft(self):
        if self.complete is not None:
            if self.correction_original is not None:
                self.save_correction()
            else:
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
        if (not self.session or self.session.result or self.correction_original is not None
                or (not self.robot_mode() and self.session.position.player != 0)):
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
        reasons = {"timeout": "Arrêt : temps imparti épuisé", "proof": "Arrêt anticipé : résultat démontré",
                   "terminal": "Arrêt : partie terminée ou aucun tour légal", "cancelled": "Calcul annulé",
                   "depth_limit": "Arrêt : profondeur maximale atteinte"}
        reason = reasons.get(analysis.stop_reason, analysis.status)
        total = time.monotonic() - self.thinking_started
        attempt = (f" · profondeur {analysis.searching_depth} inachevée" if analysis.searching_depth else
                   f" · profondeur {analysis.depth + 1} inachevée" if analysis.stop_reason == "timeout" and analysis.complete_depth else "")
        self.analysis.status.setText(f"{reason} — {total:.2f} / {self.budget_used:g} s{attempt}")
        self.refresh()

    def search_error(self, message, token):
        if token == self.generation and not self.closing:
            self.thinking = False
            self.fail(message)
            self.analysis.status.setText("Calcul interrompu. Le dernier tour légal calculé reste disponible.")
            self.refresh()

    def show_analysis(self, analysis: Analysis):
        self.advice = analysis
        self.analysis.move_label.setText((self.public_description(analysis.turn,self.session.position) + "\n" + analysis.turn.power_summary(self.session.position))
                                   if analysis.turn and self.session else analysis.status)
        proof = "Démontré" if analysis.proven else "Estimation" if analysis.score is not None else "Sans évaluation"
        score = "…" if analysis.score is None else str(analysis.score)
        self.analysis.metadata.setText(f"{proof} · score {score} pour le joueur au trait\nProfondeur {analysis.depth}"
                                       f"{' complète' if analysis.complete_depth else ' partielle'} · {analysis.nodes:,} positions · {analysis.elapsed:.2f} s\n"
                                       f"Moteur {analysis.engine}" +
                                       (f" · recherche à profondeur {analysis.searching_depth}" if analysis.searching_depth and self.thinking else ""))
        if analysis.proven:
            explanation = analysis.status
        elif analysis.turn and analysis.turn.after.extra.event==2:
            explanation="Gaea intervient immédiatement. La suite du déplacement ou de la construction dépendra de sa réponse."
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
            lines.append(f"{i + 1}. {self.public_description(turn,before) if before else turn.label()}")
            before = turn.after
        self.analysis.variation.setPlainText("\n\n".join(lines) or "Pas encore de variante calculée.")
        if self.session and not self.draft and analysis.turn:
            self.board.render_position(self.visible_position(self.session.position),self.public_highlights(analysis.turn.actions,self.session.position))

    def play_advice(self):
        if self.advice and self.advice.turn and not self.thinking and self.session and self.session.result is None:
            if self.advice.turn.after.extra.view:
                self.analysis.apply.setEnabled(False)
                self.launch("resolve",(self.session.position,self.advice.turn.actions),self.generation,self.resolved_ready,error=self.search_error)
            else:self.commit(self.advice.turn)

    def resolved_ready(self,turn,token):
        if token==self.generation and self.session and self.session.result is None:self.commit(turn)

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
        self.robot_game.setChecked(False)
        self.mode_changed()
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
        self.setup_counts=(2,2)
        self.setup_extra=Extra()
        self.setup_pair=None
        self.setup_thinking=False
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
        if self.session is not None or self.loading or self.load_failed or not self.robot_mode():
            return
        powers = (self.mine.currentData(), self.robot.currentData())
        if None in powers or powers[1] < 0:
            self.fail("Choisissez d'abord les deux pouvoirs, y compris celui du robot.")
            return
        first = placement_first(powers, self.first.currentIndex())
        opponent = tuple(self.positions[:self.setup_counts[0]]) if first == 0 else None
        if opponent and (any(c < 0 for c in opponent) or len(set(opponent)) != self.setup_counts[0]):
            self.fail("Vous vous placez en premier : saisissez vos deux bâtisseurs avant de demander le placement du robot.")
            return
        extra=self.setup_parameters(powers,(self.positions[:self.setup_counts[0]],self.positions[self.setup_counts[0]:]),public_only=True)
        if extra is None:return
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
                                  self.placement_started + budget,extra), self.placement_generation,
                    self.placement_ready, self.placement_progress, self.placement_error)

    def placement_progress(self, advice: PlacementAdvice, token):
        if token != self.placement_generation or self.session is not None or self.closing:
            return
        self.placement_advice = advice
        cells = advice.robot_cells
        pair="Placement secret du robot" if self.robot.currentData()==39 else " · ".join(f"Robot {i+1} : {coord(c)}" for i,c in enumerate(cells))
        order = "Le robot répond à votre placement." if advice.first_to_place == 0 else (
            "Le robot se place en premier ; votre placement futur n'est pas utilisé.")
        score = "…" if advice.score is None else f"{advice.score:+.1f}"
        self.placement_status.setText(f"{pair}\n{'Recherche en cours' if self.placement_thinking else 'Suggestion prête'}")
        self.analysis.placement_text.setText(f"{pair}\n{order}\n{advice.status}\n"
                                             f"Estimation {score} · {advice.elapsed:.2f} s")
        opponent = advice.opponent_cells or (-1, -1)
        preview = Position(workers=(opponent, cells), powers=(self.mine.currentData(), self.robot.currentData()),
                           player=self.first.currentIndex())
        self.analysis.placement_board.render_position(self.visible_position(preview), () if self.robot.currentData()==39 else cells)

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
        if self.session is not None or self.placement_thinking or self.placement_advice is None or not self.robot_mode():
            return
        # Only the robot's fields change. Human pieces remain theirs to place.
        split=self.setup_counts[0]
        self.positions[split:] = self.placement_advice.robot_cells
        self.clear_error()
        conflicts = [i for i in range(split) if self.positions[i] in self.positions[split:]]
        if conflicts:
            self.fail("Le robot se place en premier : modifiez vos pions qui occupent une case maintenant choisie par le robot.")
            self.slot.setCurrentIndex(conflicts[0])
        elif self.placement_advice.first_to_place == 1:
            self.slot.setCurrentIndex(0)
        self.render_placement()

    def tick(self):
        if self.thinking:
            elapsed = time.monotonic() - self.thinking_started
            depth = (f" — profondeur {self.advice.searching_depth} en cours ; dernière profondeur complète : "
                     f"{self.advice.depth if self.advice.complete_depth else 0}" if self.advice and self.advice.searching_depth else "")
            self.analysis.status.setText(f"Réflexion : {elapsed:.1f} / {self.budget_used:g} s{depth} — saisie disponible")
        if self.placement_thinking:
            elapsed = time.monotonic() - self.placement_started
            cells = (f" · proposition {coord(self.placement_advice.robot_cells[0])}, {coord(self.placement_advice.robot_cells[1])}"
                     if self.placement_advice and self.robot.currentData()!=39 else "")
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

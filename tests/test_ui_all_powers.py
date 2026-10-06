import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from dataclasses import replace
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication,QInputDialog
from santorini.engine import Position,Turn,Action
from santorini.extra import Extra,NEW_POWERS,initial_count
from santorini.native import native_next_actions
from santorini.search import Analysis
from santorini.storage import Session,Store
from santorini.ui import MainWindow
from test_ui import dispose


@pytest.fixture(scope='module')
def app():
    q=QApplication.instance() or QApplication([]);q.setQuitOnLastWindowClosed(False);return q


def select(w,power,owner=0):
    w.families['advanced'].setChecked(True)
    for i in range(w.available.count()):
        item=w.available.item(i)
        if item.data(Qt.UserRole)==power:item.setCheckState(Qt.Checked)
    combo=w.mine if owner==0 else w.robot
    combo.setCurrentIndex(combo.findData(power))


@pytest.mark.parametrize('power',sorted(NEW_POWERS))
def test_all_new_powers_can_start_and_restore(app,tmp_path,monkeypatch,power):
    w=MainWindow(Store(tmp_path),restore=False)
    monkeypatch.setattr(w,'launch',lambda *a,**k:None)
    monkeypatch.setattr(QInputDialog,'getItem',lambda parent,title,label,items,*a:(items[0],True))
    select(w,power)
    workers=[6,8]+([2] if initial_count(power)==3 else [])+[16,18]
    assert w.slot.count()==len(workers)
    w.positions=workers
    w.start_game()
    assert w.session is not None,w.error.text()
    assert w.store.load().position==w.session.position
    assert w.session.position.powers==(power,0)
    dispose(app,w)


def test_chaos_uses_named_card_buttons(app,tmp_path,monkeypatch):
    w=MainWindow(Store(tmp_path),restore=False);monkeypatch.setattr(w,'launch',lambda *a,**k:None)
    pos=Position(powers=(14,0),extra=replace(Extra(),chaos=(4,0),deck=(3,0),discard=(8,0),event=1))
    w.session=Session(pos);w.position_changed()
    w.actions_ready(native_next_actions(pos),w.input_generation)
    assert w.event_choices.layout().count()==2
    assert not w.coordinates.isVisible()
    assert w.event_choices.layout().itemAt(0).widget().text() in ('Apollo','Artemis')
    w.event_choices.layout().itemAt(0).widget().click()
    assert w.draft[0].kind=='draw'
    dispose(app,w)


def test_board_displays_public_tokens_and_masks_opponent_secrets(app,tmp_path,monkeypatch):
    w=MainWindow(Store(tmp_path),restore=False);monkeypatch.setattr(w,'launch',lambda *a,**k:None)
    pos=Position(powers=(32,39),extra=replace(Extra(),whirlpools=((7,12),(-1,-1)),talus=(2,-1)))
    w.session=Session(pos);w.refresh()
    texts=[w.board.table.item(r,c).text() for r in range(5) for c in range(5)]
    assert not any('A1' in t or 'A2' in t for t in texts)
    assert any('Tourbillon' in t for t in texts) and any('Talus' in t for t in texts)
    assert all(c<0 for c in w.visible_position(pos).workers[1])
    dispose(app,w)


def test_secret_robot_move_is_not_displayed_or_highlighted(app,tmp_path,monkeypatch):
    w=MainWindow(Store(tmp_path),restore=False);monkeypatch.setattr(w,'launch',lambda *a,**k:None)
    w.robot_game.setChecked(True)
    pos=Position(powers=(0,39),player=1);w.session=Session(pos,settings={'robot_mode':True})
    prefix=(Action('move',1,0,16,11),Action('build',1,0,11,12))
    _,turn=native_next_actions(pos,prefix)
    assert turn
    analysis=Analysis(turn,0,1,10,.01,False,True,(turn,),'Estimation')
    w.show_analysis(analysis)
    assert 'B4' not in w.analysis.move_label.text() and 'B3' not in w.analysis.move_label.text()
    assert w.public_highlights(prefix,pos)=={12}
    assert 'B4' not in w.analysis.variation.toPlainText()
    dispose(app,w)


def test_robot_secret_objective_is_computed_before_session_save(app,tmp_path,monkeypatch):
    from santorini.worker import _compute
    w=MainWindow(Store(tmp_path),restore=False);w.robot_game.setChecked(True);select(w,43,1)
    jobs=[];monkeypatch.setattr(w,'launch',lambda *a,**k:jobs.append(a))
    w.positions=[6,8,16,18];w.start_game()
    assert w.setup_thinking and w.session is None and not w.store.current.exists()
    kind,args,token,*_=jobs[0];assert kind=='secret_setup'
    pos=_compute(kind,(args[0],.02),lambda:False,lambda x:None)
    w.secret_setup_ready(pos,token)
    assert w.session and w.store.load().position==pos and not w.setup_thinking
    assert w.visible_position(pos).extra.abyss[1]==-1
    dispose(app,w)


def test_hidden_human_input_waits_for_private_view(app,tmp_path,monkeypatch):
    w=MainWindow(Store(tmp_path),restore=False)
    calls=[];monkeypatch.setattr(w,'launch',lambda *a,**k:calls.append(a))
    w.session=Session(Position(powers=(0,39),player=1),settings={'robot_mode':False})
    w.position_changed()
    assert not any(x[0]=='actions' for x in calls) and not w.options
    w.private_turn.setChecked(True)
    assert any(x[0]=='actions' for x in calls)
    dispose(app,w)


def test_nemesis_uses_named_hidden_worker_buttons(app,tmp_path,monkeypatch):
    w=MainWindow(Store(tmp_path),restore=False);monkeypatch.setattr(w,'launch',lambda *a,**k:None)
    pos=Position(workers=((0,4),(20,24)),powers=(41,39))
    w.session=Session(pos)
    w.draft=(Action('move',0,0,0,1),Action('build',0,0,1,0))
    w.actions_ready(native_next_actions(pos,w.draft),w.input_generation)
    w.action_kind.setCurrentIndex(w.action_kind.values.index('swap_hidden'))
    assert w.event_choices.layout().count()==2
    assert {w.event_choices.layout().itemAt(i).widget().text() for i in range(2)}=={'Adversaire 1','Adversaire 2'}
    assert w.coordinates.isHidden()
    assert w.public_highlights((Action('swap_hidden',0,0,1,0),),pos)==set()
    w.event_choices.layout().itemAt(0).widget().click()
    assert w.draft[-1].kind=='swap_hidden'
    dispose(app,w)


def test_arena_perfect_information_filters_random_and_hidden_powers(app,tmp_path,monkeypatch):
    from santorini.powers import ARENA_EXCLUDED,POWERS
    w=MainWindow(Store(tmp_path),restore=False)
    monkeypatch.setattr(w,'launch',lambda *a,**k:None)
    select(w,14)
    assert w.mine.currentData()==14
    w.arena_mode.setChecked(True)
    displayed={w.available.item(i).data(Qt.UserRole) for i in range(w.available.count())}
    assert displayed==set(POWERS)-{0}-ARENA_EXCLUDED
    assert len(displayed)==51 and w.mine.currentData()==0
    assert all(w.mine.findData(p)<0 and w.robot.findData(p)<0 for p in ARENA_EXCLUDED)
    for i in range(w.available.count()):
        item=w.available.item(i)
        if item.data(Qt.UserRole)==35:item.setCheckState(Qt.Checked)
    assert w.mine.findData(35)>=0
    w.arena_mode.setChecked(False)
    displayed={w.available.item(i).data(Qt.UserRole) for i in range(w.available.count())}
    assert ARENA_EXCLUDED<=displayed
    dispose(app,w)

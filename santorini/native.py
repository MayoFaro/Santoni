"""Small ctypes bridge to the dependency-free Rust engine.

Search and legality run in the computation process. The UI only uses the cheap
prefix preview. The 38 original powers retain an independent Python oracle;
the remaining cards use Rust validation and explicit rule fixtures.
"""
from __future__ import annotations

import ctypes as C
import os
import json
from .extra import Extra, requires_native
from pathlib import Path

from .engine import Action, Position, Turn, validate_turn
from .search import Analysis
from .powers import POWERS


class CAction(C.Structure):
    _fields_ = [("kind", C.c_uint8), ("player", C.c_int8), ("worker", C.c_int8),
                ("source", C.c_int8), ("target", C.c_int8)]


class CExtra(C.Structure):
    _fields_ = [("wind", C.c_int8), ("siren", C.c_int8 * 2),
        ("whirlpools", (C.c_int8 * 2) * 2), ("coins", C.c_uint32 * 2),
        ("coin_count", C.c_uint8 * 2), ("talus", C.c_int8 * 2),
        ("materials", C.c_uint32 * 2), ("reserves", C.c_uint8 * 2),
        ("chaos", C.c_uint8 * 2), ("deck", C.c_uint16 * 2), ("discard", C.c_uint16 * 2),
        ("theft_owner", C.c_int8), ("stolen", C.c_uint8), ("abyss", C.c_int8 * 2),
        ("fate", C.c_int8 * 2), ("event", C.c_uint8), ("event_owner", C.c_uint8),
        ("return_player", C.c_uint8), ("dome_cell", C.c_int8), ("dion_owner", C.c_int8),
        ("safe", C.c_uint32 * 2), ("view", C.c_uint8), ("queued", C.c_uint8), ("queued_owner", C.c_uint8), ("queued_return", C.c_uint8), ("nemesis_active", C.c_uint8), ("original", C.c_uint8 * 2)]


class CCore(C.Structure):
    _fields_ = [("heights", C.c_uint8 * 25), ("domes", C.c_uint32),
                ("workers", (C.c_int8 * 4) * 2), ("counts", C.c_uint8 * 2),
                ("powers", C.c_uint8 * 2), ("player", C.c_uint8),
                ("hero_used", C.c_uint8 * 2), ("athena_lock", C.c_uint8),
                ("adonis", C.c_int8 * 3), ("winner", C.c_int8), ("reason", C.c_uint8),
                ("extra", CExtra)]


class CResume(C.Structure):
    _fields_ = [("origin", CCore), ("actions", CAction * 128), ("length", C.c_uint32)]


class CState(C.Structure):
    _fields_ = CCore._fields_ + [("resume", CResume)]


class CTurn(C.Structure):
    _fields_ = [("actions", CAction * 128), ("length", C.c_uint32), ("after", CState)]


class CAnalysis(C.Structure):
    _fields_ = [("turns", CTurn * 3), ("length", C.c_uint32), ("score", C.c_int32),
                ("has_score", C.c_uint8), ("depth", C.c_uint32), ("nodes", C.c_uint64),
                ("elapsed", C.c_double), ("proven", C.c_uint8), ("complete", C.c_uint8), ("status", C.c_uint8)]


PUBLISH = C.CFUNCTYPE(None, C.POINTER(CAnalysis))
CANCEL = C.CFUNCTYPE(C.c_uint8)
_library = None
ACTION_KINDS = ("move", "build", "dome", "activate", "remove", "kill", "force", "place", "adonis", "wind", "whirlpool", "talus", "swap", "draw", "decline", "extra_turn", "sing", "hidden_move", "hidden_build", "swap_hidden", "probe_kill", "probe_remove", "probe_force", "probe_charon")


def encode_actions(actions):
    values=[]
    for action in actions:
        if not isinstance(action,Action) or action.kind not in ACTION_KINDS:
            raise ValueError('Action de pouvoir invalide.')
        fields=(action.player,action.worker,action.source,action.target)
        if any(type(x) is not int for x in fields) or action.player not in (0,1) or not -1<=action.worker<=3 or not -1<=action.source<=24 or not -1<=action.target<=24:
            raise ValueError('Coordonnées d’action invalides.')
        if action.kind in ('move','force','place','kill','adonis','sing','hidden_move','hidden_build','swap_hidden','probe_force','probe_charon') and (action.worker<0 or action.target<0):
            raise ValueError('Bâtisseur d’action invalide.')
        if action.kind=='swap_hidden' and not 0<=action.target<=3:
            raise ValueError('Identité adverse invalide.')
        if action.kind=='draw' and not 1<=action.target<=10:
            raise ValueError('Carte de Chaos invalide.')
        values.append(CAction(ACTION_KINDS.index(action.kind),*fields))
    if len(values)>128:raise ValueError('Tour trop long.')
    return (CAction*len(values))(*values)


def library_path():
    supplied = os.environ.get("SANTONI_ENGINE_LIB")
    if supplied:
        return Path(supplied)
    filename = "santoni_engine.dll" if os.name == "nt" else "libsantoni_engine.dylib" if __import__("sys").platform == "darwin" else "libsantoni_engine.so"
    return Path(__file__).resolve().parent.parent / "native_engine" / "target" / "release" / filename


def library():
    global _library
    if _library is None:
        candidate = C.CDLL(str(library_path()))
        if not hasattr(candidate,"santoni_state_size") or candidate.santoni_rules_version()!=3:
            raise ValueError("Bibliothèque Rust incompatible : recompilez cette branche avant de lancer l’application.")
        candidate.santoni_state_size.restype=C.c_size_t
        if candidate.santoni_state_size()!=C.sizeof(CState):
            raise ValueError("Taille de l’état Rust incompatible avec l’interface.")
        _library = candidate
        _library.santoni_search.argtypes = [C.POINTER(CState), C.c_double, C.POINTER(CAnalysis), PUBLISH, CANCEL]
        _library.santoni_search.restype = C.c_int32
        _library.santoni_legal.argtypes = [C.POINTER(CState), C.POINTER(CTurn), C.c_size_t]
        _library.santoni_legal.restype = C.c_size_t
        if hasattr(_library, "santoni_next_actions"):
            _library.santoni_next_actions.argtypes = [C.POINTER(CState), C.POINTER(CAction), C.c_size_t,
                C.POINTER(CAction), C.c_size_t, C.POINTER(CTurn), C.POINTER(C.c_uint8), CANCEL]
            _library.santoni_next_actions.restype = C.c_size_t
    return _library


def eligible(pos):
    return all(p in range(56) and POWERS[p].supported for p in pos.powers) and all(len(w) <= 4 for w in pos.workers)


def encode(pos):
    pos.validate()
    result = CState()
    result.heights[:] = pos.heights
    result.domes = pos.domes
    for p, ws in enumerate(pos.workers):
        result.workers[p][:] = tuple(ws) + (-1,) * (4 - len(ws))
        result.counts[p] = len(ws)
    result.powers[:] = pos.powers
    result.player = pos.player
    result.hero_used[:] = pos.hero_used
    result.athena_lock = pos.athena_lock
    result.adonis[:] = pos.adonis or (-1, -1, -1)
    result.winner = -1 if pos.winner is None else pos.winner
    result.reason = {"Descente de deux niveaux (Pan)": 2, "Réunion au niveau 1 (Eros)": 3,
                     "Cinq tours complètes (Chronus)": 4,"Entrée dans l’abysse (Tartarus)":5,"Lieu du Destin (Moerae)":6,"Action annulée par Hecate":7}.get(pos.reason, 1 if pos.winner is not None else 0)
    for key,value in pos.extra.to_dict().items():
        field=getattr(result.extra,key)
        if key=="whirlpools":
            for owner in range(2):field[owner][:]=value[owner]
        elif isinstance(field,C.Array):field[:]=value
        else:setattr(result.extra,key,value)
    if pos.resume:
        data=json.loads(pos.resume)
        origin=encode(Position.from_dict(data["origin"]))
        C.memmove(C.byref(result.resume.origin),C.byref(origin),C.sizeof(CCore))
        result.resume.length=len(data["actions"])
        for i,a in enumerate(data["actions"]):result.resume.actions[i]=CAction(*a)
    return result


def decode(state):
    reason = {0: "", 1: "Montée au niveau 3", 2: "Descente de deux niveaux (Pan)", 3: "Réunion au niveau 1 (Eros)", 4: "Cinq tours complètes (Chronus)", 5: "Entrée dans l’abysse (Tartarus)", 6: "Lieu du Destin (Moerae)", 7: "Action annulée par Hecate"}[state.reason]
    def unpack(value):
        return tuple(unpack(x) for x in value) if isinstance(value,C.Array) else value
    extra=Extra(**{key:unpack(getattr(state.extra,key)) for key,_ in CExtra._fields_})
    resume=""
    if hasattr(state,"resume") and state.resume.length:
        data={"origin":decode(state.resume.origin).to_dict(),"actions":[[a.kind,a.player,a.worker,a.source,a.target] for a in state.resume.actions[:state.resume.length]]}
        resume=json.dumps(data,sort_keys=True,separators=(",",":"))
    return Position(tuple(state.heights), state.domes,
                    tuple(tuple(state.workers[p][:state.counts[p]]) for p in range(2)),
                    tuple(state.powers), state.player, tuple(bool(x) for x in state.hero_used),
                    bool(state.athena_lock), None if state.adonis[0] == -1 else tuple(state.adonis),
                    None if state.winner == -1 else state.winner, reason, extra, resume)


def decode_turn(value):
    if value.length > 128:
        raise ValueError("Le moteur Rust a renvoyé un tour trop long.")
    actions = tuple(Action(ACTION_KINDS[a.kind], a.player, a.worker, a.source, a.target)
                    for a in value.actions[:value.length])
    return Turn(actions, decode(value.after))


def decode_analysis(value, position):
    variation = []
    current = position
    for item in value.turns[:value.length]:
        turn = decode_turn(item)
        if value.status==7:
            variation.append(turn)
            break
        reference = validate_turn(current, turn.actions)
        if reference.after != turn.after:
            raise ValueError("Le moteur Rust et le moteur de référence divergent.")
        variation.append(turn)
        current = turn.after
    status = ("Victoire forcée démontrée" if value.score > 0 else "Défaite forcée démontrée") if value.proven else (
        "Information secrète : estimation sur des hypothèses publiques, aucune victoire certifiée" if value.status==7 else
        "Indiquez la carte réellement tirée par Chaos" if value.status == 6 else
        "Aucun tour complet légal" if value.status == 2 else
        "Budget insuffisant pour trouver un tour légal" if value.status == 3 else
        f"Profondeur {value.depth} {'terminée' if value.complete else 'partielle'} — budget atteint")
    return Analysis(variation[0] if variation else None, value.score if value.has_score else None,
                    value.depth, value.nodes, value.elapsed, bool(value.proven), bool(value.complete),
                    tuple(variation), status,
                    searching_depth=value.depth + 1 if value.status == 4 else 0,
                    stop_reason={0: "timeout", 1: "proof", 2: "terminal", 3: "timeout", 4: "", 5: "depth_limit"}.get(value.status, ""),
                    engine="Rust")


def native_search(position, seconds, cancelled=lambda: False, publish=lambda result: None):
    if not eligible(position):
        raise ValueError("Cette configuration utilise le moteur de référence.")
    lib = library()
    if any(p > 10 for p in position.powers) and not hasattr(lib, "santoni_rules_version"):
        raise ValueError("Bibliothèque Rust ancienne : recompilez le moteur pour les pouvoirs avancés et héros.")
    state, result = encode(position), CAnalysis()
    failures = []
    @PUBLISH
    def progress(pointer):
        if failures:
            return
        try:
            publish(decode_analysis(pointer.contents, position))
        except Exception as exc:
            failures.append(exc)
    @CANCEL
    def cancel():
        return bool(failures) or cancelled()
    status = lib.santoni_search(C.byref(state), seconds, C.byref(result), progress, cancel)
    if failures:
        raise failures[0]
    if status != 0:
        raise ValueError(f"Erreur du moteur Rust ({status}).")
    return decode_analysis(result, position)


def native_turns(position):
    """Full native rule output for differential tests and offline benchmarks."""
    lib = library()
    state = encode(position)
    count = lib.santoni_legal(C.byref(state), None, 0)
    if count == C.c_size_t(-1).value:
        raise ValueError("Configuration non prise en charge par Rust.")
    buffer = (CTurn * count)()
    actual = lib.santoni_legal(C.byref(state), buffer, count)
    if actual != count:
        raise ValueError("Nombre de tours Rust incohérent.")
    return [decode_turn(item) for item in buffer]


def native_next_actions(position, prefix=(), cancelled=lambda: False):
    """Native incremental legality, without materialising all hero turns."""
    lib = library()
    if not hasattr(lib, 'santoni_next_actions'):
        # Compatibility with frozen basic-engine libraries and old installs.
        from .engine import next_actions
        from .search import Interrupted
        def check():
            if cancelled(): raise Interrupted
        return next_actions(position, prefix, check)
    state = encode(position)
    entered = encode_actions(prefix)
    choices = (CAction * 1024)()
    complete, has_complete = CTurn(), C.c_uint8()
    cancel = CANCEL(lambda: int(cancelled()))
    count = lib.santoni_next_actions(C.byref(state), entered, len(prefix), choices, len(choices),
                                    C.byref(complete), C.byref(has_complete), cancel)
    if count == C.c_size_t(-1).value:
        if cancelled():
            from .search import Interrupted
            raise Interrupted
        raise ValueError('Vérification native des actions impossible.')
    if count > len(choices):
        raise ValueError('Trop d’actions natives pour le tampon de saisie.')
    return ([Action(ACTION_KINDS[a.kind], a.player, a.worker, a.source, a.target)
             for a in choices[:count]], decode_turn(complete) if has_complete.value else None)


def native_preview(position, prefix):
    lib=library()
    lib.santoni_preview.argtypes=[C.POINTER(CState),C.POINTER(CAction),C.c_size_t,C.POINTER(CState)]
    lib.santoni_preview.restype=C.c_int32
    state=encode(position);output=CState()
    actions=encode_actions(prefix)
    status=lib.santoni_preview(C.byref(state),actions,len(prefix),C.byref(output))
    if status:raise ValueError("Aperçu Rust impossible.")
    return decode(output)


def native_expectation(position,seconds,cancelled=lambda:False):
    lib=library()
    lib.santoni_expectation.argtypes=[C.POINTER(CState),C.c_double,C.POINTER(CAnalysis),CANCEL]
    lib.santoni_expectation.restype=C.c_int32
    state=encode(position);out=CAnalysis();cancel=CANCEL(lambda:int(cancelled()))
    if lib.santoni_expectation(C.byref(state),seconds,C.byref(out),cancel):raise ValueError('Évaluation initiale de Chaos impossible.')
    return decode_analysis(out,position)


def native_iter_turns(position,prefix=(),check=lambda:None):
    lib=library()
    lib.santoni_turn_page.argtypes=[C.POINTER(CState),C.POINTER(CAction),C.c_size_t,C.c_size_t,C.POINTER(CTurn),C.c_size_t,CANCEL]
    lib.santoni_turn_page.restype=C.c_size_t
    state=encode(position)
    entered=encode_actions(prefix)
    batch=(CTurn*32)();offset=0;failure=[]
    def cancel():
        try:check();return 0
        except Exception as exc:failure.append(exc);return 1
    callback=CANCEL(cancel)
    while True:
        count=lib.santoni_turn_page(C.byref(state),entered,len(prefix),offset,batch,len(batch),callback)
        if failure:raise failure[0]
        if count==C.c_size_t(-1).value:raise ValueError('Énumération Rust interrompue.')
        for value in batch[:count]:
            check();yield decode_turn(value)
        if count<len(batch):return
        offset+=count


def native_choose_secret(position,seconds,cancelled=lambda:False):
    lib=library()
    lib.santoni_choose_secret.argtypes=[C.POINTER(CState),C.c_double,C.POINTER(CState),CANCEL]
    lib.santoni_choose_secret.restype=C.c_int32
    state=encode(position);out=CState();cancel=CANCEL(lambda:int(cancelled()))
    if lib.santoni_choose_secret(C.byref(state),seconds,C.byref(out),cancel):raise ValueError('Choix du lieu secret impossible.')
    result=decode(out);result.validate();return result

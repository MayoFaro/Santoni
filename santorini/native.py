"""Small ctypes bridge to the dependency-free Rust engine.

Calls run inside the computation process, never in the Qt event loop. Every
published native action sequence is independently checked by Python rules.
"""
from __future__ import annotations

import ctypes as C
import os
from pathlib import Path

from .engine import Action, Position, Turn, validate_turn
from .search import Analysis


class CState(C.Structure):
    _fields_ = [("heights", C.c_uint8 * 25), ("domes", C.c_uint32),
                ("workers", (C.c_int8 * 3) * 2), ("counts", C.c_uint8 * 2),
                ("powers", C.c_uint8 * 2), ("player", C.c_uint8),
                ("hero_used", C.c_uint8 * 2), ("athena_lock", C.c_uint8),
                ("adonis", C.c_int8 * 3), ("winner", C.c_int8), ("reason", C.c_uint8)]


class CAction(C.Structure):
    _fields_ = [("kind", C.c_uint8), ("player", C.c_int8), ("worker", C.c_int8),
                ("source", C.c_int8), ("target", C.c_int8)]


class CTurn(C.Structure):
    _fields_ = [("actions", CAction * 64), ("length", C.c_uint32), ("after", CState)]


class CAnalysis(C.Structure):
    _fields_ = [("turns", CTurn * 3), ("length", C.c_uint32), ("score", C.c_int32),
                ("has_score", C.c_uint8), ("depth", C.c_uint32), ("nodes", C.c_uint64),
                ("elapsed", C.c_double), ("proven", C.c_uint8), ("complete", C.c_uint8), ("status", C.c_uint8)]


PUBLISH = C.CFUNCTYPE(None, C.POINTER(CAnalysis))
CANCEL = C.CFUNCTYPE(C.c_uint8)
_library = None


def library_path():
    supplied = os.environ.get("SANTONI_ENGINE_LIB")
    if supplied:
        return Path(supplied)
    filename = "santoni_engine.dll" if os.name == "nt" else "libsantoni_engine.dylib" if __import__("sys").platform == "darwin" else "libsantoni_engine.so"
    return Path(__file__).resolve().parent.parent / "native_engine" / "target" / "release" / filename


def library():
    global _library
    if _library is None:
        _library = C.CDLL(str(library_path()))
        _library.santoni_search.argtypes = [C.POINTER(CState), C.c_double, C.POINTER(CAnalysis), PUBLISH, CANCEL]
        _library.santoni_search.restype = C.c_int32
        _library.santoni_legal.argtypes = [C.POINTER(CState), C.POINTER(CTurn), C.c_size_t]
        _library.santoni_legal.restype = C.c_size_t
    return _library


def eligible(pos):
    return all(p <= 10 for p in pos.powers) and pos.adonis is None and all(len(w) == 2 for w in pos.workers)


def encode(pos):
    pos.validate()
    result = CState()
    result.heights[:] = pos.heights
    result.domes = pos.domes
    for p, ws in enumerate(pos.workers):
        result.workers[p][:] = tuple(ws) + (-1,) * (3 - len(ws))
        result.counts[p] = len(ws)
    result.powers[:] = pos.powers
    result.player = pos.player
    result.hero_used[:] = pos.hero_used
    result.athena_lock = pos.athena_lock
    result.adonis[:] = pos.adonis or (-1, -1, -1)
    result.winner = -1 if pos.winner is None else pos.winner
    result.reason = 2 if pos.reason == "Descente de deux niveaux (Pan)" else 1 if pos.winner is not None else 0
    return result


def decode(state):
    reason = {0: "", 1: "Montée au niveau 3", 2: "Descente de deux niveaux (Pan)"}[state.reason]
    return Position(tuple(state.heights), state.domes,
                    tuple(tuple(state.workers[p][:state.counts[p]]) for p in range(2)),
                    tuple(state.powers), state.player, tuple(bool(x) for x in state.hero_used),
                    bool(state.athena_lock), None if state.adonis[0] == -1 else tuple(state.adonis),
                    None if state.winner == -1 else state.winner, reason)


def decode_turn(value):
    if value.length > 64:
        raise ValueError("Le moteur Rust a renvoyé un tour trop long.")
    actions = tuple(Action(("move", "build", "dome")[a.kind], a.player, a.worker, a.source, a.target)
                    for a in value.actions[:value.length])
    return Turn(actions, decode(value.after))


def decode_analysis(value, position):
    variation = []
    current = position
    for item in value.turns[:value.length]:
        turn = decode_turn(item)
        reference = validate_turn(current, turn.actions)
        if reference.after != turn.after:
            raise ValueError("Le moteur Rust et le moteur de référence divergent.")
        variation.append(turn)
        current = turn.after
    status = ("Victoire forcée démontrée" if value.score > 0 else "Défaite forcée démontrée") if value.proven else (
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

"""Process isolation with cancellation and snapshot IDs for Qt."""
from __future__ import annotations

import multiprocessing
import os
import threading
import time
import traceback

from PySide6.QtCore import QThread, Signal

from .engine import next_actions
from .search import Interrupted, choose_power, search


def _compute(kind, args, cancelled, publish):
    if kind == "search":
        position, budget, *end = args
        if end:
            budget = max(0.001, min(budget, end[0] - time.monotonic()))
        from .native import eligible, library_path, native_search
        if eligible(position) and library_path().exists():
            return native_search(position, budget, cancelled=cancelled, publish=publish)
        return search(position, budget, cancelled=cancelled, publish=publish)
    if kind=="secret_setup":
        from .native import native_choose_secret
        position,budget,*end=args
        if end:budget=max(.001,min(budget,end[0]-time.monotonic()))
        return native_choose_secret(position,budget,cancelled=cancelled)
    if kind=="resolve":
        from .engine import resolve_plan
        return resolve_plan(*args)
    if kind == "correction":
        session, index, actions = args
        return session.correct(index, actions)
    if kind == "power":
        return choose_power(*args, cancelled=cancelled, publish=publish)
    if kind == "placement":
        from .placement import choose_placement
        powers, first, budget, opponent, *end = args
        if end:
            budget = max(.1, min(budget, end[0] - time.monotonic()))
        return choose_placement(powers, first, budget, opponent,
                                cancelled=cancelled, publish=publish,extra=end[1] if len(end)>1 else None)
    if kind == "actions":
        from .native import eligible, library_path, native_next_actions
        if eligible(args[0]) and library_path().exists():
            return native_next_actions(*args, cancelled=cancelled)
        def check():
            if cancelled():
                raise Interrupted
        return next_actions(*args, check=check)
    if kind == "load":
        return args[0].load()
    raise ValueError("Calcul inconnu")


def _process_main(kind, args, connection, cancelled):
    try:
        if hasattr(os, "nice"):
            os.nice(5)
        last = 0.0
        last_depth = None
        def publish(value):
            nonlocal last, last_depth
            instant = time.monotonic()
            depth = getattr(value, "searching_depth", None)
            if instant - last >= 0.1 or depth != last_depth:
                connection.send(("progress", value))
                last = instant
                last_depth = depth
        result = _compute(kind, args, cancelled.is_set, publish)
        if not cancelled.is_set():
            connection.send(("result", result))
    except Interrupted:
        pass
    except Exception as exc:
        traceback.print_exc()
        connection.send(("error", str(exc)))
    finally:
        connection.close()


class Worker(QThread):
    progress = Signal(object, int)
    result = Signal(object, int)
    failed = Signal(str, int)

    def __init__(self, kind, args, generation, parent=None):
        super().__init__(parent)
        self.kind, self.args, self.generation = kind, args, generation
        self.cancelled = threading.Event()

    def cancel(self):
        self.cancelled.set()

    def run(self):
        if self.cancelled.is_set():
            return
        ctx = multiprocessing.get_context("spawn")
        reader, writer = ctx.Pipe(duplex=False)
        event = ctx.Event()
        process = ctx.Process(target=_process_main, args=(self.kind, self.args, writer, event))
        received = False
        try:
            process.start()
            writer.close()
            while not self.cancelled.is_set():
                if reader.poll(0.025):
                    try:
                        kind, value = reader.recv()
                    except EOFError:
                        break
                    if kind == "progress":
                        self.progress.emit(value, self.generation)
                    elif kind == "result":
                        received = True
                        self.result.emit(value, self.generation)
                        break
                    else:
                        received = True
                        self.failed.emit(value, self.generation)
                        break
                elif not process.is_alive():
                    break
            if not self.cancelled.is_set() and not received:
                self.failed.emit("Le calcul s'est arrêté sans résultat.", self.generation)
        except Exception as exc:
            if not self.cancelled.is_set():
                self.failed.emit(str(exc), self.generation)
        finally:
            event.set()
            if process.pid is not None:
                process.join(0.1)
                if process.is_alive():
                    process.terminate()
                    process.join(1)
                if process.is_alive():
                    process.kill()
                    process.join()
                process.close()
            reader.close()
            writer.close()

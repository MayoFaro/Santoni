"""Atomic automatic saves and full, versioned game archives."""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .engine import Action, Position, Turn, validate_turn


def now():
    return datetime.now(timezone.utc).isoformat()


def action_dict(a):
    return {"kind": a.kind, "player": a.player, "worker": a.worker,
            "source": a.source, "target": a.target}


@dataclass(frozen=True)
class Session:
    initial: Position
    history: tuple[Turn, ...] = ()
    result: dict | None = None
    started_at: str = field(default_factory=now)
    id: str = field(default_factory=lambda: uuid4().hex)
    settings: dict = field(default_factory=dict)

    @property
    def position(self):
        return self.history[-1].after if self.history else self.initial

    def append(self, turn):
        if self.result is not None:
            raise ValueError("La partie est terminée.")
        outcome = None
        if turn.after.winner is not None:
            outcome = {"winner": turn.after.winner, "reason": turn.after.reason, "recorded_at": now()}
        return replace(self, history=self.history + (turn,), result=outcome)

    def undo(self):
        if not self.history:
            return self
        return replace(self, history=self.history[:-1], result=None)

    def ended(self, winner, reason):
        return replace(self, result={"winner": winner, "reason": reason, "recorded_at": now()})

    def to_dict(self):
        return {"version": 1, "id": self.id, "started_at": self.started_at,
                "saved_at": now(), "settings": self.settings,
                "initial": self.initial.to_dict(),
                "history": [{"actions": [action_dict(a) for a in t.actions], "after": t.after.to_dict()}
                            for t in self.history], "result": self.result}

    @classmethod
    def from_dict(cls, data):
        if data.get("version") != 1:
            raise ValueError("Version de sauvegarde inconnue.")
        initial = Position.from_dict(data["initial"])
        initial.validate()
        s = cls(initial, started_at=data["started_at"], id=data["id"], settings=data.get("settings", {}))
        for item in data["history"]:
            turn = validate_turn(s.position, tuple(Action(**a) for a in item["actions"]))
            if turn.after != Position.from_dict(item["after"]):
                raise ValueError("L'historique de cette sauvegarde est incohérent.")
            s = s.append(turn)
        outcome = data.get("result")
        if outcome is not None and (not isinstance(outcome, dict) or outcome.get("winner") not in (None, 0, 1)):
            raise ValueError("Résultat de sauvegarde invalide.")
        return replace(s, result=outcome)


class Store:
    def __init__(self, root=None):
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
        self.root = Path(root or os.environ.get("SANTONI_DATA_DIR", base / "santoni"))
        self.current = self.root / "en_cours.json"

    def _write(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".save-", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, path)
            # Make the replacement durable, including the directory entry.
            if hasattr(os, "O_DIRECTORY"):
                directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def save(self, session):
        self._write(self.current, session.to_dict() if session else None)

    def load(self):
        if not self.current.exists():
            return None
        data = json.loads(self.current.read_text(encoding="utf-8"))
        return Session.from_dict(data) if data else None

    def archive(self, session):
        if session.result is None:
            session = session.ended(None, "Partie interrompue")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        path = self.root / "parties" / f"{stamp}-{session.id[:8]}.json"
        self._write(path, session.to_dict())
        return path

    def reset(self, session):
        path = self.archive(session) if session else None
        # Archive must succeed before clearing the resumable game.
        self.save(None)
        return path

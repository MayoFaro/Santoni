"""Pure, immutable game rules. A Turn always includes all of its actions.

UI and search use the same legal-turn generator. No Qt imports or disk access.
The engine implements the public powers marked supported in powers.py.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
from typing import Callable, Iterator

from .powers import POWERS


def coord(cell: int) -> str:
    return f"{'ABCDE'[cell % 5]}{cell // 5 + 1}"


def cell_at(column: str, row: int) -> int:
    return (row - 1) * 5 + "ABCDE".index(column.upper())


def border(cell: int) -> bool:
    return cell % 5 in (0, 4) or cell // 5 in (0, 4)


def neighbors(cell: int, wrap=False) -> tuple[int, ...]:
    x, y = cell % 5, cell // 5
    return tuple((ny % 5) * 5 + nx % 5
                 for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                 if dx or dy
                 for nx, ny in [(x + dx, y + dy)]
                 if wrap or (0 <= nx < 5 and 0 <= ny < 5))


NEIGHBORS = tuple(neighbors(i) for i in range(25))
WRAPPED = tuple(neighbors(i, True) for i in range(25))


@dataclass(frozen=True)
class Action:
    kind: str
    player: int = -1
    worker: int = -1
    source: int = -1
    target: int = -1

    def label(self) -> str:
        piece = f"{'Moi' if self.player == 0 else 'Robot'} {self.worker + 1}"
        if self.kind == "activate":
            return "Utiliser le héros"
        if self.kind == "move":
            return f"{piece} : {coord(self.source)} → {coord(self.target)}"
        if self.kind == "build":
            return f"Construire en {coord(self.target)}"
        if self.kind == "dome":
            return f"Dôme en {coord(self.target)}"
        if self.kind == "remove":
            return f"Retirer un bloc en {coord(self.target)}"
        if self.kind == "kill":
            return f"Éliminer {piece} en {coord(self.target)}"
        if self.kind == "force":
            return f"Forcer {piece} : {coord(self.source)} → {coord(self.target)}"
        if self.kind == "place":
            return f"Placer {piece} en {coord(self.target)}"
        if self.kind == "adonis":
            return f"Adonis : désigner {piece} en {coord(self.target)}"
        return self.kind


@dataclass(frozen=True)
class Position:
    heights: tuple[int, ...] = (0,) * 25
    domes: int = 0
    workers: tuple[tuple[int, ...], tuple[int, ...]] = ((6, 8), (16, 18))
    powers: tuple[int, int] = (0, 0)
    player: int = 0
    hero_used: tuple[bool, bool] = (False, False)
    athena_lock: bool = False
    adonis: tuple[int, int, int] | None = None
    winner: int | None = None
    reason: str = ""

    def occupied(self, cell):
        return bool(self.domes & (1 << cell)) or any(cell in ws for ws in self.workers)

    def to_dict(self):
        return {"heights": list(self.heights), "domes": self.domes,
                "workers": [list(ws) for ws in self.workers], "powers": list(self.powers),
                "player": self.player, "hero_used": list(self.hero_used),
                "athena_lock": self.athena_lock, "adonis": self.adonis,
                "winner": self.winner, "reason": self.reason}

    @classmethod
    def from_dict(cls, data):
        pos = cls(heights=tuple(data["heights"]), domes=data["domes"],
                  workers=tuple(tuple(w) for w in data["workers"]),
                  powers=tuple(data["powers"]), player=data["player"],
                  hero_used=tuple(data.get("hero_used", (False, False))),
                  athena_lock=data.get("athena_lock", False),
                  adonis=tuple(data["adonis"]) if data.get("adonis") else None,
                  winner=data.get("winner"), reason=data.get("reason", ""))
        pos.validate()
        return pos

    def validate(self):
        if len(self.heights) != 25 or any(type(h) is not int or not 0 <= h <= 3 for h in self.heights):
            raise ValueError("Les hauteurs doivent être comprises entre 0 et 3.")
        if type(self.domes) is not int or not 0 <= self.domes < 1 << 25:
            raise ValueError("Dômes invalides.")
        if self.player not in (0, 1) or self.winner not in (None, 0, 1):
            raise ValueError("Joueur invalide.")
        if len(self.powers) != 2 or any(p not in POWERS or not POWERS[p].supported for p in self.powers):
            raise ValueError("Pouvoir non pris en charge.")
        if len(self.workers) != 2 or any(not 2 <= len(w) <= 3 for w in self.workers):
            raise ValueError("Nombre de bâtisseurs invalide.")
        present = [w for ws in self.workers for w in ws if w != -1]
        if any(type(w) is not int or not 0 <= w < 25 or self.domes & (1 << w) for w in present):
            raise ValueError("Position de bâtisseur invalide.")
        if len(set(present)) != len(present):
            raise ValueError("Deux bâtisseurs ne peuvent pas occuper la même case.")
        if len(self.hero_used) != 2 or any(type(x) is not bool for x in self.hero_used):
            raise ValueError("État de héros invalide.")
        if self.adonis is not None:
            p, w, owner = self.adonis
            if p not in (0, 1) or owner != 1 - p or not 0 <= w < len(self.workers[p]):
                raise ValueError("Cible d'Adonis invalide.")


@dataclass(frozen=True)
class Turn:
    actions: tuple[Action, ...]
    after: Position

    def label(self):
        return " ; ".join(a.label() for a in self.actions)

    def description(self, before):
        lines = [self.label()]
        foe = 1 - before.player
        explicit = {a.worker for a in self.actions if a.player == foe and a.kind in ("force", "kill")}
        for i, old in enumerate(before.workers[foe]):
            new = self.after.workers[foe][i]
            if old < 0 or old == new or i in explicit:
                continue
            piece = f"{'Moi' if foe == 0 else 'Robot'} {i + 1}"
            lines.append(f"Retirer {piece} en {coord(old)}" if new < 0 else
                         f"Déplacement forcé de {piece} : {coord(old)} → {coord(new)}")
        return "\n".join(lines)

    def power_summary(self, before):
        power = before.powers[before.player]
        if power == 0:
            return "Aucun pouvoir"
        if power >= 46:
            return "Héros utilisé" if any(a.kind == "activate" for a in self.actions) else (
                "Héros déjà consommé" if before.hero_used[before.player] else "Héros conservé")
        moves = [a for a in self.actions if a.kind == "move"]
        builds = [a for a in self.actions if a.kind in ("build", "dome")]
        if power in (1, 8):
            used = any(a.target in before.workers[1 - before.player] for a in moves)
        elif power in (2, 29):
            used = len(moves) > 1
        elif power == 3:
            return "Athena : interdiction de monter active" if self.after.athena_lock else "Athena : aucune interdiction ce tour"
        elif power == 4:
            used = any(a.kind == "dome" and before.heights[a.target] < 3 for a in builds)
        elif power in (5, 6, 21):
            used = len(builds) > 1
        elif power == 7:
            used = len(moves) != 1 or any(a.worker != moves[-1].worker for a in builds)
        elif power == 9:
            return "Victoire par Pan" if self.after.reason == "Descente de deux niveaux (Pan)" else "Pan : victoire par descente non déclenchée"
        elif power == 10:
            used = self.actions[0].kind in ("build", "dome")
        elif power == 12:
            used = any(a.kind == "remove" for a in self.actions)
        elif power == 15:
            used = any(a.kind == "force" for a in self.actions)
        elif power == 27:
            used = len(builds) > 1
        elif power == 28:
            used = any(a.kind == "dome" and (a.worker != moves[-1].worker or before.heights[a.target] < 3) for a in builds)
        elif power == 30:
            used = any(a.source == a.target for a in builds)
        elif power == 45:
            used = any(abs(a.source % 5 - a.target % 5) > 1 or abs(a.source // 5 - a.target // 5) > 1
                       for a in self.actions if a.source >= 0 and a.target >= 0)
        else:
            return f"{POWERS[power].name} : effets permanents appliqués"
        return f"{POWERS[power].name} : pouvoir {'utilisé' if used else 'facultatif non utilisé'}"


def _put_worker(s, player, worker, cell):
    ws = [list(w) for w in s.workers]
    if worker == len(ws[player]):
        ws[player].append(cell)
    else:
        ws[player][worker] = cell
    return replace(s, workers=tuple(tuple(w) for w in ws))


def _chronus(s):
    if s.winner is None and any(p == 16 for p in s.powers):
        complete = sum(h == 3 and bool(s.domes & (1 << i)) for i, h in enumerate(s.heights))
        if complete >= 5:
            # Both players cannot select the same power through normal setup.
            return replace(s, winner=s.powers.index(16), reason="Cinq tours complètes (Chronus)")
    return s


def _apply(s, a):
    if a.kind == "activate":
        used = list(s.hero_used)
        used[s.player] = True
        return replace(s, hero_used=tuple(used))
    if a.kind == "adonis":
        return replace(s, adonis=(a.player, a.worker, s.player))
    if a.kind in ("move", "force", "place"):
        result = s
        if a.kind == "move":
            # Apollo swaps; Minotaur pushes. Forcing never triggers a movement win.
            foe = 1 - a.player
            if a.target in s.workers[foe]:
                victim = s.workers[foe].index(a.target)
                if s.powers[a.player] == 1:
                    result = _put_worker(result, foe, victim, a.source)
                elif s.powers[a.player] == 8:
                    x = 2 * (a.target % 5) - a.source % 5
                    y = 2 * (a.target // 5) - a.source // 5
                    result = _put_worker(result, foe, victim, y * 5 + x)
        result = _put_worker(result, a.player, a.worker, a.target)
        if a.kind == "move":
            h0, h1 = s.heights[a.source], s.heights[a.target]
            ordinary = h1 == 3 and h0 < 3
            pan = s.powers[a.player] == 9 and h0 - h1 >= 2
            if (ordinary or pan) and not (s.powers[1 - a.player] == 20 and border(a.target)):
                result = replace(result, winner=a.player,
                                 reason="Descente de deux niveaux (Pan)" if pan else "Montée au niveau 3")
            if s.powers[a.player] == 13:
                x = 2 * (a.target % 5) - a.source % 5
                y = 2 * (a.target // 5) - a.source // 5
                if 0 <= x < 5 and 0 <= y < 5 and y * 5 + x in result.workers[1 - a.player]:
                    victim = result.workers[1 - a.player].index(y * 5 + x)
                    result = _put_worker(result, 1 - a.player, victim, -1)
            if s.powers[a.player] == 19 and h1 == 1:
                others = [w for i, w in enumerate(result.workers[a.player]) if i != a.worker and w >= 0]
                if any(w in NEIGHBORS[a.target] for w in others) and not (s.powers[1 - a.player] == 20 and border(a.target)):
                    result = replace(result, winner=a.player, reason="Réunion au niveau 1 (Eros)")
        return result
    if a.kind == "kill":
        return _put_worker(s, a.player, a.worker, -1)
    if a.kind == "dome":
        return _chronus(replace(s, domes=s.domes | (1 << a.target)))
    if a.kind in ("build", "remove"):
        hs = list(s.heights)
        hs[a.target] += 1 if a.kind == "build" else -1
        return replace(s, heights=tuple(hs))
    raise ValueError(f"Action inconnue : {a.kind}")


def _can_step(s, worker, dst, active=False, no_up=False, flat=False):
    p, foe = s.player, 1 - s.player
    src = s.workers[p][worker]
    if s.domes & (1 << dst) or dst in s.workers[p]:
        return False
    delta = s.heights[dst] - s.heights[src]
    if flat and delta != 0:
        return False
    if delta > (2 if s.powers[p] == 49 and active else 1):
        return False
    if delta > 0 and (s.athena_lock or no_up):
        return False
    if delta < 0 and s.powers[foe] == 37:
        return False
    if dst in s.workers[foe]:
        if s.powers[p] == 1:
            return True
        if s.powers[p] != 8:
            return False
        x, y = 2 * (dst % 5) - src % 5, 2 * (dst // 5) - src // 5
        return 0 <= x < 5 and 0 <= y < 5 and not s.occupied(y * 5 + x)
    return True


def _builds(s, worker, atlas=False, selene=False):
    p, foe = s.player, 1 - s.player
    src = s.workers[p][worker]
    if src < 0:
        return
    adjacent = WRAPPED[src] if s.powers[p] == 45 else NEIGHBORS[src]
    for dst in adjacent:
        if s.occupied(dst):
            continue
        if s.powers[foe] == 23 and any(dst in NEIGHBORS[w] for w in s.workers[foe] if w >= 0):
            if s.heights[dst] != 3:
                continue
        ordinary = "dome" if s.heights[dst] == 3 else "build"
        if not selene:
            yield Action(ordinary, p, worker, src, dst)
        if (atlas or selene) and (ordinary != "dome" or selene):
            yield Action("dome", p, worker, src, dst)
    if s.powers[p] == 30 and s.heights[src] < 3 and not selene:
        if s.powers[foe] != 23 or not any(src in NEIGHBORS[w] for w in s.workers[foe] if w >= 0):
            yield Action("build", p, worker, src, src)


def _raw_turns(position: Position, prefix: tuple[Action, ...] = (),
                check: Callable[[], None] = lambda: None,
                skip_next: set[Action] | None = None, hero_mode=None) -> Iterator[Turn]:
    """Generate complete legal turns lazily; prefix prunes the action tree.

    The same iterator serves interactive input, validation and minimax. check()
    is called throughout generation, allowing cancellation and hard deadlines.
    """
    if position.winner is not None:
        return
    p, foe = position.player, 1 - position.player
    power = position.powers[p]
    if not POWERS[power].supported:
        raise ValueError("Pouvoir non pris en charge")
    constraint = position.adonis if position.adonis and position.adonis[0] == p else None

    def fits(actions):
        check()
        n = min(len(prefix), len(actions))
        if actions[:n] != prefix[:n]:
            return False
        return not (skip_next is not None and len(actions) > len(prefix) and actions[len(prefix)] in skip_next)

    def add(s, actions, a):
        new = actions + (a,)
        return (_apply(s, a), new) if fits(new) else None

    def finish(s, actions, up):
        if len(actions) < len(prefix) or not fits(actions):
            return
        result = replace(s, player=foe, athena_lock=power == 3 and up,
                         adonis=None if constraint else s.adonis)
        yield Turn(actions, result)

    def subsets(s, actions, options, up, remaining=None):
        # A hero may stop after any subset, including empty. Canonical order
        # avoids permutations of independent dome/remove actions.
        if s.winner is not None:
            yield from finish(s, actions, up)
            return
        yield from finish(s, actions, up)
        if remaining is None:
            remaining = tuple(range(len(options)))
        for i in remaining:
            pair = add(s, actions, options[i])
            if pair:
                following = tuple(j for j in remaining if j != i and (prefix or skip_next is not None or j > i))
                yield from subsets(*pair, options, up, following)

    def end_effects(s, actions, worker, active, up):
        if s.winner is not None:
            yield from finish(s, actions, up)
            return
        if power == 24:
            src = s.workers[p][worker]
            for victim, dst in enumerate(s.workers[foe]):
                if dst >= 0 and dst in NEIGHBORS[src] and s.heights[dst] < s.heights[src]:
                    if s.powers[foe] == 23 and any(dst in NEIGHBORS[other] for other in s.workers[foe] if other >= 0):
                        continue
                    pair = add(s, actions, Action("kill", foe, victim, dst, dst))
                    if not pair:
                        return
                    s, actions = pair
                    pair = add(s, actions, Action("build", p, worker, src, dst))
                    if not pair:
                        return
                    s, actions = pair
        if power == 12:
            yield from finish(s, actions, up)
            for idle, src in enumerate(s.workers[p]):
                if idle == worker or src < 0:
                    continue
                for dst in NEIGHBORS[src]:
                    if not s.occupied(dst) and s.heights[dst] > 0:
                        pair = add(s, actions, Action("remove", p, idle, src, dst))
                        if pair:
                            yield from finish(*pair, up)
            return
        if power == 27:
            idle = 1 - worker
            if idle < len(s.workers[p]) and s.workers[p][idle] >= 0 and s.heights[s.workers[p][idle]] == 0:
                yield from extra_builds(s, actions, idle, 3, up)
                return
        if active and power == 47:
            for victim, dst in enumerate(s.workers[foe]):
                if dst >= 0:
                    pair = add(s, actions, Action("adonis", foe, victim, dst, dst))
                    if pair:
                        yield from finish(*pair, up)
            return
        if active and power in (50, 52):
            options = []
            if power == 50:
                spaces = sorted({dst for src in s.workers[p] if src >= 0 for dst in NEIGHBORS[src] if not s.occupied(dst)})
                options = [Action("dome", p, -1, -1, dst) for dst in spaces
                           if s.powers[foe] != 23 or s.heights[dst] == 3 or not any(dst in NEIGHBORS[w] for w in s.workers[foe] if w >= 0)]
            else:
                idle = 1 - worker
                if idle < len(s.workers[p]) and s.workers[p][idle] >= 0:
                    src = s.workers[p][idle]
                    options = [Action("remove", p, idle, src, dst)
                               for ws in s.workers for dst in ws
                               if dst >= 0 and dst in NEIGHBORS[src] and s.heights[dst] > 0]
                    options.sort(key=lambda a: a.target)
            yield from subsets(s, actions, options, up)
            return
        if active and power == 54:
            yield from finish(s, actions, up)
            for dst in range(25):
                if s.occupied(dst):
                    continue
                if s.powers[foe] == 23 and s.heights[dst] != 3 and any(dst in NEIGHBORS[w] for w in s.workers[foe] if w >= 0):
                    continue
                pair = add(s, actions, Action("dome", p, worker, s.workers[p][worker], dst))
                if not pair:
                    continue
                s1, a1 = pair
                yield from finish(s1, a1, up)
                if s1.winner is not None:
                    continue
                for dst2 in range(25) if prefix or skip_next is not None else range(dst + 1, 25):
                    if s1.occupied(dst2):
                        continue
                    if s1.powers[foe] == 23 and s1.heights[dst2] != 3 and any(dst2 in NEIGHBORS[w] for w in s1.workers[foe] if w >= 0):
                        continue
                    pair2 = add(s1, a1, Action("dome", p, worker, s1.workers[p][worker], dst2))
                    if pair2:
                        yield from finish(*pair2, up)
            return
        if active and power == 55:
            for victim, dst in enumerate(s.workers[foe]):
                if dst >= 0 and any(dst in NEIGHBORS[src] and s.heights[dst] - s.heights[src] == 2 for src in s.workers[p] if src >= 0):
                    pair = add(s, actions, Action("kill", foe, victim, dst, dst))
                    if pair:
                        yield from finish(*pair, up)
            return
        yield from finish(s, actions, up)

    def extra_builds(s, actions, worker, count, up):
        if s.winner is not None:
            yield from finish(s, actions, up)
            return
        yield from finish(s, actions, up)
        if count:
            for a in _builds(s, worker):
                pair = add(s, actions, a)
                if pair:
                    yield from extra_builds(*pair, worker, count - 1, up)

    def construct(s, actions, worker, active, up):
        if s.winner is not None:
            yield from finish(s, actions, up)
            return
        builders = [(worker, False)]
        if power == 28 and len(s.workers[p]) > 1 and s.workers[p][1] >= 0:
            builders.append((1, True))  # worker 2 is the female builder.
        for builder, selene in builders:
            for a in _builds(s, builder, atlas=power == 4, selene=selene):
                pair = add(s, actions, a)
                if not pair:
                    continue
                s1, a1 = pair
                yield from end_effects(s1, a1, worker, active, up)
                if s1.winner is not None:
                    continue
                if power in (5, 6, 21):
                    for second in _builds(s1, worker):
                        if power == 5 and second.target == a.target:
                            continue
                        if power == 6 and (second.target != a.target or a.kind != "build" or second.kind != "build"):
                            continue
                        if power == 21 and border(second.target):
                            continue
                        pair2 = add(s1, a1, second)
                        if pair2:
                            yield from end_effects(*pair2, worker, active, up)

    def adjacency_ok(s, worker):
        if position.powers[foe] != 11:
            return True
        origin = position.workers[p][worker]
        if origin < 0 or not any(v in NEIGHBORS[origin] for v in position.workers[foe] if v >= 0):
            return True
        now = s.workers[p][worker]
        return any(v in NEIGHBORS[now] for v in s.workers[foe] if v >= 0)

    def move_paths(s, actions, worker, active, no_up):
        src = s.workers[p][worker]
        adjacent = WRAPPED[src] if power == 45 else NEIGHBORS[src]
        for dst in adjacent:
            if not _can_step(s, worker, dst, active, no_up):
                continue
            pair = add(s, actions, Action("move", p, worker, src, dst))
            if not pair:
                continue
            s1, a1 = pair
            up = s.heights[dst] > s.heights[src]
            if adjacency_ok(s1, worker):
                yield s1, a1, up
            if s1.winner is not None:
                continue
            if power == 2:
                for dst2 in NEIGHBORS[dst]:
                    if dst2 == src or not _can_step(s1, worker, dst2, active, no_up):
                        continue
                    pair2 = add(s1, a1, Action("move", p, worker, dst, dst2))
                    if pair2 and adjacency_ok(pair2[0], worker):
                        yield *pair2, up or s1.heights[dst2] > s1.heights[dst]
            elif power == 29 or (active and power == 48):
                # Arbitrary movement: visit each (destination, ascent flag) once
                # after the supplied prefix. During prefix entry retain paths.
                queue = deque([(s1, a1, up)])
                seen = {(dst, up)} if len(a1) >= len(prefix) else set()
                while queue:
                    ss, aa, climbed = queue.popleft()
                    current = ss.workers[p][worker]
                    if power == 29 and not border(current):
                        continue
                    for nxt in NEIGHBORS[current]:
                        if not _can_step(ss, worker, nxt, active, no_up):
                            continue
                        pair2 = add(ss, aa, Action("move", p, worker, current, nxt))
                        if not pair2:
                            continue
                        new_up = climbed or ss.heights[nxt] > ss.heights[current]
                        key = (nxt, new_up)
                        if len(pair2[1]) >= len(prefix) and key in seen:
                            continue
                        if len(pair2[1]) >= len(prefix):
                            seen.add(key)
                        if adjacency_ok(pair2[0], worker):
                            yield *pair2, new_up
                        if pair2[0].winner is None:
                            queue.append((*pair2, new_up))

    def ordinary(s, actions, active):
        if active and power == 51:
            for dst in range(25):
                if border(dst) and not s.occupied(dst) and s.heights[dst] == 0:
                    worker = len(s.workers[p])
                    pair = add(s, actions, Action("place", p, worker, -1, dst))
                    if pair:
                        yield from construct(*pair, worker, active, False)
            return
        if power == 7:
            # Hermes' flat mode can move either builder any number of times,
            # including zero, then build with either. BFS includes interleaving.
            queue = deque([(s, actions)])
            seen = {s.workers[p]} if len(actions) >= len(prefix) else set()
            while queue:
                ss, aa = queue.popleft()
                if all(adjacency_ok(ss, i) for i, w in enumerate(ss.workers[p]) if w >= 0):
                    for i, w in enumerate(ss.workers[p]):
                        if w >= 0:
                            yield from construct(ss, aa, i, active, False)
                for i, w in enumerate(ss.workers[p]):
                    if w < 0 or hypnus_blocks(i):
                        continue
                    for dst in NEIGHBORS[w]:
                        if not _can_step(ss, i, dst, flat=True):
                            continue
                        pair = add(ss, aa, Action("move", p, i, w, dst))
                        if not pair:
                            continue
                        key = pair[0].workers[p]
                        if len(pair[1]) >= len(prefix) and key in seen:
                            continue
                        if len(pair[1]) >= len(prefix):
                            seen.add(key)
                        queue.append(pair)
        for worker, src in enumerate(s.workers[p]):
            if src < 0 or hypnus_blocks(worker):
                continue
            pre = [(s, actions, False)]
            if power == 10 or (active and power == 46):
                if active and power == 46:
                    pre = []
                for a in _builds(s, worker):
                    pair = add(s, actions, a)
                    if pair:
                        if pair[0].winner is not None:
                            yield from finish(*pair, False)
                        else:
                            pre.append((*pair, power == 10))
            if power == 15:
                for victim, dst in enumerate(s.workers[foe]):
                    if dst < 0 or dst not in NEIGHBORS[src]:
                        continue
                    x, y = 2 * (src % 5) - dst % 5, 2 * (src // 5) - dst // 5
                    if 0 <= x < 5 and 0 <= y < 5 and not s.occupied(y * 5 + x):
                        pair = add(s, actions, Action("force", foe, victim, dst, y * 5 + x))
                        if pair:
                            pre.append((*pair, False))
            for ss, aa, no_up in pre:
                for moved, path, up in move_paths(ss, aa, worker, active, no_up):
                    yield from construct(moved, path, worker, active, up)

    def hypnus_blocks(worker):
        src = position.workers[p][worker]
        return position.powers[foe] == 22 and all(position.heights[src] > position.heights[w]
                for owner, ws in enumerate(position.workers) for i, w in enumerate(ws)
                if w >= 0 and (owner != p or i != worker))

    def variants():
        if hero_mode is not True:
            yield from ordinary(position, (), False)
        if hero_mode is not False and power >= 46 and not position.hero_used[p]:
            pair = add(position, (), Action("activate", p))
            if pair:
                ss, aa = pair
                if power == 53:
                    # Movement can free a corner for another victim; the
                    # order is material, so examine both victim orders.
                    def force_corners(s, actions, remaining=None):
                        yield from ordinary(s, actions, True)
                        if remaining is None:
                            remaining = tuple(range(len(s.workers[foe])))
                        for victim in remaining:
                            dst = s.workers[foe][victim]
                            if dst < 0 or not any(dst in NEIGHBORS[w] for w in s.workers[p] if w >= 0):
                                continue
                            for corner in (0, 4, 20, 24):
                                if not s.occupied(corner):
                                    nxt = add(s, actions, Action("force", foe, victim, dst, corner))
                                    if nxt:
                                        yield from force_corners(*nxt, tuple(i for i in remaining if i != victim))
                    yield from force_corners(ss, aa)
                else:
                    yield from ordinary(ss, aa, True)

    yield from variants()


def legal_turns(position: Position, prefix: tuple[Action, ...] = (),
                check: Callable[[], None] = lambda: None,
                skip_next: set[Action] | None = None) -> Iterator[Turn]:
    """Apply existential constraints without ever forcing use of a hero.

    Feasibility is tested over complete turns, independently of the entered
    prefix. If the user elects to activate a hero, its newly possible actions
    must also obey the opposing constraints.
    """
    if position.winner is not None:
        return
    p, foe = position.player, 1 - position.player
    power = position.powers[p]
    need_up = position.powers[foe] == 26 and (any(position.heights) or power == 46)
    constraint = position.adonis if position.adonis and position.adonis[0] == p else None
    if not need_up and constraint is None:
        yield from _raw_turns(position, prefix, check, skip_next)
        return

    def meets_adonis(turn):
        if constraint is None:
            return True
        player, worker, owner = constraint
        dst = turn.after.workers[player][worker]
        return dst >= 0 and any(v in NEIGHBORS[dst] for v in turn.after.workers[owner] if v >= 0)

    def raw(mode, entered=(), skip=None):
        return _raw_turns(position, entered, check, skip, mode)

    normal_up = need_up and any(_ascended(position, t.actions) for t in raw(False))
    normal_adonis = constraint is not None and any(meets_adonis(t) for t in raw(False)
                          if not normal_up or _ascended(position, t.actions))
    activated = bool(prefix and prefix[0].kind == "activate")
    if not activated:
        for turn in raw(False, prefix, skip_next):
            if (not normal_up or _ascended(position, turn.actions)) and (not normal_adonis or meets_adonis(turn)):
                yield turn
    if power >= 46 and not position.hero_used[p] and (not prefix or activated):
        # End-of-turn dome/removal subsets cannot introduce a new ascent or
        # geometric adjacency. Avoid enumerating their exponentially large
        # subsets just to establish feasibility of movement constraints.
        active_up = normal_up or (need_up and power in (46, 48, 49, 53) and
                                  any(_ascended(position, t.actions) for t in raw(True)))
        active_adonis = normal_adonis or (constraint is not None and power in (46, 48, 49, 51, 53) and any(meets_adonis(t) for t in raw(True)
                                if not active_up or _ascended(position, t.actions)))
        for turn in raw(True, prefix, skip_next):
            if (not active_up or _ascended(position, turn.actions)) and (not active_adonis or meets_adonis(turn)):
                yield turn


def _ascended(position, actions):
    s = position
    for a in actions:
        if a.kind == "move" and a.player == position.player and s.heights[a.target] > s.heights[a.source]:
            return True
        s = _apply(s, a)
    return False


def next_actions(position, prefix=(), check=lambda: None):
    """Find each possible next action once, pruning its remaining subtree."""
    options = set()
    complete = None
    for turn in legal_turns(position, tuple(prefix), check, options):
        if len(turn.actions) == len(prefix):
            complete = turn
        else:
            options.add(turn.actions[len(prefix)])
    return sorted(options, key=lambda a: (a.kind, a.player, a.worker, a.target)), complete


def validate_turn(position, actions):
    actions = tuple(actions)
    turn = next((t for t in legal_turns(position, actions) if t.actions == actions), None)
    if turn is None:
        raise ValueError("Ce tour ne respecte pas les règles ou n'est pas complet.")
    return turn


def terminal_position(position, check=lambda: None):
    if position.winner is not None:
        return position
    checked = _chronus(position)
    if checked.winner is not None:
        return checked
    if next(legal_turns(position, check=check), None) is None:
        return replace(position, winner=1 - position.player, reason="Aucun tour complet légal")
    return position


def validate_setup(workers, powers, player):
    pos = Position(workers=tuple(tuple(ws) for ws in workers), powers=tuple(powers), player=player)
    pos.validate()
    if any(len(ws) != 2 for ws in workers) or any(w < 0 for ws in workers for w in ws):
        raise ValueError("Placez les quatre bâtisseurs initiaux.")
    if powers[0] and powers[0] == powers[1]:
        raise ValueError("Les joueurs doivent choisir des pouvoirs différents.")
    for p in (0, 1):
        if powers[p] == 19:
            a, b = workers[p]
            opposite = (a % 5 == 0 and b % 5 == 4) or (b % 5 == 0 and a % 5 == 4) or (a // 5 == 0 and b // 5 == 4) or (b // 5 == 0 and a // 5 == 4)
            if not opposite:
                raise ValueError("Eros doit placer ses bâtisseurs sur deux bords opposés.")
    return pos

"""Placement search, isolated from both live game state and Qt.

The second player responds to observed pieces. The first player anticipates
opponent replies and never reads their future placement. A cheap exhaustive
screen is followed by deadline-bounded game search on the selected openings.
"""
from __future__ import annotations

import itertools
import time
from dataclasses import dataclass

from .engine import Position, validate_setup
from .extra import initial_count
from .setup import reference_setup
from .search import Interrupted, evaluate, search


def placement_first(powers, first_player):
    """Bia overrides placement order, not who takes the first game turn."""
    for power in (13,43):
        if power in powers:return powers.index(power)
    if 39 in powers:return 1-powers.index(39)
    return first_player


def legal_pairs(power, excluded=()):
    cells = [c for c in range(25) if c not in excluded]
    # Selene's female worker has a distinct identity. Do not collapse the
    # assignments (a,b) and (b,a) for this power.
    pairs = itertools.permutations(cells, 2) if power == 28 else itertools.combinations(cells, initial_count(power))
    for group in pairs:
        if power == 19:
            a,b=group
            opposite = (a % 5 == 0 and b % 5 == 4) or (b % 5 == 0 and a % 5 == 4) or (
                a // 5 == 0 and b // 5 == 4) or (b // 5 == 0 and a // 5 == 4)
            if not opposite:
                continue
        yield group


@dataclass(frozen=True)
class PlacementAdvice:
    robot_cells: tuple[int, int]
    opponent_cells: tuple[int, int] | None
    first_to_place: int
    score: float | None
    placements_evaluated: int
    replies_evaluated: int
    elapsed: float
    status: str
    refined: bool = False


def _priority(pair, power, opponent=()):
    # Ordering only: never presented as a game-theoretic score. Favor access
    # to the centre while also considering Bia's two-step elimination geometry.
    central = sum(4 - abs(c % 5 - 2) - abs(c // 5 - 2) for c in pair)
    spread = abs(pair[0] % 5 - pair[1] % 5) + abs(pair[0] // 5 - pair[1] // 5)
    value = 3 * central + min(spread, 3)
    if power == 13:
        for src in pair:
            for victim in opponent:
                dx, dy = victim % 5 - src % 5, victim // 5 - src // 5
                if (dx or dy) and dx in (-2, 0, 2) and dy in (-2, 0, 2):
                    between = (src // 5 + dy // 2) * 5 + src % 5 + dx // 2
                    if between not in pair and between not in opponent:
                        value += 20
    return value


def _opening(powers, first, robot, opponent, extra=None):
    return reference_setup((opponent,robot),powers,first,extra)


def _probe(position, seconds, cancelled):
    from .native import eligible, library_path, native_search
    if eligible(position) and library_path().exists():
        result = native_search(position, seconds, cancelled=cancelled)
    else:
        result = search(position, seconds, cancelled=cancelled)
    if result.score is None:
        return None
    return float(result.score if position.player == 1 else -result.score)


def choose_placement(powers, first_player, seconds, opponent_cells=None,
                     cancelled=lambda: False, publish=lambda advice: None, extra=None):
    start = time.monotonic()
    if not .1 <= seconds <= 600:
        raise ValueError("Le budget du placement doit être compris entre 0,1 et 600 secondes.")
    powers = tuple(powers)
    Position(powers=powers, player=first_player).validate()
    if powers[0] and powers[0] == powers[1]:
        raise ValueError("Les joueurs doivent choisir des pouvoirs différents.")
    first = placement_first(powers, first_player)
    deadline = start + seconds
    if first == 0:
        if opponent_cells is None or len(opponent_cells) != initial_count(powers[0]):
            raise ValueError("Placez d'abord vos deux bâtisseurs : le robot se place après vous.")
        opponent_cells = tuple(opponent_cells)
        if len(set(opponent_cells)) != initial_count(powers[0]) or any(type(c) is not int or not 0 <= c < 25 for c in opponent_cells):
            raise ValueError("Vos deux bâtisseurs doivent occuper deux cases distinctes du plateau.")
        if opponent_cells not in set(legal_pairs(powers[0])) and tuple(reversed(opponent_cells)) not in set(legal_pairs(powers[0])):
            raise ValueError("Votre placement ne respecte pas les contraintes de votre pouvoir (Eros).")
        candidates = list(legal_pairs(powers[1], opponent_cells))
    else:
        # Even if the UI has future human positions entered, do not peek.
        opponent_cells = None
        candidates = list(legal_pairs(powers[1]))
    candidates.sort(key=lambda pair: (-_priority(pair, powers[1], opponent_cells or ()), pair))
    if not candidates:
        raise ValueError("Aucun placement légal disponible pour le robot.")
    fallback = candidates[0]
    if opponent_cells:
        _opening(powers,first_player,fallback,opponent_cells,extra).validate()
    best = PlacementAdvice(fallback, opponent_cells, first, None, 0, 0, 0,
                           "Premier placement légal ; comparaison en cours")
    publish(best)

    def check(end=deadline):
        if cancelled() or time.monotonic() >= end:
            raise Interrupted

    rows = []
    total_replies = 0
    screened = 0
    best_value = float("-inf")
    # Reserve most of the budget for rules-aware search. In the first-player
    # case the geometric screen is an alpha-beta maximin over legal replies.
    screen_end = min(deadline, start + max(.02, seconds * .25))
    try:
        for pair in candidates:
            check(screen_end)
            replies = [opponent_cells] if opponent_cells else list(legal_pairs(powers[0], pair))
            replies.sort(key=lambda reply: (-_priority(reply, powers[0], pair), reply))
            worst = float("inf")
            worst_reply = replies[0]
            exact = True
            for reply in replies:
                check(screen_end)
                value = float(evaluate(_opening(powers, first_player, pair, reply,extra), 1))
                total_replies += 1
                if value < worst:
                    worst, worst_reply = value, reply
                if first == 1 and worst <= best_value:
                    exact = False
                    break
            screened += 1
            rows.append((worst, pair, worst_reply, exact))
            if worst > best_value:
                best_value = worst
                best = PlacementAdvice(pair, worst_reply, first, worst, screened, total_replies,
                                       time.monotonic() - start,
                                       "Comparaison géométrique ; approfondissement avec les pouvoirs à venir")
                publish(best)
    except Interrupted:
        if cancelled():
            raise

    if cancelled():
        raise Interrupted
    # Robot second: deepen every legal pair. Robot first: use a beam of good
    # screened pairs plus unexamined high-priority pairs if screening timed out.
    if first == 0:
        selected = candidates
    else:
        ranked = [row[1] for row in sorted(rows, key=lambda row: (-row[0], candidates.index(row[1])))]
        selected = list(dict.fromkeys([best.robot_cells] + ranked[:10] + candidates[:4]))[:12]
    probes = []
    for pair in selected:
        if opponent_cells:
            replies = [opponent_cells]
        else:
            # Sample the most challenging geometric replies and Bia attack
            # replies. Game search determines their score with both powers.
            replies = list(legal_pairs(powers[0], pair))
            replies.sort(key=lambda reply: (evaluate(_opening(powers, first_player, pair, reply,extra), 1),
                                            -_priority(reply, powers[0], pair), reply))
            replies = replies[:12]
        probes.append((pair, replies))
    jobs = sum(len(replies) for _, replies in probes)
    remaining = max(0, deadline - time.monotonic())
    per_probe = max(.001, remaining * .9 / max(1, jobs))
    searched_rows = 0
    refined_value = float("-inf")
    # Publish only fully examined rows: an incomplete min over replies could
    # otherwise make an unsafe first-player placement look artificially good.
    try:
        for pair, replies in probes:
            check()
            worst = float("inf")
            worst_reply = replies[0]
            informed = True
            for reply in replies:
                check()
                budget = min(per_probe, deadline - time.monotonic())
                if budget < .001:
                    raise Interrupted
                value = _probe(_opening(powers, first_player, pair, reply,extra), budget, cancelled)
                total_replies += 1
                if value is None:
                    informed = False
                    continue
                if value < worst:
                    worst, worst_reply = value, reply
            if not informed:
                continue
            searched_rows += 1
            if worst > refined_value:
                refined_value = worst
                best = PlacementAdvice(pair, worst_reply, first, worst, searched_rows, total_replies,
                                       time.monotonic() - start,
                                       "Estimation avec les deux pouvoirs" if first == 0 else
                                       "Estimation du pire résultat parmi les réponses adverses approfondies",
                                       True)
                publish(best)
    except Interrupted:
        if cancelled():
            raise
    if best.refined:
        status = (f"{searched_rows}/{len(selected)} placements approfondis contre votre placement" if first == 0 else
                  f"{searched_rows}/{len(selected)} placements approfondis ; jusqu'à 12 réponses adverses testées chacun")
    else:
        status = f"{screened}/{len(candidates)} placements comparés géométriquement ; budget insuffisant pour approfondir"
    return PlacementAdvice(best.robot_cells, best.opponent_cells, first, best.score, best.placements_evaluated,
                           total_replies, time.monotonic() - start, status, best.refined)

"""Deadline-bounded, adversarial search, independent of the native app."""
from __future__ import annotations

import time
from dataclasses import dataclass, replace

from .engine import NEIGHBORS, Position, Turn, legal_turns, validate_setup
from .powers import POWERS

MATE = 100_000


class Interrupted(Exception):
    pass


@dataclass
class Analysis:
    turn: Turn | None
    score: int | None
    depth: int
    nodes: int
    elapsed: float
    proven: bool = False
    complete_depth: bool = False
    variation: tuple[Turn, ...] = ()
    status: str = ""
    searching_depth: int = 0
    stop_reason: str = ""
    engine: str = "Python"


def evaluate(s: Position, player: int) -> int:
    if s.winner is not None:
        return MATE if s.winner == player else -MATE

    def strength(p):
        total = 0
        power = s.powers[p]
        foe = 1 - p
        for i, src in enumerate(s.workers[p]):
            if src < 0:
                continue
            h = s.heights[src]
            total += 80 + (0, 16, 52, 65)[h]
            total += 2 * (4 - abs(src % 5 - 2) - abs(src // 5 - 2))
            for dst in NEIGHBORS[src]:
                if s.occupied(dst):
                    continue
                delta = s.heights[dst] - h
                if delta > 1 or (s.powers[foe] == 37 and delta < 0):
                    continue
                if p == s.player and s.athena_lock and delta > 0:
                    continue
                total += 2 + 2 * max(0, delta)
                if s.heights[dst] == 3 and h == 2:
                    if s.powers[foe] != 20 or dst % 5 not in (0, 4) and dst // 5 not in (0, 4):
                        total += 100
                if power == 9 and delta <= -2:
                    total += 100
        # A remaining hero is a resource, not a guaranteed win.
        if power >= 46 and not s.hero_used[p]:
            total += 12
        if power == 16:
            total += 12 * sum(h == 3 and s.domes & (1 << i) != 0 for i, h in enumerate(s.heights))
        return total

    return strength(player) - strength(1 - player)


def search(position: Position, seconds: float, cancelled=lambda: False,
           publish=lambda result: None, max_depth=64) -> Analysis:
    if not 0.001 <= seconds <= 3600:
        raise ValueError("Le budget doit être compris entre 0,001 et 3600 secondes.")
    started = time.monotonic()
    deadline = started + seconds
    nodes = 0
    table = {}
    root = position.player

    def check():
        if cancelled() or time.monotonic() >= deadline:
            raise Interrupted

    def result(turn, score, depth, variation=(), complete=False, status=""):
        return Analysis(turn, score, depth, nodes, time.monotonic() - started,
                        score is not None and (score >= MATE - 1000 or (complete and score <= -MATE + 1000)),
                        complete, variation, status)

    if position.winner is not None:
        return replace(result(None, evaluate(position, root), 0, complete=True, status=position.reason), stop_reason="terminal")
    try:
        fallback = next(legal_turns(position, check=check), None)
    except Interrupted:
        return replace(result(None, None, 0, status="Délai atteint avant de trouver un tour légal ; augmenter le budget."), stop_reason="cancelled" if cancelled() else "timeout")
    if fallback is None:
        return replace(result(None, -MATE, 0, complete=True, status="Aucun tour complet légal"), stop_reason="terminal")
    best = result(fallback, None, 0, (fallback,), status="Tour légal de secours")
    publish(best)

    def ordered(s, preferred=None):
        # Batches avoid materialising millions of optional hero turns before
        # evaluating the first one. Every legal child remains searchable.
        seen = set()
        if preferred is not None:
            seen.add(preferred.after)
            yield preferred
        batch = []
        for turn in legal_turns(s, check=check):
            if turn.after in seen:
                continue
            seen.add(turn.after)
            batch.append(turn)
            if len(batch) >= 24:
                batch.sort(key=lambda t: evaluate(t.after, s.player), reverse=True)
                yield from batch
                batch.clear()
        batch.sort(key=lambda t: evaluate(t.after, s.player), reverse=True)
        yield from batch

    def minimax(s, depth, alpha, beta, ply):
        nonlocal nodes
        check()
        nodes += 1
        if s.winner is not None:
            return (MATE - ply if s.winner == root else -MATE + ply), ()
        key = (s, ply)
        cached = table.get(key)
        original_a, original_b = alpha, beta
        preferred = cached[3][0] if cached and cached[3] else None
        if cached and cached[0] >= depth:
            _, value, flag, line = cached
            if flag == "exact":
                return value, line
            if flag == "lower":
                alpha = max(alpha, value)
            else:
                beta = min(beta, value)
            if alpha >= beta:
                return value, line
        if depth == 0:
            # Immobility is a loss, including at the search horizon.
            if next(legal_turns(s, check=check), None) is None:
                return (-MATE + ply if s.player == root else MATE - ply), ()
            return evaluate(s, root), ()
        maximizing = s.player == root
        value = -MATE * 2 if maximizing else MATE * 2
        line = ()
        found = False
        for turn in ordered(s, preferred):
            found = True
            v, tail = minimax(turn.after, depth - 1, alpha, beta, ply + 1)
            if (maximizing and v > value) or (not maximizing and v < value):
                value, line = v, (turn,) + tail
            if maximizing:
                alpha = max(alpha, value)
            else:
                beta = min(beta, value)
            if alpha >= beta:
                break
        if not found:
            value = -MATE + ply if maximizing else MATE - ply
        flag = "upper" if value <= original_a else "lower" if value >= original_b else "exact"
        if len(table) < 150_000:
            table[key] = (depth, value, flag, line)
        return value, line

    attempted_depth = 0
    for depth in range(1, max_depth + 1):
        attempted_depth = depth
        publish(replace(best, searching_depth=depth, elapsed=time.monotonic() - started, nodes=nodes))
        candidate = None
        candidate_value = -MATE * 2
        alpha = -MATE * 2
        try:
            preferred = best.turn
            for turn in ordered(position, preferred):
                v, tail = minimax(turn.after, depth - 1, alpha, MATE * 2, 1)
                if v > candidate_value:
                    candidate_value, candidate = v, (turn,) + tail
                alpha = max(alpha, v)
                if v == MATE - 1:
                    best = result(turn, v, depth, (turn,) + tail, True, "Victoire immédiate démontrée")
                    best.stop_reason = "proof"
                    publish(best)
                    return best
                # During the first iteration, progressively improve the
                # fallback. Later partial depths never overwrite a full depth.
                if depth == 1:
                    best = result(candidate[0], candidate_value, 1, candidate, False,
                                  "Évaluation provisoire : profondeur 1 incomplète")
            best = result(candidate[0], candidate_value, depth, candidate, True,
                          "Victoire forcée démontrée" if candidate_value >= MATE - 1000 else
                          "Défaite forcée démontrée" if candidate_value <= -MATE + 1000 else
                          f"Profondeur {depth} terminée")
            publish(best)
            if best.proven:
                break
        except Interrupted:
            break
    best.elapsed = time.monotonic() - started
    best.searching_depth = attempted_depth if attempted_depth > best.depth else 0
    best.stop_reason = ("proof" if best.proven else "cancelled" if cancelled() else
                        "timeout" if time.monotonic() >= deadline else "depth_limit")
    if best.stop_reason == "timeout":
        best.status += " — temps imparti épuisé"
    return best


@dataclass
class PowerAdvice:
    power: int | None
    scores: dict[int, float]
    rounds: int
    elapsed: float
    status: str


def choose_power(available, opponent: int | None, player: int, seconds: float,
                 first=0, cancelled=lambda: False, publish=lambda advice: None):
    """Compare powers on identical reference openings under one total budget.

    Only fully completed equal-budget comparison rounds change the ranking. This is an
    opening estimate, not a solved ranking of powers or a game-tree proof.
    If the opponent is unknown, use the worst tested matchup (maximin).
    """
    start = time.monotonic()
    if not 0.1 <= seconds <= 3600:
        raise ValueError("Budget de choix du pouvoir invalide.")
    deadline = start + seconds
    candidates = [p for p in available if p in POWERS and POWERS[p].supported and p != opponent]
    if not candidates:
        return PowerAdvice(None, {}, 0, 0, "Aucun pouvoir disponible")
    foes = [opponent] if opponent is not None else [0] + candidates
    openings = [((5, 9), (15, 19)), ((1, 3), (21, 23)), ((6, 8), (16, 18))]
    sums = {p: [] for p in candidates}
    complete = 0
    advice = PowerAdvice(None, {}, 0, 0, "Comparaison en cours")
    # Every matchup gets the same reflection time. Search depths can differ;
    # scores remain explicitly estimates. Reserve time for publication/FFI.
    jobs = sum(sum(foe != power for foe in foes) for power in candidates)
    per_probe = max(0.001, seconds * 0.9 / max(1, jobs * len(openings)))
    # No undocumented static tier list: choose only after one fair comparison.
    for opening in openings:
        round_scores = {}
        for power in candidates:
            matchup_values = []
            for foe in foes:
                if foe == power:
                    continue
                if cancelled() or time.monotonic() >= deadline:
                    advice.elapsed = time.monotonic() - start
                    advice.status = ("Estimation sur les comparaisons terminées" if complete else
                                     "Budget insuffisant pour comparer tous les pouvoirs ; augmenter le délai ou réduire la sélection")
                    return advice
                pair = (power, foe) if player == 0 else (foe, power)
                workers = opening
                # Eros requires opposing edges; this common placement is legal
                # for all supported initial powers, including both sides Eros.
                if 19 in pair:
                    workers = ((5, 9), (15, 19))
                pos = validate_setup(workers, pair, first)
                budget = min(per_probe, deadline - time.monotonic())
                if budget < 0.001:
                    advice.elapsed = time.monotonic() - start
                    advice.status = "Budget terminé ; seules les comparaisons complètes sont retenues"
                    return advice
                from .native import eligible, library_path, native_search
                if eligible(pos) and library_path().exists() and budget >= 0.01:
                    analysis = native_search(pos, budget, cancelled=cancelled)
                else:
                    analysis = search(pos, budget, cancelled=cancelled)
                value = analysis.score
                if value is None:
                    value = evaluate(pos, player)
                elif pos.player != player:
                    value = -value
                matchup_values.append(value)
            round_scores[power] = min(matchup_values) if matchup_values else evaluate(Position(), player)
        for power, value in round_scores.items():
            sums[power].append(value)
        complete += 1
        averages = {p: sum(vs) / len(vs) for p, vs in sums.items()}
        selected = max(candidates, key=lambda p: (averages[p], -p))
        advice = PowerAdvice(selected, averages, complete, time.monotonic() - start,
                             "Estimation d'ouverture ; les placements réels peuvent changer le classement")
        publish(advice)
        if time.monotonic() >= deadline:
            break
    return advice

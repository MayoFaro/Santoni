from dataclasses import replace

import pytest

from santorini.engine import Action, Position, coord, legal_turns, next_actions, terminal_position, validate_setup, validate_turn


def position(power=0, foe=0, heights=None, workers=((6, 8), (16, 18)), domes=0, **extra):
    hs = [0] * 25
    for cell, height in (heights or {}).items():
        hs[cell] = height
    return Position(tuple(hs), domes, workers, (power, foe), **extra)


def first_move(turn):
    return next((a for a in turn.actions if a.kind == "move"), None)


def test_standard_complete_turn_and_no_side_effects():
    s = Position()
    turns = list(legal_turns(s))
    assert len(turns) == 80
    for t in turns:
        assert [a.kind for a in t.actions] == ["move", "build"]
        assert t.after.player == 1
        assert sum(t.after.heights) == 1
        assert validate_turn(s, t.actions) == t
    assert s.heights == (0,) * 25


def test_win_stops_before_construction_and_forcing_does_not_win():
    s = position(heights={6: 2, 7: 3})
    wins = [t for t in legal_turns(s) if first_move(t).target == 7]
    assert wins
    assert all(t.after.winner == 0 and len(t.actions) == 1 for t in wins if first_move(t).source == 6)
    s = position(power=1, heights={6: 3, 7: 2}, workers=((6, 8), (7, 18)))
    swap = next(t for t in legal_turns(s) if first_move(t).target == 7)
    assert swap.after.workers[1][0] == 6
    assert swap.after.winner is None


def test_no_climb_two_no_dome_entry_and_loss_by_immobility():
    s = position(heights={7: 2}, domes=1 << 5)
    assert not any(first_move(t).source == 6 and first_move(t).target in (5, 7) for t in legal_turns(s))
    closed = position(domes=sum(1 << i for i in range(25) if i not in (6, 8, 16, 18)))
    assert not list(legal_turns(closed))
    assert terminal_position(closed).winner == 1


def test_move_without_available_build_is_not_legal():
    # Only destination is empty; departing origin becomes buildable, except
    # when Limus protects it. Moving alone does not make a legal turn.
    s = position(foe=23, workers=((0, -1), (5, -1)), domes=sum(1 << i for i in range(25) if i not in (0, 1, 5)))
    assert not list(legal_turns(s))


def test_apollo_and_minotaur():
    s = position(1, workers=((6, 8), (7, 18)))
    swap = next(t for t in legal_turns(s) if first_move(t).source == 6 and first_move(t).target == 7)
    assert swap.after.workers == ((7, 8), (6, 18))
    push = position(8, workers=((6, 0), (7, 18)))
    t = next(t for t in legal_turns(push) if first_move(t).target == 7)
    assert t.after.workers[1][0] == 8
    blocked = replace(push, domes=1 << 8)
    assert not any(first_move(t).target == 7 for t in legal_turns(blocked))


def test_artemis_never_returns_to_origin_and_stops_on_win():
    s = position(2, heights={6: 2, 7: 3})
    for t in legal_turns(s):
        moves = [a for a in t.actions if a.kind == "move"]
        if len(moves) == 2:
            assert moves[1].target != moves[0].source
            assert not (moves[0].source == 6 and moves[0].target == 7)


def test_athena_lock_is_one_turn_and_counts_first_move_ascent():
    s = position(3, heights={7: 1, 16: 1, 17: 2})
    t = next(t for t in legal_turns(s) if first_move(t).source == 6 and first_move(t).target == 7)
    assert t.after.athena_lock
    reply = list(legal_turns(t.after))
    assert reply
    assert all(t.after.heights[first_move(r).target] <= t.after.heights[first_move(r).source] for r in reply)
    assert all(not r.after.athena_lock for r in reply)


def test_atlas_demeter_hephaestus():
    assert any(a.kind == "dome" and t.after.heights[a.target] == 0 for t in legal_turns(position(4)) for a in t.actions)
    demeter = list(legal_turns(position(5)))
    assert any(len(t.actions) == 3 for t in demeter)
    assert all(t.actions[1].target != t.actions[2].target for t in demeter if len(t.actions) == 3)
    smith = list(legal_turns(position(6, heights={7: 2})))
    for t in smith:
        if len(t.actions) == 3:
            assert t.actions[1].target == t.actions[2].target
            assert t.actions[1].kind == t.actions[2].kind == "build"
            assert t.after.heights[t.actions[2].target] <= 3


def test_hermes_can_stay_and_move_both_without_climbing():
    s = position(7)
    turns = list(legal_turns(s))
    assert any([a.kind for a in t.actions] == ["build"] for t in turns)
    assert any(len({a.worker for a in t.actions if a.kind == "move"}) == 2 for t in turns)
    assert all(s.heights[a.source] == s.heights[a.target] for t in turns for a in t.actions if a.kind == "move")


def test_pan_prometheus_and_hero_achilles():
    s = position(9, heights={6: 2})
    assert any(t.after.winner == 0 and len(t.actions) == 1 for t in legal_turns(s))
    s = position(10, heights={7: 1})
    for t in legal_turns(s):
        if t.actions[0].kind == "build":
            move = first_move(t)
            before = list(s.heights)
            before[t.actions[0].target] += 1
            assert before[move.target] <= before[move.source]
    achilles = position(46, heights={7: 1})
    assert any(t.actions[0].kind == "activate" and first_move(t).target == 7 for t in legal_turns(achilles))


def test_prefix_options_match_complete_turns():
    for p in [0, 1, 2, 4, 5, 6, 10, 12, 24, 28, 46, 47, 49, 51, 52, 54, 55]:
        s = position(p)
        options, complete = next_actions(s)
        assert complete is None
        assert set(options) == {t.actions[0] for t in legal_turns(s)}
        prefix = (options[0],)
        options2, complete = next_actions(s, prefix)
        expected = list(legal_turns(s, prefix))
        assert set(options2) == {t.actions[1] for t in expected if len(t.actions) > 1}
        assert bool(complete) == any(len(t.actions) == 1 for t in expected)


def test_persephone_forces_ascent_only_when_a_complete_turn_exists():
    s = position(foe=26, heights={7: 1})
    turns = list(legal_turns(s))
    assert turns
    assert all(s.heights[first_move(t).target] > s.heights[first_move(t).source] for t in turns)
    options, _ = next_actions(s)
    assert set(options) == {t.actions[0] for t in turns}
    flat = position(foe=26)
    assert list(legal_turns(flat))


def test_hero_single_use_and_jason_third_builder():
    s = position(51)
    turns = list(legal_turns(s))
    special = next(t for t in turns if t.actions[0].kind == "activate")
    assert special.after.hero_used[0]
    assert len(special.after.workers[0]) == 3
    returned = replace(special.after, player=0)
    assert not any(t.actions[0].kind == "activate" for t in legal_turns(returned))


def test_bellerophon_can_win_by_moving_one_to_three():
    s = position(49, heights={6: 1, 7: 3})
    assert not any(first_move(t).target == 7 for t in legal_turns(s) if t.actions[0].kind != "activate")
    assert any(t.after.winner == 0 and first_move(t).target == 7 for t in legal_turns(s))


def test_hera_blocks_border_movement_victory():
    s = position(9, 20, heights={6: 2})
    assert not any(t.after.winner == 0 and first_move(t).target == 0 for t in legal_turns(s))
    assert any(t.after.winner == 0 for t in legal_turns(s))


def test_eros_and_urania_setup_and_wrap():
    with pytest.raises(ValueError, match="Eros"):
        validate_setup(((6, 8), (16, 18)), (19, 0), 0)
    validate_setup(((5, 9), (15, 19)), (19, 0), 0)
    s = position(45, workers=((0, 6), (16, 18)))
    assert any(first_move(t).source == 0 and first_move(t).target == 24 for t in legal_turns(s))


def test_adonis_checks_complete_turn_and_expires():
    s = position(workers=((6, 8), (11, 18)), adonis=(0, 0, 1))
    turns = list(legal_turns(s))
    assert turns
    assert all(any(v in __import__('santorini.engine', fromlist=['NEIGHBORS']).NEIGHBORS[t.after.workers[0][0]] for v in t.after.workers[1]) for t in turns)
    assert all(t.after.adonis is None for t in turns)
    options, _ = next_actions(s)
    assert set(options) == {t.actions[0] for t in turns}


def test_opponent_power_cannot_force_hero_activation():
    s = position(49, 26, heights={7: 2})
    turns = list(legal_turns(s))
    assert any(t.actions[0].kind != "activate" for t in turns)
    active = [t for t in turns if t.actions[0].kind == "activate"]
    assert active
    assert all(first_move(t).target == 7 for t in active)
    s = position(49, heights={12: 2}, workers=((6, 8), (18, 24)), adonis=(0, 0, 1))
    turns = list(legal_turns(s))
    assert any(t.actions[0].kind != "activate" for t in turns)
    assert any(t.actions[0].kind == "activate" for t in turns)
    opts, _ = next_actions(s)
    assert set(opts) == {t.actions[0] for t in turns}


def test_five_complete_towers_win_during_optional_construction():
    hs = {0: 3, 4: 3, 20: 3, 24: 3, 7: 3}
    s = position(5, 16, heights=hs, domes=sum(1 << i for i in (0, 4, 20, 24)))
    wins = [t for t in legal_turns(s) if t.after.winner == 1]
    assert wins
    assert all(t.actions[-1].kind == "dome" for t in wins)


@pytest.mark.parametrize("power", [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 19, 20, 21, 22, 23, 24, 26, 27, 28, 30, 37, 45, 46, 47, 49, 51, 52, 53, 54, 55])
def test_generated_states_valid(power):
    s = position(power)
    for t in legal_turns(s):
        t.after.validate()

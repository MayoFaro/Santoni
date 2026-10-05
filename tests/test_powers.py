from dataclasses import replace

from santorini.engine import Action, NEIGHBORS, Position, legal_turns, next_actions, validate_turn


def state(power, foe=0, workers=((6, 8), (16, 18)), heights=None, domes=0):
    hs = [0] * 25
    for cell, h in (heights or {}).items():
        hs[cell] = h
    return Position(tuple(hs), domes, workers, (power, foe))


def moved(turn):
    return next(a for a in turn.actions if a.kind == "move")


def test_aphrodite_endpoint_restriction():
    s = state(0, 11, workers=((6, 8), (7, 18)))
    for turn in legal_turns(s):
        a = moved(turn)
        assert a.target in NEIGHBORS[7] or a.target in NEIGHBORS[18]
    assert not any(moved(t).source == 6 and moved(t).target == 0 for t in legal_turns(s))


def test_ares_removes_only_unoccupied_block_near_unmoved_worker():
    s = state(12, heights={7: 1})
    turn = next(t for t in legal_turns(s) if any(a.kind == "remove" for a in t.actions))
    removal = turn.actions[-1]
    assert removal.worker != moved(turn).worker
    assert removal.target in NEIGHBORS[turn.after.workers[0][removal.worker]]
    blocked = replace(s, domes=1 << 7)
    assert not any(a.kind == "remove" and a.target == 7 for t in legal_turns(blocked) for a in t.actions)


def test_bia_eliminates_worker_beyond_destination():
    s = state(13, workers=((6, 0), (8, 18)))
    turn = next(t for t in legal_turns(s) if moved(t).source == 6 and moved(t).target == 7)
    assert turn.after.workers[1][0] == -1


def test_charon_forces_before_movement_without_forced_win():
    s = state(15, workers=((6, 0), (7, 18)), heights={5: 3})
    turn = next(t for t in legal_turns(s) if t.actions[0].kind == "force" and t.actions[0].target == 5)
    assert turn.after.workers[1][0] == 5
    assert turn.after.winner is None
    assert moved(turn).source == 6


def test_eros_wins_by_reunion_on_level_one():
    s = state(19, heights={7: 1})
    turn = next(t for t in legal_turns(s) if moved(t).source == 6 and moved(t).target == 7)
    assert turn.after.winner == 0
    assert len(turn.actions) == 1


def test_hestia_second_build_is_never_on_perimeter():
    s = state(21)
    turns = list(legal_turns(s))
    assert any(len(t.actions) == 3 for t in turns)
    assert all(t.actions[-1].target % 5 not in (0, 4) and t.actions[-1].target // 5 not in (0, 4)
               for t in turns if len(t.actions) == 3)


def test_hypnus_blocks_unique_highest_worker():
    s = state(0, 22, heights={6: 2, 8: 1})
    turns = list(legal_turns(s))
    assert turns
    assert all(moved(t).worker == 1 for t in turns)


def test_limus_allows_only_complete_domes_in_protected_area():
    s = state(0, 23, heights={12: 3})
    turns = list(legal_turns(s))
    protected = {cell for src in s.workers[1] for cell in NEIGHBORS[src]}
    assert any(a.target == 12 and a.kind == "dome" for t in turns for a in t.actions)
    assert all(a.kind == "dome" and s.heights[a.target] == 3 for t in turns for a in t.actions
               if a.kind in ("build", "dome") and a.target in protected)


def test_medusa_mandatory_lower_neighbor_elimination_and_build():
    s = state(24, workers=((6, 0), (16, 24)), heights={6: 2, 11: 2})
    turn = next(t for t in legal_turns(s) if moved(t).target == 11)
    assert turn.after.workers[1][0] == -1
    assert turn.after.heights[16] == 1
    assert [a.kind for a in turn.actions][-2:] == ["kill", "build"]


def test_poseidon_extra_builds_with_unmoved_ground_worker():
    s = state(27)
    turn = next(t for t in legal_turns(s) if len(t.actions) == 5)
    assert all(a.worker != moved(turn).worker for a in turn.actions[2:])
    assert s.heights[s.workers[0][turn.actions[-1].worker]] == 0


def test_selene_builds_remote_dome_with_female_worker():
    s = state(28)
    turn = next(t for t in legal_turns(s) if moved(t).source == 6 and moved(t).target == 5
                and t.actions[-1].worker == 1 and t.actions[-1].kind == "dome" and t.actions[-1].target == 7)
    assert 7 not in NEIGHBORS[5]
    assert turn.after.domes & (1 << 7)


def test_triton_can_continue_only_from_perimeter():
    s = state(29, workers=((0, 8), (16, 18)))
    turns = list(legal_turns(s))
    assert any(sum(a.kind == "move" for a in t.actions) >= 3 for t in turns)
    for turn in turns:
        steps = [a for a in turn.actions if a.kind == "move"]
        assert all(a.source % 5 in (0, 4) or a.source // 5 in (0, 4) for a in steps[1:])


def test_zeus_self_build_does_not_trigger_victory():
    s = state(30, heights={6: 1, 7: 2})
    turn = next(t for t in legal_turns(s) if moved(t).source == 6 and moved(t).target == 7
                and t.actions[-1].source == t.actions[-1].target)
    assert turn.after.heights[7] == 3
    assert turn.after.winner is None


def test_hades_blocks_descent_but_not_forced_displacement():
    s = state(0, 37, heights={6: 2})
    assert not any(moved(t).source == 6 for t in legal_turns(s))
    s = state(15, 37, workers=((6, 0), (7, 18)), heights={7: 2})
    assert any(t.actions[0].kind == "force" for t in legal_turns(s))


def test_atalanta_multistep_moves_and_single_use():
    s = state(48)
    options, _ = next_actions(s, (Action("activate", 0),))
    first = next(a for a in options if a.kind == "move")
    options, _ = next_actions(s, (Action("activate", 0), first))
    assert any(a.kind == "move" for a in options)
    assert any(a.kind in ("build", "dome") for a in options)


def test_heracles_optional_domes_from_both_workers_and_any_input_order():
    s = state(50)
    ordinary = next(legal_turns(s))
    prefix = (Action("activate", 0),) + ordinary.actions
    options, complete = next_actions(s, prefix)
    assert complete is not None  # zero domes is expressly legal
    domes = [a for a in options if a.kind == "dome"]
    assert len(domes) >= 2
    a, b = sorted(domes, key=lambda a: a.target, reverse=True)[:2]
    options2, _ = next_actions(s, prefix + (a,))
    assert b in options2
    turn = validate_turn(s, prefix + (a, b))
    assert turn.after.domes & (1 << a.target)
    assert turn.after.domes & (1 << b.target)


def test_medea_removes_block_under_neighbor_and_theseus_eliminates():
    s = state(52, workers=((6, 12), (17, 24)), heights={17: 1})
    turn = next(t for t in legal_turns(s) if t.actions[0].kind == "activate" and
                moved(t).worker == 0 and any(a.kind == "remove" and a.target == 17 for a in t.actions))
    assert turn.after.heights[17] == 0
    s = state(55, workers=((6, 12), (17, 24)), heights={17: 2})
    turn = next(t for t in legal_turns(s) if t.actions[0].kind == "activate" and moved(t).worker == 0)
    assert turn.after.workers[1][0] == -1


def test_odysseus_can_free_corner_for_other_victim():
    s = state(53, workers=((6, 3), (7, 4)))
    turns = legal_turns(s)
    turn = next(t for t in turns if t.actions[0].kind == "activate" and len(t.actions) >= 5
                and t.actions[1].kind == "force" and t.actions[1].worker == 1
                and t.actions[1].target == 0 and t.actions[2].kind == "force" and t.actions[2].target == 4)
    assert turn.after.workers[1] == (4, 0)


def test_polyphemus_global_domes_in_either_input_order():
    s = state(54)
    ordinary = next(legal_turns(s))
    prefix = (Action("activate", 0),) + ordinary.actions
    options, complete = next_actions(s, prefix)
    assert complete
    dome4 = next(a for a in options if a.kind == "dome" and a.target == 4)
    options2, complete = next_actions(s, prefix + (dome4,))
    assert complete
    dome1 = next(a for a in options2 if a.kind == "dome" and a.target == 1)
    turn = validate_turn(s, prefix + (dome4, dome1))
    assert turn.after.domes & (1 << 1) and turn.after.domes & (1 << 4)

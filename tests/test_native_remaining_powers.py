from dataclasses import replace
import pytest
from santorini.engine import Position,Action,validate_turn,validate_setup
from santorini.extra import Extra
from santorini.native import native_next_actions,native_turns,native_search
from santorini.storage import Session


def a(kind,p,w,src,dst):return Action(kind,p,w,src,dst)
def complete(pos,*actions):return validate_turn(pos,actions).after

def test_chaos_card_and_random_draw():
    h=[0]*25;h[7]=3
    p=Position(heights=tuple(h),powers=(14,0),extra=replace(Extra(),chaos=(4,0),deck=(3,0),discard=(8,0)))
    q=complete(p,a('move',0,0,6,11),a('dome',0,0,11,7))
    assert q.extra.event==1 and q.powers==(14,0)
    options,_=native_next_actions(q);assert {x.target for x in options}=={1,2}
    assert native_search(q,.05).turn is None
    drawn=complete(q,a('draw',0,0,-1,2));assert drawn.extra.chaos==(2,0) and drawn.player==1

def test_circe_borrows_then_returns_power():
    p=Position(powers=(17,4))
    q=complete(p,a('move',0,0,6,11),a('dome',0,0,11,7))
    assert q.extra.theft_owner==0 and q.powers==(17,4)
    options,_=native_next_actions(q,(a('move',1,0,16,21),))
    assert not any(x.kind=='dome' for x in options)
    r=complete(q,a('move',1,0,16,12),a('build',1,0,12,16))
    assert r.player==0
    # Adjacent enemy workers at start: the previous theft expires.
    r=replace(r,workers=(r.workers[0],(12,13)))
    options,_=native_next_actions(r,(a('move',0,0,11,6),))
    assert not any(x.kind=='dome' for x in options)

def test_dionysus_extra_turn_controls_enemy_and_cannot_win():
    h=[0]*25;h[7]=3;h[16]=2;h[17]=3
    p=Position(heights=tuple(h),powers=(18,0))
    q=complete(p,a('move',0,0,6,11),a('dome',0,0,11,7))
    assert q.extra.event==3 and q.player==0
    r=complete(q,a('extra_turn',0,0,-1,12))
    s=complete(r,a('move',1,0,16,17),a('build',1,0,17,12))
    assert s.winner is None and s.workers[1][0]==17 and s.player==1

def test_morpheus_saves_and_spends_materials():
    p=Position(powers=(25,0))
    q=complete(p,a('move',0,0,6,11));assert q.extra.materials==(1,0)
    q=replace(q,player=0)
    r=complete(q,a('move',0,0,11,6),a('build',0,0,6,7),a('build',0,0,6,7))
    assert r.heights[7]==2 and r.extra.materials==(0,0)

def test_aeolus_initial_and_end_direction():
    p=Position(powers=(31,0),extra=replace(Extra(),wind=13))
    options,_=native_next_actions(p);assert a('move',0,0,6,7) not in options
    opts,done=native_next_actions(p,(a('move',0,0,6,11),a('build',0,0,11,6)))
    assert done is None and {x.target for x in opts}=={6,7,8,11,13,16,17,18}

def test_charybdis_forces_without_win_and_returns_token_on_build():
    h=[0]*25;h[23]=3
    p=Position(heights=tuple(h),powers=(32,0),extra=replace(Extra(),whirlpools=((7,23),(-1,-1))))
    q=complete(p,a('move',0,0,6,7),a('build',0,0,23,22))
    assert q.workers[0][0]==23 and q.winner is None
    q=replace(q,player=0,workers=((6,8),(16,18)))
    r=complete(q,a('move',0,0,6,11),a('build',0,0,11,7));assert r.extra.whirlpools[0][0]==-1

def test_clio_coins_block_enemy_only():
    p=Position(powers=(33,0))
    q=complete(p,a('move',0,0,6,11),a('build',0,0,11,12))
    assert q.extra.coins[0]==1<<12 and q.extra.coin_count[0]==1
    options,_=native_next_actions(q);assert all(x.target!=12 for x in options)
    r=replace(q,player=0);options,_=native_next_actions(r);assert any(x.target==12 for x in options)

def test_talus_blocks_both_players():
    p=Position(powers=(34,0))
    q=complete(p,a('move',0,0,6,11),a('build',0,0,11,6),a('talus',0,0,-1,12))
    assert q.extra.talus==(12,-1)
    for player in (0,1):
        options,_=native_next_actions(replace(q,player=player));assert all(x.target!=12 for x in options)

def test_gaea_interrupts_enemy_then_resumes_second_build():
    h=[0]*25;h[7]=3
    p=Position(heights=tuple(h),powers=(5,35),extra=replace(Extra(),reserves=(0,2)))
    q=complete(p,a('move',0,0,6,11),a('dome',0,0,11,7))
    assert q.player==1 and q.extra.event==2 and q.resume
    r=complete(q,a('place',1,2,-1,2))
    assert r.player==0 and len(r.workers[1])==3 and r.extra.reserves==(0,1)
    options,done=native_next_actions(r)
    assert done and any(x.kind=='build' for x in options) and all(x.target!=7 for x in options)
    s=complete(r,a('build',0,0,11,12))
    assert not s.resume and s.player==1 and s.workers[1][2]==2
    session=Session(p).append(validate_turn(p,(a('move',0,0,6,11),a('dome',0,0,11,7))))
    session=session.append(validate_turn(q,(a('place',1,2,-1,2),))).append(validate_turn(r,(a('build',0,0,11,12),)))
    assert Session.from_dict(session.to_dict()).position==s

def test_graeae_any_builder():
    p=Position(workers=((6,8,2),(16,18)),powers=(36,0))
    q=complete(p,a('move',0,0,6,11),a('build',0,2,2,3))
    assert q.heights[3]==1

def test_harpies_sliding_is_forced_not_a_winning_move():
    p=Position(workers=((5,8),(16,18)),powers=(0,38))
    q=complete(p,a('move',0,0,5,6),a('build',0,0,7,12))
    assert q.workers[0][0]==7 # blocked by own worker at 8

def test_hecate_cancel_move_and_build_but_apollo_swaps():
    p=Position(workers=((6,8),(7,18)),powers=(0,39))
    q=complete(p,a('move',0,0,6,7));assert q.workers==p.workers and q.player==1 and q.reason=='Action annulée par Hecate'
    q=complete(p,a('move',0,0,6,11),a('build',0,0,11,7));assert q.heights==p.heights and q.workers[0][0]==11
    r=replace(p,powers=(1,39));q=complete(r,a('move',0,0,6,7),a('build',0,0,7,12));assert q.workers[1][0]==6

def test_moerae_overrides_winning_move():
    h=[0]*25;h[6]=2;h[7]=3
    p=Position(heights=tuple(h),workers=((6,8),(16,18,24)),powers=(0,40),extra=replace(Extra(),fate=(-1,1)))
    q=complete(p,a('move',0,0,6,7));assert q.winner==1 and q.reason=='Lieu du Destin (Moerae)'

def test_nemesis_swaps_maximum_or_none():
    p=Position(workers=((0,4),(20,24)),powers=(41,0))
    prefix=(a('move',0,0,0,1),a('build',0,0,1,0))
    options,done=native_next_actions(p,prefix);assert done and any(x.kind=='swap' for x in options)
    _,partial=native_next_actions(p,prefix+(a('swap',0,0,1,20),));assert partial is None
    q=complete(p,*prefix,a('swap',0,0,1,20),a('swap',0,1,4,24));assert q.workers==((20,24),(1,4))

def test_siren_forces_up_without_winning_and_order_matters():
    h=[0]*25;h[18]=3
    p=Position(heights=tuple(h),workers=((0,4),(16,17)),powers=(42,0),extra=replace(Extra(),siren=(13,-1)))
    q=complete(p,a('force',1,1,17,18),a('force',1,0,16,17));assert q.workers[1]==(17,18) and q.winner is None

def test_tartarus_loses_instead_of_wins_including_forced():
    h=[0]*25;h[6]=2;h[7]=3
    p=Position(heights=tuple(h),powers=(0,43),extra=replace(Extra(),abyss=(-1,7)))
    q=complete(p,a('move',0,0,6,7));assert q.winner==1 and q.reason=='Entrée dans l’abysse (Tartarus)'
    p=Position(workers=((0,4),(16,18)),powers=(42,43),extra=replace(Extra(),siren=(13,-1),abyss=(-1,17)))
    q=complete(p,a('force',1,0,16,17));assert q.winner==0

def test_terpsichore_all_move_then_all_build():
    p=Position(powers=(44,0));prefix=(a('move',0,0,6,11),)
    options,done=native_next_actions(p,prefix);assert not done and all(x.kind=='move' and x.worker==1 for x in options)
    q=complete(p,*prefix,a('move',0,1,8,13),a('build',0,1,13,8),a('build',0,0,11,6));assert q.heights[6]==q.heights[8]==1


def test_gaea_can_use_both_reserves_and_reach_four_workers():
    h=[0]*25;h[7]=3;h[12]=3
    p=Position(heights=tuple(h),powers=(5,35),extra=replace(Extra(),reserves=(0,2)))
    q=complete(p,a('move',0,0,6,11),a('dome',0,0,11,7))
    r=complete(q,a('place',1,2,-1,2))
    s=complete(r,a('dome',0,0,11,12))
    t=complete(s,a('place',1,3,-1,13))
    u=complete(t)
    assert len(u.workers[1])==4 and u.extra.reserves==(0,0) and not u.resume
    assert Position.from_dict(u.to_dict())==u


def test_gaea_reacts_to_dionysus_controlled_worker_and_returns_control():
    h=[0]*25;h[7]=3;h[17]=3
    p=Position(heights=tuple(h),powers=(18,35),extra=replace(Extra(),reserves=(0,2)))
    q=complete(p,a('move',0,0,6,11),a('dome',0,0,11,7))
    r=complete(q,a('decline',1,0,-1,12))
    s=complete(r);assert s.extra.event==3 and s.player==0
    t=complete(s,a('extra_turn',0,0,-1,12))
    u=complete(t,a('move',1,0,16,21),a('dome',1,0,21,17))
    assert u.extra.event==2 and u.player==1
    v=complete(u,a('place',1,2,-1,12));assert v.player==0
    w=complete(v);assert w.extra.event==3 and w.player==0


def test_chaos_draw_does_not_erase_dionysus_choice():
    h=[0]*25;h[7]=3
    p=Position(heights=tuple(h),powers=(18,14),extra=replace(Extra(),chaos=(0,4),deck=(0,3),discard=(0,8)))
    q=complete(p,a('move',0,0,6,11),a('dome',0,0,11,7))
    assert q.extra.event==1 and q.player==1
    r=complete(q,a('draw',1,0,-1,2));assert r.extra.event==3 and r.player==0
    s=complete(r,a('decline',0,0,-1,12));assert s.player==1 and s.extra.event==0


def test_circe_disables_hero_activation_without_consuming_it():
    p=Position(powers=(17,54));q=complete(p,a('move',0,0,6,11),a('build',0,0,11,6))
    options,_=native_next_actions(q)
    assert not any(x.kind=='activate' for x in options) and not q.hero_used[1]


def test_clio_protects_worker_from_bia_medusa_and_siren():
    coin=replace(Extra(),coins=(0,1<<12),coin_count=(0,1))
    p=Position(workers=((10,0),(12,24)),powers=(13,33),extra=coin)
    q=complete(p,a('move',0,0,10,11),a('build',0,0,11,6));assert q.workers[1][0]==12
    h=[0]*25;h[11]=2;h[10]=2
    p=replace(p,powers=(24,33),heights=tuple(h));q=complete(p,a('move',0,0,10,11),a('build',0,0,11,6));assert q.workers[1][0]==12
    p=replace(p,powers=(42,33),extra=replace(coin,siren=(13,-1)))
    options,_=native_next_actions(p);assert not any(x.kind=='force' and x.source==12 for x in options)


def test_hecate_cancel_does_not_cancel_following_turn():
    p=Position(workers=((6,8),(7,18)),powers=(0,39))
    q=complete(p,a('move',0,0,6,7))
    r=complete(q,a('move',1,0,7,12),a('build',1,0,12,7))
    assert r.heights[7]==1 and not r.reason


@pytest.mark.parametrize('power,field,first,second',[(40,'fate',0,18),(43,'abyss',0,24),(39,'workers',(16,18),(20,24))])
def test_secret_search_does_not_read_opposing_secret(power,field,first,second):
    base=Position(powers=(0,power),workers=((6,8),(16,18,22) if power==40 else (16,18)))
    def make(value):
        return replace(base,workers=(base.workers[0],value)) if field=='workers' else replace(base,extra=replace(Extra(),**{field:(-1,value)}))
    results=[]
    for value in (first,second):
        calls=[0]
        def cancel():calls[0]+=1;return calls[0]>120
        result=native_search(make(value),1,cancelled=cancel)
        assert not result.proven and not result.complete_depth
        if result.turn:
            assert result.turn.after.extra.fate[1]==-1 and result.turn.after.extra.abyss[1]==-1
            if power==39:assert all(c==-1 for c in result.turn.after.workers[1])
        results.append(result.turn.actions if result.turn else None)
    assert results[0]==results[1]


def test_setup_resources_cardinality_and_directions():
    p=validate_setup(((6,8),(16,18)),(35,31),0,replace(Extra(),wind=13))
    assert p.extra.reserves==(2,0)
    with pytest.raises(ValueError,match='direction'):validate_setup(((6,8),(16,18)),(31,0),0)
    p=validate_setup(((6,8,2),(16,18)),(36,0),0);assert len(p.workers[0])==3


def test_hero_resumes_after_gaea_and_cannot_dome_new_worker():
    p=Position(powers=(50,35),extra=replace(Extra(),reserves=(0,2)))
    prefix=(a('activate',0,-1,-1,-1),a('move',0,0,6,11),a('build',0,0,11,6),a('dome',0,-1,-1,7))
    q=complete(p,*prefix);assert q.extra.event==2
    r=complete(q,a('place',1,2,-1,2))
    options,done=native_next_actions(r)
    assert done and r.hero_used[0] and all(x.target!=2 for x in options)
    s=complete(r);assert s.player==1 and s.hero_used[0] and s.workers[1][2]==2


def test_artemis_second_move_starts_after_harpies_slide():
    p=Position(workers=((5,8),(16,18)),powers=(2,38))
    prefix=(a('move',0,0,5,6),)
    options,_=native_next_actions(p,prefix)
    assert any(x.kind=='move' for x in options)
    assert all(x.source==7 for x in options if x.kind=='move')


def test_hermes_attempt_into_hecate_is_not_hidden_from_input_options():
    p=Position(workers=((6,8),(7,18)),powers=(7,39))
    options,_=native_next_actions(p);assert a('move',0,0,6,7) in options
    q=complete(p,a('move',0,0,6,7));assert q.reason=='Action annulée par Hecate'


def test_urania_wind_direction_at_wrapped_edge():
    p=Position(workers=((0,8),(16,18)),powers=(45,31),extra=replace(Extra(),wind=11))
    options,_=native_next_actions(p)
    assert a('move',0,0,0,4) not in options and a('move',0,0,0,1) in options


def test_minotaur_forced_worker_loses_on_tartarus():
    p=Position(workers=((6,0),(7,24)),powers=(8,43),extra=replace(Extra(),abyss=(-1,8)))
    q=complete(p,a('move',0,0,6,7));assert q.winner==0 and q.reason=='Entrée dans l’abysse (Tartarus)'


def test_morpheus_persephone_stock_does_not_block_input_verification():
    import time
    h=[0]*25;h[12]=1
    p=Position(heights=tuple(h),powers=(25,26),extra=replace(Extra(),materials=(20,0)))
    start=time.monotonic();options,_=native_next_actions(p)
    assert time.monotonic()-start<.5
    assert options and all(h[x.target]>h[x.source] for x in options)


def test_initial_chaos_expectation_returns_score_not_chosen_card():
    from santorini.native import native_expectation
    p=Position(powers=(14,0),extra=replace(Extra(),event=1,event_owner=0,return_player=0,deck=(3,0)))
    result=native_expectation(p,.04)
    assert result.turn is None and result.score is not None and result.nodes>2


def test_siren_hidden_force_does_not_disclose_worker_coordinates():
    p=Position(workers=((0,4),(16,18)),powers=(42,39),extra=replace(Extra(),siren=(13,-1)))
    options,_=native_next_actions(p)
    chants=[x for x in options if x.kind=='sing']
    assert chants and all(x.source==-1 and x.target==12 for x in chants)
    q=complete(p,a('sing',1,0,-1,12))
    assert q.workers[1][0]==17 and q.player==1


def test_adonis_designates_hidden_worker_without_revealing_location():
    p=Position(powers=(47,39))
    prefix=(a('activate',0,-1,-1,-1),a('move',0,0,6,11),a('build',0,0,11,6))
    options,_=native_next_actions(p,prefix)
    assert options and all(x.kind=='adonis' and x.source==-1 and x.target==12 for x in options)
    q=complete(p,*prefix,a('adonis',1,0,-1,12));assert q.adonis==(1,0,0)


@pytest.mark.parametrize('power',[40,43])
def test_native_secret_setup_is_legal_and_budgeted(power):
    import time
    from santorini.native import native_choose_secret
    from santorini.setup import reference_setup
    p=reference_setup(((6,8),(16,18)),(0,power),0)
    start=time.monotonic();q=native_choose_secret(p,.04)
    assert time.monotonic()-start<.5 and not q.extra.view
    q.validate()
    if power==43:assert not q.occupied(q.extra.abyss[1])
    else:assert 0<=q.extra.fate[1]<20 and q.extra.fate[1]%5!=4


def test_lazy_turns_and_terminal_check_with_large_morpheus_stock():
    import time
    from santorini.engine import legal_turns,terminal_position
    p=Position(powers=(25,0),extra=replace(Extra(),materials=(20,0)))
    start=time.monotonic();turn=next(legal_turns(p));checked=terminal_position(p)
    assert turn and checked.winner is None and time.monotonic()-start<.5


def test_chronus_waits_until_dionysus_extra_turns_end():
    h=[0]*25
    for cell in (0,1,2,3,7):h[cell]=3
    p=Position(heights=tuple(h),domes=sum(1<<c for c in (0,1,2,3)),powers=(18,16),player=0,
        extra=replace(Extra(),dion_owner=0))
    q=complete(p,a('move',1,0,16,11),a('dome',1,0,11,7))
    assert q.winner is None and q.extra.event==3
    r=complete(q,a('decline',0,0,-1,12))
    assert r.winner==1 and r.reason=='Cinq tours complètes (Chronus)'


def test_dionysus_controls_hecate_without_revealing_position():
    p=Position(powers=(18,39),extra=replace(Extra(),dion_owner=0))
    options,_=native_next_actions(p)
    assert options and all(x.kind=='hidden_move' and x.source==-1 for x in options)
    prefix=(a('hidden_move',1,0,-1,13),)
    options,_=native_next_actions(p,prefix)
    assert options and all(x.kind=='hidden_build' and x.source==-1 for x in options)
    q=complete(p,*prefix,a('hidden_build',1,0,-1,22))
    assert q.workers[1][0]==17 and q.heights[22]==1 and q.player==1
    result=native_search(p,.08)
    assert result.turn and not result.proven and result.turn.actions[0].kind=='hidden_move'


def test_failed_secret_dome_does_not_trigger_chaos_draw():
    h=[0]*25;h[7]=3
    p=Position(heights=tuple(h),workers=((6,8),(7,18)),powers=(14,39),extra=replace(Extra(),chaos=(4,0),deck=(3,0),discard=(8,0)))
    q=complete(p,a('move',0,0,6,11),a('dome',0,0,11,7))
    assert q.extra.event==0 and q.extra.deck==p.extra.deck and not q.domes


def test_nemesis_blind_swaps_choose_identities_and_swap_both_workers():
    p=Position(workers=((0,4),(20,24)),powers=(41,39))
    prefix=(a('move',0,0,0,1),a('build',0,0,1,0))
    options,_=native_next_actions(p,prefix)
    assert {x.target for x in options if x.kind=='swap_hidden'}=={0,1}
    q=complete(p,*prefix,a('swap_hidden',0,0,1,0),a('swap_hidden',0,1,4,1))
    assert q.workers==((20,24),(1,4))


def test_charon_blind_force_and_failed_attempt():
    p=Position(workers=((6,8),(7,18)),powers=(15,39))
    options,_=native_next_actions(p)
    assert any(x.kind=='probe_charon' and x.source==6 and x.target==5 for x in options)
    q=complete(p,a('probe_charon',0,0,6,5),a('move',0,0,6,11),a('build',0,0,11,6))
    assert q.workers[1][0]==5
    r=complete(p,a('probe_charon',0,0,6,1))
    assert r.reason=='Action annulée par Hecate' and r.player==1


def test_odysseus_blind_force():
    p=Position(workers=((6,8),(7,18)),powers=(53,39))
    prefix=(a('activate',0,-1,-1,-1),)
    options,_=native_next_actions(p,prefix)
    assert any(x.kind=='probe_force' and x.source==-1 for x in options)
    q=complete(p,*prefix,a('probe_force',1,0,-1,0),a('move',0,0,6,11),a('build',0,0,11,6))
    assert q.workers[1][0]==0


def test_medea_blind_removes_under_hidden_worker():
    h=[0]*25;h[17]=1
    p=Position(heights=tuple(h),workers=((6,12),(17,24)),powers=(52,39))
    prefix=(a('activate',0,-1,-1,-1),a('move',0,0,6,11),a('build',0,0,11,6))
    options,_=native_next_actions(p,prefix)
    assert a('probe_remove',0,1,12,17) in options
    q=complete(p,*prefix,a('probe_remove',0,1,12,17))
    assert q.heights[17]==0


def test_theseus_blind_eliminates_selected_adjacent_hidden_worker():
    h=[0]*25;h[12]=2
    p=Position(heights=tuple(h),workers=((6,8),(12,24)),powers=(55,39))
    prefix=(a('activate',0,-1,-1,-1),a('move',0,0,6,11),a('build',0,0,11,6))
    options,_=native_next_actions(p,prefix)
    assert a('probe_kill',1,-1,-1,12) in options
    q=complete(p,*prefix,a('probe_kill',1,-1,-1,12))
    assert q.workers[1][0]==-1

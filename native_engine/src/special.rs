//! Persistent board tokens, hidden objectives, chance and interrupted turns.
use super::*;
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Extra {
    pub wind: i8,
    pub siren: [i8; 2],
    pub whirlpools: [[i8; 2]; 2],
    pub coins: [u32; 2],
    pub coin_count: [u8; 2],
    pub talus: [i8; 2],
    pub materials: [u32; 2],
    pub reserves: [u8; 2],
    pub chaos: [u8; 2],
    pub deck: [u16; 2],
    pub discard: [u16; 2],
    pub theft_owner: i8,
    pub stolen: u8,
    pub abyss: [i8; 2],
    pub fate: [i8; 2],
    pub event: u8, // 1 random draw, 2 Gaea response, 3 Dionysus decision
    pub event_owner: u8,
    pub return_player: u8,
    pub dome_cell: i8,
    pub dion_owner: i8, // extra turn in progress; all wins suppressed
    pub safe: [u32; 2],
    pub view: u8,
    pub queued: u8,
    pub queued_owner: u8,
    pub queued_return: u8,
    pub nemesis_active: u8,
    pub original: [u8; 2], // only within generation, restored before publication
}
impl Default for Extra {
    fn default() -> Self {
        Self {
            wind: -1,
            siren: [-1; 2],
            whirlpools: [[-1; 2]; 2],
            coins: [0; 2],
            coin_count: [0; 2],
            talus: [-1; 2],
            materials: [0; 2],
            reserves: [0; 2],
            chaos: [0; 2],
            deck: [0; 2],
            discard: [0; 2],
            theft_owner: -1,
            stolen: 0,
            abyss: [-1; 2],
            fate: [-1; 2],
            event: 0,
            event_owner: 0,
            return_player: 0,
            dome_cell: -1,
            dion_owner: -1,
            safe: [0; 2],
            view: 0,
            queued: 0,
            queued_owner: 0,
            queued_return: 0,
            nemesis_active: 0,
            original: [255; 2],
        }
    }
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Resume {
    pub origin: Snapshot,
    pub actions: [Action; MAX_ACTIONS],
    pub length: u32,
}
impl Default for Resume {
    fn default() -> Self {
        Self {
            origin: Snapshot {
                heights: [0; 25],
                domes: 0,
                workers: [[-1; 4]; 2],
                counts: [2; 2],
                powers: [0; 2],
                player: 0,
                hero_used: [0; 2],
                athena_lock: 0,
                adonis: [-1; 3],
                winner: -1,
                reason: 0,
                extra: Extra::default(),
            },
            actions: [Action::default(); MAX_ACTIONS],
            length: 0,
        }
    }
}
impl State {
    pub(super) fn snapshot(&self) -> Snapshot {
        Snapshot {
            heights: self.heights,
            domes: self.domes,
            workers: self.workers,
            counts: self.counts,
            powers: self.powers,
            player: self.player,
            hero_used: self.hero_used,
            athena_lock: self.athena_lock,
            adonis: self.adonis,
            winner: self.winner,
            reason: self.reason,
            extra: self.extra,
        }
    }
}
impl Snapshot {
    pub(super) fn state(&self) -> State {
        State {
            heights: self.heights,
            domes: self.domes,
            workers: self.workers,
            counts: self.counts,
            powers: self.powers,
            player: self.player,
            hero_used: self.hero_used,
            athena_lock: self.athena_lock,
            adonis: self.adonis,
            winner: self.winner,
            reason: self.reason,
            extra: self.extra,
            resume: Resume::default(),
        }
    }
}
pub(super) const DIRECTIONS: [i8; 8] = [6, 7, 8, 11, 13, 16, 17, 18];
pub(super) fn vector(direction: i8) -> (i8, i8) {
    (direction % 5 - 2, direction / 5 - 2)
}
pub(super) fn direction(src: i8, dst: i8) -> i8 {
    let dx = (dst % 5 - src % 5).signum();
    let dy = (dst / 5 - src / 5).signum();
    (dy + 2) * 5 + dx + 2
}
pub(super) fn wrapped_direction(src: i8, dst: i8) -> i8 {
    let axis = |delta: i8| {
        if delta.abs() > 1 {
            -delta.signum()
        } else {
            delta.signum()
        }
    };
    (axis(dst / 5 - src / 5) + 2) * 5 + axis(dst % 5 - src % 5) + 2
}
pub(super) fn blocked(s: &State, p: usize, c: i8) -> bool {
    let viewer = if s.extra.dion_owner >= 0 {
        s.extra.dion_owner as usize
    } else {
        p
    };
    let enemy = 1 - viewer;
    let clio = if s.extra.dion_owner >= 0 {
        s.extra.original[enemy] == 33
    } else {
        s.powers[enemy] == 33
    };
    s.extra.talus.contains(&c) || (clio && s.extra.coins[card_owner(s, enemy, 33)] & (1 << c) != 0)
}
pub(super) fn card_owner(s: &State, p: usize, power: u8) -> usize {
    if s.extra.theft_owner == p as i8 && s.extra.stolen == power {
        1 - p
    } else {
        p
    }
}
pub(super) fn prepare(s: &State) -> State {
    if s.extra.original != [255; 2] {
        return *s;
    }
    let mut out = *s;
    if out.winner < 0 {
        out.reason = 0;
    }
    out.extra.original = s.powers;
    let p = s.player as usize;
    if out.extra.theft_owner == p as i8 {
        out.extra.theft_owner = -1;
        out.extra.stolen = 0;
    }
    if s.powers[p] == 17 {
        let enemy: Vec<_> = s.workers[1 - p]
            .iter()
            .copied()
            .filter(|&c| c >= 0)
            .collect();
        if enemy.len() >= 2
            && enemy
                .iter()
                .all(|&a| enemy.iter().all(|&b| a == b || !adjacent(a).contains(&b)))
        {
            out.extra.theft_owner = p as i8;
            out.extra.stolen = s.powers[1 - p];
        }
    }
    if out.extra.theft_owner >= 0 {
        let owner = out.extra.theft_owner as usize;
        out.powers[owner] = if out.extra.stolen >= 46 {
            0
        } else {
            out.extra.stolen
        };
        out.powers[1 - owner] = 0;
    }
    for owner in 0..2 {
        if out.powers[owner] == 14 {
            out.powers[owner] = out.extra.chaos[card_owner(&out, owner, 14)];
        }
    }
    if out.powers[p] == 25 {
        let owner = card_owner(&out, p, 25);
        out.extra.materials[owner] = out.extra.materials[owner].saturating_add(1);
    }
    out
}
pub(super) fn restore(s: &mut State) {
    if s.extra.original != [255; 2] {
        s.powers = s.extra.original;
        s.extra.original = [255; 2];
    }
}
pub(super) fn remove_tokens(s: &mut State, c: i8) {
    for owner in 0..2 {
        s.extra.coins[owner] &= !(1 << c);
        for cell in &mut s.extra.whirlpools[owner] {
            if *cell == c {
                *cell = -1;
            }
        }
    }
}
pub(super) fn hazards(s: &mut State, p: usize, c: i8) {
    if s.extra.dion_owner >= 0 {
        s.winner = -1;
        s.reason = 0;
        return;
    }
    if s.extra.abyss.contains(&c) {
        s.winner = (1 - p) as i8;
        s.reason = 5;
        return;
    }
    let zone = s.extra.fate[card_owner(s, 1 - p, 40)];
    if s.winner == p as i8
        && s.powers[1 - p] == 40
        && zone >= 0
        && (c % 5 - zone % 5).abs() <= 1
        && (c / 5 - zone / 5).abs() <= 1
        && c % 5 >= zone % 5
        && c / 5 >= zone / 5
    {
        s.winner = (1 - p) as i8;
        s.reason = 6;
    }
}
pub(super) fn after_action(before: &State, s: &mut State, a: Action) {
    let p = if a.player >= 0 {
        a.player as usize
    } else {
        before.player as usize
    };
    match a.kind {
        0 | 6 | 7 => {
            if a.kind == 0 {
                for victim in 0..s.counts[1 - p] as usize {
                    let old = before.workers[1 - p][victim];
                    let dst = s.workers[1 - p][victim];
                    if dst >= 0 && dst != old {
                        hazards(s, 1 - p, dst);
                    }
                }
            }
            hazards(s, p, a.target);
            if a.kind == 0 && s.winner < 0 {
                for owner in 0..2 {
                    let pools = s.extra.whirlpools[owner];
                    if pools.iter().all(|&c| c >= 0 && !occupied(before, c)) {
                        if let Some(i) = pools.iter().position(|&c| c == a.target) {
                            let dst = pools[1 - i];
                            s.workers[p][a.worker as usize] = dst;
                            hazards(s, p, dst);
                        }
                    }
                }
                if s.powers[1 - p] == 38 && s.winner < 0 {
                    let (dx, dy) = vector(if s.powers[p] == 45 {
                        wrapped_direction(a.source, a.target)
                    } else {
                        direction(a.source, a.target)
                    });
                    let mut c = s.workers[p][a.worker as usize];
                    loop {
                        let x = c % 5 + dx;
                        let y = c / 5 + dy;
                        if !(0..5).contains(&x) || !(0..5).contains(&y) {
                            break;
                        }
                        let dst = y * 5 + x;
                        if occupied(s, dst)
                            || blocked(s, p, dst)
                            || s.heights[dst as usize] > s.heights[c as usize]
                        {
                            break;
                        }
                        s.workers[p][a.worker as usize] = dst;
                        hazards(s, p, dst);
                        c = dst;
                        if s.winner >= 0 {
                            break;
                        }
                    }
                }
            }
        }
        1 | 2 | 4 => {
            remove_tokens(s, a.target);
            if a.kind == 1 && before.powers[p] == 33 {
                let owner = card_owner(before, p, 33);
                if s.extra.coin_count[owner] < 3 {
                    s.extra.coins[owner] |= 1 << a.target;
                    s.extra.coin_count[owner] += 1;
                }
            }
            if a.kind == 2 && s.winner < 0 {
                for owner in 0..2 {
                    if s.powers[owner] == 35
                        || s.extra.dion_owner >= 0 && s.extra.original[owner] == 35
                    {
                        let stock = card_owner(s, owner, 35);
                        if s.extra.reserves[stock] > 0
                            && s.counts[owner] < 4
                            && adjacent(a.target).iter().any(|&c| {
                                !occupied(s, c)
                                    && !blocked(s, owner, c)
                                    && s.heights[c as usize] == 0
                            })
                        {
                            s.extra.event = 2;
                            s.extra.event_owner = owner as u8;
                            s.extra.dome_cell = a.target;
                            break;
                        }
                    }
                }
            }
            if s.extra.dion_owner >= 0 {
                s.winner = -1;
                s.reason = 0;
            }
        }
        9 => s.extra.wind = a.target,
        10 => {
            let owner = card_owner(s, p, 32);
            if let Some(i) = s.extra.whirlpools[owner].iter().position(|&c| c < 0) {
                s.extra.whirlpools[owner][i] = a.target;
            }
        }
        11 => s.extra.talus[card_owner(s, p, 34)] = a.target,
        12 => {
            let foe = 1 - p;
            let victim = before.workers[foe]
                .iter()
                .position(|&c| c == a.target)
                .unwrap();
            s.workers[p][a.worker as usize] = a.target;
            s.workers[foe][victim] = a.source;
            hazards(s, p, a.target);
            if s.winner < 0 {
                hazards(s, foe, a.source);
            }
        }
        13 => {
            let owner = s.extra.event_owner as usize;
            let bit = 1u16 << (a.target - 1);
            s.extra.chaos[owner] = a.target as u8;
            s.extra.deck[owner] &= !bit;
            s.extra.discard[owner] |= bit;
            s.extra.event = 0;
            s.player = s.extra.return_player;
            if s.extra.queued != 0 {
                s.extra.event = s.extra.queued;
                s.extra.event_owner = s.extra.queued_owner;
                s.extra.return_player = s.extra.queued_return;
                s.player = s.extra.event_owner;
                s.extra.queued = 0;
            }
        }
        14 => {
            if s.extra.event == 2 {
                s.extra.event = 0;
            } else if s.extra.event == 3 {
                s.extra.event = 0;
                s.extra.dion_owner = -1;
                s.player = s.extra.return_player;
            }
        }
        15 => {
            s.extra.event = 0;
            s.extra.dion_owner = p as i8;
        }
        16 => {
            let src = before.workers[p][a.worker as usize];
            if src >= 0 {
                let controller = 1 - p;
                let dir = s.extra.siren[card_owner(s, controller, 42)];
                let (dx, dy) = vector(dir);
                let x = src % 5 + dx;
                let y = src / 5 + dy;
                if (0..5).contains(&x)
                    && (0..5).contains(&y)
                    && !occupied(s, y * 5 + x)
                    && !blocked(s, controller, y * 5 + x)
                {
                    s.workers[p][a.worker as usize] = y * 5 + x;
                    hazards(s, p, y * 5 + x);
                } else {
                    s.reason = 7;
                }
            }
        }
        _ => (),
    }
    if matches!(a.kind, 0 | 6 | 7) && s.winner < 0 && before.powers[p] != 39 {
        for owner in 0..2 {
            if s.powers[owner] == 43 {
                s.extra.safe[owner] |= 1 << a.target;
            }
        }
    }
    if before.extra.event == 2 && a.kind == 7 {
        let stock = card_owner(before, p, 35);
        s.extra.reserves[stock] -= 1;
        s.extra.event = 0;
    }
}
pub(super) fn schedule_draw(s: &mut State, owner: usize) {
    if s.extra.deck[owner] == 0 {
        s.extra.deck[owner] = s.extra.discard[owner];
        s.extra.discard[owner] = 0;
    }
    if s.extra.deck[owner] == 0 {
        return;
    }
    if s.extra.event == 3 {
        s.extra.queued = 3;
        s.extra.queued_owner = s.extra.event_owner;
        s.extra.queued_return = s.extra.return_player;
    }
    // Resolve the public draw before resuming the suspended choice.
    s.extra.return_player = s.player;
    s.player = owner as u8;
    s.extra.event = 1;
    s.extra.event_owner = owner as u8;
}
pub(super) fn prepare_for_turn(s: &State) -> State {
    if s.extra.dion_owner >= 0 {
        let mut out = *s;
        let owner = s.extra.dion_owner as usize;
        out.extra.original = s.powers;
        out.player = (1 - owner) as u8;
        out.powers = [0; 2];
        out.powers[1 - owner] = 18;
        out
    } else {
        prepare(s)
    }
}
pub(super) fn event_turns(s: &State, limit: usize) -> Result<Vec<Turn>, Stop> {
    let p = s.extra.event_owner as usize;
    let mut choices = Vec::new();
    match s.extra.event {
        1 => {
            for card in 1..=10 {
                if s.extra.deck[p] & (1 << (card - 1)) != 0 {
                    choices.push(action(13, p, 0, -1, card));
                }
            }
        }
        2 => {
            choices.push(action(14, p, 0, -1, 12));
            for dst in adjacent(s.extra.dome_cell) {
                if (!occupied(s, dst) || (s.powers[1 - p] == 39 && s.workers[1 - p].contains(&dst)))
                    && !blocked(s, p, dst)
                    && s.heights[dst as usize] == 0
                {
                    choices.push(action(7, p, s.counts[p] as usize, -1, dst));
                }
            }
        }
        3 => {
            choices.push(action(14, p, 0, -1, 12));
            let accept = action(15, p, 0, -1, 12);
            let next = advanced::apply_full(s, accept);
            if !advanced::generate(&next, None, &|| false, 1)?.is_empty() {
                choices.push(accept);
            }
        }
        _ => (),
    }
    let mut turns = Vec::new();
    for a in choices.into_iter().take(limit) {
        let mut after = advanced::apply_full(s, a);
        if s.extra.event == 2 {
            let n = after.resume.length as usize;
            if n >= MAX_ACTIONS {
                return Err(Stop::Deadline);
            }
            after.resume.actions[n] = a;
            after.resume.length += 1;
            after.player = s.extra.return_player;
        }
        turns.push(Turn {
            actions: vec![a],
            after,
        });
    }
    Ok(turns)
}
/// Search from the acting player's observation, without reading the opposing
/// secret. Hypotheses are public-compatible; their scores are estimates.
/// No mate proof or predicted hidden continuation is published.
pub(super) fn hidden_search(s: &State, seconds: f64, cancelled: &dyn Fn() -> bool) -> CAnalysis {
    let started = Instant::now();
    let end = started + Duration::from_secs_f64(seconds);
    let p = s.player as usize;
    let foe = 1 - p;
    let mut public = *s;
    public.extra.view = 1;
    if s.extra.event == 2 && s.powers[foe] == 39 {
        public.resume = Resume::default();
    }
    match s.powers[foe] {
        39 => {
            public.workers[foe] = s.workers[foe].map(|c| if c >= 0 { -2 } else { -1 });
            public.resume.origin.workers[foe] = public.workers[foe];
        }
        40 => public.extra.fate[foe] = -1,
        43 => public.extra.abyss[foe] = -1,
        _ => (),
    }
    public.resume.origin.extra.abyss[foe] = -1;
    public.resume.origin.extra.fate[foe] = -1;
    let mut worlds = Vec::new();
    match s.powers[foe] {
        39 => {
            let available: Vec<i8> = (0..25)
                .filter(|&c| {
                    public.domes & (1 << c) == 0
                        && !public.workers[p].contains(&c)
                        && !blocked(&public, foe, c)
                })
                .collect();
            let living: Vec<_> = (0..s.counts[foe] as usize)
                .filter(|&w| public.workers[foe][w] == -2)
                .collect();
            if living.len() == 1 {
                for &c in &available {
                    let mut world = public;
                    world.workers[foe][living[0]] = c;
                    world.resume.origin.workers[foe] = world.workers[foe];
                    worlds.push(world);
                }
            } else if living.len() >= 2 {
                for &a in &available {
                    for &b in &available {
                        if a == b {
                            continue;
                        }
                        let mut world = public;
                        world.workers[foe][living[0]] = a;
                        world.workers[foe][living[1]] = b;
                        world.resume.origin.workers[foe] = world.workers[foe];
                        worlds.push(world);
                    }
                }
            }
        }

        40 => {
            for c in 0..20 {
                if c % 5 != 4 {
                    let mut world = public;
                    world.extra.fate[foe] = c;
                    world.resume.origin.extra.fate[foe] = c;
                    worlds.push(world);
                }
            }
        }
        43 => {
            for c in 0..25 {
                if public.extra.safe[foe] & (1 << c) == 0 {
                    let mut world = public;
                    world.extra.abyss[foe] = c;
                    world.resume.origin.extra.abyss[foe] = c;
                    worlds.push(world);
                }
            }
        }
        _ => worlds.push(public),
    }
    if worlds.is_empty() {
        worlds.push(public);
    }
    let sampled: Vec<_> = (0..worlds.len().min(16))
        .map(|i| worlds[i * worlds.len() / worlds.len().min(16)])
        .collect();
    let mut search = Search {
        started,
        deadline: end,
        cancelled,
        nodes: 0,
        root: p,
        table: HashMap::new(),
    };
    let mut best: Option<Turn> = None;
    let mut best_value = -2 * MATE;
    let mut completed = 0usize;
    let mut seen = HashSet::new();
    let mut opaque_seen = HashSet::new();
    let _ = advanced::visit(&public, Some(end), cancelled, &mut |candidate| {
        if candidate
            .actions
            .iter()
            .any(|a| matches!(a.kind, 16..=23) || a.kind == 8 && a.source < 0)
        {
            if !opaque_seen.insert(candidate.actions.clone()) {
                return Ok(());
            }
        } else if !seen.insert(candidate.after) {
            return Ok(());
        }
        if best.is_none() {
            best = Some(candidate.clone());
        }
        let mut total = 0i64;
        for world in &sampled {
            search.check()?;
            let mut actual = None;
            for n in 1..=candidate.actions.len() {
                let prefix = &candidate.actions[..n];
                let mut check = |turn: Turn| {
                    if turn.actions == prefix
                        && (n == candidate.actions.len()
                            || turn.after.winner >= 0
                            || turn.after.reason == 7
                            || turn.after.extra.event == 2)
                    {
                        actual = Some(turn);
                        return Err(Stop::Enough);
                    }
                    Ok(())
                };
                match advanced::next_visit(world, Some(end), cancelled, prefix, &mut check) {
                    Ok(()) | Err(Stop::Enough) => (),
                    Err(e) => return Err(e),
                };
                if actual.is_some() {
                    break;
                }
            }
            let Some(mut actual) = actual else {
                total += -MATE as i64;
                continue;
            };
            if s.extra.event == 2 && s.powers[foe] == 39 {
                actual.after.resume = Resume::default();
                actual.after.player = s.extra.event_owner;
                actual.after.reason = 0;
            }
            // One full opponent response, then evaluate. All work stays in Rust.
            let (value, _) = search.minimax(&actual.after, 1, -2 * MATE, 2 * MATE, 1)?;
            total += value as i64;
        }
        let value = (total / sampled.len() as i64) as i32;
        completed += 1;
        if value > best_value {
            best_value = value;
            best = Some(candidate);
        }
        Ok(())
    });
    if let Some(turn) = &mut best {
        turn.after.resume = Resume::default();
        for c in turn.after.workers.iter_mut().flatten() {
            if *c == -2 {
                *c = -1;
            }
        }
    }
    let line = best.into_iter().collect::<Vec<_>>();
    let mut out = search.output(
        &line,
        if completed > 0 {
            Some(best_value.clamp(-MATE / 2, MATE / 2))
        } else {
            None
        },
        0,
        false,
        7,
        s,
    );
    out.proven = 0;
    out
}

pub(super) fn gaea_response_ok(s: &State, a: Action) -> bool {
    let p = s.extra.event_owner as usize;
    if a == action(14, p, 0, -1, 12) {
        return true;
    }
    a.kind == 7
        && a.player == p as i8
        && a.worker == s.counts[p] as i8
        && a.source == -1
        && (0..25).contains(&a.target)
        && adjacent(s.extra.dome_cell).contains(&a.target)
        && s.heights[a.target as usize] == 0
        && !blocked(s, p, a.target)
        && (!occupied(s, a.target)
            || (s.powers[1 - p] == 39 && s.workers[1 - p].contains(&a.target)))
}

pub(super) fn valid_extra(e: &Extra) -> bool {
    (e.wind == -1 || DIRECTIONS.contains(&e.wind))
        && e.siren.iter().all(|&d| d == -1 || DIRECTIONS.contains(&d))
        && e.whirlpools
            .iter()
            .flatten()
            .chain(e.talus.iter())
            .chain(e.abyss.iter())
            .all(|&c| (-1..25).contains(&c))
        && e.fate
            .iter()
            .all(|&c| c == -1 || (0..20).contains(&c) && c % 5 != 4)
        && e.coins
            .iter()
            .chain(e.safe.iter())
            .all(|&mask| mask < 1 << 25)
        && e.coin_count.iter().all(|&n| n <= 3)
        && e.reserves.iter().all(|&n| n <= 2)
        && e.chaos.iter().all(|&n| n <= 10)
        && e.deck
            .iter()
            .chain(e.discard.iter())
            .all(|&mask| mask <= 1023)
        && (-1..=1).contains(&e.theft_owner)
        && (-1..=1).contains(&e.dion_owner)
        && e.stolen <= 55
        && e.event <= 3
        && e.event_owner <= 1
        && e.return_player <= 1
        && (-1..25).contains(&e.dome_cell)
        && e.view <= 1
        && e.nemesis_active <= 1
        && (e.queued == 0 || e.queued == 3)
        && e.queued_owner <= 1
        && e.queued_return <= 1
        && (e.original == [255; 2] || e.original.iter().all(|&p| p <= 55))
}

pub(super) fn terminalize(s: &mut State) {
    if s.winner >= 0 || s.extra.dion_owner >= 0 || s.extra.event != 0 {
        return;
    }
    let mut powers = s.powers;
    if s.extra.theft_owner >= 0 {
        let owner = s.extra.theft_owner as usize;
        powers[owner] = s.extra.stolen;
        powers[1 - owner] = 0;
    }
    if (0..25)
        .filter(|&i| s.heights[i] == 3 && s.domes & (1 << i) != 0)
        .count()
        >= 5
    {
        if let Some(owner) = powers.iter().position(|&p| p == 16) {
            s.winner = owner as i8;
            s.reason = 4;
        }
    }
}

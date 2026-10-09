//! Native rules and search for the 55 supplied gods and heroes.
//! No external crates. Complete turns, interrupt ownership and chance nodes.
use std::collections::{HashMap, HashSet, VecDeque};
use std::rc::Rc;
use std::time::{Duration, Instant};
mod advanced;
mod special;

const MATE: i32 = 100_000;
const MAX_ACTIONS: usize = 128;

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct Snapshot {
    pub heights: [u8; 25],
    pub domes: u32,
    pub workers: [[i8; 4]; 2],
    pub counts: [u8; 2],
    pub powers: [u8; 2],
    pub player: u8,
    pub hero_used: [u8; 2],
    pub athena_lock: u8,
    pub adonis: [i8; 3],
    pub winner: i8,
    pub reason: u8,
    pub extra: special::Extra,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct State {
    pub heights: [u8; 25],
    pub domes: u32,
    pub workers: [[i8; 4]; 2],
    pub counts: [u8; 2],
    pub powers: [u8; 2],
    pub player: u8,
    pub hero_used: [u8; 2],
    pub athena_lock: u8,
    pub adonis: [i8; 3],
    pub winner: i8,
    pub reason: u8,
    pub extra: special::Extra,
    pub resume: special::Resume,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Default)]
pub struct Action {
    pub kind: u8, // 0 = move, 1 = block, 2 = dome
    pub player: i8,
    pub worker: i8,
    pub source: i8,
    pub target: i8,
}

#[derive(Clone, Debug)]
struct Turn {
    actions: Vec<Action>,
    after: State,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CTurn {
    pub actions: [Action; MAX_ACTIONS],
    pub length: u32,
    pub after: State,
}

impl From<&Turn> for CTurn {
    fn from(turn: &Turn) -> Self {
        assert!(turn.actions.len() <= MAX_ACTIONS);
        let mut actions = [Action::default(); MAX_ACTIONS];
        actions[..turn.actions.len()].copy_from_slice(&turn.actions);
        Self {
            actions,
            length: turn.actions.len() as u32,
            after: turn.after,
        }
    }
}

#[repr(C)]
pub struct CAnalysis {
    pub turns: [CTurn; 3],
    pub length: u32,
    pub score: i32,
    pub has_score: u8,
    pub depth: u32,
    pub nodes: u64,
    pub elapsed: f64,
    pub proven: u8,
    pub complete: u8,
    pub status: u8, // 0 budget, 1 proof, 2 terminal, 3 no fallback, 4 searching, 5 depth limit
}

fn adjacent(cell: i8) -> Vec<i8> {
    let (x, y) = (cell % 5, cell / 5);
    let mut result = Vec::with_capacity(8);
    for dy in -1..=1 {
        for dx in -1..=1 {
            let (nx, ny) = (x + dx, y + dy);
            if (dx != 0 || dy != 0) && (0..5).contains(&nx) && (0..5).contains(&ny) {
                result.push(ny * 5 + nx);
            }
        }
    }
    result
}

fn occupied(s: &State, c: i8) -> bool {
    s.domes & (1 << c) != 0 || s.workers.iter().any(|ws| ws.contains(&c))
}

fn can_step(s: &State, w: usize, dst: i8, no_up: bool, flat: bool) -> bool {
    let p = s.player as usize;
    let foe = 1 - p;
    let src = s.workers[p][w];
    if s.domes & (1 << dst) != 0 || s.workers[p].contains(&dst) {
        return false;
    }
    let delta = s.heights[dst as usize] as i8 - s.heights[src as usize] as i8;
    if delta > 1 || (flat && delta != 0) || (delta > 0 && (s.athena_lock != 0 || no_up)) {
        return false;
    }
    if s.workers[foe].contains(&dst) {
        match s.powers[p] {
            1 => return true,
            8 => {
                let x = 2 * (dst % 5) - src % 5;
                let y = 2 * (dst / 5) - src / 5;
                return (0..5).contains(&x) && (0..5).contains(&y) && !occupied(s, y * 5 + x);
            }
            _ => return false,
        }
    }
    true
}

fn action(kind: u8, p: usize, w: usize, src: i8, dst: i8) -> Action {
    Action {
        kind,
        player: p as i8,
        worker: w as i8,
        source: src,
        target: dst,
    }
}

fn apply(s: &State, a: Action) -> State {
    let mut result = *s;
    let (p, w) = (a.player as usize, a.worker as usize);
    if a.kind == 0 {
        if let Some(victim) = s.workers[1 - p].iter().position(|&v| v == a.target) {
            result.workers[1 - p][victim] = if s.powers[p] == 1 {
                a.source
            } else {
                (2 * (a.target / 5) - a.source / 5) * 5 + 2 * (a.target % 5) - a.source % 5
            };
        }
        result.workers[p][w] = a.target;
        let h0 = s.heights[a.source as usize];
        let h1 = s.heights[a.target as usize];
        if h1 == 3 && h0 < 3 {
            result.winner = p as i8;
            result.reason = 1;
        } else if s.powers[p] == 9 && h0 as i8 - h1 as i8 >= 2 {
            result.winner = p as i8;
            result.reason = 2;
        }
    } else if a.kind == 1 {
        result.heights[a.target as usize] += 1;
    } else {
        result.domes |= 1 << a.target;
    }
    result
}

#[derive(Debug)]
enum Stop {
    Deadline,
    Enough,
}
type GResult = Result<(), Stop>;

struct Generator<'a> {
    deadline: Option<Instant>,
    cancelled: &'a dyn Fn() -> bool,
    calls: usize,
    limit: usize,
    turns: Vec<Turn>,
    produced: usize,
    visitor: Option<&'a mut dyn FnMut(Turn) -> GResult>,
}

impl Generator<'_> {
    fn check(&mut self) -> GResult {
        self.calls += 1;
        if self.calls % 64 == 0 || self.calls == 1 {
            if self.deadline.is_some_and(|end| Instant::now() >= end) || (self.cancelled)() {
                return Err(Stop::Deadline);
            }
        }
        Ok(())
    }
    fn emit(&mut self, turn: Turn) -> GResult {
        self.produced += 1;
        if let Some(visitor) = &mut self.visitor {
            visitor(turn)?;
        } else {
            self.turns.push(turn);
        }
        if self.produced >= self.limit {
            return Err(Stop::Enough);
        }
        Ok(())
    }
    fn finish(&mut self, s: &State, actions: &[Action], up: bool) -> GResult {
        self.check()?;
        let mut after = *s;
        after.player = 1 - s.player;
        after.athena_lock = u8::from(s.powers[s.player as usize] == 3 && up);
        self.emit(Turn {
            actions: actions.to_vec(),
            after,
        })
    }
    fn builds(&mut self, s: &State, actions: &[Action], worker: usize, up: bool) -> GResult {
        if s.winner >= 0 {
            return self.finish(s, actions, up);
        }
        let p = s.player as usize;
        let power = s.powers[p];
        let src = s.workers[p][worker];
        for dst in adjacent(src) {
            self.check()?;
            if occupied(s, dst) {
                continue;
            }
            let first_kind = if s.heights[dst as usize] == 3 { 2 } else { 1 };
            let kinds: &[u8] = if power == 4 && first_kind == 1 {
                &[1, 2]
            } else {
                std::slice::from_ref(&first_kind)
            };
            for &kind in kinds {
                let a = action(kind, p, worker, src, dst);
                let s1 = apply(s, a);
                let mut path = actions.to_vec();
                path.push(a);
                self.finish(&s1, &path, up)?;
                if power == 5 || power == 6 {
                    for dst2 in adjacent(src) {
                        if occupied(&s1, dst2) || (power == 5 && dst2 == dst) {
                            continue;
                        }
                        let kind2 = if s1.heights[dst2 as usize] == 3 { 2 } else { 1 };
                        if power == 6 && (dst2 != dst || kind != 1 || kind2 != 1) {
                            continue;
                        }
                        let a2 = action(kind2, p, worker, src, dst2);
                        let s2 = apply(&s1, a2);
                        path.push(a2);
                        self.finish(&s2, &path, up)?;
                        path.pop();
                    }
                }
            }
        }
        Ok(())
    }
    fn moves(&mut self, s: &State, actions: &[Action], w: usize, no_up: bool) -> GResult {
        let p = s.player as usize;
        let src = s.workers[p][w];
        for dst in adjacent(src) {
            self.check()?;
            if !can_step(s, w, dst, no_up, false) {
                continue;
            }
            let a = action(0, p, w, src, dst);
            let s1 = apply(s, a);
            let mut path = actions.to_vec();
            path.push(a);
            let up = s.heights[dst as usize] > s.heights[src as usize];
            self.builds(&s1, &path, w, up)?;
            if s.powers[p] == 2 && s1.winner < 0 {
                for dst2 in adjacent(dst) {
                    if dst2 == src || !can_step(&s1, w, dst2, no_up, false) {
                        continue;
                    }
                    let a2 = action(0, p, w, dst, dst2);
                    let s2 = apply(&s1, a2);
                    path.push(a2);
                    self.builds(
                        &s2,
                        &path,
                        w,
                        up || s1.heights[dst2 as usize] > s1.heights[dst as usize],
                    )?;
                    path.pop();
                }
            }
        }
        Ok(())
    }
    fn generate(&mut self, s: &State) -> GResult {
        self.check()?;
        if s.winner >= 0 {
            return Ok(());
        }
        let p = s.player as usize;
        if s.powers[p] == 7 {
            let mut seen = HashSet::new();
            seen.insert(s.workers[p]);
            let mut queue = VecDeque::from([(*s, Vec::<Action>::new())]);
            while let Some((state, path)) = queue.pop_front() {
                self.check()?;
                for w in 0..state.counts[p] as usize {
                    let src = state.workers[p][w];
                    if src >= 0 {
                        self.builds(&state, &path, w, false)?;
                    }
                }
                for w in 0..state.counts[p] as usize {
                    let src = state.workers[p][w];
                    if src < 0 {
                        continue;
                    }
                    for dst in adjacent(src) {
                        if !can_step(&state, w, dst, false, true) {
                            continue;
                        }
                        let a = action(0, p, w, src, dst);
                        let next = apply(&state, a);
                        if seen.insert(next.workers[p]) {
                            let mut line = path.clone();
                            line.push(a);
                            queue.push_back((next, line));
                        }
                    }
                }
            }
        }
        for w in 0..s.counts[p] as usize {
            let src = s.workers[p][w];
            if src < 0 {
                continue;
            }
            self.moves(s, &[], w, false)?;
            if s.powers[p] == 10 {
                for dst in adjacent(src) {
                    if occupied(s, dst) {
                        continue;
                    }
                    let a = action(
                        if s.heights[dst as usize] == 3 { 2 } else { 1 },
                        p,
                        w,
                        src,
                        dst,
                    );
                    let s1 = apply(s, a);
                    self.moves(&s1, &[a], w, true)?;
                }
            }
        }
        Ok(())
    }
}

fn generate(
    s: &State,
    deadline: Option<Instant>,
    cancelled: &dyn Fn() -> bool,
    limit: usize,
) -> Result<Vec<Turn>, Stop> {
    if s.powers.iter().any(|&p| p > 10)
        || s.adonis != [-1; 3]
        || s.counts != [2, 2]
        || s.extra != special::Extra::default()
        || s.resume.length != 0
    {
        return advanced::generate(s, deadline, cancelled, limit);
    }
    let mut gen = Generator {
        deadline,
        cancelled,
        calls: 0,
        limit,
        turns: Vec::new(),
        produced: 0,
        visitor: None,
    };
    match gen.generate(s) {
        Ok(()) | Err(Stop::Enough) => Ok(gen.turns),
        Err(e) => Err(e),
    }
}

fn evaluate(s: &State, player: usize) -> i32 {
    if s.winner >= 0 {
        return if s.winner as usize == player {
            MATE
        } else {
            -MATE
        };
    }
    let strength = |p: usize| {
        let mut value = 0;
        for &src in &s.workers[p][..s.counts[p] as usize] {
            if src < 0 {
                continue;
            }
            let h = s.heights[src as usize];
            value += 80 + [0, 16, 52, 65][h as usize];
            value += 2 * (4 - (src % 5 - 2).abs() as i32 - (src / 5 - 2).abs() as i32);
            for dst in adjacent(src) {
                if occupied(s, dst) {
                    continue;
                }
                let delta = s.heights[dst as usize] as i32 - h as i32;
                if delta > 1
                    || (s.powers[1 - p] == 37 && delta < 0)
                    || (p == s.player as usize && s.athena_lock != 0 && delta > 0)
                {
                    continue;
                }
                value += 2 + 2 * delta.max(0);
                if s.heights[dst as usize] == 3
                    && h == 2
                    && !(s.powers[1 - p] == 20 && advanced::border(dst))
                {
                    value += 100;
                }
                if s.powers[p] == 9 && delta <= -2 {
                    value += 100;
                }
            }
        }
        if s.powers[p] >= 46 && s.hero_used[p] == 0 {
            value += 12;
        }
        if s.powers[p] == 16 {
            value += 12
                * (0..25)
                    .filter(|&c| s.heights[c] == 3 && s.domes & (1 << c) != 0)
                    .count() as i32;
        }
        value
    };
    strength(player) - strength(1 - player)
}

const TABLE_CAP: usize = 50_000;

#[derive(Clone)]
struct Entry {
    depth: u32,
    value: i32,
    flag: u8,
    line: Rc<Vec<Turn>>,
}

struct Search<'a> {
    started: Instant,
    deadline: Instant,
    cancelled: &'a dyn Fn() -> bool,
    nodes: u64,
    root: usize,
    table: HashMap<(State, u32), Entry>,
}

fn ordered_visit(
    s: &State,
    preferred: Option<&Turn>,
    deadline: Instant,
    cancelled: &dyn Fn() -> bool,
    visitor: &mut dyn FnMut(Turn) -> GResult,
) -> GResult {
    if s.powers.iter().all(|&p| p <= 10)
        && s.adonis == [-1; 3]
        && s.counts == [2, 2]
        && s.extra == special::Extra::default()
        && s.resume.length == 0
    {
        let mut seen = HashSet::new();
        let mut turns: Vec<_> = generate(s, Some(deadline), cancelled, usize::MAX)?
            .into_iter()
            .filter(|t| seen.insert(t.after))
            .collect();
        turns.sort_by_cached_key(|t| {
            (
                std::cmp::Reverse(preferred.is_some_and(|p| p.after == t.after)),
                std::cmp::Reverse(evaluate(&t.after, s.player as usize)),
            )
        });
        for turn in turns {
            visitor(turn)?;
        }
        return Ok(());
    }
    let mut seen = HashSet::new();
    if let Some(turn) = preferred {
        seen.insert(turn.after);
        visitor(turn.clone())?;
    }
    let mut batch: Vec<Turn> = Vec::with_capacity(24);
    let mut flush = |batch: &mut Vec<Turn>| -> GResult {
        batch.sort_by_cached_key(|t| std::cmp::Reverse(evaluate(&t.after, s.player as usize)));
        for turn in batch.drain(..) {
            visitor(turn)?;
        }
        Ok(())
    };
    advanced::visit(s, Some(deadline), cancelled, &mut |turn| {
        if seen.insert(turn.after) {
            batch.push(turn);
        }
        if batch.len() >= 24 {
            flush(&mut batch)?;
        }
        Ok(())
    })?;
    flush(&mut batch)
}

impl Search<'_> {
    fn check(&self) -> Result<(), Stop> {
        if Instant::now() >= self.deadline || (self.nodes % 512 == 0 && (self.cancelled)()) {
            Err(Stop::Deadline)
        } else {
            Ok(())
        }
    }
    /// Insert or refresh a transposition entry. A key already present is
    /// always updated, even once the table reached `TABLE_CAP`, since this
    /// never grows its size. A brand-new key past the cap only displaces an
    /// arbitrary existing entry that is no deeper than the incoming one:
    /// this keeps the bounded memory cost while favouring the deepest
    /// knowledge over shallow entries, instead of refusing every insertion
    /// once the table first fills up.
    fn store(&mut self, key: (State, u32), entry: Entry) {
        if self.table.len() < TABLE_CAP || self.table.contains_key(&key) {
            self.table.insert(key, entry);
            return;
        }
        let victim = self
            .table
            .iter()
            .next()
            .map(|(&k, v)| (k, v.depth))
            .filter(|&(_, depth)| depth <= entry.depth)
            .map(|(k, _)| k);
        if let Some(victim) = victim {
            self.table.remove(&victim);
            self.table.insert(key, entry);
        }
    }
    fn minimax(
        &mut self,
        s: &State,
        depth: u32,
        mut alpha: i32,
        mut beta: i32,
        ply: u32,
    ) -> Result<(i32, Vec<Turn>), Stop> {
        self.check()?;
        self.nodes += 1;
        if s.winner >= 0 {
            return Ok((
                if s.winner as usize == self.root {
                    MATE - ply as i32
                } else {
                    -MATE + ply as i32
                },
                vec![],
            ));
        }
        if s.extra.event == 1 {
            let draws = special::event_turns(s, usize::MAX)?;
            let mut total = 0i64;
            let mut all_win = true;
            let mut all_loss = true;
            for turn in &draws {
                self.check()?;
                let (v, _) = self.minimax(&turn.after, depth, -2 * MATE, 2 * MATE, ply)?;
                total += v as i64;
                all_win &= v >= MATE - 1000;
                all_loss &= v <= -MATE + 1000;
            }
            if draws.is_empty() {
                return Ok((evaluate(s, self.root), vec![]));
            }
            let mut value = (total / draws.len() as i64) as i32;
            if !all_win && !all_loss {
                value = value.clamp(-MATE / 2, MATE / 2);
            }
            return Ok((value, vec![]));
        }
        let original = (alpha, beta);
        let key = (*s, ply);
        let cached = self.table.get(&key).cloned();
        if let Some(entry) = &cached {
            if entry.depth >= depth {
                if entry.flag == 0 {
                    return Ok((entry.value, (*entry.line).clone()));
                }
                if entry.flag == 1 {
                    alpha = alpha.max(entry.value);
                } else {
                    beta = beta.min(entry.value);
                }
                if alpha >= beta {
                    return Ok((entry.value, (*entry.line).clone()));
                }
            }
        }
        if depth == 0 {
            let legal = generate(s, Some(self.deadline), self.cancelled, 1)?;
            let v = if legal.is_empty() {
                if s.player as usize == self.root {
                    -MATE + ply as i32
                } else {
                    MATE - ply as i32
                }
            } else {
                evaluate(s, self.root)
            };
            return Ok((v, vec![]));
        }
        let preferred = cached.as_ref().and_then(|e| e.line.first());
        let maximizing = s.player as usize == self.root;
        let mut value = if maximizing { -2 * MATE } else { 2 * MATE };
        let mut line = vec![];
        let mut count = 0;
        let deadline = self.deadline;
        let cancelled = self.cancelled;
        let scanned = ordered_visit(s, preferred, deadline, cancelled, &mut |turn| {
            count += 1;
            let (v, mut tail) = self.minimax(&turn.after, depth - 1, alpha, beta, ply + 1)?;
            if (maximizing && v > value) || (!maximizing && v < value) {
                value = v;
                line = vec![turn];
                line.append(&mut tail);
            }
            if maximizing {
                alpha = alpha.max(value);
            } else {
                beta = beta.min(value);
            }
            if alpha >= beta {
                return Err(Stop::Enough);
            }
            Ok(())
        });
        match scanned {
            Ok(()) | Err(Stop::Enough) => (),
            Err(e) => return Err(e),
        }
        if count == 0 {
            return Ok((
                if maximizing {
                    -MATE + ply as i32
                } else {
                    MATE - ply as i32
                },
                vec![],
            ));
        }
        let flag = if value <= original.0 {
            2
        } else if value >= original.1 {
            1
        } else {
            0
        };
        self.store(
            key,
            Entry {
                depth,
                value,
                flag,
                line: Rc::new(line.clone()),
            },
        );
        Ok((value, line))
    }
    fn output(
        &self,
        line: &[Turn],
        score: Option<i32>,
        depth: u32,
        complete: bool,
        status: u8,
        s: &State,
    ) -> CAnalysis {
        let empty = CTurn {
            actions: [Action::default(); MAX_ACTIONS],
            length: 0,
            after: *s,
        };
        let mut turns = [empty; 3];
        for (i, turn) in line.iter().take(3).enumerate() {
            turns[i] = CTurn::from(turn);
        }
        CAnalysis {
            turns,
            length: line.len().min(3) as u32,
            score: score.unwrap_or(0),
            has_score: u8::from(score.is_some()),
            depth,
            nodes: self.nodes,
            elapsed: self.started.elapsed().as_secs_f64(),
            proven: u8::from(
                score.is_some_and(|v| v >= MATE - 1000 || (complete && v <= -MATE + 1000)),
            ),
            complete: u8::from(complete),
            status,
        }
    }
}

fn supported(s: &State) -> bool {
    special::valid_extra(&s.extra)
        && s.resume.length <= MAX_ACTIONS as u32
        && (s.resume.length == 0 || supported(&s.resume.origin.state()))
        && advanced::supported(s)
        && s.counts.iter().all(|&count| (2..=4).contains(&count))
        && (s.adonis == [-1; 3]
            || ((0..=1).contains(&s.adonis[0])
                && s.adonis[2] == 1 - s.adonis[0]
                && s.adonis[1] >= 0
                && s.adonis[1] < s.counts[s.adonis[0] as usize] as i8))
        && s.player <= 1
        && s.winner >= -1
        && s.winner <= 1
        && s.heights.iter().all(|&h| h <= 3)
        && s.domes < 1 << 25
        && s.workers
            .iter()
            .all(|ws| ws.iter().all(|&c| (-1..25).contains(&c)))
}

#[no_mangle]
pub extern "C" fn santoni_rules_version() -> u32 {
    3
}

#[no_mangle]
pub extern "C" fn santoni_state_size() -> usize {
    std::mem::size_of::<State>()
}

/// Caller supplies a valid State and owns all output memory for this call.
#[no_mangle]
pub unsafe extern "C" fn santoni_search(
    state: *const State,
    seconds: f64,
    output: *mut CAnalysis,
    publish: Option<extern "C" fn(*const CAnalysis)>,
    cancelled: Option<extern "C" fn() -> u8>,
) -> i32 {
    if state.is_null()
        || output.is_null()
        || !seconds.is_finite()
        || !(0.001..=3600.0).contains(&seconds)
    {
        return -1;
    }
    let mut s = *state;
    if !supported(&s) {
        return -2;
    }
    special::terminalize(&mut s);
    let started = Instant::now();
    let cancel = || cancelled.is_some_and(|callback| callback() != 0);
    if s.extra.view == 0
        && matches!(s.powers[1 - s.player as usize], 39 | 40 | 43)
        && s.extra.event != 1
    {
        *output = special::hidden_search(&s, seconds, &cancel);
        if let Some(callback) = publish {
            callback(output);
        }
        return 0;
    }
    let mut search = Search {
        started,
        deadline: started + Duration::from_secs_f64(seconds),
        cancelled: &cancel,
        nodes: 0,
        root: s.player as usize,
        table: HashMap::new(),
    };
    if s.extra.event == 1 {
        *output = search.output(&[], None, 0, true, 6, &s);
        return 0;
    }
    let fallback = match generate(&s, Some(search.deadline), &cancel, 1) {
        Ok(mut moves) => {
            if moves.is_empty() {
                None
            } else {
                Some(moves.remove(0))
            }
        }
        Err(_) => {
            *output = search.output(&[], None, 0, false, 3, &s);
            return 0;
        }
    };
    let Some(first) = fallback else {
        let value = if s.winner >= 0 {
            evaluate(&s, search.root)
        } else {
            -MATE
        };
        *output = search.output(&[], Some(value), 0, true, 2, &s);
        return 0;
    };
    let mut best_line = vec![first];
    let mut best_score = None;
    let mut best_depth = 0;
    let mut complete = false;
    if let Some(callback) = publish {
        callback(&search.output(&best_line, best_score, 0, false, 0, &s));
    }
    'deepening: for depth in 1..=64 {
        if let Some(callback) = publish {
            callback(&search.output(&best_line, best_score, best_depth, complete, 4, &s));
        }
        let preferred = best_line.first().cloned();
        let mut value = -2 * MATE;
        let mut line = vec![];
        let deadline = search.deadline;
        let root_cancelled = search.cancelled;
        let scanned = ordered_visit(
            &s,
            preferred.as_ref(),
            deadline,
            root_cancelled,
            &mut |turn| {
                let (v, mut tail) = search.minimax(&turn.after, depth - 1, value, 2 * MATE, 1)?;
                if v > value {
                    value = v;
                    line = vec![turn];
                    line.append(&mut tail);
                }
                if depth == 1 {
                    best_line = line.clone();
                    best_score = Some(value);
                    best_depth = 1;
                }
                if v == MATE - 1 {
                    return Err(Stop::Enough);
                }
                Ok(())
            },
        );
        match scanned {
            Ok(()) | Err(Stop::Enough) => (),
            Err(_) => break 'deepening,
        }
        best_line = line;
        best_score = Some(value);
        best_depth = depth;
        complete = true;
        let current = search.output(
            &best_line,
            best_score,
            depth,
            true,
            u8::from(value.abs() >= MATE - 1000),
            &s,
        );
        if let Some(callback) = publish {
            callback(&current);
        }
        if value.abs() >= MATE - 1000 {
            break;
        }
    }
    *output = search.output(
        &best_line,
        best_score,
        best_depth,
        complete,
        if best_score.is_some_and(|v| v >= MATE - 1000 || complete && v <= -MATE + 1000) {
            1
        } else if best_depth == 64 {
            5
        } else {
            0
        },
        &s,
    );
    0
}

/// Differential-test interface. Returns required capacity, or usize::MAX on
/// invalid input. Only writes as many CTurn values as the caller allocated.
#[no_mangle]
pub unsafe extern "C" fn santoni_legal(
    state: *const State,
    output: *mut CTurn,
    capacity: usize,
) -> usize {
    if state.is_null() || !supported(&*state) || (output.is_null() && capacity > 0) {
        return usize::MAX;
    }
    let turns = generate(&*state, None, &|| false, usize::MAX).unwrap();
    for (i, turn) in turns.iter().take(capacity).enumerate() {
        *output.add(i) = CTurn::from(turn);
    }
    turns.len()
}

/// Next actions and exact completion for an entered prefix. The generator
/// prunes represented branches; large hero subsets are never materialised.
#[no_mangle]
pub unsafe extern "C" fn santoni_next_actions(
    state: *const State,
    prefix: *const Action,
    length: usize,
    output: *mut Action,
    capacity: usize,
    complete: *mut CTurn,
    has_complete: *mut u8,
    cancelled: Option<extern "C" fn() -> u8>,
) -> usize {
    if state.is_null()
        || !supported(&*state)
        || length > MAX_ACTIONS
        || (prefix.is_null() && length > 0)
        || (output.is_null() && capacity > 0)
        || complete.is_null()
        || has_complete.is_null()
    {
        return usize::MAX;
    }
    let path = if length == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(prefix, length)
    };
    let mut choices = HashSet::new();
    *has_complete = 0;
    let cancel = || cancelled.is_some_and(|callback| callback() != 0);
    let result = advanced::next_visit(&*state, None, &cancel, path, &mut |turn| {
        if let Some(next) = turn.actions.get(length) {
            choices.insert(*next);
        } else {
            *complete = CTurn::from(&turn);
            *has_complete = 1;
        }
        Ok(())
    });
    if result.is_err() {
        return usize::MAX;
    }
    let mut actions: Vec<_> = choices.into_iter().collect();
    actions.sort_by_key(|a| (a.kind, a.player, a.worker, a.target));
    for (i, a) in actions.iter().take(capacity).enumerate() {
        *output.add(i) = *a;
    }
    actions.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn initial() -> State {
        State {
            heights: [0; 25],
            domes: 0,
            workers: [[6, 8, -1, -1], [16, 18, -1, -1]],
            counts: [2, 2],
            powers: [0, 0],
            player: 0,
            hero_used: [0; 2],
            athena_lock: 0,
            adonis: [-1; 3],
            winner: -1,
            reason: 0,
            extra: special::Extra::default(),
            resume: special::Resume::default(),
        }
    }
    #[test]
    fn standard_turns() {
        let turns = generate(&initial(), None, &|| false, usize::MAX).unwrap();
        assert_eq!(turns.len(), 80);
        assert!(turns
            .iter()
            .all(|t| t.actions.len() == 2 && t.after.player == 1));
    }
    #[test]
    fn immediate_win() {
        let mut state = initial();
        state.heights[6] = 2;
        state.heights[7] = 3;
        let turns = generate(&state, None, &|| false, usize::MAX).unwrap();
        let win = turns.iter().find(|t| t.after.winner == 0).unwrap();
        assert_eq!(win.actions.len(), 1);
    }
    #[test]
    fn hermes_stays_and_moves_both() {
        let mut state = initial();
        state.powers[0] = 7;
        let turns = generate(&state, None, &|| false, usize::MAX).unwrap();
        assert!(turns.iter().any(|t| t.actions.len() == 1));
        assert!(turns
            .iter()
            .any(|t| t.actions.iter().any(|a| a.kind == 0 && a.worker == 0)
                && t.actions.iter().any(|a| a.kind == 0 && a.worker == 1)));
    }
    fn dummy_search() -> Search<'static> {
        Search {
            started: Instant::now(),
            deadline: Instant::now() + Duration::from_secs(10),
            cancelled: &|| false,
            nodes: 0,
            root: 0,
            table: HashMap::new(),
        }
    }
    fn dummy_key(tag: u32) -> (State, u32) {
        let mut s = initial();
        s.domes = tag;
        (s, 0)
    }
    fn dummy_entry(depth: u32) -> Entry {
        Entry {
            depth,
            value: 0,
            flag: 0,
            line: Rc::new(vec![]),
        }
    }
    #[test]
    fn store_grows_table_below_capacity() {
        let mut search = dummy_search();
        search.store(dummy_key(0), dummy_entry(3));
        assert_eq!(search.table.len(), 1);
        assert_eq!(search.table[&dummy_key(0)].depth, 3);
    }
    #[test]
    fn store_updates_existing_key_even_when_table_full() {
        let mut search = dummy_search();
        for i in 0..TABLE_CAP as u32 {
            search.store(dummy_key(i), dummy_entry(5));
        }
        assert_eq!(search.table.len(), TABLE_CAP);
        search.store(dummy_key(0), dummy_entry(9));
        assert_eq!(search.table.len(), TABLE_CAP);
        assert_eq!(search.table[&dummy_key(0)].depth, 9);
    }
    #[test]
    fn store_replaces_shallow_victim_with_deeper_new_entry() {
        let mut search = dummy_search();
        for i in 0..TABLE_CAP as u32 {
            search.store(dummy_key(i), dummy_entry(1));
        }
        assert_eq!(search.table.len(), TABLE_CAP);
        let new_key = dummy_key(TABLE_CAP as u32);
        search.store(new_key, dummy_entry(5));
        assert_eq!(search.table.len(), TABLE_CAP);
        assert_eq!(search.table[&new_key].depth, 5);
    }
    #[test]
    fn store_keeps_deep_entries_over_a_shallow_newcomer() {
        let mut search = dummy_search();
        for i in 0..TABLE_CAP as u32 {
            search.store(dummy_key(i), dummy_entry(9));
        }
        assert_eq!(search.table.len(), TABLE_CAP);
        let new_key = dummy_key(TABLE_CAP as u32);
        search.store(new_key, dummy_entry(1));
        assert_eq!(search.table.len(), TABLE_CAP);
        assert!(!search.table.contains_key(&new_key));
    }
    #[test]
    fn cached_entry_shares_line_allocation_instead_of_deep_cloning() {
        let mut search = dummy_search();
        let key = dummy_key(0);
        search.store(key, dummy_entry(3));
        let cached = search.table.get(&key).cloned().unwrap();
        assert!(Rc::ptr_eq(&search.table[&key].line, &cached.line));
    }
}

/// Preview an already checked input prefix; never used to decide legality.
#[no_mangle]
pub unsafe extern "C" fn santoni_preview(
    state: *const State,
    actions: *const Action,
    length: usize,
    output: *mut State,
) -> i32 {
    if state.is_null()
        || output.is_null()
        || length > MAX_ACTIONS
        || (length > 0 && actions.is_null())
    {
        return -1;
    }
    let source = *state;
    if !supported(&source) {
        return -2;
    }
    let mut current = if source.resume.length > 0 {
        let mut out = source;
        out.extra.original = source.powers;
        out.powers = source.resume.origin.powers;
        out
    } else if source.extra.event > 0 {
        source
    } else {
        special::prepare_for_turn(&source)
    };
    let entered = if length == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(actions, length)
    };
    for &a in entered {
        if a.kind > 23 || !(0..=1).contains(&a.player) {
            return -3;
        }
        if a.kind != 3 && !(0..25).contains(&a.target) {
            return -3;
        }
        let p = a.player as usize;
        if matches!(a.kind, 0 | 5..=8 | 12 | 16..=19 | 22 | 23)
            && (a.worker < 0 || a.worker as usize >= current.workers[p].len())
        {
            return -3;
        }
        if matches!(a.kind, 0 | 23) && !(0..25).contains(&a.source)
            || a.kind == 19 && a.target as usize >= current.counts[1 - p] as usize
            || a.kind == 12 && !current.workers[1 - p].contains(&a.target)
            || matches!(a.kind, 4 | 21) && current.heights[a.target as usize] == 0
            || a.kind == 1 && current.heights[a.target as usize] >= 3
            || a.kind == 13 && !(1..=10).contains(&a.target)
        {
            return -3;
        }
        current = advanced::apply_full(&current, a);
    }
    special::restore(&mut current);
    *output = current;
    0
}

/// Opening evaluation with an initial random Chaos draw. Unlike search, this
/// API returns an expected score and never suggests which card to draw.
#[no_mangle]
pub unsafe extern "C" fn santoni_expectation(
    state: *const State,
    seconds: f64,
    output: *mut CAnalysis,
    cancelled: Option<extern "C" fn() -> u8>,
) -> i32 {
    if state.is_null()
        || output.is_null()
        || !seconds.is_finite()
        || !(0.001..=3600.0).contains(&seconds)
    {
        return -1;
    }
    let s = *state;
    if !supported(&s) {
        return -2;
    }
    let started = Instant::now();
    let cancel = || cancelled.is_some_and(|f| f() != 0);
    let mut search = Search {
        started,
        deadline: started + Duration::from_secs_f64(seconds),
        cancelled: &cancel,
        nodes: 0,
        root: s.extra.return_player as usize,
        table: HashMap::new(),
    };
    let mut score = None;
    let mut completed = 0;
    for depth in 0..=64 {
        match search.minimax(&s, depth, -2 * MATE, 2 * MATE, 0) {
            Ok((v, _)) => {
                score = Some(v);
                completed = depth;
            }
            Err(_) => break,
        }
    }
    *output = search.output(&[], score, completed, score.is_some(), 8, &s);
    0
}

/// Bounded pages for callers consuming complete turns lazily. Stops generation
/// at the end of the requested page; does not materialize Morpheus/hero trees.
#[no_mangle]
pub unsafe extern "C" fn santoni_turn_page(
    state: *const State,
    prefix: *const Action,
    length: usize,
    skip: usize,
    output: *mut CTurn,
    capacity: usize,
    cancelled: Option<extern "C" fn() -> u8>,
) -> usize {
    if state.is_null()
        || !supported(&*state)
        || length > MAX_ACTIONS
        || (length > 0 && prefix.is_null())
        || output.is_null()
        || capacity == 0
    {
        return usize::MAX;
    }
    let entered = if length == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(prefix, length)
    };
    let cancel = || cancelled.is_some_and(|f| f() != 0);
    let mut seen = 0;
    let mut count = 0;
    let result = advanced::prefix_visit(&*state, None, &cancel, entered, &mut |turn| {
        if seen >= skip {
            *output.add(count) = CTurn::from(&turn);
            count += 1;
            if count == capacity {
                return Err(Stop::Enough);
            }
        }
        seen += 1;
        Ok(())
    });
    if matches!(result, Err(Stop::Deadline)) {
        usize::MAX
    } else {
        count
    }
}

/// Compare the robot's own secret objective against fully informed replies.
/// This conservative setup search never exposes the selected square to the UI.
#[no_mangle]
pub unsafe extern "C" fn santoni_choose_secret(
    state: *const State,
    seconds: f64,
    output: *mut State,
    cancelled: Option<extern "C" fn() -> u8>,
) -> i32 {
    if state.is_null()
        || output.is_null()
        || !seconds.is_finite()
        || !(0.001..=600.0).contains(&seconds)
    {
        return -1;
    }
    let s = *state;
    if !supported(&s) || !matches!(s.powers[1], 40 | 43) {
        return -2;
    }
    let started = Instant::now();
    let cancel = || cancelled.is_some_and(|f| f() != 0);
    let mut search = Search {
        started,
        deadline: started + Duration::from_secs_f64(seconds),
        cancelled: &cancel,
        nodes: 0,
        root: 1,
        table: HashMap::new(),
    };
    let cells: Vec<i8> = (0..25)
        .filter(|&c| {
            if s.powers[1] == 40 {
                c < 20 && c % 5 != 4
            } else {
                !occupied(&s, c)
            }
        })
        .collect();
    if cells.is_empty() {
        return -3;
    }
    let mut best = s;
    if s.powers[1] == 40 {
        best.extra.fate[1] = cells[0];
    } else {
        best.extra.abyss[1] = cells[0];
    }
    'deepening: for depth in 0..=32 {
        let mut best_value = -2 * MATE;
        let mut round = best;
        for &cell in &cells {
            let mut candidate = s;
            candidate.extra.view = 1;
            if s.powers[1] == 40 {
                candidate.extra.fate[1] = cell;
            } else {
                candidate.extra.abyss[1] = cell;
            }
            let value = match search.minimax(&candidate, depth, -2 * MATE, 2 * MATE, 0) {
                Ok((v, _)) => v,
                Err(_) => break 'deepening,
            };
            if value > best_value {
                best_value = value;
                round = candidate;
            }
        }
        best = round;
        if best_value.abs() >= MATE - 1000 {
            break;
        }
    }
    best.extra.view = 0;
    *output = best;
    0
}

//! Native search for ordinary Santorini and the ten basic gods.
//! No external crates. The Python reference validates every recommended turn.
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};

const MATE: i32 = 100_000;
const MAX_ACTIONS: usize = 64;

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub struct State {
    pub heights: [u8; 25],
    pub domes: u32,
    pub workers: [[i8; 3]; 2],
    pub counts: [u8; 2],
    pub powers: [u8; 2],
    pub player: u8,
    pub hero_used: [u8; 2],
    pub athena_lock: u8,
    pub adonis: [i8; 3],
    pub winner: i8,
    pub reason: u8,
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
    pub status: u8, // 0 = budget, 1 = proof, 2 = no legal turn, 3 = no fallback
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
    fn finish(&mut self, s: &State, actions: &[Action], up: bool) -> GResult {
        self.check()?;
        let mut after = *s;
        after.player = 1 - s.player;
        after.athena_lock = u8::from(s.powers[s.player as usize] == 3 && up);
        self.turns.push(Turn {
            actions: actions.to_vec(),
            after,
        });
        if self.turns.len() >= self.limit {
            return Err(Stop::Enough);
        }
        Ok(())
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
    let mut gen = Generator {
        deadline,
        cancelled,
        calls: 0,
        limit,
        turns: Vec::new(),
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
                if delta > 1 || (p == s.player as usize && s.athena_lock != 0 && delta > 0) {
                    continue;
                }
                value += 2 + 2 * delta.max(0);
                if s.heights[dst as usize] == 3 && h == 2 {
                    value += 100;
                }
                if s.powers[p] == 9 && delta <= -2 {
                    value += 100;
                }
            }
        }
        value
    };
    strength(player) - strength(1 - player)
}

#[derive(Clone)]
struct Entry {
    depth: u32,
    value: i32,
    flag: u8,
    line: Vec<Turn>,
}

struct Search<'a> {
    started: Instant,
    deadline: Instant,
    cancelled: &'a dyn Fn() -> bool,
    nodes: u64,
    root: usize,
    table: HashMap<(State, u32), Entry>,
}

impl Search<'_> {
    fn check(&self) -> Result<(), Stop> {
        if Instant::now() >= self.deadline || (self.nodes % 512 == 0 && (self.cancelled)()) {
            Err(Stop::Deadline)
        } else {
            Ok(())
        }
    }
    fn moves(&self, s: &State, preferred: Option<&Turn>) -> Result<Vec<Turn>, Stop> {
        let moves = generate(s, Some(self.deadline), self.cancelled, usize::MAX)?;
        let mut seen = HashSet::new();
        let mut unique: Vec<Turn> = moves.into_iter().filter(|t| seen.insert(t.after)).collect();
        unique.sort_by_cached_key(|t| {
            let priority = preferred.is_some_and(|p| p.after == t.after);
            (
                std::cmp::Reverse(priority),
                std::cmp::Reverse(evaluate(&t.after, s.player as usize)),
            )
        });
        self.check()?;
        Ok(unique)
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
        let original = (alpha, beta);
        let key = (*s, ply);
        let cached = self.table.get(&key).cloned();
        if let Some(entry) = &cached {
            if entry.depth >= depth {
                if entry.flag == 0 {
                    return Ok((entry.value, entry.line.clone()));
                }
                if entry.flag == 1 {
                    alpha = alpha.max(entry.value);
                } else {
                    beta = beta.min(entry.value);
                }
                if alpha >= beta {
                    return Ok((entry.value, entry.line.clone()));
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
        let moves = self.moves(s, preferred)?;
        let maximizing = s.player as usize == self.root;
        if moves.is_empty() {
            return Ok((
                if maximizing {
                    -MATE + ply as i32
                } else {
                    MATE - ply as i32
                },
                vec![],
            ));
        }
        let mut value = if maximizing { -2 * MATE } else { 2 * MATE };
        let mut line = vec![];
        for turn in moves {
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
                break;
            }
        }
        let flag = if value <= original.0 {
            2
        } else if value >= original.1 {
            1
        } else {
            0
        };
        if self.table.len() < 50_000 {
            self.table.insert(
                key,
                Entry {
                    depth,
                    value,
                    flag,
                    line: line.clone(),
                },
            );
        }
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
    s.powers.iter().all(|&p| p <= 10)
        && s.adonis == [-1; 3]
        && s.counts == [2, 2]
        && s.player <= 1
        && s.winner >= -1
        && s.winner <= 1
        && s.heights.iter().all(|&h| h <= 3)
        && s.domes < 1 << 25
        && s.workers
            .iter()
            .all(|ws| ws.iter().all(|&c| (-1..25).contains(&c)))
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
        || !(0.01..=3600.0).contains(&seconds)
    {
        return -1;
    }
    let s = *state;
    if !supported(&s) {
        return -2;
    }
    let started = Instant::now();
    let cancel = || cancelled.is_some_and(|callback| callback() != 0);
    let mut search = Search {
        started,
        deadline: started + Duration::from_secs_f64(seconds),
        cancelled: &cancel,
        nodes: 0,
        root: s.player as usize,
        table: HashMap::new(),
    };
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
        let moves = match search.moves(&s, best_line.first()) {
            Ok(m) => m,
            Err(_) => break,
        };
        let mut value = -2 * MATE;
        let mut line = vec![];
        for turn in moves {
            let (v, mut tail) = match search.minimax(&turn.after, depth - 1, value, 2 * MATE, 1) {
                Ok(r) => r,
                Err(_) => break 'deepening,
            };
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
                *output = search.output(&line, Some(v), depth, true, 1, &s);
                if let Some(callback) = publish {
                    callback(output);
                }
                return 0;
            }
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
        u8::from(best_score.is_some_and(|v| v.abs() >= MATE - 1000)),
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

#[cfg(test)]
mod tests {
    use super::*;
    fn initial() -> State {
        State {
            heights: [0; 25],
            domes: 0,
            workers: [[6, 8, -1], [16, 18, -1]],
            counts: [2, 2],
            powers: [0, 0],
            player: 0,
            hero_used: [0; 2],
            athena_lock: 0,
            adonis: [-1; 3],
            winner: -1,
            reason: 0,
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
}

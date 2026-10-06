//! Public advanced gods and heroes. Complete turns, independent of Python.
use super::*;

const ACTIVATE: u8 = 3;
const REMOVE: u8 = 4;
const KILL: u8 = 5;
const FORCE: u8 = 6;
const PLACE: u8 = 7;
const ADONIS: u8 = 8;

pub(super) fn border(c: i8) -> bool {
    c % 5 == 0 || c % 5 == 4 || c / 5 == 0 || c / 5 == 4
}
fn neighbours(s: &State, c: i8) -> Vec<i8> {
    if s.powers[s.player as usize] != 45 {
        return adjacent(c);
    }
    let mut result = Vec::new();
    for dy in -1..=1 {
        for dx in -1..=1 {
            if dx != 0 || dy != 0 {
                result.push((c / 5 + dy).rem_euclid(5) * 5 + (c % 5 + dx).rem_euclid(5));
            }
        }
    }
    result
}
fn put(s: &mut State, p: usize, w: usize, cell: i8) {
    s.workers[p][w] = cell;
    s.counts[p] = s.counts[p].max((w + 1) as u8);
}
pub(super) fn apply_full(s: &State, a: Action) -> State {
    let mut out = *s;
    if a.kind == ACTIVATE {
        out.hero_used[s.player as usize] = 1;
        return out;
    }
    if a.kind == ADONIS {
        out.adonis = [a.player, a.worker, s.player as i8];
        return out;
    }
    let p = a.player as usize;
    if s.powers[1 - p] == 39
        && s.workers[1 - p].contains(&a.target)
        && matches!(a.kind, 0..=2 | 7)
        && !(a.kind == 0 && matches!(s.powers[p], 1 | 8))
    {
        out.reason = 7;
        if s.extra.event == 2 {
            out.extra.event = 0;
        }
        return out;
    }
    if a.kind == 0
        && s.powers[p] == 8
        && s.powers[1 - p] == 39
        && s.workers[1 - p].contains(&a.target)
    {
        let x = 2 * (a.target % 5) - a.source % 5;
        let y = 2 * (a.target / 5) - a.source / 5;
        if !(0..5).contains(&x)
            || !(0..5).contains(&y)
            || occupied(s, y * 5 + x)
            || special::blocked(s, p, y * 5 + x)
        {
            out.reason = 7;
            return out;
        }
    }
    match a.kind {
        0 => {
            let foe = 1 - p;
            if let Some(v) = s.workers[foe].iter().position(|&c| c >= 0 && c == a.target) {
                if s.powers[p] == 1 {
                    put(&mut out, foe, v, a.source);
                } else if s.powers[p] == 8 {
                    put(
                        &mut out,
                        foe,
                        v,
                        (2 * (a.target / 5) - a.source / 5) * 5 + 2 * (a.target % 5) - a.source % 5,
                    );
                }
            }
            put(&mut out, p, a.worker as usize, a.target);
            let h0 = s.heights[a.source as usize];
            let h1 = s.heights[a.target as usize];
            let pan = s.powers[p] == 9 && h0 as i8 - h1 as i8 >= 2;
            if (h1 == 3 && h0 < 3 || pan) && !(s.powers[foe] == 20 && border(a.target)) {
                out.winner = p as i8;
                out.reason = if pan { 2 } else { 1 };
            }
            if s.powers[p] == 13 {
                let x = 2 * (a.target % 5) - a.source % 5;
                let y = 2 * (a.target / 5) - a.source / 5;
                if (0..5).contains(&x) && (0..5).contains(&y) && !special::blocked(s, p, y * 5 + x)
                {
                    if let Some(v) = out.workers[foe].iter().position(|&c| c == y * 5 + x) {
                        put(&mut out, foe, v, -1);
                    }
                }
            }
            if s.powers[p] == 19
                && h1 == 1
                && !(s.powers[foe] == 20 && border(a.target))
                && out.workers[p].iter().enumerate().any(|(w, &c)| {
                    w != a.worker as usize && c >= 0 && adjacent(a.target).contains(&c)
                })
            {
                out.winner = p as i8;
                out.reason = 3;
            }
        }
        FORCE | PLACE => put(&mut out, p, a.worker as usize, a.target),
        KILL => put(&mut out, p, a.worker as usize, -1),
        1 => out.heights[a.target as usize] += 1,
        REMOVE => out.heights[a.target as usize] -= 1,
        2 => {
            out.domes |= 1 << a.target;
            for owner in 0..2 {
                if out.powers[owner] == 16
                    && (0..25)
                        .filter(|&i| out.heights[i] == 3 && out.domes & (1 << i) != 0)
                        .count()
                        >= 5
                {
                    out.winner = owner as i8;
                    out.reason = 4;
                }
            }
        }
        17 | 18 => {
            let src = s.workers[p][a.worker as usize];
            if src >= 0 {
                if a.kind == 17 {
                    let (dx, dy) = special::vector(a.target);
                    let x = src % 5 + dx;
                    let y = src / 5 + dy;
                    if (0..5).contains(&x)
                        && (0..5).contains(&y)
                        && step(s, a.worker as usize, y * 5 + x, false, false, false)
                    {
                        return apply_full(s, action(0, p, a.worker as usize, src, y * 5 + x));
                    }
                } else if let Some(build) = builds(s, a.worker as usize, false, false)
                    .into_iter()
                    .find(|b| b.target == a.target)
                {
                    return apply_full(s, build);
                }
                out.reason = 7;
            }
            return out;
        }
        19 => {
            let foe = 1 - p;
            let v = a.target as usize;
            if s.extra.nemesis_active == 0
                && s.workers[p].iter().any(|&c| {
                    c >= 0
                        && s.workers[foe]
                            .iter()
                            .any(|&d| d >= 0 && adjacent(c).contains(&d))
                })
            {
                out.reason = 7;
                return out;
            }
            out.extra.nemesis_active = 1;
            let dst = s.workers[foe][v];
            out.workers[p][a.worker as usize] = dst;
            out.workers[foe][v] = a.source;
            if dst >= 0 {
                special::hazards(&mut out, p, dst);
            }
            special::hazards(&mut out, foe, a.source);
            return out;
        }
        20 => {
            if let Some(v) = s.workers[p].iter().position(|&c| c == a.target) {
                out.workers[p][v] = -1;
            } else if !s.workers[p].contains(&-2) {
                out.reason = 7;
            }
            return out;
        }
        21 => {
            if s.workers.iter().flatten().any(|&c| c == a.target) {
                out.heights[a.target as usize] -= 1;
                special::remove_tokens(&mut out, a.target);
            } else if !s.workers[1 - p].contains(&-2) {
                out.reason = 7;
            }
            return out;
        }
        22 => {
            let controller = 1 - p;
            let src = s.workers[p][a.worker as usize];
            if src == -2
                || src >= 0
                    && s.workers[controller]
                        .iter()
                        .any(|&c| c >= 0 && adjacent(c).contains(&src))
                    && !occupied(s, a.target)
            {
                out.workers[p][a.worker as usize] = a.target;
                special::hazards(&mut out, p, a.target);
            } else {
                out.reason = 7;
            }
            return out;
        }
        23 => {
            let x = 2 * (a.source % 5) - a.target % 5;
            let y = 2 * (a.source / 5) - a.target / 5;
            if (0..5).contains(&x) && (0..5).contains(&y) && !occupied(s, a.target) {
                if let Some(v) = s.workers[1 - p]
                    .iter()
                    .position(|&c| c == y * 5 + x || c == -2)
                {
                    out.workers[1 - p][v] = a.target;
                    special::hazards(&mut out, 1 - p, a.target);
                } else {
                    out.reason = 7;
                }
            } else {
                out.reason = 7;
            }
            return out;
        }
        9..=16 => (),
        _ => unreachable!(),
    }
    special::after_action(s, &mut out, a);
    if matches!(a.kind, 2 | 13..=15) {
        special::terminalize(&mut out);
    }
    out
}
fn step(s: &State, w: usize, dst: i8, active: bool, no_up: bool, flat: bool) -> bool {
    let p = s.player as usize;
    let foe = 1 - p;
    let src = s.workers[p][w];
    if special::blocked(s, p, src) {
        return false;
    }
    if special::blocked(s, p, dst)
        || (s.extra.wind >= 0
            && (if s.powers[p] == 45 {
                special::wrapped_direction(src, dst)
            } else {
                special::direction(src, dst)
            }) == s.extra.wind)
    {
        return false;
    }
    if occupied(s, dst) && !s.workers[foe].contains(&dst)
        || s.domes & (1 << dst) != 0
        || s.workers[p].contains(&dst)
    {
        return false;
    }
    let delta = s.heights[dst as usize] as i8 - s.heights[src as usize] as i8;
    if delta > if active && s.powers[p] == 49 { 2 } else { 1 }
        || flat && delta != 0
        || delta > 0 && (s.athena_lock != 0 || no_up)
        || delta < 0 && s.powers[foe] == 37
    {
        return false;
    }
    if s.workers[foe].contains(&dst) {
        if s.powers[p] == 1 {
            return true;
        }
        if s.powers[p] != 8 {
            return s.powers[foe] == 39;
        }
        let x = 2 * (dst % 5) - src % 5;
        let y = 2 * (dst / 5) - src / 5;
        return s.powers[foe] == 39
            || (0..5).contains(&x)
                && (0..5).contains(&y)
                && !occupied(s, y * 5 + x)
                && !special::blocked(s, p, y * 5 + x);
    }
    true
}
fn attempt_occupied(s: &State, c: i8) -> bool {
    let foe = 1 - s.player as usize;
    special::blocked(s, s.player as usize, c)
        || s.domes & (1 << c) != 0
        || s.workers[s.player as usize].contains(&c)
        || (s.powers[foe] != 39 && s.workers[foe].contains(&c))
}
fn builds(s: &State, w: usize, atlas: bool, selene: bool) -> Vec<Action> {
    let p = s.player as usize;
    let foe = 1 - p;
    let src = s.workers[p][w];
    let mut out = Vec::new();
    if src < 0 {
        return out;
    }
    for dst in neighbours(s, src) {
        if (occupied(s, dst) && !(s.powers[foe] == 39 && s.workers[foe].contains(&dst)))
            || special::blocked(s, p, dst)
            || s.powers[foe] == 23
                && s.heights[dst as usize] != 3
                && s.workers[foe]
                    .iter()
                    .any(|&c| c >= 0 && adjacent(c).contains(&dst))
        {
            continue;
        }
        let kind = if s.heights[dst as usize] == 3 { 2 } else { 1 };
        if !selene {
            out.push(action(kind, p, w, src, dst));
        }
        if (atlas || selene) && (kind != 2 || selene) {
            out.push(action(2, p, w, src, dst));
        }
    }
    if s.powers[p] == 30
        && s.heights[src as usize] < 3
        && !selene
        && (s.powers[foe] != 23
            || !s.workers[foe]
                .iter()
                .any(|&c| c >= 0 && adjacent(c).contains(&src)))
    {
        out.push(action(1, p, w, src, src));
    }
    out
}

struct Rules<'a> {
    origin: State,
    replay_len: usize,
    gen: Generator<'a>,
    active: bool,
    need_up: bool,
    need_adonis: bool,
    probe_up: bool,
    probe_adonis: bool,
    found: bool,
    prefix: Vec<Action>,
    filter_next: bool,
    next_seen: HashSet<Action>,
}
impl Rules<'_> {
    fn fits(&self, path: &[Action]) -> bool {
        let n = path.len().min(self.prefix.len());
        path[..n] == self.prefix[..n]
            && !(self.filter_next
                && path.len() > self.prefix.len()
                && self.next_seen.contains(&path[self.prefix.len()]))
    }

    fn finish(&mut self, s: &State, path: &[Action], up: bool) -> GResult {
        if self.interrupt(s, path)? {
            return Ok(());
        }
        self.gen.check()?;
        if path.len() < self.prefix.len() || !self.fits(path) {
            return Ok(());
        }
        let constraint = self.origin.adonis;
        let p = self.origin.player as usize;
        let adjacent_ok = if constraint[0] == p as i8 {
            let dst = s.workers[p][constraint[1] as usize];
            dst >= 0
                && s.workers[1 - p]
                    .iter()
                    .any(|&v| v >= 0 && adjacent(dst).contains(&v))
        } else {
            true
        };
        if s.reason != 7 && ((self.need_up && !up) || (self.need_adonis && !adjacent_ok)) {
            return Ok(());
        }
        if self.probe_up || self.probe_adonis {
            if (!self.probe_up || up) && (!self.probe_adonis || adjacent_ok) {
                self.found = true;
                return Err(Stop::Enough);
            }
            return Ok(());
        }
        let mut after = *s;
        after.resume = special::Resume::default();
        after.extra.nemesis_active = 0;
        after.player = if self.origin.extra.dion_owner >= 0 {
            1 - self.origin.extra.dion_owner as u8
        } else {
            1 - self.origin.player
        };
        after.athena_lock = u8::from(self.origin.powers[p] == 3 && up);
        if constraint[0] == p as i8 {
            after.adonis = [-1; 3];
        }
        special::restore(&mut after);
        if after.winner < 0 {
            let new_domes = s.domes & !self.origin.domes;
            let complete_tower = (0..25).any(|c| new_domes & (1 << c) != 0 && s.heights[c] == 3);
            if self.origin.powers[p] == 18 && complete_tower {
                after.extra.event = 3;
                after.extra.event_owner = if self.origin.extra.dion_owner >= 0 {
                    self.origin.extra.dion_owner as u8
                } else {
                    p as u8
                };
                after.extra.return_player = if self.origin.extra.dion_owner >= 0 {
                    1 - self.origin.extra.dion_owner as u8
                } else {
                    1 - p as u8
                };
                after.player = after.extra.event_owner;
            } else {
                after.extra.dion_owner = -1;
            }
            if new_domes != 0 {
                for owner in 0..2 {
                    if after.powers[owner] == 14 {
                        special::schedule_draw(&mut after, owner);
                        break;
                    }
                }
            }
        }
        if self.origin.extra.dion_owner >= 0 && after.extra.dion_owner < 0 {
            special::terminalize(&mut after);
        }
        if self.filter_next && path.len() > self.prefix.len() {
            self.next_seen.insert(path[self.prefix.len()]);
        }
        self.gen.emit(Turn {
            actions: path[self.replay_len..].to_vec(),
            after,
        })
    }
    fn add(&self, s: &State, path: &[Action], a: Action) -> (State, Vec<Action>) {
        let mut line = path.to_vec();
        line.push(a);
        let mut after = apply_full(s, a);
        if after.extra.event == 2 {
            if let Some(&response) = self.prefix.get(line.len()) {
                if special::gaea_response_ok(&after, response) {
                    after = apply_full(&after, response);
                    if after.reason == 7 {
                        after.reason = 0;
                    }
                    line.push(response);
                }
            }
        }
        (after, line)
    }
    fn adjacency_ok(&self, s: &State, w: usize) -> bool {
        let p = self.origin.player as usize;
        let foe = 1 - p;
        let src = self.origin.workers[p][w];
        if self.origin.powers[foe] != 11
            || src < 0
            || !self.origin.workers[foe]
                .iter()
                .any(|&v| v >= 0 && adjacent(src).contains(&v))
        {
            return true;
        }
        let dst = s.workers[p][w];
        dst >= 0
            && s.workers[foe]
                .iter()
                .any(|&v| v >= 0 && adjacent(dst).contains(&v))
    }
    fn hypnus(&self, w: usize) -> bool {
        let p = self.origin.player as usize;
        let src = self.origin.workers[p][w];
        self.origin.powers[1 - p] == 22
            && self.origin.workers.iter().enumerate().all(|(owner, ws)| {
                ws.iter().enumerate().all(|(i, &c)| {
                    c < 0
                        || (owner == p && i == w)
                        || self.origin.heights[src as usize] > self.origin.heights[c as usize]
                })
            })
    }
    fn interrupt(&mut self, s: &State, path: &[Action]) -> Result<bool, Stop> {
        if s.extra.event != 2 || s.winner >= 0 {
            return Ok(false);
        }
        if !self.fits(path) || path.len() < self.prefix.len() {
            return Ok(true);
        }
        if self.probe_up || self.probe_adonis {
            let mut at = self.origin;
            let mut up = false;
            for &a in path {
                if a.kind == 0 {
                    up |= at.heights[a.target as usize] > at.heights[a.source as usize];
                }
                at = apply_full(&at, a);
            }
            let constraint = self.origin.adonis;
            let p = self.origin.player as usize;
            let dst = if constraint[0] == p as i8 {
                s.workers[p][constraint[1] as usize]
            } else {
                -1
            };
            let adjacent_ok = constraint[0] != p as i8
                || dst >= 0
                    && s.workers[1 - p]
                        .iter()
                        .any(|&c| c >= 0 && adjacent(dst).contains(&c));
            if (!self.probe_up || up) && (!self.probe_adonis || adjacent_ok) {
                self.found = true;
                return Err(Stop::Enough);
            }
            return Ok(true);
        }
        let mut after = *s;
        after.resume.origin = self.origin.snapshot();
        after.resume.length = path.len() as u32;
        after.resume.actions[..path.len()].copy_from_slice(path);
        after.extra.return_player = if self.origin.extra.dion_owner >= 0 {
            self.origin.extra.dion_owner as u8
        } else {
            self.origin.player
        };
        after.player = after.extra.event_owner;
        special::restore(&mut after);
        special::terminalize(&mut after);
        if self.filter_next && path.len() > self.prefix.len() {
            self.next_seen.insert(path[self.prefix.len()]);
        }
        self.gen.emit(Turn {
            actions: path[self.replay_len..].to_vec(),
            after,
        })?;
        Ok(true)
    }
    fn subsets(
        &mut self,
        s: &State,
        path: &[Action],
        options: &[Action],
        start: usize,
        up: bool,
    ) -> GResult {
        if !self.fits(path) {
            return Ok(());
        }
        if self.interrupt(s, path)? {
            return Ok(());
        }
        self.finish(s, path, up)?;
        if s.winner >= 0 || s.reason == 7 {
            return Ok(());
        }
        for i in start..options.len() {
            if options[i].kind == 2 && attempt_occupied(s, options[i].target) {
                continue;
            }
            if path.iter().any(|a| a == &options[i]) {
                continue;
            }
            let (after, line) = self.add(s, path, options[i]);
            self.subsets(
                &after,
                &line,
                options,
                if path.len() < self.prefix.len() {
                    0
                } else {
                    i + 1
                },
                up,
            )?;
        }
        Ok(())
    }
    fn extras(&mut self, s: &State, path: &[Action], w: usize, count: usize, up: bool) -> GResult {
        if !self.fits(path) {
            return Ok(());
        }
        if self.interrupt(s, path)? {
            return Ok(());
        }
        self.finish(s, path, up)?;
        if s.winner >= 0 || s.reason == 7 {
            return Ok(());
        }
        if count > 0 {
            for a in builds(s, w, false, false) {
                let (after, line) = self.add(s, path, a);
                self.extras(&after, &line, w, count - 1, up)?;
            }
        }
        Ok(())
    }
    fn end(&mut self, s: &State, path: &[Action], w: usize, up: bool) -> GResult {
        if !self.fits(path) {
            return Ok(());
        }
        if s.winner >= 0 || s.reason == 7 {
            return self.finish(s, path, up);
        }
        if self.interrupt(s, path)? {
            return Ok(());
        }
        let p = s.player as usize;
        let foe = 1 - p;
        let power = s.powers[p];
        let mut s = *s;
        let mut path = path.to_vec();
        if matches!(power, 31 | 32 | 34 | 41) {
            return self.token_end(&s, &path, w, up);
        }
        if power == 24 {
            let src = s.workers[p][w];
            for victim in 0..s.counts[foe] as usize {
                let dst = s.workers[foe][victim];
                if dst < 0
                    || !adjacent(src).contains(&dst)
                    || special::blocked(&s, p, dst)
                    || s.heights[dst as usize] >= s.heights[src as usize]
                {
                    continue;
                }
                if s.powers[foe] == 23
                    && s.workers[foe]
                        .iter()
                        .any(|&c| c >= 0 && adjacent(c).contains(&dst))
                {
                    continue;
                }
                (s, path) = self.add(&s, &path, action(KILL, foe, victim, dst, dst));
                (s, path) = self.add(&s, &path, action(1, p, w, src, dst));
            }
        }
        if power == 12 {
            self.finish(&s, &path, up)?;
            for idle in 0..s.counts[p] as usize {
                let src = s.workers[p][idle];
                if idle == w || src < 0 {
                    continue;
                }
                for dst in adjacent(src) {
                    if !attempt_occupied(&s, dst) && s.heights[dst as usize] > 0 {
                        let (after, line) = self.add(&s, &path, action(REMOVE, p, idle, src, dst));
                        self.finish(&after, &line, up)?;
                    }
                }
            }
            return Ok(());
        }
        if power == 27 && w < 2 {
            let idle = 1 - w;
            let src = s.workers[p][idle];
            if src >= 0 && s.heights[src as usize] == 0 {
                return self.extras(&s, &path, idle, 3, up);
            }
        }
        if self.active && power == 47 {
            for victim in 0..s.counts[foe] as usize {
                let dst = s.workers[foe][victim];
                if dst >= 0 || dst == -2 {
                    let target = if s.powers[foe] == 39 { 12 } else { dst };
                    let source = if s.powers[foe] == 39 { -1 } else { dst };
                    let (after, line) =
                        self.add(&s, &path, action(ADONIS, foe, victim, source, target));
                    self.finish(&after, &line, up)?;
                }
            }
            return Ok(());
        }
        if self.active && (power == 50 || power == 52) {
            let mut options = Vec::new();
            if power == 50 {
                for dst in 0..25 {
                    if !attempt_occupied(&s, dst)
                        && s.workers[p]
                            .iter()
                            .any(|&src| src >= 0 && adjacent(src).contains(&dst))
                        && (s.powers[foe] != 23
                            || s.heights[dst as usize] == 3
                            || !s.workers[foe]
                                .iter()
                                .any(|&c| c >= 0 && adjacent(c).contains(&dst)))
                    {
                        options.push(Action {
                            kind: 2,
                            player: p as i8,
                            worker: -1,
                            source: -1,
                            target: dst,
                        });
                    }
                }
            } else if w < 2 {
                let idle = 1 - w;
                let src = s.workers[p][idle];
                if src >= 0 {
                    for (owner, ws) in s.workers.iter().enumerate() {
                        if owner == foe && s.powers[foe] == 39 {
                            continue;
                        }
                        for &dst in ws {
                            if dst >= 0
                                && adjacent(src).contains(&dst)
                                && s.heights[dst as usize] > 0
                                && !special::blocked(&s, p, dst)
                            {
                                options.push(action(REMOVE, p, idle, src, dst));
                            }
                        }
                    }
                    if s.powers[foe] == 39 {
                        for dst in adjacent(src) {
                            if s.heights[dst as usize] > 0
                                && s.domes & (1 << dst) == 0
                                && !s.workers[p].contains(&dst)
                            {
                                options.push(action(21, p, idle, src, dst));
                            }
                        }
                    }
                    options.sort_by_key(|a| a.target);
                }
            }
            return self.subsets(&s, &path, &options, 0, up);
        }
        if self.active && power == 54 {
            self.finish(&s, &path, up)?;
            let src = s.workers[p][w];
            for dst in 0..25 {
                if attempt_occupied(&s, dst)
                    || s.powers[foe] == 23
                        && s.heights[dst as usize] != 3
                        && s.workers[foe]
                            .iter()
                            .any(|&c| c >= 0 && adjacent(c).contains(&dst))
                {
                    continue;
                }
                let (after, line) = self.add(&s, &path, action(2, p, w, src, dst));
                self.finish(&after, &line, up)?;
                if after.winner >= 0 || after.reason == 7 {
                    continue;
                }
                for dst2 in if self.prefix.is_empty() { dst + 1 } else { 0 }..25 {
                    if attempt_occupied(&after, dst2)
                        || after.powers[foe] == 23
                            && after.heights[dst2 as usize] != 3
                            && after.workers[foe]
                                .iter()
                                .any(|&c| c >= 0 && adjacent(c).contains(&dst2))
                    {
                        continue;
                    }
                    let (after2, line2) = self.add(&after, &line, action(2, p, w, src, dst2));
                    self.finish(&after2, &line2, up)?;
                }
            }
            return Ok(());
        }
        if self.active && power == 55 && s.powers[foe] == 39 {
            for dst in 0..25 {
                if s.domes & (1 << dst) == 0
                    && !s.workers[p].contains(&dst)
                    && s.workers[p].iter().any(|&src| {
                        src >= 0
                            && adjacent(src).contains(&dst)
                            && s.heights[dst as usize] as i8 - s.heights[src as usize] as i8 == 2
                    })
                {
                    let (after, line) = self.add(
                        &s,
                        &path,
                        Action {
                            kind: 20,
                            player: foe as i8,
                            worker: -1,
                            source: -1,
                            target: dst,
                        },
                    );
                    self.finish(&after, &line, up)?;
                }
            }
            return Ok(());
        }
        if self.active && power == 55 {
            for victim in 0..s.counts[foe] as usize {
                let dst = s.workers[foe][victim];
                if dst >= 0
                    && !special::blocked(&s, p, dst)
                    && s.workers[p].iter().any(|&src| {
                        src >= 0
                            && adjacent(src).contains(&dst)
                            && s.heights[dst as usize] as i8 - s.heights[src as usize] as i8 == 2
                    })
                {
                    let (after, line) = self.add(&s, &path, action(KILL, foe, victim, dst, dst));
                    self.finish(&after, &line, up)?;
                }
            }
            return Ok(());
        }
        self.finish(&s, &path, up)
    }
    fn construct(&mut self, s: &State, path: &[Action], w: usize, up: bool) -> GResult {
        // Building cannot introduce a voluntary ascent after movement is over.
        if (self.probe_up || self.need_up) && !up {
            return Ok(());
        }
        if (self.probe_adonis || self.need_adonis) && s.powers[s.player as usize] == 25 {
            let c = self.origin.adonis;
            let p = s.player as usize;
            if c[0] == p as i8 {
                let dst = s.workers[p][c[1] as usize];
                if dst < 0
                    || !s.workers[1 - p]
                        .iter()
                        .any(|&v| v >= 0 && adjacent(dst).contains(&v))
                {
                    return Ok(());
                }
            }
        }
        if !self.fits(path) {
            return Ok(());
        }
        if s.winner >= 0 || s.reason == 7 {
            return self.finish(s, path, up);
        }
        if self.interrupt(s, path)? {
            return Ok(());
        }
        let p = s.player as usize;
        let power = s.powers[p];
        if power == 25 {
            return self.materials(s, path, w, up);
        }
        if power == 44 {
            return self.dance_builds(s, path, 0, up);
        }
        let mut builders = if power == 36 {
            (0..s.counts[p] as usize)
                .filter(|&i| s.workers[p][i] >= 0)
                .map(|i| (i, false))
                .collect()
        } else {
            vec![(w, false)]
        };
        if power == 28 && s.counts[p] > 1 && s.workers[p][1] >= 0 {
            builders.push((1, true));
        }
        for (builder, selene) in builders {
            for a in builds(s, builder, power == 4, selene) {
                let (after, line) = self.add(s, path, a);
                self.end(&after, &line, w, up)?;
                if after.winner >= 0 || after.reason == 7 {
                    continue;
                }
                if matches!(power, 5 | 6 | 21) {
                    for second in builds(&after, w, false, false) {
                        if power == 5 && second.target == a.target
                            || power == 6
                                && (second.target != a.target || a.kind != 1 || second.kind != 1)
                            || power == 21 && border(second.target)
                        {
                            continue;
                        }
                        let (after2, line2) = self.add(&after, &line, second);
                        self.end(&after2, &line2, w, up)?;
                    }
                }
            }
        }
        Ok(())
    }
    fn moved(&mut self, s: &State, path: &[Action], w: usize, no_up: bool) -> GResult {
        if s.winner >= 0 || s.reason == 7 {
            return self.finish(s, path, false);
        }
        if !self.fits(path) {
            return Ok(());
        }
        if self.interrupt(s, path)? {
            return Ok(());
        }
        let p = s.player as usize;
        let power = s.powers[p];
        let src = s.workers[p][w];
        for dst in neighbours(s, src) {
            self.gen.check()?;
            if !step(s, w, dst, self.active, no_up, false) {
                continue;
            }
            let (after, line) = self.add(s, path, action(0, p, w, src, dst));
            let up = s.heights[dst as usize] > s.heights[src as usize];
            if !self.fits(&line) {
                continue;
            }
            if self.adjacency_ok(&after, w) {
                self.construct(&after, &line, w, up)?;
            }
            if after.winner >= 0 || after.reason == 7 {
                continue;
            }
            if power == 2 {
                let current = after.workers[p][w];
                for dst2 in adjacent(current) {
                    if dst2 == src || !step(&after, w, dst2, self.active, no_up, false) {
                        continue;
                    }
                    let (after2, line2) = self.add(&after, &line, action(0, p, w, current, dst2));
                    if self.adjacency_ok(&after2, w) {
                        self.construct(
                            &after2,
                            &line2,
                            w,
                            up || after.heights[dst2 as usize] > after.heights[current as usize],
                        )?;
                    }
                }
            } else if power == 29 || self.active && power == 48 {
                let mut seen = if line.len() >= self.prefix.len() {
                    HashSet::from([(after.workers[p][w], up)])
                } else {
                    HashSet::new()
                };
                let mut queue = VecDeque::from([(after, line, up)]);
                while let Some((ss, aa, climbed)) = queue.pop_front() {
                    self.gen.check()?;
                    let current = ss.workers[p][w];
                    if power == 29 && !border(current) {
                        continue;
                    }
                    for nxt in adjacent(current) {
                        if !step(&ss, w, nxt, self.active, no_up, false) {
                            continue;
                        }
                        let (after2, line2) = self.add(&ss, &aa, action(0, p, w, current, nxt));
                        let new_up =
                            climbed || ss.heights[nxt as usize] > ss.heights[current as usize];
                        if !self.fits(&line2) {
                            continue;
                        }
                        if line2.len() >= self.prefix.len()
                            && !seen.insert((after2.workers[p][w], new_up))
                        {
                            continue;
                        }
                        if self.adjacency_ok(&after2, w) {
                            self.construct(&after2, &line2, w, new_up)?;
                        }
                        if after2.winner < 0 && after2.reason != 7 {
                            queue.push_back((after2, line2, new_up));
                        }
                    }
                }
            }
        }
        Ok(())
    }
    fn ordinary(&mut self, s: &State, path: &[Action]) -> GResult {
        if !self.fits(path) {
            return Ok(());
        }
        if self.interrupt(s, path)? {
            return Ok(());
        }
        if s.winner >= 0 || s.reason == 7 {
            return self.finish(s, path, false);
        }
        let p = s.player as usize;
        let foe = 1 - p;
        let power = s.powers[p];
        if power == 18 && s.extra.dion_owner >= 0 && s.extra.original[p] == 39 {
            return self.blind_dion(s, path);
        }
        if power == 44 {
            return self.dance_moves(s, path, 0, false);
        }
        if power == 42 {
            self.sing(s, path, 0)?;
        }
        if self.active && power == 51 {
            for dst in 0..25 {
                if border(dst) && !attempt_occupied(s, dst) && s.heights[dst as usize] == 0 {
                    let (after, line) =
                        self.add(s, path, action(PLACE, p, s.counts[p] as usize, -1, dst));
                    self.construct(&after, &line, s.counts[p] as usize, false)?;
                }
            }
            return Ok(());
        }
        if power == 7 && !self.probe_up {
            let mut seen = if path.len() >= self.prefix.len() {
                HashSet::from([s.workers[p]])
            } else {
                HashSet::new()
            };
            let mut queue = VecDeque::from([(*s, path.to_vec())]);
            while let Some((ss, aa)) = queue.pop_front() {
                self.gen.check()?;
                if (0..ss.counts[p] as usize)
                    .all(|w| ss.workers[p][w] < 0 || self.adjacency_ok(&ss, w))
                {
                    for w in 0..ss.counts[p] as usize {
                        if ss.workers[p][w] >= 0 {
                            self.construct(&ss, &aa, w, false)?;
                        }
                    }
                }
                for w in 0..ss.counts[p] as usize {
                    let src = ss.workers[p][w];
                    if src < 0 || self.hypnus(w) {
                        continue;
                    }
                    for dst in adjacent(src) {
                        if !step(&ss, w, dst, false, false, true) {
                            continue;
                        }
                        let (after, line) = self.add(&ss, &aa, action(0, p, w, src, dst));
                        if after.reason == 7 && self.fits(&line) {
                            self.finish(&after, &line, false)?;
                            continue;
                        }
                        if self.fits(&line)
                            && (line.len() < self.prefix.len() || seen.insert(after.workers[p]))
                        {
                            queue.push_back((after, line));
                        }
                    }
                }
            }
        }
        for w in 0..s.counts[p] as usize {
            let src = s.workers[p][w];
            if src < 0 || self.hypnus(w) {
                continue;
            }
            let mut pre = vec![(*s, path.to_vec(), false)];
            if power == 10 || self.active && power == 46 {
                if self.active && power == 46 {
                    pre.clear();
                }
                for a in builds(s, w, false, false) {
                    let (after, line) = self.add(s, path, a);
                    if after.winner >= 0 {
                        self.finish(&after, &line, false)?;
                    } else {
                        pre.push((after, line, power == 10));
                    }
                }
            }
            if power == 15 && s.powers[foe] == 39 {
                for dst in adjacent(src) {
                    let x = 2 * (src % 5) - dst % 5;
                    let y = 2 * (src / 5) - dst / 5;
                    if (0..5).contains(&x)
                        && (0..5).contains(&y)
                        && !s.workers[p].contains(&(y * 5 + x))
                        && s.domes & (1 << dst) == 0
                        && !s.workers[p].contains(&dst)
                        && !special::blocked(s, p, dst)
                    {
                        let (after, line) = self.add(s, path, action(23, p, w, src, dst));
                        pre.push((after, line, false));
                    }
                }
            }
            if power == 15 && s.powers[foe] != 39 {
                for v in 0..s.counts[foe] as usize {
                    let dst = s.workers[foe][v];
                    if dst < 0 || !adjacent(src).contains(&dst) || special::blocked(s, p, dst) {
                        continue;
                    }
                    let x = 2 * (src % 5) - dst % 5;
                    let y = 2 * (src / 5) - dst / 5;
                    if (0..5).contains(&x) && (0..5).contains(&y) && !attempt_occupied(s, y * 5 + x)
                    {
                        let (after, line) =
                            self.add(s, path, action(FORCE, foe, v, dst, y * 5 + x));
                        pre.push((after, line, false));
                    }
                }
            }
            for (ss, aa, no_up) in pre {
                self.moved(&ss, &aa, w, no_up)?;
            }
        }
        Ok(())
    }
    fn corners(&mut self, s: &State, path: &[Action], remaining: u8) -> GResult {
        if !self.fits(path) {
            return Ok(());
        }
        if s.winner >= 0 || s.reason == 7 {
            return self.finish(s, path, false);
        }
        self.ordinary(s, path)?;
        let p = s.player as usize;
        let foe = 1 - p;
        for v in 0..s.counts[foe] as usize {
            if remaining & (1 << v) == 0 {
                continue;
            }
            let dst = s.workers[foe][v];
            if s.powers[foe] == 39 {
                if dst == -1 {
                    continue;
                }
                for corner in [0, 4, 20, 24] {
                    if s.domes & (1 << corner) == 0
                        && !s.workers[p].contains(&corner)
                        && !special::blocked(s, p, corner)
                    {
                        let (after, line) = self.add(s, path, action(22, foe, v, -1, corner));
                        self.corners(&after, &line, remaining & !(1 << v))?;
                    }
                }
                continue;
            }
            if dst < 0
                || !s.workers[p]
                    .iter()
                    .any(|&c| c >= 0 && adjacent(c).contains(&dst))
            {
                continue;
            }
            for corner in [0, 4, 20, 24] {
                if !attempt_occupied(s, corner) {
                    let (after, line) = self.add(s, path, action(FORCE, foe, v, dst, corner));
                    self.corners(&after, &line, remaining & !(1 << v))?;
                }
            }
        }
        Ok(())
    }
    fn materials(&mut self, s: &State, path: &[Action], w: usize, up: bool) -> GResult {
        if !self.fits(path) {
            return Ok(());
        }
        if self.interrupt(s, path)? {
            return Ok(());
        }
        self.end(s, path, w, up)?;
        let owner = special::card_owner(s, s.player as usize, 25);
        if s.winner >= 0 || s.reason == 7 || s.extra.materials[owner] == 0 {
            return Ok(());
        }
        for a in builds(s, w, false, false) {
            let (mut after, line) = self.add(s, path, a);
            if after.reason != 7 {
                after.extra.materials[owner] -= 1;
            }
            self.materials(&after, &line, w, up)?;
        }
        Ok(())
    }
    fn blind_dion(&mut self, s: &State, path: &[Action]) -> GResult {
        let p = s.player as usize;
        let controller = s.extra.dion_owner as usize;
        for w in 0..s.counts[p] as usize {
            if s.workers[p][w] == -1 {
                continue;
            }
            for dir in special::DIRECTIONS {
                self.gen.check()?;
                let (after, line) = self.add(s, path, action(17, p, w, -1, dir));
                if !self.fits(&line) {
                    continue;
                }
                if after.reason == 7 {
                    self.finish(&after, &line, false)?;
                    continue;
                }
                for dst in 0..25 {
                    if after.domes & (1 << dst) != 0
                        || after.workers[controller].contains(&dst)
                        || special::blocked(&after, controller, dst)
                    {
                        continue;
                    }
                    let (built, full) = self.add(&after, &line, action(18, p, w, -1, dst));
                    self.finish(&built, &full, false)?;
                }
            }
        }
        Ok(())
    }
    fn dance_moves(&mut self, s: &State, path: &[Action], mask: u8, up: bool) -> GResult {
        self.gen.check()?;
        if !self.fits(path) {
            return Ok(());
        }
        if s.winner >= 0 {
            return self.finish(s, path, up);
        }
        let p = s.player as usize;
        let remaining: Vec<_> = (0..s.counts[p] as usize)
            .filter(|&w| s.workers[p][w] >= 0 && mask & (1 << w) == 0)
            .collect();
        if remaining.is_empty() {
            return self.dance_builds(s, path, 0, up);
        }
        for w in remaining {
            if self.hypnus(w) {
                continue;
            }
            let src = s.workers[p][w];
            for dst in neighbours(s, src) {
                if step(s, w, dst, false, false, false) {
                    let (after, line) = self.add(s, path, action(0, p, w, src, dst));
                    if self.adjacency_ok(&after, w) {
                        self.dance_moves(
                            &after,
                            &line,
                            mask | (1 << w),
                            up || s.heights[dst as usize] > s.heights[src as usize],
                        )?;
                    }
                }
            }
        }
        Ok(())
    }
    fn dance_builds(&mut self, s: &State, path: &[Action], mask: u8, up: bool) -> GResult {
        if (self.probe_up || self.need_up) && !up {
            return Ok(());
        }
        self.gen.check()?;
        if !self.fits(path) {
            return Ok(());
        }
        if self.interrupt(s, path)? {
            return Ok(());
        }
        if s.winner >= 0 || s.reason == 7 {
            return self.finish(s, path, up);
        }
        let p = s.player as usize;
        let remaining: Vec<_> = (0..s.counts[p] as usize)
            .filter(|&w| s.workers[p][w] >= 0 && mask & (1 << w) == 0)
            .collect();
        if remaining.is_empty() {
            return self.finish(s, path, up);
        }
        for w in remaining {
            for a in builds(s, w, false, false) {
                let (after, line) = self.add(s, path, a);
                self.dance_builds(&after, &line, mask | (1 << w), up)?;
            }
        }
        Ok(())
    }
    fn sing(&mut self, s: &State, path: &[Action], mask: u8) -> GResult {
        self.gen.check()?;
        if !self.fits(path) {
            return Ok(());
        }
        if mask != 0 {
            self.finish(s, path, false)?;
        }
        if s.winner >= 0 || s.reason == 7 {
            return Ok(());
        }
        let p = s.player as usize;
        let foe = 1 - p;
        let dir = s.extra.siren[special::card_owner(s, p, 42)];
        if dir < 0 {
            return Ok(());
        }
        let (dx, dy) = special::vector(dir);
        if s.powers[foe] == 39 {
            for w in 0..s.counts[foe] as usize {
                if mask & (1 << w) == 0 && s.workers[foe][w] != -1 {
                    let (after, line) = self.add(s, path, action(16, foe, w, -1, 12));
                    self.sing(&after, &line, mask | (1 << w))?;
                }
            }
            return Ok(());
        }
        for w in 0..s.counts[foe] as usize {
            let src = s.workers[foe][w];
            if src < 0 || mask & (1 << w) != 0 {
                continue;
            }
            let x = src % 5 + dx;
            let y = src / 5 + dy;
            if (0..5).contains(&x) && (0..5).contains(&y) {
                let dst = y * 5 + x;
                if !attempt_occupied(s, dst)
                    && !special::blocked(s, p, src)
                    && !special::blocked(s, p, dst)
                {
                    let (after, line) = self.add(s, path, action(FORCE, foe, w, src, dst));
                    self.sing(&after, &line, mask | (1 << w))?;
                }
            }
        }
        Ok(())
    }
    fn swaps(
        &mut self,
        s: &State,
        path: &[Action],
        mine: u8,
        theirs: u8,
        left: usize,
        up: bool,
    ) -> GResult {
        self.gen.check()?;
        if !self.fits(path) {
            return Ok(());
        }
        if left == 0 || s.winner >= 0 || s.reason == 7 {
            return self.finish(s, path, up);
        }
        let p = s.player as usize;
        let foe = 1 - p;
        for w in 0..s.counts[p] as usize {
            let src = s.workers[p][w];
            if src < 0 || mine & (1 << w) != 0 || special::blocked(s, p, src) {
                continue;
            }
            for v in 0..s.counts[foe] as usize {
                let dst = s.workers[foe][v];
                if dst == -1 || theirs & (1 << v) != 0 || dst >= 0 && special::blocked(s, p, dst) {
                    continue;
                }
                let a = if s.powers[foe] == 39 {
                    action(19, p, w, src, v as i8)
                } else {
                    action(12, p, w, src, dst)
                };
                let (after, line) = self.add(s, path, a);
                self.swaps(
                    &after,
                    &line,
                    mine | (1 << w),
                    theirs | (1 << v),
                    left - 1,
                    up,
                )?;
            }
        }
        Ok(())
    }
    fn token_end(&mut self, s: &State, path: &[Action], w: usize, up: bool) -> GResult {
        let p = s.player as usize;
        let power = s.powers[p];
        if power == 31 {
            for dir in special::DIRECTIONS {
                let (after, line) = self.add(s, path, action(9, p, w, -1, dir));
                self.finish(&after, &line, up)?;
            }
            return Ok(());
        }
        self.finish(s, path, up)?;
        if power == 32 {
            let owner = special::card_owner(s, p, 32);
            if s.extra.whirlpools[owner].iter().any(|&c| c < 0) {
                for dst in 0..25 {
                    if !attempt_occupied(s, dst)
                        && !special::blocked(s, p, dst)
                        && !s.extra.whirlpools.iter().flatten().any(|&c| c == dst)
                    {
                        let (after, line) = self.add(s, path, action(10, p, w, -1, dst));
                        self.finish(&after, &line, up)?;
                    }
                }
            }
        } else if power == 34 {
            let owner = special::card_owner(s, p, 34);
            let src = s.workers[p][w];
            for dst in neighbours(s, src) {
                if !attempt_occupied(s, dst)
                    && !special::blocked(s, p, dst)
                    && s.extra.talus[owner] != dst
                {
                    let (after, line) = self.add(s, path, action(11, p, w, -1, dst));
                    self.finish(&after, &line, up)?;
                }
            }
        } else if power == 41 {
            let foe = 1 - p;
            if s.powers[foe] == 39
                || !s.workers[p].iter().any(|&c| {
                    c >= 0
                        && s.workers[foe]
                            .iter()
                            .any(|&d| d >= 0 && adjacent(c).contains(&d))
                })
            {
                let count = s.workers[p]
                    .iter()
                    .filter(|&&c| c != -1 && (c < 0 || !special::blocked(s, p, c)))
                    .count()
                    .min(
                        s.workers[foe]
                            .iter()
                            .filter(|&&c| c != -1 && (c < 0 || !special::blocked(s, p, c)))
                            .count(),
                    );
                if count > 0 {
                    self.swaps(s, path, 0, 0, count, up)?;
                }
            }
        }
        Ok(())
    }

    fn run(&mut self) -> GResult {
        let s = self.origin;
        if self.active {
            let (after, line) = self.add(
                &s,
                &[],
                Action {
                    kind: ACTIVATE,
                    player: s.player as i8,
                    worker: -1,
                    source: -1,
                    target: -1,
                },
            );
            if s.powers[s.player as usize] == 53 {
                self.corners(&after, &line, 0xff)
            } else {
                self.ordinary(&after, &line)
            }
        } else {
            self.ordinary(&s, &[])
        }
    }
}

pub(super) fn supported(s: &State) -> bool {
    s.powers.iter().all(|&p| p <= 55)
}
fn raw(
    s: &State,
    deadline: Option<Instant>,
    cancelled: &dyn Fn() -> bool,
    limit: usize,
    active: bool,
    need_up: bool,
    need_adonis: bool,
    probe_up: bool,
    probe_adonis: bool,
) -> Result<(Vec<Turn>, bool), Stop> {
    let mut rules = Rules {
        origin: if s.resume.length > 0 {
            s.resume.origin.state()
        } else {
            special::prepare_for_turn(s)
        },
        replay_len: s.resume.length as usize,
        gen: Generator {
            deadline,
            cancelled,
            calls: 0,
            limit,
            turns: Vec::new(),
            produced: 0,
            visitor: None,
        },
        active,
        need_up,
        need_adonis,
        probe_up,
        probe_adonis,
        found: false,
        prefix: s.resume.actions[..s.resume.length as usize].to_vec(),
        filter_next: false,
        next_seen: HashSet::new(),
    };
    match rules.run() {
        Ok(()) | Err(Stop::Enough) => Ok((rules.gen.turns, rules.found)),
        Err(e) => Err(e),
    }
}
pub(super) fn generate(
    s: &State,
    deadline: Option<Instant>,
    cancelled: &dyn Fn() -> bool,
    limit: usize,
) -> Result<Vec<Turn>, Stop> {
    if s.winner >= 0 {
        return Ok(Vec::new());
    }
    if s.extra.event > 0 {
        return special::event_turns(s, limit);
    }
    let effective = if s.resume.length > 0 {
        s.resume.origin.state()
    } else {
        special::prepare_for_turn(s)
    };
    let p = effective.player as usize;
    let need_up = effective.powers[1 - p] == 26
        && (s.heights.iter().any(|&h| h > 0) || effective.powers[p] == 46);
    let adonis = s.adonis[0] == p as i8;
    let normal_up = need_up
        && raw(
            s,
            deadline,
            cancelled,
            usize::MAX,
            false,
            false,
            false,
            true,
            false,
        )?
        .1;
    let normal_adonis = adonis
        && raw(
            s,
            deadline,
            cancelled,
            usize::MAX,
            false,
            normal_up,
            false,
            false,
            true,
        )?
        .1;
    let mut out = raw(
        s,
        deadline,
        cancelled,
        limit,
        false,
        normal_up,
        normal_adonis,
        false,
        false,
    )?
    .0;
    if out.len() >= limit {
        return Ok(out);
    }
    if effective.powers[p] >= 46 && effective.hero_used[p] == 0 {
        let active_up = normal_up
            || need_up
                && matches!(effective.powers[p], 46 | 48 | 49 | 53)
                && raw(
                    s,
                    deadline,
                    cancelled,
                    usize::MAX,
                    true,
                    false,
                    false,
                    true,
                    false,
                )?
                .1;
        let active_adonis = normal_adonis
            || adonis
                && matches!(effective.powers[p], 46 | 48 | 49 | 51 | 53)
                && raw(
                    s,
                    deadline,
                    cancelled,
                    usize::MAX,
                    true,
                    active_up,
                    false,
                    false,
                    true,
                )?
                .1;
        out.extend(
            raw(
                s,
                deadline,
                cancelled,
                limit - out.len(),
                true,
                active_up,
                active_adonis,
                false,
                false,
            )?
            .0,
        );
    }
    Ok(out)
}

fn visit_prefix(
    s: &State,
    deadline: Option<Instant>,
    cancelled: &dyn Fn() -> bool,
    prefix: &[Action],
    filter_next: bool,
    visitor: &mut dyn FnMut(Turn) -> GResult,
) -> GResult {
    if s.winner >= 0 {
        return Ok(());
    }
    if s.extra.event > 0 {
        for turn in special::event_turns(s, usize::MAX)? {
            if turn.actions.starts_with(prefix) {
                visitor(turn)?;
            }
        }
        return Ok(());
    }
    let effective = if s.resume.length > 0 {
        s.resume.origin.state()
    } else {
        special::prepare_for_turn(s)
    };
    let p = effective.player as usize;
    let up = effective.powers[1 - p] == 26
        && (s.heights.iter().any(|&h| h > 0) || effective.powers[p] == 46);
    let adonis = s.adonis[0] == p as i8;
    let normal_up = up
        && raw(
            s,
            deadline,
            cancelled,
            usize::MAX,
            false,
            false,
            false,
            true,
            false,
        )?
        .1;
    let normal_adonis = adonis
        && raw(
            s,
            deadline,
            cancelled,
            usize::MAX,
            false,
            normal_up,
            false,
            false,
            true,
        )?
        .1;
    {
        let mut rules = Rules {
            origin: if s.resume.length > 0 {
                s.resume.origin.state()
            } else {
                special::prepare_for_turn(s)
            },
            replay_len: s.resume.length as usize,
            gen: Generator {
                deadline,
                cancelled,
                calls: 0,
                limit: usize::MAX,
                turns: Vec::new(),
                produced: 0,
                visitor: Some(visitor),
            },
            active: false,
            need_up: normal_up,
            need_adonis: normal_adonis,
            probe_up: false,
            probe_adonis: false,
            found: false,
            prefix: s.resume.actions[..s.resume.length as usize]
                .iter()
                .chain(prefix.iter())
                .copied()
                .collect(),
            filter_next,
            next_seen: HashSet::new(),
        };
        rules.run()?;
    }
    if effective.powers[p] >= 46 && effective.hero_used[p] == 0 {
        let active_up = normal_up
            || up
                && matches!(effective.powers[p], 46 | 48 | 49 | 53)
                && raw(
                    s,
                    deadline,
                    cancelled,
                    usize::MAX,
                    true,
                    false,
                    false,
                    true,
                    false,
                )?
                .1;
        let active_adonis = normal_adonis
            || adonis
                && matches!(effective.powers[p], 46 | 48 | 49 | 51 | 53)
                && raw(
                    s,
                    deadline,
                    cancelled,
                    usize::MAX,
                    true,
                    active_up,
                    false,
                    false,
                    true,
                )?
                .1;
        let mut rules = Rules {
            origin: if s.resume.length > 0 {
                s.resume.origin.state()
            } else {
                special::prepare_for_turn(s)
            },
            replay_len: s.resume.length as usize,
            gen: Generator {
                deadline,
                cancelled,
                calls: 0,
                limit: usize::MAX,
                turns: Vec::new(),
                produced: 0,
                visitor: Some(visitor),
            },
            active: true,
            need_up: active_up,
            need_adonis: active_adonis,
            probe_up: false,
            probe_adonis: false,
            found: false,
            prefix: s.resume.actions[..s.resume.length as usize]
                .iter()
                .chain(prefix.iter())
                .copied()
                .collect(),
            filter_next,
            next_seen: HashSet::new(),
        };
        rules.run()?;
    }
    Ok(())
}

pub(super) fn visit(
    s: &State,
    deadline: Option<Instant>,
    cancelled: &dyn Fn() -> bool,
    visitor: &mut dyn FnMut(Turn) -> GResult,
) -> GResult {
    visit_prefix(s, deadline, cancelled, &[], false, visitor)
}
pub(super) fn next_visit(
    s: &State,
    deadline: Option<Instant>,
    cancelled: &dyn Fn() -> bool,
    prefix: &[Action],
    visitor: &mut dyn FnMut(Turn) -> GResult,
) -> GResult {
    visit_prefix(s, deadline, cancelled, prefix, true, visitor)
}

pub(super) fn prefix_visit(
    s: &State,
    deadline: Option<Instant>,
    cancelled: &dyn Fn() -> bool,
    prefix: &[Action],
    visitor: &mut dyn FnMut(Turn) -> GResult,
) -> GResult {
    visit_prefix(s, deadline, cancelled, prefix, false, visitor)
}

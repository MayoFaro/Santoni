use crate::{generate, Action, State};

pub const PLANE_BYTES: usize = 150;
pub const ACTION_SPACE: usize = 1250;

pub fn is_in_scope(s: &State) -> bool {
    s.powers == [0, 0] && s.counts == [2, 2]
}

/// Chaque tour du sous-jeu sans pouvoir est exactement [déplacement, construction].
/// Réutilise `generate`/`apply`, déjà différentiel-testés (tests/test_native.py) —
/// aucune règle n'est réimplémentée ici.
pub fn legal_children(s: &State) -> Vec<(Action, Action, State)> {
    debug_assert!(is_in_scope(s));
    generate(s, None, &|| false, usize::MAX)
        .unwrap()
        .into_iter()
        .map(|t| {
            assert_eq!(t.actions.len(), 2, "tour hors périmètre sans-pouvoir");
            (t.actions[0], t.actions[1], t.after)
        })
        .collect()
}

/// `None` si la position n'est pas terminale. `Some(1)` si `to_move` a gagné,
/// `Some(-1)` s'il a perdu — y compris par absence de coup légal (pas de pat
/// dans Santorini de base : qui ne peut pas jouer perd).
pub fn terminal_value(s: &State, to_move: u8) -> Option<i32> {
    if s.winner != -1 {
        return Some(if s.winner as u8 == to_move { 1 } else { -1 });
    }
    if s.player == to_move && legal_children(s).is_empty() {
        return Some(-1);
    }
    None
}

pub fn encode_planes(s: &State) -> [u8; PLANE_BYTES] {
    let mut planes = [0u8; PLANE_BYTES];
    let me = s.player as usize;
    let foe = 1 - me;
    for &c in &s.workers[me] {
        if c >= 0 { planes[c as usize] = 1; }
    }
    for &c in &s.workers[foe] {
        if c >= 0 { planes[25 + c as usize] = 1; }
    }
    for cell in 0..25usize {
        let h = s.heights[cell];
        if h == 1 { planes[50 + cell] = 1; }
        if h == 2 { planes[75 + cell] = 1; }
        if h == 3 && s.domes & (1 << cell) == 0 { planes[100 + cell] = 1; }
        if s.domes & (1 << cell) != 0 { planes[125 + cell] = 1; }
    }
    planes
}

/// Index fixe dans [0, ACTION_SPACE) pour un tour (déplacement puis
/// construction/dôme). `worker_slot` est l'indice (0 ou 1) du bâtisseur
/// déplacé dans `s.workers[s.player]`, pas son identité fixe.
pub fn action_index(move_action: &Action, build_action: &Action, worker_slot: usize) -> usize {
    debug_assert!(worker_slot < 2);
    let dest = move_action.target as usize;
    let build = build_action.target as usize;
    worker_slot * 625 + dest * 25 + build
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::State;

    fn base_state() -> State {
        // Position de départ : bâtisseurs en coins opposés, aucune construction.
        let mut s: State = unsafe { std::mem::zeroed() };
        s.workers = [[0, 4, -1, -1], [20, 24, -1, -1]];
        s.counts = [2, 2];
        s.powers = [0, 0];
        s.winner = -1;
        s
    }

    #[test]
    fn rejects_positions_outside_the_no_power_subgame() {
        let mut s = base_state();
        s.powers = [3, 0];
        assert!(!is_in_scope(&s));
        let mut s2 = base_state();
        s2.counts = [3, 2];
        assert!(!is_in_scope(&s2));
        assert!(is_in_scope(&base_state()));
    }

    #[test]
    fn legal_children_matches_generate_exactly() {
        let s = base_state();
        let via_generate = generate(&s, None, &|| false, usize::MAX).unwrap();
        let via_mcts = legal_children(&s);
        assert_eq!(via_generate.len(), via_mcts.len());
        let mut after_states: Vec<State> = via_generate.iter().map(|t| t.after).collect();
        let mut mcts_after: Vec<State> = via_mcts.iter().map(|(_, _, after)| *after).collect();
        after_states.sort_by_key(|s| format!("{:?}", s));
        mcts_after.sort_by_key(|s| format!("{:?}", s));
        assert_eq!(after_states, mcts_after);
    }
}

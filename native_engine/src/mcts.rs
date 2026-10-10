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
    assert!(is_in_scope(s));
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
    assert!(is_in_scope(s));
    if s.winner != -1 {
        return Some(if s.winner as u8 == to_move { 1 } else { -1 });
    }
    if s.player == to_move && legal_children(s).is_empty() {
        return Some(-1);
    }
    None
}

pub fn encode_planes(s: &State) -> [u8; PLANE_BYTES] {
    assert!(is_in_scope(s));
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

pub trait LeafEvaluator {
    fn evaluate(&mut self, states: &[State]) -> Vec<(Vec<f32>, f32)>;
}

struct Edge {
    mv: Action,
    bld: Action,
    after: State,
    prior: f32,
    visits: u32,
    value_sum: f32,
}

struct Node {
    edges: Vec<Edge>,
}

pub struct Mcts {
    root: State,
    c_puct: f32,
    nodes: std::collections::HashMap<State, Node>,
}

impl Mcts {
    pub fn new(root: State, c_puct: f32) -> Self {
        Self { root, c_puct, nodes: std::collections::HashMap::new() }
    }

    pub fn run(&mut self, simulations: u32, evaluator: &mut dyn LeafEvaluator) {
        assert!(
            terminal_value(&self.root, self.root.player).is_none(),
            "Mcts::run appelé sur une racine déjà terminale"
        );
        // Expansion initiale de la racine : ne compte pas comme une des
        // `simulations`. Sans elle, la toute première itération ne ferait
        // qu'évaluer et peupler les arêtes de la racine sans jamais
        // descendre ni backuper sur l'une d'elles (aucun parent au-dessus de
        // la racine pour recevoir ce backup) — `visit_counts` totaliserait
        // alors `simulations - 1`.
        if !self.nodes.contains_key(&self.root) {
            self.simulate_one(self.root, evaluator);
        }
        for _ in 0..simulations {
            self.simulate_one(self.root, evaluator);
        }
    }

    /// Retourne la valeur de `state` du point de vue du joueur au trait à
    /// cet état (convention négamax : chaque niveau de récursion inverse la
    /// valeur renvoyée par l'enfant avant de l'utiliser).
    fn simulate_one(&mut self, state: State, evaluator: &mut dyn LeafEvaluator) -> f32 {
        let to_move = state.player;
        if let Some(v) = terminal_value(&state, to_move) {
            return v as f32;
        }
        if !self.nodes.contains_key(&state) {
            let children = legal_children(&state);
            let (priors, value) = evaluator.evaluate(&[state]).remove(0);
            let edges = children
                .into_iter()
                .map(|(mv, bld, after)| {
                    let worker_slot = state.workers[to_move as usize]
                        .iter()
                        .position(|&c| c == mv.source)
                        .expect("le coup joué doit déplacer un bâtisseur du joueur au trait");
                    let idx = action_index(&mv, &bld, worker_slot);
                    Edge { mv, bld, after, prior: priors[idx], visits: 0, value_sum: 0.0 }
                })
                .collect();
            self.nodes.insert(state, Node { edges });
            return value;
        }
        let c_puct = self.c_puct;
        let node = self.nodes.get(&state).unwrap();
        let total_visits: u32 = node.edges.iter().map(|e| e.visits).sum::<u32>().max(1);
        let sqrt_total = (total_visits as f32).sqrt();
        let score = |e: &Edge| {
            let q = if e.visits == 0 { 0.0 } else { e.value_sum / e.visits as f32 };
            q + c_puct * e.prior * sqrt_total / (1.0 + e.visits as f32)
        };
        let best = node
            .edges
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| score(a).partial_cmp(&score(b)).unwrap())
            .map(|(i, _)| i)
            .expect("un noeud non terminal expansé a au moins un coup légal");
        let next_state = node.edges[best].after;
        let value = -self.simulate_one(next_state, evaluator);
        let node = self.nodes.get_mut(&state).unwrap();
        node.edges[best].visits += 1;
        node.edges[best].value_sum += value;
        value
    }

    /// Coups légaux à la racine avec leur nombre de visites, pour construire
    /// la cible de politique de l'entraînement.
    pub fn visit_counts(&self) -> Vec<(Action, Action, u32)> {
        self.nodes[&self.root]
            .edges
            .iter()
            .map(|e| (e.mv, e.bld, e.visits))
            .collect()
    }
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

#[cfg(test)]
mod mcts_tree_tests {
    use super::*;

    struct UniformEvaluator;
    impl LeafEvaluator for UniformEvaluator {
        fn evaluate(&mut self, states: &[State]) -> Vec<(Vec<f32>, f32)> {
            states.iter().map(|_| (vec![1.0 / ACTION_SPACE as f32; ACTION_SPACE], 0.0)).collect()
        }
    }

    /// État par défaut valide (pas un `mem::zeroed()` brut : `extra`/`resume`/
    /// `adonis` ont des sentinelles non nulles — cf. `special::Extra::default()`
    /// et `lib.rs` `tests::initial()` — sinon `generate` bascule sur le chemin
    /// `advanced::generate` et peut générer les coups du mauvais joueur).
    fn default_state() -> State {
        State {
            heights: [0; 25],
            domes: 0,
            workers: [[-1, -1, -1, -1], [-1, -1, -1, -1]],
            counts: [2, 2],
            powers: [0, 0],
            player: 0,
            hero_used: [0, 0],
            athena_lock: 0,
            adonis: [-1, -1, -1],
            winner: -1,
            reason: 0,
            extra: crate::special::Extra::default(),
            resume: crate::special::Resume::default(),
        }
    }

    fn state_with_immediate_win() -> State {
        let mut s = default_state();
        s.workers = [[0, 1, -1, -1], [20, 24, -1, -1]];
        s.heights[6] = 2; // adjacent à la case 0, le bâtisseur 0 peut y monter
        s
    }

    #[test]
    fn root_with_one_legal_move_gets_all_simulations_on_that_move() {
        let s = state_with_immediate_win();
        let mut mcts = Mcts::new(s, 1.5);
        let mut eval = UniformEvaluator;
        mcts.run(50, &mut eval);
        let counts = mcts.visit_counts();
        let total: u32 = counts.iter().map(|(_, _, n)| n).sum();
        assert_eq!(total, 50);
    }

    #[test]
    fn terminal_value_reports_loss_for_player_with_no_legal_move() {
        let mut s = default_state();
        // Bâtisseurs du joueur au trait complètement enfermés par des dômes.
        s.workers = [[12, 13, -1, -1], [0, 1, -1, -1]];
        s.domes = !0u32 & !((1 << 12) | (1 << 13));
        s.player = 0;
        assert_eq!(terminal_value(&s, 0), Some(-1));
    }
}

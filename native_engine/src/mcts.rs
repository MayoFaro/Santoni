use crate::{generate, special, Action, State};

pub const PLANE_BYTES: usize = 150;
pub const ACTION_SPACE: usize = 1250;

/// Plateau vide avec toutes les sentinelles neutres attendues par
/// `generate` (`adonis == [-1; 3]`, `extra == Extra::default()`,
/// `resume.length == 0`) plutôt qu'un `std::mem::zeroed()` : ce dernier
/// laisse `adonis`/`extra` à zéro, ce qui fait basculer `generate` sur le
/// chemin `advanced::generate` et peut produire des coups pour le mauvais
/// bâtisseur (ce même piège est déjà documenté sur
/// `mcts_tree_tests::default_state` plus bas dans ce fichier, et évité par
/// `lib.rs::tests::initial`). Défini ici (plutôt que dans `lib.rs`, où les
/// contraintes du plan interdisent toute modification au-delà des deux
/// `pub mod` des Tasks 1/3) grâce à la règle d'orphelin de Rust : `Default`
/// est un trait étranger (`std`) mais `State` est un type local à ce
/// crate, donc `impl Default for State` est permis depuis n'importe quel
/// module du crate — y compris `mcts`, qui a accès au module privé
/// `special` en tant que descendant de la racine du crate qui le déclare.
/// Exposé publiquement pour que du code hors crate — ex.
/// `bin/selfplay_worker.rs`, qui a besoin d'un état de départ valide pour
/// une partie d'auto-jeu — puisse construire un état de départ correct
/// sans avoir accès au module privé `special`.
impl Default for State {
    fn default() -> Self {
        State {
            heights: [0; 25],
            domes: 0,
            workers: [[-1; 4]; 2],
            counts: [2, 2],
            powers: [0, 0],
            player: 0,
            hero_used: [0, 0],
            athena_lock: 0,
            adonis: [-1, -1, -1],
            winner: -1,
            reason: 0,
            extra: special::Extra::default(),
            resume: special::Resume::default(),
        }
    }
}

pub fn is_in_scope(s: &State) -> bool {
    s.powers == [0, 0] && s.counts == [2, 2]
}

/// Chaque tour du sous-jeu sans pouvoir est [déplacement, construction] —
/// sauf lorsque le déplacement monte sur une case de hauteur 3 : la partie
/// se termine immédiatement (`apply` positionne `winner`) et `Generator::
/// builds` (lib.rs) clôt le tour sans jamais émettre d'action de
/// construction (`t.actions.len() == 1`). Pour conserver la forme fixe
/// `(Action, Action, State)` attendue par `Mcts`/`action_index` sans
/// propager un `Option<Action>` dans toute l'API déjà testée des Tasks 1-2,
/// on synthétise une construction factice ciblant la case d'arrivée du
/// déplacement : elle n'est jamais appliquée (l'état `t.after` est déjà
/// terminal) et ne sert qu'à produire un `action_index` défini.
/// Réutilise `generate`/`apply`, déjà différentiel-testés (tests/test_native.py) —
/// aucune règle n'est réimplémentée ici.
pub fn legal_children(s: &State) -> Vec<(Action, Action, State)> {
    assert!(is_in_scope(s));
    generate(s, None, &|| false, usize::MAX)
        .unwrap()
        .into_iter()
        .map(|t| {
            assert!(
                t.actions.len() == 1 || t.actions.len() == 2,
                "tour hors périmètre sans-pouvoir"
            );
            let mv = t.actions[0];
            let bld = if t.actions.len() == 2 {
                t.actions[1]
            } else {
                debug_assert!(t.after.winner >= 0, "tour à une action sans victoire immédiate");
                Action { kind: 1, player: mv.player, worker: mv.worker, source: mv.target, target: mv.target }
            };
            (mv, bld, t.after)
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
    /// Évalue un lot de positions de feuille, renvoyant pour chacune un vecteur
    /// de logits (priors) et une valeur scalaire. **Contrat d'ordre crucial** :
    /// les résultats renvoyés doivent être dans le **même ordre** que les états
    /// d'entrée — le i-ème élément du retour correspond au i-ème état du slice
    /// `states`. Cet ordre est maintenant correctness-critical depuis que `Mcts::run`
    /// batch-évalue plusieurs feuilles en un seul appel (avant restructuration,
    /// `evaluate` n'était jamais appelé qu'avec un seul état, rendant l'ordre
    /// implicite). Les implémentations doivent respecter rigoureusement cet ordre :
    /// toute violation silencieuse sans test détectant le désordre causera des
    /// backups de valeurs sur les mauvais chemins et une destruction silencieuse
    /// de la qualité du jeu. **Contrat de longueur tout aussi strict** : le
    /// `Vec` renvoyé doit avoir exactement une entrée par état d'entrée (même
    /// longueur que `states`) ; `Mcts::run` vérifie cette invariante par un
    /// `assert_eq!` avant de consommer le résultat.
    fn evaluate(&mut self, states: &[State]) -> Vec<(Vec<f32>, f32)>;
}

struct Edge {
    mv: Action,
    bld: Action,
    after: State,
    prior: f32,
    visits: u32,
    value_sum: f32,
    virtual_loss: u32,
}

struct Node {
    edges: Vec<Edge>,
}

/// Résultat d'une passe de sélection (une descente racine→feuille, sans
/// appel réseau) : soit l'état terminal rencontré a une valeur déjà connue
/// (`Terminal`), soit la descente s'est arrêtée sur une feuille jamais vue
/// qui doit être évaluée par le réseau (`Leaf`). Dans les deux cas,
/// `path` porte la séquence (état, index d'arête choisie) traversée depuis
/// la racine, nécessaire pour le backup différé une fois la vraie valeur
/// connue.
enum SelectOutcome {
    Terminal { path: Vec<(State, usize)>, value: f32 },
    Leaf { path: Vec<(State, usize)>, leaf_state: State },
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

    /// Exécute `simulations` simulations MCTS, regroupées en vagues d'au
    /// plus `leaf_batch_size` feuilles évaluées en un seul appel à
    /// `evaluator.evaluate`. `leaf_batch_size == 1` reproduit exactement le
    /// comportement séquentiel d'avant ce correctif (aucune perte virtuelle
    /// n'a d'effet observable avec une seule sélection par vague) — voir
    /// les tests de non-régression du module `mcts_tree_tests`.
    pub fn run(&mut self, simulations: u32, leaf_batch_size: usize, evaluator: &mut dyn LeafEvaluator) {
        assert!(
            terminal_value(&self.root, self.root.player).is_none(),
            "Mcts::run appelé sur une racine déjà terminale"
        );
        assert!(leaf_batch_size >= 1, "leaf_batch_size doit être au moins 1");
        self.ensure_root_expanded(evaluator);
        let mut done = 0u32;
        while done < simulations {
            let wave_size = (simulations - done).min(leaf_batch_size as u32);
            let mut pending: Vec<(Vec<(State, usize)>, State)> = Vec::new();
            for _ in 0..wave_size {
                match self.select_one() {
                    SelectOutcome::Terminal { path, value } => self.backup_path(&path, value),
                    SelectOutcome::Leaf { path, leaf_state } => pending.push((path, leaf_state)),
                }
            }
            if !pending.is_empty() {
                let leaf_states: Vec<State> = pending.iter().map(|(_, s)| *s).collect();
                let results = evaluator.evaluate(&leaf_states);
                assert_eq!(
                    results.len(),
                    pending.len(),
                    "LeafEvaluator::evaluate doit renvoyer un résultat par état en entrée"
                );
                for ((path, leaf_state), (priors, value)) in pending.into_iter().zip(results.into_iter()) {
                    self.expand_leaf(leaf_state, &priors);
                    self.backup_path(&path, value);
                }
            }
            done += wave_size;
        }
    }

    /// Expansion initiale de la racine (évaluation + peuplement des arêtes),
    /// idempotente : ne fait rien si déjà expansée. Extraite de `run` pour
    /// que l'appelant self-play puisse injecter du bruit de Dirichlet
    /// (`mix_root_noise`) entre l'expansion et le reste des simulations ;
    /// l'appeler directement est un no-op si `run` l'a déjà fait.
    pub fn ensure_root_expanded(&mut self, evaluator: &mut dyn LeafEvaluator) {
        if !self.nodes.contains_key(&self.root) {
            let (priors, _value) = evaluator.evaluate(&[self.root]).remove(0);
            self.expand_leaf(self.root, &priors);
        }
    }

    /// Mélange du bruit de Dirichlet dans les priors des arêtes de la
    /// racine — exploration en self-play uniquement, jamais sur le chemin
    /// d'arbitrage (`run_single_move` dans `bin/selfplay_worker.rs`), qui
    /// doit rester le jeu le plus fort possible. Doit être appelée après
    /// `ensure_root_expanded` (les arêtes doivent déjà exister) et avant le
    /// reste des simulations de `run`, sinon le bruit n'affecte jamais la
    /// sélection UCB.
    pub fn mix_root_noise(&mut self, noise: &[f32], epsilon: f32) {
        let node = self
            .nodes
            .get_mut(&self.root)
            .expect("mix_root_noise appelé avant l'expansion de la racine (cf. ensure_root_expanded)");
        assert_eq!(
            node.edges.len(),
            noise.len(),
            "le bruit doit avoir une valeur par coup légal à la racine"
        );
        // `Edge::prior` est déjà une distribution de probabilité à ce stade
        // (normalisée par `expand_leaf`, qui normalise désormais tout nœud,
        // pas seulement la racine) ; `noise` (sortie de `dirichlet_noise`,
        // bin/selfplay_worker.rs) en est une aussi — les mélanger directement
        // a un sens de probabilité correct, sans re-softmaxer.
        for (edge, &n) in node.edges.iter_mut().zip(noise) {
            edge.prior = edge.prior * (1.0 - epsilon) + n * epsilon;
        }
    }

    /// Une passe de sélection : descend depuis la racine via UCB (en tenant
    /// compte de la perte virtuelle posée par les sélections précédentes de
    /// la même vague), sans jamais appeler le réseau. S'arrête sur un état
    /// terminal (valeur déjà connue) ou sur une feuille jamais vue (à
    /// évaluer par lot). Pose une perte virtuelle d'une unité sur chaque
    /// arête traversée ; `backup_path` l'annule.
    fn select_one(&mut self) -> SelectOutcome {
        let mut path: Vec<(State, usize)> = Vec::new();
        let mut state = self.root;
        loop {
            let to_move = state.player;
            if let Some(v) = terminal_value(&state, to_move) {
                return SelectOutcome::Terminal { path, value: v as f32 };
            }
            if !self.nodes.contains_key(&state) {
                return SelectOutcome::Leaf { path, leaf_state: state };
            }
            let c_puct = self.c_puct;
            let node = self.nodes.get_mut(&state).unwrap();
            let total_effective: u32 =
                node.edges.iter().map(|e| e.visits + e.virtual_loss).sum::<u32>().max(1);
            let sqrt_total = (total_effective as f32).sqrt();
            let score = |e: &Edge| {
                let q = if e.visits == 0 { 0.0 } else { e.value_sum / e.visits as f32 };
                q + c_puct * e.prior * sqrt_total / (1.0 + e.visits as f32 + e.virtual_loss as f32)
            };
            let best = node
                .edges
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| score(a).partial_cmp(&score(b)).unwrap())
                .map(|(i, _)| i)
                .expect("un noeud non terminal expansé a au moins un coup légal");
            node.edges[best].virtual_loss += 1;
            let next_state = node.edges[best].after;
            path.push((state, best));
            state = next_state;
        }
    }

    /// Crée le noeud d'une feuille nouvellement évaluée : une arête par
    /// coup légal, avec son prior tiré de `priors` à l'index
    /// `action_index`, normalisé en distribution de probabilité. Ne fait
    /// aucun backup — appelé juste avant `backup_path` par
    /// `run`/`ensure_root_expanded`.
    fn expand_leaf(&mut self, state: State, priors: &[f32]) {
        let to_move = state.player;
        let children = legal_children(&state);
        let mut edges: Vec<Edge> = children
            .into_iter()
            .map(|(mv, bld, after)| {
                let worker_slot = state.workers[to_move as usize]
                    .iter()
                    .position(|&c| c == mv.source)
                    .expect("le coup joué doit déplacer un bâtisseur du joueur au trait");
                let idx = action_index(&mv, &bld, worker_slot);
                Edge { mv, bld, after, prior: priors[idx], visits: 0, value_sum: 0.0, virtual_loss: 0 }
            })
            .collect();
        // `priors` sont les logits bruts renvoyés par `LeafEvaluator::evaluate`
        // (cf. le contrat du trait) — pas une distribution de probabilité : un
        // réseau non entraîné les renvoie centrés sur 0, ~moitié négatifs
        // (mesuré en revue finale). `select_one` (PUCT) utilise `prior` comme
        // P(s,a) dans son terme d'exploration et suppose une vraie
        // distribution de probabilité sur les coups légaux de CE nœud ; sans
        // ce softmax, un prior négatif supprime activement l'exploration d'un
        // coup, et un prior à l'échelle de logits arbitraire peut écraser le
        // terme Q borné [-1,1] (ou l'inverse). Normalisé ici pour tout nœud —
        // racine et non-racine, en self-play et en arbitrage — alors que
        // l'ancienne normalisation (dans `mix_root_noise`) ne touchait que la
        // racine en self-play, laissant chaque autre nœud (et la racine de
        // `run_single_move`, qui n'appelle jamais `mix_root_noise`) tourner
        // sur des logits bruts non normalisés.
        let max = edges.iter().map(|e| e.prior).fold(f32::NEG_INFINITY, f32::max);
        let mut total = 0.0f32;
        for edge in edges.iter_mut() {
            edge.prior = (edge.prior - max).exp();
            total += edge.prior;
        }
        for edge in edges.iter_mut() {
            edge.prior /= total;
        }
        self.nodes.insert(state, Node { edges });
    }

    /// Remonte `path` (de la feuille vers la racine) en annulant la perte
    /// virtuelle posée par `select_one` et en appliquant la vraie valeur,
    /// inversée à chaque niveau (convention négamax — `leaf_value` est la
    /// valeur du point de vue du joueur au trait à la feuille/l'état
    /// terminal, exactement ce que renvoient `LeafEvaluator::evaluate` et
    /// `terminal_value`).
    fn backup_path(&mut self, path: &[(State, usize)], leaf_value: f32) {
        let mut value = -leaf_value;
        for &(state, edge_idx) in path.iter().rev() {
            let node = self.nodes.get_mut(&state).unwrap();
            let edge = &mut node.edges[edge_idx];
            edge.virtual_loss -= 1;
            edge.visits += 1;
            edge.value_sum += value;
            value = -value;
        }
    }

    /// Coups légaux à la racine avec leur nombre de visites et l'état
    /// résultant, pour construire la cible de politique de l'entraînement
    /// et avancer la partie sans recalculer `legal_children` séparément.
    pub fn visit_counts(&self) -> Vec<(Action, Action, State, u32)> {
        self.nodes[&self.root]
            .edges
            .iter()
            .map(|e| (e.mv, e.bld, e.after, e.visits))
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

    /// Couverture dédiée (jusqu'ici seulement incidente via le smoke test
    /// Python) du cas à une seule action de `legal_children` : un
    /// déplacement qui monte à hauteur 3 gagne immédiatement
    /// (`Generator::builds` dans lib.rs clôt le tour sans construction),
    /// et `legal_children` doit synthétiser une construction factice
    /// ciblant la case d'arrivée plutôt que paniquer sur l'assertion de
    /// longueur de tour.
    ///
    /// Utilise `State::default()` (défini plus haut dans ce fichier) plutôt
    /// que le `base_state()` ci-dessus construit par `mem::zeroed()` :
    /// celui-ci laisse `adonis`/`extra`/`resume` à zéro, ce qui fait
    /// basculer `generate` sur le chemin `advanced::generate` et — piège
    /// déjà documenté sur `mcts_tree_tests::default_state` — génère alors
    /// les coups des bâtisseurs du mauvais joueur (vérifié empiriquement :
    /// avec `base_state()`, ce test ne trouvait aucun coup gagnant 0→1, le
    /// générateur n'énumérant que les coups des bâtisseurs 20/24 bien que
    /// `s.player == 0`).
    #[test]
    fn legal_children_synthesizes_a_placeholder_build_for_winning_moves() {
        let mut s = State {
            workers: [[0, 4, -1, -1], [20, 24, -1, -1]],
            ..Default::default()
        };
        s.heights[0] = 2;
        s.heights[1] = 3; // adjacente à la case 0, non dômée : coup gagnant pour le bâtisseur 0
        let children = legal_children(&s);
        let found = children
            .iter()
            .find(|(mv, _, _)| mv.source == 0 && mv.target == 1)
            .expect("le coup gagnant 0->1 doit apparaître parmi les coups légaux");
        let (mv, bld, after) = found;
        assert_eq!(after.winner, 0, "monter à hauteur 3 doit déclarer le joueur au trait vainqueur");
        assert_eq!(bld.target, mv.target, "la construction factice cible la case d'arrivée du déplacement");
        assert_eq!(bld.source, mv.target);
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
        mcts.run(50, 1, &mut eval);
        let counts = mcts.visit_counts();
        let total: u32 = counts.iter().map(|(_, _, _, n)| n).sum();
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

    /// Pin de la régression trouvée en revue sur `selfplay_worker.rs` : le
    /// brief du plan appelait `terminal_value(&state, 0)` en dur dans la
    /// boucle de `play_one_game`. La branche « joueur bloqué » de
    /// `terminal_value` ne se déclenche que si `to_move == s.player` ; avec
    /// un `0` codé en dur, elle ne se déclenche jamais quand c'est le
    /// joueur 1 qui est bloqué (`s.player == 1 != 0`), et la position
    /// terminale est manquée (`None` au lieu de `Some(-1)`). C'est ce même
    /// scénario que `terminal_value_reports_loss_for_player_with_no_legal_move`
    /// ci-dessus, mais avec les rôles inversés (joueur 1 bloqué plutôt que
    /// joueur 0) pour isoler précisément le biais de perspective corrigé
    /// par `terminal_value(&state, state.player)` dans
    /// `bin/selfplay_worker.rs`.
    #[test]
    fn terminal_value_reports_loss_for_player_one_stuck_even_though_hardcoded_zero_misses_it() {
        let mut s = default_state();
        // Bâtisseurs du joueur au trait (joueur 1) complètement enfermés
        // par des dômes ; le joueur 0 reste libre.
        s.workers = [[0, 1, -1, -1], [12, 13, -1, -1]];
        s.domes = !0u32 & !((1 << 12) | (1 << 13));
        s.player = 1;
        assert_eq!(
            terminal_value(&s, 0),
            None,
            "le `0` codé en dur de l'ancienne version de play_one_game manque bien cette position terminale"
        );
        assert_eq!(
            terminal_value(&s, s.player),
            Some(-1),
            "interroger depuis la perspective du joueur au trait détecte correctement le blocage"
        );
    }

    #[test]
    fn mix_root_noise_biases_simulations_toward_the_noised_edge() {
        let mut s = default_state();
        s.workers = [[12, 13, -1, -1], [0, 1, -1, -1]];
        let mut mcts = Mcts::new(s, 1.5);
        let mut eval = UniformEvaluator;
        mcts.ensure_root_expanded(&mut eval);
        let n_edges = mcts.visit_counts().len();
        assert!(n_edges > 1, "fixture doit offrir plusieurs coups légaux pour que le test ait un sens");
        let mut noise = vec![0.0f32; n_edges];
        noise[0] = 1.0;
        mcts.mix_root_noise(&noise, 1.0);
        mcts.run(200, 1, &mut eval);
        let counts = mcts.visit_counts();
        let noised_visits = counts[0].3;
        assert!(
            counts.iter().all(|(_, _, _, n)| *n <= noised_visits),
            "l'arête bruitée doit recevoir au moins autant de visites que les autres"
        );
        assert!(
            noised_visits > 200 / n_edges as u32,
            "l'arête bruitée doit recevoir nettement plus que sa part uniforme"
        );
    }

    #[test]
    #[should_panic(expected = "mix_root_noise appelé avant l'expansion de la racine")]
    fn mix_root_noise_panics_before_root_expansion() {
        let s = default_state();
        let mut mcts = Mcts::new(s, 1.5);
        mcts.mix_root_noise(&[1.0], 0.25);
    }

    /// Évaluateur renvoyant des logits bruts non normalisés (n'importe quelle
    /// échelle, y compris négative) plutôt qu'une distribution de
    /// probabilité — c'est exactement ce que renvoie un réseau non entraîné
    /// (mesuré en revue finale : moyenne ~0, écart-type ~0.5, ~moitié de
    /// valeurs négatives), et ce que `SocketEvaluator` (bin/selfplay_worker.rs)
    /// transmet tel quel depuis le serveur d'inférence.
    struct RawLogitEvaluator;
    impl LeafEvaluator for RawLogitEvaluator {
        fn evaluate(&mut self, states: &[State]) -> Vec<(Vec<f32>, f32)> {
            states
                .iter()
                .map(|_| {
                    let mut p = vec![0.0f32; ACTION_SPACE];
                    for (i, v) in p.iter_mut().enumerate() {
                        *v = ((i % 5) as f32 - 2.0) * 0.75; // {-1.5,-0.75,0,0.75,1.5}, pas une distribution
                    }
                    (p, 0.0)
                })
                .collect()
        }
    }

    #[test]
    fn expand_leaf_normalizes_raw_logit_priors_into_a_probability_distribution() {
        // Pin de la correction du bug trouvé lors de cette investigation :
        // `expand_leaf` doit transformer les logits bruts de
        // `RawLogitEvaluator` (certains négatifs, aucune contrainte de somme)
        // en une vraie distribution de probabilité sur CHAQUE nœud qu'il
        // crée — pas seulement la racine (c'était l'ancien comportement,
        // via `mix_root_noise`, qui ne touchait jamais `run_single_move` ni
        // aucun nœud non-racine).
        let mut s = default_state();
        s.workers = [[12, 13, -1, -1], [0, 1, -1, -1]];
        let mut mcts = Mcts::new(s, 1.5);
        let mut eval = RawLogitEvaluator;
        // Assez de simulations pour forcer l'expansion d'au moins un nœud
        // non-racine, pas seulement la racine elle-même.
        mcts.run(50, 1, &mut eval);
        assert!(mcts.nodes.len() > 1, "au moins un nœud non-racine doit avoir été exploré");
        for node in mcts.nodes.values() {
            assert!(!node.edges.is_empty());
            let sum: f32 = node.edges.iter().map(|e| e.prior).sum();
            assert!(
                (sum - 1.0).abs() < 1e-4,
                "les priors d'un nœud doivent sommer à 1.0, trouvé {sum}"
            );
            for e in &node.edges {
                assert!(e.prior >= 0.0, "un prior normalisé ne peut pas être négatif, trouvé {}", e.prior);
            }
        }
    }

    #[test]
    fn mix_root_noise_still_biases_toward_the_noised_edge_with_raw_logit_scale_priors() {
        let mut s = default_state();
        s.workers = [[12, 13, -1, -1], [0, 1, -1, -1]];
        let mut mcts = Mcts::new(s, 1.5);
        let mut eval = RawLogitEvaluator;
        mcts.ensure_root_expanded(&mut eval);
        let n_edges = mcts.visit_counts().len();
        assert!(n_edges > 1, "fixture doit offrir plusieurs coups légaux pour que le test ait un sens");
        let mut noise = vec![0.0f32; n_edges];
        // L'arête 0 reçoit le prior brut le plus négatif (((0 % 5) as f32 -
        // 2.0) * 0.75 == -1.5) : même après normalisation par `expand_leaf`,
        // c'est la plus petite probabilité du lot, donc le mélange à
        // epsilon=0.25 doit encore suffire à la faire ressortir.
        noise[0] = 1.0;
        mcts.mix_root_noise(&noise, 0.25); // DIRICHLET_EPSILON (bin/selfplay_worker.rs)
        mcts.run(400, 1, &mut eval);
        let counts = mcts.visit_counts();
        let noised_visits = counts[0].3;
        let max_other = counts.iter().skip(1).map(|(_, _, _, n)| *n).max().unwrap();
        assert!(
            noised_visits > max_other,
            "avec epsilon=0.25 et un bruit concentré, l'arête bruitée doit recevoir plus de visites \
             que n'importe quelle autre même à l'échelle de logits bruts non normalisés ; \
             trouvé noised={noised_visits}, max autre={max_other}"
        );
    }

    #[test]
    fn visit_counts_returns_the_after_state_matching_legal_children() {
        let mut s = default_state();
        s.workers = [[12, 13, -1, -1], [0, 1, -1, -1]];
        let mut mcts = Mcts::new(s, 1.5);
        let mut eval = UniformEvaluator;
        mcts.ensure_root_expanded(&mut eval);
        let counts = mcts.visit_counts();
        let legal = legal_children(&s);
        assert_eq!(counts.len(), legal.len());
        for ((cmv, cbld, cafter, _), (lmv, lbld, lafter)) in counts.iter().zip(legal.iter()) {
            assert_eq!(cmv, lmv);
            assert_eq!(cbld, lbld);
            assert_eq!(cafter, lafter);
        }
    }

    #[test]
    fn run_produces_exactly_simulations_total_visits_with_batching() {
        let mut s = default_state();
        s.workers = [[12, 13, -1, -1], [0, 1, -1, -1]];
        let mut mcts = Mcts::new(s, 1.5);
        let mut eval = UniformEvaluator;
        mcts.run(37, 8, &mut eval); // 37 n'est pas un multiple de 8
        let counts = mcts.visit_counts();
        let total: u32 = counts.iter().map(|(_, _, _, n)| n).sum();
        assert_eq!(total, 37);
    }

    #[test]
    fn forced_move_with_batching_does_not_panic() {
        let s = state_with_immediate_win(); // une seule arête légale
        let mut mcts = Mcts::new(s, 1.5);
        let mut eval = UniformEvaluator;
        mcts.run(50, 8, &mut eval);
        let counts = mcts.visit_counts();
        let total: u32 = counts.iter().map(|(_, _, _, n)| n).sum();
        assert_eq!(total, 50);
    }

    struct RecordingEvaluator {
        call_sizes: Vec<usize>,
    }
    impl LeafEvaluator for RecordingEvaluator {
        fn evaluate(&mut self, states: &[State]) -> Vec<(Vec<f32>, f32)> {
            self.call_sizes.push(states.len());
            states.iter().map(|_| (vec![1.0 / ACTION_SPACE as f32; ACTION_SPACE], 0.0)).collect()
        }
    }

    #[test]
    fn run_batches_multiple_leaves_into_a_single_evaluate_call() {
        let mut s = default_state();
        s.workers = [[12, 13, -1, -1], [0, 1, -1, -1]];
        let mut mcts = Mcts::new(s, 1.5);
        let mut eval = RecordingEvaluator { call_sizes: Vec::new() };
        mcts.run(64, 8, &mut eval);
        assert!(
            eval.call_sizes.iter().any(|&n| n > 1),
            "au moins un appel à evaluate() doit recevoir plusieurs états à la fois, trouvé {:?}",
            eval.call_sizes
        );
    }
}

//! Worker self-play autonome : joue des parties complètes du sous-jeu
//! sans pouvoir avec un évaluateur local jetable (politique uniforme,
//! valeur nulle) et les écrit sur disque via `mcts_io::write_game_atomically`.
//! Un vrai réseau de neurones remplacera `UniformEvaluator` au Task 7.
//!
//! Aucune dépendance externe (règle du projet, cf. `native_engine/src/lib.rs`) :
//! l'arrêt propre sur SIGINT appelle directement `signal(2, ...)` de la libc
//! via une déclaration `extern "C"` manuelle plutôt qu'une crate comme `ctrlc`.
use santoni_engine::mcts::{legal_children, terminal_value, Mcts, LeafEvaluator, ACTION_SPACE};
use santoni_engine::mcts_io::{write_game_atomically, GameRecord};
use santoni_engine::State;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};

struct UniformEvaluator;
impl LeafEvaluator for UniformEvaluator {
    fn evaluate(&mut self, states: &[State]) -> Vec<(Vec<f32>, f32)> {
        states.iter().map(|_| (vec![1.0 / ACTION_SPACE as f32; ACTION_SPACE], 0.0)).collect()
    }
}

fn parse_arg(args: &[String], flag: &str) -> String {
    let i = args.iter().position(|a| a == flag).expect("argument manquant");
    args[i + 1].clone()
}

/// Position de départ du sous-jeu sans pouvoir : bâtisseurs en coins
/// opposés, aucune construction. `State::default()` (ajouté à `lib.rs`)
/// pose les sentinelles neutres attendues par `generate`
/// (`adonis == [-1; 3]`, `extra`/`resume` par défaut) — un
/// `std::mem::zeroed()` brut, utilisé initialement ici en suivant
/// l'esquisse de la tâche, fait au contraire basculer `generate` sur le
/// chemin `advanced::generate` et panique en jouant le coup du mauvais
/// bâtisseur (confirmé expérimentalement avant ce correctif).
fn initial_state() -> State {
    State {
        workers: [[0, 4, -1, -1], [20, 24, -1, -1]],
        ..Default::default()
    }
}

fn play_one_game(simulations_per_move: u32, evaluator: &mut dyn LeafEvaluator) -> GameRecord {
    let mut state = initial_state();
    let mut moves = Vec::new();
    loop {
        // `terminal_value(&state, 0)` (the plan brief's literal check) misses
        // the case where player 1 is the one with no legal move: its
        // stuck-player branch only fires when `to_move == s.player`, so with
        // a hardcoded `0` it never fires while `s.player == 1`. The position
        // is then silently treated as non-terminal, `Mcts::new`/`Mcts::run`
        // gets called on an already-terminal root, and `Mcts::run`'s own
        // assertion panics — aborting the whole worker and losing every
        // remaining game in the batch. Checking from `state.player`'s own
        // perspective makes both branches (win and stuck) fire correctly in
        // all cases, then flipping the sign when `state.player == 1`
        // converts the result to the fixed player-0 perspective that
        // `GameRecord.outcome` carries throughout the rest of this function.
        if let Some(v) = terminal_value(&state, state.player) {
            let outcome = if state.player == 0 { v } else { -v };
            return GameRecord { moves, outcome: outcome as i8 };
        }
        let mut mcts = Mcts::new(state, 1.5);
        mcts.run(simulations_per_move, evaluator);
        let counts = mcts.visit_counts();
        let total: u32 = counts.iter().map(|(_, _, n)| n).sum();
        let (mv, bld, after) = {
            let legal = legal_children(&state);
            let best = counts.iter().enumerate().max_by_key(|(_, (_, _, n))| *n).unwrap().0;
            legal[best].clone()
        };
        let worker_slot = state.workers[state.player as usize]
            .iter()
            .position(|&c| c == mv.source)
            .unwrap();
        moves.push((santoni_engine::mcts::action_index(&mv, &bld, worker_slot), total));
        state = after;
    }
}

/// Drapeau d'interruption positionné par le gestionnaire SIGINT et vérifié
/// uniquement entre deux parties, jamais pendant l'écriture d'un fichier :
/// l'écriture est déjà atomique par renommage (Task 3), donc le seul
/// contrat à tenir ici est « ne plus démarrer de nouvelle partie après le
/// signal, laisser ce qui est déjà commité tel quel ». Un `AtomicBool`
/// statique immuable évite tout `static mut` (et son lint `static_mut_refs`) :
/// le gestionnaire ne touche qu'à ce seul mot atomique, donc il n'y a pas de
/// risque de donnée partagée non synchronisée malgré l'exécution hors du
/// flot normal du programme.
static INTERRUPTED: AtomicBool = AtomicBool::new(false);

extern "C" fn handle_sigint(_signum: i32) {
    INTERRUPTED.store(true, Ordering::SeqCst);
}

// Déclaration directe de `signal(2)` de la libc : aucune crate externe
// (`ctrlc` ou autre) n'est autorisée dans ce projet. `#[link(name = "c")]`
// est explicite plutôt que supposé implicite, pour que `ldd` fasse
// apparaître `libc.so` de façon garantie quelle que soit la plateforme.
#[link(name = "c")]
extern "C" {
    #[link_name = "signal"]
    fn libc_signal(signum: i32, handler: usize) -> usize;
}

fn install_sigint_handler() {
    const SIGINT: i32 = 2;
    unsafe {
        libc_signal(SIGINT, handle_sigint as *const () as usize);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let games: u32 = parse_arg(&args, "--games").parse().unwrap();
    let out_dir = PathBuf::from(parse_arg(&args, "--out-dir"));
    let simulations_per_move: u32 = parse_arg(&args, "--simulations-per-move").parse().unwrap();

    install_sigint_handler();

    let mut evaluator = UniformEvaluator;
    for i in 0..games {
        if INTERRUPTED.load(Ordering::SeqCst) {
            break;
        }
        let record = play_one_game(simulations_per_move, &mut evaluator);
        let path = out_dir.join(format!("game-{:04}.json", i));
        write_game_atomically(&path, &record).expect("écriture de partie impossible");
    }
}

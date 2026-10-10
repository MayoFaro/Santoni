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
use santoni_engine::{Action, State};
use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

struct UniformEvaluator;
impl LeafEvaluator for UniformEvaluator {
    fn evaluate(&mut self, states: &[State]) -> Vec<(Vec<f32>, f32)> {
        states.iter().map(|_| (vec![1.0 / ACTION_SPACE as f32; ACTION_SPACE], 0.0)).collect()
    }
}

/// Évaluateur qui délègue chaque lot de feuilles au serveur d'inférence
/// Python (Task 7) via le protocole binaire du Task 6
/// (`santoni_az/inference_protocol.py`) : requête = `u32` LE (nombre
/// d'états) puis `count * PLANE_BYTES` octets (un encodage de plans par
/// état, `santoni_engine::mcts::encode_planes`) ; réponse = `u32` LE
/// (nombre d'items) puis, pour chaque item, `ACTION_SPACE` `f32` LE
/// (logits de politique) suivis d'un `f32` LE (valeur). Tous les offsets
/// ci-dessous sont calculés de la même façon que côté Python
/// (`inference_protocol.pack_request`/`unpack_response`) : get this wrong
/// and every policy/value the self-play worker sees is silently corrupted.
///
/// Le serveur d'inférence (Task 7, `InferenceServer._client_loop` /
/// `_drain_and_process_once`) traite une seule requête par connexion puis
/// ferme la socket cliente (`conn.close()` après l'envoi de la réponse) —
/// comportement confirmé par `tests/test_az_inference_server.py`, qui ouvre
/// une nouvelle `socket.socket()` à chaque requête plutôt que de réutiliser
/// une connexion. On se connecte donc à nouveau à chaque appel à
/// `evaluate` (une connexion persistante ouverte une seule fois dans
/// `connect` se voit couper par le serveur après le premier aller-retour,
/// et toute écriture suivante échoue avec `BrokenPipe` — confirmé en
/// pratique avant ce correctif) ; on ne garde donc que le chemin de la
/// socket, pas un `UnixStream` déjà ouvert.
struct SocketEvaluator {
    path: String,
}

impl SocketEvaluator {
    // Ne se connecte pas ici : chaque appel à `evaluate` ouvre sa propre
    // connexion (le serveur étant un aller-retour par connexion, cf. plus
    // haut). Une connexion de validation ouverte puis aussitôt abandonnée
    // ici serait reçue par le serveur comme une connexion sans aucune
    // donnée avant fermeture -- gérée proprement par le vrai serveur Python
    // (`InferenceServer._client_loop` ferme simplement si `len(header) < 4`)
    // mais inutile : la première vraie connexion dans `evaluate` échoue déjà
    // bruyamment (`.expect(...)`) si le chemin est invalide.
    fn connect(path: &str) -> Self {
        Self { path: path.to_string() }
    }
}

impl LeafEvaluator for SocketEvaluator {
    fn evaluate(&mut self, states: &[State]) -> Vec<(Vec<f32>, f32)> {
        let mut stream = UnixStream::connect(&self.path).expect("connexion au serveur d'inférence impossible");
        let count = states.len() as u32;
        let mut request = count.to_le_bytes().to_vec();
        for s in states {
            request.extend_from_slice(&santoni_engine::mcts::encode_planes(s));
        }
        stream.write_all(&request).expect("écriture socket impossible");
        let mut header = [0u8; 4];
        stream.read_exact(&mut header).expect("lecture entête réponse impossible");
        let n = u32::from_le_bytes(header) as usize;
        let item_size = ACTION_SPACE * 4 + 4;
        let mut body = vec![0u8; n * item_size];
        stream.read_exact(&mut body).expect("lecture corps réponse impossible");
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let base = i * item_size;
            let mut policy = Vec::with_capacity(ACTION_SPACE);
            for j in 0..ACTION_SPACE {
                let o = base + j * 4;
                policy.push(f32::from_le_bytes([body[o], body[o + 1], body[o + 2], body[o + 3]]));
            }
            let vo = base + ACTION_SPACE * 4;
            let value = f32::from_le_bytes([body[vo], body[vo + 1], body[vo + 2], body[vo + 3]]);
            result.push((policy, value));
        }
        result
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

/// Mode « un seul coup » : arbitre une position isolée quelconque du
/// sous-jeu sans pouvoir (fournie par une campagne Python, Task 12) plutôt
/// que de jouer des parties complètes depuis l'ouverture comme le mode
/// self-play continu ci-dessous. Lit `size_of::<State>()` octets bruts
/// depuis `input_state` — mêmes octets que ceux produits par
/// `santorini.native.encode(position)` côté Python, puisque `State`/
/// `Action` (lib.rs) et `CState`/`CAction` (santorini/native.py) décrivent
/// la même disposition mémoire `#[repr(C)]`, déjà garantie identique par
/// `santoni_state_size()` (vérifié par `native.py::library()` à chaque
/// chargement de la bibliothèque).
///
/// Format de sortie : un `u32` petit-boutiste (nombre d'actions du tour),
/// puis ce nombre d'`Action` brutes, dans le même ordre que celui attendu
/// par `CAction.from_buffer_copy` côté Python. Le préfixe est indispensable
/// parce qu'un tour n'a **pas** toujours deux actions : monter sur une case
/// de hauteur 3 gagne la partie immédiatement, le générateur clôt alors le
/// tour sans construction (`t.actions.len() == 1`, cf. le commentaire de
/// `mcts::legal_children`). `legal_children` synthétise une construction
/// factice pour garder une forme fixe en interne, mais l'écrire ici ferait
/// rejeter par `validate_turn` (côté Python) chaque tour gagnant : la
/// correspondance avec le vrai tour légal doit être exacte. Le préfixe
/// `u32` LE est le même encodage de compteur que le protocole d'inférence
/// (`santoni_az/inference_protocol.py::pack_request`), donc symétrique à
/// lire avec `struct.unpack('<I', ...)`.
///
/// `is_in_scope` rejette (via `assert!`) toute position hors du sous-jeu
/// sans pouvoir, exactement comme `legal_children`/`terminal_value` dans
/// `mcts.rs` : ce mode ne sait arbitrer que les positions que `Mcts` sait
/// déjà traiter, pas question de deviner un comportement sur le reste du
/// moteur.
fn run_single_move(input_state: &PathBuf, output_move: &PathBuf, seconds: f64, evaluator: &mut dyn LeafEvaluator) {
    let bytes = std::fs::read(input_state).expect("lecture de l'état d'entrée impossible");
    assert_eq!(bytes.len(), std::mem::size_of::<State>(), "taille d'état incompatible");
    // `read_unaligned` et non `read` : `bytes` est un `Vec<u8>` issu de
    // `std::fs::read`, dont l'allocation n'offre aucune garantie
    // d'alignement pour un `State` (align_of::<State>() > 1). En pratique
    // l'allocateur sur-aligne presque toujours, mais une lecture alignée
    // sur un pointeur non garanti aligné est un comportement indéfini —
    // `read_unaligned` est correct par construction et de coût identique
    // ici (une seule lecture par processus).
    let state: State = unsafe { std::ptr::read_unaligned(bytes.as_ptr() as *const State) };
    assert!(santoni_engine::mcts::is_in_scope(&state), "position hors périmètre sans-pouvoir");

    let deadline = Instant::now() + Duration::from_secs_f64(seconds);
    let mut mcts = Mcts::new(state, 1.5);
    const BATCH: u32 = 16;
    loop {
        mcts.run(BATCH, evaluator);
        if Instant::now() >= deadline {
            break;
        }
    }
    let counts = mcts.visit_counts();
    let legal = legal_children(&state);
    let best = counts.iter().enumerate().max_by_key(|(_, (_, _, n))| *n).unwrap().0;
    let (mv, bld, after) = legal[best];

    // Un tour gagnant n'a qu'une action. Dans le sous-jeu sans pouvoir,
    // `apply` (lib.rs) ne positionne `winner` que sur un déplacement montant
    // à hauteur 3 — la seconde cause possible (descente de Pan) exige
    // `powers[p] == 9`, exclu par `is_in_scope` ci-dessus. Donc
    // `after.winner >= 0` est exactement équivalent à « le tour légal réel
    // n'avait qu'une action », et la construction portée par
    // `legal_children` dans ce cas est la factice qu'il synthétise.
    // L'assertion le vérifie plutôt que de le supposer : si la convention de
    // `mcts::legal_children` changeait, on échouerait bruyamment ici au lieu
    // d'écrire silencieusement un tour que `validate_turn` rejetterait.
    let winning = after.winner >= 0;
    if winning {
        assert!(
            bld.source == mv.target && bld.target == mv.target,
            "tour gagnant sans construction factice : convention de legal_children modifiée"
        );
    }
    let actions: &[Action] = if winning { &[mv] } else { &[mv, bld] };

    let mut out = (actions.len() as u32).to_le_bytes().to_vec();
    for action in actions {
        out.extend_from_slice(unsafe {
            std::slice::from_raw_parts(action as *const Action as *const u8, std::mem::size_of::<Action>())
        });
    }
    let tmp = output_move.with_extension("bin.tmp");
    std::fs::write(&tmp, &out).expect("écriture du coup impossible");
    std::fs::rename(&tmp, output_move).expect("renommage du coup impossible");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--single-move") {
        let input_state = PathBuf::from(parse_arg(&args, "--input-state"));
        let output_move = PathBuf::from(parse_arg(&args, "--output-move"));
        let seconds: f64 = parse_arg(&args, "--seconds").parse().unwrap();
        let socket_path = args.iter().position(|a| a == "--inference-socket").map(|i| args[i + 1].clone());
        let mut evaluator: Box<dyn LeafEvaluator> = match socket_path {
            Some(path) => Box::new(SocketEvaluator::connect(&path)),
            None => Box::new(UniformEvaluator),
        };
        run_single_move(&input_state, &output_move, seconds, evaluator.as_mut());
        return;
    }
    let games: u32 = parse_arg(&args, "--games").parse().unwrap();
    let out_dir = PathBuf::from(parse_arg(&args, "--out-dir"));
    let simulations_per_move: u32 = parse_arg(&args, "--simulations-per-move").parse().unwrap();

    install_sigint_handler();

    let socket_path = args.iter().position(|a| a == "--inference-socket").map(|i| args[i + 1].clone());
    let mut evaluator: Box<dyn LeafEvaluator> = match socket_path {
        Some(path) => Box::new(SocketEvaluator::connect(&path)),
        None => Box::new(UniformEvaluator),
    };
    for i in 0..games {
        if INTERRUPTED.load(Ordering::SeqCst) {
            break;
        }
        let record = play_one_game(simulations_per_move, evaluator.as_mut());
        let path = out_dir.join(format!("game-{:04}.json", i));
        write_game_atomically(&path, &record).expect("écriture de partie impossible");
    }
}

#[cfg(test)]
mod socket_evaluator_tests {
    // `Mcts::run` (native_engine/src/mcts.rs) only ever calls
    // `evaluator.evaluate(&[state])` with a single-element slice today, so
    // the end-to-end Python test (tests/test_selfplay_end_to_end.py) never
    // actually drives `SocketEvaluator` with more than one state per
    // request. This test exercises that multi-item path directly: a fake
    // in-process server hand-encodes a 3-item response with distinct,
    // non-symmetric values per item (so a wrong offset shows up as a
    // mismatch, not an accidental match against uniform/repeated data), and
    // we check every float lands back at the right (policy, value) slot in
    // the right item.
    use super::*;
    use std::os::unix::net::UnixListener;

    #[test]
    fn evaluate_decodes_a_multi_item_response_at_the_right_offsets() {
        let dir = std::env::temp_dir().join(format!("selfplay_worker_socket_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("création du répertoire temporaire");
        let socket_path = dir.join("fake_infer.sock");
        let _ = std::fs::remove_file(&socket_path);
        let listener = UnixListener::bind(&socket_path).expect("bind socket factice");
        let socket_path_str = socket_path.to_str().unwrap().to_string();

        let items: Vec<(Vec<f32>, f32)> = (0..3)
            .map(|i| {
                let policy: Vec<f32> = (0..ACTION_SPACE).map(|j| (i * 10_000 + j) as f32 * 0.001).collect();
                (policy, i as f32 + 0.5)
            })
            .collect();

        let server_items = items.clone();
        let server = std::thread::spawn(move || {
            let (mut conn, _) = listener.accept().expect("accept socket factice");
            let mut header = [0u8; 4];
            conn.read_exact(&mut header).expect("lecture entête requête (test)");
            let n = u32::from_le_bytes(header) as usize;
            // Drain exactly the request body the real SocketEvaluator sends
            // (n * PLANE_BYTES), mirroring what the real Python server reads,
            // so the client's write_all cannot block on an unread buffer.
            let mut body = vec![0u8; n * 150];
            conn.read_exact(&mut body).expect("lecture corps requête (test)");

            let mut response = (server_items.len() as u32).to_le_bytes().to_vec();
            for (policy, value) in &server_items {
                for p in policy {
                    response.extend_from_slice(&p.to_le_bytes());
                }
                response.extend_from_slice(&value.to_le_bytes());
            }
            conn.write_all(&response).expect("écriture réponse factice");
        });

        let mut evaluator = SocketEvaluator::connect(&socket_path_str);
        let states = vec![initial_state(), initial_state(), initial_state()];
        let decoded = evaluator.evaluate(&states);
        server.join().expect("jointure thread serveur factice");

        assert_eq!(decoded.len(), items.len());
        for (i, ((got_policy, got_value), (want_policy, want_value))) in decoded.iter().zip(items.iter()).enumerate() {
            assert_eq!(got_policy.len(), ACTION_SPACE, "item {i}: taille de politique inattendue");
            assert_eq!(got_policy, want_policy, "item {i}: politique mal décodée (offset incorrect)");
            assert_eq!(got_value, want_value, "item {i}: valeur mal décodée (offset incorrect)");
        }

        let _ = std::fs::remove_file(&socket_path);
    }
}

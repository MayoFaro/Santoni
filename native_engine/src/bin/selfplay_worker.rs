//! Worker self-play autonome : joue des parties complètes du sous-jeu
//! sans pouvoir avec un évaluateur local jetable (politique uniforme,
//! valeur nulle) et les écrit sur disque via `mcts_io::write_game_atomically`.
//! Un vrai réseau de neurones remplacera `UniformEvaluator` au Task 7.
//!
//! Aucune dépendance externe (règle du projet, cf. `native_engine/src/lib.rs`) :
//! l'arrêt propre sur SIGINT appelle directement `signal(2, ...)` de la libc
//! via une déclaration `extern "C"` manuelle plutôt qu'une crate comme `ctrlc`.
use santoni_engine::mcts::{terminal_value, Mcts, LeafEvaluator, ACTION_SPACE};
use santoni_engine::mcts_io::{write_game_atomically, GameRecord, PositionRecord};
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
/// Python via le protocole binaire (`santoni_az/inference_protocol.py`) :
/// requête = `u32` LE (nombre d'états) puis `count * PLANE_BYTES` octets (un
/// encodage de plans par état, `santoni_engine::mcts::encode_planes`) ;
/// réponse = `u32` LE (nombre d'items) puis, pour chaque item, `ACTION_SPACE`
/// `f32` LE (logits de politique) suivis d'un `f32` LE (valeur). Tous les
/// offsets ci-dessous sont calculés de la même façon que côté Python
/// (`inference_protocol.pack_request`/`unpack_response`) : get this wrong and
/// every policy/value the self-play worker sees is silently corrupted.
///
/// La connexion est persistante : une seule `UnixStream` est ouverte au
/// premier appel à `evaluate` puis réutilisée pour tous les appels suivants,
/// y compris entre deux parties successives (le serveur d'inférence,
/// `InferenceServer._client_loop`, boucle lui aussi sur la même connexion
/// plutôt que de la fermer après un seul échange). Ceci élimine le
/// connect()/accept()/spawn-de-thread/close() répété à chaque feuille
/// évaluée (mesuré : ~241 connexions en 5s à un seul coup → 1 ; environ
/// 10% d'un cœur de CPU système récupéré sous forte cadence). Mesuré aussi :
/// ceci ne change quasiment pas le débit (46.9 → 47.1 req/s à
/// `flush_interval_s` par défaut) — le coût dominant par aller-retour n'est
/// pas la connexion (~21ms mesurés, pas 100-200ms) mais l'intervalle de
/// scrutation `InferenceServer.flush_interval_s` lui-même (7.3x de gain
/// mesuré en l'abaissant seul, sans ce correctif). En
/// cas d'erreur d'E/S sur la connexion existante (pipe brisé, serveur
/// redémarré), `evaluate` referme le flux mort, en rouvre un nouveau, et
/// rejoue la même requête une fois ; un second échec se propage normalement
/// (panic), ce cas restant aussi rare qu'avant ce correctif.
struct SocketEvaluator {
    path: String,
    stream: Option<UnixStream>,
}

impl SocketEvaluator {
    // Ne se connecte pas ici : la connexion s'ouvre au premier appel à
    // `evaluate` (dans `ensure_connected`), pas à la construction — un
    // worker qui démarre avant que le serveur d'inférence soit prêt doit
    // pouvoir être instancié sans échouer immédiatement ; l'échec, s'il a
    // lieu, se manifeste seulement au premier vrai besoin de la connexion.
    fn connect(path: &str) -> Self {
        Self { path: path.to_string(), stream: None }
    }

    fn ensure_connected(&mut self) -> &mut UnixStream {
        if self.stream.is_none() {
            let stream = UnixStream::connect(&self.path).expect("connexion au serveur d'inférence impossible");
            self.stream = Some(stream);
        }
        self.stream.as_mut().unwrap()
    }

    fn try_evaluate(&mut self, states: &[State]) -> std::io::Result<Vec<(Vec<f32>, f32)>> {
        let stream = self.ensure_connected();
        let count = states.len() as u32;
        let mut request = count.to_le_bytes().to_vec();
        for s in states {
            request.extend_from_slice(&santoni_engine::mcts::encode_planes(s));
        }
        stream.write_all(&request)?;
        let mut header = [0u8; 4];
        stream.read_exact(&mut header)?;
        let n = u32::from_le_bytes(header) as usize;
        assert_eq!(
            n, states.len(),
            "le serveur d'inférence a renvoyé un nombre de résultats différent du nombre d'états envoyés"
        );
        let item_size = ACTION_SPACE * 4 + 4;
        let mut body = vec![0u8; n * item_size];
        stream.read_exact(&mut body)?;
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
        Ok(result)
    }
}

impl LeafEvaluator for SocketEvaluator {
    fn evaluate(&mut self, states: &[State]) -> Vec<(Vec<f32>, f32)> {
        match self.try_evaluate(states) {
            Ok(result) => result,
            Err(_) => {
                // Connexion morte (pipe brisé, serveur redémarré) : on
                // referme et rouvre une nouvelle connexion, puis on rejoue
                // la même requête une fois.
                self.stream = None;
                self.try_evaluate(states).expect(
                    "connexion au serveur d'inférence impossible après une tentative de reconnexion",
                )
            }
        }
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

/// Bruit de Dirichlet au niveau de la racine + température d'ouverture :
/// sans ceci, le MCTS self-play est entièrement déterministe (mêmes
/// priors, même UCB, même coup choisi à chaque partie) — vérifié en revue
/// finale du chantier initial : 4 parties self-play produisaient des
/// fichiers strictement identiques. `DIRICHLET_ALPHA`/`DIRICHLET_EPSILON`
/// reprennent les valeurs standard AlphaZero pour un facteur de branchement
/// modéré (échecs : alpha=0.3, epsilon=0.25). `TEMPERATURE_PLIES` limite
/// l'échantillonnage proportionnel aux visites aux tout premiers coups de
/// la partie ; au-delà, exploitation pure (argmax), comme dans AlphaZero
/// une fois l'ouverture passée.
const DIRICHLET_ALPHA: f32 = 0.3;
const DIRICHLET_EPSILON: f32 = 0.25;
const TEMPERATURE_PLIES: usize = 6;

/// Nombre de feuilles regroupées en un seul appel réseau par vague de
/// sélection MCTS (`Mcts::run`, native_engine/src/mcts.rs) — réduit le
/// nombre d'allers-retours avec le serveur d'inférence par coup joué (à
/// `simulations_per_move = 32`, 4 vagues au lieu de 32 appels séparés).
/// S'applique au self-play (`play_one_game`) et à l'arbitrage
/// (`run_single_move`) : le regroupement de feuilles est une technique de
/// performance pure, pas une source de bruit, donc les deux chemins en
/// bénéficient également — contrairement au bruit de Dirichlet et à la
/// température d'ouverture ci-dessus, qui restent self-play uniquement.
const LEAF_BATCH_SIZE: usize = 8;

/// Générateur xorshift64* minimal (aucune dépendance externe — règle du
/// projet). N'a pas besoin d'être cryptographiquement sûr : seul le service
/// rendu est de casser le déterminisme entre parties/process, pas une
/// garantie de sécurité.
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 0x9e3779b97f4a7c15 } else { seed } }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Flottant uniforme dans [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Normale standard (Box-Muller), utilisée par `gamma_sample`.
    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

/// Graine distincte par partie et par processus : PID (distinct entre
/// workers concurrents, change au redémarrage — même raisonnement que le
/// nom de fichier des parties, cf. `main` plus bas), index de partie, et
/// horloge système, combinés par hachage plutôt que concaténés
/// arithmétiquement pour éviter toute structure visible entre graines
/// consécutives.
fn seed_for_game(game_index: u32) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    std::process::id().hash(&mut hasher);
    game_index.hash(&mut hasher);
    std::time::SystemTime::now().hash(&mut hasher);
    hasher.finish()
}

/// Échantillonne une loi Gamma(shape, 1) par la méthode de Marsaglia-Tsang
/// (2000), avec le correctif usuel pour `shape < 1` (Y = X · U^(1/shape),
/// X ~ Gamma(shape+1, 1)).
fn gamma_sample(rng: &mut Xorshift64, shape: f64) -> f64 {
    if shape < 1.0 {
        let u = rng.next_f64().max(1e-12);
        return gamma_sample(rng, shape + 1.0) * u.powf(1.0 / shape);
    }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let (x, v) = loop {
            let x = rng.next_normal();
            let v = 1.0 + c * x;
            if v > 0.0 {
                break (x, v);
            }
        };
        let v3 = v * v * v;
        let u = rng.next_f64();
        if u < 1.0 - 0.0331 * x.powi(4) || u.ln() < 0.5 * x * x + d * (1.0 - v3 + v3.ln()) {
            return d * v3;
        }
    }
}

/// Échantillon de Dirichlet(alpha, ..., alpha) sur `n` catégories : `n`
/// tirages Gamma(alpha, 1) indépendants, normalisés à somme 1. `n == 1` est
/// un cas valide (ply à un seul coup légal) : le seul tirage se normalise
/// trivialement à 1.0.
fn dirichlet_noise(rng: &mut Xorshift64, n: usize, alpha: f32) -> Vec<f32> {
    let samples: Vec<f64> = (0..n).map(|_| gamma_sample(rng, alpha as f64).max(1e-9)).collect();
    let total: f64 = samples.iter().sum();
    samples.iter().map(|s| (s / total) as f32).collect()
}

/// Tirage pondéré par `weights` (ex. les nombres de visites MCTS) : plus un
/// coup a été visité, plus il a de chances d'être choisi. `weights` doit
/// être non vide avec une somme strictement positive.
fn sample_weighted(rng: &mut Xorshift64, weights: &[u32]) -> usize {
    let total: u64 = weights.iter().map(|&w| w as u64).sum();
    assert!(total > 0, "sample_weighted appelé avec des poids tous nuls");
    let mut r = (rng.next_f64() * total as f64) as u64;
    for (i, &w) in weights.iter().enumerate() {
        if r < w as u64 {
            return i;
        }
        r -= w as u64;
    }
    weights.len() - 1
}

fn play_one_game(simulations_per_move: u32, evaluator: &mut dyn LeafEvaluator, rng_seed: u64) -> GameRecord {
    let mut state = initial_state();
    let mut positions = Vec::new();
    let mut rng = Xorshift64::new(rng_seed);
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
            return GameRecord { positions, outcome: outcome as i8 };
        }
        let planes = santoni_engine::mcts::encode_planes(&state);
        let mut mcts = Mcts::new(state, 1.5);
        mcts.ensure_root_expanded(evaluator);
        let n_edges = mcts.visit_counts().len();
        let noise = dirichlet_noise(&mut rng, n_edges, DIRICHLET_ALPHA);
        mcts.mix_root_noise(&noise, DIRICHLET_EPSILON);
        mcts.run(simulations_per_move, LEAF_BATCH_SIZE, evaluator);
        // `visit_counts()` porte directement l'état résultant (`after`) de
        // chaque coup légal : pas de second appel à `legal_children` dont
        // l'ordre devrait, par convention non vérifiée par le compilateur,
        // coïncider avec celui de `visit_counts` (cf. revue finale — un tel
        // désalignement silencieux mélangerait la distribution de visites
        // d'une position avec l'état résultant d'une autre).
        let counts = mcts.visit_counts();
        let visits: Vec<(usize, u32)> = counts
            .iter()
            .map(|(mv, bld, _after, n)| {
                let worker_slot = state.workers[state.player as usize]
                    .iter()
                    .position(|&c| c == mv.source)
                    .unwrap();
                (santoni_engine::mcts::action_index(mv, bld, worker_slot), *n)
            })
            .collect();
        let chosen = if positions.len() < TEMPERATURE_PLIES {
            let weights: Vec<u32> = counts.iter().map(|(_, _, _, n)| *n).collect();
            sample_weighted(&mut rng, &weights)
        } else {
            counts.iter().enumerate().max_by_key(|(_, (_, _, _, n))| *n).unwrap().0
        };
        let after = counts[chosen].2;
        positions.push(PositionRecord { planes, visits });
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
        mcts.run(BATCH, LEAF_BATCH_SIZE, evaluator);
        if Instant::now() >= deadline {
            break;
        }
    }
    let counts = mcts.visit_counts();
    let best = counts.iter().enumerate().max_by_key(|(_, (_, _, _, n))| *n).unwrap().0;
    let (mv, bld, after, _) = counts[best];

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
        let record = play_one_game(simulations_per_move, evaluator.as_mut(), seed_for_game(i));
        // Le PID dans le nom de fichier rend le nom globalement unique, sans
        // aucune coordination entre workers. `game-{:04}.json` (compteur
        // local repartant de 0 à chaque processus) collisionnait dans deux
        // situations très réelles :
        //   - `worker_count > 1` (orchestrateur) : N processus écrivent tous
        //     `game-0000.json` dans le même `run_dir/games` ;
        //   - reprise après arrêt : le nouveau processus repart de 0 et
        //     écrase les parties de la session précédente.
        // Dans les deux cas la perte est silencieuse *et* invisible pour
        // l'entraîneur : `Trainer._consumed_games` est indexé par nom de
        // fichier et contient déjà l'ancien nom, donc le nouveau contenu au
        // même chemin n'est jamais relu. Un PID est distinct pour chaque
        // processus vivant et change au redémarrage : il ferme les deux
        // trous d'un coup. Le préfixe `game-` et le suffixe `.json` sont
        // conservés pour que tous les `glob("game-*.json")` existants
        // (trainer, orchestrateur, tests) continuent de matcher.
        let path = out_dir.join(format!("game-{}-{:06}.json", std::process::id(), i));
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

    #[test]
    fn evaluate_reuses_the_same_connection_across_calls() {
        let dir = std::env::temp_dir().join(format!("selfplay_worker_reuse_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("création du répertoire temporaire");
        let socket_path = dir.join("fake_infer_reuse.sock");
        let _ = std::fs::remove_file(&socket_path);
        let listener = UnixListener::bind(&socket_path).expect("bind socket factice");
        let socket_path_str = socket_path.to_str().unwrap().to_string();

        let accept_count = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let accept_count_server = accept_count.clone();
        let server = std::thread::spawn(move || {
            // Une seule connexion attendue : si evaluate() en rouvrait une à
            // chaque appel, ce unique accept() ne verrait jamais la seconde
            // requête (détecté ci-dessous via un timeout plutôt qu'un
            // blocage permanent du thread client).
            let (mut conn, _) = listener.accept().expect("accept socket factice");
            accept_count_server.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            for _ in 0..2 {
                let mut header = [0u8; 4];
                conn.read_exact(&mut header).expect("lecture entête requête (test)");
                let n = u32::from_le_bytes(header) as usize;
                let mut body = vec![0u8; n * 150];
                conn.read_exact(&mut body).expect("lecture corps requête (test)");
                let mut response = 1u32.to_le_bytes().to_vec();
                response.extend_from_slice(&vec![0u8; ACTION_SPACE * 4]);
                response.extend_from_slice(&0.0f32.to_le_bytes());
                conn.write_all(&response).expect("écriture réponse factice");
            }
        });

        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let mut evaluator = SocketEvaluator::connect(&socket_path_str);
            let states = vec![initial_state()];
            evaluator.evaluate(&states);
            evaluator.evaluate(&states);
            let _ = tx.send(());
        });
        rx.recv_timeout(std::time::Duration::from_secs(5)).expect(
            "evaluate() n'a pas abouti à temps : la connexion a probablement été rouverte au lieu d'être réutilisée",
        );

        server.join().expect("jointure thread serveur factice");
        assert_eq!(accept_count.load(std::sync::atomic::Ordering::SeqCst), 1);
        let _ = std::fs::remove_file(&socket_path);
    }

    #[test]
    fn evaluate_reconnects_transparently_after_a_server_restart() {
        let dir = std::env::temp_dir().join(format!("selfplay_worker_reconnect_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("création du répertoire temporaire");
        let socket_path = dir.join("fake_infer_reconnect.sock");
        let _ = std::fs::remove_file(&socket_path);
        let listener = UnixListener::bind(&socket_path).expect("bind socket factice");
        let socket_path_str = socket_path.to_str().unwrap().to_string();

        let server = std::thread::spawn(move || {
            for i in 0..2u32 {
                let (mut conn, _) = listener.accept().expect("accept socket factice");
                let mut header = [0u8; 4];
                conn.read_exact(&mut header).expect("lecture entête requête (test)");
                let n = u32::from_le_bytes(header) as usize;
                let mut body = vec![0u8; n * 150];
                conn.read_exact(&mut body).expect("lecture corps requête (test)");
                let value = i as f32 + 0.5;
                let mut response = 1u32.to_le_bytes().to_vec();
                response.extend_from_slice(&vec![0u8; ACTION_SPACE * 4]);
                response.extend_from_slice(&value.to_le_bytes());
                conn.write_all(&response).expect("écriture réponse factice");
                // Fermeture explicite : simule un serveur qui redémarre
                // entre deux requêtes du même worker, sans compter sur le
                // moment exact où `Drop` fermerait le descripteur.
                conn.shutdown(std::net::Shutdown::Both).ok();
            }
        });

        let mut evaluator = SocketEvaluator::connect(&socket_path_str);
        let states = vec![initial_state()];
        let first = evaluator.evaluate(&states);
        assert_eq!(first[0].1, 0.5);
        let second = evaluator.evaluate(&states);
        assert_eq!(second[0].1, 1.5);
        server.join().expect("jointure thread serveur factice");
        let _ = std::fs::remove_file(&socket_path);
    }
}

#[cfg(test)]
mod exploration_tests {
    use super::*;

    #[test]
    fn same_seed_reproduces_the_same_game() {
        let mut eval_a = UniformEvaluator;
        let mut eval_b = UniformEvaluator;
        let a = play_one_game(4, &mut eval_a, 12345);
        let b = play_one_game(4, &mut eval_b, 12345);
        assert_eq!(a.outcome, b.outcome);
        assert_eq!(a.positions.len(), b.positions.len());
        for (pa, pb) in a.positions.iter().zip(&b.positions) {
            assert_eq!(pa.planes, pb.planes);
            assert_eq!(pa.visits, pb.visits);
        }
    }

    #[test]
    fn different_seeds_produce_different_games() {
        let mut eval_a = UniformEvaluator;
        let mut eval_b = UniformEvaluator;
        let a = play_one_game(4, &mut eval_a, 1);
        let b = play_one_game(4, &mut eval_b, 2);
        assert!(!a.positions.is_empty());
        assert!(!b.positions.is_empty());
        // Avant ce correctif, 4 parties self-play produisaient des fichiers
        // strictement identiques (revue finale du chantier initial) quelle
        // que soit la graine, puisqu'aucune n'était même consultée.
        let diverges = a.positions.len() != b.positions.len()
            || a.positions.iter().zip(&b.positions).any(|(pa, pb)| pa.planes != pb.planes);
        assert!(diverges, "deux graines distinctes doivent produire des trajectoires différentes");
    }

    #[test]
    fn dirichlet_noise_sums_to_one_and_has_one_entry_per_category() {
        let mut rng = Xorshift64::new(42);
        let noise = dirichlet_noise(&mut rng, 5, DIRICHLET_ALPHA);
        assert_eq!(noise.len(), 5);
        let total: f32 = noise.iter().sum();
        assert!((total - 1.0).abs() < 1e-4, "la somme du bruit de Dirichlet doit être 1, trouvé {total}");
    }

    #[test]
    fn dirichlet_noise_handles_a_single_category_without_panicking() {
        let mut rng = Xorshift64::new(7);
        let noise = dirichlet_noise(&mut rng, 1, DIRICHLET_ALPHA);
        assert_eq!(noise, vec![1.0]);
    }

    #[test]
    fn sample_weighted_always_returns_the_only_index_when_forced() {
        let mut rng = Xorshift64::new(99);
        for _ in 0..20 {
            assert_eq!(sample_weighted(&mut rng, &[7]), 0);
        }
    }
}

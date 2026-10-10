"""200 parties appariées, candidat (MCTS + réseau, sous-jeu sans pouvoir) contre
le moteur de référence figé, budget de temps égal par tour. Même corpus et même
méthode de bootstrap par groupe de placement que
`tools/hermes_canonicalisation_campaign.py` ; reprend directement `digest`,
`verify_reference`, `atomic_json`, `bootstrap` de ce script (voir ses lignes
45-61 et 213-221 pour la provenance).

Corpus : `experiments/no-power-corpus-20261010/campaign.json`, produit par
`tools/generate_no_power_corpus.py` (graine 20261010). Volontairement **pas**
le corpus historique `experiments/selfplay-100-5s-20261005/campaign.json` des
campagnes précédentes : celui-ci est à 88 % composé de jobs avec pouvoirs
(12 de ses 100 jobs seulement ont `powers == [0, 0]`), alors que le mode
`--single-move` du worker candidat `assert!(is_in_scope(&state))` et panique
sur toute position hors du sous-jeu sans pouvoir — une campagne de 200
parties sur ce corpus s'effondrait donc en parties `'error'` et l'IC
bootstrap (conditionné à `len(finished) == 200`) n'était jamais calculé. Le
nouveau corpus garde schéma, motif de variantes et granularité de groupe de
placement identiques (25 ouvertures × 2 orientations × 2 premiers joueurs,
groupes `(matchup, opening)` de 4 jobs), avec `powers == [0, 0]` partout.
`main()` revérifie cet invariant job par job dès le chargement, pour qu'un
corpus inadapté échoue en une seconde et non au bout de plusieurs heures.

Décodage des actions : le worker (`--single-move`, Task 11) écrit un `u32`
petit-boutiste donnant le nombre d'actions du tour, puis ce nombre de
`CAction` bruts. Ce nombre vaut 1 ou 2 : monter sur une case de hauteur 3
gagne immédiatement et le tour se clôt sans construction. Lire deux actions
en dur (ce que faisait la première version) faisait rejeter par
`validate_turn` — qui exige une correspondance exacte avec le tour légal
réel — *tout* tour gagnant du candidat, transformant chaque victoire
candidate en partie `'error'`. Le `kind` de chaque action est décodé via
`ACTION_KINDS`, exactement comme `santorini.native.decode_turn`, au lieu de
supposer `"move"`/`"build"` en dur : sur une position de milieu/fin de partie
quelconque du corpus, la deuxième action peut être un dôme (`kind == 2`).

Moteur de référence : le côté référence passe par `freeze_reference`, qui
copie le `.so` *actuellement* compilé dans `native_engine/target/release`
(le moteur alpha-bêta de production réel, canonicalisation comprise) dans
le dossier de sortie de la campagne, avec un manifeste sha256 — exactement
le motif `freeze` de `tools/hermes_canonicalisation_campaign.py`, appliqué
ici au côté référence plutôt qu'au côté candidat. Sans ça, une recompilation
en cours de campagne (même sans rapport avec ce script) changerait
silencieusement ce que « la référence » désigne pour les parties déjà
jouées. L'archive historique `experiments/references/b698745` que
`verify_reference` vérifie est une vérification de provenance différente et
sans rapport : cette archive précède les travaux de génération progressive
et de canonisation déjà fusionnés dans `main`, ce n'est donc pas le moteur
de production actuel — elle n'est jamais chargée pour la recherche.
"""
from __future__ import annotations

import argparse
import ctypes as C
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import multiprocessing
import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'experiments/references/b698745'
CORPUS = ROOT / 'experiments/no-power-corpus-20261010/campaign.json'
WORKER_BINARY = ROOT / 'native_engine' / 'target' / 'release' / 'selfplay_worker'

sys.path.insert(0, str(ROOT))
from santorini.engine import Action, Position, terminal_position, validate_setup, validate_turn  # noqa: E402
from santorini.native import ACTION_KINDS, encode, CAction, native_search  # noqa: E402
from santorini import native  # noqa: E402
from santoni_az.network import PolicyValueNet  # noqa: E402
from santoni_az.inference_server import InferenceServer  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_reference():
    meta = json.loads((REFERENCE / 'reference.json').read_text())
    for name, key in [('libsantoni_engine.so', 'binary_sha256'), ('source.tar.gz', 'archive_sha256')]:
        if digest(REFERENCE / name) != meta[key]:
            raise ValueError(f'Reference integrity mismatch: {name}')
    return meta


def freeze_reference(directory):
    """Snapshot the engine currently built at `native.library_path()` (today's
    live `native_engine/target/release` build, i.e. the actual production
    alpha-beta engine, canonicalisation work included) as the reference under
    test for this run, so a rebuild mid-campaign -- even one triggered by
    something unrelated to this script -- can never silently change what
    'the reference' means partway through. Deliberately NOT the historical
    `experiments/references/b698745` snapshot `verify_reference` checks above:
    that archive predates the progressive-generation + canonicalisation work
    already merged into main, so it is not today's production engine; it is
    kept here only as an unrelated provenance check on a different archive.
    Structurally the same as `tools/hermes_canonicalisation_campaign.py::freeze`
    (copy the live `.so` into the campaign directory + sha256 manifest; skip
    the copy if the manifest already exists so re-running the same `--out`
    directory never silently re-snapshots)."""
    target = directory / 'reference'
    target.mkdir(exist_ok=True)
    original = native.library_path()
    binary = target / Path(original).name
    manifest = target / 'manifest.json'
    if not manifest.exists():
        shutil.copy2(original, binary)
        atomic_json(manifest, {'engine_sha256': digest(binary)})
    frozen = json.loads(manifest.read_text())
    if digest(binary) != frozen['engine_sha256']:
        raise ValueError('Reference snapshot integrity mismatch')
    return binary, frozen


def load_checkpoint_net(path):
    data = torch.load(path, weights_only=False)
    net = PolicyValueNet()
    net.load_state_dict(data['model_state'])
    net.eval()
    return net


def freeze_checkpoint(directory, checkpoint_path):
    """Snapshot the checkpoint file currently at `checkpoint_path` into the
    campaign directory, exactly the same rationale as `freeze_reference`
    above applied to the candidate side: pin precisely what is being
    measured, so a checkpoint file changing on disk mid-campaign (e.g.
    training continuing to write to the same path, or simply pointing a
    resumed run at a different file) can never silently change what 'the
    candidate' means partway through, and is caught loudly instead."""
    target = directory / 'candidate'
    target.mkdir(exist_ok=True)
    binary = target / 'checkpoint.pt'
    manifest = target / 'manifest.json'
    if not manifest.exists():
        shutil.copy2(checkpoint_path, binary)
        generation = torch.load(binary, weights_only=False)['generation']
        atomic_json(manifest, {'checkpoint_sha256': digest(binary), 'generation': generation})
    frozen = json.loads(manifest.read_text())
    if digest(binary) != frozen['checkpoint_sha256']:
        raise ValueError('Candidate checkpoint snapshot integrity mismatch')
    return binary, frozen


def set_engine(lib):
    if os.environ.get('SANTONI_ENGINE_LIB') != str(lib):
        os.environ['SANTONI_ENGINE_LIB'] = str(lib)
        native._library = None
    native.library()


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def bootstrap(groups):
    rng = random.Random(20261010)
    values = list(groups.values())
    samples = []
    for _ in range(5000):
        selected = [rng.choice(values) for _ in values]
        samples.append(sum(sum(v) for v in selected) / sum(len(v) for v in selected))
    samples.sort()
    return [samples[125], samples[4874]]


def candidate_move(position, seconds, socket_path, scratch_dir):
    state = encode(position)
    input_path = scratch_dir / 'state.bin'
    output_path = scratch_dir / 'move.bin'
    input_path.write_bytes(bytes(state))
    if output_path.exists():
        output_path.unlink()
    cmd = [str(WORKER_BINARY), '--single-move', '--input-state', str(input_path),
           '--output-move', str(output_path), '--seconds', str(seconds),
           '--inference-socket', str(socket_path)]
    started = time.monotonic()
    result = subprocess.run(cmd, timeout=seconds + 10, capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError(f'Candidate worker failed: {result.stderr}')
    raw = output_path.read_bytes()
    # Préfixe `u32` LE = nombre d'actions réellement écrites (1 ou 2). Un tour
    # gagnant (montée à hauteur 3) n'a qu'une action : le générateur clôt le
    # tour sans construction. `validate_turn` exigeant la correspondance
    # exacte avec le tour légal, en soumettre deux transformait chaque
    # victoire du candidat en partie 'error'. Décodage générique du `kind`
    # (voir santorini.native.decode_turn) : ne jamais supposer "move"/"build"
    # en dur, la deuxième action peut être un dôme (kind 2) sur une position
    # arbitraire du corpus.
    (count,) = struct.unpack_from('<I', raw, 0)
    expected = 4 + count * C.sizeof(CAction)
    if count not in (1, 2) or len(raw) != expected:
        raise ValueError(f'Candidate move file malformed: {len(raw)} bytes for {count} actions')
    actions = []
    for index in range(count):
        item = CAction.from_buffer_copy(raw, 4 + index * C.sizeof(CAction))
        actions.append(Action(ACTION_KINDS[item.kind], item.player, item.worker,
                              item.source, item.target))
    turn = validate_turn(position, tuple(actions))
    return turn, time.monotonic() - started


def reference_move(position, seconds, reference_binary):
    # Route through the frozen snapshot, not whatever happens to be built at
    # native.library_path() right now -- mirrors how the candidate side is
    # already pinned to a specific worker binary rather than "whatever's
    # currently built".
    set_engine(reference_binary)
    analysis = native_search(position, seconds)
    if analysis.turn is None:
        raise ValueError('Reference engine returned no legal turn')
    return analysis.turn, analysis.elapsed


def play(job, directory, seconds, socket_path, reference_binary):
    initial = job['initial']
    path = Path(directory) / f"game-{job['number']:04d}.json"
    position = validate_setup(initial['workers'], initial['powers'], initial['first'])
    data = {'job': job, 'status': 'running', 'decisions': []}
    atomic_json(path, data)
    try:
        with tempfile.TemporaryDirectory() as scratch:
            scratch_dir = Path(scratch)
            for ply in range(150):
                checked = terminal_position(position)
                if checked.winner is not None:
                    position = checked
                    break
                is_candidate = position.player == job['variant_player']
                if is_candidate:
                    turn, elapsed = candidate_move(position, seconds, socket_path, scratch_dir)
                else:
                    turn, elapsed = reference_move(position, seconds, reference_binary)
                data['decisions'].append({'ply': ply + 1, 'player': position.player,
                                           'backend': 'candidate' if is_candidate else 'reference',
                                           'elapsed': elapsed})
                position = turn.after
                atomic_json(path, {**data, 'position': position.to_dict()})
        data['status'] = 'finished' if position.winner is not None else 'inconclusive'
    except Exception as exc:
        data.update(status='error', error=f'{type(exc).__name__}: {exc}')
    data['winner'] = position.winner
    atomic_json(path, data)
    return {'number': job['number'], 'status': data['status'],
            'variant_won': position.winner == job['variant_player'] if position.winner is not None else None}


def compute_pending_jobs(directory, jobs):
    """Coups restants à jouer : ignore les parties déjà terminées, met en
    quarantaine (`attempts/`, suffixe horodaté) les parties incomplètes
    d'une exécution précédente plutôt que de les écraser silencieusement sur
    place. Même motif que `tools/hermes_canonicalisation_campaign.py`
    (lignes ~367-379)."""
    pending = []
    for job in jobs:
        path = directory / f"game-{job['number']:04d}.json"
        if path.exists():
            old = json.loads(path.read_text())
            if old['job'] != job:
                raise ValueError(f'Job mismatch: {path}')
            if old['status'] == 'finished':
                continue
            attempts = directory / 'attempts'
            attempts.mkdir(exist_ok=True)
            path.rename(attempts / f'{path.stem}-{time.time_ns()}.json')
        pending.append(job)
    return pending


def check_manifest_matches(manifest_path, config):
    """Écrit `campaign.json` au premier lancement sur ce dossier ; aux
    lancements suivants, vérifie que la configuration n'a pas changé plutôt
    que de mélanger silencieusement deux campagnes incompatibles (graine,
    checkpoint, moteur de référence...) dans le même répertoire."""
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if {k: v for k, v in previous.items() if k != 'started_at'} != config:
            raise ValueError('Campaign provenance/parameters differ; choose a new output folder')
    else:
        atomic_json(manifest_path, {'started_at': time.time(), **config})
        shutil.copy2(Path(__file__), manifest_path.parent / 'runner.py')


def summary(directory, corpus):
    games = [json.loads(p.read_text()) for p in sorted(directory.glob('game-*.json'))]
    finished = [g for g in games if g['status'] == 'finished']
    groups = defaultdict(list)
    for game in finished:
        job = game['job']
        won = int(game['winner'] == job['variant_player'])
        key = (tuple(job['initial']['matchup']), job['initial']['opening'])
        groups[key].append(won)
    wins = sum(sum(v) for v in groups.values())
    result = {'expected': 200, 'completed': len(finished), 'candidate_wins': wins,
               'reference_wins': len(finished) - wins,
               'win_rate': wins / len(finished) if finished else None,
               'errors': [{'game': g['job']['number'], 'error': g.get('error')} for g in games
                          if g['status'] in ('error', 'inconclusive')]}
    if len(finished) == 200:
        result['placement_bootstrap_95'] = bootstrap(groups)
        result['placement_groups'] = len(groups)
    atomic_json(directory / 'summary.json', result)
    return result


def report(directory, result):
    lines = ['# Candidat MCTS + réseau contre le moteur de référence figé (sous-jeu sans pouvoir)', '',
             f"Parties terminées : {result['completed']}/{result['expected']}.", '']
    if result['win_rate'] is not None:
        lines.append(f"Victoires du candidat : {result['candidate_wins']} / {result['completed']} "
                      f"({result['win_rate']*100:.1f} %).")
    if 'placement_bootstrap_95' in result:
        lines.append(f"Intervalle bootstrap à 95 % par groupe de placement "
                      f"({result['placement_groups']} groupes) : {result['placement_bootstrap_95']}.")
    lines += ['', f"Erreurs ou parties non conclues : {len(result['errors'])}."]
    (directory / 'rapport.md').write_text('\n'.join(lines) + '\n')


def check_corpus_is_no_power(jobs_all):
    """Échoue immédiatement si le corpus chargé contient le moindre job avec
    pouvoir.

    Le candidat ne sait jouer que le sous-jeu sans pouvoir : `run_single_move`
    (native_engine/src/bin/selfplay_worker.rs) `assert!(is_in_scope(&state))`
    et panique sinon. Sans ce contrôle, un corpus inadapté — par exemple le
    corpus historique `selfplay-100-5s-20261005`, à 88 % avec pouvoirs — ne se
    manifeste que partie par partie, chacune marquée 'error' après avoir
    dépensé son budget de temps : plusieurs heures de calcul pour un
    `summary.json` sans IC bootstrap (celui-ci exige `len(finished) == 200`).
    Mieux vaut échouer dans la première seconde."""
    offenders = [job.get('number', index + 1) for index, job in enumerate(jobs_all)
                 if list(job.get('powers', [])) != [0, 0]]
    if offenders:
        raise ValueError(
            f'Corpus {CORPUS} is not a no-power corpus: {len(offenders)} of {len(jobs_all)} '
            f'jobs have powers != [0, 0] (first offenders: {offenders[:10]}). The candidate '
            'engine only plays the no-power subgame; regenerate a corpus with '
            'tools/generate_no_power_corpus.py.')
    if not jobs_all:
        raise ValueError(f'Corpus {CORPUS} contains no job')


def physical_cpus():
    cores = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        base = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key = ((base / 'physical_package_id').read_text(), (base / 'core_id').read_text())
        cores.setdefault(key, cpu)
    return list(cores.values())


def initialize(cpus):
    os.nice(10)
    identity = multiprocessing.current_process()._identity[0]
    os.sched_setaffinity(0, {cpus[(identity - 1) % len(cpus)]})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--seconds', type=float, default=30.0)
    parser.add_argument('--out', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--limit', type=int, default=None,
                         help='Only run the first N scheduled games (pilot runs); '
                              'same convention as tools/hermes_canonicalisation_campaign.py --limit.')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    verify_reference()
    directory = Path(args.out)
    directory.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (directory / '.campaign.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    reference_binary, reference_manifest = freeze_reference(directory)
    print(f"REFERENCE frozen: {reference_binary} sha256={reference_manifest['engine_sha256']}", flush=True)
    candidate_binary, candidate_manifest = freeze_checkpoint(directory, Path(args.checkpoint))
    net = load_checkpoint_net(candidate_binary)
    socket_path = str(directory / 'infer.sock')
    server = InferenceServer(socket_path, net)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(0.1)
    print(f"CANDIDATE checkpoint generation={candidate_manifest['generation']} "
          f"sha256={candidate_manifest['checkpoint_sha256']}", flush=True)
    corpus = json.loads(CORPUS.read_text())
    jobs_all = corpus['jobs']
    check_corpus_is_no_power(jobs_all)
    jobs = [{'number': i + 1, 'variant_player': player, 'initial': job}
            for i, (job, player) in enumerate((job, player) for job in jobs_all for player in (0, 1))]
    if args.limit:
        jobs = jobs[:args.limit]
    config = {'protocol': 'mcts-az-vs-reference-no-power-30s-v1', 'seconds': args.seconds, 'jobs': jobs,
              'seed': corpus['seed'], 'reference_engine_sha256': reference_manifest['engine_sha256'],
              'candidate_checkpoint_sha256': candidate_manifest['checkpoint_sha256'],
              'candidate_checkpoint_generation': candidate_manifest['generation'],
              'corpus_sha256': digest(CORPUS), 'runner_sha256': digest(Path(__file__))}
    try:
        check_manifest_matches(directory / 'campaign.json', config)
    except ValueError as exc:
        parser.error(str(exc))
    pending = compute_pending_jobs(directory, jobs)
    cpus = physical_cpus()[:args.workers]
    print(f'START {len(pending)}/{len(jobs)} remaining; {args.seconds}s/move; CPUs {cpus}', flush=True)
    try:
        with ProcessPoolExecutor(max_workers=len(cpus), mp_context=multiprocessing.get_context('spawn'),
                                  initializer=initialize, initargs=(cpus,)) as pool:
            futures = {pool.submit(play, job, str(directory), args.seconds, socket_path, str(reference_binary))
                       for job in pending}
            while futures:
                done, futures = wait(futures, timeout=30, return_when=FIRST_COMPLETED)
                for future in done:
                    print('GAME ' + json.dumps(future.result()), flush=True)
                result = summary(directory, corpus)
                report(directory, result)
                print(f"PROGRESS {result['completed']}/{len(jobs)}; errors={len(result['errors'])}", flush=True)
    finally:
        server.stop()
    result = summary(directory, corpus)
    report(directory, result)
    print('FINAL ' + json.dumps(result, ensure_ascii=False), flush=True)
    if result['completed'] != len(jobs):
        raise SystemExit('Campaign requires review: incomplete games')


if __name__ == '__main__':
    main()

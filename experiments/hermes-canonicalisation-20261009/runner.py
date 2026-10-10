"""200 paired games, candidate (visit_base + canonical_key) vs frozen `main`
at `b698745`: 30 seconds or completed depth 6, first stop reached.

Adapted from `tools/baseline_depth_campaign.py` and
`tools/options_depth_campaign.py` (worktree `defensive-threats`,
branch `feature/defensive-threats`): same canonical 100-configuration
corpus (`experiments/selfplay-100-5s-20261005/campaign.json`, seed 20261005),
same 30s-or-completed-depth6 protocol, same bootstrap-by-placement-group.
Not a literal copy: this branch changes only `native_engine/src/lib.rs`
(confirmed by `git diff b698745..HEAD --stat`), so there is no Python rules
drift to cross-validate with a second frozen Python package -- the single
current `santorini` package validates turns from both engines. The only
two frozen artefacts needed are the two compiled `.so` binaries.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import replace
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'experiments/references/b698745'
CORPUS = ROOT / 'experiments/selfplay-100-5s-20261005/campaign.json'
SECONDS = 30.0
MAX_DEPTH = 6

sys.path.insert(0, str(ROOT))
from santorini.engine import Action, validate_setup, validate_turn  # noqa: E402
from santorini.storage import Session, action_dict, now  # noqa: E402
from santorini.powers import POWERS  # noqa: E402
from santorini import native  # noqa: E402


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_reference():
    meta = json.loads((REFERENCE / 'reference.json').read_text())
    for name, key in [('libsantoni_engine.so', 'binary_sha256'), ('source.tar.gz', 'archive_sha256')]:
        if digest(REFERENCE / name) != meta[key]:
            raise ValueError(f'Reference integrity mismatch: {name}')
    return meta


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def freeze(directory):
    """Snapshot the engine currently built in native_engine/target/release as
    the candidate under test, so a rebuild mid-campaign can never silently
    change what is being measured."""
    target = directory / 'candidate'
    target.mkdir(exist_ok=True)
    original = native.library_path()
    binary = target / Path(original).name
    manifest = target / 'manifest.json'
    if not manifest.exists():
        shutil.copy2(original, binary)
        atomic_json(manifest, {'engine_sha256': digest(binary),
                                'source_commit': os.environ.get('HERMES_CANDIDATE_COMMIT', 'uncommitted-worktree-head')})
    frozen = json.loads(manifest.read_text())
    if digest(binary) != frozen['engine_sha256']:
        raise ValueError('Candidate snapshot integrity mismatch')
    return binary, frozen


def set_engine(lib):
    if os.environ.get('SANTONI_ENGINE_LIB') != str(lib):
        os.environ['SANTONI_ENGINE_LIB'] = str(lib)
        native._library = None
    native.library()


def capped_search(position, lib, seconds=SECONDS, max_depth=MAX_DEPTH):
    """Same semantics as `tools/baseline_depth_campaign.py::capped_search`
    in the defensive-threats worktree: use the engine's own cancellation
    hook, stop early on a completed depth 6 (treated the same as the
    engine's own early stop on a demonstrated win/loss), never modify
    engine code or ranking."""
    set_engine(lib)
    started = time.monotonic()
    reached = None
    attempted = 0

    def progress(analysis):
        nonlocal reached, attempted
        if reached is not None:
            return
        attempted = max(attempted, analysis.searching_depth, analysis.depth)
        if analysis.complete_depth and analysis.depth >= max_depth:
            reached = analysis

    analysis = native.native_search(
        position, seconds, publish=progress,
        cancelled=lambda: reached is not None or time.monotonic() - started >= seconds)
    if reached is not None:
        if reached.depth != max_depth:
            raise ValueError('Engine skipped the requested completed depth')
        analysis = replace(reached, elapsed=analysis.elapsed, nodes=analysis.nodes,
                            searching_depth=0, stop_reason='depth_limit',
                            status=f'Profondeur {max_depth} entièrement achevée')
    if analysis.depth > max_depth or attempted > max_depth:
        raise ValueError('Engine exceeded the depth limit')
    return analysis, {'attempted_depth': attempted, 'depth_cap_reached': reached is not None,
                       'wall_elapsed': time.monotonic() - started}


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


def scheduled_jobs(configurations):
    # Each of the 100 configurations played with both sides swapped: 200
    # paired games, interleaved so partial progress stays comparable.
    return [{'number': i + 1, 'config': job['number'], 'variant_player': player, 'initial': job}
            for i, (job, player) in enumerate((job, player) for job in configurations for player in (0, 1))]


def search_turn(position, variant, candidate_lib, reference_lib):
    return capped_search(position, candidate_lib if variant else reference_lib)


def play(job, directory, candidate_lib, reference_lib):
    initial = job['initial']
    path = Path(directory) / f"game-{job['number']:04d}.json"
    session = Session(validate_setup(initial['workers'], initial['powers'], initial['first']),
                       settings={'hermes_canonicalisation': True, 'seconds': SECONDS, 'max_depth': MAX_DEPTH,
                                 'variant_player': job['variant_player'], 'config': job['config']})
    started = time.monotonic()
    data = {'job': job, 'status': 'running', 'started_at': now(), 'decisions': [], 'session': session.to_dict()}
    atomic_json(path, data)
    try:
        for ply in range(150):
            before = session.position
            variant = before.player == job['variant_player']
            analysis, limits = search_turn(before, variant, candidate_lib, reference_lib)
            row = {k: getattr(analysis, k) for k in (
                'depth', 'score', 'proven', 'complete_depth', 'elapsed', 'nodes', 'stop_reason')}
            row.update(limits, player=before.player, ply=ply + 1,
                       backend='candidate' if variant else 'original',
                       budget_seconds=SECONDS, max_depth=MAX_DEPTH,
                       variation=[{'actions': [action_dict(a) for a in t.actions], 'after': t.after.to_dict()}
                                  for t in analysis.variation])
            data['decisions'].append(row)
            if analysis.turn is None:
                if analysis.stop_reason == 'terminal' and analysis.score == -100000:
                    session = session.ended(1 - before.player, 'Aucun tour complet légal')
                else:
                    raise ValueError('No completed legal turn within limits')
            else:
                turn = validate_turn(before, analysis.turn.actions)
                if turn.after != analysis.turn.after:
                    raise ValueError('Rules/engine mismatch on the chosen turn')
                session = session.append(turn)
            data.update(session=session.to_dict(), elapsed=time.monotonic() - started)
            atomic_json(path, data)
            if session.result is not None:
                break
        data['status'] = 'finished' if session.result is not None else 'inconclusive'
        if data['status'] == 'inconclusive':
            data['error'] = '150-turn limit reached'
        Session.from_dict(session.to_dict())
        if session.result:
            # A minimax proof says a winning line exists against every
            # reply; it does not compel the eventual opponent to keep
            # playing that line. A different final winner is kept as an
            # audit hint, not a contradiction.
            divergences = []
            for decision in data['decisions']:
                if decision['proven'] and decision['score'] is not None:
                    predicted = decision['player'] if decision['score'] > 0 else 1 - decision['player']
                    if predicted != session.result['winner']:
                        divergences.append({'ply': decision['ply'], 'predicted_winner': predicted,
                                             'final_winner': session.result['winner']})
            if divergences:
                data['proof_outcome_divergences'] = divergences
    except Exception as exc:
        data.update(status='error', error=f'{type(exc).__name__}: {exc}')
    data.update(session=session.to_dict(), elapsed=time.monotonic() - started, finished_at=now())
    atomic_json(path, data)
    return {'number': job['number'], 'config': job['config'], 'status': data['status'],
            'turns': len(session.history), 'winner': session.result['winner'] if session.result else None,
            'variant_won': session.result['winner'] == job['variant_player'] if session.result else None,
            'error': data.get('error')}


def bootstrap(groups):
    rng = random.Random(20261009)
    values = list(groups.values())
    samples = []
    for _ in range(5000):
        selected = [rng.choice(values) for _ in values]
        samples.append(sum(sum(v) for v in selected) / sum(len(v) for v in selected))
    samples.sort()
    return [samples[125], samples[4874]]


def summary(directory):
    games = [json.loads(p.read_text()) for p in sorted(directory.glob('game-*.json'))]
    finished = [g for g in games if g['status'] == 'finished']
    groups = defaultdict(list)
    by_matchup = defaultdict(lambda: {'games': 0, 'candidate_wins': 0})
    for game in finished:
        job = game['job']
        won = game['session']['result']['winner'] == job['variant_player']
        key = (tuple(job['initial']['matchup']), job['initial']['opening'])
        groups[key].append(int(won))
        name = '/'.join(POWERS[p].name for p in job['initial']['powers'])
        by_matchup[name]['games'] += 1
        by_matchup[name]['candidate_wins'] += won
    wins = sum(sum(v) for v in groups.values())
    metrics = {}
    for backend in ('candidate', 'original'):
        decisions = [d for g in games for d in g['decisions'] if d['backend'] == backend]
        metrics[backend] = {
            'decisions': len(decisions),
            'depths': dict(Counter(d['depth'] for d in decisions)),
            'stop_reasons': dict(Counter(d['stop_reason'] for d in decisions)),
            'depth_cap_reached': sum(d['depth_cap_reached'] for d in decisions),
            'mean_seconds': sum(d['elapsed'] for d in decisions) / len(decisions) if decisions else None,
        }
    result = {'updated_at': now(), 'expected': 200, 'completed': len(finished),
              'candidate_wins': wins, 'original_wins': len(finished) - wins,
              'win_rate': wins / len(finished) if finished else None,
              'statuses': dict(Counter(g['status'] for g in games)),
              'by_matchup': dict(by_matchup), 'metrics': metrics,
              'errors': [{'game': g['job']['number'], 'error': g.get('error')} for g in games
                         if g['status'] in ('error', 'inconclusive')],
              'divergences': sum(len(g.get('proof_outcome_divergences', [])) for g in games)}
    if len(finished) == 200:
        result['placement_bootstrap_95'] = bootstrap(groups)
        result['placement_groups'] = len(groups)
    atomic_json(directory / 'summary.json', result)
    return result


def report(directory, result):
    lines = ['# Candidat (génération progressive + canonisation) contre original `b698745` figé',
             '', '30 secondes ou profondeur 6 entièrement achevée, première butée atteinte.', '',
             f"Parties terminées : {result['completed']}/{result['expected']}.", '',
             'Les 100 configurations du corpus canonique '
             '(`experiments/selfplay-100-5s-20261005/campaign.json`, graine 20261005) sont '
             'rejouées camp inversé : 200 parties. Seul `native_engine/src/lib.rs` change entre '
             'le candidat et la référence ; les règles Python sont identiques des deux côtés.', '',
             f"Victoires du candidat : {result['candidate_wins']} / {result['completed']} "
             f"({result['win_rate']*100:.1f} %)." if result['win_rate'] is not None else '', '']
    if 'placement_bootstrap_95' in result:
        lines.append(f"Intervalle bootstrap à 95 %, regroupé par placement "
                      f"({result['placement_groups']} groupes, 5000 tirages) : "
                      f"{result['placement_bootstrap_95']}.")
    lines += ['', '| Confrontation | Parties | Victoires du candidat |',
              '| --- | ---: | ---: |']
    for name, row in result['by_matchup'].items():
        lines.append(f"| {name} | {row['games']} | {row['candidate_wins']} |")
    lines += ['', '| Camp | Décisions | Profondeurs | Arrêts | Butée atteinte | Temps moyen |',
              '| --- | ---: | --- | --- | ---: | ---: |']
    for backend, m in result['metrics'].items():
        mean = f"{m['mean_seconds']:.3f}s" if m['mean_seconds'] is not None else '—'
        lines.append(f"| {backend} | {m['decisions']} | {m['depths']} | {m['stop_reasons']} | "
                      f"{m['depth_cap_reached']} | {mean} |")
    lines += ['', f"Erreurs ou parties non conclues : {len(result['errors'])}.",
              f"Contradictions preuve/vainqueur final : {result['divergences']}.", '',
              'Les 25 groupes de placements corrélés (matchup x ouverture) ne sont pas 100 '
              'placements indépendants ; voir le rapport pour la méthode de bootstrap.', '']
    (directory / 'rapport.md').write_text('\n'.join(lines))


def self_check(directory, candidate_lib, reference_lib):
    p = validate_setup(((6, 8), (16, 18)), (0, 0), 0)
    for label, lib in [('candidate', candidate_lib), ('original', reference_lib)]:
        depth, cap = capped_search(p, lib, 2, 2)
        assert depth.complete_depth and depth.depth == 2 and cap['depth_cap_reached']
        assert depth.stop_reason == 'depth_limit'
        assert validate_turn(p, depth.turn.actions).after == depth.turn.after
        timed, time_limit = capped_search(p, lib, .02, 6)
        assert timed.depth <= 6 and not time_limit['depth_cap_reached']
        print(f'  {label}: depth-cap ok, timeout ok', flush=True)
    atomic_json(directory / 'self-check.json', {'passed': True,
                                                 'candidate_engine_sha256': digest(candidate_lib),
                                                 'reference_engine_sha256': digest(reference_lib)})
    print('SELF-CHECK PASS: candidate + original, depth cap, timeout, legal turn', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--limit', type=int, default=None,
                         help='Only run the first N scheduled games (pilot runs).')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    meta = verify_reference()
    directory = Path(args.output).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (directory / '.campaign.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    candidate_lib, candidate_meta = freeze(directory)
    reference_lib = REFERENCE / 'libsantoni_engine.so'
    if args.self_check:
        self_check(directory, candidate_lib, reference_lib)
        return
    corpus = json.loads(CORPUS.read_text())
    jobs_all = corpus['jobs']
    assert len(jobs_all) == 100 and [j['number'] for j in jobs_all] == list(range(1, 101))
    jobs = scheduled_jobs(jobs_all)
    if args.limit:
        jobs = jobs[:args.limit]
    cpus = physical_cpus()[:args.workers]
    cpu_model = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines()
                       if line.startswith('model name')), 'unknown')
    config = {'protocol': 'hermes-canonicalisation-candidate-vs-b698745-30s-depth6-v1',
              'seconds': SECONDS, 'max_depth': MAX_DEPTH, 'jobs': jobs, 'seed': corpus['seed'],
              'reference_engine_commit': meta['engine_commit'], 'reference_engine_sha256': meta['binary_sha256'],
              'candidate_engine_sha256': candidate_meta['engine_sha256'],
              'candidate_source_commit': candidate_meta['source_commit'],
              'corpus_sha256': digest(CORPUS), 'runner_sha256': digest(Path(__file__)),
              'workers': len(cpus), 'cpus': cpus, 'cpu_model': cpu_model,
              'platform': platform.platform(), 'python': platform.python_version(),
              'limitations': ['25 correlated placement groups (matchup x opening), not 100 independent placements.',
                               'Engine\'s own early stop on exact proof retained.',
                               'Wall-clock search outcomes may vary with hardware and load.']}
    manifest = directory / 'campaign.json'
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if {k: v for k, v in previous.items() if k != 'started_at'} != config:
            parser.error('Campaign provenance/parameters differ; choose a new output folder')
    else:
        atomic_json(manifest, {'started_at': now(), **config})
        shutil.copy2(Path(__file__), directory / 'runner.py')
    initial = summary(directory)
    report(directory, initial)
    if args.prepare_only:
        print(f'PREPARED: {len(jobs)} games, 30 seconds or completed depth 6, frozen reference verified', flush=True)
        return
    if shutil.disk_usage(directory).free < 200 * 1024 * 1024:
        parser.error('At least 200 MiB free disk space required')
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
    print(f'START {len(pending)}/{len(jobs)} remaining; 30s or completed depth6; CPUs {cpus}', flush=True)
    with ProcessPoolExecutor(max_workers=len(cpus), mp_context=multiprocessing.get_context('spawn'),
                              initializer=initialize, initargs=(cpus,)) as pool:
        futures = {pool.submit(play, job, str(directory), str(candidate_lib), str(reference_lib)) for job in pending}
        while futures:
            done, futures = wait(futures, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                print('GAME ' + json.dumps(future.result()), flush=True)
            result = summary(directory)
            report(directory, result)
            print(f"PROGRESS {result['completed']}/{len(jobs)}; errors={len(result['errors'])}", flush=True)
    result = summary(directory)
    report(directory, result)
    print('FINAL ' + json.dumps(result, ensure_ascii=False), flush=True)
    if result['completed'] != len(jobs) or result['divergences']:
        raise SystemExit('Campaign requires review: incomplete games or proof/outcome divergences')


if __name__ == '__main__':
    main()

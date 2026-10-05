"""Paired 35s/5s evaluation using an integrity-checked frozen engine and rules."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import random
import shutil
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'experiments/references/initial-cb732bb'
SOURCE = REFERENCE / 'source'
# Both native search and independent Python validation use the frozen snapshot.
sys.path.insert(0, str(SOURCE))
os.environ['SANTONI_ENGINE_LIB'] = str(REFERENCE / 'libsantoni_engine.so')
from santorini.engine import Position, validate_setup, validate_turn
from santorini.native import native_search
from santorini.powers import POWERS
from santorini.storage import Session, action_dict, now


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_reference():
    meta = json.loads((REFERENCE / 'reference.json').read_text())
    for path, expected in [(REFERENCE / 'libsantoni_engine.so', meta['binary_sha256']),
                           (REFERENCE / 'source.tar.gz', meta['archive_sha256'])]:
        if digest(path) != expected:
            raise ValueError(f'Reference integrity mismatch: {path}')
    for name, expected in meta['source_sha256'].items():
        if digest(SOURCE / name) != expected:
            raise ValueError(f'Frozen source integrity mismatch: {name}')
    return meta


def paired_jobs(original):
    result = []
    for job in original['jobs']:
        for long_player in (0, 1):
            result.append({**job, 'number': len(result) + 1,
                           'original_game': job['number'], 'long_player': long_player,
                           'budgets': [35 if p == long_player else 5 for p in (0, 1)]})
    return result


def initialise(cpus):
    os.nice(10)
    identity = multiprocessing.current_process()._identity[0]
    os.sched_setaffinity(0, {cpus[(identity - 1) % len(cpus)]})


def physical_cpus():
    cores = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        base = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key = ((base / 'physical_package_id').read_text(), (base / 'core_id').read_text())
        cores.setdefault(key, cpu)
    return list(cores.values())


def play(job, seconds, directory):
    path = Path(directory) / f"game-{job['number']:03d}.json"
    position = validate_setup(job['workers'], job['powers'], job['first'])
    session = Session(position, settings={'selfplay': True, 'seconds': seconds, **job})
    data = {'job': job, 'started_at': now(), 'decisions': [], 'status': 'running',
            'session': session.to_dict()}
    started = time.monotonic()
    atomic_json(path, data)
    try:
        for ply in range(150):
            before = session.position
            attempted = 0
            def progress(a):
                nonlocal attempted
                attempted = max(attempted, a.searching_depth, a.depth)
            wall = time.monotonic()
            budget = seconds[before.player]
            analysis = native_search(before, budget, publish=progress)
            metadata = {key: getattr(analysis, key) for key in
                        ('score', 'depth', 'nodes', 'elapsed', 'proven', 'complete_depth', 'status', 'stop_reason', 'engine')}
            metadata.update(budget_seconds=budget, player=before.player, ply=ply+1, wall_elapsed=time.monotonic()-wall,
                            attempted_depth=attempted,
                            variation=[{'actions': [action_dict(a) for a in t.actions], 'after': t.after.to_dict()}
                                       for t in analysis.variation])
            data['decisions'].append(metadata)
            if analysis.turn is None:
                if analysis.stop_reason == 'terminal' and analysis.score == -100000:
                    session = session.ended(1-before.player, 'Aucun tour complet légal')
                else:
                    data['status'] = 'inconclusive'
                    data['error'] = 'No completed legal turn found within budget'
                break
            turn = validate_turn(before, analysis.turn.actions)
            if turn.after != analysis.turn.after:
                raise ValueError('Native/reference state mismatch')
            session = session.append(turn)
            data.update(session=session.to_dict(), elapsed=time.monotonic()-started)
            atomic_json(path, data)
            if session.result is not None:
                break
        else:
            data['status'] = 'inconclusive'
            data['error'] = '150-turn limit reached'
        if session.result is not None:
            data['status'] = 'finished'
    except Exception as exc:
        data['status'] = 'error'
        data['error'] = f'{type(exc).__name__}: {exc}'
    data.update(session=session.to_dict(), elapsed=time.monotonic()-started, finished_at=now())
    atomic_json(path, data)
    return {'number': job['number'], 'status': data['status'], 'turns': len(session.history),
            'elapsed': data['elapsed'], 'result': session.result, 'error': data.get('error')}


def summary(directory):
    games = [json.loads(p.read_text()) for p in sorted(directory.glob('game-*.json'))]
    finished = [g for g in games if g['status'] == 'finished']
    wins = Counter()
    by_matchup = {}
    decisions = defaultdict(list)
    clusters = defaultdict(list)
    pairs = defaultdict(list)
    for game in finished:
        job = game['job']
        won = game['session']['result']['winner'] == job['long_player']
        wins['35' if won else '5'] += 1
        key = '/'.join(POWERS[p].name for p in job['matchup'])
        row = by_matchup.setdefault(key, {'games': 0, 'wins_35': 0, 'wins_5': 0})
        row['games'] += 1
        row['wins_35' if won else 'wins_5'] += 1
        clusters[(tuple(job['matchup']), job['opening'])].append(int(won))
        pairs[job['original_game']].append(int(won))
        for d in game['decisions']:
            decisions[str(d['budget_seconds'])].append(d)
    metrics = {}
    for budget, ds in decisions.items():
        metrics[budget] = {
            'searches': len(ds), 'median_depth': statistics.median(d['depth'] for d in ds),
            'mean_depth': statistics.mean(d['depth'] for d in ds),
            'search_seconds': sum(d['elapsed'] for d in ds),
            'mean_wall_seconds': statistics.mean(d['wall_elapsed'] for d in ds),
            'max_wall_seconds': max(d['wall_elapsed'] for d in ds),
            'stop_reasons': dict(Counter(d['stop_reason'] for d in ds)),
            'proofs': sum(d['proven'] for d in ds)}
    result = {'updated_at': now(), 'completed': len(finished), 'expected': 200,
              'statuses': dict(Counter(g['status'] for g in games)),
              'wins_by_budget': dict(wins), 'by_matchup': by_matchup,
              'search_metrics': metrics,
              'paired_results': dict(Counter(str(sum(v)) + '/2 wins at 35s' for v in pairs.values() if len(v) == 2)),
              'errors': [{'game': g['job']['number'], 'error': g.get('error')} for g in games if g['status'] in ('error', 'inconclusive')]}
    # Resample whole original placements, retaining their correlated variants.
    if len(finished) == 200:
        rng = random.Random(20261005)
        groups = list(clusters.values())
        samples = []
        for _ in range(10000):
            chosen = [rng.choice(groups) for _ in groups]
            samples.append(sum(sum(v) for v in chosen) / sum(len(v) for v in chosen))
        samples.sort()
        result['win_rate_35'] = wins['35'] / 200
        result['placement_cluster_bootstrap_95'] = [samples[250], samples[9749]]
        result['independent_placement_groups'] = len(groups)
    atomic_json(directory / 'summary.json', result)
    return result


def critical_rechecks(directory, original_directory, cpu):
    """Separate identical-position experiment; never overwrite original records."""
    os.sched_setaffinity(0, {cpu})
    corpus = json.loads((original_directory / 'critical-positions.json').read_text())
    cases = {(16, 26), (18, 28)}
    for item in corpus:
        if 'game' in item and 'ply' in item:
            cases.add((item['game'], item['ply']))
    for game, ply in sorted(cases):
        record = json.loads((original_directory / f'game-{game:03d}.json').read_text())
        state = Position.from_dict(record['session']['initial'] if ply == 1 else record['session']['history'][ply - 2]['after'])
        for budget in (5, 35):
            path = directory / f'recheck-{game:03d}-{ply:03d}-{budget}s.json'
            if path.exists():
                continue
            result = native_search(state, budget)
            data = {'game': game, 'ply': ply, 'budget_seconds': budget,
                    'position': state.to_dict(), 'score': result.score, 'depth': result.depth,
                    'proven': result.proven, 'elapsed': result.elapsed, 'stop_reason': result.stop_reason,
                    'actions': [action_dict(a) for a in result.turn.actions] if result.turn else [],
                    'variation': [{'actions': [action_dict(a) for a in t.actions], 'after': t.after.to_dict()} for t in result.variation]}
            atomic_json(path, data)
            print(f'RECHECK {game}:{ply} {budget}s depth={result.depth} score={result.score}', flush=True)


def write_report(directory, result, audit):
    config = json.loads((directory / 'campaign.json').read_text())
    lines = ['# Campagne moteur initial : 35 s contre 5 s', '',
             f"Parties terminées : {result['completed']}/200.", '',
             'Les 100 configurations initiales sont chacune jouées deux fois, avec inversion des budgets.',
             'Même moteur et mêmes règles figés pour les deux joueurs ; aucune amélioration stratégique.', '',
             '**Limite de comparabilité historique :** binaire recompilé depuis les mêmes sources, sur un autre environnement matériel. '
             'La première campagne utilisait 12 cœurs ; celle-ci en utilise ' + str(len(config['cpus'])) + '.', '',
             f"Victoires à 35 s : {result['wins_by_budget'].get('35', 0)} ; à 5 s : {result['wins_by_budget'].get('5', 0)}."]
    if 'win_rate_35' in result:
        lo, hi = result['placement_cluster_bootstrap_95']
        lines += [f"Taux de victoire à 35 s : {result['win_rate_35']:.1%}. Intervalle bootstrap à 95 %, regroupé par placement : [{lo:.1%}, {hi:.1%}].",
                  'Cet intervalle dépend des 25 groupes de placements disponibles ; il ne démontre pas une supériorité universelle.']
    lines += ['', '| Budget | Recherches | Profondeur médiane | Temps moyen réel |', '|---|---:|---:|---:|']
    for budget, row in sorted(result['search_metrics'].items(), key=lambda item: int(item[0])):
        lines.append(f"| {budget} s | {row['searches']} | {row['median_depth']} | {row['mean_wall_seconds']:.3f} s |")
    lines += ['', 'Les profondeurs agrégées concernent des positions différentes et ne constituent pas une comparaison à position fixe.', '',
              '| Confrontation | Parties | Victoires 35 s | Victoires 5 s |', '|---|---:|---:|---:|']
    for name, row in result['by_matchup'].items():
        lines.append(f"| {name} | {row['games']} | {row['wins_35']} | {row['wins_5']} |")
    lines += ['', f"Erreurs ou parties non conclues : {len(result['errors'])}.",
              f"Historiques invalides : {len(audit['audit_errors'])}. Contradictions de preuves : {len(audit['proof_contradictions'])}.", '',
              '## Positions identiques', '', '| Position | Budget | Profondeur | Score heuristique | Démontré |', '|---|---:|---:|---:|---|']
    for path in sorted(directory.glob('recheck-*.json')):
        d = json.loads(path.read_text())
        lines.append(f"| {d['game']}/{d['ply']} | {d['budget_seconds']} s | {d['depth']} | {d['score']} | {d['proven']} |")
    lines += ['', 'Les scores heuristiques ne sont pas des probabilités de victoire. Les actions et variantes complètes sont conservées dans les fichiers recheck.',
              'Les réanalyses sont réalisées séparément, après les matchs, sans concurrence du banc de parties.', '']
    (directory / 'rapport.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    meta = verify_reference()
    original_directory = ROOT / 'experiments/selfplay-100-5s-20261005'
    original = json.loads((original_directory / 'campaign.json').read_text())
    scheduled = paired_jobs(original)
    assert len(scheduled) == 200
    cpus = physical_cpus()[:args.workers]
    directory = Path(args.output).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    # Prevent simultaneous resumptions from racing over the same game files.
    import fcntl
    lock = (directory / '.campaign.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = {'started_at': now(), 'protocol': 'paired-budgets-v1', 'jobs': scheduled,
              'engine_sha256': meta['binary_sha256'], 'reference_manifest_sha256': digest(REFERENCE / 'reference.json'),
              'runner_sha256': digest(Path(__file__)), 'original_campaign_sha256': digest(original_directory / 'campaign.json'),
              'workers': len(cpus), 'cpus': cpus, 'seed': original['seed'],
              'platform': platform.platform(), 'python': platform.python_version(),
              'cpu_model': meta['cpu_model'], 'original_engine_sha256': original['engine_sha256'],
              'original_cpus': original['cpus'],
              'limitations': ['Original binary absent: rebuilt from unchanged engine sources.',
                              'Different hardware; original campaign used 12 physical-core workers.',
                              'Wall-time searches are not bit-for-bit deterministic.']}
    manifest = directory / 'campaign.json'
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if {k: v for k, v in previous.items() if k != 'started_at'} != {k: v for k, v in config.items() if k != 'started_at'}:
            parser.error('Campaign provenance/parameters differ; use another output directory')
    else:
        atomic_json(manifest, config)
    if args.prepare_only:
        print('PREPARED 200 paired games; integrity verified', flush=True)
        return
    if shutil.disk_usage(directory).free < 200 * 1024 * 1024:
        parser.error('At least 200 MiB of free disk space is required before starting the campaign')
    pending = []
    for job in scheduled:
        path = directory / f"game-{job['number']:03d}.json"
        if path.exists():
            game = json.loads(path.read_text())
            if game['job'] != job:
                raise ValueError(f'Job mismatch in {path}')
            if game['status'] == 'finished':
                continue
            # Preserve interrupted/failed attempts instead of silently overwriting evidence.
            attempts = directory / 'attempts'
            attempts.mkdir(exist_ok=True)
            path.rename(attempts / f'{path.stem}-{time.time_ns()}.json')
        pending.append(job)
    print(f'START {len(pending)}/200 remaining games; 35s vs 5s; CPUs {cpus}', flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=len(cpus), mp_context=multiprocessing.get_context('spawn'),
                                 initializer=initialise, initargs=(cpus,)) as pool:
            futures = [pool.submit(play, job, job['budgets'], str(directory)) for job in pending]
            for future in as_completed(futures):
                item = future.result()
                result = summary(directory)
                print(f"DONE {result['completed']}/200 game={item['number']} status={item['status']} turns={item['turns']} wins={result['wins_by_budget']}", flush=True)
    critical_rechecks(directory, original_directory, cpus[0])
    sys.path.insert(0, str(SOURCE / 'tools'))
    from analyse_selfplay import analyse
    audit = analyse(directory)
    result = summary(directory)
    write_report(directory, result, audit)
    print('FINAL ' + json.dumps(result, ensure_ascii=False), flush=True)
    if result['completed'] != 200 or audit['audit_errors'] or audit['proof_contradictions']:
        raise SystemExit('Campaign requires review: incomplete games or audit findings')


if __name__ == '__main__':
    main()

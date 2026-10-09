"""Equal-budget comparison of distinct native engines in isolated subprocesses."""
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
import select
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
sys.path.insert(0, str(ROOT))
from santorini.engine import Action, Position, validate_setup, validate_turn
from santorini.storage import Session, action_dict, now


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def worker(binary):
    os.environ['SANTONI_ENGINE_LIB'] = str(Path(binary).resolve())
    from santorini.native import native_search, library
    library()  # Load before advertising readiness, once per isolated subprocess.
    binary_hash = digest(binary)
    print(json.dumps({'ready': binary_hash}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        state = Position.from_dict(request['position'])
        attempted = 0
        def progress(result):
            nonlocal attempted
            attempted = max(attempted, result.depth, result.searching_depth)
        start = time.monotonic()
        result = native_search(state, request['seconds'], publish=progress)
        answer = {key: getattr(result, key) for key in
                  ('score', 'depth', 'nodes', 'elapsed', 'proven', 'stop_reason')}
        answer.update(wall_elapsed=time.monotonic() - start, attempted_depth=attempted,
                      binary_sha256=binary_hash,
                      turn={'actions': [action_dict(a) for a in result.turn.actions],
                            'after': result.turn.after.to_dict()} if result.turn else None,
                      variation=[{'actions': [action_dict(a) for a in turn.actions],
                                  'after': turn.after.to_dict()} for turn in result.variation])
        print(json.dumps(answer), flush=True)


class EngineProcess:
    def __init__(self, binary):
        self.binary = Path(binary).resolve()
        self.expected_hash = digest(self.binary)
        self.process = subprocess.Popen([sys.executable, str(SCRIPT), 'worker', str(self.binary)],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True, bufsize=1)
        try:
            ready = self._read(30)
            if ready.get('ready') != self.expected_hash:
                raise ValueError('Le processus ne charge pas le binaire attendu.')
        except BaseException:
            self.close()
            raise

    def _read(self, timeout):
        if not select.select([self.process.stdout], [], [], timeout)[0]:
            raise TimeoutError('Le processus moteur ne répond pas.')
        line = self.process.stdout.readline()
        if not line:
            detail = self.process.stderr.read()
            raise RuntimeError(f'Processus moteur arrêté : {detail}')
        return json.loads(line)

    def analyse(self, position, seconds):
        self.process.stdin.write(json.dumps({'position': position.to_dict(), 'seconds': seconds}) + '\n')
        self.process.stdin.flush()
        result = self._read(seconds + 30)
        if result['binary_sha256'] != self.expected_hash:
            raise ValueError('Le binaire a changé pendant la recherche.')
        # The bridge already validates every published PV; the supervisor checks
        # the final PV again independently of subprocess state and engine choice.
        before = position
        for row in result['variation']:
            turn = validate_turn(before, tuple(Action(**a) for a in row['actions']))
            if turn.after != Position.from_dict(row['after']):
                raise ValueError('Variation native incohérente.')
            before = turn.after
        if result['turn']:
            turn = validate_turn(position, tuple(Action(**a) for a in result['turn']['actions']))
            if turn.after != Position.from_dict(result['turn']['after']):
                raise ValueError('Tour natif incohérent.')
        return result

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def physical_cpus():
    cores = {}
    for cpu in sorted(os.sched_getaffinity(0)):
        base = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key = ((base / 'physical_package_id').read_text(), (base / 'core_id').read_text())
        cores.setdefault(key, cpu)
    return list(cores.values())


def initialise(cpus):
    os.nice(10)
    identity = multiprocessing.current_process()._identity[0]
    os.sched_setaffinity(0, {cpus[(identity - 1) % len(cpus)]})


def paired_jobs(seed, matchups):
    rng = random.Random(seed)
    jobs = []
    for pair_number, powers in enumerate(matchups, 1):
        cells = rng.sample(range(25), 4)
        first = rng.randrange(2)
        for candidate_player in (0, 1):
            jobs.append({'number': len(jobs) + 1, 'pair': pair_number, 'powers': list(powers),
                         'workers': [cells[:2], cells[2:]], 'first': first,
                         'candidate_player': candidate_player})
    return jobs


def position_job(case, binaries, seconds, directory, index):
    state = Position.from_dict(case['position'])
    results = {}
    # Alternate execution order across positions; both runs share the same core.
    names = ['baseline', 'candidate'] if index % 2 == 0 else ['candidate', 'baseline']
    for name in names:
        with EngineProcess(binaries[name]) as engine:
            results[name] = engine.analyse(state, seconds)
    data = {'case': case, 'seconds': seconds, 'execution_order': names, 'results': results}
    write_json(Path(directory) / f'position-{index + 1:03d}.json', data)
    return {'case': case['id'], 'depths': {name: value['depth'] for name, value in results.items()},
            'same_successor': results['baseline']['turn']['after'] == results['candidate']['turn']['after']
            if all(value['turn'] for value in results.values()) else None}


def match_job(job, binaries, seconds, directory, turn_limit):
    path = Path(directory) / f'game-{job["number"]:03d}.json'
    session = Session(validate_setup(job['workers'], job['powers'], job['first']),
                      settings={'version_bench': True, 'seconds': seconds, **job})
    data = {'job': job, 'started_at': now(), 'status': 'running', 'decisions': [], 'session': session.to_dict()}
    write_json(path, data)
    try:
        with EngineProcess(binaries['baseline']) as baseline, EngineProcess(binaries['candidate']) as candidate:
            for ply in range(1, turn_limit + 1):
                state = session.position
                name = 'candidate' if state.player == job['candidate_player'] else 'baseline'
                analysis = (candidate if name == 'candidate' else baseline).analyse(state, seconds)
                data['decisions'].append({'ply': ply, 'player': state.player, 'version': name, **analysis})
                if analysis['turn'] is None:
                    if analysis['stop_reason'] == 'terminal' and analysis['score'] == -100000:
                        session = session.ended(1 - state.player, 'Aucun tour complet légal')
                    else:
                        data.update(status='inconclusive', error='Aucun tour complet dans le budget')
                    break
                turn = validate_turn(state, tuple(Action(**a) for a in analysis['turn']['actions']))
                session = session.append(turn)
                data['session'] = session.to_dict()
                write_json(path, data)
                if session.result:
                    break
            else:
                data.update(status='inconclusive', error=f'Plafond de {turn_limit} tours atteint')
        if session.result:
            data['status'] = 'finished'
    except Exception as exc:
        data.update(status='error', error=f'{type(exc).__name__}: {exc}')
    data.update(session=session.to_dict(), finished_at=now())
    write_json(path, data)
    return {'number': job['number'], 'status': data['status'], 'turns': len(session.history),
            'winner': session.result['winner'] if session.result else None,
            'candidate_won': session.result['winner'] == job['candidate_player'] if session.result else None}


def match_summary(directory):
    records = [json.loads(p.read_text()) for p in sorted(Path(directory).glob('game-*.json'))]
    wins, statuses, pairs, metrics, contradictions = Counter(), Counter(), defaultdict(list), defaultdict(list), []
    for record in records:
        statuses[record['status']] += 1
        # Replaying persisted histories also detects supervisor/serialization errors.
        session = Session.from_dict(record['session'])
        if record['status'] != 'finished':
            continue
        winner = session.result['winner']
        name = 'candidate' if winner == record['job']['candidate_player'] else 'baseline'
        wins[name] += 1
        pairs[record['job']['pair']].append(name)
        for decision in record['decisions']:
            metrics[decision['version']].append(decision)
            if decision['proven'] and decision['score'] is not None:
                predicted = decision['player'] if decision['score'] > 0 else 1 - decision['player']
                if predicted != winner:
                    contradictions.append({'game': record['job']['number'], 'ply': decision['ply']})
    result = {'statuses': dict(statuses), 'wins': dict(wins), 'pairs': dict(pairs),
              'proof_contradictions': contradictions,
              'metrics': {name: {'searches': len(rows), 'median_depth': statistics.median(r['depth'] for r in rows),
                          'mean_seconds': statistics.mean(r['elapsed'] for r in rows),
                          'stop_reasons': dict(Counter(r['stop_reason'] for r in rows))}
                          for name, rows in metrics.items()}}
    write_json(Path(directory) / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='mode', required=True)
    internal = subs.add_parser('worker')
    internal.add_argument('binary', type=Path)
    for mode in ('positions', 'matches'):
        command = subs.add_parser(mode)
        command.add_argument('--baseline', type=Path, required=True)
        command.add_argument('--candidate', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
        command.add_argument('--seconds', type=float, default=5)
        command.add_argument('--workers', type=int, default=4)
        if mode == 'positions':
            command.add_argument('--positions', type=Path, required=True)
        else:
            command.add_argument('--seed', type=int, default=20261007)
            command.add_argument('--matchups', default='0:0,7:6,1:8,2:3,4:9,5:10')
            command.add_argument('--turn-limit', type=int, default=120)
    args = parser.parse_args()
    if args.mode == 'worker':
        worker(args.binary)
        return
    if args.output.exists():
        parser.error('Utiliser un nouveau dossier de sortie.')
    if args.workers < 1 or not .001 <= args.seconds <= 3600:
        parser.error('Budget ou nombre de processus invalide.')
    cpus = physical_cpus()[:args.workers]
    binaries = {'baseline': str(args.baseline.resolve()), 'candidate': str(args.candidate.resolve())}
    fingerprints = {name: digest(path) for name, path in binaries.items()}
    if fingerprints['baseline'] == fingerprints['candidate']:
        parser.error('Les deux binaires doivent être distincts.')
    if args.mode == 'positions':
        jobs = json.loads(args.positions.read_text())['cases']
        for case in jobs:
            Position.from_dict(case['position'])
    else:
        matchups = [tuple(map(int, pair.split(':'))) for pair in args.matchups.split(',')]
        jobs = paired_jobs(args.seed, matchups)
        for job in jobs:
            validate_setup(job['workers'], job['powers'], job['first'])
    args.output.mkdir(parents=True)
    manifest = {'version': 1, 'started_at': now(), 'mode': args.mode, 'seconds': args.seconds,
                'cpus': cpus, 'platform': platform.platform(), 'python': sys.version,
                'binary_sha256': fingerprints, 'jobs': jobs, 'runner_sha256': digest(SCRIPT),
                'rules_sha256': digest(ROOT / 'santorini/engine.py'),
                'turn_limit': args.turn_limit if args.mode == 'matches' else None}
    write_json(args.output / 'manifest.json', manifest)
    with ProcessPoolExecutor(max_workers=len(cpus), mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialise, initargs=(cpus,)) as pool:
        if args.mode == 'positions':
            futures = [pool.submit(position_job, case, binaries, args.seconds, str(args.output), index)
                       for index, case in enumerate(jobs)]
        else:
            futures = [pool.submit(match_job, job, binaries, args.seconds, str(args.output), args.turn_limit)
                       for job in jobs]
        for future in as_completed(futures):
            print(json.dumps(future.result(), ensure_ascii=False), flush=True)
            if args.mode == 'matches':
                match_summary(args.output)
    if args.mode == 'matches':
        print(json.dumps(match_summary(args.output), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

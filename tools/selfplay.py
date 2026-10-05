"""Reproducible, headless self-play; never reads or writes live game saves."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from santorini.engine import Position, validate_setup, validate_turn
from santorini.native import library_path, native_search
from santorini.powers import POWERS
from santorini.storage import Session, action_dict, now

MATCHUPS = [(0, 0), (7, 6), (1, 8), (2, 3), (4, 9),
            (5, 10), (1, 3), (2, 9), (4, 5), (8, 10)]
COUNTS = [3, 3, 3, 3, 3, 2, 2, 2, 2, 2]


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def jobs(seed):
    rng = random.Random(seed)
    result = []
    for pair, count in zip(MATCHUPS, COUNTS):
        for opening in range(count):
            cells = rng.sample(range(25), 4)
            for reverse in (False, True):
                for first in (0, 1):
                    # Rotate the second base-game pair instead of duplicating it.
                    board = [24-c for c in cells] if pair == (0, 0) and reverse else cells
                    powers = tuple(reversed(pair)) if reverse else pair
                    result.append(dict(number=len(result)+1, matchup=pair, opening=opening,
                                       powers=powers, workers=[board[:2], board[2:]], first=first))
    assert len(result) == 100
    return result


def initialise(cpus):
    if hasattr(os, 'nice'):
        os.nice(10)
    if cpus and hasattr(os, 'sched_setaffinity'):
        identity = multiprocessing.current_process()._identity[0]
        os.sched_setaffinity(0, {cpus[(identity-1) % len(cpus)]})


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
            analysis = native_search(before, seconds, publish=progress)
            metadata = {key: getattr(analysis, key) for key in
                        ('score', 'depth', 'nodes', 'elapsed', 'proven', 'complete_depth', 'status', 'stop_reason', 'engine')}
            metadata.update(player=before.player, ply=ply+1, wall_elapsed=time.monotonic()-wall,
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
    games = [json.loads(p.read_text()) for p in sorted(Path(directory).glob('game-*.json'))]
    done = [g for g in games if g['status'] == 'finished']
    decisions = [d for g in done for d in g['decisions']]
    by_match = {}
    for g in done:
        key = '/'.join(POWERS[p].name for p in g['job']['matchup'])
        row = by_match.setdefault(key, {'games': 0, 'wins_by_power': {}, 'first_player_wins': 0,
                                       'turns': [], 'reasons': {}})
        winner = g['session']['result']['winner']
        power = POWERS[g['job']['powers'][winner]].name
        row['games'] += 1
        row['wins_by_power'][power] = row['wins_by_power'].get(power, 0)+1
        row['first_player_wins'] += winner == g['job']['first']
        row['turns'].append(len(g['session']['history']))
        reason = g['session']['result']['reason']
        row['reasons'][reason] = row['reasons'].get(reason, 0)+1
    result = {'updated_at': now(), 'statuses': dict(Counter(g['status'] for g in games)),
              'completed': len(done), 'total_turns': sum(len(g['session']['history']) for g in done),
              'depths': dict(Counter(d['depth'] for d in decisions)),
              'stop_reasons': dict(Counter(d['stop_reason'] for d in decisions)),
              'average_search_seconds': sum(d['elapsed'] for d in decisions)/max(1, len(decisions)),
              'by_matchup': by_match,
              'errors': [{'game': g['job']['number'], 'error': g.get('error')} for g in games if g['status'] in ('error','inconclusive')]}
    atomic_json(Path(directory)/'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--seconds', type=float, default=5)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--seed', type=int, default=20261005)
    args = parser.parse_args()
    if args.workers < 1 or not .001 <= args.seconds <= 3600:
        parser.error("workers must be positive; seconds must be between .001 and 3600")
    directory = Path(args.output).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    cpus = sorted(os.sched_getaffinity(0))
    cores = {}
    for cpu in cpus:
        topology = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        key = (topology.joinpath('physical_package_id').read_text(), topology.joinpath('core_id').read_text())
        cores.setdefault(key, cpu)
    selected = list(cores.values())[:args.workers]
    scheduled = jobs(args.seed)
    config = {'started_at': now(), 'seconds': args.seconds, 'workers': len(selected), 'cpus': selected,
              'seed': args.seed, 'jobs': scheduled, 'engine_sha256': hashlib.sha256(library_path().read_bytes()).hexdigest(),
              'notes': 'Same engine on both sides. 25 openings crossed with first player and power assignment. No learning or engine edits.'}
    print(f"START {len(scheduled)} games; {args.seconds}s/turn; {len(selected)} workers", flush=True)
    pending = []
    for job in scheduled:
        file = directory/f"game-{job['number']:03d}.json"
        if file.exists() and json.loads(file.read_text()).get('status') == 'finished':
            continue
        pending.append(job)
    manifest = directory/'campaign.json'
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        expected_jobs = json.loads(json.dumps(scheduled))
        if (previous.get('seconds') != args.seconds or previous.get('seed') != args.seed
                or previous.get('jobs') != expected_jobs):
            parser.error('Existing campaign parameters differ; use a new output directory')
        if pending and previous.get('engine_sha256') != config['engine_sha256']:
            parser.error('Engine differs from the incomplete campaign; use a new output directory')
    else:
        atomic_json(manifest, config)
    if not pending:
        print('FINAL ' + json.dumps(summary(directory), ensure_ascii=False), flush=True)
        return
    with ProcessPoolExecutor(max_workers=len(selected), mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialise, initargs=(selected,)) as pool:
        futures = [pool.submit(play, job, args.seconds, str(directory)) for job in pending]
        for future in as_completed(futures):
            item = future.result()
            result = summary(directory)
            print(f"DONE {result['completed']}/100 game={item['number']} status={item['status']} turns={item['turns']} elapsed={item['elapsed']:.1f}s result={item['result']}", flush=True)
    print('FINAL ' + json.dumps(summary(directory), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

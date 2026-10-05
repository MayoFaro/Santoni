"""Audit recorded self-play and report observations without changing strategy."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from santorini.engine import Position
from santorini.powers import POWERS
from santorini.storage import Session


def analyse(directory):
    directory = Path(directory)
    games = [json.loads(p.read_text()) for p in sorted(directory.glob('game-*.json'))]
    finished = [g for g in games if g['status'] == 'finished']
    decisions = [d for g in finished for d in g['decisions']]
    reasons = Counter(g['session']['result']['reason'] for g in finished)
    depth_by_power = defaultdict(list)
    proof_contradictions, audit_errors, early_warnings = [], [], []
    first_wins = 0
    usage = defaultdict(Counter)
    power_wins = Counter()
    for g in finished:
        job = g['job']; winner = g['session']['result']['winner']
        first_wins += winner == job['first']
        power_wins[job['powers'][winner]] += 1
        try:
            session = Session.from_dict(g['session'])
        except Exception as exc:
            audit_errors.append({'game': job['number'], 'error': str(exc)})
            continue
        first_loss_proof = None
        before_turn = session.initial
        for turn in session.history:
            power = POWERS[before_turn.powers[before_turn.player]].name
            usage[power][turn.power_summary(before_turn)] += 1
            before_turn = turn.after
        for d in g['decisions']:
            player = d['player']; ply = d['ply']
            depth_by_power[job['powers'][player]].append(d['depth'])
            if d['proven'] and d['score'] is not None:
                predicted = player if d['score'] > 0 else 1-player
                if predicted != winner:
                    proof_contradictions.append({'game': job['number'], 'ply': ply, 'predicted': predicted, 'winner': winner})
                if player != winner and first_loss_proof is None:
                    first_loss_proof = ply
        loser = 1-winner
        losing = [d for d in g['decisions'] if d['player'] == loser]
        if first_loss_proof is not None:
            preceding = next((d for d in reversed(losing) if d['ply'] < first_loss_proof), None)
            early_warnings.append({'game': job['number'], 'first_loss_proof_ply': first_loss_proof,
                                   'turns_remaining': len(session.history)-first_loss_proof+1,
                                   'prior_loser_decision': preceding,
                                   'loser_power': POWERS[job['powers'][loser]].name})
    depths = Counter(d['depth'] for d in decisions)
    stops = Counter(d['stop_reason'] for d in decisions)
    result = {'completed': len(finished), 'statuses': dict(Counter(g['status'] for g in games)),
              'turns': sum(len(g['session']['history']) for g in finished), 'decisions': len(decisions),
              'first_player_wins': first_wins, 'victory_reasons': dict(reasons), 'depths': dict(depths),
              'median_depth': statistics.median([d['depth'] for d in decisions]) if decisions else None,
              'stop_reasons': dict(stops), 'median_search_seconds': statistics.median([d['elapsed'] for d in decisions]) if decisions else None,
              'depth_by_power': {POWERS[p].name: {'median': statistics.median(v), 'mean': statistics.mean(v), 'turns': len(v)} for p,v in depth_by_power.items()},
              'power_usage': {p:dict(c) for p,c in usage.items()},
              'proof_contradictions': proof_contradictions, 'audit_errors': audit_errors,
              'first_loss_proofs': early_warnings}
    (directory/'analysis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'first_loss_proofs'},ensure_ascii=False,indent=2))
    return result


if __name__ == '__main__':
    analyse(sys.argv[1])

"""Reanalyse recorded pre-turn positions with a larger deadline."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from santorini.engine import Action, Position, coord
from santorini.native import native_search
from santorini.storage import action_dict


def recheck(directory, game, ply, seconds):
    file = Path(directory)/f'game-{game:03d}.json'
    record = json.loads(file.read_text())
    before = record['session']['initial'] if ply == 1 else record['session']['history'][ply-2]['after']
    state = Position.from_dict(before)
    original = record['decisions'][ply-1]
    print(f'START game={game} ply={ply} old_score={original["score"]} old_depth={original["depth"]}',flush=True)
    result = native_search(state, seconds)
    actual = tuple(Action(**a) for a in record['session']['history'][ply-1]['actions'])
    data = {'game':game,'ply':ply,'budget':seconds,'original_score':original['score'],
            'original_depth':original['depth'],'score':result.score,'depth':result.depth,
            'proven':result.proven,'reason':result.stop_reason,'elapsed':result.elapsed,
            'same_move':result.turn is not None and result.turn.actions==actual,
            'old_actions':[action_dict(a) for a in actual],
            'suggested_actions':[action_dict(a) for a in result.turn.actions] if result.turn else [],
            'position':state.to_dict(),
            'variation':[{'actions':[action_dict(a) for a in t.actions],'after':t.after.to_dict()} for t in result.variation]}
    (Path(directory)/f'recheck-{game:03d}-{ply:03d}.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in data.items() if k not in ('position','variation')},ensure_ascii=False),flush=True)
    return data


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('directory');parser.add_argument('--cases',required=True);parser.add_argument('--seconds',type=float,default=20)
    args=parser.parse_args()
    os.nice(15)
    for case in args.cases.split(','):
        game,ply=map(int,case.split(':'));recheck(args.directory,game,ply,args.seconds)

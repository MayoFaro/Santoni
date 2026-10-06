"""Deterministic rule exercise; random actions, not a strength tournament.

Only artificial sessions are created. Includes save/load replay of every game.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from santorini.extra import NEW_POWERS
from santorini.native import native_next_actions,library_path
from santorini.setup import reference_setup
from santorini.storage import Session


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--seed',type=int,default=20261006)
    parser.add_argument('--turns',type=int,default=32)
    parser.add_argument('--opponents',type=int,nargs='+',default=[0,5,26,35])
    args=parser.parse_args()
    rng=random.Random(args.seed);counts=Counter();games=[];started=time.monotonic()
    for power in sorted(NEW_POWERS):
        for opponent in args.opponents:
            if power==opponent:continue
            cells=rng.sample(range(25),4)
            session=Session(reference_setup((cells[:2],cells[2:]),(power,opponent),rng.randrange(2)),settings={'rule_exercise':True,'seed':args.seed})
            for turn_number in range(args.turns):
                pos=session.position;prefix=()
                for action_number in range(128):
                    options,complete=native_next_actions(pos,prefix)
                    if complete and (not options or rng.random()<.6):break
                    if not options:
                        if not prefix:session=session.ended(1-pos.player,'Aucun tour complet légal')
                        break
                    prefix+=(rng.choice(options),)
                else:raise RuntimeError('128 actions without completing a turn')
                if complete:
                    complete.after.validate()
                    counts.update(action.kind for action in complete.actions)
                    session=session.append(complete)
                elif session.result is None:raise RuntimeError(f'Incomplete turn {power}/{opponent}: {prefix}')
                if session.result:break
            saved=session.to_dict()
            restored=Session.from_dict(saved)
            assert restored.position==session.position
            games.append({'powers':[power,opponent],'turns':len(session.history),'winner':session.result.get('winner') if session.result else None,'session':saved})
        print(f'power {power}: {len(games)} games replayed',flush=True)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    result={'seed':args.seed,'max_turns':args.turns,'opponents':args.opponents,'engine_sha256':hashlib.sha256(library_path().read_bytes()).hexdigest(),'elapsed':time.monotonic()-started,'games':len(games),'recorded_turns':sum(g['turns'] for g in games),'actions':dict(counts),'replay_errors':0,'note':'Random legal action exercise and archive replay, not a strategic evaluation.'}
    (out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (out/'games.json').write_text(json.dumps(games,ensure_ascii=False,separators=(',',':'))+'\n')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()

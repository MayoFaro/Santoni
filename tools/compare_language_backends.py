"""Compare compute diagnostics at equal time; this does not measure playing strength."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from santorini.engine import Position,validate_turn
from santorini.native import native_search,library_path
from santorini.search import search


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=.25)
    parser.add_argument('--repeats',type=int,default=3)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for power in (0,11,27,29,46,50,54):
        for repeat in range(args.repeats):
            state=Position(powers=(power,0));results={}
            order=[('python',search),('rust',native_search)]
            if repeat%2:order.reverse()
            for name,backend in order:
                start=time.monotonic();result=backend(state,args.seconds)
                wall=time.monotonic()-start
                if result.turn:assert validate_turn(state,result.turn.actions).after==result.turn.after
                results[name]={'nodes':result.nodes,'depth':result.depth,'elapsed':result.elapsed,
                               'wall_seconds':wall,'proven':result.proven,'complete_depth':result.complete_depth,
                               'actions':[a.__dict__ for a in result.turn.actions] if result.turn else None}
            rows.append({'power':power,'repeat':repeat,'results':results})
            print(json.dumps({'power':power,'repeat':repeat,'depths':{k:v['depth'] for k,v in results.items()},
                              'nodes':{k:v['nodes'] for k,v in results.items()}}),flush=True)
    data={'seconds':args.seconds,'repeats':args.repeats,'rows':rows,
          'warning':'Same rules and evaluation; move ordering and implementation differ. Node/depth gains do not prove more wins.',
          'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
             [ROOT/'native_engine/src/lib.rs',ROOT/'native_engine/src/advanced.rs',ROOT/'santorini/engine.py',ROOT/'santorini/search.py',ROOT/'santorini/native.py',library_path()]}}
    (args.output/'results.json').write_text(json.dumps(data,indent=2)+'\n')

if __name__=='__main__':main()

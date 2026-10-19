"""Falls and times of policies on seeds outside the official 1-5: robust.py EVENT NAME A.onnx VX ..."""
import sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from arena.course import compile_course
from arena.manifest import load_event
from arena.policy import load_policy
from arena.runner import run_once

SEEDS = range(6, 26)
EVENT = sys.argv[1]


def one(args):
    path, vx = args
    man = load_event(EVENT); m = compile_course(man); pol = load_policy(path)
    out = []
    for s in SEEDS:
        o = run_once(man, pol, s, model=m, command=man.command_at(vx))
        out.append(o.time_s if o.status == "finish" else None)
    return out


EVENT = sys.argv[1]
argv = sys.argv[1:]
jobs = [(p, float(v)) for p, v in zip(argv[2::3], argv[3::3])]
with ProcessPoolExecutor(4) as ex:
    res = list(ex.map(one, jobs))
for name, (p, vx), r in zip(argv[1::3], jobs, res):
    fin = [t for t in r if t is not None]
    print(f"{name:14s} @{vx:.2f}: finished {len(fin)}/20, falls {20-len(fin)}, "
          f"finish times mean {np.mean(fin) if fin else float('nan'):.3f} s (min {min(fin) if fin else 0:.2f}, max {max(fin) if fin else 0:.2f})")

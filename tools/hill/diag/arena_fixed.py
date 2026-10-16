import sys, numpy as np
from arena.manifest import load_event
from arena.course import compile_course
from arena.policy import load_policy
from arena.runner import run_once
ev = sys.argv[1]; p = sys.argv[2]
man = load_event(ev); model = compile_course(man)
for vx in [float(v) for v in sys.argv[3].split(',')]:
    c = np.asarray(man.command, float).copy(); c[0] = vx
    rs = [run_once(man, load_policy(p), s, model=model, command=c) for s in range(1, 17)]
    t = [r.time_s for r in rs if r.status.startswith('fin')]
    print('ARENA', p.split('/')[-1][-24:], 'vx', vx, 'fin', len(t), '/16 mean', round(float(np.mean(t)), 3) if t else None, flush=True)

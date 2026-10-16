import sys, numpy as np, arena.actuator as Act
lo, hi = int(sys.argv[1]), int(sys.argv[2]); Act.DELAY_STEPS = (lo, hi)
from arena.manifest import load_event
from arena.course import compile_course
from arena.policy import load_policy
from arena.runner import run_once
man = load_event('hill-climb-1m5'); model = compile_course(man)
for p in sys.argv[3:]:
    c = np.asarray(man.command, float).copy(); c[0] = 0.3
    rs = [run_once(man, load_policy(p), s, model=model, command=c) for s in range(1, 6)]
    print(f'delay{lo}-{hi}', p.split('/')[-1][-20:], [(r.status[:4], r.time_s) for r in rs])

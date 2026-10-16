import sys, numpy as np, arena.actuator as Act
if sys.argv[1] == 'off': Act.MAX_CURRENT = 1e6
from arena.manifest import load_event
from arena.course import compile_course
from arena.policy import load_policy
from arena.runner import run_once
man = load_event('hill-climb-1m5'); model = compile_course(man)
for p in sys.argv[2:]:
    c = np.asarray(man.command, float).copy(); c[0] = 0.3
    print(sys.argv[1], p.split('/')[-1][-22:], [(r.status[:4], r.time_s) for r in (run_once(man, load_policy(p), s, model=model, command=c) for s in range(1, 6))])

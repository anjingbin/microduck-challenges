import sys,numpy as np
from arena.manifest import load_event
from arena.course import compile_course
from arena.policy import load_policy
from arena.runner import run_once
man=load_event('hill-climb-1m5'); model=compile_course(man); pol=load_policy(sys.argv[1])
for vx in [0.15,0.25,0.4,0.6,0.8,1.0]:
    c=np.asarray(man.command,dtype=np.float64).copy(); c[0]=vx
    print(vx,[(r.status[0],r.time_s) for r in (run_once(man,pol,s,model=model,command=c) for s in range(1,6))],flush=True)

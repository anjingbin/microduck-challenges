import numpy as np
from arena.manifest import load_event
from arena.course import compile_course
from arena.policy import load_policy
from arena.runner import run_once
man=load_event('roller-sprint-2m'); model=compile_course(man)
for name in ['xin','namenu']:
    pol=load_policy(f'/workspace/arena_compare/namenu_roller/{name}.onnx')
    for vx in ([1.6] if name=='xin' else [0.9]):
        c=np.asarray(man.command,dtype=np.float64).copy(); c[0]=vx
        res=[run_once(man,pol,s,model=model,command=c) for s in range(1,31)]
        print(name,vx,'seeds1-5',[ (r.status,r.time_s) for r in res[:5]],' seeds1-30 finish',sum(r.status=='finish' for r in res),'/30',flush=True)

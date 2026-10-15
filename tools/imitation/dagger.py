import numpy as np, torch, torch.nn as nn, time, sys, onnx
from arena.manifest import load_event
from arena.course import compile_course
from arena.policy import load_policy
from arena.runner import run_once
torch.set_num_threads(8)
man=load_event('roller-sprint-2m'); model=compile_course(man)
expert=load_policy('/workspace/arena_compare/namenu_roller/xin.onnx')
print('expert inputs',[i.name for i in expert.session.get_inputs()],[o.name for o in expert.session.get_outputs()])
CMD=np.asarray(man.command,dtype=np.float64).copy(); CMD[0]=1.6
net=nn.Sequential(nn.Linear(61,256),nn.ELU(),nn.Linear(256,256),nn.ELU(),nn.Linear(256,128),nn.ELU(),nn.Linear(128,14))
class Mix:
    def __init__(s,beta,rng): s.beta=beta; s.rng=rng; s.O=[]; s.A=[]
    def act(s,obs):
        a_e=np.asarray(expert.act(obs),dtype=np.float32).reshape(-1)
        if np.all(np.isfinite(obs)) and np.abs(obs).max()<100 and np.all(np.isfinite(a_e)) and np.abs(a_e).max()<20:
            s.O.append(np.asarray(obs,dtype=np.float32).copy()); s.A.append(a_e)
        else: return np.zeros(14,dtype=np.float32)
        if s.rng.random()<s.beta: return a_e
        with torch.no_grad(): return np.clip(net(torch.from_numpy(np.asarray(obs,dtype=np.float32))[None])[0].numpy(),-5,5)
import copy
best=(-1,99,None)
O=[];A=[]; rng=np.random.default_rng(0); t0=time.time(); seed=100
opt=torch.optim.Adam(net.parameters(),lr=1e-3)
for it in range(int(sys.argv[1]) if len(sys.argv)>1 else 10):
    beta=max(0.0,1.0-it*0.2); fin=0; times=[]
    for k in range(30):
        p=Mix(beta,rng); seed=(k%5)+1 if k<10 else seed+1
        cmd=CMD.copy(); cmd[0]=rng.uniform(1.3,1.9) if k%2 else 1.6
        r=run_once(man,p,seed,model=model,command=cmd)
        O+=p.O; A+=p.A; fin+= r.status=='finish'; times.append(r.time_s or 40)
    X=torch.tensor(np.array(O)); Y=torch.tensor(np.array(A))
    for ep in range(300):
        idx=torch.randint(0,len(X),(1024,)); loss=((net(X[idx])-Y[idx])**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    S=Mix(0.0,rng); ev=[run_once(man,S,sd,model=model,command=CMD) for sd in range(1,6)]
    print(f"EVAL seeds1-5 @1.6: {[(r.status,r.time_s) for r in ev]}",flush=True)
    nf=sum(r.status=='finish' for r in ev); mt=np.mean([r.time_s or 40 for r in ev])
    if (nf,-mt)>(best[0],-best[1]): best=(nf,mt,copy.deepcopy(net.state_dict())); print('BEST',nf,round(mt,3),flush=True)
    print(f"it{it} beta={beta:.1f} data={len(X)} finish {fin}/30 mean_t={np.mean(times):.2f} loss={loss.item():.4f} {time.time()-t0:.0f}s",flush=True)
net.load_state_dict(best[2]); net.eval()
torch.onnx.export(net,torch.zeros(1,61),'/workspace/imitation/student.onnx',input_names=[expert.session.get_inputs()[0].name],output_names=[expert.session.get_outputs()[0].name],opset_version=17,dynamo=False)
print('exported')

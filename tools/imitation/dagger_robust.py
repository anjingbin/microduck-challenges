import numpy as np, torch, torch.nn as nn, time, sys, copy, os
from arena.manifest import load_event
from arena.course import compile_course
from arena.policy import load_policy
from arena.runner import run_once
from arena.evaluate import evaluate
torch.set_num_threads(8)
H=int(os.environ.get('H',512)); NOISE=float(os.environ.get('NOISE',0.1)); ROUNDS=int(sys.argv[1]); TAG=os.environ.get('TAG','r')
man=load_event('roller-sprint-2m'); model=compile_course(man)
expert=load_policy('/workspace/arena_compare/namenu_roller/xin.onnx')
IN=expert.session.get_inputs()[0].name; OUT=expert.session.get_outputs()[0].name
BASE=np.asarray(man.command,dtype=np.float64)
net=nn.Sequential(nn.Linear(61,H),nn.ELU(),nn.Linear(H,H),nn.ELU(),nn.Linear(H,H//2),nn.ELU(),nn.Linear(H//2,14))
def ok(x,m): return np.all(np.isfinite(x)) and np.abs(x).max()<m
class Stop(Exception): pass
class Mix:
    def __init__(s,beta,noise,rng): s.beta,s.noise,s.rng=beta,noise,rng; s.O=[]; s.A=[]
    def act(s,obs):
        if len(s.O)>=150: raise Stop
        a_e=np.asarray(expert.act(obs),dtype=np.float32).reshape(-1)
        if not(ok(obs,100) and ok(a_e,20)): return np.zeros(14,dtype=np.float32)
        s.O.append(np.asarray(obs,dtype=np.float32).copy()); s.A.append(a_e)
        if s.rng.random()<s.beta: a=a_e
        else:
            with torch.no_grad(): a=net(torch.from_numpy(np.asarray(obs,dtype=np.float32))[None])[0].numpy()
        return np.clip(a+s.noise*s.rng.standard_normal(14).astype(np.float32),-5,5)
def export(path):
    net.eval(); torch.onnx.export(net,torch.zeros(1,61),path,input_names=[IN],output_names=[OUT],opset_version=17,dynamo=False); net.train()
O=[];A=[]; rng=np.random.default_rng(int(os.environ.get('SEED',0))); t0=time.time(); rs=1000
opt=torch.optim.Adam(net.parameters(),lr=1e-3); best=(-1,99,None)
sweep=np.round(np.arange(1.0,2.01,0.1),1)
for it in range(ROUNDS):
    beta=max(0.0,1.0-it*0.25)
    for k in range(40):
        if k<25: sd=k%5+1; vx=rng.choice([1.4,1.5,1.6,1.7,1.8])
        elif k<33: sd=1; vx=rng.choice(sweep)
        else: rs+=1; sd=rs; vx=rng.uniform(1.2,2.0)
        c=BASE.copy(); c[0]=vx
        p=Mix(beta, 0.0 if k%3==0 else NOISE*rng.uniform(0.3,1.5), rng)
        try: run_once(man,p,int(sd),model=model,command=c)
        except Stop: pass
        O+=p.O; A+=p.A
    X=torch.tensor(np.array(O)); Y=torch.tensor(np.array(A))
    for ep in range(600):
        idx=torch.randint(0,len(X),(2048,)); loss=((net(X[idx])-Y[idx])**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    if it>=3:
        f=f'/workspace/imitation/{TAG}_it{it}.onnx'; export(f)
        ev=evaluate(man,load_policy(f)); s=(ev.seeds_finished,-ev.score)
        print(f"it{it} data={len(X)} loss={loss.item():.4f} OFFICIAL {ev.score:.3f}s {ev.seeds_finished}/5 @{ev.command_vx} {time.time()-t0:.0f}s",flush=True)
        if s>(best[0],-best[1]): best=(ev.seeds_finished,ev.score,f); os.system(f'cp {f} /workspace/imitation/{TAG}_best.onnx'); print('BEST',best,flush=True)
    else: print(f"it{it} data={len(X)} loss={loss.item():.4f} {time.time()-t0:.0f}s",flush=True)
print('DONE',best)

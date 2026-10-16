import sys,json,numpy as np, mujoco
import arena.runner as R
from arena.manifest import load_event
from arena.course import compile_course
from arena.policy import load_policy
from arena.constants import CTRL_DT
man=load_event('hill-climb-1m5'); model=compile_course(man)
course=mujoco.mj_name2id(model,1,'course'); feetb=[mujoco.mj_name2id(model,1,n) for n in ('ankle_left','ankle_right')]
sites=[mujoco.mj_name2id(model,6,n) for n in ('left_foot','right_foot')]
REC=[]; orig=R._record
def rec(frame,data,adr):
    orig(frame,data,adr)
    c=[0,0]
    for i in range(data.ncon):
        g1,g2=data.contact[i].geom1,data.contact[i].geom2; b1,b2=model.geom_bodyid[g1],model.geom_bodyid[g2]
        for k,fb in enumerate(feetb):
            if {b1,b2}=={fb,course} or (fb in (b1,b2) and model.body_rootid[b1 if b2==fb else b2]==0): c[k]=1
    REC.append((data.qpos[:3].copy(),data.qpos[3:7].copy(),data.site_xpos[sites].copy(),c))
R._record=rec
class Wrap:
    def __init__(s,p): s.p=p; s.A=[]
    def act(s,o): a=s.p.act(o); s.A.append(np.asarray(a).reshape(-1).copy()); return a
out={}
for name,path,vx in [(a.split('=')[0],a.split('=')[1],float(a.split('=')[2])) for a in sys.argv[1:]]:
    res={}
    for seed in [1,2,3,4,5]:
        REC.clear(); w=Wrap(load_policy(path)); c=np.asarray(man.command,float).copy(); c[0]=vx
        r=R.run_once(man,w,seed,model=model,command=c)
        n=len(REC); settle=round(man.start.settle_s/CTRL_DT)+1
        P=np.array([x[0] for x in REC])[settle:]; Q=np.array([x[1] for x in REC])[settle:]; F=np.array([x[2] for x in REC])[settle:]; C=np.array([x[3] for x in REC])[settle:]
        A=np.array(w.A)[settle-1:]
        s8,c8=np.sin(np.radians(8)),np.cos(np.radians(8))
        along=(P[:,0]-P[0,0])*c8+(P[:,2]-P[0,2])*s8
        w_,x,y,z=Q.T; pitch=np.degrees(np.arcsin(np.clip(2*(w_*y-z*x),-1,1)))
        hgt=(P[:,2]-P[0,2])*c8-(P[:,0]-P[0,0])*s8   # height change normal to slope
        fh=(F[:,:,2]*c8-F[:,:,0]*s8); fh=fh-fh.min()   # foot height normal to slope
        both_air=float(((C[:,0]==0)&(C[:,1]==0)).mean())
        td=[np.where(np.diff(C[:,k])==1)[0] for k in (0,1)]
        steps=sum(len(t) for t in td); dur=len(P)*CTRL_DT
        v=np.diff(along)/CTRL_DT
        da=np.diff(A,axis=0)
        res[seed]=dict(status=r.status,t=r.time_s,steps=steps,step_hz=round(steps/dur,2),stride_m=round(float(along[-1])/max(steps,1),3),
            air_both=round(both_air,2),clear_cm=round(float(np.percentile(fh,95))*100,1),pitch_med=round(float(np.median(pitch)),1),pitch_max=round(float(pitch.max()),1),
            h_rel_cm=round(float(np.median(hgt))*100,1),v_prof=[round(float(x),2) for x in v[::10]],act_abs=round(float(np.abs(A).mean()),2),act_rate=round(float(np.abs(da).mean()),3),
            jrange=[round(float(x),2) for x in (A.max(0)-A.min(0))])
    out[name]=res
    s=res[3]; print(f"== {name} @{vx}: "+", ".join(f"s{k}:{v['status'][:2]}{v['t']}" for k,v in res.items()))
    for k in ['step_hz','stride_m','air_both','clear_cm','pitch_med','pitch_max','h_rel_cm','act_abs','act_rate','v_prof','jrange']: print(f"   {k}: {s[k]}")
json.dump(out,open('/workspace/hill/analysis.json','w'),default=str)

import glob,sys,numpy as np, mujoco
from arena.manifest import load_event
from arena.course import compile_course
from arena.constants import CTRL_DT
model=compile_course(load_event('roller-sprint-2m')); d=mujoco.MjData(model)
from arena.runner import JOINT_NAMES
jn=list(JOINT_NAMES); qadr=[int(model.joint(n).qposadr[0]) for n in jn]
tires=['tire','tire_2','tire_3','tire_4']; tid=[mujoco.mj_name2id(model,1,t) for t in tires]
for name in sys.argv[1:]:
    tr=np.load(glob.glob(f'/workspace/arena_compare/{name}/traces/*.npz')[0])['trace']
    print("==",name,len(tr),"steps; trace cols",tr.shape[1])
    rows=[]
    for k,f in enumerate(tr):
        d.qpos[:]=model.qpos0; d.qpos[0:7]=f[0:7]; d.qpos[qadr]=f[7:21]; mujoco.mj_kinematics(model,d)
        R=d.xmat[1].reshape(3,3); pitch=np.degrees(np.arcsin(-R[2,0]))
        tz=[d.xpos[b][2] for b in tid]; tx=[(R.T@(d.xpos[b]-d.xpos[1]))[0] for b in tid]
        rows.append((pitch,tz,tx))
    for k in list(range(0,len(tr),5))+[len(tr)-1]:
        p,tz,tx=rows[k]; print(f" t={k*CTRL_DT:.2f} pitch={p:5.1f} tireZ={np.round(tz,3).tolist()} tireXbody={np.round(tx,3).tolist()}")
    q=tr[:,7:21]; print(" joint mean/std:"); [print(f"   {jn[i]:22s} {q[:,i].mean():+.3f} {q[:,i].std():.3f}") for i in range(14)]
    tz=np.array([r[1] for r in rows]); P=np.array([r[0] for r in rows])
    print(" contact frac (z<0.02) [L_toe,L_heel,R_toe,R_heel]:", np.round((tz<0.02).mean(0),2).tolist(), " mean z:", np.round(tz.mean(0),3).tolist())
    print(" airborne frac (all >0.02):", round(float((tz.min(1)>0.02).mean()),2), " pitch mean/first0.3s:", round(P.mean(),1), round(P[:15].mean(),1))

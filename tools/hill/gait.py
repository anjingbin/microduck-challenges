import glob,sys,numpy as np, mujoco
from arena.manifest import load_event
from arena.course import compile_course
from arena.constants import CTRL_DT
m=load_event('roller-sprint-2m'); model=compile_course(m); d=mujoco.MjData(model)
qadr=[model.jnt_qposadr[j] for j in range(model.njnt) if model.jnt_type[j]!=0]
jn=[mujoco.mj_id2name(model,3,j) for j in range(model.njnt) if model.jnt_type[j]!=0][:14]
for name in sys.argv[1:]:
    tr=np.load(glob.glob(f'{name}/traces/*.npz')[0])['trace']
    n=len(tr); x=tr[:,0]-tr[0,0]; vx=np.diff(tr[:,0])/CTRL_DT
    xq,yq=tr[:,4],tr[:,5]; gz=-(1-2*(xq*xq+yq*yq)); tilt=np.degrees(np.arccos(np.clip(-gz,-1,1)))
    feetz=[];
    for f in tr:
        d.qpos[:]=model.qpos0; d.qpos[0:7]=f[0:7]; d.qpos[qadr[:14]]=f[7:21]; mujoco.mj_kinematics(model,d)
        zs=[d.xpos[b][2] for b in range(model.nbody) if 'tire' in (mujoco.mj_id2name(model,1,b) or '') or 'wheel' in (mujoco.mj_id2name(model,1,b) or '')]
        feetz.append(zs)
    feetz=np.array(feetz)
    q=tr[:,7:21]; amp=q.max(0)-q.min(0)
    top=np.argsort(-amp)[:4]
    print(f"== {name}: t={n*CTRL_DT:.2f}s mean_vx={x[-1]/(n*CTRL_DT):.2f} peak_vx={vx.max():.2f} vx@0.5s/1s={vx[min(24,len(vx)-1)]:.2f}/{vx[min(49,len(vx)-1)]:.2f}")
    print(f"   trunk z {tr[:,2].min():.3f}-{tr[:,2].max():.3f} (median {np.median(tr[:,2]):.3f}); tilt median/max {np.median(tilt):.0f}/{tilt.max():.0f} deg")
    if feetz.size: print(f"   wheel bodies {feetz.shape[1]}, max lift above min {np.round(feetz.max(0)-feetz.min(0),3).tolist()}")
    print("   biggest joint swings:", [(jn[i], round(float(amp[i]),2)) for i in top])
    print("   vx profile every 0.2s:", np.round(vx[::10],2).tolist())

import numpy as np, mujoco, mujoco_warp as mjw, warp as wp, arena.runner as R
from arena.manifest import load_event
from arena.course import compile_course
from arena.policy import load_policy
np.set_printoptions(precision=4, suppress=True, linewidth=200)
man = load_event('hill-climb-1m5'); model = compile_course(man)
snaps = []
orig = mujoco.mj_step
def step(m, d):
    if len(snaps) < 400: snaps.append((d.qpos.copy(), d.qvel.copy(), d.ctrl.copy(), m.dof_frictionloss.copy(), m.dof_damping.copy()))
    return orig(m, d)
R.mujoco.mj_step = step
c = np.asarray(man.command, float).copy(); c[0] = 0.6
R.run_once(man, load_policy('/workspace/hill/es2.onnx'), 1, model=model, command=c)
R.mujoco.mj_step = orig
wp.init()
for k in [5, 120, 200, 260, 280, 290]:
    qp, qv, u, fl, dmp = snaps[k]
    m = model; m.dof_frictionloss[:] = fl; m.dof_damping[:] = dmp
    d = mujoco.MjData(m); d.qpos[:] = qp; d.qvel[:] = qv; d.ctrl[:] = u; mujoco.mj_forward(m, d)
    ncon = d.ncon
    dc = mujoco.MjData(m); dc.qpos[:] = qp; dc.qvel[:] = qv; dc.ctrl[:] = u
    for _ in range(4): orig(m, dc)
    mw = mjw.put_model(m); dw = mjw.put_data(m, d, nconmax=200, njmax=600)
    for _ in range(4): mjw.step(mw, dw)
    qvw = dw.qvel.numpy()[0]
    mjw.forward(mw, dw0 := mjw.put_data(m, d, nconmax=200, njmax=600)) if False else None
    d2 = mujoco.MjData(m); d2.qpos[:] = qp; d2.qvel[:] = qv; d2.ctrl[:] = u; mujoco.mj_forward(m, d2)
    dw2 = mjw.put_data(m, d2, nconmax=200, njmax=600); mjw.forward(mw, dw2)
    nw = int(dw2.nacon.numpy()[0]) if hasattr(dw2, 'nacon') else -1
    cw = dw2.contact; 
    print('   C contacts', [(int(d2.contact[i].geom[0]), int(d2.contact[i].geom[1]), np.round(d2.contact[i].pos, 4).tolist(), round(float(d2.contact[i].dist), 5)) for i in range(d2.ncon)])
    print('   W contacts', nw, [(cw.geom.numpy()[i].tolist(), np.round(cw.pos.numpy()[i], 4).tolist(), round(float(cw.dist.numpy()[i]), 5)) for i in range(max(nw, 0))])
    print(f'snap {k} ncon {ncon}  |dqvel| trunk C {np.abs(dc.qvel[:6]-qv[:6]).sum():.4f}  C-vs-Warp trunk {np.abs(dc.qvel[:6]-qvw[:6]).max():.4f} joints {np.abs(dc.qvel[6:]-qvw[6:]).max():.4f}')

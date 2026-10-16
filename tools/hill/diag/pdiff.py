import sys, mujoco, numpy as np, pickle
def grab(m):
    d = {}
    nm = lambda t, i: mujoco.mj_id2name(m, t, i)
    for b in range(m.nbody):
        n = nm(mujoco.mjtObj.mjOBJ_BODY, b)
        if n is None: continue
        n = n.replace('robot/', '')
        d['body:'+n] = np.r_[m.body_mass[b], m.body_inertia[b], m.body_ipos[b], m.body_pos[b], m.body_quat[b]]
    for j in range(m.njnt):
        n = (nm(mujoco.mjtObj.mjOBJ_JOINT, j) or 'free').replace('robot/', '')
        dof = m.jnt_dofadr[j]
        d['jnt:'+n] = np.r_[m.jnt_range[j], m.dof_armature[dof], m.dof_damping[dof], m.dof_frictionloss[dof], m.jnt_stiffness[j], m.qpos0[m.jnt_qposadr[j]] if m.jnt_type[j]!=0 else 0]
    for a in range(m.nu):
        n = (nm(mujoco.mjtObj.mjOBJ_ACTUATOR, a) or str(a)).replace('robot/', '')
        d['act:'+n] = np.r_[m.actuator_gainprm[a][:3], m.actuator_biasprm[a][:3], m.actuator_ctrlrange[a], m.actuator_forcerange[a], m.actuator_trnid[a][0]]
    for s in range(m.nsensor):
        n = (nm(mujoco.mjtObj.mjOBJ_SENSOR, s) or str(s)).replace('robot/', '')
        d['sens:'+n] = np.r_[m.sensor_type[s], m.sensor_objtype[s], m.sensor_noise[s], m.sensor_cutoff[s]]
    for s in range(m.nsite):
        n = (nm(mujoco.mjtObj.mjOBJ_SITE, s) or str(s)).replace('robot/', '')
        d['site:'+n] = np.r_[m.site_pos[s], m.site_quat[s]]
    return d
if sys.argv[1] == 'arena':
    from arena.manifest import load_event; from arena.course import compile_course
    pickle.dump(grab(compile_course(load_event('hill-climb-1m5'))), open('/workspace/hill/diag/pa.pkl', 'wb'))
elif sys.argv[1] == 'mjlab':
    sys.path.insert(0, '/workspace/microduck_train/tools/hill'); import hill_exact as H
    env = H.make_env(H.make_cfg(1)); pickle.dump(grab(env.sim.mj_model), open('/workspace/hill/diag/pm.pkl', 'wb'))
else:
    a = pickle.load(open('/workspace/hill/diag/pa.pkl', 'rb')); m = pickle.load(open('/workspace/hill/diag/pm.pkl', 'rb'))
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    print('only arena', sorted(set(a) - set(m))); print('only mjlab', sorted(set(m) - set(a)))
    for k in sorted(set(a) & set(m)):
        if a[k].shape != m[k].shape or not np.allclose(a[k], m[k], atol=1e-4, rtol=1e-3): print(k, '\n  A', a[k], '\n  M', m[k])

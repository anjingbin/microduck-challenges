import sys, numpy as np, mujoco
np.set_printoptions(precision=4, suppress=True)
def dump(m, tag):
    o = m.opt
    print(tag, 'dt', o.timestep, 'integ', o.integrator, 'cone', o.cone, 'impratio', o.impratio, 'iters', o.iterations, 'ls', o.ls_iterations, 'tol', o.tolerance, 'solver', o.solver, 'noslip', o.noslip_iterations, 'disable', o.disableflags, 'enable', o.enableflags, 'grav', o.gravity)
    for g in range(m.ngeom):
        if m.geom_contype[g] or m.geom_conaffinity[g]:
            n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g); b = m.geom_bodyid[g]
            if b == 0 or 'foot' in str(n) or 'wheel' in str(n) or 'tire' in str(n) or g < 3 or n is None:
                print('  ', n, 'type', m.geom_type[g], 'body', b, 'fric', m.geom_friction[g], 'solref', m.geom_solref[g], 'solimp', m.geom_solimp[g], 'condim', m.geom_condim[g], 'margin', m.geom_margin[g], 'gap', m.geom_gap[g], 'prio', m.geom_priority[g], 'solmix', m.geom_solmix[g])
    print('  nbody', m.nbody, 'mass', m.body_mass.sum(), 'armature', m.dof_armature[6:9], 'dof_damp', m.dof_damping[6:9], 'frictionloss', m.dof_frictionloss[6:9])
    print('  pair', m.npair, 'exclude', m.nexclude)
if sys.argv[1] == 'arena':
    from arena.manifest import load_event; from arena.course import compile_course
    dump(compile_course(load_event('hill-climb-1m5')), 'ARENA')
else:
    sys.path.insert(0, '/workspace/microduck_train/tools/hill'); import hill_exact as H, os
    env = H.make_env(H.make_cfg(1)) if hasattr(H, 'make_cfg') else None
    dump(env.unwrapped.sim.mj_model if hasattr(env, 'unwrapped') else env.sim.mj_model, 'MJLAB')

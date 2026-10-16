import sys, mujoco, numpy as np
np.set_printoptions(precision=4, suppress=True)
def dump(m):
    for g in range(m.ngeom):
        b = m.geom_bodyid[g]
        if m.geom_contype[g] or m.geom_conaffinity[g]:
            print(g, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b), mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g), 'type', m.geom_type[g], 'ct/ca', m.geom_contype[g], m.geom_conaffinity[g], 'size', m.geom_size[g], 'pos', m.geom_pos[g], 'mesh', m.geom_dataid[g])
if sys.argv[1] == 'arena':
    from arena.manifest import load_event; from arena.course import compile_course
    dump(compile_course(load_event('hill-climb-1m5')))
else:
    sys.path.insert(0, '/workspace/microduck_train/tools/hill'); import hill_exact as H
    env = H.make_env(H.make_cfg(1)); dump(env.sim.mj_model)

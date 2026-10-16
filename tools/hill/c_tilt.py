"""MuJoCo C: Arena hill (rotated floor) vs flat floor + tilted gravity, same policy/seeds."""
import sys, math, numpy as np, mujoco
import arena.runner as R
from arena.manifest import load_event
from arena.course import compile_course
from arena.policy import load_policy
man = load_event('hill-climb-1m5'); th = math.radians(8)
def run(path, mode, vx=0.3):
    model = compile_course(man)
    if mode == 'tilt':
        b = mujoco.mj_name2id(model, 1, 'course'); model.body_quat[b] = [1, 0, 0, 0]
        model.opt.gravity[:] = [-9.81 * math.sin(th), 0, -9.81 * math.cos(th)]
        R._WORLD_DOWN = np.array([-math.sin(th), 0, -math.cos(th)])
        R.ramp_dir = lambda s: np.array([1.0, 0, 0])
        orig = R._perturb.__wrapped__ if hasattr(R._perturb, '__wrapped__') else R._perturb
        def pert(data, adr, manifest, seed):
            orig(data, adr, manifest, seed)
            qy = data.qpos[3:7].copy(); qp = np.array([math.cos(th/2), 0, math.sin(th/2), 0])
            out = np.zeros(4); mujoco.mju_mulQuat(out, qp, qy); data.qpos[3:7] = out
        pert.__wrapped__ = orig; R._perturb = pert
    else:
        R._WORLD_DOWN = np.array([0.0, 0, -1.0])
        import arena.course as C; R.ramp_dir = C.ramp_dir
        if hasattr(R._perturb, '__wrapped__'): R._perturb = R._perturb.__wrapped__
    pol = load_policy(path); c = np.asarray(man.command, float).copy(); c[0] = vx
    return [(r.status[:4], r.time_s) for r in (R.run_once(man, pol, s, model=model, command=c) for s in range(1, 6))]
for p in sys.argv[1:]:
    print(p.split('/')[-1], 'SLOPE', run(p, 'slope'), flush=True)
    print(p.split('/')[-1], 'TILT ', run(p, 'tilt'), flush=True)

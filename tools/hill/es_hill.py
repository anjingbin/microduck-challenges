"""ES fine-tune of our own dive1 policy weights directly on the Arena hill sim (CPU, free). Objective = Arena-style mean time."""
import sys, os, time, json, numpy as np, onnx
from onnx import numpy_helper as nh
from multiprocessing import Pool
SRC = sys.argv[1]; ITERS = int(sys.argv[2]); OUT = sys.argv[3]
SIG = float(os.environ.get('SIG', 0.004)); LR = float(os.environ.get('LR', 0.002)); POP = int(os.environ.get('POP', 16))
VXS = [float(v) for v in os.environ.get('VXS', '0.3,0.25,0.35').split(',')]
m = onnx.load(SRC); W = {i.name: nh.to_array(i).astype(np.float64) for i in m.graph.initializer}
KEYS = ['mlp.0.weight','mlp.0.bias','mlp.2.weight','mlp.2.bias','mlp.4.weight','mlp.4.bias','mlp.6.weight','mlp.6.bias']
shapes = [W[k].shape for k in KEYS]; sizes = [W[k].size for k in KEYS]
theta0 = np.concatenate([W[k].ravel() for k in KEYS]); scale = np.concatenate([np.full(W[k].size, np.std(W[k]) + 1e-3) for k in KEYS])
MEAN, DIV = W['obs_normalizer._mean'][0], W['onnx::Div_24'][0]
def unpack(th):
    out, o = [], 0
    for s, n in zip(shapes, sizes): out.append(th[o:o+n].reshape(s)); o += n
    return out
elu = lambda x: np.where(x > 0, x, np.expm1(np.minimum(x, 0)))
class Stop(Exception): pass
class Pol:
    def __init__(s, th): s.p = unpack(th); s.n = 0
    def act(s, obs):
        s.n += 1
        if s.n > 175: raise Stop
        x = (np.asarray(obs, np.float64) - MEAN) / DIV; p = s.p
        x = elu(p[0] @ x + p[1]); x = elu(p[2] @ x + p[3]); x = elu(p[4] @ x + p[5])
        return (p[6] @ x + p[7]).astype(np.float32)
_G = {}
def _init():
    from arena.manifest import load_event
    from arena.course import compile_course
    _G['man'] = load_event('hill-climb-1m5'); _G['model'] = compile_course(_G['man'])
def run(args):
    th, seed, vx = args
    from arena.runner import run_once
    man = _G['man']; c = np.asarray(man.command, float).copy(); c[0] = vx
    try: r = run_once(man, Pol(th), seed, model=_G['model'], command=c)
    except Stop: return 60.0
    return r.time_s if r.status == 'finish' else 60.0
def fitness(pool, ths):
    jobs = [(th, s, vx) for th in ths for vx in VXS for s in (1, 2, 3, 4, 5)]
    t = np.array(pool.map(run, jobs, chunksize=4)).reshape(len(ths), len(VXS), 5)
    best = t.mean(2).min(1)                 # arena-like: best vx's 5-seed mean
    soft = np.minimum(t, 3.0).mean((1, 2))  # smooth shaping: falls capped at 3 s
    return best, soft, t
if __name__ == '__main__':
    rng = np.random.default_rng(0); theta = theta0.copy(); log = []
    with Pool(8, initializer=_init) as pool:
        b0, s0, t0 = fitness(pool, [theta]); print('start', b0[0], s0[0], flush=True); bestv = (b0[0], theta.copy())
        for it in range(ITERS):
            eps = rng.standard_normal((POP, theta.size)) * scale
            ths = [theta + SIG * e for e in eps] + [theta - SIG * e for e in eps]
            b, s, _ = fitness(pool, ths)
            f = -s; f = (f.argsort().argsort() / (len(f) - 1) - 0.5)
            g = (f[:POP, None] - f[POP:, None]) * eps
            theta = theta + LR * g.mean(0) / SIG * 1.0
            bc, sc, tc = fitness(pool, [theta])
            print(f'it{it} cur best={bc[0]:.3f} soft={sc[0]:.3f} fin={int((tc<60).sum())}/{tc.size} popbest={b.min():.3f}', flush=True)
            if bc[0] < bestv[0]: bestv = (bc[0], theta.copy())
    p = unpack(bestv[1])
    for k, v in zip(KEYS, p):
        for i in m.graph.initializer:
            if i.name == k: i.CopyFrom(nh.from_array(v.astype(np.float32), k))
    onnx.save(m, OUT); print('saved', OUT, 'best', bestv[0])

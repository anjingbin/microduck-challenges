"""OpenAI-style evolution strategy on a policy's MLP weights, scored by the Arena runner.

Antithetic Gaussian perturbations scaled per tensor by mean |w|, centred-rank fitness, Adam
step. The objective is the sub-step mean finish time over the event's seeds at one commanded
speed (a fall or timeout counts the time limit), so a gain shows before it crosses a whole step.
Every --eval-every generations the current mean is exported and scored with the full evaluate().

    .venv/bin/python remix/sprint-nb/es_weights.py SRC.onnx OUTDIR --event sprint-2m --vx 0.5
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "hill-so"))
import hc_weights as hw  # noqa: E402  (remix/hill-so: weight loading, perturbation, Arena scoring)
from arena.evaluate import evaluate
from arena.policy import load_policy

_G = {}


def flat(params):
    return np.concatenate([params[k].ravel() for k in hw.PARAMS]).astype(np.float64)


def unflat(vec, like):
    out, i = {}, 0
    for k in hw.PARAMS:
        n = like[k].size
        out[k] = vec[i:i + n].reshape(like[k].shape).astype(np.float32)
        i += n
    return out


def _init(src, event, vx, like, scale):
    hw._init(src, event, vx)
    _G.update(like=like, scale=scale)


def _task(args):
    theta, sigma, seed, sign = args
    eps = np.random.default_rng(seed).standard_normal(theta.size)
    b, o, f = hw.score(unflat(theta + sign * sigma * _G["scale"] * eps, _G["like"]))
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("outdir")
    ap.add_argument("--event", default="sprint-2m")
    ap.add_argument("--vx", type=float, default=0.5)
    ap.add_argument("--sigma", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--pairs", type=int, default=16)
    ap.add_argument("--gens", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=25)
    a = ap.parse_args()

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    like = hw.load_params(a.src)
    theta = flat(like)
    scale = np.concatenate([np.full(like[k].size, float(np.abs(like[k]).mean())) for k in hw.PARAMS])
    hw._init(a.src, a.event, a.vx)
    print(f"start: {hw.score(like)}", flush=True)
    pool = mp.Pool(a.workers, initializer=_init, initargs=(a.src, a.event, a.vx, like, scale))
    m, v, b1, b2 = np.zeros_like(theta), np.zeros_like(theta), 0.9, 0.999
    log = open(out / "log.jsonl", "a")
    t0, best = time.time(), None
    for gen in range(a.gens):
        seeds = [a.seed * 10_000_000 + gen * 1000 + j for j in range(a.pairs)]
        jobs = [(theta, a.sigma, s, sg) for s in seeds for sg in (1, -1)]
        objs = np.array(pool.map(_task, jobs)).reshape(a.pairs, 2)
        ranks = np.empty(objs.size)
        ranks[np.argsort(objs.ravel())] = np.linspace(0.5, -0.5, objs.size)  # lower time -> higher
        ranks = ranks.reshape(a.pairs, 2)
        g = np.zeros_like(theta)
        for j, s in enumerate(seeds):
            g += (ranks[j, 0] - ranks[j, 1]) * np.random.default_rng(s).standard_normal(theta.size)
        g = g / (2 * a.pairs * a.sigma)  # ascent direction in eps space
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        step = a.lr * (m / (1 - b1 ** (gen + 1))) / (np.sqrt(v / (1 - b2 ** (gen + 1))) + 1e-8)
        theta = theta + scale * step
        log.write(json.dumps({"gen": gen, "t": round(time.time() - t0, 1), "pop_min": float(objs.min()),
                              "pop_mean": float(objs.mean())}) + "\n")
        log.flush()
        if gen % 5 == 0:
            print(f"gen {gen} t={time.time()-t0:.0f}s pop min {objs.min():.4f} mean {objs.mean():.4f}", flush=True)
        if (gen + 1) % a.eval_every == 0:
            p = out / f"es_{gen+1}.onnx"
            p.write_bytes(hw.to_bytes(a.src, unflat(theta, like)))
            ev = evaluate(hw._G["man"], load_policy(p))
            print(f"gen {gen+1} mean policy: full evaluate {ev.score:.3f}@{ev.command_vx} {ev.seeds_finished}/5",
                  flush=True)


if __name__ == "__main__":
    main()

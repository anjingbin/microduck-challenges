"""Elitist hill climb on the weights of a frozen-architecture Arena policy.

Each generation perturbs the current best weights with Gaussian noise scaled by each
tensor's mean |w|, scores every candidate with the Arena's own runner on the event's
official seeds at one commanded speed, and keeps the best if it improves
(whole-step board first, sub-step finish time as tie-breaker). A candidate whose board
reaches --save-at is exported and re-scored with the full evaluate() (sweep included).

    .venv/bin/python -I hc_weights.py SRC.onnx OUTDIR --event hill-climb-1m5 --vx 0.30
"""

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

from arena.course import compile_course
from arena.evaluate import evaluate
from arena.manifest import load_event
from arena.policy import load_policy
from arena.runner import run_once

PARAMS = [f"mlp.{i}.{k}" for i in (0, 2, 4, 6) for k in ("weight", "bias")]
_G = {}


def load_params(src):
    m = onnx.load(src)
    return {i.name: numpy_helper.to_array(i).copy() for i in m.graph.initializer if i.name in PARAMS}


def to_bytes(src, params):
    m = onnx.load(src)
    for i in m.graph.initializer:
        if i.name in params:
            i.CopyFrom(numpy_helper.from_array(params[i.name].astype(np.float32), i.name))
    return m.SerializeToString()


class Sess:
    def __init__(self, data):
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.intra_op_num_threads = 1
        self.s = ort.InferenceSession(data, opts, providers=["CPUExecutionProvider"])
        self.i = self.s.get_inputs()[0].name

    def act(self, obs):
        x = obs.reshape(1, -1).astype(np.float32)
        return self.s.run(None, {self.i: x})[0].reshape(-1).astype(np.float32)


def _init(src, event, vx):
    man = load_event(event)
    _G.update(src=src, man=man, model=compile_course(man), cmd=man.command_at(vx))


def score(params):
    man = _G["man"]
    pol = Sess(to_bytes(_G["src"], params))
    L, limit = man.course.length_m, man.time_limit_s
    times, objs, fin = [], [], 0
    for seed in man.seeds:
        o = run_once(man, pol, seed, model=_G["model"], command=_G["cmd"])
        if o.status == "finish":
            fin += 1
            times.append(o.time_s)
            objs.append(o.time_s - (o.progress_m - L) / (o.progress_m / o.time_s))
        else:
            times.append(limit)
            objs.append(limit)
    return float(np.mean(times)), float(np.mean(objs)), fin


def _task(args):
    base, sigma, seed = args
    rng = np.random.default_rng(seed)
    cand = {k: v + (sigma * float(np.abs(v).mean()) * rng.standard_normal(v.shape)).astype(np.float32)
            for k, v in base.items()}
    return seed, score(cand)


def perturbed(base, sigma, seed):
    rng = np.random.default_rng(seed)
    return {k: v + (sigma * float(np.abs(v).mean()) * rng.standard_normal(v.shape)).astype(np.float32)
            for k, v in base.items()}


def key(b, o, f):
    return (-f, round(b, 4), o)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("outdir")
    ap.add_argument("--event", default="hill-climb-1m5")
    ap.add_argument("--vx", type=float, default=0.30)
    ap.add_argument("--sigma", type=float, default=0.01)
    ap.add_argument("--lam", type=int, default=16)
    ap.add_argument("--gens", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--save-at", type=float, default=0.784)
    a = ap.parse_args()

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    _init(a.src, a.event, a.vx)
    best_p = load_params(a.src)
    best = score(best_p)
    print(f"start: board {best[0]:.3f} obj {best[1]:.5f} {best[2]}/5", flush=True)
    saved = a.save_at + 0.004
    sigma = a.sigma
    pool = mp.Pool(a.workers, initializer=_init, initargs=(a.src, a.event, a.vx))
    log = open(out / "log.jsonl", "a")
    t0, fails = time.time(), 0
    for gen in range(a.gens):
        seeds = [a.seed * 10_000_000 + gen * 1000 + j for j in range(a.lam)]
        res = pool.map(_task, [(best_p, sigma, s) for s in seeds])
        s_best, r_best = min(res, key=lambda r: key(*r[1]))
        if key(*r_best) < key(*best):
            best_p, best, fails = perturbed(best_p, sigma, s_best), r_best, 0
        else:
            fails += 1
            if fails >= 15:  # stuck: widen a little, then come back
                sigma = min(sigma * 1.5, 4 * a.sigma)
                fails = 0
        log.write(json.dumps({"gen": gen, "t": round(time.time() - t0, 1), "best_board": best[0],
                              "best_obj": best[1], "fin": best[2], "sigma": sigma}) + "\n")
        log.flush()
        if gen % 10 == 0:
            print(f"gen {gen} t={time.time()-t0:.0f}s best {best[0]:.3f}/{best[1]:.5f} {best[2]}/5 sigma {sigma:.4f}",
                  flush=True)
        if best[2] == 5 and round(best[0], 4) < round(saved, 4):
            p = out / f"cand_{best[0]:.3f}.onnx"
            p.write_bytes(to_bytes(a.src, best_p))
            ev = evaluate(_G["man"], load_policy(p))
            print(f"gen {gen} NEW board {best[0]:.3f} -> full evaluate {ev.score:.3f}@{ev.command_vx} "
                  f"{ev.seeds_finished}/5 -> {p.name}", flush=True)
            saved = best[0]


if __name__ == "__main__":
    main()

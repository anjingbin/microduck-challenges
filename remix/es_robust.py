"""ES on a policy's MLP weights against many seeds, for a policy that is fast AND stable off the official seeds.

Same estimator as sprint-nb/es_weights.py (antithetic, centred ranks, Adam, per-tensor noise
scale), but each candidate races --train-seeds (not the Arena's official 1-5); a fall or a
timeout costs the event's time limit, so the search cannot buy speed with falls. Every
--eval-every generations the mean policy is exported and raced on --val-seeds (never trained
on) and on the official seeds.

    .venv/bin/python remix/es_robust.py BASE.onnx OUT --event hill-climb-1m5 --vx 0.30
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "hill-so"))
import hc_weights as hw  # noqa: E402
from arena.course import compile_course  # noqa: E402
from arena.manifest import load_event  # noqa: E402
from arena.policy import load_policy  # noqa: E402
from arena.runner import run_once  # noqa: E402

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


def race(pol, seeds):
    """Mean sub-step finish time (fall/timeout = time limit) and finishes over seeds."""
    man, model, cmd = _G["man"], _G["model"], _G["cmd"]
    L, limit = man.course.length_m, man.time_limit_s
    objs, fin = [], 0
    for s in seeds:
        o = run_once(man, pol, s, model=model, command=cmd)
        if o.status == "finish":
            fin += 1
            objs.append(o.time_s - (o.progress_m - L) / (o.progress_m / o.time_s))
        else:
            objs.append(limit)
    return float(np.mean(objs)), fin


def _init(src, event, vx, seeds, like, scale):
    man = load_event(event)
    _G.update(src=src, man=man, model=compile_course(man), cmd=man.command_at(vx), seeds=seeds,
              like=like, scale=scale)


def _task(args):
    theta, sigma, seed, sign = args
    eps = np.random.default_rng(seed).standard_normal(theta.size)
    pol = hw.Sess(hw.to_bytes(_G["src"], unflat(theta + sign * sigma * _G["scale"] * eps, _G["like"])))
    return race(pol, _G["seeds"])[0]


def seeds_arg(s):
    a, b = s.split("-")
    return list(range(int(a), int(b) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("outdir")
    ap.add_argument("--event", required=True)
    ap.add_argument("--vx", type=float, required=True)
    ap.add_argument("--train-seeds", type=seeds_arg, default=seeds_arg("6-15"))
    ap.add_argument("--val-seeds", type=seeds_arg, default=seeds_arg("16-35"))
    ap.add_argument("--sigma", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--pairs", type=int, default=12)
    ap.add_argument("--gens", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=20)
    a = ap.parse_args()

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    like = hw.load_params(a.src)
    theta = flat(like)
    scale = np.concatenate([np.full(like[k].size, float(np.abs(like[k]).mean())) for k in hw.PARAMS])
    _init(a.src, a.event, a.vx, a.train_seeds, like, scale)

    def report(tag, path):
        pol = load_policy(path)
        t_obj, t_fin = race(pol, a.train_seeds)
        v_obj, v_fin = race(pol, a.val_seeds)
        o_obj, o_fin = race(pol, _G["man"].seeds)
        print(f"{tag}: train {t_obj:.3f} ({t_fin}/{len(a.train_seeds)}) | val {v_obj:.3f} "
              f"({v_fin}/{len(a.val_seeds)}) | official {o_obj:.3f} ({o_fin}/5)", flush=True)

    report("start", Path(a.src))
    pool = mp.Pool(a.workers, initializer=_init,
                   initargs=(a.src, a.event, a.vx, a.train_seeds, like, scale))
    m, v, b1, b2 = np.zeros_like(theta), np.zeros_like(theta), 0.9, 0.999
    log = open(out / "log.jsonl", "a")
    t0 = time.time()
    for gen in range(a.gens):
        seeds = [a.seed * 10_000_000 + gen * 1000 + j for j in range(a.pairs)]
        jobs = [(theta, a.sigma, s, sg) for s in seeds for sg in (1, -1)]
        objs = np.array(pool.map(_task, jobs)).reshape(a.pairs, 2)
        ranks = np.empty(objs.size)
        ranks[np.argsort(objs.ravel())] = np.linspace(0.5, -0.5, objs.size)
        ranks = ranks.reshape(a.pairs, 2)
        g = np.zeros_like(theta)
        for j, s in enumerate(seeds):
            g += (ranks[j, 0] - ranks[j, 1]) * np.random.default_rng(s).standard_normal(theta.size)
        g = g / (2 * a.pairs * a.sigma)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        theta = theta + scale * a.lr * (m / (1 - b1 ** (gen + 1))) / (np.sqrt(v / (1 - b2 ** (gen + 1))) + 1e-8)
        log.write(json.dumps({"gen": gen, "t": round(time.time() - t0, 1), "pop_min": float(objs.min()),
                              "pop_mean": float(objs.mean())}) + "\n")
        log.flush()
        if (gen + 1) % a.eval_every == 0:
            p = out / f"es_{gen+1}.onnx"
            p.write_bytes(hw.to_bytes(a.src, unflat(theta, like)))
            report(f"gen {gen+1} t={time.time()-t0:.0f}s", p)


if __name__ == "__main__":
    main()

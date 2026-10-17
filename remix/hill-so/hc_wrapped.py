"""Weight hill climb of the network inside a start-offset wrapper (so_hill.py).

The offset table stays fixed (from a so_hill run's best.npy); hc_weights.py perturbs the
inner network's weights; candidates are scored through so_hill.Wrapped, exactly as the
exported graph computes them. A candidate reaching --save-at is exported and re-scored
with the full evaluate().

    .venv/bin/python -I hc_wrapped.py SRC.onnx BEST.npy OUTDIR --k-end 65 --every 3
"""

import argparse
import json
import multiprocessing as mp
import tempfile
import time
from pathlib import Path

import numpy as np

import hc_weights as hw
import so_hill as so
from arena.evaluate import evaluate
from arena.policy import load_policy

_G = {}


def _init(src, tab, event, vx):
    so._init(src, so.np.arange(1), vx)  # sets man/model/cmd for so_hill
    _G.update(src=src, tab=tab)


def score(params):
    data = hw.to_bytes(_G["src"], params)
    man = so._G["man"]
    pol = so.Wrapped(data, _G["tab"])
    L, limit = man.course.length_m, man.time_limit_s
    times, objs, fin = [], [], 0
    for seed in man.seeds:
        o = so.run_once(man, pol, seed, model=so._G["model"], command=so._G["cmd"])
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
    return seed, score(hw.perturbed(base, sigma, seed))


def export(src, params, tab, path):
    with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
        f.write(hw.to_bytes(src, params))
        f.flush()
        so.export(f.name, tab, path, "start offsets by CMA-ES + inner weights by elitist hill climb")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("best_npy")
    ap.add_argument("outdir")
    ap.add_argument("--vx", type=float, default=0.30)
    ap.add_argument("--k-end", type=int, default=65)
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--sigma", type=float, default=0.0003)
    ap.add_argument("--lam", type=int, default=16)
    ap.add_argument("--gens", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--save-at", type=float, default=0.784)
    a = ap.parse_args()

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    knots = np.arange(so.K_FIRST, a.k_end + 1, a.every)
    tab = so.table(np.load(a.best_npy), knots)
    _init(a.src, tab, None, a.vx)
    best_p = hw.load_params(a.src)
    best = score(best_p)
    print(f"start: board {best[0]:.3f} obj {best[1]:.5f} {best[2]}/5", flush=True)
    saved = a.save_at + 0.004
    sigma = a.sigma
    pool = mp.Pool(a.workers, initializer=_init, initargs=(a.src, tab, None, a.vx))
    log = open(out / "log.jsonl", "a")
    t0, fails = time.time(), 0
    for gen in range(a.gens):
        seeds = [a.seed * 10_000_000 + gen * 1000 + j for j in range(a.lam)]
        res = pool.map(_task, [(best_p, sigma, s) for s in seeds])
        s_best, r_best = min(res, key=lambda r: hw.key(*r[1]))
        if hw.key(*r_best) < hw.key(*best):
            best_p, best, fails = hw.perturbed(best_p, sigma, s_best), r_best, 0
        else:
            fails += 1
            if fails >= 15:
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
            export(a.src, best_p, tab, p)
            np.savez(out / f"cand_{best[0]:.3f}.npz", **best_p)
            ev = evaluate(so._G["man"], load_policy(p))
            print(f"gen {gen} NEW board {best[0]:.3f} -> full evaluate {ev.score:.3f}@{ev.command_vx} "
                  f"{ev.seeds_finished}/5 -> {p.name}", flush=True)
            saved = best[0]


if __name__ == "__main__":
    main()

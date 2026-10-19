"""Elitist hill climb of an Arena policy's weights on the roller sprint's official seeds.

Each generation perturbs the elite's MLP weights (Gaussian, scaled to each tensor's std),
scores the candidates at one commanded speed on the event's five seeds, and keeps the best
if it beats the elite. Objective: mean over seeds of the finish time minus the overshoot past
the line divided by the run's mean speed (a sub-step tie-breaker; a fall counts the time
limit). A new elite is confirmed with the Arena's full evaluate() and saved.

    .venv/bin/python logs/remix/hillclimb.py SRC.onnx OUTDIR [--vx 0.7] [--sigmas 0.002,0.005,0.01,0.02]
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

from arena.constants import NUM_JOINTS, OBS_SIZE
from arena.course import compile_course
from arena.evaluate import evaluate
from arena.manifest import load_event
from arena.runner import run_once

EVENT = "roller-sprint-2m"
PARAMS = [f"mlp.{i}.{k}" for i in (0, 2, 4, 6) for k in ("weight", "bias")]


class OrtPolicy:
    def __init__(self, model_bytes: bytes):
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.intra_op_num_threads = 1
        self.s = ort.InferenceSession(model_bytes, opts, providers=["CPUExecutionProvider"])
        self.i = self.s.get_inputs()[0].name
        self.o = self.s.get_outputs()[0].name

    def act(self, obs):
        out = self.s.run([self.o], {self.i: obs.reshape(1, OBS_SIZE).astype(np.float32)})[0]
        return np.asarray(out, dtype=np.float32).reshape(NUM_JOINTS)


def build(base: onnx.ModelProto, weights: dict) -> bytes:
    m = onnx.ModelProto()
    m.CopyFrom(base)
    for init in m.graph.initializer:
        if init.name in weights:
            init.CopyFrom(numpy_helper.from_array(weights[init.name].astype(np.float32), init.name))
    return m.SerializeToString()


_G = {}


def _init(src: str, vx: float):
    _G["base"] = onnx.load(src)
    _G["manifest"] = load_event(EVENT)
    _G["model"] = compile_course(_G["manifest"])
    _G["command"] = _G["manifest"].command_at(vx)


def score(weights: dict) -> tuple[float, float, int]:
    """(objective, board mean at this speed, seeds finished)."""
    man = _G["manifest"]
    pol = OrtPolicy(build(_G["base"], weights))
    L, limit = man.course.length_m, man.time_limit_s
    objs, times, fin = [], [], 0
    for seed in man.seeds:
        o = run_once(man, pol, seed, model=_G["model"], command=_G["command"])
        if o.status == "finish":
            fin += 1
            times.append(o.time_s)
            objs.append(o.time_s - (o.progress_m - L) / (o.progress_m / o.time_s))
        else:
            times.append(limit)
            objs.append(limit)
    return float(np.mean(objs)), float(np.mean(times)), fin


def _score_task(args):
    seed, weights = args
    return seed, score(weights)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("outdir")
    ap.add_argument("--vx", type=float, default=0.7)
    ap.add_argument("--sigmas", default="0.002,0.005,0.01,0.02",
                    help="relative to each tensor's std; candidates cycle through them")
    ap.add_argument("--pop", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--generations", type=int, default=100000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    base = onnx.load(a.src)
    inits = {i.name: numpy_helper.to_array(i).copy() for i in base.graph.initializer}
    elite = {k: inits[k].astype(np.float64) for k in PARAMS}
    scale = {k: float(elite[k].std()) for k in PARAMS}
    rng = np.random.default_rng(a.seed)

    _init(a.src, a.vx)
    best_obj, best_board, best_fin = score(elite)
    best_full = evaluate(_G["manifest"], OrtPolicy(build(base, elite))).score
    log = open(out / "log.jsonl", "a")
    print(f"start obj {best_obj:.5f} board@{a.vx} {best_board:.3f} {best_fin}/5 full {best_full:.3f}", flush=True)
    sigmas = [float(x) for x in a.sigmas.split(",")]
    pool = mp.Pool(a.workers, initializer=_init, initargs=(a.src, a.vx))
    t0 = time.time()
    for gen in range(a.generations):
        seeds = rng.integers(0, 2**31, a.pop)
        cands = {}
        for j, s in enumerate(seeds):
            r = np.random.default_rng(int(s))
            sg = sigmas[j % len(sigmas)]
            cands[int(s)] = {k: elite[k] + sg * scale[k] * r.standard_normal(elite[k].shape) for k in PARAMS}
        results = pool.map(_score_task, [(s, w) for s, w in cands.items()])
        # The board's own time first (whole steps), the sub-step tie-breaker second.
        ok = [(round(board, 4), obj, s) for s, (obj, board, fin) in results if fin == 5]
        improved = False
        if ok:
            board, obj, s = min(ok)
            if (board, obj) < (round(best_board, 4), best_obj):
                elite, best_obj, best_board = cands[s], obj, board
                improved = True
        rec = {"gen": gen, "t": round(time.time() - t0, 1), "best_obj": best_obj,
               "best_board": best_board, "improved": improved,
               "finished_frac": len(ok) / a.pop}
        if improved:
            ev = evaluate(_G["manifest"], OrtPolicy(build(base, elite)))
            rec.update(full=ev.score, full_vx=ev.command_vx, full_fin=ev.seeds_finished)
            Path(out / "elite.onnx").write_bytes(build(base, elite))
            if ev.seeds_finished == 5 and ev.score < best_full:
                best_full = ev.score
                Path(out / f"best_{ev.score:.3f}.onnx").write_bytes(build(base, elite))
            print(f"gen {gen} t={rec['t']}s obj {best_obj:.5f} board@{a.vx} {best_board:.3f} "
                  f"full {ev.score:.3f}@{ev.command_vx} {ev.seeds_finished}/5", flush=True)
        elif gen % 20 == 0:
            print(f"gen {gen} t={rec['t']}s obj {best_obj:.5f} board {best_board:.3f} finished {len(ok)}/{a.pop}", flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()


if __name__ == "__main__":
    main()

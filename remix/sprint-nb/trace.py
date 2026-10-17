"""Per-step position, height and pitch of sprint-2m runs, seed 1: trace.py A.onnx VX [B.onnx VX ...]."""
import sys, numpy as np
from arena.manifest import load_event
from arena.course import compile_course
from arena.policy import load_policy
from arena.runner import run_once
man = load_event("sprint-2m"); model = compile_course(man)
for path, vx in zip(sys.argv[1::2], sys.argv[2::2]):
    pol = load_policy(path); vx = float(vx)
    o = run_once(man, pol, 1, model=model, command=man.command_at(vx))
    t = o.trace
    x = t[:, 0] - t[0, 0]; z = t[:, 2]
    w, qx, qy, qz = t[:, 3], t[:, 4], t[:, 5], t[:, 6]
    pitch = np.degrees(np.arcsin(np.clip(2 * (w * qy - qz * qx), -1, 1)))
    print(f"== {path.split('/')[-2]} vx {vx}: {o.status} {o.time_s} s, {len(t)} steps")
    for k in range(0, len(t), 5):
        print(f"  t={0.02*(k+1):.2f} x={x[k]:.3f} z={z[k]:.3f} pitch={pitch[k]:+.0f}")
    print(f"  last: x={x[-1]:.3f} z={z[-1]:.3f} pitch={pitch[-1]:+.0f}  start z={z[0]:.3f}")

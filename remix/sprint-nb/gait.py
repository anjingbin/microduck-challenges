"""Gait comparison on sprint-2m, seed 1: gait.py NAME A.onnx VX [NAME B.onnx VX ...].

Speed by quarter, stride period and length, trunk height, pitch and roll, joint ranges.
"""
import sys
import numpy as np
from arena.constants import JOINT_NAMES
from arena.course import compile_course
from arena.manifest import load_event
from arena.policy import load_policy
from arena.runner import run_once

man = load_event("sprint-2m")
model = compile_course(man)
L = ["l_hip_yaw", "l_hip_roll", "l_hip_pitch", "l_knee", "l_ankle"]
for name, path, vx in zip(sys.argv[1::3], sys.argv[2::3], sys.argv[3::3]):
    o = run_once(man, load_policy(path), 1, model=model, command=man.command_at(float(vx)))
    t = o.trace
    x, z = t[:, 0] - t[0, 0], t[:, 2]
    w, qx, qy, qz = t[:, 3:7].T
    pitch = np.degrees(np.arcsin(np.clip(2 * (w * qy - qz * qx), -1, 1)))
    roll = np.degrees(np.arctan2(2 * (w * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy)))
    q = t[:, 7:]
    v = np.diff(x) / 0.02
    hp = q[:, 2] - q[:, 2].mean()  # left hip pitch
    zc = np.where(np.diff(np.sign(hp[5:])) > 0)[0]  # rising zero crossings = one stride each
    stride_t = np.diff(zc).mean() * 0.02 if len(zc) > 1 else float("nan")
    n = len(t)
    print(f"== {name}: {o.status} {o.time_s} s ({n} steps) at {vx} m/s")
    print(f"  speed m/s by quarter: " + " ".join(f"{v[i*len(v)//4:(i+1)*len(v)//4].mean():.2f}" for i in range(4))
          + f" | first 0.2 s {x[9]:.2f} m")
    print(f"  stride period {stride_t:.3f} s -> {1/stride_t:.2f} strides/s, stride length {np.mean(v[10:])*stride_t:.3f} m")
    print(f"  trunk z mean {z.mean():.3f} (min {z.min():.3f}); pitch mean {pitch.mean():+.1f} last {pitch[-1]:+.0f}; "
          f"roll range {roll.min():+.0f}..{roll.max():+.0f}")
    amp = q[10:].max(0) - q[10:].min(0)
    mean = q[10:].mean(0)
    print("  joint      " + " ".join(f"{j[:13]:>13}" for j in JOINT_NAMES[:5]) + "  neck/head mean")
    print("  range rad  " + " ".join(f"{a:13.2f}" for a in amp[:5]) + "  " + " ".join(f"{m:+.2f}" for m in mean[5:9]))
    print("  mean rad   " + " ".join(f"{m:+13.2f}" for m in mean[:5]))
    print("  R range    " + " ".join(f"{a:13.2f}" for a in amp[9:14]))

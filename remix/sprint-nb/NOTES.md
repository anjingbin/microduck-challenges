# bubu sprint nb: a new sprint-2m base trained from scratch, then ES on the Arena scorer

Published as [dadudu/microduck-bubu-sprint](https://huggingface.co/dadudu/microduck-bubu-sprint)
`bubu-nb4-es2`, entered on `sprint-2m`: **0.952 s, 5/5 at 0.30 m/s** (run `01M56230N6HRHM18C0AHQ1`),
32nd of 99 entries, 10th of 25 players on 2026-10-17. Own lineage, no other entry's weights.

`publish/policy.onnx` (sha256 `f98082ff…`) is nb4 checkpoint 850 (1.328 s), improved by
`es_weights.py` (sigma 0.02, lr 0.01, ~1300 generations, 0.968 s, entered as `bubu-nb4-es`), then a
finer pass (sigma 0.005, lr 0.003, generation 425). Local seeds: 0.96 / 0.94 / 0.94 / 0.96 / 0.96.

## Training runs (HF Jobs, l4x1, `src/microduck_challenges/sprint_2m/`)

Best checkpoint by local `arena score`; every 50 iterations scored with `score_run.sh`.

| Run | Commit | Start | Change | Best | 5/5 checkpoints |
|---|---|---|---|---|---|
| nb1 | 46c48b1 | scratch, 1000 it | forward_speed w5, finish bonus 30, dive near finish, no pushes, no standing envs, upright 1.0 std² 0.1, air_time 0.05-0.2 s | 900: 1.528 s | most |
| nb2 | 46c48b1 | nb1 999, 1500 it | same | 800: 1.464 s | ~half |
| nb3 | 614edf9 | nb2 800 | action_rate -0.05, finish 60 | 150: 1.536 s | refuted: ~6x jerkier, slower |
| nb4 | fac88cc | nb2 800 | free a long rolling stride: air_time 0.10-0.35 s, pose 0, upright 0.5 std² 0.3, body_ang_vel -0.01, head_pose_tracking 0.5, no head bias | 850: 1.328 s | 5 of 21 |
| nb5 | 6cd3884 | nb4 850 | forward_speed along the lane, heading_hold std 1.0 | none (1.08-1.16 s finishes, one fall each) | 0 of 21 |
| nb6 | 1cf33b8 | nb5 650 | lane_hold w0.2 std 0.3 m | 600/650: 1.140 s | 4 of 21 |

Spent about $4.8 of an $8 budget. Warm starts: the checkpoint was staged (not committed: LFS
objects cannot be pushed to this public fork) for the job's source snapshot, with
`MICRODUCK_WARM_START=1 ... --agent.resume True --agent.load-run warm_<run> --agent.load-checkpoint model_<N>.pt`;
the checkpoints stay in `dadudu/mjlab-sprint2m-microduck-<run>`.

## ES (`es_weights.py`, local CPU, free)

| Start | Base | After ES |
|---|---|---|
| fast10 (bubu-sprint-fast) | 1.204 s | 1.008 s |
| nb1 900 | 1.528 s | 1.192 s (stopped at 115 generations) |
| nb2 800 | 1.464 s | 0.992 s |
| nb4 850 | 1.328 s | 0.968 s, then 0.952 s with sigma 0.005 |
| nb5 650 | fell once | 1.016 s |
| nb6 600 | 1.140 s | ~1.00 s at 525 generations |

Every base stalls at 0.95-1.01 s; a faster PPO base did not give a faster ES result.

## What the gait comparison showed (`gait.py`, seed 1)

suraj-phanindra/microduck-sprint-hc5-aifund (0.86 s) wins on stride, not cadence: 3.3 strides/s
at 0.74 m, ~18° lean, trunk rolling -49..+23° and twisting -49..+13°, neck dropped to -0.8 rad,
a 58° dive over the line, 1.5 cm off line at 2 m. nb2 800 (1.46 s): 3.7 strides/s at 0.39 m,
5° lean, ±10° roll, 31 cm off line. ES moves every base toward hc5's gait. Same lesson as
Tien Kung Omni's 400 m gait at the 2026 World Humanoid Robot Games: at speed the torso twists
to balance the legs' swing.

## Scripts

From the repo root, workshop environment (`uv sync --group workshop`).

```bash
.venv/bin/python remix/sprint-nb/es_weights.py BASE.onnx OUT --vx 0.30 --sigma 0.02 --lr 0.01 --workers 4
remix/sprint-nb/score_run.sh dadudu/mjlab-sprint2m-microduck-nb4 nb4   # every checkpoint, exported as publish does
.venv/bin/python remix/sprint-nb/gait.py hc5 hc5.onnx 1.15 nb4 onnx/nb4_850.onnx 0.30
.venv/bin/python remix/sprint-nb/trace.py onnx/nb4_850.onnx 0.30
```

`es_weights.py` imports `remix/hill-so/hc_weights.py`. `score_run.sh` writes `onnx/<run>_<N>.onnx`
and `publish-eval/`; delete `publish-eval/` afterwards.

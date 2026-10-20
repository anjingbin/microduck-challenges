# Off-seed robustness of the dadudu entries (2026-10-19)

The Arena scores every race on its official seeds 1-5. Every entry below was tuned, directly or
through its base, against those seeds. `robust.py` races each policy on seeds 6-25 instead (same
course, same perturbation rules, the speed the Arena picked for it) and counts falls:

```bash
.venv/bin/python remix/robust.py EVENT NAME A.onnx VX [NAME B.onnx VX ...]
```

Simulation only; a real robot is another question.

## sprint-2m

| Policy | Arena (seeds 1-5) | Seeds 6-25 finished | Falls | Mean finish |
|---|---|---|---|---|
| bubu-first (stock recipe, pushes on) | 4.928 s | 20/20 | 0 | 4.97 s |
| bubu-fast4-350 | 1.372 s | 19/20 | 1 | 1.38 s |
| bubu-fast10-75 | 1.204 s | 17/20 | 3 | 1.23 s |
| nb4 checkpoint 850 | 1.328 s | 18/20 | 2 | 1.27 s |
| nb6 checkpoint 600 | 1.140 s | 18/20 | 2 | 1.14 s |
| **bubu-nb4-es** (`sprint-nb/publish/es_0.968.onnx`) | 0.968 s | **18/20** | 2 | 1.07 s |
| bubu-nb4-es2 (`es2_0.952.onnx`) | 0.952 s | 14/20 | 6 | 1.01 s |
| bubu-nb4-es2-finish (`finish_0.944.onnx`) | 0.944 s | 14/20 | 6 | 1.00 s |
| bubu-nb4-es2-mid (`policy.onnx`, entered) | 0.940 s | 14/20 | 6 | 1.03 s |

The first ES pass (sigma 0.02) kept the base's robustness; the fine pass (sigma 0.005) and the
offset tables traded it for the official seeds.

## hill-climb-1m5

| Policy | Arena | Seeds 6-25 finished | Falls | Mean finish |
|---|---|---|---|---|
| bubu-hill-so (`hill-so/publish/policy.onnx`, entered) | 0.784 s | 2/20 | 18 | 0.85 s |
| juicenv/microduck-hill-es9 (its base) | 0.788 s | 2/20 | 18 | 0.87 s |
| bubu-hill-es2 (own) | 1.164 s | 15/20 | 5 | 1.24 s |
| bubu-hill-dive1-75 (own) | 1.224 s | 19/20 | 1 | 1.24 s |

The 0.78 s lineage (namenu -> suraj hc -> juicenv es -> remixes) is ES/hill-climbed on the
official seeds and falls on nine seeds in ten elsewhere.

## roller-sprint-2m

| Policy | Arena | Seeds 6-25 finished | Falls | Mean finish |
|---|---|---|---|---|
| bubu-roller-so (`roller-so/publish/policy.onnx`, entered) | 0.932 s | 11/20 | 9 | 1.01 s |
| suraj-phanindra/microduck-roller-hc5-aifund (its base) | 0.940 s | 15/20 | 5 | 1.00 s |
| DenRakEiw/microduck-senate_duck_0928 (1st) | 0.928 s | 7/20 | 13 | 1.01 s |
| bubu-distill-xin (own, distilled from Xin) | 1.000 s | 8/20 | 12 | 1.04 s |
| bubu-roll5-100 (own PPO) | 2.264 s | 20/20 | 0 | 2.28 s |
| bubu-roll2-250 (own PPO) | 2.332 s | 19/20 | 1 | 2.34 s |

## Robust ES (`es_robust.py`): fast and stable

ES on the weights against seeds 6-15 (a fall costs the time limit), held-out seeds 16-35 for
picking the generation, then checked on fresh seeds 36-55 and the official ones.

| Policy | Arena (official) | Fresh seeds 36-55 | Mean finish |
|---|---|---|---|
| hill: bubu-hill-dive1-75 (base) | 1.224 s | 19/20 | 1.239 s |
| **hill: dive1 + robust ES, generation 300 (`hill-robust/policy.onnx`)** | **1.008 s** | **20/20** | **0.994 s** |
| hill: bubu-hill-so (entered) | 0.784 s | 1/20 | 0.82 s |
| roller: suraj hc5 (base) | 0.940 s | 15/20 | 0.987 s |
| roller: hc5 + robust ES, generation 380 | 0.992 s | 16/20 | 0.999 s |
| roller: bubu-roller-so (entered) | 0.932 s | 10/20 | 1.002 s |

Hill (`--vx 0.30 --sigma 0.02 --lr 0.01 --pairs 12 --workers 2 --seed 41`): held-out finishes
stayed 20/20 up to generation 300, then the policy started fitting the 10 training seeds
(held-out 18/20 by 340). Roller: no real gain in 500 generations; hc5 falls on a quarter of
other seeds and 10 training seeds did not move that.

### Hill, round 2

From round 1's generation 300, `es_robust.py --train-seeds 6-25 --val-seeds 26-45 --workers 4
--seed 43` (20 training seeds against the round-1 overfit). Held-out seeds stayed 20/20 or 19/20
throughout, the official time kept falling. Checked on 40 fresh seeds 56-95:

| Policy | Arena (official) | Fresh seeds 56-95 | Mean finish |
|---|---|---|---|
| round 1, generation 300 (`hill-robust/round1_g300.onnx`) | 1.008 s @0.30 | 36/40 | 1.032 s |
| round 2, generation 340 | 0.964 s @0.75 | 36/40 | 0.975 s |
| round 2, generation 420 | 0.944 s @0.85 | 36/40 | 0.974 s |
| **round 2, generation 460 (`hill-robust/policy.onnx`)** | **0.932 s @0.65** | **38/40** | **0.973 s** |

Round 1's "20/20 on seeds 36-55" was a good draw: on 40 other seeds it falls on 4.

### Roller, 20 training seeds

From suraj-phanindra/microduck-roller-hc5-aifund @ 17c37f18 (not ours: credit it if published),
`es_robust.py --event roller-sprint-2m --vx 0.70 --train-seeds 6-25 --val-seeds 26-45 --workers 4
--seed 44`, 640 generations. Training finishes went 15/20 -> 20/20 by generation 60; held-out
seeds swung between 15 and 20/20, so only generations at 20/20 there were checked on seeds 56-95:

| Policy | Arena (official) | Fresh seeds 56-95 | Mean finish |
|---|---|---|---|
| hc5 (base) | 0.940 s @0.7 | 30/40 | 0.991 s |
| bubu-roller-so (entered, offsets on hc5) | 0.932 s @0.7 | 28/40 | 0.995 s |
| generation 240 | 0.996 s @0.5 | 37/40 | 0.998 s |
| generation 280 | 0.996 s @0.5 | 36/40 | 0.996 s |
| generation 540 | 0.992 s @0.4 | 38/40 | 1.005 s |
| generation 580 | 0.996 s @0.6 | 39/40 | 1.000 s |
| **generation 620 (`roller-robust/policy.onnx`)** | **0.984 s @0.9** | **40/40** | **0.996 s** |

About 0.05 s slower than hc5 on the official seeds, about the same off them, and it stopped
falling: 10 falls in 40 for hc5, none for generation 620.

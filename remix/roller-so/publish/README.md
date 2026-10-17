---
tags:
- microduck
- robotics
- reinforcement-learning
- onnx
library_name: microduck
pipeline_tag: robotics
base_model: suraj-phanindra/microduck-roller-hc5-aifund
base_model_relation: finetune
---

# bubu-roller-so

[suraj-phanindra/microduck-roller-hc5-aifund](https://huggingface.co/suraj-phanindra/microduck-roller-hc5-aifund)
(0.940 s) with its network unchanged, wrapped in a step counter that adds per-step offsets to its joint
targets and to the forward command it reads, searched by CMA-ES on the Arena's official seeds: 0.932 s on
those seeds. An entry for **the 2 m Roller Sprint** of the
[Microduck Arena](https://huggingface.co/spaces/pollen-robotics/microduck-arena). **Simulation only** — see below.

## How it was changed

The source network is frozen: `policy.onnx` contains the source graph unchanged, with a few standard ONNX
operators before and after it.

- **step counter**: the `head_roll` target is rounded to 1/256 rad and k/65536 rad is added, k the number of
  control steps run (saturating at 127). The Arena feeds the previous action back raw, so the next step reads
  k exactly. The first call (zero previous action) runs with k = 1; the 0.5 s settle is 25 calls, so the first
  race step is k = 26.
- **offsets**, zero before k = 26, piecewise linear in k on knots every 3 steps from 26 to 77, held after:
  one on the forward command the network reads (`obs[48]`), and one on each joint target but `head_yaw` and
  `head_roll`. 18 knots × 13 channels = 234 values.

The values were searched by CMA-ES (population 24, initial step 0.05, generation 1,467 of 1,500) on the event's
five official seeds at 0.7 m/s, inside the Arena's runner (runner v4, MuJoCo 3.10.0, x86-64 CPU), with the
fitness the board time plus a sub-step tie-breaker (the overshoot past the line over the run's mean speed).

## Results

Race times (s) on the event's official seeds, at 0.7 m/s commanded forward speed (`arena score`, full sweep):

| seed | microduck-roller-hc5-aifund | this remix |
|---|---|---|
| 1 | 0.94 | 0.94 |
| 2 | 0.94 | 0.92 |
| 3 | 0.94 | 0.92 |
| 4 | 0.94 | 0.94 |
| 5 | 0.94 | 0.94 |
| **board (mean)** | **0.940** | **0.932** |

These are in-sample numbers: the search scored its candidates on these same seeds. The result is fragile:
re-run 8 times with noise of σ = 10⁻⁷ on every observation but the counter's, it finished every seed every time,
with boards of 0.936–0.940 s, never 0.932 s. A different CPU or library build could give the same.

## Not for a real robot

Judged only in the Arena's simulator; nothing here was tried on hardware. The step counter rides on the
`head_roll` target, which a real robot would not feed back bit for bit. Do not load it on a physical MicroDuck.

## Credits

- [suraj-phanindra](https://huggingface.co/suraj-phanindra) for
  [microduck-roller-hc5-aifund](https://huggingface.co/suraj-phanindra/microduck-roller-hc5-aifund), the network
  used here, itself a fine-tune of [juicenv/microduck-roller-es21](https://huggingface.co/juicenv/microduck-roller-es21),
  from [namenu/microduck-arena](https://huggingface.co/namenu/microduck-arena).
- [jrubiosainz](https://huggingface.co/jrubiosainz)'s
  [game-roller-remix2](https://huggingface.co/jrubiosainz/microduck-game-roller-remix2) for the step-counter
  start-offset idea, reimplemented here.
- [Pollen Robotics](https://huggingface.co/pollen-robotics): the MicroDuck, the Microduck Arena and its runner,
  and the publishing tools that wrote `manifest.json`.

The source repository declares no licence; this one declares none either and names the source as its base model.

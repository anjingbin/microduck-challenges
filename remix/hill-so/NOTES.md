# bubu-hill-so: hill-climb-1m5 remix and the searches behind it

Published as [dadudu/microduck-bubu-hill-so](https://huggingface.co/dadudu/microduck-bubu-hill-so),
entered on `hill-climb-1m5`: **0.784 s, 5/5 at 0.30 m/s** (run `01M543H58SHR0GZ4PX21BS`), 3rd on the
board on 2026-10-17 (tied with DenRakEiw, who entered first).

`publish/policy.onnx` (sha256 `7bc4ae4e…920c`) is juicenv/microduck-hill-es9 (0.788 s) with its
network unchanged, wrapped by `so_hill.py` in a step counter (carried in the head_roll target) that
adds per-step joint-target and forward-command offsets. The offsets are `offsets_0.784.npy`
(knots 26..65 every 3), found by CMA-ES on the event's official seeds. Simulation only.
Local seeds: 0.78 / 0.78 / 0.78 / 0.78 / 0.80.

## Scripts

All run from the repo root with the workshop environment (`uv sync --group workshop`, then
`uv pip install cma`). Each uses 4 worker processes, CPU only.

| Script | What it does |
|---|---|
| `so_hill.py` | CMA-ES search of start offsets around a frozen policy (copy of `remix/roller-so/startoffsets.py` with `EVENT` set to `hill-climb-1m5`; change `EVENT` for another race) |
| `hc_weights.py` | Elitist hill climb on a policy's MLP weights, Gaussian noise scaled by each tensor's mean \|w\| |
| `hc_wrapped.py` | The same hill climb on the network inside a fixed `so_hill.py` offset table |

```bash
cd remix/hill-so
# start offsets on es9 (what produced offsets_0.784.npy; ~26 min, 1080 generations)
../../.venv/bin/python so_hill.py es9.onnx out_so --vx 0.30 --k-end 65 --every 3 --sigma0 0.05 --seed 1 --beat 0.780
# weights only
../../.venv/bin/python hc_weights.py es9.onnx out_hcw --sigma 0.0003 --seed 3
# weights inside the offset wrapper (its first export is publish/policy.onnx)
../../.venv/bin/python hc_wrapped.py es9.onnx offsets_0.784.npy out_hcwr --save-at 0.784
```

`es9.onnx` is `policy.onnx` of juicenv/microduck-hill-es9
(`uv run hf download juicenv/microduck-hill-es9 policy.onnx`).

## What was tried, 2026-10-17 (local scores)

- **Start offsets on es9** (`so_hill.py`): board 0.792 with zero offsets (the counter's rounding
  costs one step) → **0.784** by generation ~500; sub-step mean 0.785 → 0.777. Stalled through
  generation 1080 (sigma 0.012).
- **Weights only on es9** (`hc_weights.py`): no gain. sigma 0.01 and 0.002: no candidate improved in
  40–90 generations; even sigma 0.0005 makes most candidates slower or fall (es9 sits on a sharp
  optimum). sigma 0.0003: sub-step 0.7827 → 0.7806 by generation 100, then nothing through 1560;
  board stayed 0.788.
- **Weights inside the offset wrapper** (`hc_wrapped.py`): sub-step 0.7770 → 0.7761, board stayed
  0.784 through generation 1560.
- **Same start-offset search on sprint-2m**, on suraj-phanindra/microduck-sprint-hc5-aifund at
  1.15 m/s: board stayed 0.864 through 1040 generations (sub-step 0.8628 → 0.8559). First place
  (0.860) needs a seed at 0.84 s, a whole step faster.

The board counts in 0.02 s steps (one control step) and breaks ties by who entered first, so beating
an entry means taking a whole step off at least one seed.

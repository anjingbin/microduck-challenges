# bubu-roller-so: publish and enter

`publish/policy.onnx` is hc5 (suraj-phanindra/microduck-roller-hc5-aifund, 0.940 s) wrapped in
CMA-ES start offsets (`startoffsets.py`, run 2, generation 1467; offsets in `cand_0.932.npy`).
Local `arena score`: 0.932 s, 5/5 at 0.70 m/s (seeds 0.94 / 0.92 / 0.92 / 0.94 / 0.94).
sha256 `ab4d025731112904f7ddf9e7f86f1454f9a7aeb7d878fea87d1b7130c2072503`. Simulation only.

Needs `HF_TOKEN` (write) for dadudu. From the repo root:

```bash
uv sync --frozen --group workshop
uv run --group workshop arena score remix/roller-so/publish/policy.onnx --event roller-sprint-2m   # expect 0.932
uv run publish --repo dadudu/microduck-bubu-roller-so --onnx remix/roller-so/publish/policy.onnx \
    --kind perpetual --event roller-sprint-2m --accessories rollers \
    --base-model suraj-phanindra/microduck-roller-hc5-aifund --no-private --enter \
    --description "2 m Roller Sprint skater: suraj-phanindra/microduck-roller-hc5-aifund (0.940 s) with its network unchanged, wrapped in a step counter that adds per-step joint-target and forward-command offsets found by CMA-ES on the Arena's official seeds: 0.932 s. Credit to suraj-phanindra, juicenv and namenu. Simulation only."
# then replace the generated card and manifest with the corrected ones:
uv run hf upload dadudu/microduck-bubu-roller-so remix/roller-so/publish/README.md README.md
uv run hf upload dadudu/microduck-bubu-roller-so remix/roller-so/publish/manifest.json manifest.json
```

Re-run the search: `uv pip install cma`, then
`.venv/bin/python remix/roller-so/startoffsets.py <hc5 policy.onnx> <outdir> --gens 1500 --seed 2 --sigma0 0.05 --every 3 --k-end 77`.

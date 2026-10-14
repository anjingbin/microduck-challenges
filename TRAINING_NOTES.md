# Bubu training notes — sprint-2m

- Task: `Mjlab-Sprint2m-MicroDuck` (Microduck Arena event `sprint-2m`)
- Run name: `bubu-first`
- Backend: Hugging Face Jobs, flavor `l4x1` ($0.80/h), timeout `5h` (spend cap ~$4)
- Config: 4096 envs, 3000 iterations, seed 1, tensorboard logger, no wandb
- Base: upstream `pollen-robotics/microduck-challenges` main, unchanged recipe
- Command:

```bash
uv run train Mjlab-Sprint2m-MicroDuck --env.scene.num-envs 4096 --agent.max-iterations 3000 \
    --agent.seed 1 --agent.logger tensorboard --agent.run-name bubu-first \
    --hf-jobs --no-wandb --flavor l4x1 --timeout 5h
```

- Note: Pollen's reference picked checkpoint 1000 as fastest; evaluate several checkpoints locally
  (`uv run --group workshop arena score policy.onnx --event sprint-2m`) before publishing.
- Nothing published to the Arena yet.

## Result (2026-10-14)

- HF Job `6acf139c5ccb47ba0f791477` (l4x1), ~2h15m, ≈ $1.8
- Final checkpoint exported to ONNX, sha256 `cc5097e8b2ba28d6…`
- Published: https://huggingface.co/dadudu/microduck-bubu-sprint (public)
- Arena sprint-2m: **4.928 s** mean, 5/5 seeds finished, at 0.55 m/s; rank 49 of 57 entries
  (run https://huggingface.co/spaces/pollen-robotics/microduck-arena#/r/01M4WPNH655Q12YG9TS5PH)
- Top of board at the time: 0.908 s. Next: try earlier checkpoints (Pollen's best reference was ckpt 1000) and higher speeds.

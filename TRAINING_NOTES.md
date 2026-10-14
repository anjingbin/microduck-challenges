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

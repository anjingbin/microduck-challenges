# sprint-2m speed campaign (branch bubu-sprint-fast)

Budget: <= $10 HF Jobs total. Spent so far: fast1 ~$0.6 (bubu-first ~$1.8 was before the campaign).

## fast1 — commit 2021e0a
- Job 6acf442b5ccb47ba0f797c71, l4x1, 1000 iters, 44m38s, ~$0.6
- Ckpt repo: dadudu/mjlab-sprint2m-microduck-20261014-165817 (local: /workspace/ckpt_fast1)
- Changes: new reward forward_speed = clamp(body vx, -1, 3) weight 3.0; track_linear_velocity 2.0->0.5;
  upright 2.0->0.5 with std^2 0.05->0.3; pose 1.0->0.5; fell_over limit 70 deg (unchanged default)
- Local Arena scores (sprint-2m):
  - model_250: 2.436 s @0.20 m/s, 5/5
  - model_500: 1.852 s @1.05, 5/5
  - model_750: 1.784 s @0.20, 5/5
  - model_999: 1.736 s @0.25, 5/5  <- best (onnx: /workspace/ckpt_fast1/model_999.onnx)
- Note: speed now nearly independent of command (best at 0.20-0.25 m/s), as intended.

## Next idea (fast2)
- Raise forward_speed weight / cap (3 -> 5), drop track_linear_velocity to 0, lower air_time/pose,
  ease action_rate curriculum; warm-start from fast1 model_999 if supported; 1000 iters, timeout 2h.

## fast2 — commit e21ce32 (launched 17:49 Beijing)
- Job 6acf50165ccb47ba0f798c90, l4x1, 1000 iters, timeout 2h
- Changes vs fast1: forward_speed weight 3->5, cap 3->4 m/s; track_linear_velocity 0.5->0;
  pose 0.5->0.3; air_time 3.0->1.5; action_rate curriculum clamped at -0.3 (was ramp to -1.0). Fresh start (no warm start).
- Result: COMPLETED 44m54s, ~$0.6. Ckpt repo dadudu/mjlab-sprint2m-microduck-20261014-174907 (local /workspace/ckpt_fast2)
  - model_250: 2.100 s @0.45, 5/5
  - model_500: 1.680 s @0.35, 5/5
  - model_750: 1.472 s @0.20, 5/5  <- best so far (onnx: /workspace/ckpt_fast2/model_750.onnx)
  - model_999: 20.820 s @0.55, 3/5 (falls; unstable late)
- Campaign spend: ~$1.2 of $10.
- Next idea: score intermediate ckpts more densely (save_interval 100), keep fast2 rewards but add a mild fall penalty / keep action_rate ramp, or 750-iter run.

## fast3 — commit 19e10a9 (launched 18:46 Beijing)
- Job 6acf5d785ccb47ba0f799b87, l4x1, 1000 iters, ~46m, ~$0.6; ckpt repo dadudu/mjlab-sprint2m-microduck-20261014-184614 (local /workspace/ckpt_fast3)
- Changes vs fast2: fall_penalty (is_terminated) weight -10; upright 0.5->0.8; save_interval 250->100. Fresh start.
- Result: COMPLETED. Local Arena scores (sprint-2m):
  - model_300: 1.808 s @0.25, 5/5
  - model_400: 1.724 s @0.30, 5/5
  - model_800: 1.484 s @0.55, 5/5  <- best of fast3 (onnx: /workspace/ckpt_fast3/model_800.onnx)
  - model_900: 50.000 s, 0/5 (collapse)
  - model_999: 30.516 s @0.45, 2/5 (collapse)
- Still behind fast2 best (model_750 1.472 s). Campaign spend: ~$1.8 of $10.
- Next idea: warm-start from fast2 model_750 / fast3 model_800; or raise fall_penalty earlier and stop/early-pick before late collapse; or shorten to 800 iters.

## fast4 — commit f3fcd05 (launched 19:38 Beijing)
- Job 6acf69b4c66288145223f186, l4x1, +500 iters, timeout 2h; ckpt repo dadudu/mjlab-sprint2m-microduck-20261014-193824
- Warm start from fast2 model_750 (checkpoint tracked at logs/rsl_rl/sprint_2m/warm_fast2/), --agent.resume,
  MICRODUCK_WARM_START=1; fixed LR 3e-4; fast3 rewards (fall penalty -10, upright 0.8); save every 50.
- Result: COMPLETED (~25 min incl. setup, ~$0.35). Warm start worked: model_0 already scores 1.448 s (≈ fast2 model_750).
  (Job log's iteration counter restarted at 0/500 and episode length started low because curricula restart.)
  - model_0 1.448 / 50 1.456 / 100 1.436 (5/5)
  - 150 11.172 (4/5), 200 11.192 (4/5), 250 20.852 (3/5), 300 11.100 (4/5)
  - **model_350: 1.372 s @0.35 m/s, 5/5  <- campaign best** (onnx: /workspace/ckpt_fast4/model_350.onnx)
  - 400 11.112 (4/5), 450 20.804 (3/5), 499 20.788 (3/5)
- Fine-tuning is noisy: one-seed falls appear at many checkpoints. Campaign spend: ~$2.15 of $10.

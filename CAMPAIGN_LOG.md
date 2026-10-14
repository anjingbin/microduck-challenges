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
- Job 6acf5d785ccb47ba0f799b87, l4x1, 1000 iters, timeout 2h; ckpt repo dadudu/mjlab-sprint2m-microduck-20261014-184614
- Changes vs fast2: fall_penalty (is_terminated) weight -10; upright 0.5->0.8; save_interval 250->100. Fresh start.

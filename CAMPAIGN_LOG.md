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

## fast5 — commit a8098e9 (launched 21:03 Beijing)
- Job 6acf7d925ccb47ba0f79a393, l4x1, +500 iters, timeout 2h; ckpt dadudu/mjlab-sprint2m-microduck-20261014-210309
- Warm start from fast4 model_350; fall_penalty -10→-20; upright 0.8→1.2; new lean_forward reward (clamp body gravity_x 0..0.75) weight 1.5; fell_over 75°; fixed LR 3e-4; save every 50.
- Result: COMPLETED ~25 min, ~$0.35. Warm start OK (model_0 = 1.372 s).
  - model_0: 1.372 s @0.85, 5/5 (seed)
  - **model_100: 1.352 s @0.55, 5/5  <- campaign best** (onnx: /workspace/ckpt_fast5/model_100.onnx)
  - most later ckpts 4/5 or worse (noise continues)
  - model_750 in repo is leftover warm_fast2 seed (1.472 s), ignore
- Campaign spend: ~$2.5 of $10. Still far from 0.9 s.

## fast6 — commit 5a11134 (launched 21:58 Beijing)
- Job 6acf8a965ccb47ba0f79ac44, l4x1, +400 iters, timeout 2h; ckpt dadudu/mjlab-sprint2m-microduck-20261014-215842
- Warm from fast5 model_100; LR 1e-4 fixed; save every 25.
- New: progress along spawn heading; finish_bonus (w20, pays 1+max(0,4-t) on reaching 2 m); reached_2m termination (as time_out, so no fall penalty);
  lean_near_finish (w3, forward pitch paid only after 1.5 m); episode 20s -> 8s; spawn xy ±0.05 m.
  (Command speed was already randomized 0.2–1.2 and velocity pushes already on.)
- Result: COMPLETED ~33 min, ~$0.45. Warm start OK (model_0 ≈1.37 s).
  - **model_75: 1.284 s @1.10 m/s, 5/5  <- campaign best** (onnx: /workspace/ckpt_fast6/2026-10-14_14-00-13_fast6_model_75.onnx)
  - model_25 1.332 / model_175 1.288 / model_399 1.556 (all 5/5)
  - Many mid/late ckpts 4/5 (~11 s). Finish bonus + shorter episodes helped early; late instability remains.
- Ideas that helped: finish_bonus + denser ckpts (caught 75/175). Dive-near-finish + lower LR: mixed. Campaign spend ~$2.95 of $10.

## fast7 — commit 8a1bcd8 (launched 22:45 Beijing)
- Job 6acf95935ccb47ba0f79bb3a, l4x1, +300 iters, timeout 2h; ckpt dadudu/mjlab-sprint2m-microduck-20261014-224535
- Warm from fast6 model_75; finish_bonus 20→30; lean_near_finish 3→5; LR 1e-4; save every 25.
- Result: COMPLETED (sat ~15 min in SCHEDULING; ~$0.3–0.45 billed).
  - **model_200: 1.228 s @0.20 m/s, 5/5  <- campaign best** (onnx: /workspace/ckpt_fast7/2026-10-14_15-04-39_fast7_model_200.onnx)
  - model_125 1.244 / 150 1.268 / 175 1.244 / model_0 1.264 (5/5); others 4/5 or 3/5
- Stronger finish shaping gave a stable window at 125–200. Campaign spend ~$3.4 of $10.

## fast8 — commit cb29bb5 (launched 00:12 Beijing)
- Job 6acfa9cc5ccb47ba0f79d565; ckpt dadudu/mjlab-sprint2m-microduck-20261015-001152
- Warm fast7 model_200; finish_bonus 30→40; +300 iters LR 1e-4, save 25. Automated via run_round.sh.
- Result: COMPLETED, 16 min running (~$0.3). Best model_0 1.224 s 5/5 (≈ warm start); model_200 1.232 5/5; rest 3/5–4/5. finish 40 destabilized.

## fast9 — commit 7a4abfe (launched 00:37)
- Job 6acfafaf5ccb47ba0f79ded1; ckpt dadudu/mjlab-sprint2m-microduck-20261015-003658
- Warm fast8 model_0; finish 40→30; fall_penalty -20→-40; upright 1.2→1.5.
- fast9 result: COMPLETED (~45 min in SCHEDULING, ~25 min running, ~$0.35). Worse: best model_100 1.352 s 5/5; most 2/5–4/5. Stronger fall/upright hurt.

## fast10 — fast7 settings again (finish 30, lean 5, fall -20, upright 1.2), warm from fast8 model_0 (1.224 s)
- fast10 result: job 6ad00cf55ccb47ba0f7a2053, 16 min running (~$0.3). **model_75: 1.204 s @0.50, 5/5 <- overall best** (/workspace/ckpt_fast10/2026-10-14_23-17-25_fast10_model_75.onnx); others 3/5–4/5.
- Campaign spend ≈ $4.0 of $10 (fast1–fast10). Publishing fast10 model_75 to Arena.
- Published fast10 model_75 as bubu-fast10-75: Arena 1.204 s 5/5, run 01M4YCMY86P9EW09FCE6KA

## dive1 — namenu-style lean (launched ~08:30 Beijing)
- Warm from fast10 model_75 (1.204 s); +400 iters, LR 1e-4, save 25.
- lean_forward (whole episode) weight 1.5→4, cap 0.75→0.85 (~58°); upright 1.2→0.3; pose 0.3→0.15; episode 8→5 s; finish bonus 30, lean_near_finish 5, fall -20, fell_over 75° kept.
- dive1 result: job 6ad01dfd5ccb47ba0f7a2942, 20 min running (~$0.35). **No checkpoint finished 5/5** (best 4/5 ≈ 10.9–11.3 s; late ckpts 0–2/5).
  - Gait clearly changed: model_275 finishers 1.28–2.24 s; replay trace tilt mean 47°, peak 168° (it pitches past vertical / tumbles and recovers, then lands over the line).
  - Verdict: aggressive lean from step 0 breaks the 5/5 reliability; fast10 model_75 (1.204 s) stays the best. Campaign ≈ $4.35 of $10.

## dive2 — gentle namenu-style lean (launched ~09:17 Beijing)
- Warm fast10 model_75; +350 iters LR 1e-4 save 25.
- lean_forward w2 cap 0.42 (~25°); lean_near_finish w5 from 1.7 m; upright 1.0; pose 0.3; fall -25; fell_over 70°; episode 6 s; finish 30.

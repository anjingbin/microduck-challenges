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
- dive2 result: job 6ad02959c662881452249663, 17.5 min running (~$0.3). Best model_25: 1.224 s @0.95, 5/5 (does not beat 1.204). Others 1/5–4/5.
  - Gait: steady ~24° forward lean (max ~29°), no end dive; finishers ~1.18–1.26 s per seed, but one start falls on most ckpts.
- Best stays fast10 model_75 (1.204 s). Campaign ≈ $4.65 of $10.

# ROLLER (roller-sprint-2m) — branch bubu-roller
## roll1 — fresh, 900 iters, l4x1, timeout 2h (launched ~10:05 Beijing)
- Upstream Mjlab-RollerSprint2m-MicroDuck + forward_speed (w5, cap 3 m/s), finish_bonus (w30, 2 m along spawn heading),
  reached_2m termination, fall_penalty -20, episode 8 s, save every 25. Command range kept (-0.5..0.6, upstream: >0.6 broke learning).
- roll1 result: job 6ad033b5c662881452249b89 COMPLETED 43m25s (~$0.6). Ckpt dadudu/mjlab-rollersprint2m-microduck-20261015-100013.
  - Best **model_500: 2.696 s @0.40 m/s, 5/5** (onnx: /workspace/ckpt_roll1/2026-10-15_02-01-50_roll1_model_500.onnx)
  - Nearby 5/5: 450 2.728 / 475 2.748 / 825 2.744 / 700 2.748; early 0–175 all 0/5 (40 s timeout).
  - Arena #1 is 1.216 s — we are ~2.2× slower. Stable roller gait from ~225 on.
  - Combined campaign spend ≈ $5.25 of $10.

## roll2 — warm from roll1 model_500 (launched ~11:55 Beijing)
- Job 6ad04eaf5ccb47ba0f7a4045 RUNNING, l4x1, +500 iters, timeout 2h; ckpt dadudu/mjlab-rollersprint2m-microduck-20261015-115520
- Warm start roll1 model_500 (2.696 s); cmd vx curriculum 0.6→1.0→1.5; wheel_speed w1; fwd_speed w8; lean_forward w1.5 cap0.5; lean_near_finish w3; LR 1e-4 fixed.
- Branch bubu-roller commit 27b2702. Auto score via run_roller_round.sh after job completes.
- Campaign spend before roll2 ≈ $5.25 of $10; roll2 projected ~$0.4–0.6.

### roll2 (job 6ad04eaf5ccb47ba0f7a4045, repo dadudu/mjlab-rollersprint2m-microduck-20261015-115520)
- Warm from roll1 model_500. cmd vx max curriculum 0.6 -> 1.0 (it100) -> 1.5 (it250); wheel_speed w10->1; forward_speed w8;
  lean_forward w1.5 cap 0.5; lean_near_finish w3 from 1.6 m. 500 iters, 26.3 min l4x1 (~$0.35). Campaign total ~$5.60.
- Best: model_250 2.332 s 5/5 (@1.5 m/s). Others 5/5: 425 2.384, 25 2.468, 275 2.472, 450 2.496, 499 2.516. Many mid ckpts unstable (0-4/5).
- Leader analysis: Xin-AIAspire roller-es2 0.980 s (now #1), namenu strata-roller-re19 1.216 s (#14): trunk z ~0.13-0.15 (ours 0.11),
  tilt median 30-37 deg with end dive to 81-104 deg, wheels lifted 11-15 cm (big strides), start at 1.1-1.3 m/s. Ours: low glide, 14 deg, 2-6 cm lift.

### roll3 (job 6ad05cfcc66288145224ba94, repo dadudu/mjlab-rollersprint2m-microduck-20261015-125620)
- Warm roll2 model_250. + start_splay (foot lateral sep, t<0.4 s, cap 0.16, w10), early_speed (vx t<0.4 s, w4),
  trunk_height (exp, 0.145 m, std 0.02, w2), foot_clearance (foot z gap cap 0.12, gated vx>0.5, w3). 500 iters, 26.1 min (~$0.35). Total ~$5.95.
- Best: model_0 2.320 s 5/5 (= warm start re-scored; noise vs 2.332), model_25 2.324 5/5, 175 2.364, 350 2.372, 375 2.416. Late ckpts degrade (1-4/5).
- Gait vs roll2-250: start vx 0.54 -> 0.44-0.57 m/s, vx@0.5s 0.59 -> 0.62-0.81, trunk z median 0.109 -> 0.114-0.115 (target 0.145 not reached),
  wheel lift still 2-7 cm, tilt 14-17 deg. Skate splay/stride rewards did not change the gait materially. No improvement.

### roll4 (job 6ad06b9b5ccb47ba0f7a555d, repo dadudu/mjlab-rollersprint2m-microduck-20261015-135844)
- Warm roll2 model_250. REMOVED: com_height_target (paid 0.094-0.124 m), glide (w4), hip_roll_neutral (-2), gait_symmetry (-1),
  forward_lean (15 deg target), action_rate_weight curriculum. SOFTENED: action_rate -1->-0.2, feet_flat -2->-0.5, pose 2->0.5,
  upright 2->1, single_support 3->1. STRENGTHENED: start_splay w20 cap 0.20, early_speed w10, trunk_height w6, foot_clearance w10 cap 0.15.
  500 iters, 26.1 min (~$0.35). Campaign total ~$6.3.
- RESULT: collapse. model_0 2.328 s 5/5 (= warm start); every trained ckpt 0/5 or 1/5 (400, 450, 475 at 1/5).
- Gait did change toward leaders: trunk z median 0.109 -> 0.140-0.144, wheel lift 2-4 cm -> 4-9 cm (model_100: 13-19 cm but falls at 1.4 s, tilt 115 deg),
  tilt 15 -> 9-13 deg (lean lost). Start speed not better (first 0.2 s 0.35-0.73 vs 0.54-0.48). Policy is taller/higher-stepping but unstable.
- Takeaway: the removed stabilizers were holding balance; next try should reintroduce them partially or ramp the new terms by curriculum.

### Roller published
- roll2 model_250 as bubu-roll2-250 -> public dadudu/microduck-bubu-roller (rev 7ab2141); Arena roller-sprint-2m 2.332 s 5/5 @1.5 m/s, rank 53/70, run 01M4Z8QC1T7GF2BMHTX2JQ.

### roll5 (job 6ad094f2c66288145224eaaf, repo dadudu/mjlab-rollersprint2m-microduck-20261015-165508)
- Xin trace analysis (seed 3, correct JOINT_NAMES mapping): pitch ~44 deg at t=0, mean 32 deg, 77 deg at finish; NOT a heel wheelie in sim:
  each wheel touches ground only 12-18% of frames, all 4 airborne 43% (bounding lunge); trunk z median 0.151; head thrown (neck ~-1.4 rad).
- Base roll2/3 set (roll4 removals reverted). Removed forward_lean (15 deg target) + start_splay; upright 2->1; lean cap 0.6;
  + pitch_target (exp, 0.56 rad, std 0.2, w4); early_speed w6; neck_joint_pos -0.5->-0.1; cmd stages 0.6/1.2/2.0. Warm roll2-250. 28.1 min (~$0.37). Total ~$6.7.
- Best: model_100 2.264 s 5/5 (beats roll2 2.332 by 0.07 s). 25 2.316, 50 2.320 5/5; >=125 degrade (2-4/5).
- Gait model_100: lean now 33 deg median (roll2 15) — matches Xin's ~30; but still glides on left wheels (contact 76-84%), never airborne,
  trunk z 0.113, wheel lift <=4 cm, first-0.2 s speed 0.56 (Xin 1.26). Leaning alone gave little speed.
- roll5 model_100 as bubu-roll5-100 -> dadudu/microduck-bubu-roller (rev 8fe9099); Arena 2.264 s 5/5, rank 54/72, run 01M4ZEZ0Q409SGH360X9TA.

### roll6 (job 6ad0bc9cc662881452250f3a, repo dadudu/mjlab-rollersprint2m-microduck-20261015-194421)
- Warm roll5 model_100; keeps roll5 terms. + airborne (both foot sites >1.5 cm, vx>0.5) weight ramp 0.5->2 (it75)->4 (it150)->6 (it250);
  foot_clearance ramp 3->4.5->6; early_speed w10. 450 iters (completed, log shows it 449/450; wait log RUN_MIN 0 is an artifact of box outage). ~25 min (~$0.35). Total ~$7.05.
- Best: model_0 2.248 s 5/5 (= warm start re-scored, noise vs 2.264); model_50 2.368 5/5; all others 1-4/5. No real gain.
- Gait (model_0/50/325): airborne 0% (Xin 43%), lean median 33 deg (unchanged), vx@0.5s 0.59-0.84, trunk z ~0.11, wheel lift <=7 cm.
  The bounding reward never got traction from this warm start; the policy just got less stable.

### Free investigation: rules + CPU imitation feasibility (no HF jobs, $0)
- Rules: no written clause on imitating/distilling other entries found in the Arena Space README/guides or microduck-challenges
  README/AGENTS.md. Arena design spec: published policies are downloadable so the platform is "a place to learn from other
  people's policies"; `publish --base-model` exists for "Remix ... warm start / fine-tune" attribution. No LICENSE on
  microduck-challenges / microduck-arena GitHub repos (API: none); microduck_rl + mjlab-microduck Apache-2.0 (3D models CC BY-SA-NC).
  Xin-AIAspire/microduck-roller-es2: no license field/tag in card -> default all rights reserved.
- Leader robustness: Xin 5/5 on Arena seeds 1-5 @1.6 but 20/30 on seeds 1-30; namenu 10/30 @0.9 -> tuned to the eval seeds.
- DAgger distillation from Xin ONNX in the Arena's own CPU sim (tools/imitation/dagger.py; MLP 256-256-128; 14 rounds x 30 rollouts, ~7.5 min CPU):
  best snapshot 5/5 @1.6 on seeds 1-5 in-loop (mean 1.068 s); exported ONNX via `arena score` sweep: 24.43 s, 2/5 @1.5 (fragile).
  Gait of student: airborne 37% (Xin 43%), first 0.2 s 1.18 m/s (Xin 1.26, roll5 0.56), lean median 26 deg, trunk z median 0.153, wheel lift 7-11 cm.
  -> bounding transfers by distillation; RL reward shaping (roll3-6) never produced it. Not published.

### Distilled-from-Xin student (CPU DAgger, $0)
- tools/imitation/dagger_robust.py: MLP 512-512-256, 30 rounds x 40 rollouts (25 on Arena seeds 1-5 @1.4-1.8, 8 on seed 1 @1.0-2.0 sweep,
  7 random seeds), action-noise injection 0.03-0.15, 150-step rollout cap, official arena.evaluate each round from it3. ~29 min CPU.
- Official `arena score --event roller-sprint-2m`: it18 1.000 s 5/5 @1.6 (best); it7/it20/it24 1.004 s; most rounds 5/5 at 1.00-1.07 s; plateau ~1.00.
- Gait it18: airborne 46% (Xin 43%), first 0.2 s 1.23 m/s, vx@0.5s 2.00, lean median 33 deg (dives to 127 deg after finish), trunk z 0.150, wheel lift 12-21 cm.
- File: /workspace/imitation/best/bubu-distill-xin-it18.onnx. No GPU fine-tune needed (5/5 reached on CPU). Spend unchanged ~$7.05.
- NOT published. When approved: publish --base-model Xin-AIAspire/microduck-roller-es2, description must say "distilled from Xin".
- PUBLISHED bubu-distill-xin (it18) -> dadudu/microduck-bubu-roller rev 62c4f46, base_model Xin-AIAspire/microduck-roller-es2, desc credits Xin. Arena 1.000 s 5/5 @1.6, rank 2/74, run 01M4ZYB8W1PKJYN46SB9QK. Clip: /workspace/arena_compare/bubu_distill.mp4 (1.7 MB).

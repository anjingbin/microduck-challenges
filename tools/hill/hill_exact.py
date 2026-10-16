"""Exact-course hill env: sprint env (flat plane) + gravity tilted 8 deg (== infinite 8 deg incline, uphill = +x),
spawn yaw +-2 deg, cmd 0.15-1.0, finish at 1.5 m along slope. CPU eval of an ONNX."""
import math, sys, numpy as np, torch
import microduck_challenges
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv
import os
DEG = 0.0 if os.environ.get('FLAT') else 8.0
S8, C8 = math.sin(math.radians(DEG)), math.cos(math.radians(DEG))
DIST = float(os.environ.get('DIST', 1.5))

def make_cfg(n, train=False, rew=None):
    c = load_env_cfg('Mjlab-Sprint2m-MicroDuck'); c.scene.num_envs = n
    c.sim.mujoco.gravity = (-9.81 * S8, 0.0, -9.81 * C8)
    if os.environ.get('ARENA_GEOMS', '1') == '1':  # Arena builds robot_allcollisions.xml stripped to 11 geoms == mjlab standup spec
        import dataclasses as _dc; from mjlab_microduck.robot.microduck_constants import get_standup_spec
        c.scene.entities['robot'] = _dc.replace(c.scene.entities['robot'], spec_fn=get_standup_spec)
        c.sim.nconmax = max(c.sim.nconmax or 0, 64)
    c.events['reset_base'].params['pose_range']['yaw'] = (-0.035, 0.035)
    c.events['reset_base'].params['pose_range']['pitch'] = (math.radians(DEG), math.radians(DEG))  # upright w.r.t. gravity, as on the Arena slope
    c.commands['twist'].ranges.lin_vel_x = (0.15, 1.0)
    c.commands['twist'].ranges.heading = (0.0, 0.0)
    c.terminations['reached_2m'].params = {**(c.terminations['reached_2m'].params or {}), 'length_m': 1.5}
    if 'finish_bonus' in c.rewards: c.rewards['finish_bonus'].params = {**(c.rewards['finish_bonus'].params or {}), 'length_m': 1.5}
    if 'lean_near_finish' in c.rewards: c.rewards['lean_near_finish'].params['start_m'] = 1.2
    if not train:
        c.observations['actor'].enable_corruption = False
        for k in list(c.events):
            if k.startswith('randomize') or k in ('push_robot','foot_friction','encoder_bias','base_com'): c.events.pop(k)
        for a in c.scene.entities['robot'].articulation.actuators:
            if hasattr(a, 'vin_range'): a.vin_range = (7.35, 7.35); a.vin_drop_gain_range = (0.1, 0.1)
        for tn in ('base_ang_vel', 'projected_gravity'):
            c.observations['actor'].terms[tn].delay_max_lag = 0
        for k in list(c.curriculum):
            if 'com' in k: c.curriculum.pop(k)
        for g in ('actor',):
            for tn in ('projected_gravity','base_ang_vel'):
                t = c.observations[g].terms[tn]
                if t.params and 'max_angle_deg' in t.params: t.params['max_angle_deg'] = 0.0
    if rew: rew(c)
    return c

def make_env(cfg):
    if os.environ.get('ARENA_DELAY'):
        import arena_delay; arena_delay.install(lo=int(os.environ.get('AD_LO',3)), hi=int(os.environ.get('AD_HI',6))) if not getattr(arena_delay, '_done', False) else None; arena_delay._done = True
    e = ManagerBasedRlEnv(cfg, device='cpu')
    e.scene['robot'].data.gravity_vec_w[:] = torch.tensor([-S8, 0.0, -C8])
    return e

def eval_onnx(path, vx=0.3, n=16, T=4.5, settle=25):
    """Arena timing: policy runs during a settle of `settle` steps; clock + progress origin start after it."""
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    cfg = make_cfg(n); cfg.episode_length_s = 20.0
    for k in ('head_pose', 'body_pose'): cfg.commands[k].zero_command_prob = 1.0
    cfg.commands['twist'].ranges.lin_vel_x = (vx, vx); cfg.commands['twist'].rel_standing_envs = 0.0
    cfg.commands['twist'].rel_forward_envs = 1.0; cfg.commands['twist'].resampling_time_range = (100.0, 100.0)
    cfg.terminations.pop('reached_2m', None); cfg.terminations.pop('time_out', None)
    env = make_env(cfg); obs, _ = env.reset()
    rob = env.scene['robot']
    done_t = np.full(n, np.nan); fell = np.zeros(n, bool); x0 = None; tip = np.zeros(n, int)
    for k in range(settle + int(T / env.step_dt)):
        env.command_manager.get_term('twist').vel_command_b[:, 0] = vx
        O = obs['actor'].numpy().astype(np.float32)
        a = np.concatenate([sess.run(None, {sess.get_inputs()[0].name: O[i:i+1]})[0] for i in range(n)])
        obs, r, term, trunc, info = env.step(torch.from_numpy(a))
        x = rob.data.root_link_pos_w[:, 0].numpy().copy()
        gz = rob.data.projected_gravity_b[:, 2].numpy()
        if k == settle - 1: x0 = x.copy(); continue
        if k < settle: continue
        tip = np.where(gz > -0.5, tip + 1, 0)
        for i in range(n):
            if np.isnan(done_t[i]) and not fell[i]:
                if tip[i] >= 15 or term[i]: fell[i] = True
                elif x[i] - x0[i] >= DIST: done_t[i] = (k - settle + 1) * env.step_dt
    return done_t, fell

if __name__ == '__main__':
    d, f = eval_onnx(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.3)
    print('EXACT finish', int(np.sum(~np.isnan(d))), '/', len(d), 'fell', int(f.sum()), 'mean t', np.nanmean(d), np.round(d, 2).tolist())

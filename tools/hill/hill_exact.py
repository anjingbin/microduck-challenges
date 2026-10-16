"""Exact-course hill env: sprint env (flat plane) + gravity tilted 8 deg (== infinite 8 deg incline, uphill = +x),
spawn yaw +-2 deg, cmd 0.15-1.0, finish at 1.5 m along slope. CPU eval of an ONNX."""
import math, sys, numpy as np, torch
import microduck_challenges
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv
S8, C8 = math.sin(math.radians(8)), math.cos(math.radians(8))

def make_cfg(n, train=False, rew=None):
    c = load_env_cfg('Mjlab-Sprint2m-MicroDuck'); c.scene.num_envs = n
    c.sim.mujoco.gravity = (-9.81 * S8, 0.0, -9.81 * C8)
    c.events['reset_base'].params['pose_range']['yaw'] = (-0.035, 0.035)
    c.commands['twist'].ranges.lin_vel_x = (0.15, 1.0)
    c.commands['twist'].ranges.heading = (0.0, 0.0)
    c.terminations['reached_2m'].params = {**(c.terminations['reached_2m'].params or {}), 'length_m': 1.5}
    if 'finish_bonus' in c.rewards: c.rewards['finish_bonus'].params = {**(c.rewards['finish_bonus'].params or {}), 'length_m': 1.5}
    if 'lean_near_finish' in c.rewards: c.rewards['lean_near_finish'].params['start_m'] = 1.2
    if not train:
        c.observations['actor'].enable_corruption = False
        for k in list(c.events):
            if k.startswith('randomize') or k in ('push_robot','foot_friction','encoder_bias','base_com'): c.events.pop(k)
        for k in list(c.curriculum):
            if 'com' in k: c.curriculum.pop(k)
        for g in ('actor',):
            for tn in ('projected_gravity','base_ang_vel'):
                t = c.observations[g].terms[tn]
                if t.params and 'max_angle_deg' in t.params: t.params['max_angle_deg'] = 0.0
    if rew: rew(c)
    return c

def make_env(cfg):
    e = ManagerBasedRlEnv(cfg, device='cpu')
    e.scene['robot'].data.gravity_vec_w[:] = torch.tensor([-S8, 0.0, -C8])
    return e

def eval_onnx(path, vx=0.3, n=16, T=4.0):
    import onnxruntime as ort
    from microduck_challenges.sprint_2m.env import _bubu_progress
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    env = make_env(make_cfg(n)); obs, _ = env.reset()
    done_t = np.full(n, np.nan); fell = np.zeros(n, bool)
    for k in range(int(T / env.step_dt)):
        env.command_manager.get_term('twist').vel_command_b[:, 0] = vx
        O = obs['actor'].numpy().astype(np.float32); a = np.concatenate([sess.run(None, {sess.get_inputs()[0].name: O[i:i+1]})[0] for i in range(n)])
        obs, r, term, trunc, info = env.step(torch.from_numpy(a))
        prog = _bubu_progress(env).numpy()
        for i in range(n):
            if np.isnan(done_t[i]) and not fell[i]:
                if term[i] and prog[i] < 1.4: fell[i] = True
                elif prog[i] >= 1.45 or (term[i] or trunc[i]): done_t[i] = (k + 1) * env.step_dt
    q = env.scene['robot'].data.root_link_quat_w; p = env.scene['robot'].data.root_link_pos_w
    print('final prog', np.round(prog, 2).tolist()); print('pos x,y', np.round(p[:, :2].numpy(), 2).tolist()[:6])
    return done_t, fell

if __name__ == '__main__':
    d, f = eval_onnx(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.3)
    print('EXACT finish', int(np.sum(~np.isnan(d))), '/', len(d), 'fell', int(f.sum()), 'mean t', np.nanmean(d), np.round(d, 2).tolist())

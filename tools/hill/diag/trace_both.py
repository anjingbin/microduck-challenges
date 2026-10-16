import sys, math, numpy as np
P = '/workspace/hill/es2.onnx'; N = 60; th = math.radians(8)
mode = sys.argv[1]
if mode == 'arena':
    import os, mujoco, arena.runner as R, arena.actuator as Act
    if os.environ.get('FL'):
        import arena.course as Co; Co.FORCE_LIMIT = float(os.environ['FL'])
    if os.environ.get('AD_LO'): Act.DELAY_STEPS = (int(os.environ['AD_LO']), int(os.environ['AD_HI']))
    from arena.manifest import load_event
    from arena.course import compile_course
    from arena.policy import load_policy
    man = load_event('hill-climb-1m5'); model = compile_course(man)
    R._perturb = lambda data, adr, m, s: None   # no perturbation: identical start
    pol = load_policy(P); O, A, X = [], [], []
    class W:
        def act(s, o):
            a = pol.act(o); O.append(np.array(o)); A.append(np.array(a).reshape(-1))
            if len(O) >= N: raise StopIteration
            return a
    orig = R._record
    def rec(f, d, adr): orig(f, d, adr); X.append(np.r_[d.qpos[:7], d.qpos[adr.qpos]])
    R._record = rec
    c = np.asarray(man.command, float).copy(); c[0] = 0.3
    try: R.run_once(man, W(), 1, model=model, command=c)
    except StopIteration: pass
    np.savez('/workspace/hill/diag/arena%s%s.npz' % (os.environ.get('AD_LO',''), os.environ.get('FL','')), O=np.array(O), A=np.array(A), X=np.array(X))
else:
    sys.path.insert(0, '/workspace/microduck_train/tools/hill'); import hill_exact as H, torch, onnxruntime as ort
    cfg = H.make_cfg(1); cfg.events['reset_base'].params['pose_range'].update({'x': (0, 0), 'y': (0, 0), 'yaw': (0, 0), 'z': (0.1188, 0.1188)})
    cfg.episode_length_s = 20.0
    for k in ('head_pose', 'body_pose'): cfg.commands[k].zero_command_prob = 1.0
    cfg.commands['twist'].ranges.lin_vel_x = (0.3, 0.3); cfg.commands['twist'].rel_standing_envs = 0.0
    cfg.commands['twist'].rel_forward_envs = 1.0; cfg.commands['twist'].resampling_time_range = (100.0, 100.0)
    env = H.make_env(cfg); obs, _ = env.reset(); sess = ort.InferenceSession(P, providers=['CPUExecutionProvider'])
    O, A, X = [], [], []; rob = env.scene['robot']
    for k in range(N):
        env.command_manager.get_term('twist').vel_command_b[:, 0] = 0.3
        o = obs['actor'].numpy().astype(np.float32)
        a = sess.run(None, {sess.get_inputs()[0].name: o})[0]
        O.append(o[0]); A.append(a[0])
        obs, *_ = env.step(torch.from_numpy(a))
        X.append(np.r_[rob.data.root_link_pos_w[0].numpy(), rob.data.root_link_quat_w[0].numpy(), rob.data.joint_pos[0].numpy()])
    import os; np.savez('/workspace/hill/diag/mjlab%s.npz' % (('_ad' if os.environ.get('ARENA_DELAY') else '') + os.environ.get('AD_LO','')), O=np.array(O), A=np.array(A), X=np.array(X))
print('saved', mode)

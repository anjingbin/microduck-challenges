import sys, math, numpy as np
P = '/workspace/hill/es2.onnx'; N = 60; th = math.radians(8)
mode = sys.argv[1]
if mode == 'arena':
    import mujoco, arena.runner as R
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
    np.savez('/workspace/hill/diag/arena.npz', O=np.array(O), A=np.array(A), X=np.array(X))
else:
    sys.path.insert(0, '/workspace/microduck_train/tools/hill'); import hill_exact as H, torch, onnxruntime as ort
    cfg = H.make_cfg(1); cfg.events['reset_base'].params['pose_range'].update({'x': (0, 0), 'y': (0, 0), 'yaw': (0, 0), 'z': (0.1188, 0.1188)})
    cfg.episode_length_s = 20.0
    env = H.make_env(cfg); obs, _ = env.reset(); sess = ort.InferenceSession(P, providers=['CPUExecutionProvider'])
    O, A, X = [], [], []; rob = env.scene['robot']
    for k in range(N):
        env.command_manager.get_term('twist').vel_command_b[:, 0] = 0.3
        o = obs['actor'].numpy().astype(np.float32)
        a = sess.run(None, {sess.get_inputs()[0].name: o})[0]
        O.append(o[0]); A.append(a[0])
        obs, *_ = env.step(torch.from_numpy(a))
        X.append(np.r_[rob.data.root_link_pos_w[0].numpy(), rob.data.root_link_quat_w[0].numpy(), rob.data.joint_pos[0].numpy()])
    np.savez('/workspace/hill/diag/mjlab.npz', O=np.array(O), A=np.array(A), X=np.array(X))
print('saved', mode)

"""Make mjlab's BAM actuator command delay identical to the Arena's (arena/actuator.py Servos + runner._step/_start).

Arena per control step: Servos.drive() runs 4x inside the decimation loop and once more after it; every call
appends the (constant) target to a ring of depth 7 and applies ring[-1-L], L ~ U{3..6} drawn per call.
=> at in-loop substep j (0..3): current target if L <= j, previous step's if L <= j+5, else two steps back.
Reset: drive(zeros) backfills the ring with 0 rad targets (mjlab: post-reset write of joint_pos_target=0),
so the first control step partly drives toward 0 rad exactly like the Arena.
"""
import dataclasses, torch
from mjlab.actuator.actuator import Actuator

S = {'reset_ids': None}

def install(decimation=4, lo=3, hi=6):
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    orig_reset_idx = ManagerBasedRlEnv._reset_idx
    def _reset_idx(self, env_ids=None):
        r = orig_reset_idx(self, env_ids)
        S['reset_ids'] = env_ids
        return r
    ManagerBasedRlEnv._reset_idx = _reset_idx
    orig_apply = Actuator.apply_delay

    def apply_delay(self, cmd):
        if getattr(self.cfg, 'delay_max_lag', 0) == 0 or self.command_field != 'position':
            return orig_apply(self, cmd)
        tgt = cmd.position_target
        if getattr(self, '_ad_hist', None) is None or self._ad_hist[0].shape != tgt.shape:
            self._ad_hist = [tgt.clone() for _ in range(3)]; self._ad_n = 0
        ids = S['reset_ids']
        if ids is not None:   # the post-reset write: backfill reset envs with this (zero) target
            for h in self._ad_hist: h[ids] = tgt[ids]
            S['_seen'] = S.get('_seen', set()) | {id(self)}
            return dataclasses.replace(cmd, position_target=self._ad_hist[0].clone())
        j = self._ad_n % decimation; self._ad_n += 1
        if j == 0:
            self._ad_hist = [tgt.clone(), self._ad_hist[0], self._ad_hist[1]]
        L = torch.randint(lo, hi + 1, tgt.shape, device=tgt.device)
        back = torch.where(L <= j, 0, torch.where(L <= j + 5, 1, 2))
        out = torch.gather(torch.stack(self._ad_hist, 0), 0, back.unsqueeze(0)).squeeze(0)
        return dataclasses.replace(cmd, position_target=out)
    Actuator.apply_delay = apply_delay

    from mjlab.scene.scene import Scene
    orig_w = Scene.write_data_to_sim
    def write_data_to_sim(self, *a, **k):
        r = orig_w(self, *a, **k)
        S['reset_ids'] = None   # reset flag lasts exactly one write
        return r
    Scene.write_data_to_sim = write_data_to_sim

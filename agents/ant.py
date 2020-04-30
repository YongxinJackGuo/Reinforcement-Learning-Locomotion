import numpy as np
from gym import utils
from gym.envs.mujoco import ant_v3
import torch

class Ant():
    def __init__(self, env, horizon):
        """
        :param env: environment
        :param horizon: time steps per batch
        """
        self.env = env
        self.horizon = horizon
        self.ob_dim = env.observation_space.shape[0]  # 111 observation dim
        self.ac_dim = env.action_space.shape[0]  # 8 action dim

    def get_traj_per_batch(self, pi, val_net):
        """
        :param pi: policy net
        :param val_net: value net
        :return: A batch of trajectory information
        """
        # trajectory sequence: sarsarsar...
        # initialization
        rew = 0
        ob = self.env.reset()
        ac = self.env.action_space.sample()
        t = 0

        cur_ep_rew = 0
        cur_ep_len = 0
        ep_rew = []
        ep_len = []

        rews = np.zeros(self.horizon, 'float32')
        dones = np.zeros(self.horizon, 'int32')
        acs = np.array([ac for _ in range(self.horizon)])
        obs = np.array([ob for _ in range(self.horizon)])
        vpreds = np.zeros(self.horizon, 'float32')


        while True:
            obs[t] = ob
            # action update
            ac, vpred = pi.sample_action(ob), val_net(ob)
            acs[t] = ac
            vpreds[t] = vpred
            # step update
            ob, rew, done, _ = self.env.step(ac)
            rews[t] = rew
            dones[t] = done

            cur_ep_len += 1
            cur_ep_rew += rew

            if t > 0 and t % (self.horizon - 1) == 0:
                yield {"ob" : obs, "ac" : acs, "vpreds" : vpreds,
                       "rews" : rews, "done": dones, "ep_rew" : ep_rew,
                       "ep_len" : ep_len}

            # one episode ends
            if done:
                ep_rew.append(cur_ep_rew)
                ep_len.append(cur_ep_len)
                cur_ep_rew = 0
                cur_ep_len = 0
                ob = self.env.reset()

            t += 1


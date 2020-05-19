import numpy as np
from gym import utils
from gym.envs.mujoco import ant_v3
from gym.envs.mujoco import hopper
import torch


class Ant():
    def __init__(self, env, horizon, epsi):
        """
        :param env: environment
        :param horizon: time steps per batch
        """
        self.env = env
        self.horizon = horizon
        self.epsi = epsi
        self.ob_dim = env.observation_space.shape[0]  # 111 observation dim
        self.ac_dim = env.action_space.shape[0]  # 8 action dim

    def get_traj_per_batch(self, pi, val_net=None, gamma=1):
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
        vpreds = torch.zeros(self.horizon)
        values = torch.zeros(self.horizon)

        while True:
            obs[t] = ob
            # action update
            ac = pi.sample_action(ob).detach().numpy()
            acs[t] = ac
            if val_net is not None:
                vpred = val_net(ob)
                vpreds[t] = vpred
            # step update
            ob, rew, done, _ = self.env.step(ac)
            rews[t] = rew
            dones[t] = done

            cur_ep_len += 1
            cur_ep_rew += rew

            if t > 0 and t % (self.horizon - 1) == 0:
                # ob, ac, rews, done, ep_rew, ep_len : nparray
                # vpreds: torch
                ep_len.append(cur_ep_len)
                pass_time = 0
                for ep in range(len(ep_len)):
                    temp_value = 0
                    for i in reversed(range(ep_len[ep])):
                        temp_value = temp_value * gamma + rews[i + pass_time]
                        values[i + pass_time] = temp_value
                    pass_time += ep_len[ep]
                return {"ob": obs, "ac": acs, "vpreds": vpreds,
                        "rews": rews, "done": dones, "ep_rew": ep_rew,
                        "ep_len": ep_len, 'values': values}

            # one episode ends
            if done or cur_ep_len == self.epsi:
                ep_rew.append(cur_ep_rew)
                ep_len.append(cur_ep_len)
                cur_ep_rew = 0
                cur_ep_len = 0
                ob = self.env.reset()

            t += 1

import numpy as np
from gym import utils
from gym.envs.mujoco import ant_v3
import torch


class FNNclassifier(torch.nn.Module):
    def __init__(self, in_features, num_act):
        super().__init__()
        self.fc1 = torch.nn.Linear(in_features, 100)
        self.fc2 = torch.nn.Linear(100, 50)
        self.fc3 = torch.nn.Linear(50, 25)
        self.fc4 = torch.nn.Linear(25, num_act)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)
        x = torch.tanh(x)
        x = self.fc4(x)
        x = torch.tanh(x)
        return x


env = ant_v3.AntEnv(ctrl_cost_weight=1E-6, contact_cost_weight=1E-3, healthy_reward=0.05)
observation = torch.tensor(env.reset())  # 游戏回到初始状态
policy = FNNclassifier(111, 8)
value = FNNclassifier(111, 8)
policy = policy.double()
batch_size = 2000
reward_old = 0
reward_new = 0
for iteration in range(100):
    for t_step in range(batch_size):
        env.render()
        tensor_action = policy.forward(observation)
        action = tensor_action.detach().numpy()
        observation_np, reward, done, info = env.step(action)
        reward_new = reward
        observation = torch.tensor(observation_np)
        if done:
            pass

env.close()
# recorder.close()

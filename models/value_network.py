import torch
from models import policy_network
from agents import ant
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

class Value_Net(torch.nn.Module):
    def __init__(self, ob_dim, value_dim):
        if value_dim is None: value_dim = 1

        # Define the network architecture
        super(Value_Net, self).__init__()
        self.fc1 = torch.nn.Linear(ob_dim, 100)
        self.fc2 = torch.nn.Linear(100, 50)
        self.fc3 = torch.nn.Linear(50, 25)
        self.fc4 = torch.nn.Linear(25, value_dim)
        # self.tanh1 = torch.nn.Tanh()
        # self.tanh2 = torch.nn.Tanh()
        # self.tanh3 = torch.nn.Tanh()

    def forward(self, obs):
        obs = torch.Tensor(obs)
        x = self.fc1(obs)
        # x = self.tanh1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        # x = self.tanh2(x)
        x = self.fc3(x)
        x = torch.tanh(x)
        # x = self.tanh3(x)
        x = self.fc4(x)
        return x

    def net_initial(self, agent, policy_net, max_iter=30, lr=0.01):
        # initial the value network for the first time
        dic = agent.get_traj_per_batch(policy_net)
        reward = dic['rews']
        ep_len = dic['ep_len']
        obs = dic['ob']
        values = dic['values']
        reward_tensor = torch.Tensor(reward)
        dataset = TensorDataset(torch.Tensor(obs), values)

        loss_func = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for iter in range(max_iter):
            dataloader = DataLoader(dataset, batch_size=500, shuffle=True)
            for batch_idx, (data, target) in enumerate(dataloader):
                optimizer.zero_grad()
                predict = self.forward(data)
                loss = loss_func(predict.flatten(), target)
                loss.backward()
                optimizer.step()
        predict_test = self.forward(obs)
        loss = loss_func(predict_test.flatten(), values)
        print('loss_initial %f' % loss)

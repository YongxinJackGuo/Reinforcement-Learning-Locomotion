import torch
import math
from utils import common as U

class Policy_Net(torch.nn.Module):
    # TODO: The output of policy net is mean (mu) and standard deviation (std)
    #  of a batch of state input.
    def __init__(self, ob_dim, ac_dim):

        # Define the network architecture
        super().__init__()
        self.fc1 = torch.nn.Linear(ob_dim, 100)
        self.fc2 = torch.nn.Linear(100, 50)
        self.fc3 = torch.nn.Linear(50, 25)
        self.fc4 = torch.nn.Linear(25, ac_dim)
        # self.tanh1 = torch.nn.Tanh()
        # self.tanh2 = torch.nn.Tanh()
        # self.tanh3 = torch.nn.Tanh()
        # self.tanh4 = torch.nn.Tanh()

    def forward(self, obs):
        obs = torch.Tensor(obs)
        x = self.fc1(obs)
        x = torch.tanh(x)
        # x = self.tanh1(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        # x = self.tanh2(x)
        x = self.fc3(x)
        x = torch.tanh(x)
        # x = self.tanh3(x)
        x = self.fc4(x)
        mu = x
        log_std = torch.zeros_like(mu)-0.5
        std = torch.exp(log_std)
        return mu, std

    # Compute KL divergence between two Gaussian distributions
    def get_KL(self, obs):
        # TODO； compute KL-divergence using Equation:
        #  KL(p, q) = log(std_q / std_p) + (std_p + (mean_p - mean_q)^2) / 2std_q^2 - 0.5
        # Compute mu, std of next policy as variable. (Requires grad)
        mu_next, std_next = self.forward(obs)
        # Compute mu, std of current policy as fixed constant. (Detached from autograd)
        mu_cur, std_cur = mu_next.detach(), std_next.detach()
        # Compute KL divergence of next policy distributin w.r.t current policy distribution
        KL = torch.log(std_cur / std_next) + (std_next + (mu_next - mu_cur).pow(2)) / (2 * std_cur.pow(2)) - 0.5

        return KL.sum(dim=1)  # sum up over each batch. The return dim will be N.(batch size)

    def get_log_prob(self, obs, acs):
        acs = torch.from_numpy(acs)
        PI = torch.tensor([math.pi])
        mu, std = self.forward(obs)
        # Compute log probability of actions within a batch
        log_prob = -torch.log(std * torch.sqrt(2 * PI)) - 0.5 * ((acs - mu) / std).pow(2)
        return log_prob

    def sample_action(self, obs):
        mu, std = self.forward(obs)
        selected_ac = torch.normal(mu, std)

        return selected_ac

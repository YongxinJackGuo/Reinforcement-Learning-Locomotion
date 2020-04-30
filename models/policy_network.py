import torch
import math
from utils import common as U

class Policy_Net():
    # TODO: The output of policy net is mean (mu) and standard deviation (std)
    #  of a batch of state input.
    def __init__(self, ob_dim, ac_dim, *hid_layers):
        if hid_layers is None: hid_layers = [128, 128]

        # Define the network architecture

    def forward(self, obs):
        mu = None
        std = None
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
        PI = torch.tensor([math.pi])
        mu, std = self.forward(obs)
        # Compute log probability of actions within a batch
        log_prob = -torch.log(mu * torch.sqrt(2 * PI)) - 0.5 * ((acs - mu) / std).pow(2)
        return log_prob

    def sample_action(self, obs):
        mu, std = self.forward(obs)
        selected_ac = torch.normal(mu, std)

        return selected_ac

import numpy as np
from gym import utils
from gym.envs.mujoco import ant_v3
from agents.ant import Ant
from models.policy_network import Policy_Net
from models.value_network import Value_Net
import torch

seed = 42
torch.manual_seed(seed)
np.random.seed(42)

class Args():
    def __init__(self):
        self.max_iters = None
        self.horizon = None  # Batch size: time steps per batch
        self.l2_reg = None  # L2 regularization lambda for value loss function
        self.max_KL = None  # Max KL divergence threshold for TRPO update
        # Environemnt
        self.env = ant_v3.AntEnv(ctrl_cost_weight=1E-6, contact_cost_weight=1E-3, healthy_reward=0.05)
        self.env.seed(seed)
        self.agent = Ant(self.env, self.horizon)  # create agent
        self.pi_net = Policy_Net(self.agent.ob_dim, self.agent.ac_dim)  # Create Policy Network
        self.value_net = Value_Net(self.agent.ob_dim, self.agent.ac_dim)  # Create Value Network
        self.value_net_lr = None  # Declare value net learning rate
        self.LBFGS_iters = None  # Declare the number of update times for value_net parameters in one TRPO update
        self.cg_iters = 10  # Declare the number of iterations for conjugate gradient algorithm
        self.cg_threshold = 1e-10  # Eearly stopping threshold for conjugate gradient
        self.line_search_alpha = None  # Line search decay rate for TRPO
        self.search_iters = 10  # Line search iterations
args = Args()



def update():
    return





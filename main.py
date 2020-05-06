import numpy as np
from gym import utils
from gym.envs.mujoco import ant_v3
from gym.envs.mujoco import Walker2dEnv
from gym.envs.mujoco import HopperEnv
from agents.ant import Ant
from models.policy_network import Policy_Net
from models.value_network import Value_Net
import torch
from utils import common
from core import trpo
from matplotlib import pyplot as plt
import time

# seed = 42
# torch.manual_seed(seed)
# np.random.seed(42)


class Args():
    def __init__(self):
        self.max_iters = 1000
        self.horizon = 100000 # Batch size: time steps per batch
        self.episode_long = 1000
        self.l2_reg = 1E-4  # L2 regularization lambda for value loss function
        self.max_KL = 0.01 # Max KL divergence threshold for TRPO update
        # Environemnt
        self.env = Walker2dEnv() #ant_v3.AntEnv(ctrl_cost_weight=1E-6, contact_cost_weight=1E-3, healthy_reward=0.05)
        # self.env.seed(seed)
        self.agent = Ant(self.env, self.horizon,self.episode_long)  # create agent
        self.pi_net = Policy_Net(self.agent.ob_dim, self.agent.ac_dim)  # Create Policy Network
        self.value_net = Value_Net(self.agent.ob_dim, 1)  # Create Value Network
        self.value_net_lr = 1  # Declare value net learning rate
        self.LBFGS_iters = 20  # Declare the number of update times for value_net parameters in one TRPO update
        self.cg_iters = 500  # Declare the number of iterations for conjugate gradient algorithm
        self.cg_threshold = 1e-2  # Eearly stopping threshold for conjugate gradient
        self.line_search_alpha = 0.5  # Line search decay rate for TRPO
        self.search_iters = 10  # Line search iterations
        self.gamma = 1
        self.Lambda = 0.95



def train(args):
    env = args.env
    value_net = args.value_net
    pi_net = args.pi_net
    agent = args.agent
    Lambda = args.Lambda
    gamma = args.gamma
    value_net.net_initial(agent, pi_net)
    value_list = list()
    len_list = list()
    for iters in range(args.max_iters):
        bacth_dic = agent.get_traj_per_batch(pi_net, value_net)#.__next__()
        lens = sum(bacth_dic['ep_len']) / len(bacth_dic['ep_len'])
        value = sum(bacth_dic['rews']) / len(bacth_dic['ep_len'])
        print("len %f" % lens)
        print(value)
        len_list.append(lens)
        value_list.append(value)
        adv, Q = common.get_adv(bacth_dic['rews'], gamma, Lambda, bacth_dic['ep_len'],
                                bacth_dic['vpreds'])
        success, value_net, pi_net = trpo.trpo_update(pi_net, value_net, bacth_dic['ac'], Q,
                                                      bacth_dic['ob'], adv, args)
        # assert success is True, "Linear Search False"
        print('iter %d' % (iters))
        print(success)
        if iters % 100 == 0:
            np.savetxt('hopper.csv', [np.array(value_list)], delimiter=',', fmt='%f')
            np.savetxt('hopper_len.csv', [np.array(len_list)], delimiter=',', fmt='%f')
    args.value_net = value_net
    args.pi_net = pi_net
    return args

def show(args, iter=10000):
    env = args.env
    pi_net = args.pi_net
    observation = env.reset()
    for i in range(iter):
        env.render()
        action = pi_net.sample_action(observation).detach().numpy()
        observation_np, reward, done, info = env.step(action)
        if done:
            print(i)
            break
    env.close()


args = Args()
# show(args)
args_after_train = train(args)

for i in range(10):
    # show(args)
    show(args_after_train)



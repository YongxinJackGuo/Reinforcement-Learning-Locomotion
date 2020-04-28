import torch
from utils import common as CM

class Policy_Net():
    # TODO: The output of policy net is mean (mu) and standard deviation (std)
    #  of a batch of state input
    def __init__(self, ob_dim, ac_dim, *hid_layers):
        if hid_layers is None: hid_layers = [128, 128]

        return None

    def forward(self):
        return None

    def get_KL(self):
        CM.compute_KL()
        return None

    def get_neg_log(self):

        return None

    def sample_action(self):

        return None

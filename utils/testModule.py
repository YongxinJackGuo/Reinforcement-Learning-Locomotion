import torch
from utils import common as U


def update(pi):
    cur_params = U.get_flat_param(pi)
    new_params = 10 * cur_params
    U.set_flat_param(pi, new_params)



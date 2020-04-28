# Common utilty functions

import numpy as np

# Compute KL divergence between two Gaussian distributions
def compute_KL(p, q):
    # TODO； compute KL-divergence using Equation:
    #  KL(p, q) = log(std_q / std_p) + (std_p + (mean_p - mean_q)^2) / 2std_q^2 - 0.5

    return (p + q)

# Perform conjugate gradient
def cg(Ax, b, cg_iter, threshold):
    # TODO: compute conjugate gradient (cg) descent

    return None

# Perform Hessian-vector product
def compute_hvp():
    # TODO: compute Hessian-vector product from KL
    #  In this case, the Hessian matrix is Fisher Information Matrix (FIM)

    return None

# Get General Advantage Estimate (GAE)
def get_adv():
    # TODO: get estimate advantage function GAE


    return None

# Perform backtracking line search with exponential decay to obtain final update
def line_search():
    # TODO: Perfrom line search at each TRPO

    return None


class compute_Test:
    def __init__(self, a):
        self.a = a

    def get_value(self):
        return self.a




# Common utilty functions
import torch
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
    # TODO: Perform line search at each TRPO

    return None

def get_flat_param(model):
    # TODO: Get the network parameters
    flat_param = torch.cat([param.view(-1) for param in model.parameters()])

    return flat_param

def set_flat_param(model, flat_param):
    # TODO: Set the network parameters
    
    return None

def get_flat_grad(model):
    # TODO: Get the network parameter gradients
    flat_grad = torch.cat([param.grad.view(-1) for param in model.parameters()])

    return None

def set_flat_grad(model, flat_grad):
    # TODO: Set the network parameter gradients

    return None

class compute_Test:
    def __init__(self, a):
        self.a = a

    def get_value(self):
        return self.a




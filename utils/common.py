# Common utilty functions
import torch
import numpy as np


# Perform conjugate gradient
def cg(b, kl, pi, cg_iter, threshold):
    # TODO: compute conjugate gradient (cg) descent
    # Initialize x
    x = torch.zeros(b.shape)
    # r is non-conjugate component, also the residual, d is conjugate component
    r = d = b.clone()  # Originally, it is r = d = b - Hx but Hx is zero anyway as the initialization
    for _ in range(cg_iter):
        beta = -1 * torch.dot(d, r) / torch.dot(d, compute_hvp(kl, d, pi))  # stepsize along for solution
        x += beta * d  # update solution along conjugate direction
        r_prev = r # store old r
        r -= beta * compute_hvp(kl, d, pi)  # update non-conjugate direction
        delta = torch.dot(r, r) / torch.dot(r_prev, r_prev)  # update stepsize for conjugate direction
        d = r - torch.dot(delta, d)
        if r < threshold:
            break
    return x

# Perform Hessian-vector product
def compute_hvp(kl, v, pi):
    # TODO: compute Hessian-vector product from KL
    #  In this case, the Hessian matrix is Fisher Information Matrix (FIM)
    kl = kl.mean()
    grad = torch.autograd.grad(kl, pi.parameters(), create_graph=True)  # Compute d(KL)/dx
    flat_grad = torch.cat([g.view(-1) for g in grad])  # Flatten
    grad_v = torch.dot(flat_grad, v)  # Compute d(KL)/dx * v
    grad_grad_v = torch.autograd.grad(grad_v, pi.parameters())  # Compute d/dx * d(KL)/dx * v. Hessian-vector product
    flat_grad_grad_v = torch.cat([gg.view(-1) for gg in grad_grad_v])  # Flatten

    return flat_grad_grad_v  # Can add a damping term to it

# Compute KL divergence between two different policy nets
def compute_KL(new_stats, old_stats):
    """
    :param new_stats: Tuple of size 2: (mean, std). Mean and std are both torch.tensors with with size N-by-acs_dim.
    :param old_stats: Tuple of size 2: (mean, std). Mean and std are both torch.tensors with with size N-by_acs_dim.
    :return: A scalar KL value.
    """
    # TODO； compute KL-divergence using Equation, this function evaluates two policy
    #  distribution similarity for line search.
    #  KL(p, q) = log(std_q / std_p) + (std_p + (mean_p - mean_q)^2) / 2std_q^2 - 0.5
    mu_new, std_new = new_stats[0], new_stats[1]
    mu_old, std_old = old_stats[0], old_stats[1]
    # Compute KL divergence of next policy distributin w.r.t current policy distribution
    kl = torch.log(std_old / std_new) + (std_new + (mu_new - mu_old).pow(2)) / (2 * std_old.pow(2)) - 0.5

    return kl.sum(dim=1).mean()

# Get General Advantage Estimate (GAE)
def get_adv():
    # TODO: get estimate advantage function GAE

    # Remember to normalize the advantages for training stability. (x - mu) / std

    return

# Perform backtracking line search with exponential decay to obtain final update
def line_search(stepsize, pi, obs, acs, args):
    # TODO: Perform line search at each TRPO

    alpha, search_iters, max_KL = args.line_search_alpha, args.search_iters, args.max_KL
    cur_params = get_flat_param(pi)  # get initial policy stats
    old_stats = (pi(obs))  # put initial policy stats into a tuple
    old_log_prob = pi.get_log_prob(obs, acs)  # old policy log probability for actions

    for alpha in [alpha**i for i in range(search_iters)]:
        new_stepsize = stepsize * alpha
        new_params = cur_params + new_stepsize
        set_flat_param(pi, new_params)  # update the model
        new_stats = (pi(obs))  # store new model stats
        new_log_prob = pi.get_log_prob(obs, acs)  # new policy log probability for actions

        # compute criteria
        L = (new_log_prob - old_log_prob).mean()
        kl = compute_KL(old_stats, new_stats)
        if kl < 1.5 * max_KL and L > 0:  # Trust Region Condition
            print('Step {:.2f} size accepted'.format(new_stepsize))
            return True, new_params  # Step size accept

    return False, new_params

def get_flat_param(model):
    # TODO: Get the network parameters
    flat_param = torch.cat([param.view(-1) for param in model.parameters()])  # flatten

    return flat_param

def set_flat_param(model, flat_param):
    # TODO: Set the network parameters
    count = 0
    for param in model.parameters():
        num_elem = param.numel()
        param.data.copy_(flat_param[count: count + num_elem].reshape_as(param))
        count = num_elem

def get_flat_grad(model):
    # TODO: Get the network parameter gradients
    flat_grad = torch.cat([param.grad.view(-1) for param in model.parameters()])

    return None

def set_flat_grad(model, flat_grad):
    # TODO: Set the network parameter gradients

    return None






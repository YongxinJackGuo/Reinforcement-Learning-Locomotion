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
    kl = kl.mean()
    grad = torch.autograd.grad(kl, pi.parameters(), create_graph=True)
    for _ in range(cg_iter):
        hvp = compute_hvp(kl, d, pi, grad)
        beta = torch.dot(r, r) / torch.dot(d, hvp)  # stepsize along for solution
        x += beta * d  # update solution along conjugate direction
        r_prev = r.clone() # store old r
        r -= beta * hvp  # update non-conjugate direction
        r_curr = torch.dot(r, r)
        delta = r_curr / torch.dot(r_prev, r_prev)  # update stepsize for conjugate direction
        d = r - delta * d

        if r_curr < threshold:
            return x
    print(r_curr)
    return False

# Perform Hessian-vector product
def compute_hvp(kl, v, pi, grad, damping = 0.01):
    # TODO: compute Hessian-vector product from KL
    #  In this case, the Hessian matrix is Fisher Information Matrix (FIM)
    # grad = torch.autograd.grad(kl, pi.parameters(), create_graph=True)  # Compute d(KL)/dx
    flat_grad = torch.cat([g.view(-1) for g in grad])  # Flatten
    grad_v = torch.dot(flat_grad, v)  # Compute d(KL)/dx * v
    grad_grad_v = torch.autograd.grad(grad_v, pi.parameters(), retain_graph=True)  # Compute d/dx * d(KL)/dx * v. Hessian-vector product
    flat_grad_grad_v = torch.cat([gg.contiguous().view(-1) for gg in grad_grad_v])  # Flatten
    return flat_grad_grad_v + v * damping# Can add a damping term to it

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
    # print(mu_new)
    # print('-----------------------------')
    mu_old, std_old = old_stats[0], old_stats[1]
    # print(mu_old)
    # Compute KL divergence of next policy distributin w.r.t current policy distribution
    kl = torch.log(std_old / std_new) + (std_new.pow(2) + (mu_new - mu_old).pow(2)) / (2 * std_old.pow(2)) - 0.5
    # print(kl)
    return kl.sum(dim=1).mean()

# Get General Advantage Estimate (GAE)
def get_adv(rwd, gamma, Lambda, ep_len, value):
    # TODO: get estimate advantage function GAE
    # Remember to normalize the advantages for training stability. (x - mu) / std
    delta = torch.zeros_like(value)
    adv = torch.zeros_like(value)
    pass_time = 0
    for ep in range(len(ep_len)):
        temp_gae = 0
        for i in reversed(range(ep_len[ep])):
            if i == ep_len[ep]-1:
                delta[i + pass_time] = rwd[i+pass_time] - value[i + pass_time]
            else:
                delta[i + pass_time] = rwd[i + pass_time] + gamma * value[i+1+pass_time] - value[i+pass_time]
            temp_gae = temp_gae * gamma * Lambda + delta[i+pass_time]
            adv[i+pass_time] = temp_gae
        pass_time += ep_len[ep]
    mean, std = adv.mean(), adv.std()
    norm_adv = (adv - mean) / std
    Q = adv + value
    return norm_adv, Q

# Perform backtracking line search with exponential decay to obtain final update
def line_search(stepsize, pi, obs, acs, args, adv):
    # TODO: Perform line search at each TRPO
    alpha, search_iters, max_KL = args.line_search_alpha, args.search_iters, args.max_KL
    cur_params = get_flat_param(pi)  # get initial policy stats
    old_params = cur_params.clone()
    old_stats = (pi(obs))  # put initial policy stats into a tuple
    valid_indices = torch.where(adv > 0)[0]
    invalid_indices = torch.where(adv < 0)[0]
    obs_for_prob = torch.Tensor(obs)[valid_indices, :].detach().numpy()
    acs_for_prob = torch.Tensor(acs)[valid_indices, :].detach().numpy()
    obs_for_prob_n = torch.Tensor(obs)[invalid_indices, :].detach().numpy()
    acs_for_prob_n = torch.Tensor(acs)[invalid_indices, :].detach().numpy()
    old_log_prob = pi.get_log_prob(obs_for_prob, acs_for_prob)  # old policy log probability for actions
    old_log_prob_n = pi.get_log_prob(obs_for_prob_n, acs_for_prob_n)
    for alpha in [alpha**i for i in range(search_iters)]:
        new_stepsize = stepsize * alpha
        new_params = cur_params + new_stepsize
        set_flat_param(pi, new_params)  # update the model
        new_stats = (pi(obs))  # store new model stats
        # print(new_stats)
        new_log_prob = pi.get_log_prob(obs_for_prob, acs_for_prob)  # new policy log probability for actions
        new_log_prob_n = pi.get_log_prob(obs_for_prob_n, acs_for_prob_n)
        # compute criteria
        L = (new_log_prob - old_log_prob).mean()
        L_n = (new_log_prob_n - old_log_prob_n).mean()
        kl = compute_KL(new_stats, old_stats)
        # print(kl)
        # print(L_n)
        if kl < 1.5 * max_KL and L > 0 and L_n <0:  # Trust Region Condition
            print('Step {:.6f} size accepted'.format(new_stepsize.detach().numpy()[0]))
            return True, new_params  # Step size accept

    return False, old_params

def get_flat_param(model):
    # TODO: Get the network parameters
    flat_param = torch.cat([param.view(-1) for param in model.parameters()])

    return flat_param

def set_flat_param(model, flat_param):
    # TODO: Set the network parameters
    count = 0
    for param in model.parameters():
        num_elem = param.numel()
        param.data.copy_(flat_param[count: count + num_elem].reshape_as(param))
        count += num_elem

def get_flat_grad(model):
    # TODO: Get the network parameter gradients
    flat_grad = torch.cat([param.grad.view(-1) for param in model.parameters()])

    return None

def set_flat_grad(model, flat_grad):
    # TODO: Set the network parameter gradients

    return None






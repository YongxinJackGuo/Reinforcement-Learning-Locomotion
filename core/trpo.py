from torch import optim
import torch.nn as nn
import torch
from utils import common as U
import scipy.optimize

# Trust Region Policy Optimization Update per Batch

def get_flat_params_from(model):
    params = []
    for param in model.parameters():
        params.append(param.view(-1))

    flat_params = torch.cat(params)
    return flat_params



def trpo_update(policy_net, value_net, batch_actions, batch_values, batch_states, adv, args):

    value_net_lr, l2_reg, max_KL = args.value_net_lr, args.l2_reg, args.max_KL
    LBFGS_iters, cg_iters, cg_threshold = args.LBFGS_iters, args.cg_iters, args.cg_threshold

    # def get_flat_grad_from(inputs, grad_grad=False):
    #     grads = []
    #     for param in inputs:
    #         if grad_grad:
    #             grads.append(param.grad.grad.view(-1))
    #         else:
    #             if param.grad is None:
    #                 grads.append(torch.zeros(param.view(-1).shape))
    #             else:
    #                 grads.append(param.grad.view(-1))
    #
    #     flat_grad = torch.cat(grads)
    #     return flat_grad
    #
    # def get_flat_params_from(model):
    #     params = []
    #     for param in model.parameters():
    #         params.append(param.view(-1))
    #
    #     flat_params = torch.cat(params)
    #     return flat_params
    #
    #
    #
    # def get_value_loss(flat_params):
    #     U.set_flat_param(value_net, torch.Tensor(flat_params))
    #     for param in value_net.parameters():
    #         if param.grad is not None:
    #             param.grad.data.fill_(0)
    #     values_pred = value_net(batch_states)
    #     value_loss = (values_pred - batch_values).pow(2).mean()
    #
    #     # weight decay
    #     for param in value_net.parameters():
    #         value_loss += param.pow(2).sum() * l2_reg
    #     value_loss.backward(retain_graph=True)
    #     return value_loss.item(), get_flat_grad_from(value_net.parameters()).detach().numpy().astype(float)
    #
    # flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss,
    #                                                         get_flat_params_from(value_net).detach().numpy(),
    #                                                         maxiter=25)
    # U.set_flat_param(value_net, torch.Tensor(flat_params))



    # TODO: compute the policy_net loss function for computing the gradient g later on
    #-------------Compute the policy gradient first---------------------
    # Policy probability at the next state. Requires gradient
    log_prob_next = policy_net.get_log_prob(batch_states, batch_actions)
    # Policy probability at current state. Detached from autograd
    log_prob_cur = policy_net.get_log_prob(batch_states, batch_actions).detach()

    # Construct a policy loss Function: L = (pi_prob_next / pi_prob_cur) * Advantages
    L_policy = (torch.exp(log_prob_next - log_prob_cur)).t() * adv# negative sign for minimization convention
    L_policy = L_policy.sum()/adv.shape[0]

    # Compute the policy gradient
    pi_grads = torch.autograd.grad(L_policy, policy_net.parameters())
    pi_grads = torch.cat([pi_grad.view(-1) for pi_grad in pi_grads])  # Flatten the gradient


    # TODO: update the actor, the policy_net parameter. Cannot used optimizer here.
    #  Main functions used here: cg(), compute_hvp()

    # Get KL-divergence as pytorch Variable
    KL = policy_net.get_KL(batch_states)
    # Compute x = H^-1 * g using conjugate gradient method
    x = U.cg(pi_grads, KL, policy_net, cg_iters, cg_threshold)  # -pi_grads because we took a minus sign at L before
    if x is False:
        return False, value_net, policy_net
    # Compute Hx
    KL = KL.mean()
    grad = torch.autograd.grad(KL, policy_net.parameters(), create_graph=True)
    Hx = U.compute_hvp(KL, x, policy_net, grad)
    stepsize = torch.sqrt(2 * max_KL / (x.dot(Hx))) * x

    # Backtracking line search with exponential dacay
    success, update_net_param = U.line_search(stepsize, policy_net, batch_states, batch_actions, args, adv)
    U.set_flat_param(policy_net, flat_param=update_net_param)  # Update the policy net with accepted network parameters

    # TODO: update the critic, the value_net parameters. One optimizer_val here.
    # Use L-BFGS algorithm for minimizing the value loss function (限域拟牛顿法)

    # define a loss function for value network
    # Run L-BFGS couple times

    dic = args.agent.get_traj_per_batch(policy_net)
    values = dic['values']
    val_optimizer = optim.LBFGS(value_net.parameters(), lr=value_net_lr, max_iter=20)
    val_MSELoss = nn.MSELoss()
    def closure():
        val_pred = value_net(batch_states)
        val_loss = val_MSELoss(val_pred.flatten(), values)
        # Compute L2 regularization term for value_net loss function
        for params in value_net.parameters():
            val_loss += l2_reg * params.pow(2).sum()
        val_optimizer.zero_grad()
        val_loss.backward(retain_graph=True)
        # print(val_loss)
        return val_loss
    val_optimizer.step(closure)  # update value_net parameters
    return success, value_net, policy_net
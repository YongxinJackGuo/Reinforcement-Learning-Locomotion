from torch import optim
import torch.nn as nn
import torch
from utils import common as U

# Trust Region Policy Optimization Update per Batch

def trpo_update(policy_net, value_net, batch_actions, batch_values, batch_states, adv, args):

    value_net_lr, l2_reg, max_KL = args.value_net_lr, args.l2_reg, args.max_KL
    LBFGS_iters, cg_iters, cg_threshold = args.LBFGS_iters, args.cg_iters, args.cg_threshold

    # TODO: compute the policy_net loss function for computing the gradient g later on
    #-------------Compute the policy gradient first---------------------
    # Policy probability at the next state. Requires gradient
    log_prob_next = policy_net.get_log_prob(batch_states, batch_actions)
    # Policy probability at current state. Detached from autograd
    log_prob_cur = policy_net.get_log_prob(batch_states, batch_actions).detach()

    # Construct a policy loss Function: L = (pi_prob_next / pi_prob_cur) * Advantages
    test = (torch.exp(log_prob_next - log_prob_cur))
    L_policy = -1 * (torch.exp(log_prob_next - log_prob_cur)).t() * adv# negative sign for minimization convention
    L_policy = L_policy.sum()/adv.shape[0]

    # Compute the policy gradient
    pi_grads = torch.autograd.grad(L_policy, policy_net.parameters())
    pi_grads = torch.cat([pi_grad.view(-1) for pi_grad in pi_grads])  # Flatten the gradient


    # TODO: update the actor, the policy_net parameter. Cannot used optimizer here.
    #  Main functions used here: cg(), compute_hvp()

    # Get KL-divergence as pytorch Variable
    KL = policy_net.get_KL(batch_states)
    # Compute x = H^-1 * g using conjugate gradient method
    x = U.cg(-pi_grads, KL, policy_net, cg_iters, cg_threshold)  # -pi_grads because we took a minus sign at L before
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
    val_optimizer = optim.LBFGS(value_net.parameters(), lr=value_net_lr, max_iter=20)
    # define a loss function for value network
    # Run L-BFGS couple times
    val_MSELoss = nn.MSELoss()
    dic = args.agent.get_traj_per_batch(policy_net)
    values = dic['values']
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
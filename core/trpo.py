from torch import optim
import torch.nn as nn
import torch

# Trust Region Policy Optimization Update per Batch

def trpo_update(policy_net, value_net, batch_actions, batch_values, adv, args):

    value_net_lr, l2_reg, max_KL = args.value_net_lr, args.l2_reg, args.max_KL
    LBFGS_iters = args.LBFGS_iters

    # TODO: update the critic, the value_net parameters. One optimizer_val here.
    # Use L-BFGS algorithm for minimizing the value loss function (限域拟牛顿法)
    val_optimizer = optim.LBFGS(value_net.parameters(), lr=value_net_lr)
    # Run L-BFGS couple times
    for _ in range(LBFGS_iters):
        val_pred = value_net(batch_actions)

        val_MSELoss = nn.MSELoss() # define a loss function for value network
        val_loss = val_MSELoss(val_pred, batch_values)

        # Compute L2 regularization term for value_net loss function
        for params in value_net.parameters():
            val_loss += l2_reg * params.pow(2)

        val_optimizer.zero_grad()
        val_loss.backward()
        val_optimizer.step()  # update value_net parameters


    # TODO: compute the policy_net loss function for computing the gradient g later on


    # TODO: update the actor, the policy_net parameter. Cannot used optimizer here.
    #  Functions used here: cg(), compute_hvp(),

    return None
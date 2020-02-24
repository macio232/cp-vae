from __future__ import print_function

import torch
from torch.autograd import Variable

import numpy as np
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# ======================================================================================================================
def train_vae(epoch, args, train_loader, model, optimizer):
    # set loss to 0
    train_loss = 0
    train_re = 0
    train_kl = 0
    train_kl_cont = 0
    train_kl_discr = 0
    # set model in training mode
    model.train()

    # start training

    anneal_rate = args.anneal_rate
    gumble_tau_min = args.gumbel_tau_min
    anneal_interval = args.anneal_interval

    if epoch % anneal_interval == 0:
        args.gumbel_tau = np.maximum(args.gumbel_tau * np.exp(-anneal_rate * epoch), gumble_tau_min)

    if args.warmup == 0:
        beta = args.beta
    else:
        beta = 1.* epoch / args.warmup
        if beta > 1.:
            beta = 1.
    print('beta: {}'.format(beta))
    print('temp: {}'.format(args.gumbel_tau))

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        # dynamic binarization

        if args.dynamic_binarization:
            x = torch.bernoulli(data)
        else:
            x = data

        # reset gradients
        optimizer.zero_grad()
        loss, RE, KL, KL_cont, KL_discr = model.calculate_loss(x, beta, average=True)
        # backward pass
        loss.backward()
        # optimization
        optimizer.step()

        train_loss += loss.item()
        train_re += -RE.item()
        train_kl += KL.item()
        if args.prior == 'MoG' or args.prior ==  'standard':
            train_kl_discr = 0.0
            train_kl_cont = 0.0
        else:

            train_kl_cont += KL_cont.item()
            train_kl_discr += KL_discr.item()

    # calculate final loss
    train_loss /= len(train_loader)  # loss function already averages over batch size
    train_re /= len(train_loader)  # re already averages over batch size
    train_kl /= len(train_loader)  # kl already averages over batch size
    train_kl_cont /= len(train_loader)  # kl of continuous latent already averages over batch size
    train_kl_discr /= len(train_loader)  # kl of discrete latent already averages over batch size

    return model, train_loss, train_re, train_kl, train_kl_cont, train_kl_discr

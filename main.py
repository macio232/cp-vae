from __future__ import print_function
import argparse

import torch
import torch.optim as optim

from utils.nn import AdamNormGrad

import os

import datetime
from utils.load_data import load_dataset

# # # # # # # # # # #
# START EXPERIMENTS # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # #
#

# Training settings
parser = argparse.ArgumentParser(description='CP-VAE')

parser.add_argument('--fixed_var', action='store_true', default=False,
                    help='Identity var matrix')

# arguments for optimization
parser.add_argument('--batch_size', type=int, default=100, metavar='BStrain',
                    help='input batch size for training (default: 100)')
parser.add_argument('--test_batch_size', type=int, default=100, metavar='BStest',
                    help='input batch size for testing (default: 100)')
parser.add_argument('--epochs', type=int, default=10, metavar='E',
                    help='number of epochs to train (default: 2000)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')
parser.add_argument('--early_stopping_epochs', type=int, default=50, metavar='ES',
                    help='number of epochs for early stopping')

parser.add_argument('--warmup', type=int, default=100, metavar='WU',
                    help='number of epochs for warmu-up')
# warmup is used to boost the generative capacity of the decoder

parser.add_argument('--beta', type=float, default= 1.0,
                    help='hyperparameter to scale KL term in ELBO')

# Gumbel-Softmax parametes

parser.add_argument('--temp', type=float, default=1.0, metavar='TEMP',
                    help='Gumbel-Softmax initial temperature (default: 1.0)')

parser.add_argument('--temp_min', type=float, default=0.5, metavar='TEMP_MIN',
                    help='minimum Gumbel-Softmax temperature (default: 0.5)')

parser.add_argument('--anneal_rate', type=float, default=0.00003, metavar='ANR',
                    help='annealing rate for Gumbel-Softmax (default: 0.00003)')
parser.add_argument('--anneal_interval', type=float, default=1, metavar='ANIN',
                    help='annealing interval for Gumbel-Softmax  (default: 100)')


# cuda
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
# random seed
parser.add_argument('--seed', type=int, default=14, metavar='S',
                    help='random seed (default: 14)')

# model: latent size, input_size, so on
parser.add_argument('--z1_size', type=int, default=10, metavar='M1',
                    help='latent size')
parser.add_argument('--z2_size', type=int, default=0, metavar='M2',
                    help='latent size')
parser.add_argument('--disc_size', type=int, default=10, metavar='D1',
                    help='discrete latent size/dim')

parser.add_argument('--input_size', type=int, default=[1, 28, 28], metavar='D',
                    help='input size')
parser.add_argument('--hidden_size', type=int, default= 300, metavar='D',
                    help='hidden size')

parser.add_argument('--activation', type=str, default=None, metavar='ACT',
                    help='activation function')

parser.add_argument('--gumbel_hard', action='store_true', default=True,
                    help='Use if sample class instead of soft-max')


# model: model name, prior
parser.add_argument('--model_name', type=str, default='cp_vae_z_xc', metavar='MN',
                    help='model name: vae, joint_vae, cp_vae, cp_vae_z_xc')

parser.add_argument('--prior', type=str, default='conditional', metavar='P',
                    help='prior: standard, conditional, MoG, joint')

parser.add_argument('--input_type', type=str, default='binary', metavar='IT',
                    help='type of the input: binary, gray, continuous')

# experiment
parser.add_argument('--S', type=int, default=5000, metavar='SLL',
                    help='number of samples used for approximating log-likelihood')
parser.add_argument('--MB', type=int, default=100, metavar='MBLL',
                    help='size of a mini-batch used for approximating log-likelihood')

# dataset
parser.add_argument('--dataset_name', type=str, default='dynamic_mnist', metavar='DN',
                    help='name of the dataset: static_mnist, dynamic_mnist, omniglot, caltech101silhouettes,'
                         ' histopathologyGray, freyfaces, cifar10, gaussian_data')

parser.add_argument('--gaussian_mean1', type=float, default=0.1, metavar='GM1',
                    help='mean for gaussian data')
parser.add_argument('--gaussian_var1', type=float, default=0.1, metavar='GV1',
                    help='var for gaussian data')

parser.add_argument('--gaussian_mean2', type=float, default=0.99, metavar='GM2',
                    help='mean for gaussian data')
parser.add_argument('--gaussian_var2', type=float, default=0.1, metavar='GV2',
                    help='var for gaussian data')


parser.add_argument('--dynamic_binarization', action='store_true', default=False,
                    help='allow dynamic binarization')

# reconstruction'
# TODO: Check if actually does anything
parser.add_argument('--no_recon_oneHot', action='store_true', default=False,
                    help='enables to pick the most likely sample to reconstruct the input (in validation - test)')
# so if you want to reconstruct the input using prob of each category write in the comment line: --no_recon_oneHot.


# Capacity gamma|KL -C| common for continuous and discrete
parser.add_argument('--use_capacity', action='store_true', default=False,
                    help='gamma|KL - C| parameter fot KL-term')
# so if you want to use gamma|KL -C| write in the comment line: --use_capacity

parser.add_argument('--gamma', type=float, default=30, metavar='GAMMA',
                    help='gamma parameter for capacity  (default: 30)')
parser.add_argument('--num_iter', type=int, default=25000,
                    help= 'when to stop increasing the capacity (default: 25000)')


# Capacity continuous lattent variable
parser.add_argument('--min_capacity_cont', type=float, default=0.0, metavar='MIN_CAPc',
                    help='minimum capacity of continuous latent variable (default: 0)')
parser.add_argument('--max_capacity_cont', type=float, default=5.0, metavar='MAX_CAPc',
                    help='maximum capacity of continuous latent variable (default: 0.0)')


# Capacity discrete lattent variable
parser.add_argument('--min_capacity_discr', type=float, default=0.0, metavar='MIN_CAPc',
                    help='minimum capacity of discrete latent variable (default: 0.5)')
parser.add_argument('--max_capacity_discr', type=float, default=15.0, metavar='MAX_CAPc',
                    help='maximum capacity of discrete latent variable (default: 0.0)')

# visualization
parser.add_argument('--latent', action='store_true', default=True,
                    help='allow latent space visualization')


args = parser.parse_args()
args.recon_oneHot = not args.no_recon_oneHot
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def run(args, kwargs):
    args.model_signature = str(datetime.datetime.now())[0:19]

    if args.fixed_var:
        model_name = args.dataset_name + '_' + args.model_name + '_' + args.prior + '_wu(' + str(args.warmup) + ')' + '_z1_' + str(args.z1_size) + '_c_' \
                     + str(args.disc_size) +'_beta_'+ str(args.beta) + '_lr_' + str(args.lr)+ '_fixed_var_mean' + str(args.fixed_var_mean) + '_ghard_' + str(args.gumbel_hard)

    elif args.use_capacity:
        model_name = args.dataset_name + '_' + args.model_name + '_' + args.prior + '_wu(' + str(args.warmup) + ')' + '_z1_' + str(args.z1_size) + '_c_' + str(args.disc_size)\
                     + '_lr_' + str(args.lr) + '_gamma_' + str(args.gamma) + '_num_iter_' + str(args.num_iter) \
                     + '_capacity_cont_' + str( args.max_capacity_cont) + '_capacity_discr_' + str(args.max_capacity_discr) + '_ghard_' + str(args.gumbel_hard)


    else:
        model_name = args.dataset_name + '_' + args.model_name + '_' + args.prior + '_wu(' + str(args.warmup) + ')' + '_z1_' + str(args.z1_size) + '_c_' \
                     + str(args.disc_size) +'_beta_'+ str(args.beta) + '_lr_' + str(args.lr) + '_ghard_' + str(args.gumbel_hard)


    # DIRECTORY FOR SAVING
    snapshots_path = 'snapshots/'
    dir = snapshots_path + args.model_signature + '_' + model_name +  '/'

    if not os.path.exists(dir):
        os.makedirs(dir)

    # LOAD DATA=========================================================================================================
    print('load data')

    # loading data
    train_loader, val_loader, test_loader, args = load_dataset(args, **kwargs)

    # CREATE MODEL======================================================================================================
    print('create model')
    # importing model
    if args.model_name == 'vae':
        from models.vae import VAE
    elif args.model_name == 'cp_vae':
        from models.cp_vae import VAE
    elif args.model_name == 'cp_vae_z_xc':
        from models.cp_vae_z_xc import VAE
    elif args.model_name == 'joint_vae':
        from models.joint_vae import VAE
    else:
        raise Exception('Wrong name of the model!')

    model = VAE(args)
    if args.cuda:
        model.cuda()

    optimizer = AdamNormGrad(model.parameters(), lr=args.lr)

    # ======================================================================================================================
    print(args)
    with open(dir + 'vae_experiment_log.txt', 'a') as f:
        print(args, file=f)

    # ======================================================================================================================
    print('perform experiment')
    from utils.experiments import experiment_vae
    experiment_vae(args, train_loader, val_loader, test_loader, model, optimizer, dir, model_name = args.model_name)
    # ======================================================================================================================
    print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
    with open(dir + 'vae_experiment_log.txt', 'a') as f:
        print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-\n', file=f)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#

if __name__ == "__main__":
    run(args, kwargs)

# # # # # # # # # # #
# END EXPERIMENTS # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # #

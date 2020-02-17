from __future__ import print_function

import numpy as np

import math

from scipy.misc import logsumexp


import torch
import torch.utils.data
import torch.nn as nn
from torch.nn import Linear
from torch.autograd import Variable

from utils.distributions import log_Bernoulli, log_Normal_diag, log_Normal_standard, log_Logistic_256
from utils.visualization import plot_histogram
from utils.nn import he_init, GatedDense, NonLinear

from models.Model import Model
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

#=======================================================================================================================
class VAE(Model):
    def __init__(self, args):
        super(VAE, self).__init__(args)

        # encoder: q(z | x)
        self.encoder_layers = nn.Sequential(
            GatedDense(np.prod(self.args.input_size), self.args.hidden_size),
            GatedDense(self.args.hidden_size, self.args.hidden_size))

        self.encoder_mean = Linear(self.args.hidden_size, self.args.z1_size)
        self.encoder_log_var = NonLinear(self.args.hidden_size, self.args.z1_size, activation=nn.Hardtanh(min_val=-6.,max_val=2.))

        # decoder: p(x | z)
        self.decoder_layers = nn.Sequential(
            GatedDense(self.args.z1_size, self.args.hidden_size),
            GatedDense(self.args.hidden_size, self.args.hidden_size))


        if self.args.input_type == 'binary':
            self.decoder_mean = NonLinear(self.args.hidden_size, np.prod(self.args.input_size), activation=nn.Sigmoid())
        elif self.args.input_type == 'gray' or self.args.input_type == 'continuous':
            self.decoder_mean = NonLinear(self.args.hidden_size, np.prod(self.args.input_size), activation=nn.Sigmoid())
            self.decoder_logvar = NonLinear(self.args.hidden_size, np.prod(self.args.input_size), activation=nn.Hardtanh(min_val=-4.5,max_val=0))


        ## initialize: mu, sigma parameters for the MoG prior
        if self.args.prior == 'MoG':

            if self.args.fixed_var:
                self.prior_means = Variable(torch.rand([self.args.disc_size, self.args.z1_size]), requires_grad = True)  # K x D
                self.prior_vars = Variable(torch.ones([self.args.disc_size, self.args.z1_size]), requires_grad = False)

            else:
                self.prior_means = Variable(torch.rand(self.args.disc_size, self.args.z1_size), requires_grad = True)  # K x D
                self.prior_vars = Variable(torch.rand(self.args.disc_size, self.args.z1_size), requires_grad = True)   # K x D

            if self.args.cuda:
                self.prior_means = self.prior_means.cuda()
                self.prior_vars = self.prior_vars.cuda()


        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                he_init(m)

    # AUXILIARY METHODS
    def calculate_loss(self, x, beta=1.,temp=1., average=False):
        '''
        :param x: input image(s)
        :param beta: a hyperparam for warmup
        :param average: whether to average loss or not
        :return: value of a loss function
        '''
        # pass through VAE
        x_mean, x_logvar, z_q, z_q_mean, z_q_logvar = self.forward(x)

        # RE
        if self.args.input_type == 'binary':
            RE = log_Bernoulli(x, x_mean, dim=1)
        elif self.args.input_type == 'gray' or self.args.input_type == 'continuous':
            RE = -log_Logistic_256(x, x_mean, x_logvar, dim=1)
        else:
            raise Exception('Wrong input type!')

        # KL
        if self.args.prior == 'standard':
            log_p_z = self.log_p_z(z_q)
            log_q_z = log_Normal_diag(z_q, z_q_mean, z_q_logvar, dim=1)
            KL = -(log_p_z - log_q_z)

        elif self.args.prior == 'MoG' :
            log_MoG_prior = 0
            for i in range(self.args.disc_size):
                samples = torch.zeros (1, self.args.disc_size)
                if self.args.cuda:
                    samples = torch.cuda.FloatTensor (1, self.args.disc_size).fill_ (0)

                samples[np.arange (1), i] = 1.

                z_sample_rand_discr = Variable (samples)

                if self.args.cuda:
                    z_sample_rand_discr = z_sample_rand_discr.cuda ()

                gen_mean = z_sample_rand_discr.mm (self.prior_means)
                gen_logvar = z_sample_rand_discr.mm (self.prior_vars)
                log_MoG_prior += log_Normal_diag(z_q, gen_mean, gen_logvar, dim=1)

            log_q_z_cont = log_Normal_diag(z_q, z_q_mean, z_q_logvar, dim=1)

            KL = -( 1/self.args.disc_size * log_MoG_prior - log_q_z_cont)

        else:
            raise Exception('Wrong name of prior!')


        loss = - RE + beta * KL

        if average:
            loss = torch.mean(loss)
            RE = torch.mean(RE)
            KL = torch.mean(KL)

        KL_cont = 0.0
        KL_discr = 0.0

        return loss, RE, KL, KL_cont, KL_discr


    def calculate_lower_bound(self, X_full, MB=100):
        # CALCULATE LOWER BOUND:
        lower_bound = 0.
        RE_all = 0.
        KL_all = 0.

        I = int(math.ceil(X_full.size(0) / MB))

        for i in range(I):
            x = X_full[i * MB: (i + 1) * MB].view(-1, np.prod(self.args.input_size))

            loss, RE, KL, _, _ = self.calculate_loss(x,average=True)

            RE_all += RE.cpu().item()
            KL_all += KL.cpu().item()
            lower_bound += loss.cpu().item()

        lower_bound /= I

        return lower_bound

    # ADDITIONAL METHODS
    def generate_x(self, N=25):
        z_sample_rand = Variable( torch.FloatTensor(N, self.args.z1_size).normal_() )
        if self.args.cuda:
            z_sample_rand = z_sample_rand.cuda()

        samples_rand, _ = self.decoder(z_sample_rand)

        return samples_rand

    def generate_x_MoG(self):
        samples = torch.zeros(1, self.args.disc_size)
        if self.args.cuda:
            samples = torch.cuda.FloatTensor(1, self.args.disc_size).fill_(0)

        samples[np.arange(1), np.random.randint(0, self.args.disc_size, 1)] = 1.

        z_sample_rand_discr = Variable(samples)

        if self.args.cuda:
            z_sample_rand_discr = z_sample_rand_discr.cuda()

        gen_mean = z_sample_rand_discr.mm(self.prior_means)
        gen_logvar = z_sample_rand_discr.mm(self.prior_vars)
        gen_var = (torch.exp(gen_logvar))

        z_sample_rand_cont2 = []
        for i in range(self.args.z1_size):
            z_sample_rand_cont1 = Variable (torch.FloatTensor (1, 1).normal_ (gen_mean[0,i].item(), gen_var[0,i].item()))
            z_sample_rand_cont2.append(z_sample_rand_cont1) # -> list
        z_sample_rand_cont = torch.cat(z_sample_rand_cont2, dim=1) #  list -> tensor
        if self.args.cuda:
            z_sample_rand_cont = z_sample_rand_cont.cuda()


        samples_rand_mean, samples_rand_var = self.decoder(z_sample_rand_cont)

        return samples_rand_mean, samples_rand_var, self.prior_means, self.prior_vars

    def generate_specific_category_MoG(self, category):
        samples = torch.zeros(1, self.args.disc_size)
        if self.args.cuda:
            samples = torch.cuda.FloatTensor(1, self.args.disc_size).fill_(0)

        samples[np.arange(1), category] = 1.

        z_sample_rand_discr =  Variable(samples)

        if self.args.cuda:
            z_sample_rand_discr = z_sample_rand_discr.cuda()

        gen_mean = z_sample_rand_discr.mm(self.prior_means)
        gen_logvar = z_sample_rand_discr.mm(self.prior_vars)
        gen_var = (torch.exp(gen_logvar))


        z_sample_rand_cont2 = []
        for i in range(self.args.z1_size):
            z_sample_rand_cont1 = Variable (torch.FloatTensor (1, 1).normal_ (gen_mean[0,i].item(), gen_var[0,i].item()))
            z_sample_rand_cont2.append(z_sample_rand_cont1) # -> list
        z_sample_rand_cont = torch.cat(z_sample_rand_cont2, dim=1) #  list -> tensor
        if self.args.cuda:
            z_sample_rand_cont = z_sample_rand_cont.cuda()

        samples_rand, _ = self.decoder(z_sample_rand_cont)

        return samples_rand, _


    def reconstruct_x(self, x):
        x_mean, _, _, _, _ = self.forward(x)
        return x_mean

    # THE MODEL: VARIATIONAL POSTERIOR
    def encoder(self, x):
        x = self.encoder_layers(x)

        z_q_mean = self.encoder_mean(x)
        z_q_logvar = self.encoder_log_var(x)
        return z_q_mean, z_q_logvar

    # THE MODEL: GENERATIVE DISTRIBUTION
    def decoder(self, z):
        z = self.decoder_layers(z)

        x_mean = self.decoder_mean(z)
        if self.args.input_type == 'binary':
            x_logvar = 0.
        else:
            x_mean = torch.clamp(x_mean, min=0.+1./512., max=1.-1./512.)
            x_logvar = self.decoder_logvar(z)
        return x_mean, x_logvar

    # the prior
    def log_p_z(self, z):
        if self.args.prior == 'standard':
            log_prior = log_Normal_standard(z, dim=1)

        else:
            raise Exception('Wrong name of the prior!')

        return log_prior

    # THE MODEL: FORWARD PASS
    def forward(self, x):
        # z ~ q(z | x)
        z_q_mean, z_q_logvar = self.encoder(x)
        z_q = self.reparameterize(z_q_mean, z_q_logvar)

        # x_mean = p(x|z)
        x_mean, x_logvar = self.decoder(z_q)

        return x_mean, x_logvar, z_q, z_q_mean, z_q_logvar

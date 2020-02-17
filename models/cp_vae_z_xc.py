from __future__ import print_function

import numpy as np

import math

from scipy.misc import logsumexp


import torch
import torch.utils.data
import torch.nn as nn
from torch.nn import Linear
from torch.autograd import Variable
from torch.nn import functional as F
import torch.distributions as dis


from utils.distributions import log_Bernoulli, log_Normal_diag, log_Normal_standard, log_Logistic_256
from utils.visualization import plot_histogram
from utils.nn import he_init, GatedDense, NonLinear, CReLU

from models.Model import Model
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

#=======================================================================================================================
class VAE(Model):
    def __init__(self, args):
        super(VAE, self).__init__(args)

        self.latent_size = self.args.z1_size + self.args.disc_size

        # encoder: q(z, c | x)
        self.encoder_layers = nn.Sequential(
            GatedDense(np.prod(self.args.input_size), self.args.hidden_size),
            GatedDense(self.args.hidden_size, self.args.hidden_size)
        )
        
        self.encoder_mean = Linear(self.args.hidden_size + self.args.disc_size, self.args.z1_size )
        self.encoder_log_var = NonLinear(self.args.hidden_size + self.args.disc_size, self.args.z1_size, activation=nn.Hardtanh(min_val=-6.,max_val=2.))
        # self.encoder_discr = Linear(self.args.hidden_size, self.args.disc_size)
        self.encoder_discr = NonLinear(self.args.hidden_size, self.args.disc_size, activation = nn.ELU())

        # decoder: p(x | z, c)
        self.decoder_layers = nn.Sequential(
            GatedDense(self.latent_size, self.args.hidden_size),
            GatedDense(self.args.hidden_size, self.args.hidden_size)
        )



        if self.args.input_type == 'binary':
            self.decoder_mean = NonLinear(self.args.hidden_size, np.prod(self.args.input_size), activation=nn.Sigmoid())
        elif self.args.input_type == 'gray' or self.args.input_type == 'continuous':
            self.decoder_mean = NonLinear(self.args.hidden_size, np.prod(self.args.input_size), activation=nn.Sigmoid())
            self.decoder_logvar = NonLinear(self.args.hidden_size, np.prod(self.args.input_size), activation=nn.Hardtanh(min_val=-4.5,max_val=0))


        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus(beta=0.5)

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                he_init(m)

        ## initialize: mu, sigma parameters
        if self.args.prior == 'conditional':
            self.prior_means = Variable (torch.rand([self.args.disc_size, self.args.z1_size]), requires_grad=True)  # K x D

            if self.args.fixed_var:
                self.prior_vars = Variable(torch.ones([self.args.disc_size, self.args.z1_size]), requires_grad = False)  # K x D
            else:
                self.prior_vars = Variable(torch.rand(self.args.disc_size, self.args.z1_size), requires_grad = True)   # K x D
                # self.prior_vars = torch.ones(self.args.disc_size, self.args.z1_size)  # K x D

            if self.args.cuda:
                self.prior_means = self.prior_means.cuda()
                self.prior_vars = self.prior_vars.cuda()

         #   self.prior_vars = self.softplus(self.prior_vars)

    # THE MODEL: VARIATIONAL POSTERIOR ENCODER
    def encoder(self, x):

        x = self.encoder_layers(x)

        z_q_discr = self.encoder_discr(x)
        z_q_discr_r = self.reparameterize_discrete (z_q_discr)

        x_cont = torch.cat([x, z_q_discr_r], 1)
        z_q_mean = self.encoder_mean(x_cont)
        z_q_logvar = self.encoder_log_var(x_cont)


        return z_q_mean, z_q_logvar, z_q_discr

    # THE MODEL: GENERATIVE DISTRIBUTION DECODER
    def decoder(self, z):
        z = self.decoder_layers(z)

        x_mean = self.decoder_mean(z)
        if self.args.input_type == 'binary':
            x_logvar = 0.
        else:
            x_mean = torch.clamp(x_mean, min=0.+1./512., max=1.-1./512.)
            x_logvar = self.decoder_logvar(z)
        return x_mean, x_logvar


    # THE MODEL: FORWARD PASS
    def forward(self, x):
        '''
        :param x: input x
        :return: x_mean, x_logvar, z_q, z_q_cont_r, z_q_discr_r, z_q_mean, z_q_logvar, z_q_discr

        z ~ q(z | x)
        z_q_mean -> linear, z_q_logvar-> NonLinear, z_q_discr-> linear
        reparmeterize_discrete: z_q_discr -> F.softmax and then sample_gumbel_softmax (see Model -> reparmeterize_discrete)
        z_q_mean: (batch_size, z1_size)
        z_q_logvar: (batch_size, z1_size)
        z_q_discr: (batch_size, disc_size)
        z_q_cont_r: (batch_size, z1_size)
        z_q_discr_r: (batch_size, disc_size)
        z_q : (batch_size, z1_size + disc_size)

        '''

        z_q_mean, z_q_logvar, z_q_discr = self.encoder(x)
        z_q_cont_r = self.reparameterize (z_q_mean, z_q_logvar)
        z_q_discr_r = self.reparameterize_discrete (z_q_discr)
        z_q = torch.cat([z_q_cont_r, z_q_discr_r], 1)

        x_mean, x_logvar = self.decoder(z_q)

        return x_mean, x_logvar, z_q, z_q_cont_r, z_q_discr_r, z_q_mean, z_q_logvar, z_q_discr


    def calculate_loss(self, x, beta=1.,  average=False):
        '''
        :param x: input image(s)
        :param beta: a hyperparam for warmup
        :param average: whether to average loss or not
        :return: value of a loss function, RE loss (A term), total regularizer (B+ C), B term, C term

        L =   E_(q(z,c | x)) [log p(x | z, c)]          ->  A term = RE
            - E_(q(c | x)) [ KL(q(z | x, c) || p(z | c))   ->  B term Continuous latent variables
            - KL(q(c | x) || p(c))                      ->  C term Categorical latent variables
        '''

        # pass through VAE
        x_mean, x_logvar, z_q, z_q_cont_r, z_q_discr_r, z_q_mean, z_q_logvar, z_q_discr = self.forward(x)

        # RE / A term
        if self.args.input_type == 'binary':
            RE = log_Bernoulli(x, x_mean, dim=1)

        elif self.args.input_type == 'gray' or self.args.input_type == 'continuous':
            RE = -log_Logistic_256(x, x_mean, x_logvar, dim=1)
        else:
            raise Exception('Wrong input type!')


        #  B term
        #  KL for continuous latent variables

        if self.args.prior == 'conditional':
            KL_cont = 0.0
            disc_size = self.args.disc_size


            ''' 
                compute KL_cont using the def KL_continuous function in Models.py (self.KL_continuous)
                1. initialize mu, sigma gaussian parameters -> self.prior_means, self.prior_vars <- using def initialize_Gaussian_params 
                2. For each categorical latent variable compute KL_cont = z_q_discr_r[:,i] * self.KL_continuous (z_q_mean, z_q_logvar, self.prior_means[i,:], self.prior_vars[i,:], dim=1)
                3. sum them to get the final KL_cont
            '''

            q_c_x = F.softmax(z_q_discr, dim=-1)

            for i in range(disc_size):
                # KL_cont += z_q_discr_r[:,i] * self.KL_continuous (z_q_mean, z_q_logvar, self.prior_means[i,:], self.prior_vars[i,:], dim=-1)
                KL_cont += q_c_x[:, i] * self.KL_continuous(z_q_mean, z_q_logvar, self.prior_means[i, :],
                                                           self.softplus(self.prior_vars[i, :]), dim=-1)


        else:
            raise ValueError('Wrong Prior Type')


        # C term
        #  KL for discrete latent variables

        KL_discr = self.KL_discrete(z_q_discr,  dim=1)

        # total loss
        KL = KL_cont + KL_discr

        loss = - RE + beta * KL
        #
        if average:
            loss = torch.mean(loss)
            RE = torch.mean(RE)
            KL = torch.mean(KL)
            KL_cont = torch.mean(KL_cont)
            KL_discr = torch.mean(KL_discr)

        return loss, RE, KL, KL_cont, KL_discr



    def calculate_likelihood(self, X, dir, mode='test', S=5000, MB=100):
        '''
         X: test or train dataset
         S: number of samples used for approximating log-likelihood
         MB  : mini batch size
         return : LogL

         R: number of MB
         (need corrections)
        '''

        # set auxiliary variables for number of training and test sets
        N_test = X.size(0)

        # init list
        likelihood_test = []

        if S <= MB:
            R = 1
        else:
            R = S / MB
            S = MB

        for j in range(N_test):
            if j % 100 == 0:
                print('{:.2f}%'.format(j / (1. * N_test) * 100))
            # Take x*
            x_single = X[j].unsqueeze(0)

            a = []
            for r in range (0, int (R)):
                # Repeat it for all training points
                x = x_single.expand (S, x_single.size (1))


                for c in range(0, self.args.disc_size):

                    x_mean, x_logvar, z_q, z_q_cont_r, z_q_discr_r, z_q_mean, z_q_logvar, z_q_discr = self.forward (x)

                    samples = torch.zeros (1, self.args.disc_size)
                    if self.args.cuda:
                        samples = torch.cuda.FloatTensor (1, self.args.disc_size).fill_ (0)

                    samples[np.arange (1), c] = 1.

                    z_sample_rand_discr = Variable (samples)
                    if self.args.cuda:
                        z_sample_rand_discr = z_sample_rand_discr.cuda ()
                    #
                    gen_mean = z_sample_rand_discr.mm (self.prior_means)
                    gen_logvar = z_sample_rand_discr.mm (self.prior_vars)
                    log_p_z_c = log_Normal_diag (z_q_cont_r, gen_mean, gen_logvar, dim=1)

                    log_q_z_cont = log_Normal_diag (z_q_cont_r, z_q_mean, z_q_logvar, dim=1)

                    KL = -(log_p_z_c - log_q_z_cont)

                    loos, RE, _, _, _ = self.calculate_loss(x)
                    total_KL = KL + torch.log(z_q_discr).view(100) - log_q_c_x
                    log_q_c_x = torch.log(torch.Tensor(1).fill_(self.args.disc_size))
                    if self.args.cuda:
                        log_q_c_x = torch.log (torch.cuda.FloatTensor (1).fill_ (self.args.disc_size))
                    a_tmp = -RE + beta*total_KL

                    a.append( -a_tmp.cpu().data.numpy() )

            # calculate max
            a = np.asarray(a)
            # print('a.shape')
            #
            # print(a.shape)
            a = np.reshape(a, (a.shape[0] * a.shape[1], 1))
            # logsumexp(): returns the log of the sum of exponentials of input elements.
            likelihood_x = logsumexp( a )
            likelihood_test.append(likelihood_x - np.log(len(a)))

        likelihood_test = np.array(likelihood_test)

        plot_histogram(-likelihood_test, dir, mode)

        return -np.mean(likelihood_test)

    def calculate_lower_bound(self, X_full, MB=100):

        '''
        X_full: the whole xtest or train dataset
        MB: size of MB
        return: ELBO

        I: nuber of MBs
        '''

        # CALCULATE LOWER BOUND:
        lower_bound = 0.
        RE_all = 0.
        KL_all = 0.
        KL_cont_all = 0.
        KL_discr_all = 0.

        # ceil(): returns ceiling value of x - the smallest integer not less than x
        I = int(math.ceil(X_full.size(0) / MB))

        for i in range(I):
            x = X_full[i * MB: (i + 1) * MB].view(-1, np.prod(self.args.input_size))

            loss, RE, KL, KL_cont, KL_discr = self.calculate_loss(x, average=True)

            RE_all += RE.cpu().item()
            KL_all += KL.cpu().item()
            KL_cont_all += KL_cont.cpu().item()
            KL_discr_all += KL_discr.cpu ().item()
            lower_bound += loss.cpu().item()

        lower_bound /= I

        return lower_bound



    def reconstruct_x(self, x):
        z_q_mean, z_q_logvar, z_q_discr = self.encoder(x)
        z_q_cont_r = self.reparameterize (z_q_mean, z_q_logvar)

        z_q_discr_r = self.reparameterize_discrete_reconstracrion (z_q_discr)
        if self.args.no_recon_oneHot:
            z_q_discr_r = self.reparameterize_discrete (z_q_discr)

        z_q = torch.cat([z_q_cont_r, z_q_discr_r], 1)

        x_mean, _ = self.decoder(z_q)

        return x_mean

    # ADDITIONAL METHODS

    def generate_cont_x(self, average_q_c_x, marginal = False):
        '''
        :argument: marginal = False sample from the uniform categorical prior, otherwise from the marginal categorical posterior
        return: generations  (only 1 sample)

        Specify number of samples that you want to generate (N=25) in evaluation.py
        Sample uniformly:
        1. sample a random category between (1, disc_size): one hot vector (ex. [0, 1, 0, 0]) -> z_sample_rand_discr
        2. multiply it with the learned means of the prior (prior_means: size (disc_size, z1_size ) ) such that to 'isolate'
            the corresponding mean to the sampled category -> gen_mean : size (1, z1_size)
        2. same as 2. for var
        3. for each pair (gen_mean, gen_logvar ) sample z from the corresponding Normal distribution -> z_sample_rand_cont
        4. concatenate continuous and categorical samples (z_sample_rand_cont, z_sample_rand_discr) -> z_sample_rand and
            pass it to the decoder

        '''
        samples = torch.zeros(1, self.args.disc_size)
        if self.args.cuda:
            samples = torch.cuda.FloatTensor(1, self.args.disc_size).fill_(0)

        samples[np.arange(1), np.random.randint(0, self.args.disc_size, 1)] = 1.

        if marginal == True:
            loc = dis.Categorical (average_q_c_x).sample ()
            samples[np.arange(1), loc] = 1.

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

        z_sample_rand = torch.cat([z_sample_rand_cont, z_sample_rand_discr], 1)
        samples_rand_mean, samples_rand_var = self.decoder(z_sample_rand)

        return samples_rand_mean, samples_rand_var, self.prior_means, self.prior_vars


    def generate_specific_x_cat(self, category):
        '''
        :argument the category from which we want to sample
        :return generation from a specific category (only 1 sample)

        Specify the number of samples that you want to generate (N=25) in evaluation.py

        1. sample a random category between (1, disc_size): one hot vector (ex. [0, 1, 0, 0]) -> z_sample_rand_discr
        2. multiply it with the learned means of the prior (prior_means: size (disc_size, z1_size ) ) such that to 'isolate'
            the corresponding mean to the sampled category -> gen_mean : size (1, z1_size)
        2. same as 2. for var
        3. for each pair (gen_mean, gen_logvar ) sample z from the corresponding Normal distribution -> z_sample_rand_cont
            *******think if you sample from N(gen_mean, gen_logvar ) or N(gen_mean, exp( gen_logvar )) *******
            ******* Ok if I sample from .normal_ (gen_mean[0,i].item(), np.exp(gen_var[0,i]) I have much worse generations *******
            ******* check for use logvar - exp (logvar) everywhere  *******
        4. concatenate continuous and categorical samples (z_sample_rand_cont, z_sample_rand_discr) -> z_sample_rand and
            it as input to the decoder

        '''
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

        z_sample_rand = torch.cat([z_sample_rand_cont, z_sample_rand_discr], 1)

        samples_rand, _ = self.decoder(z_sample_rand)

        return samples_rand



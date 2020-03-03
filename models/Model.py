from __future__ import print_function

import numpy as np

import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from utils.nn import normal_init, NonLinear
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

#=======================================================================================================================
class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()

        self.args = args

    # AUXILIARY METHODS


    def reparameterize(self, mu, logvar, order, model):
        output = dict()
        for klass, values in mu.items():
            if model.curvature[klass] == 'euclidean':
                z_var = logvar[klass].mul(0.5).exp_()
                q_z = torch.distributions.normal.Normal(values, z_var).rsample()
                for klass_idx, org_idx in enumerate(order[klass]):
                    output[org_idx] = q_z[klass_idx, :]
        return torch.stack([output[i] for i in range(len(output))])

    def reparameterize_discrete(self, logits, hard=False, *args, **kwargs):
        """
        Samples from a gumbel-softmax distribution using the reparameterization
        trick.

        The sample_gumbel_softmax() argument should be unormalized log-probs
        -> apply softmax at the output of the encoder to make it
           prob and after take the log (or equivalently log_softmax)
        """

        return F.gumbel_softmax(logits, hard=hard, *args, **kwargs)

    def reparameterize_discrete_reconstracrion(self, logits ):
        """

        In validation / test pick the most likely sample to reconstruct the input
        ----------
        one_hot_vector : MB x disc_size

        loc : indicates the location of the highest prob
        scatter_(dim, index, src) → Tensor
        On dim (1) scatter the value src (1) at the indices with the highest prob (loc)
        scatter_  accepts only 2D tensors => use view
        """

        _, loc = torch.max(logits, dim=1)

        one_hot_vector = torch.zeros(logits.size())    # MB x disc_size

        # scatter_(dim, index, src) → Tensor
        # On dim (1) scatter the value src (1) at the indices with the highest prob (loc)
        # scatter_  accepts only 2D tensors => use viewß

        one_hot_vector.scatter_(1, loc.view(-1, 1).data.cpu(), 1)
        if self.args.cuda:
            one_hot_vector = one_hot_vector.cuda()
        # print('------q_z_discr----------')
        # print(F.log_softmax(logits, dim=-1))
        # print('------GUMBLE_q_z_discr----------')
        # print(self.sample_gumbel_softmax (F.log_softmax(logits, dim=-1)))
        #
        # print('------loc----------')
        # print(loc)
        # print('------one_hot_vector----------')
        # print(one_hot_vector)
        return one_hot_vector


    def sample_gumbel_softmax(self , logits, EPS = 1e-10,  hard=False ):
        """
        Samples from a gumbel-softmax distribution using the reparameterization
        trick.
        https://blog.evjang.com/2016/11/tutorial-categorical-variational.html
        1.  Sample from Gumbel(0, 1)
        2. Draw a sample from the Gumbel-Softmax distribution

        Args
        ----------
        logits : torch.Tensor
           logits: [MB, disc_size] unnormalized log-probs -> apply softmax at the output of the encoder to make it
           prob an log (or equivalently log_softmax): def reparameterize_discrete
        hard: if True, take argmax, but differentiate w.r.t. soft sample y

        Returns:
        ----------
        [MB, disc_size] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
        """

        # Sample from Gumbel(0, 1)
        # torch.rand returns a tensor filled with rn from a uniform distr. on the interval [0, 1)
        unif = torch.rand (logits.size ())
        if self.args.cuda:
            unif = unif.cuda ()
        gumbel = -torch.log (-torch.log (unif + EPS) + EPS)
        gumbel = Variable(gumbel)

        # Draw a sample from the Gumbel-Softmax distribution
        y = logits + gumbel
        ttt = self.args.temp

        # log_logits = torch.log(logits + EPS)
        # y = log_logits + gumbel
        gumbel_softmax_samples = F.softmax ( y / ttt, dim=-1)


        if hard:
            gumbel_softmax_samples_max , _ = torch.max(gumbel_softmax_samples, dim=gumbel_softmax_samples.dim() - 1,
                                 keepdim=True)

            gumbel_softmax_samples_hard = Variable(
                torch.eq(gumbel_softmax_samples_max.data, gumbel_softmax_samples.data).type(float_type(use_cuda))
            )
            gumbel_softmax_samples_hard_diff = gumbel_softmax_samples_hard - gumbel_softmax_samples
            gumbel_softmax_samples = gumbel_softmax_samples_hard_diff.detach() + gumbel_softmax_samples

        return gumbel_softmax_samples




    def KL_discrete(self, logit, average=False, dim=None, EPS = 1e-12 ):
        '''Give as imput the Gumble-Softmax reparametrized output of the encoder
        (output of the encoder is not activated)
        KL(q(c | x) || p(c)) = sum_k q(c | x) (log q(c | x) - log p(c))      -> A
                             = sum_k q(c | x) log q(c | x) -  log p(c) sum_k q(c | x)
                             = sum_k q(c | x) log q(c | x) - log (1 / K) * 1
                             =  sum_k q(c | x) log q(c | x) + log (K)       -> B
                             (tested, A, B same result)
        '''

        ####    B   ####
        disc_dim = logit.size (-1)
        log_K = torch.Tensor([np.log(disc_dim)])   # log p_c is normally 1 / discr_dim

        if self.args.cuda:
            log_K = log_K.cuda()

        # apply softmax at the output of the encoder to make it
        # prob and after take the log (or equivalently log_softmax)

        q_c_x = F.softmax(logit, dim=-1)
	    # log_q_y = torch.Tensor([np.log(q_c_x + EPS)])
        # log_q_c_x = F.log_softmax(logits, dim=-1)

        # Calculate negative entropy of each row: sum_discr_dim
        neg_entropy = torch.sum(q_c_x * torch.log(q_c_x + EPS), dim=-1)

        # KL loss with uniform categorical variable
        KL_discr = log_K + neg_entropy


        return KL_discr






    def KL_continuous(self, mean_q, log_var_q, mean_p, log_var_p, average=False, dim=None):
        #
        '''

        return:
        KL_cont =  (log_var_p / log_var_q + torch.exp(log_var_q) / torch.exp(log_var_p) + torch.pow(mean_p - mean_q, 2) / torch.exp(log_var_p))

        '''
        # # Matrix calculations
        # # Determinants of diagonal covariances pv, qv
        # dlog_var_p = log_var_q.prod()
        # dlog_var_q = log_var_q.prod(dim)
        # # Inverse of diagonal covariance var_q
        # inv_var_p = 1. / np.exp(log_var_p)
        # # Difference between means pm, qm
        # diff = mean_q - mean_p
        # KL_cont = (0.5 *
        #         ((dlog_var_p / dlog_var_q)  # log |\Sigma_p| / |\Sigma_q|
        #          + (inv_var_p * log_var_q).sum(dim)  # + tr(\Sigma_p^{-1} * \Sigma_q)
        #          + (diff * inv_var_p * diff).sum(dim)  # + (\mu_q-\mu_p)^T\Sigma_p^{-1}(\mu_q-\mu_p)
        #          - len(mean_q)))  # - D : size_z

        KL_cont = 0.5 * ( log_var_p -  log_var_q    #log s_p,i^2 / s_q_i^2
                         + torch.exp( log_var_q )/ torch.exp( log_var_p )   #s_q_i^2 / s_p_i^2
                         + torch.pow( mean_q - mean_p, 2 )/ torch.exp( log_var_p )      #(m_p_i - m_q_i)^2 / s_p_i^2
                         - 1)       # dim_z -> after sum D

        if average:
            return torch.mean(KL_cont, dim)
        else:
            return torch.sum(KL_cont, dim)


        return KL_cont

    # def MI_discrete(self, logit, average=False, dim=None, EPS = 1e-12 ):



    def calculate_loss(self):
        return 0.

    def calculate_likelihood(self):
        return 0.

    def calculate_lower_bound(self):
        return 0.

    # THE MODEL: FORWARD PASS
    def forward(self, x):
        return 0.

#=======================================================================================================================
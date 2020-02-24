from __future__ import print_function

import torch
from torch.autograd import Variable

from utils.visualization import plot_images, visualize_latent, plot_manifold, plot_scatter, visualize_latent_no_target
import numpy as np

import time

import os
from torch.nn import functional as F
from sklearn.manifold import TSNE

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# ======================================================================================================================
def evaluate_vae(args, model, train_loader, data_loader, epoch, dir, mode):
    # set loss to 0
    evaluate_loss = 0
    evaluate_re = 0
    evaluate_kl = 0
    evaluate_kl_cont = 0
    evaluate_kl_discr = 0
    # set model to evaluation mode
    model.eval()


    # evaluate
    for batch_idx, (data, target) in enumerate(data_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)

        x = data

        # calculate loss function
        loss, RE, KL, KL_cont, KL_discr = model.calculate_loss(x, average=True)

        evaluate_loss += loss.item()
        evaluate_re += -RE.item()
        evaluate_kl += KL.item()
        if args.prior == 'MoG' or args.prior == 'standard':
            evaluate_kl_discr = 0.0
            evaluate_kl_cont = 0.0
        else:
            evaluate_kl_cont += KL_cont.item()
            evaluate_kl_discr += KL_discr.item()


        # print N digits
        if batch_idx == 1 and mode == 'validation':
            if epoch == 1:
                pass
                if not os.path.exists(dir + 'figures/'):
                    os.makedirs(dir + 'figures/')
                # VISUALIZATION: plot real images
                plot_images(args, data.data.cpu().numpy()[0:9], dir + 'figures/', 'real', size_x=3, size_y=3)
            x_mean = model.reconstruct_x(x)
            plot_images(args, x_mean.data.cpu().numpy()[0:9], dir + 'reconstruction_val_', str(epoch), size_x=3, size_y=3)

            if epoch%100 == 0 :
                x_mean = model.reconstruct_x(x)
                plot_images(args, x_mean.data.cpu().numpy()[0:9], dir + 'figures/', 'reconstruction_val_' + str(epoch), size_x=3, size_y=3)

    if mode == 'test':
        # load all data
        # grab the test data by iterating over the loader
        # there is no standardized tensor_dataset member across pytorch datasets
        test_data, test_target = [], []
        for data, lbls in data_loader:
            test_data.append(data)
            test_target.append(lbls)

        test_data, test_target = [torch.cat(test_data, 0), torch.cat(test_target, 0).squeeze()]

        # grab the train data by iterating over the loader
        # there is no standardized tensor_dataset member across pytorch datasets
        full_data = []
        for data, _ in train_loader:
            full_data.append(data)

        full_data = torch.cat(full_data, 0)

        if args.cuda:
            test_data, test_target, full_data = test_data.cuda(), test_target.cuda(), full_data.cuda()

        if args.dynamic_binarization:
            full_data = torch.bernoulli(full_data)

        # VISUALIZATION: plot real test images (25 first)
        plot_images(args, test_data.data.cpu().numpy()[0:25], dir + 'figures/', 'real_test', size_x=5, size_y=5)

        # VISUALIZATION: plot reconstructions
        samples = model.reconstruct_x(test_data[0:25])

        plot_images(args, samples.data.cpu().numpy(), dir + 'figures/', 'reconstructions_test', size_x=5, size_y=5)

        # # VISUALIZATION: plot generations
        if args.prior == 'conditional' :

            # marginal categorical posterior
            z_q_mean, z_q_logvar, z_q_discr_r, z_q_discr = model.encoder(full_data)
            q_c_x = F.softmax (z_q_discr, dim=-1)
            average_q_c_x = torch.mean( q_c_x, dim=0)

            #############################################
            # Accuracy
            # values, indices = torch.max(q_c_x, 1)
            # num_correct = torch.sum(test_target == indices)
            #
            # accuracy = (float(num_correct) / len(test_target) )* 100

            #############################################


            samples_rand1_marginal =[]
            for i in range(25):
                sample_rand_marginal, _, _, _ = model.generate_cont_x (average_q_c_x, marginal = True)
                samples_rand1_marginal.append(sample_rand_marginal)
            samples_rand_marginal = torch.cat(samples_rand1_marginal)

            plot_images(args, samples_rand_marginal.data.cpu().numpy(), dir + 'figures/', 'generations_marginal', size_x=5, size_y=5) #size_x=5, size_y=5)


            samples_rand1 =[]
            for i in range(25):
                sample_rand, samples_rand_var, prior_means, prior_vars = model.generate_cont_x (average_q_c_x, marginal = False)
                samples_rand1.append(sample_rand)
            samples_rand = torch.cat(samples_rand1)

            plot_images(args, samples_rand.data.cpu().numpy(), dir + 'figures/', 'generations', size_x=5, size_y=5) #size_x=5, size_y=5)

            # # VISUALIZATION: conditional generations
            for j in range(args.disc_size):
                samples_rand_specific =[]
                for i in range(25):
                    sample_rand_specific = model.generate_specific_x_cat (j)
                    samples_rand_specific.append(sample_rand_specific)
                sample_rand_specific = torch.cat(samples_rand_specific)

                plot_images(args, sample_rand_specific.data.cpu().numpy(), dir + 'figures/' , 'generations_specific_cat' + str(j), size_x=5, size_y=5)

        # # # VISUALIZATION: plot generations

        if args.prior == 'standard':
            samples_rand = model.generate_x(25)
            plot_images(args, samples_rand.data.cpu().numpy(), dir + 'figures/' , 'generations', size_x=5, size_y=5)

        if args.prior == 'MoG':

            samples_rand_cz =[]

            for i in range(25):
                zc_sample_means, _, _, _ = model.generate_x_MoG ()
                samples_rand_cz.append(zc_sample_means)
            samples_means = torch.cat(samples_rand_cz)

            plot_images(args, samples_means.data.cpu().numpy(), dir + 'figures/', 'generations', size_x=5, size_y=5) #size_x=5, size_y=5)

            # # VISUALIZATION: plot generations from a specific category
            for j in range(args.disc_size):
                samples_rand_specific =[]
                for i in range(25):
                    sample_rand_specific = model.generate_specific_category_MoG (j)
                    samples_rand_specific.append(sample_rand_specific)
                sample_rand_specific = torch.cat(samples_rand_specific)

                plot_images(args, sample_rand_specific.data.cpu().numpy(), dir + 'figures/' , 'generations_specific_cat' + str(j), size_x=5, size_y=5)

            # VISUALIZATION: latent space
        if args.latent is True and args.dataset_name != 'celeba':
            if args.prior == 'standard' or args.prior == 'MoG':
                z_mean_recon, z_logvar_recon = model.encoder (test_data)
                vis_data = TSNE (n_components=2).fit_transform (z_mean_recon.data.cpu ().numpy ().astype ('float64'))
                vis_x = vis_data[:, 0]
                vis_y = vis_data[:, 1]

                print ("latent visualization without target")

                visualize_latent_no_target (vis_x, vis_y,
                                            dir + 'figures/' + '/only_latent' + args.model_name + '_' + 'c_' + str (
                                                args.disc_size))

                print ("latent visualization")
                visualize_latent (vis_x, vis_y, test_target,
                                  dir + 'figures/' + '/latent_' + args.model_name + '_' + 'c_' + str (args.disc_size))


            elif args.prior == 'conditional':
                z_mean_recon, z_logvar_recon, z_q_discr = model.encoder(test_data)
                vis_data = TSNE(n_components=2).fit_transform(z_mean_recon.data.cpu().numpy().astype('float64'))
                vis_x = vis_data[:, 0]
                vis_y = vis_data[:, 1]

                print("latent visualization without target")

                visualize_latent_no_target(vis_x,vis_y, dir + 'figures/' + '/only_latent'+ args.model_name + '_' + 'c_' + str(args.disc_size))

                print("latent visualization")
                visualize_latent(vis_x,vis_y,test_target, dir + 'figures/' + '/latent_'+ args.model_name + '_' + 'c_' + str(args.disc_size))



        # CALCULATE lower-bound
        t_ll_s = time.time()
        elbo_test = model.calculate_lower_bound(test_data, MB=args.MB)
        t_ll_e = time.time()
        print('Test lower-bound value {:.2f} in time: {:.2f}s'.format(elbo_test, t_ll_e - t_ll_s))

        # CALCULATE log-likelihood
        t_ll_s = time.time()
        elbo_train = model.calculate_lower_bound(full_data, MB=args.MB)
        t_ll_e = time.time()
        print('Train lower-bound value {:.2f} in time: {:.2f}s'.format(elbo_train, t_ll_e - t_ll_s))

        # CALCULATE log-likelihood
        t_ll_s = time.time()
        log_likelihood_test = 0. # model.calculate_likelihood(test_data, dir, mode='test', S=args.S, MB=args.MB)
        t_ll_e = time.time()
        print('Test log_likelihood value {:.2f} in time: {:.2f}s'.format(log_likelihood_test, t_ll_e - t_ll_s))

        # CALCULATE log-likelihood
        t_ll_s = time.time()
        log_likelihood_train = 0. #model.calculate_likelihood(full_data, dir, mode='train', S=args.S, MB=args.MB)) #commented because it takes too much time
        t_ll_e = time.time()
        print('Train log_likelihood value {:.2f} in time: {:.2f}s'.format(log_likelihood_train, t_ll_e - t_ll_s))

    # calculate final loss
    evaluate_loss /= len(data_loader)  # loss function already averages over batch size
    evaluate_re /= len(data_loader)  # re already averages over batch size
    evaluate_kl /= len(data_loader)  # kl already averages over batch size
    evaluate_kl_cont /= len(data_loader)  # kl of continuous latent already averages over batch size
    evaluate_kl_discr /= len(data_loader)  # kl of discrete latent already averages over batch size

    if mode == 'test':
        return evaluate_loss, evaluate_re, evaluate_kl, evaluate_kl_cont, evaluate_kl_discr, log_likelihood_test, log_likelihood_train, elbo_test, elbo_train
    else:
        return evaluate_loss, evaluate_re, evaluate_kl, evaluate_kl_cont, evaluate_kl_discr

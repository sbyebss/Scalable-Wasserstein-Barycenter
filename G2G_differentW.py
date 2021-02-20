from __future__ import print_function
import GPUtil
import optimal_transport_modules.log_utils as LLU
from optimal_transport_modules.icnn_modules import *
from optimal_transport_modules.generate_data import *
from optimal_transport_modules.record_mean_cov import *
import optimal_transport_modules
from optimal_transport_modules.cfg import CfgGMM as Cfg_class
from torchvision.utils import make_grid, save_image
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

args = Cfg_class()
gpus_choice = GPUtil.getFirstAvailable(
    order='random', maxLoad=0.5, maxMemory=0.5, attempts=5, interval=900, verbose=False)
PTU.set_gpu_mode(True, gpus_choice[0])

args.MEAN, args.COV = select_mean_and_cov(args.TRIAL)
args.INPUT_DIM = args.MEAN[0].shape[1]
args.NUM_DISTRIBUTION = len(args.MEAN)
for i in range(args.NUM_DISTRIBUTION):
    args.NUM_GMM_COMPONENT.append(args.MEAN[i].shape[0])
if args.INPUT_DIM <= 2:
    args.high_dim_flag = False
else:
    args.high_dim_flag = True
"""""""""""""""""""""""""""""""""""""""""""""""""""
                Storing path
"""""""""""""""""""""""""""""""""""""""""""""""""""
results_save_path, model_save_path, results, testresults = LLU.init_path(args)
kwargs = {'num_workers': 1, 'pin_memory': True}

##### For computing the constraint loss of negtive weights ######


def compute_constraint_loss(list_of_params, begin_idx, end_idx):
    loss_val = 0
    p_idx = 0
    for p in list_of_params:
        if p_idx >= begin_idx and p_idx < end_idx:
            loss_val += torch.relu(-p).pow(2).sum()
        elif p_idx >= end_idx:
            break
        p_idx += 1
    return loss_val


"""""""""""""""""""""""""""""""""""""""""""""""""""
                        Data
"""""""""""""""""""""""""""""""""""""""""""""""""""
total_data = marginal_data_gmm_3loop_picnn(args)
train_loader = torch.utils.data.DataLoader(
    total_data, batch_size=args.BATCH_SIZE, shuffle=False, **kwargs)

"""""""""""""""""""""""""""""""""""""""""""""""""""
                Plot Original Distribution
"""""""""""""""""""""""""""""""""""""""""""""""""""
# if args.high_dim_flag == False:
#     # if False:
#     original_dist_plot_path = []
#     for i in range(args.NUM_DISTRIBUTION):
#         original_dist_plot_path.append(results_save_path + '/distribution{0}_GMM{1}.png'.format(
#             i + 1, args.TRIAL))

#     right_place = 7
#     left_place = -7

#     for i in range(args.NUM_DISTRIBUTION):
#         fig = plt.figure()
#         JG = sns.jointplot(total_data[:, :, i].detach().numpy()[:, 0],
#                            total_data[:, :, i].detach().numpy()[:, 1], kind='kde', joint_kws={'shade_lowest': False})
#         JG.ax_joint.set_xlim(left_place, right_place)
#         JG.ax_joint.set_ylim(left_place, right_place)
#         plt.savefig(original_dist_plot_path[i])
#         plt.close()

"""""""""""""""""""""""""""""""""""""""""""""""""""
                 Neural Networks
"""""""""""""""""""""""""""""""""""""""""""""""""""
if args.expanded_PICNN:
    convex_f = nn.ModuleList(
        [PICNN_expanded(
            args.INPUT_DIM, args.NUM_DISTRIBUTION,
            args.NUM_NEURON_fg_weight,
            args.NUM_NEURON_fg_sample,
            args.fg_activation,
            args.NUM_LAYERS) for i in range(
            args.NUM_DISTRIBUTION)])
    convex_g = nn.ModuleList(
        [PICNN_expanded(
            args.INPUT_DIM, args.NUM_DISTRIBUTION,
            args.NUM_NEURON_fg_weight,
            args.NUM_NEURON_fg_sample,
            args.fg_activation,
            args.NUM_LAYERS) for i in range(
            args.NUM_DISTRIBUTION)])
else:
    convex_f = nn.ModuleList(
        [PICNN_LastInp_Quadratic(
            args.INPUT_DIM, args.NUM_DISTRIBUTION,
            args.NUM_NEURON_fg_weight,
            args.NUM_NEURON_fg_sample,
            args.fg_activation,
            args.NUM_LAYERS) for i in range(
            args.NUM_DISTRIBUTION)])
    convex_g = nn.ModuleList(
        [PICNN_LastInp_Quadratic(
            args.INPUT_DIM, args.NUM_DISTRIBUTION,
            args.NUM_NEURON_fg_weight,
            args.NUM_NEURON_fg_sample,
            args.fg_activation,
            args.NUM_LAYERS) for i in range(
            args.NUM_DISTRIBUTION)])

if args.h_PICNN_flag:
    generator_h = Different_Weights_PICNN(
        args.INPUT_DIM, args.NUM_DISTRIBUTION, args.NUM_NEURON_h_weight, args.NUM_NEURON_h_sample, args.h_activation, args.NUM_LAYERS_h, args.h_full_activation)
else:
    generator_h = Different_Weights_NormalNet(
        args.INPUT_DIM, args.OUTPUT_DIM, args.NUM_DISTRIBUTION, args.NUM_NEURON_h, args.h_activation, args.NUM_LAYERS_h, h_full_activation=args.h_full_activation)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        Initialization with some positive parameters
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# Form a list of positive weight parameters in f/g_positive_params and
# also initialize them with positive values
f_positive_params = []
g_positive_params = []

for i in range(args.NUM_DISTRIBUTION):
    for p in list(convex_f[i].parameters()):
        if hasattr(p, 'be_positive'):
            f_positive_params.append(p)

    for p in list(convex_g[i].parameters()):
        if hasattr(p, 'be_positive'):
            g_positive_params.append(p)

    convex_f[i].cuda(PTU.device)
    convex_g[i].cuda(PTU.device)

generator_h.cuda(PTU.device)
len_g_params = len(g_positive_params)
num_parameters = 0.0
for i in range(args.NUM_DISTRIBUTION):
    num_parameters_each_distribution = sum(
        [l.nelement() for l in convex_f[i].parameters()])
    num_parameters += num_parameters_each_distribution

optimizer_f = []
optimizer_g = []

for i in range(args.NUM_DISTRIBUTION):
    optimizer_f.append(optim.Adam(convex_f[i].parameters(), lr=args.LR_f))
    optimizer_g.append(
        optim.Adam(convex_g[i].parameters(), lr=args.LR_g))
optimizer_h = optim.Adam(
    generator_h.parameters(),
    lr=args.LR_h)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    Training function definition
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
############## For each function here, it's an epoch ##################


def train(epoch):
    convex_f.train()
    convex_g.train()
    generator_h.train()

    # These values are just for saving data
    w2_loss_value_epoch = 0
    g_OT_loss_value_epoch = [0] * args.NUM_DISTRIBUTION
    g_constraints_loss_value_epoch = 0
    remaining_f_loss_value_epoch = [0] * args.NUM_DISTRIBUTION
    mu_2moment_loss_value_epoch = 0
    miu_mean_value_epoch = 0
    miu_var_value_epoch = 0

    # Iterate over one train_loader, the batch_idx is determined by tot
    for batch_idx, real_data in enumerate(train_loader):
        real_data = real_data.cuda(PTU.device)
        miu_i = real_data[:, :, :args.NUM_DISTRIBUTION]
        epsilon = real_data[:, :, args.NUM_DISTRIBUTION]
        miu_i = Variable(miu_i, requires_grad=True)
        epsilon = Variable(epsilon)

        # containing four distribution
        g_OT_loss_value_batch = [0] * args.NUM_DISTRIBUTION
        g_constraints_loss_value_batch = 0  # containing four g networks
        remaining_f_loss_value_batch = [0] * args.NUM_DISTRIBUTION
        mu_2moment_loss_value_batch = 0
        miu_mean_value_batch = torch.zeros([args.INPUT_DIM])
        miu_var_value_batch = np.zeros(
            [args.INPUT_DIM, args.INPUT_DIM])

        ######################################################
        #                Medium Loop Begin                   #
        ######################################################
        ######### Here iterate over a given number: args.N_Fnet_ITERS=4 ##
        for medium_iter in range(1, args.N_Fnet_ITERS + 1):

            ######################################################
            #                Inner Loop Begin                   #
            ######################################################
            ######### Here iterate over a given number: args.N_Gnet_ITERS=16 ##
            for inner_iter in range(1, args.N_Gnet_ITERS + 1):

                loss_g = torch.ones(args.NUM_DISTRIBUTION)
                g_positive_constraints_loss = torch.zeros(
                    args.NUM_DISTRIBUTION)
                for i in range(args.NUM_DISTRIBUTION):
                    optimizer_g[i].zero_grad()

                    # Get the gradient of g(y):=g(miu_i_data)
                    tmp_miu_i = miu_i[:, :, i]
                    g_of_y = convex_g[i](tmp_miu_i).sum()
                    grad_g_of_y = torch.autograd.grad(
                        g_of_y, tmp_miu_i, create_graph=True)[0]

                    # For each distribution you need to calculate a f(gradient of y)
                    # it's the mean of the batch
                    # FIXME add wegihts
                    f_grad_g_y = torch.dot(convex_f[i](grad_g_of_y).reshape(-1),
                                           miu_i[:, -args.NUM_DISTRIBUTION + i, 0]) / args.BATCH_SIZE
                    # FIXME add wegihts
                    # ? The 1st loss part useful for f/g parameters
                    loss_g[i] = f_grad_g_y - torch.dot((grad_g_of_y[:, :args.INPUT_DIM] * miu_i[:, :args.INPUT_DIM, i]).sum(dim=1),
                                                       miu_i[:, -args.NUM_DISTRIBUTION + i, 0]) / args.BATCH_SIZE
                    g_OT_loss_value_batch[i] += loss_g[i].item()

                total_loss_g = loss_g.sum()
                total_loss_g.backward()
                # ? The 2nd loss part useful for g parameters:
                # FIXME add wegihts
                # if args.LAMBDA_CVX > 0:
                for i in range(args.NUM_DISTRIBUTION):
                    g_positive_constraints_loss[i] = miu_i[:, -args.NUM_DISTRIBUTION + i, 0].mean() * args.LAMBDA_CVX * compute_constraint_loss(
                        list_of_params=g_positive_params, begin_idx=int(len_g_params * i / args.NUM_DISTRIBUTION), end_idx=int(len_g_params * (i + 1) / args.NUM_DISTRIBUTION))

                total_g_positive_constraints_loss = g_positive_constraints_loss.sum()
                total_g_positive_constraints_loss.backward()
                g_constraints_loss_value_batch += total_g_positive_constraints_loss.item()

                # ! update g
                for i in range(args.NUM_DISTRIBUTION):
                    optimizer_g[i].step()

                # Just for the last iteration keep the gradient on f intact
                if inner_iter != args.N_Gnet_ITERS:
                    for i in range(args.NUM_DISTRIBUTION):
                        optimizer_f[i].zero_grad()

            ######################################################
            #                Inner Loop Ends                     #
            ######################################################
            # Generator generates the miu samples
            miu = generator_h(epsilon)
            # TODO change the miu[] because miu includes the weights
            miu_mean = miu[:, :args.INPUT_DIM].mean(dim=0).cpu()
            miu_var = np.cov(miu[:, :args.INPUT_DIM].cpu().detach().numpy().T)
            miu_mean_value_batch += miu_mean
            miu_var_value_batch += miu_var

            remaining_f_loss = torch.ones(args.NUM_DISTRIBUTION)
            # FIXME add wegihts
            # ? The 3rd loss part useful for f/h parameters
            for i in range(args.NUM_DISTRIBUTION):
                remaining_f_loss[i] = - torch.dot(convex_f[i](miu).reshape(-1),
                                                  miu_i[:, -args.NUM_DISTRIBUTION + i, 0]) / args.BATCH_SIZE

                remaining_f_loss_value_batch[i] += remaining_f_loss[i].item()
            total_remaining_f_loss = remaining_f_loss.sum()
            total_remaining_f_loss.backward(retain_graph=True)

            # Flip the gradient sign for parameters in convex f
            # Because we need to solve "sup" of the loss for f
            for p in list(convex_f.parameters()):
                p.grad.copy_(-p.grad)
            # ! update f
            for i in range(args.NUM_DISTRIBUTION):
                optimizer_f[i].step()

            # Clamp the positive constraints on the convex_f_params
            for p in f_positive_params:
                p.data.copy_(torch.relu(p.data))

            if medium_iter != args.N_Fnet_ITERS:
                optimizer_h.zero_grad()

        ######################################################
        #               Medium Loop Ends                     #
        ######################################################
        # ? The 4th loss part useful for h parameters:
        # TODO don't need to worry about * args.NUM_DISTRIBUTION, but needs to cut the output
        # mu_2moment_loss_value_batch is total 4 distributions combined F
        mu_2moment_loss = 0.5 * \
            miu[:, :args.INPUT_DIM].pow(2).sum(dim=1).mean()
        mu_2moment_loss_value_batch += mu_2moment_loss.item()

        # ! update h
        mu_2moment_loss.backward()
        # The four parts loss gradients are accumulated
        optimizer_h.step()

        miu_mean_value_batch = miu_mean_value_batch / args.N_Fnet_ITERS
        miu_var_value_batch = miu_var_value_batch / args.N_Fnet_ITERS

        g_OT_loss_value_batch[:] = [
            item / (args.N_Gnet_ITERS * args.N_Fnet_ITERS) for item in g_OT_loss_value_batch]
        g_constraints_loss_value_batch /= (args.N_Gnet_ITERS *
                                           args.N_Fnet_ITERS)
        remaining_f_loss_value_batch[:] = [
            item / args.N_Fnet_ITERS for item in remaining_f_loss_value_batch]

        ##### Calculate W2 batch loss ###############
        # FIXME different weights!!
        w2_loss_value_batch = sum(g_OT_loss_value_batch) + \
            sum(remaining_f_loss_value_batch) + \
            mu_2moment_loss_value_batch + \
            0.5 * (miu_i[:, :args.INPUT_DIM, :].pow(2).sum(dim=1) *
                   miu_i[:, -args.NUM_DISTRIBUTION:, 0]).sum(dim=1).mean().item()
        w2_loss_value_batch *= 2

        ##### Calculate all epoch loss ###############
        w2_loss_value_epoch += w2_loss_value_batch
        miu_mean_value_epoch += miu_mean_value_batch
        miu_var_value_epoch += miu_var_value_batch

        g_OT_loss_value_epoch = [
            a + b for a,
            b in zip(
                g_OT_loss_value_epoch,
                g_OT_loss_value_batch)]
        g_constraints_loss_value_epoch += g_constraints_loss_value_batch
        remaining_f_loss_value_epoch = [
            a + b for a,
            b in zip(
                remaining_f_loss_value_epoch,
                remaining_f_loss_value_batch)]
        mu_2moment_loss_value_epoch += mu_2moment_loss_value_batch

        if batch_idx % args.log_interval == 0:
            logging.info('Train_Epoch: {} [{}/{} ({:.0f}%)] avg_dstb_g_OT_loss: {:.4f} avg_dstb_remaining_f_loss: {:.4f} mu_2moment_loss: {:.4f} g_constraint_loss: {:.4f} W2_loss: {:.4f} '.format(
                epoch,
                batch_idx * len(real_data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                sum(g_OT_loss_value_batch),
                sum(remaining_f_loss_value_batch),
                mu_2moment_loss_value_batch,
                miu_mean_value_batch.mean().tolist(),
                miu_var_value_batch.mean().tolist(),
                g_constraints_loss_value_batch,
                w2_loss_value_batch
            ))

    w2_loss_value_epoch /= len(train_loader)
    g_OT_loss_value_epoch[:] = [
        item / len(train_loader) for item in g_OT_loss_value_epoch]
    g_constraints_loss_value_epoch /= len(train_loader)
    remaining_f_loss_value_epoch[:] = [
        item / len(train_loader) for item in remaining_f_loss_value_epoch]
    mu_2moment_loss_value_epoch /= len(train_loader)
    miu_mean_value_epoch /= len(train_loader)
    miu_var_value_epoch /= len(train_loader)
    results.add(
        epoch=epoch,
        w2_loss_train_samples=w2_loss_value_epoch,
        g_OT_train_loss=g_OT_loss_value_epoch,
        g_constraints_train_loss=g_constraints_loss_value_epoch,
        remaining_f_train_loss=remaining_f_loss_value_epoch,
        mu_2moment_train_loss=mu_2moment_loss_value_epoch,
        miu_mean_train=miu_mean_value_epoch.tolist(),
        miu_var_train=miu_var_value_epoch.tolist()
    )
    results.save()
    return w2_loss_value_epoch, g_OT_loss_value_epoch, g_constraints_loss_value_epoch, remaining_f_loss_value_epoch, mu_2moment_loss_value_epoch


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    Real Training Process
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
total_w2_epoch_loss_list = []
total_g_OT_epoch_loss_list = []
total_g_constraint_epoch_loss_list = []
total_remaining_f_epoch_loss_list = []
total_mu_2moment_epoch_loss_list = []

for epoch in range(1, args.epochs + 1):
    w2_loss_value_epoch, g_OT_loss_value_epoch, g_constraints_loss_value_epoch, remaining_f_loss_value_epoch, mu_2moment_loss_value_epoch = train(
        epoch)

    total_w2_epoch_loss_list.append(w2_loss_value_epoch)
    total_g_OT_epoch_loss_list.append(g_OT_loss_value_epoch)
    total_g_constraint_epoch_loss_list.append(g_constraints_loss_value_epoch)
    total_remaining_f_epoch_loss_list.append(remaining_f_loss_value_epoch)
    total_mu_2moment_epoch_loss_list.append(mu_2moment_loss_value_epoch)

    if args.schedule_learning_rate:
        if epoch % args.lr_schedule_per_epoch == 0:
            for i in range(args.NUM_DISTRIBUTION):
                optimizer_f[i].param_groups[0]['lr'] = optimizer_f[i].param_groups[0]['lr'] * \
                    args.lr_schedule_scale
                optimizer_g[i].param_groups[0]['lr'] = optimizer_g[i].param_groups[0]['lr'] * \
                    args.lr_schedule_scale
            optimizer_h.param_groups[0]['lr'] = optimizer_h.param_groups[0]['lr'] * \
                args.lr_schedule_scale

    if epoch % 5 == 0:
        LLU.dump_nn(generator_h, convex_f, convex_g, epoch,
                    model_save_path, num_distribution=args.NUM_DISTRIBUTION, save_f=args.save_f)

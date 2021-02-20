from __future__ import print_function
import logging
import GPUtil

from torch.autograd import Variable
import torch.utils.data
import torch.optim as optim
import torch
import optimal_transport_modules.log_utils as LLU
import optimal_transport_modules.generate_NN as g_NN
import optimal_transport_modules.pytorch_utils as PTU
import optimal_transport_modules.plot_utils as PLU
import optimal_transport_modules.generate_data as g_data
from optimal_transport_modules.cfg import Cfg3digit as Cfg_class

cfg = Cfg_class()

gpus_choice = GPUtil.getFirstAvailable(
    order='random', maxLoad=0.5, maxMemory=0.5, attempts=1, interval=900, verbose=False)
PTU.set_gpu_mode(True, gpus_choice[0])
cfg.LR_g = cfg.LR_f
results_save_path, model_save_path, results, testresults = LLU.init_path(cfg)
kwargs = {'num_workers': 4, 'pin_memory': True}
##### For computing the constraint loss of negtive weights ######


def compute_constraint_loss(list_of_params):
    loss_val = 0
    for p in list_of_params:
        loss_val += torch.relu(-p).pow(2).sum()
    return loss_val


"""""""""""""""""""""""""""""""""""""""""""""""""""
            Data and neural network setup
"""""""""""""""""""""""""""""""""""""""""""""""""""
train_data = g_data.marginal_mnist_3loop_ficnn_handle(cfg)
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=cfg.BATCH_SIZE, shuffle=True, **kwargs)

PLU.plot_2dmarginal(cfg, train_data, results_save_path, -6, 6)

convex_f, convex_g, generator_h = g_NN.generate_FixedWeight_NN(cfg)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        Initialization with some positive parameters
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
f_positive_params, g_positive_params = [], []

for i in range(cfg.NUM_DISTRIBUTION):
    for p in list(convex_f[i].parameters()):
        if hasattr(p, 'be_positive'):
            f_positive_params.append(p)

    for p in list(convex_g[i].parameters()):
        if hasattr(p, 'be_positive'):
            g_positive_params.append(p)

    convex_f[i].cuda(PTU.device)
    convex_g[i].cuda(PTU.device)
generator_h.cuda(PTU.device)

optimizer_f, optimizer_g = [], []

for i in range(cfg.NUM_DISTRIBUTION):
    optimizer_f.append(optim.Adam(
        convex_f[i].parameters(), lr=cfg.LR_g))
    optimizer_g.append(
        optim.Adam(convex_g[i].parameters(), lr=cfg.LR_g))
optimizer_h = optim.Adam(
    generator_h.parameters(),
    lr=cfg.LR_h)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    Training function definition
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def train(epoch):
    convex_f.train()
    convex_g.train()
    generator_h.train()

    # These values are just for saving data
    w2_loss_value_epoch = 0
    g_ot_loss_value_epoch = [0] * cfg.NUM_DISTRIBUTION
    g_constraints_loss_value_epoch = 0
    remaining_f_loss_value_epoch = [0] * cfg.NUM_DISTRIBUTION
    mu_2moment_loss_value_epoch = 0

    for batch_idx, real_data in enumerate(train_loader):
        if cfg.NUM_DISTRIBUTION == 1:
            real_data = real_data[0].cuda(PTU.device)
        else:
            real_data = real_data.cuda(PTU.device)

        miu_i = real_data[:, :, 0:cfg.NUM_DISTRIBUTION]
        miu_i = Variable(miu_i, requires_grad=True)
        if cfg.convolution_flag is True:
            epsilon = g_data.torch_normal_gaussian(
                cfg.INPUT_DIM, N_TEST=cfg.BATCH_SIZE, kernel_size=1)
        else:
            epsilon = g_data.torch_normal_gaussian(
                cfg.INPUT_DIM, N_TEST=cfg.BATCH_SIZE)
        epsilon = epsilon.cuda(PTU.device)
        epsilon = Variable(epsilon)

        # containing four distribution
        g_OT_loss_value_batch = [0] * cfg.NUM_DISTRIBUTION
        g_constraints_loss_value_batch = 0  # containing four g networks
        remaining_f_loss_value_batch = [0] * cfg.NUM_DISTRIBUTION
        mu_2moment_loss_value_batch = 0

        ######################################################
        #                Medium Loop Begin                   #
        ######################################################
        ######### Here iterate over a given number: cfg.N_Fnet_ITERS=4 ##
        for medium_iter in range(1, cfg.N_Fnet_ITERS + 1):

            ######################################################
            #                Inner Loop Begin                   #
            ######################################################
            ######### Here iterate over a given number: cfg.N_Gnet_ITERS=16 ##
            for inner_iter in range(1, cfg.N_Gnet_ITERS + 1):

                loss_g = torch.ones(cfg.NUM_DISTRIBUTION)
                for i in range(cfg.NUM_DISTRIBUTION):
                    optimizer_g[i].zero_grad()

                    # Get the gradient of g(y):=g(miu_i_data)
                    tmp_miu_i = miu_i[:, :, i]
                    g_of_y = convex_g[i](tmp_miu_i).sum()
                    grad_g_of_y = torch.autograd.grad(
                        g_of_y, tmp_miu_i, create_graph=True)[0]

                    # For each distribution you need to calculate a f(gradient of y)
                    # it's the mean of the batch
                    f_grad_g_y = convex_f[i](grad_g_of_y).mean()
                    # The 1st loss part useful for f/g parameters
                    loss_g[i] = f_grad_g_y - torch.dot(
                        grad_g_of_y.reshape(-1), miu_i[:, :, i].reshape(-1)) / cfg.BATCH_SIZE
                    g_OT_loss_value_batch[i] += loss_g[i].item()

                total_loss_g = loss_g.sum()
                total_loss_g.backward()
                # The 2nd loss part useful for g parameters:
                g_positive_constraints_loss = cfg.LAMBDA_CVX * \
                    compute_constraint_loss(
                        g_positive_params)
                g_constraints_loss_value_batch += g_positive_constraints_loss.item()
                g_positive_constraints_loss.backward()

                # ! update g
                for i in range(cfg.NUM_DISTRIBUTION):
                    optimizer_g[i].step()

                # Just for the last iteration keep the gradient on f intact
                if inner_iter != cfg.N_Gnet_ITERS:
                    for i in range(cfg.NUM_DISTRIBUTION):
                        optimizer_f[i].zero_grad()

            ######################################################
            #                Inner Loop Ends                     #
            ######################################################
            if cfg.convolution_flag is True:
                miu = generator_h(epsilon).reshape(cfg.BATCH_SIZE, -1)
            else:
                miu = generator_h(epsilon)
            remaining_f_loss = torch.ones(cfg.NUM_DISTRIBUTION)
            # The 3rd loss part useful for f/h parameters
            for i in range(cfg.NUM_DISTRIBUTION):
                remaining_f_loss[i] = - convex_f[i](miu).mean()
                remaining_f_loss_value_batch[i] += remaining_f_loss[i].item()
            total_remaining_f_loss = remaining_f_loss.sum()
            total_remaining_f_loss.backward(retain_graph=True)

            # Flip the gradient sign for parameters in convex f
            # Because we need to solve "sup" of the loss for f
            for p in list(convex_f.parameters()):
                p.grad.copy_(-p.grad)
            # ! update f
            for i in range(cfg.NUM_DISTRIBUTION):
                optimizer_f[i].step()

            # Clamp the positive constraints on the convex_f_params
            for p in f_positive_params:
                p.data.copy_(torch.relu(p.data))

            if medium_iter != cfg.N_Fnet_ITERS:
                optimizer_h.zero_grad()

        ######################################################
        #               Medium Loop Ends                     #
        ######################################################
        # The 4th loss part useful for h parameters:
        # mu_2moment_loss_value_batch is total 4 distributions combined F
        mu_2moment_loss = 0.5 * \
            miu.pow(2).sum(dim=1).mean() * cfg.NUM_DISTRIBUTION
        mu_2moment_loss_value_batch += mu_2moment_loss.item() / cfg.NUM_DISTRIBUTION

        # ! update h
        mu_2moment_loss.backward()
        optimizer_h.step()

        g_OT_loss_value_batch[:] = [
            item / (cfg.N_Gnet_ITERS * cfg.N_Fnet_ITERS) for item in g_OT_loss_value_batch]
        remaining_f_loss_value_batch[:] = [
            item / cfg.N_Fnet_ITERS for item in remaining_f_loss_value_batch]
        g_constraints_loss_value_batch /= (cfg.N_Gnet_ITERS *
                                           cfg.N_Fnet_ITERS)

        ##### Calculate W2 batch loss ###############
        w2_loss_value_batch = (sum(g_OT_loss_value_batch) + sum(remaining_f_loss_value_batch)) / cfg.NUM_DISTRIBUTION + \
            mu_2moment_loss_value_batch + 0.5 * \
            miu_i.pow(2).sum(dim=1).mean().item()
        w2_loss_value_batch *= 2
        # miu_i.pow(2).sum(dim=1).mean().item() is already the mean of all distributions

        ##### Calculate all epoch loss ###############
        w2_loss_value_epoch += w2_loss_value_batch
        g_ot_loss_value_epoch = [
            a + b for a,
            b in zip(
                g_ot_loss_value_epoch,
                g_OT_loss_value_batch)]
        g_constraints_loss_value_epoch += g_constraints_loss_value_batch
        remaining_f_loss_value_epoch = [
            a + b for a,
            b in zip(
                remaining_f_loss_value_epoch,
                remaining_f_loss_value_batch)]
        mu_2moment_loss_value_epoch += mu_2moment_loss_value_batch

        if batch_idx % cfg.log_interval == 0:
            logging.info('Train_Epoch: {} [{}/{} ({:.0f}%)] avg_dstb_g_OT_loss: {:.4f} avg_dstb_remaining_f_loss: {:.4f} mu_2moment_loss: {:.4f} g_constraint_loss: {:.4f} W2_loss: {:.4f} '.format(
                epoch,
                batch_idx * len(real_data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                sum(g_OT_loss_value_batch) / cfg.NUM_DISTRIBUTION,
                sum(remaining_f_loss_value_batch) / cfg.NUM_DISTRIBUTION,
                mu_2moment_loss_value_batch,
                g_constraints_loss_value_batch,
                w2_loss_value_batch
            ))

    w2_loss_value_epoch /= len(train_loader)
    g_ot_loss_value_epoch[:] = [
        item / len(train_loader) for item in g_ot_loss_value_epoch]
    g_constraints_loss_value_epoch /= len(train_loader)
    remaining_f_loss_value_epoch[:] = [
        item / len(train_loader) for item in remaining_f_loss_value_epoch]
    mu_2moment_loss_value_epoch /= len(train_loader)

    results.add(epoch=epoch,
                w2_loss_train_samples=w2_loss_value_epoch,
                g_OT_train_loss=g_ot_loss_value_epoch,
                g_constraints_train_loss=g_constraints_loss_value_epoch,
                remaining_f_train_loss=remaining_f_loss_value_epoch,
                mu_2moment_train_loss=mu_2moment_loss_value_epoch
                )
    results.save()


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    Real Training Process
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

for epoch_realtrain in range(1, cfg.epochs + 1):
    train(epoch_realtrain)
    if cfg.schedule_learning_rate:
        if epoch_realtrain % cfg.lr_schedule_per_epoch == 0:
            for i in range(cfg.NUM_DISTRIBUTION):
                optimizer_f[i].param_groups[0]['lr'] *= cfg.lr_schedule_scale
                optimizer_g[i].param_groups[0]['lr'] *= cfg.lr_schedule_scale
            optimizer_h.param_groups[0]['lr'] *= cfg.lr_schedule_scale

    if epoch_realtrain % 1 == 0:
        LLU.dump_nn(generator_h, convex_f, convex_g,
                    epoch_realtrain, model_save_path)

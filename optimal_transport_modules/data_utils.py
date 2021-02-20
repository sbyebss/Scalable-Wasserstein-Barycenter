from __future__ import print_function
import ot
import torch
import numpy as np
from sklearn.neighbors import KernelDensity
from torch.utils.data import Dataset
import jacinle.io as io
import optimal_transport_modules.pytorch_utils as PTU
import optimal_transport_modules.generate_data as g_data
from optimal_transport_modules.record_mean_cov import select_mean_and_cov

'''
PyTorch type
'''


def kde_Gaussian_fitting(miu, bandwidth):
    kde_analyzer = KernelDensity(
        kernel='gaussian', bandwidth=bandwidth).fit(miu)
    return kde_analyzer


def second_moment_no_average(batch_dim):
    return batch_dim.pow(2).sum(dim=1)


def second_moment_single_dist(batch_dim):
    return batch_dim.pow(2).sum(dim=1).mean()


def second_moment_all_dist(batch_dim_dist):
    return batch_dim_dist.pow(2).sum(dim=1).mean(dim=0)


def inprod_average(batch_dim_1, batch_dim_2):
    assert batch_dim_1.shape[0] == batch_dim_2.shape[0]
    batch_size = batch_dim_1.shape[0]
    inner_product_avg = torch.dot(batch_dim_1.reshape(-1),
                                  batch_dim_2.reshape(-1)) / batch_size
    return inner_product_avg


def inprod(batch_dim_1, batch_dim_2):
    innner_product = torch.dot(batch_dim_1.reshape(-1),
                               batch_dim_2.reshape(-1))
    return innner_product


def grad_of_function(input_samples, network):
    g_of_y = network(input_samples).sum()
    gradient = torch.autograd.grad(
        g_of_y, input_samples, create_graph=True)[0]
    return gradient


def two_loop_loss_in_W2(convex_f_list, grad_g_of_y, miu_i, dist_weight, idx_dist):
    n_dist = dist_weight.shape[0]

    #! The 2nd loss part useful for f/g parameters
    f_grad_g_y = convex_f_list[idx_dist](grad_g_of_y).mean()

    #! The 4th loss part useful for f/g parameters
    for j in range(n_dist):
        f_grad_g_y -= dist_weight[j] * convex_f_list[j](grad_g_of_y).mean()

    #! The 1st loss part useful for g parameters
    inner_product = inprod_average(grad_g_of_y, miu_i)

    #! The 3rd loss part useful for g parameters
    half_moment_grad_of_g = 0.5 * second_moment_single_dist(grad_g_of_y)

    loss_gi = (f_grad_g_y - inner_product +
               half_moment_grad_of_g) * dist_weight[idx_dist]
    return loss_gi


'''
localized POT library
'''


def w2_distance_samples_solver(sample1_n_d, sample2_n_d):
    # see here for details
    # https://pythonot.github.io/all.html#ot.emd
    # https://pythonot.github.io/all.html#ot.emd2
    assert sample1_n_d.shape == sample2_n_d.shape
    num_sample = sample1_n_d.shape[0]
    a = np.ones([num_sample]) / num_sample
    b = np.ones([num_sample]) / num_sample
    tmp_marginal_1 = np.expand_dims(sample1_n_d, axis=0)
    tmp_marginal_2 = np.expand_dims(sample2_n_d, axis=1)
    M = tmp_marginal_1 - tmp_marginal_2
    M = np.sum(np.abs(M)**2, axis=2)
    return ot.emd2(a, b, M)


def free_support_barycenter(measures_locations, measures_weights, X_init, b=None, weights=None, numItermax=100, stopThr=1e-7, use_sinkhorn=False):
    g_sinkhorn_reg = 0.1
    iter_count = 0
    N = len(measures_locations)
    k = X_init.shape[0]
    d = X_init.shape[1]
    if b is None:
        b = np.ones((k,)) / k
    if weights is None:
        weights = np.ones((N,)) / N

    X = X_init

    log_dict = {}
    displacement_square_norm = stopThr + 1.
    while (displacement_square_norm > stopThr and iter_count < numItermax):
        T_sum = np.zeros((k, d))
        for (measure_locations_i, measure_weights_i, weight_i) in zip(measures_locations, measures_weights, weights.tolist()):
            M_i = ot.dist(X, measure_locations_i)
            if use_sinkhorn:
                T_i = ot.bregman.sinkhorn(
                    b, measure_weights_i, M_i, g_sinkhorn_reg)
            else:
                T_i = ot.emd(b, measure_weights_i, M_i)
            T_sum = T_sum + weight_i * \
                np.reshape(1. / b, (-1, 1)) * \
                np.matmul(T_i, measure_locations_i)

        displacement_square_norm = np.sum(np.square(T_sum - X))

        X = T_sum
        print('iteration %d, displacement_square_norm=%f\n',
              iter_count, displacement_square_norm)

        iter_count += 1

    return X


'''
MNIST utils
'''


class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)


# def extract_three_number(total_data):
#     idx_train = (total_data.targets == 0) + (total_data.targets ==
#                                              1) + (total_data.targets == 7)
#     total_data.targets = total_data.targets[idx_train]
#     total_data.data = total_data.data[idx_train]
#     return total_data


class CustomMnistDataset(Dataset):
    def __init__(self, data, target, transform=None):

        self.data = data
        self.target = target
        self.transform = transform

    def __len__(self):
        assert len(self.target) == len(self.data)
        return len(self.target)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_idxed = self.data[idx]
        target_idxed = self.target[idx].float()
        # sample = {'data': data_idxed, 'target': target_idxed}

        if self.transform:
            data_idxed = self.transform(data_idxed)

        return [data_idxed, target_idxed]


'''
Gaussian utils
'''


def get_gmm_param(trial, cond=-1):
    if cond > 0:
        MEAN, COV = select_mean_and_cov(trial, range_cond=cond)
    else:
        MEAN, COV = select_mean_and_cov(trial)
    INPUT_DIM = MEAN[0].shape[1]
    OUTPUT_DIM = INPUT_DIM
    NUM_DISTRIBUTION = len(MEAN)
    NUM_GMM_COMPONENT = []
    for i in range(NUM_DISTRIBUTION):
        NUM_GMM_COMPONENT.append(MEAN[i].shape[0])
    high_dim_flag = INPUT_DIM > 2
    return MEAN, COV, INPUT_DIM, OUTPUT_DIM, NUM_DISTRIBUTION, NUM_GMM_COMPONENT, high_dim_flag


'''
Average the 2 layer neural networks
'''


def average_nn(args, **kwargs):
    averaged_parameters = np.zeros([args.N_SAMPLES, args.INPUT_DIM])
    tmp_data = averaged_parameters
    n_samp_of_subset = int(args.N_SAMPLES / args.NUM_DISTRIBUTION)
    for i in range(args.NUM_DISTRIBUTION):
        model_param = io.load(args.get_nn(**kwargs) +
                              f"/subset_{i+1}_samples_{args.subset_samples}/trial_26/storing_models/nn_2layer_epoch200.pt")

        assert args.N_SAMPLES == model_param['layer1.weight'].shape[0]
        tmp_data[:, :-1] = PTU.torch2numpy(model_param['layer1.weight'])
        tmp_data[:, -
                 1] = PTU.torch2numpy(model_param['last_layer.weight'].squeeze())
        if i == args.NUM_DISTRIBUTION - 1:
            averaged_parameters[(i * n_samp_of_subset)
                                 :] = tmp_data[(i * n_samp_of_subset):]
        else:
            averaged_parameters[i * n_samp_of_subset:
                                (i + 1) * n_samp_of_subset] = tmp_data[i * n_samp_of_subset:
                                                                       (i + 1) * n_samp_of_subset]

    return averaged_parameters


'''
get marginal data handle
'''


def get_marginal_list(cfg, type_data='2block'):
    if type_data == '2block':
        marginal_data = g_data.marginal_data_blocks_3loop_ficnn(
            cfg)[:, :, :-1]
    elif type_data == 'circ_squa':
        marginal_data = g_data.marginal_data_circ_squ_3loop_ficnn(
            cfg)[:, :, :-1]
    elif type_data == 'mnist0-1':
        marginal_data = g_data.marginal_mnist_3loop_ficnn_handle(
            cfg)
    elif type_data == '3digit':
        marginal_data = g_data.marginal_data_3digit_3loop_ficnn(
            cfg)[:, :, :-1]
    elif type_data == 'ellipse':
        marginal_data = g_data.marginal_data_ellipse_3loop_ficnn(
            cfg)[:, :, :-1]
    elif type_data == 'line':
        marginal_data = g_data.marginal_data_line_3loop_ficnn(
            cfg)[:, :, :-1]
    elif type_data == 'usps_mnist':
        marginal_data = g_data.marginal_usps_3loop_ficnn_handle(
            cfg)[0][torch.randperm(5000), :, :-1]
    elif type_data == 'mnist_group':
        if cfg.N_TEST == 25:
            idx_digit = torch.zeros(25).long()
            for idx in range(5):
                idx_digit[idx * 5:(idx + 1) * 5] = 5000 * idx + torch.arange(5)
            marginal_data = g_data.marginal_mnist_3loop_ficnn_handle(
                cfg)[idx_digit]
        else:
            marginal_data = g_data.marginal_mnist_3loop_ficnn_handle(
                cfg)[torch.randperm(25000)]
    elif type_data == 'cifar':
        marginal_data = g_data.marginal_cifar_handle(cfg)
    elif type_data == 'gmm':
        marginal_data = g_data.marginal_data_gmm_3loop_ficnn(
            cfg)[:, :, :-1]
    return marginal_data.permute(2, 0, 1)

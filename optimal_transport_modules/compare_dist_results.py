from __future__ import print_function
import ot
from scipy.stats import entropy
import numpy as np
import scipy.linalg as ln
from numpy import linalg as LA
import torch
from optimal_transport_modules.log_utils import ResultsLog
import optimal_transport_modules.generate_data as g_data
import optimal_transport_modules.generate_NN as g_NN
import optimal_transport_modules.data_utils as DTU


def normalize(np_array):
    np_array /= np_array.sum()
    return np_array


def entropy_between_ICNN_reference(save_path, brct_KDE_ICNN, brct_standard_refer):
    entropyresults = ResultsLog(save_path)
    KL_between_ICNN_and_reference = entropy(brct_KDE_ICNN, brct_standard_refer)
    L2_between_ICNN_and_reference = LA.norm(
        brct_KDE_ICNN - brct_standard_refer, 2) / LA.norm(brct_standard_refer, 2)
    L1_between_ICNN_and_reference = LA.norm(
        brct_KDE_ICNN - brct_standard_refer, 1) / LA.norm(brct_standard_refer, 1)
    entropyresults.add(
        KL_between_ICNN_and_reference=KL_between_ICNN_and_reference)
    entropyresults.add(
        L1_between_ICNN_and_reference=L1_between_ICNN_and_reference)
    entropyresults.add(
        L2_between_ICNN_and_reference=L2_between_ICNN_and_reference)


def ideal_mean_and_cov_originD(args, weights_distribution):
    mean_ideal = []
    for i in range(args.NUM_DISTRIBUTION):
        mean_ideal.append(args.MEAN[i] * weights_distribution[i])
    mean_ideal_originD = sum(mean_ideal).mean(axis=0)
    Sn = np.asmatrix(np.eye(args.INPUT_DIM))

    # si represents the -1/2 power, the s means the 1/2 power and S represents S    matrix itself.
    num_itr = 0
    while True:
        num_itr += 1
        s = ln.sqrtm(Sn)
        si = LA.inv(s)
        ans_medium = np.asmatrix(np.zeros_like(Sn))
        for i in range(args.NUM_DISTRIBUTION):
            ans_medium += weights_distribution[i] * ln.sqrtm(
                np.matmul(np.matmul(s, np.asmatrix(args.COV[i][0, :, :])), s)
            )
        Sn_1 = np.matmul(ans_medium, ans_medium)
        Sn_1 = np.matmul(np.matmul(si, Sn_1), si)

        if np.power(Sn_1 - Sn, 2).sum() <= 1e-10:
            break
        Sn = Sn_1
    COV_ideal_originD = Sn_1

    return mean_ideal_originD, COV_ideal_originD


def mean_cov_from_samples(miu):
    return mean_real_originD(miu), cov_real_originD(miu)


def cov_real_originD(miu):
    if type(miu) is torch.Tensor:
        return np.cov(miu.detach().numpy().T)
    return np.cov(miu.T)


def mean_real_originD(miu):
    if type(miu) is torch.Tensor:
        return np.mean(miu.detach().numpy(), 0)
    return np.mean(miu, 0)


def Frobenius_relative_error(COV_ideal_originD, COV_real_originD):
    return LA.norm(
        COV_ideal_originD - COV_real_originD) / LA.norm(COV_ideal_originD)


def Frobenius_absolute_error(COV_ideal_originD, COV_real_originD):
    return LA.norm(
        COV_ideal_originD - COV_real_originD)


def BW2_distance(mean_ideal, mean_ours, cov_ideal, cov_ours):
    under_squre = ln.sqrtm(cov_ours)@cov_ideal@ln.sqrtm(cov_ours)
    # print(0.5 * LA.norm(mean_ideal - mean_ours)**2)
    # print(0.5 * np.trace(cov_ideal) + 0.5 *
    #       np.trace(cov_ours) - np.trace(ln.sqrtm(under_squre)))
    return 0.5 * LA.norm(mean_ideal - mean_ours)**2 + 0.5 * np.trace(cov_ideal) + 0.5 * np.trace(cov_ours) - np.trace(ln.sqrtm(under_squre))


def gaussian_compare_package(miu, cfg):
    miu = miu if isinstance(miu, np.ndarray) else miu.cpu()
    cov_ours = cov_real_originD(miu)
    weights_dist = np.ones(cfg.NUM_DISTRIBUTION) / cfg.NUM_DISTRIBUTION
    _, cov_ideal = ideal_mean_and_cov_originD(cfg, weights_dist)
    mean_ours = mean_real_originD(miu)
    mean_ideal = np.zeros_like(mean_ours)
    BW2_UVP = 100 * BW2_distance(
        mean_ideal, mean_ours, cov_ideal, cov_ours) * 2 / np.trace(cov_ideal)
    # print(miu.shape[1])
    # print(np.trace(cov_ideal) / miu.shape[1])
    # print(cov_ideal)
    # print(np.diag(cov_ours))
    return BW2_UVP


def bayesian_compare_package(miu, miu_full_posterior):
    mean_compared, mean_ideal = mean_real_originD(
        miu), mean_real_originD(miu_full_posterior)
    # mean_compared = mean_real_originD(
    #     miu) + np.load('input_data/bayesian_inference/bike_posterior/subset_mean.npy')
    # mean_ideal = np.load(
    #     'input_data/bayesian_inference/bike_posterior/full_mean.npy')
    # print(mean_compared)
    # print(mean_ideal)
    cov_compared, cov_ideal = cov_real_originD(
        miu), cov_real_originD(miu_full_posterior)

    mean_error = Frobenius_absolute_error(mean_ideal, mean_compared)
    Frob_error = Frobenius_absolute_error(cov_ideal, cov_compared)
    BW2_UVP = 100 * BW2_distance(
        mean_ideal, mean_compared, cov_ideal, cov_compared) * 2 / np.trace(cov_ideal)
    # iclr_uvp = 100 * calculate_frechet_distance(
    #     mean_compared, cov_compared,
    #     mean_ideal, cov_ideal,
    # ) / np.trace(cov_ideal)
    # print('absolute mean_error')
    # print(mean_error)
    # print('absolute cov_error')
    # print(Frob_error)
    # print('BW2 UVP')
    # print(BW2_UVP)
    return BW2_UVP


def cuturi_input_package(cfg):
    avg_measure_weights = np.ones(cfg.N_SAMPLES) / cfg.N_SAMPLES
    measure_weights = g_data.repeat_list(
        avg_measure_weights, cfg.NUM_DISTRIBUTION)
    X_init = np.random.normal(0., 1., (cfg.N_SAMPLES, cfg.INPUT_DIM))
    dirac_weight_barycenter_sample = avg_measure_weights
    return avg_measure_weights, measure_weights, X_init, dirac_weight_barycenter_sample


def ideal_projected_param(args, weights_distribution, miu):
    mean_ideal_originD, COV_ideal_originD = ideal_mean_and_cov_originD(
        args, weights_distribution)
    root_of_Sn = ln.sqrtm(COV_ideal_originD)
    COV_real_originD = np.cov(miu.T)

    if args.high_dim_flag:
        projection_component = np.zeros(
            [args.NUM_DISTRIBUTION, args.INPUT_DIM])

        projection_component[0, 1::2] = 1 / np.sqrt(args.INPUT_DIM / 2)
        projection_component[1, ::2] = 1 / np.sqrt(args.INPUT_DIM / 2)

        miu = np.matmul(miu, projection_component.T)

        mean_ideal = np.matmul(projection_component, mean_ideal)
        COV_ideal = np.matmul(
            np.matmul(projection_component, COV_ideal_originD), projection_component.T)
    else:
        COV_ideal = COV_ideal_originD

    W2_list = []
    for i in range(args.NUM_DISTRIBUTION):
        W2_list.append(
            LA.norm(args.MEAN[i][0, :] - mean_ideal_originD, 2)**2 +
            np.matrix.trace(COV_ideal_originD +
                            np.asmatrix(args.COV[i][0, :, :]) - 2 * ln.sqrtm(
                                np.matmul(
                                    np.matmul(root_of_Sn, np.asmatrix(args.COV[i][0, :, :])), root_of_Sn)
                            )
                            )
        )

    W2_total = sum(W2_list) / args.NUM_DISTRIBUTION
    half_moment = 0.5 * \
        (np.trace(COV_ideal_originD) +
         np.inner(mean_ideal_originD, mean_ideal_originD))

    return miu, mean_ideal, COV_ideal_originD, COV_real_originD, COV_ideal, W2_total, half_moment

#!mnist


def test_output_classi(model, device, test_loader):
    model.eval()
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            output = model(data)
    return output


def test_correct_rate_classi(output, device, test_loader):

    for _, target in test_loader:
        target = target.cuda(device)
        test_loss = torch.nn.functional.mse_loss(
            output, target, reduction='sum').item()

        rounded_output = torch.round(output)
        correct = (rounded_output == target).sum().item()

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def test_mnist_classifier(model, device, test_loader, cfg=None):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            target = (target == 7).float()
            data, target = data.to(device), target.to(device)
            output = model(data)
            # test_loss += torch.nn.functional.mse_loss(
            #     output, target, reduction='sum').item()
            test_loss += (-target * torch.log(output) -
                          (1 - target) * torch.log(1 - output)).sum()

            # rounded_output = torch.round(output)
            # rounded_output = (output > output.mean()).float()
            rounded_output = (output > 0.5).float()
            correct += (rounded_output == target).sum().item()
            # import matplotlib.pyplot as plt
            # plt.scatter(np.arange(len(output)),output.cpu())
            # plt.savefig('vis0.png')
            # plt.close()

    test_loss /= len(test_loader.dataset)

    # print(cfg.NUM_NEURON)
    if cfg != None and hasattr(cfg, 'idx_subset'):
        print(cfg.idx_subset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

#! generate barycenter data


def barycenter_sampler(cfg, device, results_save_path=None, load_epoch=None, usps=False):
    if results_save_path is None:
        results_save_path = cfg.get_save_path()

    if type(cfg).__name__ == 'CfgNN':
        epsilon = g_data.torch_normal_gaussian(
            cfg.INPUT_DIM, N_TEST=cfg.N_TRAIN_SAMPLES, device=device)
    elif usps is True:
        _, mean, cov = g_data.marginal_usps_3loop_ficnn_handle(
            cfg)
        epsilon = g_data.torch_samples_generate_Gaussian(
            cfg.N_TEST, mean, cov, device=device)
    else:
        epsilon = g_data.torch_normal_gaussian(
            cfg.INPUT_DIM, N_TEST=cfg.N_TEST, device=device)
    generator_h = g_NN.generate_FixedWeight_h_NN(cfg)

    if load_epoch is None:
        generator_h = g_NN.load_generator_h(
            results_save_path, generator_h, epochs=cfg.epochs, device=device)
    else:
        generator_h = g_NN.load_generator_h(
            results_save_path, generator_h, epochs=load_epoch, device=device)
    miu = generator_h(epsilon)
    return miu


def barycenter_pushforward(cfg, device, results_save_path=None, load_epoch=None, type_data='2block', return_marginal=False):
    miu_list = []
    if results_save_path is None:
        results_save_path = cfg.get_save_path()

    marginal = DTU.get_marginal_list(cfg, type_data=type_data).float()
    _, generator_g_list = g_NN.generate_FixedWeight_fg_NN(cfg)
    for idx in range(cfg.NUM_DISTRIBUTION):
        if marginal[0].shape[0] > 25:
            tmp_margin = marginal[idx][50:(50 + cfg.N_TEST)]
        else:
            tmp_margin = marginal[idx]
        if load_epoch is None:
            generator_gi = g_NN.load_generator_fg(
                results_save_path, idx, generator_g_list[idx], cfg.epochs, device=device)
        else:
            generator_gi = g_NN.load_generator_fg(
                results_save_path, idx, generator_g_list[idx], load_epoch, device=device)

        tmp_margin = tmp_margin.cuda(device)
        tmp_margin = torch.autograd.Variable(tmp_margin, requires_grad=True)
        generator_gi.cuda(device)

        g_of_miu_i = generator_gi(tmp_margin).sum()
        miu = torch.autograd.grad(
            g_of_miu_i, tmp_margin)[0]
        miu_list.append(miu)
    if return_marginal:
        if marginal[0].shape[0] > 25:
            return miu_list, marginal[:, 50:(50 + cfg.N_TEST)]
        else:
            return miu_list, marginal
    else:
        return miu_list


def barycenter_backward(barycenter, cfg, device, results_save_path=None, load_epoch=None):
    marginal_list = []
    if results_save_path is None:
        results_save_path = cfg.get_save_path()

    generator_f_list, _ = g_NN.generate_FixedWeight_fg_NN(cfg)
    for idx in range(cfg.NUM_DISTRIBUTION):
        if load_epoch is None:
            generator_fi = g_NN.load_generator_fg(
                results_save_path, idx, generator_f_list[idx], cfg.epochs, choice='f', device=device)
        else:
            generator_fi = g_NN.load_generator_fg(
                results_save_path, idx, generator_f_list[idx], load_epoch, choice='f', device=device)

        barycenter = barycenter.cuda(device)
        barycenter = torch.autograd.Variable(barycenter, requires_grad=True)
        generator_fi.cuda(device)

        g_of_miu_i = generator_fi(barycenter).sum()
        miu = torch.autograd.grad(
            g_of_miu_i, barycenter)[0]
        marginal_list.append(miu)
    return marginal_list


def cuturi_barycenter_sampler(measures_locations, cfg, use_sinkhorn=False):
    _, measure_weights, X_init, _ = cuturi_input_package(cfg)
    barycenter_samples = DTU.free_support_barycenter(
        measures_locations, measure_weights, X_init, use_sinkhorn=use_sinkhorn)
    return barycenter_samples

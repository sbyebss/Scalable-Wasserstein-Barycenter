import os
import torch.utils.data
import torch
from math import isclose
import numpy as np
from scipy.stats import multivariate_normal
from torchvision import datasets, transforms
import optimal_transport_modules.pytorch_utils as PTU
import optimal_transport_modules.data_utils as DTU
import jacinle.io as io

#! gaussian


def generate_uniform_weights(args, choice_weight, NUM_SAMPLES, NUM_WEIGHTS_repeated):
    if choice_weight == 0:
        # ? This is fixed average weight
        num_of_weights_pair = NUM_SAMPLES / \
            (NUM_WEIGHTS_repeated * args.BATCH_SIZE)
        weights_vector_compress = np.ones(
            [int(num_of_weights_pair), args.NUM_DISTRIBUTION]) / args.NUM_DISTRIBUTION
    # ? This is random uniform distribution
    elif choice_weight == 1:
        alpha = np.ones(args.NUM_DISTRIBUTION)
        num_of_weights_pair = NUM_SAMPLES / \
            (NUM_WEIGHTS_repeated * args.BATCH_SIZE)
        weights_vector_compress = np.random.dirichlet(
            alpha, int(num_of_weights_pair))
        np.random.shuffle(weights_vector_compress)
    # ? This is linspace uniform distribution(only useful for 2 distribution)
    elif choice_weight == 2:
        num_of_weights_pair = NUM_SAMPLES / args.BATCH_SIZE / NUM_WEIGHTS_repeated
        a1 = np.linspace(0 + 1e-5, 1 - 1e-5, num=num_of_weights_pair,
                         endpoint=True).reshape(-1, 1)
        np.random.shuffle(a1)
        a2 = 1 - a1
        weights_vector_compress = np.concatenate((a1, a2), axis=1)
    # ? This is linspace uniform distribution(only useful for 3 distribution)
    elif choice_weight == 3:
        weights_vector_compress = np.array(
            [[1, 0, 3], [0, 1, 3],
             [2, 0, 2], [1, 1, 2], [0, 2, 2],
             [3, 0, 1], [2, 1, 1], [1, 2, 1], [0, 3, 1],
             [3, 1, 0], [2, 2, 0], [1, 3, 0]]) / 4

    weights_vector = torch.from_numpy(
        np.repeat(weights_vector_compress, repeats=NUM_WEIGHTS_repeated * args.BATCH_SIZE, axis=0))
    return weights_vector


def gaussian_data(num_samples, args, loop3_flag=1):
    total_data = torch.randn(
        num_samples, args.INPUT_DIM, args.NUM_DISTRIBUTION + loop3_flag)

    for i in range(args.NUM_DISTRIBUTION):
        weight_GMM = int(num_samples / args.NUM_GMM_COMPONENT[i])
        for j in range(args.NUM_GMM_COMPONENT[i]):
            total_data[(j * weight_GMM):((j + 1) * weight_GMM), :, i] = torch.from_numpy(np.random.multivariate_normal(
                args.MEAN[i][j, :], args.COV[i][j], weight_GMM))
        index_column = torch.randperm(num_samples)
        total_data[:, :, i] = total_data[index_column, :, i]
    return total_data


def marginal_data_gmm_3loop_ficnn(args):
    train_data = gaussian_data(args.N_TRAIN_SAMPLES, args)
    return train_data


def marginal_data_gmm_3loop_picnn(args):
    NUM_DISTRIBUTION = args.NUM_DISTRIBUTION
    # train data genrating
    total_data = torch.randn(
        args.N_TRAIN_SAMPLES, args.INPUT_DIM + NUM_DISTRIBUTION, NUM_DISTRIBUTION + 1)

    for i in range(NUM_DISTRIBUTION):
        # weight_GMM >= 1
        weight_GMM = int(args.N_TRAIN_SAMPLES /
                         args.NUM_GMM_COMPONENT[i])
        for j in range(args.NUM_GMM_COMPONENT[i]):
            total_data[(j * weight_GMM):((j + 1) * weight_GMM), :args.INPUT_DIM, i] = torch.from_numpy(np.random.multivariate_normal(
                args.MEAN[i][j, :], args.COV[i][j], weight_GMM))
        index_column = torch.randperm(args.N_TRAIN_SAMPLES)
        total_data[:, :, i] = total_data[index_column, :, i]

    dist_weights_train_vector = generate_uniform_weights(
        args=args, choice_weight=args.choice_weight, NUM_SAMPLES=args.N_TRAIN_SAMPLES, NUM_WEIGHTS_repeated=30)
    for i in range(NUM_DISTRIBUTION + 1):
        total_data[:, -NUM_DISTRIBUTION:, i] = dist_weights_train_vector
    return total_data


#!ellipse


def ellipse_data(num_samples, args):
    total_data = torch.randn(
        num_samples, args.INPUT_DIM_fg, args.NUM_DISTRIBUTION + 1)
    origin_data = io.load(
        'input_data/ellipse/tst_circle.mat')
    central = origin_data['central_dot']
    focal_length = origin_data['focal_length']

    for i in range(args.NUM_DISTRIBUTION):
        total_data[:, :, i] = torch.stack(
            [central[i, 0] + focal_length[i, 0] * torch.cos(torch.linspace(0, 2 * np.pi, args.N_TRAIN_SAMPLES)),
             central[i, 1] + focal_length[i, 1] * torch.sin(torch.linspace(0, 2 * np.pi, args.N_TRAIN_SAMPLES))], dim=1)
        index_column = torch.randperm(args.N_TRAIN_SAMPLES)
        total_data[:, :, i] = total_data[index_column, :, i]
    return total_data


def marginal_data_ellipse_3loop_ficnn(cfg):
    train_data = ellipse_data(cfg.N_TRAIN_SAMPLES, cfg)
    return train_data

#!line


def line_data(num_samples, args):
    total_data = torch.randn(
        num_samples, args.INPUT_DIM_fg, args.NUM_DISTRIBUTION + 1)
    line_components = io.load(
        'input_data/line/components.txt')

    for i in range(args.NUM_DISTRIBUTION):
        total_data[:, 0, i] = torch.linspace(
            line_components[2 * i, 0], line_components[2 * i + 1, 0], args.N_TRAIN_SAMPLES)
        total_data[:, 1, i] = torch.linspace(
            line_components[2 * i, 1], line_components[2 * i + 1, 1], args.N_TRAIN_SAMPLES)
        index_column = torch.randperm(args.N_TRAIN_SAMPLES)
        total_data[:, :, i] = total_data[index_column, :, i]

    return total_data


def marginal_data_line_3loop_ficnn(cfg):
    train_data = line_data(cfg.N_TRAIN_SAMPLES, cfg)
    return train_data

#!blocks


def marginal_block(num_sample, x1, x2, y1, y2, z1, z2):
    xs = (x2 - x1) * np.random.rand(num_sample, 1) + x1
    ys = (y2 - y1) * np.random.rand(num_sample, 1) + y1
    zs = (z2 - z1) * np.random.rand(num_sample, 1) + z1
    return PTU.numpy2torch(np.concatenate([xs, ys, zs], axis=1))


def blocks_data(num_samples, cfg):
    total_data = torch.randn(
        num_samples, cfg.INPUT_DIM, cfg.NUM_DISTRIBUTION + 1)

    # total_data[:, :, 0] = marginal_block(
    #     num_samples, cfg.farest_point - cfg.block_side_s, cfg.farest_point, 0, cfg.block_side_l, 0, cfg.block_side_s)
    # total_data[:, :, 1] = marginal_block(
    #     num_samples, 0, cfg.block_side_s, cfg.farest_point - cfg.block_side_s, cfg.farest_point, cfg.farest_point - cfg.block_side_l, cfg.farest_point)
    total_data[:, :, 0] = marginal_block(
        num_samples, - cfg.block_side_s / 2, cfg.block_side_s / 2, - cfg.block_side_l / 2, cfg.block_side_l / 2, - cfg.block_side_s / 2, cfg.block_side_s / 2)
    total_data[:, :, 1] = marginal_block(
        num_samples, - cfg.block_side_s / 2, cfg.block_side_s / 2, - cfg.block_side_s / 2, cfg.block_side_s / 2, - cfg.block_side_l / 2, cfg.block_side_l / 2)

    return total_data


def marginal_data_blocks_3loop_ficnn(cfg):
    train_data = blocks_data(cfg.N_TRAIN_SAMPLES, cfg)
    return train_data

#! color transfer


def get_marginal_color_list(cfg):
    return io.load(
        cfg.color_data_path + '/color_input.pth')


def color_data(num_samples, cfg):
    total_data = torch.randn(
        num_samples, cfg.INPUT_DIM, cfg.NUM_DISTRIBUTION + 1)
    marginal_data = get_marginal_color_list(cfg)
    for idx in range(cfg.NUM_DISTRIBUTION):
        total_data[:, :, idx] = marginal_data[idx]
    return total_data


def marginal_data_color_transfer_ficnn(cfg):
    train_data = color_data(cfg.N_TRAIN_SAMPLES, cfg)
    return train_data

#! circle & square


def marginal_square(num_sample, side):
    return torch.rand(
        num_sample, 2) * 2 * side - side


def marginal_ring(num_sample, ring_out, ring_inside):
    length = np.sqrt(np.random.uniform(
        0, 1, num_sample) * (ring_out**2 - ring_inside**2) + ring_inside**2)
    angle = np.pi * np.random.uniform(0, 2, num_sample)
    x_coordinate = (length * np.cos(angle)).reshape(-1, 1)
    y_coordinate = (length * np.sin(angle)).reshape(-1, 1)
    ring_data = np.concatenate(
        (x_coordinate, y_coordinate), axis=1)
    return torch.from_numpy(ring_data)


def circ_squ_data(num_samples, cfg):
    total_data = torch.randn(
        num_samples, cfg.INPUT_DIM, cfg.NUM_DISTRIBUTION + 1)
    total_data[:, :, 0] = marginal_square(
        num_samples, cfg.square_radius)
    total_data[:, :, 1] = marginal_ring(
        num_samples, cfg.ring_out, cfg.ring_inside)
    return total_data


def marginal_data_circ_squ_3loop_ficnn(cfg):
    train_data = circ_squ_data(cfg.N_TRAIN_SAMPLES, cfg)
    return train_data


def marginal_data_circ_squ_3loop_picnn(cfg):
    total_data = torch.randn(
        cfg.N_TRAIN_SAMPLES, cfg.INPUT_DIM + cfg.NUM_DISTRIBUTION, cfg.NUM_DISTRIBUTION + 1)

    # ?distribution1: square
    total_data[:, 0:2, 0] = marginal_square(
        cfg.N_TRAIN_SAMPLES, cfg.square_radius)
    # ?distribution2: ring
    length = np.sqrt(np.random.uniform(
        0, 1, cfg.N_TRAIN_SAMPLES) * (cfg.ring_out**2 - cfg.ring_inside**2) + cfg.ring_inside**2)
    angle = np.pi * np.random.uniform(0, 2, cfg.N_TRAIN_SAMPLES)
    total_data[:, 0, 1] = torch.from_numpy(length * np.cos(angle))
    total_data[:, 1, 1] = torch.from_numpy(length * np.sin(angle))

    # ? generate weights
    dist_weights_train_vector = generate_uniform_weights(
        args=cfg, choice_weight=cfg.choice_weight, NUM_SAMPLES=cfg.N_TRAIN_SAMPLES, NUM_WEIGHTS_repeated=30)
    for i in range(cfg.NUM_DISTRIBUTION + 1):
        total_data[:, -cfg.NUM_DISTRIBUTION:, i] = dist_weights_train_vector
    return total_data

#! bayesian


def marginal_data_generator_bayesian_3loop(args):
    total_data = torch.randn(
        args.N_TRAIN_SAMPLES, args.INPUT_DIM_fg, args.NUM_DISTRIBUTION + 1
        # , dtype=torch.float64
    )

    for i in range(args.NUM_DISTRIBUTION):
        total_data[:, :, i] = PTU.numpy2torch(np.load(
            args.posterior_path + f"biketrip_subset{i}_samples.npy"))
        total_data[:, :, i] = bayesian_coeff_process(
            total_data[:, :, i], args.SCALE)
    return total_data

#! neural network


def nn2layer_handle_train(args, idx_epoch=200):
    total_data = torch.randn(
        args.N_TRAIN_SAMPLES, args.INPUT_DIM_fg, args.NUM_DISTRIBUTION + 1
    )
    for i in range(args.NUM_DISTRIBUTION):
        model_param = io.load(args.get_nn() +
                              f"/subset_{i+1}/trial_26/storing_models/nn_2layer_epoch{idx_epoch}.pt")
        total_data[:, :-1, i] = model_param['layer1.weight'] * args.SCALE
        total_data[:, -1, i] = model_param['last_layer.weight'].squeeze() * \
            args.SCALE
        # *saved tmp
        # param3=io.load('/home/jfan97/Study_hard/barycenter/July20/barycenter_clean/data/Results_of_classification/distribution_5/digit17/input_dim_785/layers_0/neuron_1024/activ_celu/lr0.001/schedule_learning_rate:Yes/lr_schedule:20/batch_100/train_sample_2000/subset_3/trial_26/storing_models/nn_2layer_epoch200.pt')
        # param3['layer1.weight'].mean(axis=0)
    return total_data


#! mnist

def generate_mnist_classify_set(num_samples=200, train=True, subset=True):
    total_data = datasets.MNIST(
        root='./input_data/mnist_data', train=True)
    saved_data = torch.ones([num_samples, 785])
    saved_target = torch.zeros([num_samples, 1])
    idx_train = (total_data.targets == 1) + (total_data.targets == 7)
    for idx in range(5):
        idx_list = torch.linspace(
            idx * num_samples, (idx + 1) * num_samples, num_samples).int().long()
        saved_target = total_data.targets[idx_train][idx_list].reshape(-1, 1)
        saved_data[:, :-
                   1] = total_data.data[idx_train][idx_list].reshape(num_samples, -1).float() / 255
        saved_data[:, :-1] = (saved_data[:, :-1] - 0.1307) / 0.3081
        saved_dataset = DTU.CustomMnistDataset(saved_data, saved_target)

        torch.save(
            saved_dataset, f'./input_data/mnist_data/mnist_17_classi{idx+1}_samples{num_samples}.pt')
        total_data = datasets.MNIST(
            root='./input_data/mnist_data', train=True)


def mnist_classi_handle_train(cfg):
    fname = cfg.train_data_path + \
        f'/mnist_17_classi{cfg.idx_subset}_samples{cfg.subset_samples}.pt'
    if os.path.isfile(fname) is False:
        generate_mnist_classify_set(num_samples=cfg.subset_samples)
    return io.load(fname)


def mnist_classi_handle_test(test_data_path):
    kwargs = {'num_workers': 4, 'pin_memory': True}
    test_data = io.load(test_data_path)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=len(test_data), shuffle=True, **kwargs)
    return test_loader


def marginal_mnist_3loop_ficnn_handle(cfg):
    if cfg.NUM_DISTRIBUTION == 2:
        if isclose(cfg.num_digit, 1):
            total_data = marginal_data_3digit_3loop_ficnn(cfg)
        elif isclose(cfg.num_digit, 4):
            total_data = torch.load(
                cfg.mnist_data_path + '/mnist_0-4_vs_5-9.pt')
        else:
            # if cfg.two_digit == 17:
            #     total_data = torch.load(cfg.mnist_data_path + '/mnist_3dist_std1.pt')[
            #         :cfg.N_TRAIN_SAMPLES, :, 1:]
            # else:
            total_data = torch.load(cfg.mnist_data_path + '/mnist_3dist_std1.pt')[
                :cfg.N_TRAIN_SAMPLES, :, 0:2]

    elif cfg.NUM_DISTRIBUTION == 1:
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), DTU.ReshapeTransform((-1, 1))])
        total_data = datasets.MNIST(
            root='input_data/mnist_data', train=True, transform=transform, download=False)
    return total_data


def marginal_data_3digit_3loop_ficnn(cfg):
    cfg.N_TRAIN_SAMPLES = 60000
    total_density = torch.load(cfg.mnist_data_path + '/density_of_two3.pt')
    first3_density = total_density[0, :, :].numpy()
    second3_density = total_density[1, :, :].numpy()

    train_data = torch.randn(
        cfg.N_TRAIN_SAMPLES, cfg.INPUT_DIM, cfg.NUM_DISTRIBUTION + 1)

    train_data[:, :, 0] = sample_from_2d_matrix(
        first3_density, cfg.N_TRAIN_SAMPLES)
    train_data[:, :, 1] = sample_from_2d_matrix(
        second3_density, cfg.N_TRAIN_SAMPLES)
    return train_data


def sample_from_2d_matrix(n_n_matrix, num_samples):
    inds = np.random.choice(
        np.arange(28**2), p=n_n_matrix.reshape(-1), size=num_samples)
    inds = inds.astype('float')
    sample_xy = (np.array([inds % 28, inds // 28]).T - 14) / 2
    sample_xy[:, 1] *= -1
    noise = np.random.rand(inds.shape[0], 2) * 0.5 - 0.25
    # noise = np.random.randn(inds.shape[0], 2) * 0.1
    sample_xy += noise
    return PTU.numpy2torch(sample_xy)


def marginal_usps_3loop_ficnn_handle(cfg):
    train_data = torch.randn(
        cfg.N_TRAIN_SAMPLES, cfg.INPUT_DIM_fg, cfg.NUM_DISTRIBUTION + 1)
    dicti = io.load(cfg.usps_mnist_path)
    train_data[:, :, :-1] = dicti["train_data"]
    # This means Trial 1.0 is using standard gaussian, others are using pre-designed mean and cov
    if abs(cfg.TRIAL - 1.0) < 0.9:
        return train_data, np.zeros(cfg.INPUT_DIM), np.eye(cfg.INPUT_DIM)
    return train_data, dicti["mean"], dicti["cov"]

#! cifar


def marginal_cifar_handle(cfg):
    total_data = torch.randn(
        cfg.N_TRAIN_SAMPLES, cfg.INPUT_DIM_fg, cfg.NUM_DISTRIBUTION + 1)
    if cfg.NUM_DISTRIBUTION == 1:
        total_data[:, :, 0] = io.load('input_data/cifar_data/cifar10.pt')
    return total_data

#! cuturi


def bayesian_coeff_process(np_n_dim, scale):
    np_n_dim -= np_n_dim.mean(axis=0)
    return scale * np_n_dim


def mean_field_process(np_n_dim, scale):
    return scale * np_n_dim


def marginal_bayesian_cuturi(args):
    measures_locations = []
    for i in range(args.NUM_DISTRIBUTION):
        measures_locations.append(
            np.load(args.posterior_path + f"biketrip_subset{i}_samples.npy")[:args.N_SAMPLES])
        # _, Sigma, _ = np.linalg.svd(np.cov(measures_locations[i].T))
        measures_locations[i] = bayesian_coeff_process(
            measures_locations[i], args.SCALE)
        # _, Sigma, _ = np.linalg.svd(np.cov(measures_locations[i].T))
    return measures_locations


def marginal_mean_field_cuturi(args, **kwargs):
    measures_locations = []
    tmp_data = np.zeros([args.N_SAMPLES, args.INPUT_DIM])
    for i in range(args.NUM_DISTRIBUTION):
        model_param = io.load(args.get_nn(**kwargs) +
                              f"/subset_{i+1}_samples_{args.subset_samples}/trial_26/storing_models/nn_2layer_epoch200.pt")

        tmp_data[:, :-1] = PTU.torch2numpy(model_param['layer1.weight'])
        tmp_data[:, -
                 1] = PTU.torch2numpy(model_param['last_layer.weight'].squeeze())
        measures_locations.append(tmp_data)
        measures_locations[i] = mean_field_process(
            measures_locations[i], args.SCALE)

    return measures_locations


def marginal_mnist_cuturi(args):
    measures_locations = []

    for i in range(args.NUM_DISTRIBUTION):
        measures_locations.append(
            PTU.torch2numpy(torch.load('/home/jfan97/Study_hard/barycenter/July20/barycenter_clean/mnist_data/mnist_270.pt')[:args.N_SAMPLES, :, i + 1]))
    return measures_locations


def marginal_Gaussian_cuturi(args):
    measures_locations = []
    for i in range(args.NUM_DISTRIBUTION):
        weight_GMM = int(args.N_SAMPLES / args.NUM_GMM_COMPONENT[i])
        for j in range(args.NUM_GMM_COMPONENT[i]):
            measures_locations.append(torch.randn(
                args.N_SAMPLES, args.INPUT_DIM))
            measures_locations[i] = np_samples_generate_Gaussian(
                args.MEAN[i][j, :], args.COV[i][j], weight_GMM)
        measures_locations[i] = np.random.permutation(measures_locations[i])
    return measures_locations


def real_posterior_generator_bayesian_3loop(args):
    full_posterior = np.load(
        args.posterior_path + f"biketrip_total_samples.npy")
    full_posterior = bayesian_coeff_process(full_posterior, args.SCALE)
    return full_posterior[:args.N_TEST]


def push_back_bayesian_samples(miu, cfg):
    return miu / cfg.SCALE


def points_on_triangle(v, n):
    """
    Give n random points uniformly on a triangle.

    The vertices of the triangle are given by the shape
    (2, 3) array *v*: one vertex per row.
    """
    x = np.sort(np.random.rand(2, n), axis=0)
    return np.column_stack([x[0], x[1] - x[0], 1.0 - x[1]]) @ v


# * torch type


def torch_normal_gaussian(INPUT_DIM, **kwargs):
    N_TEST = kwargs.get('N_TEST')
    device = kwargs.get('device')
    kernel_size = kwargs.get('kernel_size')
    if N_TEST is None:
        epsilon_test = torch.randn(INPUT_DIM)
    elif kernel_size is None:
        epsilon_test = torch.randn(N_TEST, INPUT_DIM)
    else:
        epsilon_test = torch.randn(N_TEST, INPUT_DIM, kernel_size, kernel_size)
    return epsilon_test.cuda(device)


def torch_samples_generate_Gaussian(n, mean, cov, **kwargs):
    device = kwargs.get('device')
    return torch.from_numpy(
        np.random.multivariate_normal(mean, cov, n)).float().cuda(device)
# * numpy type


def repeat_list(ndarray, repeat_times):
    return [ndarray] * repeat_times


def np_samples_generate_Gaussian(mean, cov, n):
    Gaussian_sample = np.random.multivariate_normal(mean, cov, n)
    return Gaussian_sample


def np_PDF_generate_multi_normal_NN_1(pos_n_n_2, mean, cov):
    rv = multivariate_normal(mean, cov)
    multi_normal_nn_1 = rv.pdf(pos_n_n_2)
    return multi_normal_nn_1


def np_PDF_generate_multi_normal_N_N(pos_n_n_2, mean, cov):
    multi_normal_n_n = np_PDF_generate_multi_normal_NN_1(
        pos_n_n_2, mean, cov).reshape(-1, 1)[:, 0]
    return multi_normal_n_n


def np_generate_kde_NN_1(pos_nn_2, kde_analyzer):
    kde_nn_1 = kde_analyzer.score_samples(pos_nn_2)
    kde_nn_1 = np.exp(kde_nn_1)
    return kde_nn_1

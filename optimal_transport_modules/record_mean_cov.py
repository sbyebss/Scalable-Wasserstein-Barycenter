import numpy as np
import math
import sklearn

low_condition_dim = np.array([2, 16, 64, 128, 256])
high_condition_dim = np.array([16, 64, 100, 128, 2, 256])


def random_diagonal(n_dim, seed):
    BIAS = (seed - 1) * 4
    FIX_CHOICE = np.array([1, 2, 3]) + BIAS
    rng = np.random.RandomState(seed)
    digonal_elements = rng.choice(FIX_CHOICE, n_dim)
    return np.diag(digonal_elements)


def centered_mean_cov(dim, test_idx, **kwargs):
    centered_mean = [np.zeros([1, dim]),
                     np.zeros([1, dim]),
                     np.zeros([1, dim])]
    random_cov = cov_find_seed(test_idx, dim, **kwargs)
    return centered_mean, random_cov


def given_singular_spd_cov(n_dim, random_state=None, range_sing=[0.5, 5]):
    generator = sklearn.utils.check_random_state(random_state)
    A = generator.rand(n_dim, n_dim)
    U, _, V = np.linalg.svd(np.dot(A.T, A))
    X = np.dot(
        np.dot(U, np.diag(range_sing[0] + generator.rand(n_dim) * (range_sing[1] - range_sing[0]))), V)
    return X


def given_condition_spd_cov(n_dim, random_state, range_cond=10000):
    generator = sklearn.utils.check_random_state(random_state)
    A = generator.rand(n_dim, n_dim)
    U, _, V = np.linalg.svd(np.dot(A.T, A))
    if type(range_cond) is list:
        cond_n = generator.uniform(range_cond[0], range_cond[1])
    elif type(range_cond) is int:
        cond_n = range_cond
    s = generator.rand(n_dim)
    idx_max = np.argmax(s)
    s_min = np.min(s)
    s[idx_max] = s_min * cond_n
    X = np.dot(np.dot(U, np.diag(s)), V)
    return X


def cov_find_seed(trial, n_dim, **kwargs):
    seed = np.array([1, 10, 15]) + 2 * round((trial - int(trial)) * 10)
    # if trial < 26:
    #     cov = [np.expand_dims(given_condition_spd_cov(n_dim, seed[0], **kwargs), axis=0),
    #            np.expand_dims(given_condition_spd_cov(
    #                n_dim, seed[1], **kwargs), axis=0),
    #            np.expand_dims(given_condition_spd_cov(n_dim, seed[2], **kwargs), axis=0)]
    if trial >= 20:
        cov = [np.expand_dims(given_singular_spd_cov(n_dim, seed[0]), axis=0),
               np.expand_dims(given_singular_spd_cov(n_dim, seed[1]), axis=0),
               np.expand_dims(given_singular_spd_cov(n_dim, seed[2]), axis=0)]
    else:
        raise Exception('Not suitable for gaussian setup.')
    return cov


def select_mean_and_cov(test, **kwargs):
    if test == 1:
        mean = [np.array([[10 * math.cos(0 * math.pi / 5), 10 * math.sin(0 * math.pi / 5)],
                          [10 * math.cos(1 * math.pi / 5), 10 *
                           math.sin(1 * math.pi / 5)],
                          [10 * math.cos(2 * math.pi / 5), 10 *
                           math.sin(2 * math.pi / 5)],
                          [10 * math.cos(3 * math.pi / 5), 10 *
                           math.sin(3 * math.pi / 5)],
                          [10 * math.cos(4 * math.pi / 5), 10 *
                           math.sin(4 * math.pi / 5)],
                          [10 * math.cos(5 * math.pi / 5), 10 *
                           math.sin(5 * math.pi / 5)],
                          [10 * math.cos(6 * math.pi / 5), 10 *
                           math.sin(6 * math.pi / 5)],
                          [10 * math.cos(7 * math.pi / 5), 10 *
                           math.sin(7 * math.pi / 5)],
                          [10 * math.cos(8 * math.pi / 5), 10 *
                           math.sin(8 * math.pi / 5)],
                          [10 * math.cos(9 * math.pi / 5), 10 *
                           math.sin(9 * math.pi / 5)]])]
        cov = [np.array([[[0.1, 0], [0, 0.1]],
                         [[0.1, 0], [0, 0.1]],
                         [[0.1, 0], [0, 0.1]],
                         [[0.1, 0], [0, 0.1]],
                         [[0.1, 0], [0, 0.1]],
                         [[0.1, 0], [0, 0.1]],
                         [[0.1, 0], [0, 0.1]],
                         [[0.1, 0], [0, 0.1]],
                         [[0.1, 0], [0, 0.1]],
                         [[0.1, 0], [0, 0.1]]])]
    # lowD GMM[4,4] 2 marginal
    elif int(test) == 2:
        mean = np.array([[[4, 4], [4, -4], [-4, -4], [-4, 4]],
                         [[0, 4], [4, 0], [0, -4], [-4, 0]]])
        cov = [np.array([[[1, 0], [0, 1]],
                         [[1, 0], [0, 1]],
                         [[1, 0], [0, 1]],
                         [[1, 0], [0, 1]]]),
               np.array([[[1, 0], [0, 1]],
                         [[1, 0], [0, 1]],
                         [[1, 0], [0, 1]],
                         [[1, 0], [0, 1]]])]
    elif test == 3:
        mean = [np.array([[-4, 4], [4, 4], [0, -4]]),
                np.array([[0, 4], [-4, -4], [4, -4]])]
        cov = [np.array([[[2, -1], [-1, 2.5]],
                         [[4, -3], [-3, 4.5]],
                         [[3.5, -2], [-2, 3]]]),
               np.array([[[2.5, 1.2], [1.2, 2]],
                         [[2, 2.8], [2.8, 4.5]],
                         [[5, 2.5], [2.5, 3]]])]
    elif test == 4:
        mean = [np.array([[10, 10]]),
                np.array([[0, 0]]),
                np.array([[2, 0]]),
                np.array([[0, 2]])]
        cov = [np.array([[[2, 1], [1, 1]]]),
               np.array([[[1.5, -1], [-1, 1.5]]]),
               np.array([[[3, -2], [-2, 3]]]),
               np.array([[[2, 0], [0, 2]]])]
    elif test == 5:
        mean = [np.array([[4, 4],
                          [4, -4],
                          [-4, 4],
                          [-4, -4]]),
                np.array([[4, 4],
                          [-4, 4],
                          [0, -4]]),
                np.array([[4, -4],
                          [-4, -4],
                          [0, 4]])]
        cov = [np.array([[[1, 0], [0, 1]],
                         [[1, 0], [0, 1]],
                         [[1, 0], [0, 1]],
                         [[1, 0], [0, 1]]]),
               np.array([[[1, 0], [0, 1]],
                         [[1, 0], [0, 1]],
                         [[1, 0], [0, 1]]]),
               np.array([[[1, 0], [0, 1]],
                         [[1, 0], [0, 1]],
                         [[1, 0], [0, 1]]])]

    # lowD GMM[4,4] 2 marginal
    elif test == 2.1 or test == 2.12 or test == 2.13 or test == 2.14:
        mean = np.array([[[4, 4], [4, -4], [-4, -4], [-4, 4]],
                         [[0, 4], [4, 0], [0, -4], [-4, 0]]])
        cov = [np.array([[[1, 0], [0, 1]],
                         [[1, 0], [0, 1]],
                         [[1, 0], [0, 1]],
                         [[1, 0], [0, 1]]]),
               np.array([[[1, 0], [0, 1]],
                         [[1, 0], [0, 1]],
                         [[1, 0], [0, 1]],
                         [[1, 0], [0, 1]]])]
    elif test == 15:
        mean = np.array([[[0, 0]],
                         [[0, 0]],
                         [[0, 0]]])
        cov = [np.array([[[0.5, 0], [0, 2]]]),
               np.array([[[2, 1], [1, 1]]]),
               np.array([[[2, -1], [-1, 1]]])]
    elif test == 18.2 or test == 18.3 or test == 18.4:
        mean = np.array([[[0, 0]],
                         [[0, 0]]])
        cov = [np.array([[[0.0001, 0], [0, 2]]]),
               np.array([[[2, 1], [1.999, 1]]])]
    # elif test > 20 and test < 26:
    #     dim = high_condition_dim[int(test - 20)]
    #     mean, cov = centered_mean_cov(dim, test, **kwargs)
    elif test > 20 and test < 26:
        dim = low_condition_dim[int(test - 20)]
        mean, cov = centered_mean_cov(dim, test, **kwargs)

    elif test > 26 and test < 31:
        dim = low_condition_dim[int(test - 26)]
        mean, cov = centered_mean_cov(dim, test, **kwargs)

    return mean, cov

import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function
import torch.nn.functional as tnnf
import numpy as np
from scipy.stats import multivariate_normal


class GaussianMixture_2D():
    def __init__(self, mean, cov):
        self.total_matrix_of_pdf = []
        # self.original_pdf_of_each_dist = []
        self.num_distribution = len(mean)
        self.mean = mean
        self.cov = cov

    def get_num_dist(self):
        return len(self.mean)

    def get_num_GMM_component(self):
        num_GMM_component = []
        for i in range(self.num_distribution):
            num_GMM_component.append(self.mean[i].shape[0])
        return num_GMM_component

    def get_i_dist_PDF(self, pos, idx_dist):
        num_GMM_component = self.get_num_GMM_component()
        x = pos[:, :, 0]
        tmp_matrix_pdf = np.zeros_like(x)
        weights_GMM = 1 / num_GMM_component[idx_dist]
        # adder_PDF_of_each_component = np.zeros_like(x.reshape(-1, 1))
        for j in range(num_GMM_component):
            rv = multivariate_normal(
                self.mean[idx_dist][j, :], self.cov[idx_dist][j])
            # adder_PDF_of_each_component += weights_GMM * \
            #     rv.pdf(pos).reshape(-1, 1)
            tmp_matrix_pdf += weights_GMM * rv.pdf(pos) / rv.pdf(pos).sum()
        return tmp_matrix_pdf

    def get_total_matrix_of_pdf(self, pos):
        for idx_dist in range(self.num_distribution):
            self.total_matrix_of_pdf.append(self.get_i_dist_PDF(pos, idx_dist))
        return self.total_matrix_of_pdf

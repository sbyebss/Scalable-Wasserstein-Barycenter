import torch
import pdb
import torch.nn as nn
import math
from torch.autograd import Variable
from torch.autograd import Function
import numpy as np

'''''''''''''''''''''''''''''''''''''''''
            f net and g net
'''''''''''''''''''''''''''''''''''''''''

####################### Basic accessories setup ###############################


def get_activation(activation, leaky_relu_slope=0.6):
    if activation == 'relu':
        return nn.ReLU(True)
    elif activation == 'elu':
        return nn.ELU()
    elif activation == 'leaky_relu':
        return nn.LeakyReLU(leaky_relu_slope)
    elif activation == 'celu':
        return nn.CELU()
    elif activation == 'selu':
        return nn.SELU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'softsign':
        return nn.Softsign()
    elif activation == 'Prelu':
        return nn.PReLU()
    elif activation == 'Rrelu':
        return nn.RReLU(0.5, 0.8)
    elif activation == 'hardshrink':
        return nn.Hardshrink()
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'softsign':
        return nn.Softsign()
    elif activation == 'tanhshrink':
        return nn.Tanhshrink()
    else:
        raise NotImplementedError('activation [%s] is not found' % activation)


class ConvexLinear(nn.Linear):
    def __init__(self, *kargs, **kwargs):
        super(ConvexLinear, self).__init__(*kargs, **kwargs)

        if not hasattr(self.weight, 'be_positive'):
            self.weight.be_positive = 1.0

    def forward(self, input):
        out = nn.functional.linear(input, self.weight, self.bias)
        return out

#################### Self-defined Neural Network for \mu_i ########################

# ! f/g
# * PICNN


class PICNN_LastInp_Quadratic(nn.Module):
    def __init__(self, input_dim, num_distribution, hidden_u, hidden_z, activation, num_layer):
        super(PICNN_LastInp_Quadratic, self).__init__()

        # num_layer = the number excluding the last layer
        self.input_dim = input_dim
        self.num_distribution = num_distribution
        self.hidden_u = hidden_u
        self.hidden_z = hidden_z
        self.activation = activation
        self.num_layer = num_layer

        # These are weights(with bias) and matrix(without bias)
        self.Wt1_normal = nn.Linear(
            self.num_distribution, self.hidden_u, bias=True)
        self.Wy1_normal = nn.Linear(self.input_dim, self.hidden_z, bias=True)
        self.Wu1_matrix = nn.Linear(
            self.num_distribution, self.hidden_z, bias=False)

        self.activ_1 = get_activation(self.activation)
        # begin to define my own normal and convex and activation
        self.Wt_normal = nn.ModuleList([nn.Linear(
            self.hidden_u, self.hidden_u, bias=True) for i in range(2, self.num_layer + 1)])
        self.Wy_normal = nn.ModuleList([nn.Linear(
            self.input_dim, self.hidden_z, bias=True) for i in range(2, self.num_layer + 1)])
        self.Wu_matrix = nn.ModuleList([nn.Linear(
            self.hidden_u, self.hidden_z, bias=False) for i in range(2, self.num_layer + 1)])
        self.Wz_convex = nn.ModuleList([ConvexLinear(
            self.hidden_z, self.hidden_z, bias=False) for i in range(2, self.num_layer + 1)])

        self.activation = nn.ModuleList(
            [get_activation(self.activation) for i in range(2, self.num_layer + 1)])

        self.Wu_final_matrix = nn.Linear(self.hidden_u, 1, bias=False)
        self.Wz_final_convex = ConvexLinear(self.hidden_z, 1, bias=False)
        self.Wy_final_normal = nn.Linear(self.input_dim, 1, bias=True)

    def forward(self, input):
        y = input[:, :self.input_dim]
        u = input[:, -self.num_distribution:]

        z = self.activ_1(self.Wy1_normal(y) + self.Wu1_matrix(u))
        u = self.activ_1(self.Wt1_normal(u))

        for i in range(self.num_layer - 1):
            z = self.activation[i](self.Wz_convex[i](
                z) + self.Wy_normal[i](y) + self.Wu_matrix[i](u))
            u = self.activation[i](self.Wt_normal[i](u))

        z = self.Wz_final_convex(
            z) + self.Wy_final_normal(y) + self.Wu_final_matrix(u)

        return z


class PICNN_expanded(nn.Module):
    def __init__(self, input_dim, num_distribution, hidden_u, hidden_z, activation, num_layer):
        super(PICNN_expanded, self).__init__()

        # num_layer = the number excluding the last layer
        self.input_dim = input_dim
        self.num_distribution = num_distribution
        self.hidden_u = hidden_u
        self.hidden_z = hidden_z
        self.activation = activation
        self.num_layer = num_layer

        # These are weights(with bias) and matrix(without bias)
        self.Wt1_normal = nn.Linear(
            self.num_distribution, self.hidden_u, bias=True)
        self.Wy1_normal = nn.Linear(self.input_dim, self.hidden_z, bias=True)
        self.Wu1_matrix = nn.Linear(
            self.num_distribution, self.hidden_z, bias=False)

        self.activ_1 = get_activation(self.activation)
        # begin to define my own normal and convex and activation
        self.Wt_normal = nn.ModuleList([nn.Linear(
            self.hidden_u, self.hidden_u, bias=True) for i in range(2, self.num_layer + 1)])
        self.Wy_normal = nn.ModuleList([nn.Linear(
            self.input_dim, self.hidden_z, bias=True) for i in range(2, self.num_layer + 1)])
        self.Wu_matrix = nn.ModuleList([nn.Linear(
            self.hidden_u, self.hidden_z, bias=False) for i in range(2, self.num_layer + 1)])
        self.Wz_convex = nn.ModuleList([ConvexLinear(
            self.hidden_z, self.hidden_z, bias=False) for i in range(2, self.num_layer + 1)])

        # point-wise matrix
        self.Wuz_positive = nn.ModuleList([nn.Linear(
            self.hidden_u, self.hidden_z, bias=True) for i in range(2, self.num_layer + 1)])
        self.Wuy_normal = nn.ModuleList([nn.Linear(
            self.hidden_u, self.input_dim, bias=True) for i in range(2, self.num_layer + 1)])

        self.activation = nn.ModuleList(
            [get_activation(self.activation) for i in range(2, self.num_layer + 1)])

        # final layer
        self.Wu_final_matrix = nn.Linear(self.hidden_u, 1, bias=False)
        self.Wz_final_convex = ConvexLinear(self.hidden_z, 1, bias=False)
        self.Wy_final_normal = nn.Linear(self.input_dim, 1, bias=True)

        # point-wise matrix
        self.Wuz_final_positive = nn.Linear(self.hidden_u, 1, bias=True)
        self.Wuy_final_normal = nn.Linear(
            self.hidden_u, self.input_dim, bias=True)

    def forward(self, input):
        y = input[:, :self.input_dim]
        u = input[:, -self.num_distribution:]

        z = self.activ_1(self.Wy1_normal(y) + self.Wu1_matrix(u))
        u = self.activ_1(self.Wt1_normal(u))

        for i in range(self.num_layer - 1):
            z = self.activation[i](
                self.Wz_convex[i](z * torch.relu(self.Wuz_positive[i](u)))
                + self.Wy_normal[i](y * self.Wuy_normal[i](u))
                + self.Wu_matrix[i](u)
            )
            u = self.activation[i](self.Wt_normal[i](u))

        z = self.Wz_final_convex(z * torch.relu(self.Wuz_final_positive(u))) \
            + self.Wy_final_normal(y * self.Wuy_final_normal(u)) \
            + self.Wu_final_matrix(u)

        return z

# * ICNN


class ICNN_LastInp_Quadratic(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation, num_layer):
        super(ICNN_LastInp_Quadratic, self).__init__()
        # torch.set_default_dtype(torch.float64)
        # num_layer = the number excluding the last layer
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.num_layer = num_layer

        self.fc1_normal = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.activ_1 = get_activation(self.activation)

        # begin to define my own normal and convex and activation
        self.normal = nn.ModuleList([nn.Linear(
            self.input_dim, self.hidden_dim, bias=True) for i in range(2, self.num_layer + 1)])

        self.convex = nn.ModuleList([ConvexLinear(
            self.hidden_dim, self.hidden_dim, bias=False) for i in range(2, self.num_layer + 1)])

        self.activation = nn.ModuleList(
            [get_activation(self.activation) for i in range(2, self.num_layer + 1)])

        self.last_convex = ConvexLinear(self.hidden_dim, 1, bias=False)
        self.last_linear = nn.Linear(self.input_dim, 1, bias=True)

    def forward(self, input):

        x = self.activ_1(self.fc1_normal(input)).pow(2)

        for i in range(self.num_layer - 1):
            x = self.activation[i](self.convex[i](
                x).add(self.normal[i](input)))

        x = self.last_convex(x).add(self.last_linear(input).pow(2))

        return x


'''''''''''''''''''''''''''''''''''''''''
                h net
'''''''''''''''''''''''''''''''''''''''''


class BasicBlock(nn.Module):
    def __init__(self, hidden_dim, activation, leaky_relu_slope, **kwargs):
        super(BasicBlock, self).__init__()
        self.activation = activation
        self.hidden_dim = hidden_dim

        self.fc1_block = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc2_block = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.activation_block = get_activation(
            self.activation, leaky_relu_slope)

    def forward(self, input):
        x = self.activation_block(self.fc1_block(input))
        x = self.activation_block(self.fc2_block(x)) + input
        return x

# ! h different weights, serving for NWB-II


class Different_Weights_NormalNet(nn.Module):
    def __init__(self, input_dim, output_dim, num_distribution, hidden_dim, activation, num_layer, batchnormalization_flag=False, dropout_flag=False, h_full_activation=True):
        super(Different_Weights_NormalNet, self).__init__()

        # num_layer = the number excluding the last layer
        self.input_dim = input_dim + num_distribution
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.num_layer = num_layer
        self.num_distribution = num_distribution
        self.h_full_activation = h_full_activation
        self.dropout_flag = dropout_flag
        self.batchnormalization_flag = batchnormalization_flag

        self.fc1_normal = nn.Linear(self.input_dim, self.hidden_dim)
        self.activ = get_activation(self.activation)

        self.atv_list = nn.ModuleList(
            # self.atvt_list = nn.ModuleList(
            [get_activation(self.activation) for i in range(self.num_layer)])
        self.linearblock = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim) for i in range(self.num_layer)])
        if batchnormalization_flag:
            self.batchnormal = nn.ModuleList(
                [nn.BatchNorm1d(self.hidden_dim) for i in range(self.num_layer)])
        if dropout_flag:
            self.dropout = nn.ModuleList(
                [nn.Dropout(0.2) for i in range(self.num_layer)])

        self.last_normal = nn.Linear(self.hidden_dim, self.output_dim)
        self.activ_last = get_activation(self.activation)

    def forward(self, input):

        x = self.activ(self.fc1_normal(input))

        for i in range(self.num_layer):
            if self.batchnormalization_flag is False and self.dropout_flag is False:
                x = self.atvt_list[i](self.linearblock[i](x))
                # x = self.atv_list[i](self.linearblock[i](x))
            elif self.batchnormalization_flag is True and self.dropout_flag is False:
                x = self.atvt_list[i](
                    self.batchnormal[i]
                    (self.linearblock[i](x)))
            elif self.batchnormalization_flag is False and self.dropout_flag is True:
                # x = self.atvt_list[i](
                x = self.atv_list[i](
                    self.dropout[i]
                    (self.linearblock[i](x)))
            else:
                x = self.atvt_list[i](
                    self.dropout[i]
                    (self.batchnormal[i]
                     (self.linearblock[i](x))))

        if self.h_full_activation == True:
            x = self.activ_last(self.last_normal(x))
        else:
            x = self.last_normal(x)

        x = torch.cat([x, input[:, -self.num_distribution:]], axis=1)
        return x


class Different_Weights_linear(nn.Module):
    def __init__(self, input_dim, output_dim, num_distribution, hidden_dim, num_layer, batchnormalization_flag=False, dropout_flag=False):
        super(Different_Weights_linear, self).__init__()

        # num_layer = the number excluding the last layer
        self.input_dim = input_dim + num_distribution
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.num_distribution = num_distribution
        self.dropout_flag = dropout_flag
        self.batchnormalization_flag = batchnormalization_flag

        self.fc1_normal = nn.Linear(self.input_dim, self.hidden_dim)

        self.linearblock = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim) for i in range(self.num_layer)])
        if batchnormalization_flag:
            self.batchnormal = nn.ModuleList(
                [nn.BatchNorm1d(self.hidden_dim) for i in range(self.num_layer)])
        if dropout_flag:
            self.dropout = nn.ModuleList(
                [nn.Dropout(0.2) for i in range(self.num_layer)])

        self.last_normal = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, input):

        x = self.fc1_normal(input)

        for i in range(self.num_layer):
            if self.batchnormalization_flag is False and self.dropout_flag is False:
                x = (self.linearblock[i](x))
            elif self.batchnormalization_flag is True and self.dropout_flag is False:
                x = (self.batchnormal[i]
                     (self.linearblock[i](x)))
            elif self.batchnormalization_flag is False and self.dropout_flag is True:
                x = (self.dropout[i]
                     (self.linearblock[i](x)))
            else:
                x = (self.dropout[i]
                     (self.batchnormal[i]
                      (self.linearblock[i](x))))

        x = self.last_normal(x)
        x = torch.cat([x, input[:, -self.num_distribution:]], axis=1)
        return x


class Different_Weights_PICNN(nn.Module):
    def __init__(self, input_dim, num_distribution, hidden_u, hidden_z, activation, num_layer, h_full_activation):
        super(Different_Weights_PICNN, self).__init__()
        # num_layer = the number excluding the last layer
        self.input_dim = input_dim
        self.output_dim = input_dim
        self.num_distribution = num_distribution
        self.hidden_u = hidden_u
        self.hidden_z = hidden_z
        self.activation = activation
        self.num_layer = num_layer
        self.h_full_activation = h_full_activation

        # These are weights(with bias) and matrix(without bias)
        self.Wt1_normal = nn.Linear(
            self.num_distribution, self.hidden_u, bias=True)
        self.Wy1_normal = nn.Linear(self.input_dim, self.hidden_z, bias=True)
        self.Wu1_matrix = nn.Linear(
            self.num_distribution, self.hidden_z, bias=False)

        self.activ_1 = get_activation(self.activation)
        # begin to define my own normal and convex and activation
        self.Wt_normal = nn.ModuleList([nn.Linear(
            self.hidden_u, self.hidden_u, bias=True) for i in range(2, self.num_layer + 1)])
        self.Wy_normal = nn.ModuleList([nn.Linear(
            self.input_dim, self.hidden_z, bias=True) for i in range(2, self.num_layer + 1)])
        self.Wu_matrix = nn.ModuleList([nn.Linear(
            self.hidden_u, self.hidden_z, bias=False) for i in range(2, self.num_layer + 1)])
        self.Wz_convex = nn.ModuleList([nn.Linear(
            self.hidden_z, self.hidden_z, bias=False) for i in range(2, self.num_layer + 1)])

        self.activ = nn.ModuleList(
            [get_activation(self.activation) for i in range(2, self.num_layer + 1)])

        self.Wu_final_matrix = nn.Linear(
            self.hidden_u, self.output_dim, bias=False)
        self.Wz_final_convex = nn.Linear(
            self.hidden_z, self.output_dim, bias=False)
        self.Wy_final_normal = nn.Linear(
            self.input_dim, self.output_dim, bias=True)
        self.activ_last = get_activation(self.activation)

    # Input is of size
    def forward(self, input):
        y = input[:, :self.input_dim]
        u = input[:, -self.num_distribution:]

        z = self.activ_1(self.Wy1_normal(y) + self.Wu1_matrix(u))
        u = self.activ_1(self.Wt1_normal(u))

        for i in range(self.num_layer - 1):
            z = self.activ[i](self.Wz_convex[i](
                z) + self.Wy_normal[i](y) + self.Wu_matrix[i](u))
            u = self.activ[i](self.Wt_normal[i](u))
        if self.h_full_activation == True:
            z = self.activ_last(self.Wz_final_convex(
                z) + self.Wy_final_normal(y) + self.Wu_final_matrix(u))
        else:
            z = self.Wz_final_convex(
                z) + self.Wy_final_normal(y) + self.Wu_final_matrix(u)
        z = torch.cat([z, input[:, -self.num_distribution:]], axis=1)
        return z


# ! h average weights


# * convolution
class Average_Weights_Convolution(nn.Module):
    def __init__(self, input_dim, activation, num_layer=0, dropout_flag=False):
        super(Average_Weights_Convolution, self).__init__()

        # num_layer = the number excluding the last layer
        self.input_dim = input_dim
        self.activation = activation
        self.num_layer = num_layer
        self.dropout_flag = dropout_flag
        if self.num_layer == 4:
            self.main_module = nn.Sequential(
                # (input_dimx1x1)
                nn.ConvTranspose2d(in_channels=self.input_dim,
                                   out_channels=256, kernel_size=4, stride=1, padding=0),
                nn.BatchNorm2d(num_features=256),
                get_activation(self.activation),
                # State (1024x4x4)
                nn.ConvTranspose2d(in_channels=256, out_channels=128,
                                   kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(num_features=128),
                get_activation(self.activation),
                # State (512x8x8)
                nn.ConvTranspose2d(in_channels=128, out_channels=64,
                                   kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(num_features=64),
                get_activation(self.activation),
                # State (256x16x16)
                nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=2, stride=2, padding=2))
            # output of main module --> Image (Cx28x28)
        if self.num_layer == 3:
            self.main_module = nn.Sequential(
                # (input_dimx1x1)
                nn.ConvTranspose2d(in_channels=self.input_dim,
                                   out_channels=128, kernel_size=4, stride=1, padding=0),
                nn.BatchNorm2d(num_features=128),
                get_activation(self.activation),

                # State (256x4x4)
                nn.ConvTranspose2d(in_channels=128, out_channels=64,
                                   kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(num_features=64),
                get_activation(self.activation),

                # State (512x8x8)
                nn.ConvTranspose2d(in_channels=64, out_channels=1,
                                   kernel_size=2, stride=4, padding=1),
                # State (1x28x28)
                get_activation(self.activation))
        if self.num_layer == 5:
            self.main_module = nn.Sequential(
                # (input_dimx1x1)
                nn.ConvTranspose2d(in_channels=self.input_dim,
                                   out_channels=128, kernel_size=4, stride=1, padding=0),
                nn.BatchNorm2d(num_features=128),
                get_activation(self.activation),
                # State (1024x4x4)
                nn.ConvTranspose2d(in_channels=128, out_channels=64,
                                   kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(num_features=64),
                get_activation(self.activation),
                # State (512x8x8)
                nn.ConvTranspose2d(in_channels=64, out_channels=32,
                                   kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(num_features=32),
                get_activation(self.activation),
                # State (256x16x16)
                nn.ConvTranspose2d(in_channels=32, out_channels=16,
                                   kernel_size=2, stride=2, padding=2),
                nn.BatchNorm2d(num_features=16),
                get_activation(self.activation),

                # State (16x28x28)
                nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0))
            # output of main module --> Image (Cx28x28)

    def forward(self, input):
        x = self.main_module(input)
        return x


class Average_Weights_NormalNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, activation, num_layer, batchnormalization_flag=True, dropout_flag=False, h_full_activation=True, final_actv='Prelu'):
        super(Average_Weights_NormalNet, self).__init__()

        # num_layer = the number excluding the last layer
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.final_actv = final_actv
        self.num_layer = num_layer
        self.h_full_activation = h_full_activation
        self.dropout_flag = dropout_flag
        self.batchnormalization_flag = batchnormalization_flag

        self.fc1_normal = nn.Linear(self.input_dim, self.hidden_dim)
        self.activ = get_activation(self.activation)

        self.atvt_list = nn.ModuleList(
            [get_activation(self.activation) for i in range(self.num_layer)])
        self.linearblock = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim) for i in range(self.num_layer)])
        if batchnormalization_flag:
            self.batchnormal = nn.ModuleList(
                [nn.BatchNorm1d(self.hidden_dim) for i in range(self.num_layer)])
        if dropout_flag:
            self.dropout = nn.ModuleList(
                [nn.Dropout(0.2) for i in range(self.num_layer)])

        self.last_normal = nn.Linear(self.hidden_dim, self.output_dim)
        self.activ_last = get_activation(self.final_actv)

    def forward(self, input):

        x = self.activ(self.fc1_normal(input))

        for i in range(self.num_layer):
            if self.batchnormalization_flag == 0 and self.dropout_flag == 0:
                # x = self.activ(self.linearblock[i](x))
                x = self.atvt_list[i](self.linearblock[i](x))
            elif self.batchnormalization_flag == 1 and self.dropout_flag == 0:
                x = self.atvt_list[i](
                    self.batchnormal[i]
                    (self.linearblock[i](x)))
            elif self.batchnormalization_flag == 0 and self.dropout_flag == 1:
                x = self.atvt_list[i](
                    self.dropout[i]
                    (self.linearblock[i](x)))
            else:
                x = self.atvt_list[i](
                    self.dropout[i]
                    (self.batchnormal[i]
                     (self.linearblock[i](x))))

        if self.h_full_activation:
            x = self.activ_last(self.last_normal(x))
        else:
            x = self.last_normal(x)

        return x


class Fully_connected(nn.Module):
    def __init__(self, input_dim=785, output_dim=1, hidden_dim=1024, num_layer=1, activation='celu', final_actv='celu', full_activ=True, bias=False, reduction='mean'):
        super(Fully_connected, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.num_layer = num_layer
        self.full_activ = full_activ
        self.final_actv = final_actv
        self.reduction = reduction

        self.layer1 = nn.Linear(
            self.input_dim, self.hidden_dim, bias=bias)
        self.linearblock = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim, bias=bias) for i in range(self.num_layer)])
        self.last_layer = nn.Linear(
            self.hidden_dim, self.output_dim, bias=bias)

        self.layer1_activ = get_activation(self.activation)
        self.atvt_list = nn.ModuleList(
            [get_activation(self.activation) for i in range(self.num_layer)])
        self.last_layer_activ = get_activation(self.final_actv)

    def forward(self, input):

        x = self.layer1_activ(self.layer1(input))

        for i in range(self.num_layer):
            x = self.atvt_list[i](self.linearblock[i](x))

        if self.full_activ:
            if self.reduction is 'mean':
                x = self.last_layer_activ(self.last_layer(x) / self.hidden_dim)
            else:
                x = self.last_layer_activ(self.last_layer(x))
        else:
            if self.reduction is 'mean':
                x = self.last_layer(x) / self.hidden_dim
            else:
                x = self.last_layer(x)

        return x


class Average_Weights_Linear(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layer):
        super(Average_Weights_Linear, self).__init__()

        # num_layer = the number excluding the last layer
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer

        self.fc1_normal = nn.Linear(self.input_dim, self.hidden_dim)

        self.linearblock = nn.ModuleList(
            [nn.Linear(self.hidden_dim, self.hidden_dim) for i in range(self.num_layer)])

        self.last_normal = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, input):

        x = self.fc1_normal(input)

        for i in range(self.num_layer):
            x = self.linearblock[i](x)

        x = self.last_normal(x)

        return x

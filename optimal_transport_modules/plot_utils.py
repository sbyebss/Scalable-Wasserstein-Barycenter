from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torchvision.utils as vutils
from mpl_toolkits.mplot3d import Axes3D
import jacinle.io as io
from sklearn.neighbors import KernelDensity

# * handles


def sns_scatter_handle(data_np_n_2, left_place, right_place, save_path, figsize=(10, 10), opacity=1, scatter_size=1, new_fig=True):
    parameter = Axis_Params(left_place, right_place,
                            figsize, opacity=opacity, scatter_size=scatter_size, new_fig=new_fig)
    sns_scatter_alone(data_np_n_2, parameter)
    save_fig(save_path)


def plot_2dmarginal(cfg, train_data, results_save_path, left_place, right_place):
    original_dist_plot_path = []
    for i in range(cfg.NUM_DISTRIBUTION + 1):
        original_dist_plot_path.append(f'/distribution{i+1}.png')

    for i in range(cfg.NUM_DISTRIBUTION):
        sns_scatter_handle(train_data[:, :, i].detach().numpy(
        ), left_place, right_place, results_save_path + original_dist_plot_path[i])


def plt_scatter_3dhandle(data_np_n_2, left_place, right_place, save_path, figsize=(10, 10), opacity=1, scatter_size=10):
    parameter = Axis_Params_3d(
        left_place=left_place, right_place=right_place, figsize=figsize, opacity=opacity, scatter_size=scatter_size, x_rotate=20, z_rotate=45)
    plt_scatter_3d_alone(data_np_n_2, parameter)
    save_fig(save_path)


def plot_3dmarginal(train_data, results_save_path, left_place, right_place):
    plt_total_data = train_data.detach().numpy()
    plt_data = np.concatenate(
        [plt_total_data[:, :, 0], plt_total_data[:, :, 1]], axis=0)
    plt_scatter_3dhandle(plt_data, left_place, right_place,
                         results_save_path + '/2blocks.png')


def plot_rgb_cloud_alone(cloud, save_path, num_point=1024):
    index = np.random.choice(cloud.shape[0], num_point)
    ind_order = np.array([2, 1, 0])
    selected_cloud = cloud[index][:, ind_order]
    parameter = Axis_Params_3d(left_place=0, right_place=1,
                               colors=selected_cloud
                               #    , xlabel='Red', ylabel='Green', zlabel='Blue'
                               )
    plt_scatter_3d_alone(selected_cloud, parameter)
    save_fig(save_path)

# * set the grid and index


def grid_NN_2_generator(num_grid, left_place, right_place):
    x, y = xyIndex_generator(num_grid, left_place, right_place)
    x_plot = x.reshape(-1, 1)[:, 0]
    y_plot = y.reshape(-1, 1)[:, 0]
    pos_plot = np.stack((x_plot, y_plot)).T
    return pos_plot


def grid_N_N_2_generator(num_grid, left_place, right_place):
    x, y = xyIndex_generator(num_grid, left_place, right_place)
    pos = np.dstack((x, y))
    return pos


def xyIndex_generator(num_grid, left_place, right_place):
    grid_size = (right_place - left_place) / num_grid
    x, y = np.mgrid[left_place:right_place:grid_size,
                    left_place: right_place: grid_size]
    return x, y

# * scatter


def plt_scatter_3d_alone(sample_n_3, ax_params):
    fig = plt.figure(figsize=ax_params.figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(sample_n_3[:, 0], sample_n_3[:, 1],
               sample_n_3[:, 2], alpha=ax_params.opacity, s=ax_params.scatter_size, c=ax_params.colors)
    set_matplotlib_axis(ax, ax_params)
    ax.view_init(elev=ax_params.x_rotate, azim=ax_params.z_rotate)
    # you will need this line to change the Z-axis
    ax.autoscale(enable=False, axis='both')
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_xbound(ax_params.left_place, ax_params.right_place)
    ax.set_ybound(ax_params.left_place, ax_params.right_place)
    ax.set_zbound(ax_params.left_place, ax_params.right_place)
    ax.set_xlabel(ax_params.xlabel)
    ax.set_ylabel(ax_params.ylabel)
    ax.set_zlabel(ax_params.zlabel)
    return ax


def sns_scatter_alone(sample_nn_2, ax_params):
    if ax_params.new_fig == True:
        plt.figure(figsize=ax_params.figsize)
    ax = sns.scatterplot(
        sample_nn_2[:, 0], sample_nn_2[:, 1],
        alpha=ax_params.opacity,
        s=ax_params.scatter_size,
        legend=False)
    ax.axis("off")
    ax.set_xlim(ax_params.left_place, ax_params.right_place)
    ax.set_ylim(ax_params.left_place, ax_params.right_place)
    return ax


def plt_scatter_alone(sample_nn_2, ax_params):
    if ax_params.new_fig == True:
        plt.figure(figsize=ax_params.figsize)
    plt.scatter(
        sample_nn_2[:, 0], sample_nn_2[:, 1],
        edgecolors='black', color='gold')
    plt.axis("off")
    plt.xlim(ax_params.left_place, ax_params.right_place)
    plt.ylim(ax_params.left_place, ax_params.right_place)

# * contour


class dim2_plot:
    def __init__(self, num_grid=100, left_place=-10, right_place=10, bandwidth=0.9, label_size=15):
        self.ax_param = Axis_Params(
            left_place, right_place, label_font_size=label_size, bandwidth_kde=bandwidth, num_grid=num_grid)
        self.x = None
        self.y = None
        self.barycenter_density = None

    def contour_from_sample(self, sample_n_2, save_path):
        self.x, self.y, self.barycenter_density = sample2density(
            sample_n_2, self.ax_param)
        contour_alone(self.x, self.y, self.barycenter_density,
                      save_path, self.ax_param)

    def scatter(self, sample_n_2, save_path):
        plt_scatter_alone(sample_n_2, self.ax_param)
        save_fig(save_path)


def sample2density(sample_n_2, ax_params):
    grid_size = (ax_params.right_place -
                 ax_params.left_place) / ax_params.num_grid
    x, y = np.mgrid[ax_params.left_place:ax_params.right_place:grid_size,
                    ax_params.left_place: ax_params.right_place: grid_size]
    x_plot = x.reshape(-1, 1)[:, 0]
    y_plot = y.reshape(-1, 1)[:, 0]
    pos_plot = np.stack((x_plot, y_plot)).T

    kde = KernelDensity(
        kernel='gaussian', bandwidth=ax_params.bandwidth).fit(sample_n_2)
    brct_KDE_log = kde.score_samples(pos_plot)
    brct_KDE = np.exp(brct_KDE_log)
    brct_KDE_plot = brct_KDE.reshape(
        ax_params.num_grid, ax_params.num_grid)
    return x, y, brct_KDE_plot


def contour_alone(x, y, z, save_path, ax_params):
    plt.figure(figsize=ax_params.figsize)
    plt.contour(x, y, z)
    plt.xlim(ax_params.left_place, ax_params.right_place)
    plt.ylim(ax_params.left_place, ax_params.right_place)
    plt.tick_params(axis='both', which='major',
                    labelsize=ax_params.label_font_size)
    plt.savefig(save_path)
    ax = plt.gca()
    return ax

# * mnist


def mnist_alone(sample, save_path, gan=False, range_sample=None):
    if sample.shape[0] == 784:
        sample = sample.reshape(1, 1, 28, 28)
    else:
        side_length = int(np.sqrt(sample.shape[1]))
        sample = sample.reshape(sample.shape[0], 1, side_length, side_length)
    if gan is False:
        vutils.save_image(sample.cpu().data, save_path, normalize=True, range=(
            -0.4242, 2.8215), scale_each=True, nrow=int(np.sqrt(sample.shape[0])))
    else:
        # * the default range here would be [-1,1], or we hope it around this range
        sample[sample < -0.95] = -1
        sample[sample > 0.95] = 1
        vutils.save_image(sample.cpu().data, save_path, normalize=True,
                          range=range_sample,
                          scale_each=True, nrow=int(np.sqrt(sample.shape[0])))


def mnist_one_number(sample, save_path):
    sample = sample.reshape(1, 1, 28, 28)
    vutils.save_image(sample.cpu().data, save_path,
                      normalize=True, scale_each=True)


def cifar_alone(sample, save_path, gan=False, range_sample=None):
    sample = sample.reshape(sample.shape[0], 32, 32, 3).permute(0, 3, 1, 2)
    vutils.save_image(sample.cpu().data, save_path, normalize=True,
                      range=range_sample,
                      scale_each=True, nrow=sample.shape[0])


def sns_kdeplot_alone(sample_nn_2, ax_params, bandwidth):
    plt.figure(figsize=ax_params.figsize)
    ax = sns.kdeplot(sample_nn_2[:, 0],
                     sample_nn_2[:, 1], shade=True, bw=bandwidth)
    return ax


def sns_jointplot_alone(sample_x, sample_y, ax_params, save_path):
    plt.figure()
    jg = sns.jointplot(sample_x,
                       sample_y, kind='kde', joint_kws={'shade_lowest': False})
    jg.ax_joint.set_xlim(ax_params.left_place, ax_params.right_place)
    jg.ax_joint.set_ylim(ax_params.left_place, ax_params.right_place)
    plt.savefig(save_path)
    plt.close()


def error_bar(result_n_expr_n_repeat, label, x_axis=np.array([2, 16, 64, 128, 256]), line_width=6):
    mean = result_n_expr_n_repeat.mean(axis=1)
    std = result_n_expr_n_repeat.std(axis=1)
    plt.errorbar(x_axis, mean, std,
                 label=label, elinewidth=line_width)
#---------------#


def set_sns_axis(ax, ax_params):
    ax.set_xlim(ax_params.left_place, ax_params.right_place)
    ax.set_ylim(ax_params.left_place, ax_params.right_place)
    ax.collections[0].set_alpha(0)
    ax.title.set_text(ax_params.title)
    return ax


def set_matplotlib_axis(ax, ax_params):
    ax.set_xlim(ax_params.left_place, ax_params.right_place)
    ax.set_ylim(ax_params.left_place, ax_params.right_place)
    ax.tick_params(axis='both', which='major',
                   labelsize=ax_params.axis_font_size)
    ax.set_title(ax_params.title, fontsize=ax_params.title_font_size)
    return ax

#---------------#

#! save


def save_fig(path, tight_flag=True):
    if tight_flag is True:
        plt.savefig(path, bbox_inches='tight')
    else:
        plt.savefig(path)
    plt.close()

#! class of axis


class Axis_Params():
    def __init__(self, left_place, right_place, figsize=(10, 10), title='', label_font_size=15, axis_font_size=15, title_font_size=22, opacity=1, scatter_size=20, new_fig=True, bandwidth_kde=0.9, num_grid=100):
        self.left_place = left_place
        self.right_place = right_place
        self.title = title
        self.label_font_size = label_font_size
        self.axis_font_size = axis_font_size
        self.title_font_size = title_font_size
        self.figsize = figsize
        self.opacity = opacity
        self.scatter_size = scatter_size
        self.new_fig = new_fig
        self.bandwidth = bandwidth_kde
        self.num_grid = num_grid


class Axis_Params_3d(Axis_Params):
    def __init__(self, x_rotate=None, z_rotate=None, colors=None, xlabel=None, ylabel=None, zlabel=None, *kargs, **kwargs):
        super(Axis_Params_3d, self).__init__(*kargs, **kwargs)
        self.x_rotate = x_rotate
        self.z_rotate = z_rotate
        self.colors = colors
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.zlabel = zlabel

#! application: lines


def draw_marginal_lines(path):
    plt.figure(figsize=(10, 10))
    X = io.load_txt(path)

    for i in range(int(X.shape[0] / 2)):
        x_values = X[2 * i:2 * (i + 1), 0]
        y_values = X[2 * i:2 * (i + 1), 1]

        plt.plot(x_values, y_values, 'k')
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.axis('off')

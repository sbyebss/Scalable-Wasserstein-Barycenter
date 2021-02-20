from __future__ import print_function
import GPUtil
import sys
import os
sys.path.append(
    "/home/jfan97/Study_hard/barycenter/July20/barycenter_clean")

import optimal_transport_modules.pytorch_utils as PTU
import optimal_transport_modules.generate_data as g_data
import optimal_transport_modules.generate_NN as g_NN
from optimal_transport_modules.cfg import CfgMnist as Cfg_class
import optimal_transport_modules.icnn_modules as NN_modules
import optimal_transport_modules.plot_utils as PLU
import optimal_transport_modules.compare_dist_results as CDR
import pytorch_fid.fid_score as fid_score
import jacinle.io as io

cfg = Cfg_class()

gpus_choice = GPUtil.getFirstAvailable(
    order='random', maxLoad=0.5, maxMemory=0.5, attempts=1, interval=900, verbose=False)
PTU.set_gpu_mode(True, gpus_choice[0])
# PTU.set_gpu_mode(True, 5)

#! GAN
cfg.N_TEST = 64
cfg.LR_g = cfg.LR_f
results_save_path = cfg.get_save_path()

mnist_sample = CDR.barycenter_sampler(
    cfg, PTU.device, results_save_path=results_save_path, load_epoch=cfg.load_epoch)
PLU.mnist_alone(mnist_sample, results_save_path +
                f'/{cfg.load_epoch}_{cfg.repeat}.png', gan=True, range_sample=(-1, 1))

# cfg.N_TEST = 10000
# mnist_sample = CDR.barycenter_sampler(
#     cfg, PTU.device, results_save_path=results_save_path, load_epoch=cfg.load_epoch)
# path_separate_image = results_save_path + "/fid_images"
# os.makedirs(
#     path_separate_image, exist_ok=True)
# for idx in range(cfg.N_TEST):
#     PLU.mnist_alone(mnist_sample[idx], path_separate_image +
#                     f'/{cfg.load_epoch}_{idx}.png', gan=True, range_sample=(-1, 1))

# score = [fid_score.calculate_fid_given_paths(
#         [path_separate_image, 'input_data/inception_score/mnist_official'], 50, 'cuda:7', 2048)]
# io.dump(results_save_path + '/score.txt', score)

#! separate the 0-1
cfg.N_TEST = 25
cfg.LR_g = cfg.LR_f
# results_save_path = cfg.get_save_path2()
results_save_path = 'data/Results_of_MNIST/distribution_2/10bryc/h_input_dim_16/fg_FICNN/h_nonResNet/h_batchnml:Yes_dropout:Yes/convolution_No/layers_fg_4_h_3/neuron_1024/h_activ_Prelu/lr_g0.0001lr_f_0.0001lr_h0.001/schedule_learning_rate:Yes/lr_schedule:100/gIterate_6_fIterate_4/batch_100/train_sample_1500/trial_15_last_hFull_activation'
tmp_epoch = 154
cfg.INPUT_DIM = 16
cfg.NUM_LAYERS_h = 3
cfg.NUM_LAYERS = 4
cfg.dropout = 1
for idx in range(10):
    mnist_sample = CDR.barycenter_sampler(
        cfg, PTU.device, results_save_path=results_save_path, load_epoch=tmp_epoch)
    # PLU.mnist_alone(mnist_sample, results_save_path +
    #                 f'/{tmp_epoch}_{idx}_other.png')
    tmp_epoch2 = 500
    results_save_path2 = 'data/Results_of_MNIST/distribution_2/10bryc/h_input_dim_16/fg_FICNN/h_nonResNet/h_batchnml:Yes_dropout:Yes/convolution_No/layers_fg_4_h_3/neuron_fg1024_h1024/h_activ_Prelu/lr_g0.0001lr_f_0.0001lr_h0.001/schedule_learning_rate:Yes/lr_schedule:100/gIterate_6_fIterate_4/batch_100/train_sample_1500/sign_0/trial_15.2_last_hFull_activation'
    marginal_list = CDR.barycenter_backward(
        mnist_sample, cfg, PTU.device, results_save_path=results_save_path2, load_epoch=tmp_epoch2)

    for idx_f in range(cfg.NUM_DISTRIBUTION):
        PLU.mnist_alone(marginal_list[idx_f], results_save_path2 +
                        f'/from_h_{tmp_epoch2}_f{idx_f}_other_repeat{idx}.png')
# miu_list = CDR.barycenter_pushforward(
#     cfg, PTU.device, results_save_path=results_save_path, load_epoch=tmp_epoch, type_data='mnist0-1')

# for idx in range(cfg.NUM_DISTRIBUTION):
#     PLU.mnist_alone(miu_list[idx], results_save_path +
#                     f'/{tmp_epoch}_g{idx}.png')

# for idx_g in [0, 1]:
#     marginal_list = CDR.barycenter_backward(
#         miu_list[idx_g], cfg, PTU.device, results_save_path=results_save_path, load_epoch=tmp_epoch)

#     for idx in range(cfg.NUM_DISTRIBUTION):
#         PLU.mnist_alone(marginal_list[idx], results_save_path +
#                         f'/from_g{idx_g}_{tmp_epoch}_f{idx}.png')

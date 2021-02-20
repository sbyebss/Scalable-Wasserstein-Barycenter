from __future__ import print_function
import GPUtil
import sys
import os
sys.path.append(
    "/home/jfan97/Study_hard/barycenter/July20/barycenter_clean")

import optimal_transport_modules.pytorch_utils as PTU
import optimal_transport_modules.generate_data as g_data
import optimal_transport_modules.generate_NN as g_NN
from optimal_transport_modules.cfg import CfgUSPS as Cfg_class
import optimal_transport_modules.icnn_modules as NN_modules
import optimal_transport_modules.plot_utils as PLU
import optimal_transport_modules.compare_dist_results as CDR
import pytorch_fid.fid_score as fid_score
import jacinle.io as io

cfg = Cfg_class()

gpus_choice = GPUtil.getFirstAvailable(
    order='random', maxLoad=0.5, maxMemory=0.5, attempts=5, interval=900, verbose=False)
PTU.set_gpu_mode(True, gpus_choice[0])
cfg.NUM_NEURON = cfg.NUM_NEURON_h
#! h
# cfg.N_TEST = 64
# cfg.LR_g = cfg.LR_f
results_save_path = cfg.get_save_path()

usps_flag = abs(cfg.TRIAL - 1.0) > 1e-3
mnist_sample = CDR.barycenter_sampler(
    cfg, PTU.device, results_save_path=results_save_path,
    load_epoch=cfg.load_epoch, usps=usps_flag
)
print(mnist_sample.max(), mnist_sample.min())
PLU.mnist_alone(mnist_sample, results_save_path +
                f'/{cfg.load_epoch}_{cfg.repeat}.png', gan=True)

#! g

miu_list, marginal_list = CDR.barycenter_pushforward(
    cfg, PTU.device, results_save_path=results_save_path, load_epoch=cfg.load_epoch, type_data='usps_mnist', return_marginal=True)

for idx in range(cfg.NUM_DISTRIBUTION):
    PLU.mnist_alone(miu_list[idx], results_save_path +
                    f'/{cfg.load_epoch}_g{idx}_{cfg.repeat}.png', gan=True)
    # PLU.mnist_alone(marginal_list[idx], results_save_path +
    #                 f'/{cfg.load_epoch}_m{idx}_{cfg.repeat}.png', gan=True, range_sample=(-1, 1))

#! f backward
marginal_list = CDR.barycenter_backward(
    mnist_sample, cfg, PTU.device, results_save_path=results_save_path, load_epoch=cfg.load_epoch)

for idx_f in range(cfg.NUM_DISTRIBUTION):
    PLU.mnist_alone(marginal_list[idx_f], results_save_path +
                    f'/from_h_{cfg.load_epoch}_f{idx_f}_repeat{cfg.repeat}.png', gan=True)

import sys
sys.path.append(
    "/home/jfan97/Study_hard/barycenter/July20/barycenter_clean")
import numpy as np
import GPUtil

from optimal_transport_modules.cfg import CfgGMM as Cfg_class
import optimal_transport_modules.compare_dist_results as CDR
import optimal_transport_modules.pytorch_utils as PTU
import optimal_transport_modules.plot_utils as PLU
import optimal_transport_modules.data_utils as DTU
import jacinle.io as io

cfg = Cfg_class()

gpus_choice = GPUtil.getFirstAvailable(
    order='random', maxLoad=0.5, maxMemory=0.5, attempts=1, interval=900, verbose=False)
PTU.set_gpu_mode(True, gpus_choice[0])

#! For the error:
# tmp_epoch = 25
# low_condition_result = np.zeros([3, 5])  # [5 experiments,5 repeat]
# idx_low_c = 0
# # for trial in [26, 27, 28, 29, 30]:
# for trial in [28, 29, 24]:
#     idx_low_c += 1
#     for seed in range(5):
#         cfg.TRIAL = trial + (seed + 1) * 0.1
#         cfg.NUM_LAYERS_h = 3
#         cfg.MEAN, cfg.COV, cfg.INPUT_DIM, cfg.OUTPUT_DIM, cfg.NUM_DISTRIBUTION, cfg.NUM_GMM_COMPONENT, cfg.high_dim_flag = DTU.get_gmm_param(
#             cfg.TRIAL)

#         if cfg.INPUT_DIM == 16:
#             cfg.NUM_NEURON_h = 32
#         elif cfg.INPUT_DIM == 64:
#             cfg.NUM_NEURON_h = 100
#         elif cfg.INPUT_DIM == 128:
#             cfg.NUM_NEURON_h = 160
#         elif cfg.INPUT_DIM == 256:
#             cfg.NUM_NEURON_h = 280
#         elif cfg.INPUT_DIM == 2:
#             cfg.NUM_NEURON_h = 10
#         cfg.NUM_NEURON = cfg.INPUT_DIM * 2

#         # cfg.NUM_NEURON_h = cfg.INPUT_DIM * 2
#         # cfg.NUM_NEURON = cfg.NUM_NEURON_h

#         miu = CDR.barycenter_sampler(
#             cfg, PTU.device, load_epoch=tmp_epoch)

#         low_condition_result[idx_low_c - 1,
#                              seed] = CDR.gaussian_compare_package(miu, cfg)
#         print(low_condition_result)

# io.dump('error_bar_exp/NWB1_h_dim.npy', low_condition_result)

#! For the plot
cfg.MEAN, cfg.COV, cfg.INPUT_DIM, cfg.OUTPUT_DIM, cfg.NUM_DISTRIBUTION, cfg.NUM_GMM_COMPONENT, cfg.high_dim_flag = DTU.get_gmm_param(
    cfg.TRIAL)

results_save_path = cfg.get_save_path()
# results_save_path = '/home/jfan97/Study_hard/barycenter/July20/barycenter_clean/data/Results_of_Gauss2Gauss/low_dimension/h_Linear:No_Normal/distribution_2/GMM_component_[4,4]/input_dim_2/fg_FICNN/init_trunc_inv_sqrt/layers_fg_3_h_2/neuron_10/lambda_cvx_0.1_mean_0.0/optim_Adamlr_g0.001betas_0.5_0.99lr_f_0.001lr_h0.001/gIterate_6_fIterate_4/batch_200/trial_2.5_last_inp_qudr'
plt_handle = PLU.dim2_plot()


# miu_list = CDR.barycenter_pushforward(
#     cfg, PTU.device, results_save_path=results_save_path, load_epoch=cfg.load_epoch, type_data='gmm')
# for idx in range(cfg.NUM_DISTRIBUTION):
#     plt_handle.contour_from_sample(
#         miu_list[idx].cpu().detach().numpy(),
#         results_save_path + f'/{cfg.load_epoch}_g{idx}.png')
#     plt_handle.scatter(miu_list[idx].cpu().detach().numpy(
#     ), results_save_path + f'/{cfg.load_epoch}_g{idx}_scatter.png')

miu = CDR.barycenter_sampler(
    cfg, PTU.device, results_save_path=results_save_path, load_epoch=cfg.load_epoch)
plt_handle.contour_from_sample(
    miu.cpu().detach().numpy(),
    results_save_path + f'/{cfg.load_epoch}_h.png')
plt_handle.scatter(miu.cpu().detach().numpy(),
                   results_save_path + f'/{cfg.load_epoch}_h_scatter.png')

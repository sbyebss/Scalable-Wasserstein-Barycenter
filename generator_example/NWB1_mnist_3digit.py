from __future__ import print_function
import sys
sys.path.append(
    "/home/jfan97/Study_hard/barycenter/July20/barycenter_clean")
import GPUtil

import optimal_transport_modules.pytorch_utils as PTU
from optimal_transport_modules.cfg import Cfg3digit as Cfg_class
import optimal_transport_modules.plot_utils as PLU
import optimal_transport_modules.compare_dist_results as CDR

cfg = Cfg_class()


cfg.N_TEST = 100000

gpus_choice = GPUtil.getFirstAvailable(
    order='random', maxLoad=0.5, maxMemory=0.5, attempts=1, interval=900, verbose=False)
PTU.set_gpu_mode(True, gpus_choice[0])

#! normal
cfg.LR_g = cfg.LR_f
results_save_path = cfg.get_save_path()
# tmp_epoch = 77
miu = CDR.barycenter_sampler(
    cfg, PTU.device, results_save_path=results_save_path, load_epoch=cfg.load_epoch)
PLU.sns_scatter_handle(
    miu.cpu().detach().numpy(), -6, 6,
    results_save_path + f'/{cfg.load_epoch}_{cfg.N_TEST}_{cfg.opacity}_{cfg.scatter_size}.png', opacity=cfg.opacity, scatter_size=cfg.scatter_size)

miu_list = CDR.barycenter_pushforward(
    cfg, PTU.device, results_save_path=results_save_path, load_epoch=cfg.load_epoch, type_data='3digit')

for idx in range(cfg.NUM_DISTRIBUTION):
    PLU.sns_scatter_handle(
        miu_list[idx].cpu().detach().numpy(), -6, 6,
        results_save_path + f'/{cfg.load_epoch}_g{idx}_{cfg.N_TEST}_{cfg.opacity}_{cfg.scatter_size}.png', opacity=cfg.opacity, scatter_size=cfg.scatter_size)

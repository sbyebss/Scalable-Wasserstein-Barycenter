import sys
sys.path.append(
    "/home/jfan97/Study_hard/barycenter/July20/barycenter_clean")
import GPUtil

from optimal_transport_modules.cfg import CfgEllipse as Cfg_class
import optimal_transport_modules.pytorch_utils as PTU
import optimal_transport_modules.plot_utils as PLU
import optimal_transport_modules.compare_dist_results as CDR

if __name__ == "__main__":
    cfg = Cfg_class()

    cfg.N_TEST = 200
    gpus_choice = GPUtil.getFirstAvailable(
        order='random', maxLoad=0.5, maxMemory=0.5, attempts=1, interval=900, verbose=False)
    PTU.set_gpu_mode(True, gpus_choice[0])

    results_save_path = cfg.get_save_path()
    tmp_epoch = 23
    # miu = CDR.barycenter_sampler(
    #     cfg, PTU.device, results_save_path=results_save_path, load_epoch=tmp_epoch)
    # PLU.sns_scatter_handle(
    #     miu.cpu().detach().numpy(), -12, 12,
    #     results_save_path + f'/{tmp_epoch}_{cfg.N_TEST}_{cfg.opacity}_{cfg.scatter_size}.png', scatter_size=cfg.scatter_size)

    miu_list = CDR.barycenter_pushforward(
        cfg, PTU.device, results_save_path=results_save_path, load_epoch=tmp_epoch, type_data='ellipse')

    for idx in range(cfg.NUM_DISTRIBUTION):
        PLU.sns_scatter_handle(
            miu_list[idx].cpu().detach().numpy(), -12, 12,
            results_save_path + f'/{tmp_epoch}_g{idx}_{cfg.N_TEST}_{cfg.opacity}_{cfg.scatter_size}.png', scatter_size=cfg.scatter_size)

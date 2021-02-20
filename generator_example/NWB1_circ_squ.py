import GPUtil
import sys
sys.path.append(
    "/home/jfan97/Study_hard/barycenter/July20/barycenter_clean")

from optimal_transport_modules.cfg import CfgCircSqu as Cfg_class
import optimal_transport_modules.pytorch_utils as PTU
import optimal_transport_modules.plot_utils as PLU
import optimal_transport_modules.compare_dist_results as CDR

if __name__ == "__main__":
    cfg = Cfg_class()

    cfg.N_TEST = 100000
    gpus_choice = GPUtil.getFirstAvailable(
        order='random', maxLoad=0.5, maxMemory=0.5, attempts=1, interval=900, verbose=False)
    PTU.set_gpu_mode(True, gpus_choice[0])
    # PTU.set_gpu_mode(False)

    results_save_path = f'/home/jfan97/Study_hard/barycenter/July20/barycenter_clean/data/Results_of_circ_squ/distribution_2/input_dim_2/fg_FICNN/h_nonResNet/h_batchnml:Yes_dropout:No/layers_fg_{cfg.NUM_LAYERS}_h_{cfg.NUM_LAYERS_h}/neuron_fg{cfg.NUM_NEURON}_h{cfg.NUM_NEURON_h}/h_activ_Prelu/lr_g0.001lr_f_0.001lr_h0.001/schedule_learning_rate:Yes/lr_schedule:20/gIterate_6_fIterate_4/batch_100/train_sample_60000/sign_0/trial_8.61_last_hFull_activation'

    # tmp_epoch = 51

    miu = CDR.barycenter_sampler(
        cfg, PTU.device, results_save_path=results_save_path, load_epoch=cfg.load_epoch)
    PLU.sns_scatter_handle(
        miu.cpu().detach().numpy(), -6, 6,
        results_save_path + f'/{cfg.load_epoch}_{cfg.N_TEST}_{cfg.opacity}_{cfg.scatter_size}.png', opacity=cfg.opacity, scatter_size=cfg.scatter_size)

    miu_list = CDR.barycenter_pushforward(
        cfg, PTU.device, results_save_path=results_save_path, load_epoch=cfg.load_epoch, type_data='circ_squa')

    for idx in range(cfg.NUM_DISTRIBUTION):
        PLU.sns_scatter_handle(
            miu_list[idx].cpu().detach().numpy(), -6, 6,
            results_save_path + f'/{cfg.load_epoch}_g{idx}_{cfg.N_TEST}_{cfg.opacity}_{cfg.scatter_size}.png', opacity=cfg.opacity, scatter_size=cfg.scatter_size)

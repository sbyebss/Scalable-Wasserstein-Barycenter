import GPUtil
import sys
sys.path.append(
    "/home/jfan97/Study_hard/barycenter/July20/barycenter_clean")
import torch

from optimal_transport_modules.cfg import CfgColor as Cfg_class
import optimal_transport_modules.pytorch_utils as PTU
import optimal_transport_modules.generate_data as g_data
import optimal_transport_modules.generate_NN as g_NN
import optimal_transport_modules.plot_utils as PLU
import optimal_transport_modules.compare_dist_results as CDR
import jacinle.image.imgio as imgio

if __name__ == "__main__":
    cfg = Cfg_class()

    cfg.N_TEST = 1024
    cfg.LR_g = cfg.LR_f

    # cfg.NUM_NEURON_h = 16
    # cfg.NUM_NEURON = 16
    # cfg.NUM_LAYERS = 3
    # cfg.NUM_LAYERS_h = 3
    # cfg.final_actv = 'sigmoid'

    gpus_choice = GPUtil.getFirstAvailable(
        order='random', maxLoad=0.5, maxMemory=0.5, attempts=1, interval=900, verbose=False)
    PTU.set_gpu_mode(True, gpus_choice[0])
    results_save_path = cfg.get_save_path()

    # load_epoch = 80
    miu = CDR.barycenter_sampler(cfg, PTU.device, load_epoch=cfg.load_epoch)

    miu = torch.clamp(miu, 0, 1)
    PLU.plot_rgb_cloud_alone(
        miu.cpu().detach().numpy(), results_save_path + f'/{cfg.load_epoch}_h_cloud_{cfg.N_TEST}_repeat{cfg.repeat}.png')

    marginal = g_data.get_marginal_color_list(cfg).float()
    _, generator_g_list = g_NN.generate_FixedWeight_fg_NN(cfg)
    for idx in range(3):
        tmp_margin = marginal[idx]
        generator_gi = g_NN.load_generator_fg(
            results_save_path, idx + 1, generator_g_list[idx], cfg.load_epoch)

        tmp_margin = tmp_margin.cuda(PTU.device)
        tmp_margin = torch.autograd.Variable(tmp_margin, requires_grad=True)
        generator_gi.cuda(PTU.device)

        g_of_miu_i = generator_gi(tmp_margin).sum()
        miu = torch.autograd.grad(
            g_of_miu_i, tmp_margin)[0].cpu().clamp(0, 1).detach().numpy()

        PLU.plot_rgb_cloud_alone(
            miu, results_save_path + f'/{cfg.load_epoch}_g{idx+1}_cloud_{cfg.N_TEST}_repeat{cfg.repeat}.png')

        # im = miu.reshape(1080, 1920, 3) * 255
        # imgio.imwrite(results_save_path +
        #               f'/picture{idx}_baryc.png', im)

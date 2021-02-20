import sys
sys.path.append(
    "/home/jfan97/Study_hard/barycenter/July20/barycenter_clean")
import GPUtil

from optimal_transport_modules.cfg import CfgBayesian as Cfg_class
import optimal_transport_modules.compare_dist_results as CDR
import optimal_transport_modules.pytorch_utils as PTU
import optimal_transport_modules.generate_data as g_data


if __name__ == "__main__":
    cfg = Cfg_class()

    gpus_choice = GPUtil.getFirstAvailable(
        order='random', maxLoad=0.5, maxMemory=0.5, attempts=1, interval=900, verbose=False)
    PTU.set_gpu_mode(True, gpus_choice[0])
    repeat = 100
    mean_record = 0
    for idx in range(repeat):
        miu = CDR.barycenter_sampler(
            cfg, PTU.device, load_epoch=cfg.load_epoch).cpu()
        miu_full_posterior = g_data.real_posterior_generator_bayesian_3loop(
            cfg)

        # miu, miu_full_posterior = g_data.push_back_bayesian_samples(
        #     miu, cfg), g_data.push_back_bayesian_samples(miu_full_posterior, cfg)
        # print("NUM_LAYERS_h")
        # print(cfg.NUM_LAYERS_h)
        # print('scale=')
        # print(cfg.SCALE)
        mean_record += CDR.bayesian_compare_package(miu, miu_full_posterior)
print("epoch=")
print(cfg.load_epoch)
print(mean_record / repeat)

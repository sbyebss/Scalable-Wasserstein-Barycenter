import numpy as np
import matplotlib.pyplot as plt
import jacinle.io as io
import sys
sys.path.append(
    "/home/jfan97/Study_hard/barycenter/July20/barycenter_clean")
import optimal_transport_modules.plot_utils as PLU
import optimal_transport_modules.data_utils as DTU
import subprocess

# subprocess.run(["scp", "jfan97@fjj:/home/jfan97/Study_hard/Wasserstein_Barycenter/CWB/experiments/gaussian/low_cond_total1e-4.npy", "error_bar_exp/crwb_small_cond_total1e-4.npy"
#                 ])

#! load nwb
# nwb_large = io.load('error_bar_exp/NWB1_large_cond.npy')
nwb_small = io.load('error_bar_exp/NWB1_small_cond.npy')
nwb_small[-3:, :] = io.load('error_bar_exp/NWB1_h_dim.npy')

#! load crwb
# crwb_large = io.load('error_bar_exp/crwb_large_cond.npy')
crwb_small_1e4 = io.load('error_bar_exp/crwb_small_cond_total1e-4.npy')

#! load cdwb
# fsb_large = io.load('error_bar_exp/FSB_large_cond.npy')
fsb_small = io.load('error_bar_exp/FSB_small_cond.npy')

#! load cwb
# cwb_small = io.load('error_bar_exp/cwb_small_cond.npy')
cwb_small = np.zeros([5, 5])
idx_low_c = 0
for trial in [26, 27, 28, 29, 30]:
    idx_low_c += 1 * (trial >= 26)
    for seed in range(5):
        TRIAL = trial + (seed + 1) * 0.1
        _, _, INPUT_DIM, _, _, _, _ = DTU.get_gmm_param(
            TRIAL)
        cwb_small[idx_low_c - 1, seed] = io.load(
            f'../../continuous2_barycenter/results/gaussian/pretrain/lr_0.001/cond_-1/dim{INPUT_DIM}/trial{TRIAL}/BW2_UVP.npy')
        print(cwb_small)

fig = plt.figure(figsize=(10, 10))
PLU.error_bar(nwb_small, 'NWB')
PLU.error_bar(cwb_small, 'CWB')
PLU.error_bar(fsb_small, 'CDWB')
PLU.error_bar(crwb_small_1e4, r'CRWB with $\epsilon=10^{-4}$',
              x_axis=np.array([2, 16, 32, 64, 72, 100, 128, 256]))

plt.tick_params(axis='both', which='major', labelsize=25)
plt.xlabel('Dimension', fontsize=25)
plt.ylabel('BW2-UVP', fontsize=25)
plt.legend(prop={"size": 25}, loc='upper left')
plt.savefig('./error_bar_exp/error_bar_small_cond.png', bbox_inches='tight')
plt.close()

fig = plt.figure(figsize=(10, 10))
PLU.error_bar(nwb_small, 'NWB')
PLU.error_bar(cwb_small, 'CWB')

plt.tick_params(axis='both', which='major', labelsize=25)
plt.xlabel('Dimension', fontsize=25)
plt.legend(prop={"size": 25}, loc='upper left')
plt.savefig('./error_bar_exp/error_bar_inner.png', bbox_inches='tight')
plt.close()

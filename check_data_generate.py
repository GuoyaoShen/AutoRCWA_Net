import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.data_utils import generate_data, generate_dataset, load_data

path_data = './data/fRT_1.npz'

# ========== data generation
# param_w, R, T = generate_data(num_data=500, w_range=np.array([0.3,0.6]), path_all_data=path_data, w_decimal=3, import_list=True ,use_log=True)
param_w, R, T = generate_data(num_data=20, w_range=np.array([0.3,0.6]), path_all_data=path_data, w_decimal=2, import_list=True ,use_log=True)
print(param_w.shape)
print(R.shape)
print(T.shape)


# ========== dataset & dataloader generation
# dataset, dataloader = generate_dataset(path_data, idx_pick_param=[], BTSZ=1)


# ========== visualize data
# param_w, spectra_R, spectra_T = load_data(path_data)
# print(param_w.shape)
# print(spectra_R.shape)
# print(spectra_T.shape)
#
# fig = plt.figure(1, figsize=(21,12))
# fig.suptitle('R', fontsize="x-large")
# for idx_fig in range(9):
#     plt.subplot(3, 3, idx_fig+1)
#     plt.scatter(np.arange(spectra_R.shape[1]), spectra_R[idx_fig], c='b')
#     plt.title('sample idx: '+str(idx_fig)+' || w ='+str(param_w[idx_fig]/1e-3)+'mm')
#
#
# fig = plt.figure(2, figsize=(21,12))
# fig.suptitle('T', fontsize="x-large")
# for idx_fig in range(9):
#     plt.subplot(3, 3, idx_fig+1)
#     plt.scatter(np.arange(spectra_T.shape[1]), spectra_T[idx_fig], c='b')
#     plt.title('sample idx: '+str(idx_fig)+' || w ='+str(param_w[idx_fig]/1e-3)+'mm')
#
# plt.show()
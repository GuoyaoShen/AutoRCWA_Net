import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils.data_utils as UdataU
from utils.data_utils import generate_data, generate_dataset, load_data, generate_data_absorber

# path_data = './data/fRT_1.npz'

# ========== data generation
# # param_w, R, T = generate_data(num_data=500, w_range=np.array([0.3,0.6]), path_all_data=path_data, w_decimal=3, import_list=True ,use_log=True)
# # param_w, R, T = generate_data(num_data=20, w_range=np.array([0.3,0.6]), path_all_data=path_data, w_decimal=2, import_list=True ,use_log=True)
#
# # params range
# params_range = [[50,150], [50,150]]
# params_decimal = [0,0]
# path_material_name = 'absorber'
#
# # solver setting: [freq_step, freq_truncate, params_mesh, PQ_order, source, device]
# '''
# solver setting should not being changed while sampling and training!
# '''
# params_mesh = [512, 512]
# order = 9  # RCWA accuracy, higher to be more accurate
# PQ_order = [order, order]
# ginc = [0, 0, 1]  # orig [0,0,1], incident source
# EP = [1, 0, 0]  # orig [0,1,0]
# source = [ginc, EP]
# device = 'gpu'
# freq_step = 4  # freq step size, bigger to save more time, while less sampling freq points
# freq_truncate = 1.7  # 'none' to no truncate
# solver_setting = [freq_step, freq_truncate, params_mesh, PQ_order, source, device]
#
# # call RCWA solver
# params_list, R, T = generate_data_absorber(10, params_range, params_decimal, solver_setting_list=solver_setting,
#                                            path_material_name=path_material_name, import_list=False, use_log=True)
# print(params_list.shape)
# print(R.shape)
# print(T.shape)


# ========== dataset & dataloader generation
# dataset, dataloader = generate_dataset(path_data, idx_pick_param=[], BTSZ=1)
path_material_name = 'absorber'
path_all_data = './data/' + path_material_name + '/all_data_' + path_material_name + '.npz'
dataset, dataloader = UdataU.generate_dataset_absorber(path_all_data, idx_pick_param=[], BTSZ=1)
print(len(dataset))


# ========== visualize data
path_material_name = 'absorber'
path_all_data = './data/' + path_material_name + '/all_data_' + path_material_name + '.npz'
params_list, spectra_R, spectra_T = load_data(path_all_data)
print(params_list.shape)
print(spectra_R.shape)
print(spectra_T.shape)

fig = plt.figure(1, figsize=(21,12))
fig.suptitle('R', fontsize="x-large")
for idx_fig in range(9):
    plt.subplot(3, 3, idx_fig+1)
    plt.scatter(np.arange(spectra_R.shape[1]), spectra_R[idx_fig], c='b')
    plt.plot(np.arange(spectra_R.shape[1]), spectra_R[idx_fig], c='r')
    plt.title('sample idx: '+str(idx_fig)+' || params_list ='+str(params_list[idx_fig])+'mm')


fig = plt.figure(2, figsize=(21,12))
fig.suptitle('T', fontsize="x-large")
for idx_fig in range(9):
    plt.subplot(3, 3, idx_fig+1)
    plt.scatter(np.arange(spectra_T.shape[1]), spectra_T[idx_fig], c='b')
    plt.plot(np.arange(spectra_T.shape[1]), spectra_T[idx_fig], c='r')
    plt.title('sample idx: '+str(idx_fig)+' || params_list ='+str(params_list[idx_fig])+'mm')

plt.show()
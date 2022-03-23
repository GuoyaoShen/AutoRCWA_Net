import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

import re
import os

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

from Simple_RCWA.utils import data_utils
from Simple_RCWA.utils import calc_utils
from Simple_RCWA.utils import rcwa_utils

pi = np.pi

# ====== Unit Define ======
meters = 1
centimeters = 1e-2 * meters
millimeters = 1e-3 * meters
micrometres = 1e-6 * meters

# ====== Constant Define ======
c0 = 3e8
e0 = 8.85e-12
u0 = 1.256e-6
yeta0 = np.sqrt(u0/e0)



def generate_data_absorber(num_data,
                           params_range,
                           solver_setting_list,
                           params_list,
                           use_log=False,
                           flag_spectra_search_rerun=False,
                           rerun_params=[]):
    '''
    Generate data for ellipse hole absorber using Simple_RCWA package.

    params_range: [[range1 for D1], [range2 for D2]].
                  A list of the params range from the Material class's parameter list_layer_params.
                  Each entry is a np arange: np.arange(range_start, range_end, range_step).
    solver_setting_list: RCWA solver setting.
                         The elements inside are inherented from the Material class.
                         [freq, params_eps, params_geometry, params_mesh, PQ_order, list_layer_funcs, source, device].
                         freq: numpy array of frequencies to solve rcwa, (N_freq,).
                         params_eps: np array of eps for all layers, each entry in the list is shape of (N_freq,).
                         params_geometry: [Lx,Ly,[d1,...,dn]], 2D geometry params and thickness for all layers.
                         params_mesh: [Nx,Ny], mesh number for 2D geometry.
                         PQ_order: a list of [PQ_x, PQ_y], each entry should be a singular value.
                         list_layer_funcs: a list of functions [f1,...,fn] applied to each layer to define patterns
                                           inside each layer, deleting materials.
                         source: source of incident light, a list, [ginc, EP], each entry (ginc, EP) is also a list,
                                 both ginc and EP should be a unit vector.
                         device: 'cpu' for CPU using numpy; 'gpu' or 'cuda' for GPU using cupy.
    params_list: A np array containing all the sampled params in all past rounds, shape [N_sample, N_params].
                 When the first time to start, an empty array of shape [0, N_params] should be passed in.
    use_log: flag to show calculation log info.

    ------ Params below are for spectra search rerun on picked params specifically ------
    flag_spectra_search_rerun: if True, this will run RCWA solver on picked params by spectra search.
    rerun_params: a numpy array containing picked params by spectra search.
    '''

    # ====== Material property definition ======
    # solver_setting_list: [freq, eps, params_geometry, params_mesh, PQ_order, list_layer_funcs, source, device]
    freq = solver_setting_list[0]
    params_eps = solver_setting_list[1]


    # ====== Material structure definition ======
    # solver_setting_list: [freq, eps, params_geometry, params_mesh, PQ_order, list_layer_funcs, source, device]
    params_geometry = solver_setting_list[2]
    params_mesh = solver_setting_list[3]
    PQ_order = solver_setting_list[4]
    list_layer_funcs = solver_setting_list[5]
    source = solver_setting_list[6]
    device = solver_setting_list[7]


    # initialize R and T
    R = np.array([]).reshape((0, freq.shape[0]))
    T = np.array([]).reshape((0, freq.shape[0]))

    if flag_spectra_search_rerun:  # spectra search rerun
        '''
        Run RCWA on spectra search picked params.
        param_list will not be updated in this case.
        '''
        D1 = rerun_params[:, 0] * micrometres
        D2 = rerun_params[:, 1] * micrometres

        for idx_simu in range(rerun_params.shape[0]):
            print('[', (idx_simu+1), '/', rerun_params.shape[0], ']')
            list_layer_params = [[D1[idx_simu], D2[idx_simu]]]

            Si_square_hole = rcwa_utils.Material(freq, params_eps, params_geometry, params_mesh, PQ_order,
                                                 list_layer_funcs, list_layer_params, source, device, use_log)
            R_total, T_total = Si_square_hole.rcwa_solve()

            R_total = R_total[np.newaxis, ...]
            T_total = T_total[np.newaxis, ...]
            R = np.concatenate((R, R_total), axis=0)
            T = np.concatenate((T, T_total), axis=0)

        return rerun_params, R, T

    else:  # normally generate rcwa simulation data
        '''
        Run RCWA to generate the training data.
        param_list will be updated in this case.
        '''
        # ================= Generate Data using RCWA
        N_param = num_data
        num_param = 0

        N_possible = []
        for idx_params, ele_params_range in enumerate(params_range):
            N_possible_i = len(ele_params_range)
            N_possible.append(N_possible_i)
        # if import_list:
        #     N_needed += params_list.shape[0]
        N_needed = num_data + params_list.shape[0]
        print('N_possible:', N_possible, ',', np.prod(N_possible), 'in total')
        print('N_needed:', N_needed)
        if N_needed > np.prod(N_possible):
            raise ValueError('Too many sample points! Make sure: num_data + num_list < (params_range[1]-params_range[0]) * 10**params_decimal, for all params')
        else:
            print('Sample points number available, continue calculating...')

        params_list_sampled = np.array([]).reshape((0, len(params_range)))  # initiate sampled params list
        while num_param < N_param:  # solving loop
            # ====== Sampling in the parameter space ======
            D1 = np.random.choice(params_range[0])
            D2 = np.random.choice(params_range[1])
            params = np.array([D1,D2])

            # params 'in-list-check', update param_list (for 'same-param-check')
            if np.any(np.all(params_list - params == 0, axis=-1)):  # if params already in the list, continue
                continue
            else:
                params_list = np.concatenate((params_list, params[np.newaxis, ...]), axis=0)
            # if get passed, new params is a new combination

            # update sampled params list
            params_list_sampled = np.concatenate((params_list_sampled, params[np.newaxis, ...]), axis=0)
            num_param += 1

            # ====== RCWA solver ======
            # params being changed
            D1 = D1 * micrometres  # two axes of the ellipse hole
            D2 = D2 * micrometres
            list_layer_params = [[D1, D2]]

            if use_log:
                print('----------------')
                print('[', (num_param), '/', N_param, '] [D1, D2] =', params)

            # call RCWA solver
            Si_square_hole = rcwa_utils.Material(freq, params_eps, params_geometry, params_mesh, PQ_order,
                                                 list_layer_funcs, list_layer_params, source, device, use_log)
            R_total, T_total = Si_square_hole.rcwa_solve()
            R_total = R_total[np.newaxis, ...]
            T_total = T_total[np.newaxis, ...]
            R = np.concatenate((R, R_total), axis=0)
            T = np.concatenate((T, T_total), axis=0)

        return params_list_sampled, R, T, params_list


def generate_dataset(PATH_ZIPSET, idx_pick_param=[], BTSZ=10):
    '''
    Generate torch dataset and dataloader from zipped numpy dataset.
    :param PATH_ZIPSET: path for zipped numpy dataset
    :param idx_pick_param: list of idx of selected design params, default as empty list
    :param BTSZ: batch size, default as 10
    :return: dataset, dataloader: torch dataset and dataloader
    '''

    data = np.load(PATH_ZIPSET)
    params = data['params']
    # params_list = params_list[..., np.newaxis]
    spectra_R = data['R'] #[N,N_freq]
    spectra_T = data['T']

    if idx_pick_param:  # select param
        params = params[..., idx_pick_param]

    # concat reflection and transmission spectras as one
    spectra_R = np.expand_dims(spectra_R, 1)  #[N,1,N_freq]
    spectra_T = np.expand_dims(spectra_T, 1)
    spectra_RT = np.concatenate((spectra_R, spectra_T), axis=1)  #[N,2,N_freq]
    # print(spectra_RT.shape)

    tensor_x = torch.Tensor(params)  # transform to torch tensor
    tensor_y = torch.Tensor(spectra_RT)

    # generate torch dataset
    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset, batch_size=BTSZ, shuffle=True)

    return dataset, dataloader


def generate_pseudo_params(params_range):
    '''
    Generate pseudo params.
    Preparation before generating pseudo dataset. This is also a preparation for spectra search.

    params_range: [[range1 for D1], [range2 for D2]], a list, each entry is a list consist:
                  [range_start, range_end, step_size].
    '''

    for idx_params, params in enumerate(itertools.product(*[range(*ele_range) for ele_range in params_range])):
        params = np.array(params)[np.newaxis, ...]
        if idx_params == 0:
            pseudo_params = params
        else:
            pseudo_params = np.concatenate((pseudo_params, params), axis=0)
    return pseudo_params


def generate_pseudo_data(pseudo_params, net, device, PATH_pseudo_dataset='', flag_save_pseudo_data=False):
    '''
    Generate pseudo dataset with the trained network.
    Preparation for spectra search.

    pseudo_params: numpy array.
    net: trained network.
    device: torch available device.
    PATH_pseudo_dataset: path to save pseudo dataset.
    flag_save_pseudo_data: 'True' to save pseudo data.
    '''

    # params as torch tensor
    X = torch.tensor(pseudo_params).float().to(device)
    net = net.to(device)
    net.eval()

    # input to model and get spectra
    y_pred = net(X)
    y_pred_np = y_pred.cpu().detach().numpy()
    spectra_R = y_pred_np[:, 0, :]
    spectra_T = y_pred_np[:, 1, :]
    if flag_save_pseudo_data:
        np.savez(PATH_pseudo_dataset, params=pseudo_params, R=spectra_R, T=spectra_T)
        print('Pseudo data saved')

    return pseudo_params, spectra_R, spectra_T


def spectra_search(pseudo_data, target_data, order=2, N_top=10):
    '''
    Perform spectra search on pseudo data with L_{order} norm.

    pseudo_data: a list, [pseudo_params, spectra_R, spectra_T], each entry is a numpy array.
    target_data: a list, [tg_idx_freq_R, tg_value_R, tg_idx_freq_T, tg_value_T], each entry is a numpy array. If NO
                 target spectra on R or T, pass in an empty list: [] or an empty numpy array: np.array([])
    N_top: top N best match spectra.
    '''

    pseudo_params = pseudo_data[0]  # # [N_pseudo, N_params]
    spectra_R = pseudo_data[1]  # [N_pseudo, N_freq]
    spectra_T = pseudo_data[2]
    spectra_pseudo = np.concatenate((spectra_R, spectra_T), axis=-1)  # [N_pseudo, 2*N_freq]

    tg_idx_freq_R = target_data[0]  # [N_tg,]
    tg_value_R = target_data[1]
    tg_idx_freq_T = target_data[2]
    tg_value_T = target_data[3]
    if tg_idx_freq_R.size != 0 and tg_idx_freq_T.size == 0:  # only search Reflection
        print('# search R')
        tg_idx_freq = tg_idx_freq_R
        tg_value = tg_value_R

    elif tg_idx_freq_R.size == 0 and tg_idx_freq_T.size != 0:  # only search Transmission
        print('# search T')
        tg_idx_freq = tg_idx_freq_T + spectra_R.shape[-1]
        tg_value = tg_value_T

    elif tg_idx_freq_R.size != 0 and tg_idx_freq_T.size != 0:  # search both spectra
        print('# search both R and T')
        tg_idx_freq_T = tg_idx_freq_T + spectra_R.shape[-1]
        tg_idx_freq = np.concatenate((tg_idx_freq_R, tg_idx_freq_T))
        tg_value = np.concatenate((tg_value_R, tg_value_T))

    else:
        print('[Warning] Nothing is being spectra searched!')
        return np.array([]), np.array([]), np.array([]), np.array([])

    # print('#tg_idx_freq.shape:', tg_idx_freq.shape)
    # print('#tg_idx_freq:', tg_idx_freq)
    # print('#tg_value.shape:', tg_value.shape)
    # print('#tg_value:', tg_value)

    spectra_pseudo = spectra_pseudo[:, tg_idx_freq]  # [N_pseudo, N_tg]
    dist = np.linalg.norm(spectra_pseudo-tg_value, ord=order, axis=1)  # distance calculation, spectra search, [N_pseudo,]

    idx_sorted = np.argsort(dist)
    idx_pick = idx_sorted[0:N_top]  # pick top N best match

    dist_pick = dist[idx_pick, ...]  # distance
    param_pick = pseudo_params[idx_pick, ...]  # picked param
    R_pick = spectra_R[idx_pick, ...]
    T_pick = spectra_T[idx_pick, ...]
    return param_pick, R_pick, T_pick, dist_pick


def load_data(PATH_ZIPSET):
    data = np.load(PATH_ZIPSET)
    params = data['params']
    spectra_R = data['R']  # [N,N_freq]
    spectra_T = data['T']

    return params, spectra_R, spectra_T
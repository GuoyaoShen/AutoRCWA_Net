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

# ================= Unit Define
meters = 1
centimeters = 1e-2 * meters
millimeters = 1e-3 * meters
micrometres = 1e-6 * meters

# ================= Constant Define
c0 = 3e8
e0 = 8.85e-12
u0 = 1.256e-6
yeta0 = np.sqrt(u0/e0)


def generate_data(num_data, w_range, path_all_data='./data/fRT.npz', w_decimal=3, import_list=False ,use_log=False):
    '''
    Generate data using Simple_RCWA package.
    '''

    pi = np.pi

    # ================= Unit Define
    meters = 1
    centimeters = 1e-2 * meters
    millimeters = 1e-3 * meters

    # ================= Constant Define
    c0 = 3e8
    e0 = 8.85e-12
    u0 = 1.256e-6
    yeta0 = np.sqrt(u0 / e0)

    path_gold = './Simple_RCWA/Au3-Drude.txt'
    eps_gold_file = data_utils.load_property_txt(path_gold, 2743, 3636)
    eps_gold = eps_gold_file[:, 1] + eps_gold_file[:, 2] * 1j

    path_SiNx = './Simple_RCWA/SiNx_property.mat'
    eps_SiNx = data_utils.load_property_mat(path_SiNx)
    eps_SiNx = eps_SiNx['eps_SiNx_real'] + eps_SiNx['eps_SiNx_imag'] * 1j

    freq = eps_gold_file[:, 0] * 1e12

    Ly = 0.005 * millimeters  # period along y


    # ================= Generate Data using RCWA
    N_w = num_data
    num_w = 0
    # import w_weight list
    path_weight = './data/w_list.npz'
    if import_list:
        w_weight_list = np.load(path_weight)
        w_weight_list = w_weight_list['w_weight_list']
    else:
        w_weight_list = []
    param_w = np.array([]).reshape(0)
    R = np.array([]).reshape((0,893))
    T = np.array([]).reshape((0,893))

    # Param Sampling Space Check
    N_needed = num_data
    N_possible = (w_range[1] - w_range[0]) * 10 ** w_decimal
    if import_list:
        N_needed += w_weight_list.shape[0]
    print('N_possible:', N_possible)
    print('N_needed:', N_needed)
    if N_needed > N_possible:
        raise ValueError('Too many sample points! Make sure: num_data + num_list < (w_range[1]-w_range[0]) * 10**w_decimal')
    else:
        print('Sample points number available, continue calculating...')

    while num_w < N_w:  # solving loop
        # ================= Sampling in Parameter Space
        w_weight = np.random.uniform(w_range[0], w_range[1])
        w_weight = np.around(w_weight, w_decimal)
        if np.any(np.isin(w_weight_list, w_weight)):
            continue
        else:  # not in list, available w
            w_weight_list = np.append(w_weight_list, w_weight)
            num_w += 1

        # ================= RCWA Solver
        w = w_weight * Ly

        if use_log:
            print('[', (num_w), '/', N_w, '] w_weight =', w_weight)
        R_total, T_total = rcwa_utils.rcwa_solver(freq, eps_gold, eps_SiNx, w=w, use_logger=use_log)

        w = w.reshape(1)
        param_w = np.concatenate((param_w, w))
        R_total = R_total[np.newaxis,...]
        T_total = T_total[np.newaxis, ...]
        R = np.concatenate((R, R_total), axis=0)
        T = np.concatenate((T, T_total), axis=0)

        # ================= Spectra Plot
        # plt.figure(1)
        # plt.plot(freq, R_total)
        # plt.figure(2)
        # plt.plot(freq, T_total)
        # plt.show()
        path = './data/detail_data/fRT_w' + str(w_weight) + '.npz'
        np.savez(path, freq=freq, R=R_total, T=T_total)
        if use_log:
            # print('\n')
            print('\nFILE SAVED, w_weight =', w_weight)
            # print(w.shape)
            # print(R_total.shape)
            # print(T_total.shape)
            # print(param_w.shape)
            # print(R.shape)
            # print(T.shape)
            print('----------------')

    # save w_weight list
    np.savez(path_weight, w_weight_list=w_weight_list)
    # save all data
    path_data = path_all_data
    np.savez(path_data, param_w=param_w, R=R, T=T)
    print('All data saved.')

    return param_w, R, T


# def generate_dataset(PATH_ZIPSET, idx_pick_param=[], BTSZ=10):
#     '''
#     Generate torch dataset and dataloader from zipped numpy dataset.
#     :param PATH_ZIPSET: path for zipped numpy dataset
#     :param idx_pick_param: list of idx of selected design params, default as empty list
#     :param BTSZ: batch size, default as 10
#     :return: dataset, dataloader: torch dataset and dataloader
#     '''
#
#     data = np.load(PATH_ZIPSET)
#     param_w = data['param_w']
#     param_w = param_w[..., np.newaxis]
#     spectra_R = data['R'] #[N,893]
#     spectra_T = data['T']
#     N_data = spectra_R.shape[0]
#
#     # [Lx,Ly,d1,d2,d3]
#     Lx = 0.005 * millimeters  # period along x
#     Ly = 0.005 * millimeters  # period along y
#     d1 = 0.00015 * millimeters  # thickness of layer 1
#     d2 = 0.0005 * millimeters  # thickness of layer 2
#     d3 = 0.00015 * millimeters  # thickness of layer 3
#     param_other = np.array([Lx,Ly,d1,d2,d3])
#     param_other = np.tile(param_other, (N_data,1))
#     param = np.concatenate((param_other, param_w), axis=-1)
#     # print(param.shape)
#
#
#     if idx_pick_param:  # select param
#         param = param[..., idx_pick_param]
#
#     # concat reflection and transmission spectras as one
#     spectra_R = np.expand_dims(spectra_R, 1)  #[N,1,893]
#     spectra_T = np.expand_dims(spectra_T, 1)
#     spectra_RT = np.concatenate((spectra_R, spectra_T), axis=1)  #[N,2,893]
#     # print(spectra_RT.shape)
#
#     tensor_x = torch.Tensor(param)  # transform to torch tensor
#     tensor_y = torch.Tensor(spectra_RT)
#
#     # generate torch dataset
#     dataset = TensorDataset(tensor_x, tensor_y)
#     dataloader = DataLoader(dataset, batch_size=BTSZ, shuffle=True)
#
#     return dataset, dataloader


def generate_data_absorber(num_data, params_range, params_decimal, solver_setting_list, path_material_name,
                           import_list=False, use_log=False, flag_spectra_search_rerun=False, rerun_params=[]):
    '''
    Generate data for absorber using Simple_RCWA package.

    params_range: [[range1 for D1], [range2 for D2]], a list, each entry is a list consist: [range_start, range_end].
    params_decimal: [decimal for D1, decimal for D2], a list, each one is the desired decimal num, recommend for both to
                    be the same (i.e., same step size).
    solver_setting_list: RCWA solver setting, [params_mesh, PQ_order, source, device].
    path_material_name: name for the material, this will be used to automatically generate folders for corresponding
                        data.
    flag_spectra_search_rerun: if True, this will run rcwa solver on picked params by spectra search.
    rerun_params: a numpy array containing picked params by spectra search.
    '''

    # ================= Material Property Define
    freq_step = solver_setting_list[0]
    freq_truncate = solver_setting_list[1]
    # path_absorber = './Simple_RCWA/material_property/permittivity_absorber.txt'
    path_absorber = './Simple_RCWA/material_property/' + path_material_name + '.txt'
    eps_absorber_file = data_utils.load_property_txt(path_absorber)
    if freq_truncate != 'none' and freq_truncate > eps_absorber_file[0, 0] and freq_truncate < eps_absorber_file[-1, 0]:
        N_freq_stop = np.argmax(eps_absorber_file[:, 0] > freq_truncate)
        eps_absorber_file = eps_absorber_file[:N_freq_stop]
        print('Freq truncate.')
    else:
        print('No freq truncate.')
    eps_absorber_file = eps_absorber_file[::freq_step]  # solve rcwa with a step size
    eps_absorber = eps_absorber_file[:, 1] + eps_absorber_file[:, 2] * 1j

    freq = eps_absorber_file[:, 0] * 1e12
    print('freq.shape:', freq.shape)

    # ================= Material Structure Define
    a = 160.
    t = 75.

    Lx = a * micrometres  # period along x
    Ly = a * micrometres  # period along y
    d1 = t * micrometres  # thickness of layer 1

    # D1 = 130 * micrometres  # two axes of the ellipse hole
    # D2 = 150 * micrometres

    params_eps = [eps_absorber]
    params_geometry = [Lx, Ly, [d1]]
    # params_mesh = [512, 512]
    # order = 9
    # PQ_order = [order, order]
    list_layer_funcs = [rcwa_utils.layerfunc_absorber_ellipse_hole]
    # list_layer_params = [[D1, D2]]
    # ginc = [0, 0, 1]  # orig [0,0,1], incident source
    # EP = [1, 0, 0]  # orig [0,1,0]
    # source = [ginc, EP]
    # device = 'gpu'

    # [freq_step, freq_truncate, params_mesh, PQ_order, source, device]
    params_mesh = solver_setting_list[2]
    PQ_order = solver_setting_list[3]
    source = solver_setting_list[4]
    device = solver_setting_list[5]

    # initialize R and T
    R = np.array([]).reshape((0, freq.shape[0]))
    T = np.array([]).reshape((0, freq.shape[0]))

    if flag_spectra_search_rerun:  # spectra search rerun
        D1 = rerun_params[:, 0] * micrometres
        D2 = rerun_params[:, 1] * micrometres

        # # [freq_step, freq_truncate, params_mesh, PQ_order, source, device]
        # params_mesh = solver_setting_list[2]
        # PQ_order = solver_setting_list[3]
        # source = solver_setting_list[4]
        # device = solver_setting_list[5]

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
        # ================= Generate Data using RCWA
        N_param = num_data
        num_param = 0
        # import params list
        # path_params_list = './data/params_list_absorber.npz'
        path_params_list = './data/' + path_material_name + '/params_list_' + path_material_name + '.npz'
        if not os.path.exists('./data/' + path_material_name + '/'):
            os.makedirs('./data/' + path_material_name + '/')
        if import_list:
            params_list = np.load(path_params_list)
            params_list = params_list['params_list']
        else:
            params_list = []
        # R = np.array([]).reshape((0, freq.shape[0]))
        # T = np.array([]).reshape((0, freq.shape[0]))

        # Param Sampling Space Check
        N_needed = num_data
        N_possible = []
        for idx_params, ele_params_range in enumerate(params_range):
            N_possible_i = (params_range[idx_params][1] - params_range[idx_params][0]) * 10 ** params_decimal[idx_params]
            N_possible.append(N_possible_i)
        if import_list:
            N_needed += params_list.shape[0]
        print('N_possible:', N_possible, ',', np.prod(N_possible), 'in total')
        print('N_needed:', N_needed)
        if N_needed > np.prod(N_possible):
            raise ValueError('Too many sample points! Make sure: num_data + num_list < (params_range[1]-params_range[0]) * 10**params_decimal, for all params')
        else:
            print('Sample points number available, continue calculating...')

        while num_param < N_param:  # solving loop
            # ================= Sampling in Parameter Space
            D1 = np.random.uniform(params_range[0][0], params_range[0][1])
            D1 = np.around(D1, params_decimal[0])
            D2 = np.random.uniform(params_range[1][0], params_range[1][1])
            D2 = np.around(D2, params_decimal[1])
            params = np.array([D1,D2])

            # params 'in-list-check', update param_list (for 'same-param-check')
            if params_list==[]:
                params_list = params[np.newaxis, ...]
            else:
                # if np.any(np.all(params_list-params==np.array([0,0]), axis=-1)):  # if params already in the list, continue
                if np.any(np.all(params_list-params==0, axis=-1)):  # if params already in the list, continue
                    continue
                else:
                    params_list = np.concatenate((params_list, params[np.newaxis, ...]), axis=0)
            # if get passed, new params is a new combination

            # update current params
            if num_param == 0:
                params_list_current = params[np.newaxis, ...]
            else:
                params_list_current = np.concatenate((params_list_current, params[np.newaxis, ...]), axis=0)

            num_param += 1

            # ================= RCWA Solver
            D1_temp = D1
            D2_temp = D2
            # params being changed
            D1 = D1 * micrometres  # two axes of the ellipse hole
            D2 = D2 * micrometres
            list_layer_params = [[D1, D2]]

            if use_log:
                print('----------------')
                print('[', (num_param), '/', N_param, '] [D1, D2] =', params)

            # #[freq_step, freq_truncate, params_mesh, PQ_order, source, device]
            # params_mesh = solver_setting_list[2]
            # PQ_order = solver_setting_list[3]
            # source = solver_setting_list[4]
            # device = solver_setting_list[5]
            Si_square_hole = rcwa_utils.Material(freq, params_eps, params_geometry, params_mesh, PQ_order,
                                                 list_layer_funcs, list_layer_params, source, device, use_log)
            R_total, T_total = Si_square_hole.rcwa_solve()

            R_total = R_total[np.newaxis, ...]
            T_total = T_total[np.newaxis, ...]
            R = np.concatenate((R, R_total), axis=0)
            T = np.concatenate((T, T_total), axis=0)

            # ================= Save Detail Data
            # path = './data/absorber/detail_data/fRT_D1_' + str(D1_temp) + '_D2_' + str(D2_temp) + '.npz'
            # path = './data/' + path_material_name + '/detail_data/fRT_D1_' + str(D1_temp) + '_D2_' + str(D2_temp) + '.npz'
            # np.savez(path, freq=freq, R=R_total, T=T_total)  # save detail data
            # if use_log:
                # print('\nFILE SAVED, [D1,D2] =', params)
                # print('\n----------------')

        # save params list
        np.savez(path_params_list, params_list=params_list)
        # save all data
        # path_all_data = './data/' + path_material_name + '/all_data_' + path_material_name + '.npz'
        # np.savez(path_all_data, params_list=params_list, R=R, T=T)
        # print('All data saved.')

        param_other = np.array([a, t])
        param_other = np.tile(param_other, (params_list_current.shape[0], 1))
        params_list_all = np.concatenate((params_list_current, param_other), axis=-1)  # [params list, param other]

        return params_list_all, R, T


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
    Generate pseudo data with trained network.

    pseudo_params: numpy array.
    net: trained network.
    device: torch available device.
    PATH_pseudo_dataset: path to save pseudo dataset.
    flag_save_pseudo_data: 'True' to save pseudo data.
    '''

    # params as torch tensor
    X = torch.tensor(pseudo_params).float().to(device)
    net = net.to(device)

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
                 target spectra on R or T, pass in an empty list: [].
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
        tg_idx_freq = tg_idx_freq_R
        tg_value = tg_value_R

    elif tg_idx_freq_R.size == 0 and tg_idx_freq_T.size != 0:  # only search Transmission
        tg_idx_freq = tg_idx_freq_T + spectra_R.shape[-1]
        tg_value = tg_value_T

    elif tg_idx_freq_R.size != 0 and tg_idx_freq_T.size != 0:  # search both spectra
        tg_idx_freq = np.concatenate((tg_idx_freq_R, tg_idx_freq_T))
        tg_value = np.concatenate((tg_value_R, tg_value_T))

    else:
        print('[Warning] Nothing is being spectra searched!')
        return np.array([]), np.array([]), np.array([]), np.array([])

    spectra_pseudo = spectra_pseudo[:, tg_idx_freq]  # [N_pseudo, N_tg]
    dist = np.linalg.norm(spectra_pseudo-tg_value, ord=order, axis=1)  # distance calculation, spectra search

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
    # param_w = param_w[..., np.newaxis]
    spectra_R = data['R']  # [N,N_freq]
    spectra_T = data['T']

    return params, spectra_R, spectra_T
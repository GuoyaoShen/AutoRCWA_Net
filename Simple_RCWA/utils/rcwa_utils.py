import numpy as np
from numpy import linalg as LA
import scipy
from scipy import linalg as SLA
import cupy as cp
from cupy import linalg as CLA
import cupyx
from cupyx.scipy.sparse import csr_matrix, bmat
import torch
from torch import linalg as TLA

import cmath
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image

import skimage
from skimage import draw

from time import sleep
import sys

from Simple_RCWA.utils import data_utils
from Simple_RCWA.utils import calc_utils


pi = np.pi
# ====== Unit Define ======
meters = 1
centimeters = 1e-2 * meters
millimeters = 1e-3 * meters
micrometres = 1e-6 * meters

# ====== Constant Define ======
c0 = 3e8
# c0 = 299792458
e0 = 8.85e-12
u0 = 1.256e-6
yeta0 = np.sqrt(u0/e0)

#======================================================================================================================
# 0.1 version, not easy to apply new structure

def rcwa_solver(freq, eps_gold, eps_SiNx, w=0.52*0.005*millimeters, use_logger=False):
    # ================= Calculation Start
    R_total = np.zeros((len(freq),))
    T_total = np.zeros((len(freq),))

    for i_freq in range(len(freq)):
        lam0 = c0 / freq[i_freq]
        ginc = np.array([0, 0, 1])
        EP = np.array([0, 1, 0])

        # === Device Params
        ur1 = 1.  # permeability in reflection region
        er1 = 1.  # permeability in reflection region
        ur2 = 1.  # permeability in transmission region
        er2 = 1.  # permeability in transmission region
        urd = 1.  # permeability of device
        erd = np.conjugate(eps_SiNx[i_freq])  # permeability of device

        Lx = 0.005 * millimeters  # period along x
        Ly = 0.005 * millimeters  # period along y
        d1 = 0.00015 * millimeters  # thickness of layer 1
        d2 = 0.0005 * millimeters  # thickness of layer 2
        d3 = 0.00015 * millimeters  # thickness of layer 3
        # w = 0.52 * Ly  # length of one side of square

        # === RCWA Params
        Nx = 512
        Ny = np.round(Nx * Ly / Lx).astype(int)
        PQ = 1 * np.array([1, 31])

        # === Define Structure in Layers
        nxc = np.floor(Nx / 2)
        nyc = np.floor(Ny / 2)
        ER1 = 1 * np.ones((Nx, Ny))
        ER2 = erd * np.ones((Nx, Ny))
        ER3 = np.conjugate(eps_gold[i_freq]) * np.ones((Nx, Ny))
        ER = np.concatenate([ER1[..., np.newaxis], ER2[..., np.newaxis], ER3[..., np.newaxis]], axis=-1)  # [512,512,3]

        UR1 = urd * np.ones((Nx, Ny))
        UR2 = urd * np.ones((Nx, Ny))
        UR3 = urd * np.ones((Nx, Ny))
        UR = np.concatenate([UR1[..., np.newaxis], UR2[..., np.newaxis], UR3[..., np.newaxis]], axis=-1)  # [512,512,3]

        L = np.array([d1, d2, d3])

        # === Cross Sectional Grid
        dx = Lx / Nx  # grid resolution along x
        dy = Ly / Ny  # grid resolution along y
        xa = np.arange(Nx) * dx  # x axis array
        xa = xa - np.mean(xa)  # center x axis at zero
        ya = np.arange(Ny) * dy  # y axis array
        ya = ya - np.mean(ya)  # center y axis at zero
        x_axis, y_axis = np.meshgrid(xa, ya)

        ny1 = np.round(nxc - ((w / Ly) * Nx) / 2).astype(int)
        ny2 = np.round(nxc + ((w / Ly) * Nx) / 2).astype(int)
        ER[ny1 - 1:ny2, ny1 - 1:ny2, 0] = np.conjugate(eps_gold[i_freq])
        mm, nn, ll = ER.shape
        for i_ll in range(ll):
            URC_i = calc_utils.convmat(UR[..., i_ll][..., np.newaxis], PQ[0], PQ[1])
            ERC_i = calc_utils.convmat(ER[..., i_ll][..., np.newaxis], PQ[0], PQ[1])
            # print('ERC_i',ERC_i)
            # print('ERC_i', ERC_i.shape)
            if i_ll == 0:
                URC = URC_i[..., np.newaxis]
                ERC = ERC_i[..., np.newaxis]
            else:
                URC = np.concatenate((URC, URC_i[..., np.newaxis]), axis=-1)
                ERC = np.concatenate((ERC, ERC_i[..., np.newaxis]), axis=-1)

        # ====== Wave Vector Expansion ======
        nr1 = np.sqrt(ur1 * er1)  # refractive index of reflection medium, eta
        nr2 = np.sqrt(ur2 * er2)  # refractive index of transmission medium, eta
        k0 = 2 * pi / lam0
        p = np.arange(-(np.floor(PQ[0] / 2)), (np.floor(PQ[0] / 2) + 1))
        q = np.arange(-(np.floor(PQ[1] / 2)), (np.floor(PQ[1] / 2) + 1))
        kx_inc = ginc[0] / LA.norm(ginc) * k0 * nr1  # normal incidence
        ky_inc = ginc[1] / LA.norm(ginc) * k0 * nr1
        kz_inc = ginc[2] / LA.norm(ginc) * k0 * nr1

        KX = (kx_inc - 2 * pi * p / Lx) / k0
        KY = (ky_inc - 2 * pi * q / Ly) / k0
        KY, KX = np.meshgrid(KY, KX)
        # KX = np.diag(KX.squeeze())
        # KY = np.diag(KY.squeeze())
        KX = np.diag(KX.T.flatten())
        KY = np.diag(KY.T.flatten())

        # normalized reflection Kz, no minus sign ahead
        KZr = np.conj(np.sqrt((er1 * ur1 * np.eye(KX.shape[0]) - KX ** 2 - KY ** 2 + 0j).astype(complex)))
        # normalized transmission Kz
        KZt = np.conj(np.sqrt((er2 * ur2 * np.eye(KY.shape[0]) - KX ** 2 - KY ** 2 + 0j).astype(complex)))

        kx_inc = kx_inc / k0
        ky_inc = ky_inc / k0
        kz_inc = kz_inc / k0

        # === Compute Eigen-modes of Free Space
        KZ = np.conj(np.sqrt((np.eye(KX.shape[0]) - KX ** 2 - KY ** 2 + 0j).astype(complex)))

        P = np.block([[KX @ KY, np.eye(KX.shape[0]) - KX ** 2],
                      [KY ** 2 - np.eye(KX.shape[0]), -KX @ KY]])

        Q = P
        OMEGA_SQ = P @ Q
        W0 = np.block([[np.eye(KX.shape[0]), np.zeros((KX.shape[0], KX.shape[1]))],
                       [np.zeros((KX.shape[0], KX.shape[1])), np.eye(KX.shape[0])]])
        lam = 1j * np.block([[KZ, np.zeros((KZ.shape[0], KZ.shape[1]))],
                             [np.zeros((KZ.shape[0], KZ.shape[1])), KZ]])

        V0 = Q @ LA.inv(lam)

        # === Initialize Device Scattering Matrix
        S11 = np.zeros((P.shape[0], P.shape[1]))
        S12 = np.eye(P.shape[0])
        S21 = S12
        S22 = S11
        SG = {'S11': S11, 'S12': S12, 'S21': S21, 'S22': S22}

        # === Main Loop
        uu, vv, ww = ER.shape
        for ii in range(ww):
            P_ii = np.block([[KX @ LA.inv(ERC[:, :, ii]) @ KY, URC[:, :, ii] - KX @ LA.inv(ERC[:, :, ii]) @ KX],
                             [KY @ LA.inv(ERC[:, :, ii]) @ KY - URC[:, :, ii], -KY @ LA.inv(ERC[:, :, ii]) @ KX]])
            Q_ii = np.block([[KX @ LA.inv(URC[:, :, ii]) @ KY, ERC[:, :, ii] - KX @ LA.inv(URC[:, :, ii]) @ KX],
                             [KY @ LA.inv(URC[:, :, ii]) @ KY - ERC[:, :, ii], -KY @ LA.inv(URC[:, :, ii]) @ KX]])
            OMEGA_SQ_ii = P_ii @ Q_ii
            [lam_sq_ii, W_ii] = LA.eig(OMEGA_SQ_ii)
            # [lam_sq_ii, W_ii] = SLA.eig(OMEGA_SQ_ii,left=False, right=True)
            # lam_sq_ii is the same as matlab, W_ii is different
            lam_sq_ii = np.diag(lam_sq_ii)
            lam_ii = np.sqrt(lam_sq_ii)
            V_ii = Q_ii @ W_ii @ LA.inv(lam_ii)
            A0_ii = LA.inv(W_ii) @ W0 + LA.inv(V_ii) @ V0
            B0_ii = LA.inv(W_ii) @ W0 - LA.inv(V_ii) @ V0

            X_ii = SLA.expm(-lam_ii * k0 * L[ii])

            S11 = LA.inv(A0_ii - X_ii @ B0_ii @ LA.inv(A0_ii) @ X_ii @ B0_ii) @ \
                  (X_ii @ B0_ii @ LA.inv(A0_ii) @ X_ii @ A0_ii - B0_ii)
            S12 = LA.inv(A0_ii - X_ii @ B0_ii @ LA.inv(A0_ii) @ X_ii @ B0_ii) @ \
                  (X_ii @ (A0_ii - B0_ii @ LA.inv(A0_ii) @ B0_ii))
            S21 = S12
            S22 = S11
            S = {'S11': S11, 'S12': S12, 'S21': S21, 'S22': S22}
            SG = calc_utils.star(SG, S)

        # === Compute Reflection Side Connection S-Matrix
        Q_ref = ur1 ** (-1) * np.block([[KX @ KY, ur1 * er1 * np.eye(KX.shape[0]) - KX ** 2],
                                        [KY ** 2 - ur1 * er1 * np.eye(KY.shape[0]), -KY @ KX]])

        W_ref = np.block([[np.eye(KX.shape[0]), np.zeros((KX.shape[0], KX.shape[1]))],
                          [np.zeros((KY.shape[0], KY.shape[1])), np.eye(KY.shape[0])]])

        lam_ref = np.block([[1j * KZr, np.zeros((KZr.shape[0], KZr.shape[1]))],
                            [np.zeros((KZr.shape[0], KZr.shape[1])), 1j * KZr]])

        V_ref = Q_ref @ LA.inv(lam_ref)
        Ar = LA.inv(W0) @ W_ref + LA.inv(V0) @ V_ref
        Br = LA.inv(W0) @ W_ref - LA.inv(V0) @ V_ref

        S11 = -LA.inv(Ar) @ Br
        S12 = 2 * LA.inv(Ar)
        S21 = 0.5 * (Ar - Br @ LA.inv(Ar) @ Br)
        S22 = Br @ LA.inv(Ar)
        Sref = {'S11': S11, 'S12': S12, 'S21': S21, 'S22': S22}

        # === Compute Transmission Side Connection S-Matrix
        Q_trn = ur2 ** (-1) * np.block([[KX @ KY, ur2 * er2 * np.eye(KX.shape[0]) - KX ** 2],
                                        [KY ** 2 - ur2 * er2 * np.eye(KY.shape[0]), -KY @ KX]])

        W_trn = np.block([[np.eye(KX.shape[0]), np.zeros((KX.shape[0], KX.shape[1]))],
                          [np.zeros((KY.shape[0], KY.shape[1])), np.eye(KY.shape[0])]])

        lam_trn = np.block([[1j * KZt, np.zeros((KZt.shape[0], KZt.shape[1]))],
                            [np.zeros((KZt.shape[0], KZt.shape[1])), 1j * KZt]])

        V_trn = Q_trn @ LA.inv(lam_trn)
        At = LA.inv(W0) @ W_trn + LA.inv(V0) @ V_trn
        Bt = LA.inv(W0) @ W_trn - LA.inv(V0) @ V_trn

        S11 = Bt @ LA.inv(At)
        S12 = 0.5 * (At - Bt @ LA.inv(At) @ Bt)
        S21 = 2 * LA.inv(At)
        S22 = -LA.inv(At) @ Bt
        Strn = {'S11': S11, 'S12': S12, 'S21': S21, 'S22': S22}

        SG = calc_utils.star(Sref, SG)
        SG = calc_utils.star(SG, Strn)

        # === Compute Reflected and Transmitted Fields
        delta = np.concatenate(([np.zeros((1, np.floor(PQ[0] * PQ[1] / 2).astype(int))),
                                 np.array([[1]]),
                                 np.zeros((1, np.floor(PQ[0] * PQ[1] / 2).astype(int)))]), axis=-1).T

        ate = np.array([[0, 1, 0]]).T
        atm = np.array([[1, 0, 0]]).T

        esrc = np.concatenate((EP[0] * delta, EP[1] * delta), axis=0)
        csrc = LA.inv(W_ref) @ esrc

        cref = SG['S11'] @ csrc
        ctrn = SG['S21'] @ csrc

        rall = W_ref @ cref
        tall = W_trn @ ctrn

        nExp = rall.shape[0]
        rx = rall[:int(nExp / 2)]
        ry = rall[int(nExp / 2):]

        tx = tall[:int(nExp / 2)]
        ty = tall[int(nExp / 2):]

        rz = -LA.inv(KZr) @ (KX @ rx + KY @ ry)
        tz = -LA.inv(KZt) @ (KX @ tx + KY @ ty)

        R_ref = np.real(-KZr / kz_inc) * (np.abs(rx) ** 2 + np.abs(ry) ** 2 + np.abs(rz) ** 2)
        R_total[i_freq] = np.sum(np.abs(R_ref))

        T_ref = np.real(ur1 / ur2 * KZr / kz_inc) * (np.abs(tx) ** 2 + np.abs(ty) ** 2 + np.abs(tz) ** 2)
        T_total[i_freq] = np.sum(np.abs(T_ref))

        if use_logger:
            # === print a progressbar
            sys.stdout.write('\r')
            # the exact output you're looking for:
            # sys.stdout.write("[%-20s] %d%%" % ('=' * i_freq, (100/len(freq)) * (i_freq+1)))
            sys.stdout.write("Calculation Progress: %d%%" % ((100 / len(freq)) * (i_freq + 1)))
            sys.stdout.flush()

        # if i_freq==100:
        #     break

    return R_total, T_total


def rcwa_solver_Si_test_orig(freq, eps_Si, L_param=200, w_param=80, t_param=60, use_logger=False, PQ_order=11):
    # ================= Calculation Start
    R_total = np.zeros((len(freq),))
    T_total = np.zeros((len(freq),))

    for i_freq in range(len(freq)):
        lam0 = c0 / freq[i_freq]
        ginc = np.array([0, 0, 1])
        EP = np.array([0, 1, 0])

        # === Device Params
        ur1 = 1.  # permeability in reflection region
        er1 = 1.  # permeability in reflection region
        ur2 = 1.  # permeability in transmission region
        er2 = 1.  # permeability in transmission region
        urd = 1.  # permeability of device
        erd = np.conjugate(eps_Si[i_freq])  # permeability of device

        Lx = L_param * 1e-3 * millimeters  # period along x
        Ly = L_param * 1e-3 * millimeters # period along y
        d1 = t_param * 1e-3 * millimeters  # thickness of layer 1
        # d2 = 0.0005 * millimeters  # thickness of layer 2
        # d3 = 0.00015 * millimeters  # thickness of layer 3
        w = w_param * 1e-3 * millimeters  # length of one side of square

        # === RCWA Params
        Nx = 512
        Ny = np.round(Nx * Ly / Lx).astype(int)
        PQ = 1 * np.array([PQ_order, PQ_order])  # this must be singular value

        # === Define Structure in Layers
        nxc = np.floor(Nx / 2)
        nyc = np.floor(Ny / 2)
        # ER1 = 1 * np.ones((Nx, Ny))
        # ER2 = erd * np.ones((Nx, Ny))
        # ER3 = np.conjugate(eps_gold[i_freq]) * np.ones((Nx, Ny))
        # ER = np.concatenate([ER1[..., np.newaxis], ER2[..., np.newaxis], ER3[..., np.newaxis]], axis=-1)  # [512,512,3]
        ER = erd * np.ones((Nx, Ny))
        ER = ER[..., np.newaxis]

        # UR1 = urd * np.ones((Nx, Ny))
        # UR2 = urd * np.ones((Nx, Ny))
        # UR3 = urd * np.ones((Nx, Ny))
        # UR = np.concatenate([UR1[..., np.newaxis], UR2[..., np.newaxis], UR3[..., np.newaxis]], axis=-1)  # [512,512,3]
        UR = urd * np.ones((Nx, Ny))
        UR = UR[..., np.newaxis]

        # L = np.array([d1, d2, d3])
        L = np.array([d1])

        # === Cross Sectional Grid
        dx = Lx / Nx  # grid resolution along x
        dy = Ly / Ny  # grid resolution along y
        xa = np.arange(Nx) * dx  # x axis array
        xa = xa - np.mean(xa)  # center x axis at zero
        ya = np.arange(Ny) * dy  # y axis array
        ya = ya - np.mean(ya)  # center y axis at zero
        x_axis, y_axis = np.meshgrid(xa, ya)

        ny1 = np.round(nxc - ((w / Ly) * Nx) / 2).astype(int)
        ny2 = np.round(nxc + ((w / Ly) * Nx) / 2).astype(int)
        # ER[ny1 - 1:ny2, ny1 - 1:ny2, 0] = np.conjugate(eps_gold[i_freq])
        ER[ny1 - 1:ny2, ny1 - 1:ny2, 0] = 1.  # square hole in the middle
        mm, nn, ll = ER.shape
        for i_ll in range(ll):
            URC_i = calc_utils.convmat(UR[..., i_ll][..., np.newaxis], PQ[0], PQ[1])
            ERC_i = calc_utils.convmat(ER[..., i_ll][..., np.newaxis], PQ[0], PQ[1])
            # print('ERC_i',ERC_i)
            # print('ERC_i', ERC_i.shape)
            if i_ll == 0:
                URC = URC_i[..., np.newaxis]
                ERC = ERC_i[..., np.newaxis]
            else:
                URC = np.concatenate((URC, URC_i[..., np.newaxis]), axis=-1)
                ERC = np.concatenate((ERC, ERC_i[..., np.newaxis]), axis=-1)

        # ====== Wave Vector Expansion ======
        nr1 = np.sqrt(ur1 * er1)  # refractive index of reflection medium, eta
        nr2 = np.sqrt(ur2 * er2)  # refractive index of transmission medium, eta
        k0 = 2 * pi / lam0
        p = np.arange(-(np.floor(PQ[0] / 2)), (np.floor(PQ[0] / 2) + 1))
        q = np.arange(-(np.floor(PQ[1] / 2)), (np.floor(PQ[1] / 2) + 1))
        kx_inc = ginc[0] / LA.norm(ginc) * k0 * nr1  # normal incidence
        ky_inc = ginc[1] / LA.norm(ginc) * k0 * nr1
        kz_inc = ginc[2] / LA.norm(ginc) * k0 * nr1

        KX = (kx_inc - 2 * pi * p / Lx) / k0
        KY = (ky_inc - 2 * pi * q / Ly) / k0
        # print('KX', KX)
        # print('KY', KY)

        KY_temp, KX_temp = np.meshgrid(KY, KX)
        KX = np.diag(KX_temp.T.flatten())  # p now is not a flooat, it's a vector, No squeeze
        KY = np.diag(KY_temp.T.flatten())
        # print('diag KX', np.diag(KX))
        # print('diag KY', np.diag(KY))
        # print('diag KX+KY', np.diag(KX+KY))

        if (1 in KX+KY):  # prevent from singular matrix
            print('[WARNING] SINGULAR MATRIX!!! freq:', freq[i_freq], 'i_freq', i_freq)
            R_total[i_freq] = R_total[i_freq-1]
            T_total[i_freq] = T_total[i_freq-1]
            continue

        # normalized reflection Kz, no minus sign ahead
        KZr = np.conj(np.sqrt((er1 * ur1 * np.eye(KX.shape[0]) - KX ** 2 - KY ** 2 + 0j).astype(complex)))
        # normalized transmission Kz
        KZt = np.conj(np.sqrt((er2 * ur2 * np.eye(KY.shape[0]) - KX ** 2 - KY ** 2 + 0j).astype(complex)))

        kx_inc = kx_inc / k0
        ky_inc = ky_inc / k0
        kz_inc = kz_inc / k0

        # === Compute Eigen-modes of Free Space
        KZ = np.conj(np.sqrt((np.eye(KX.shape[0]) - KX ** 2 - KY ** 2 + 0j).astype(complex)))

        P = np.block([[KX @ KY, np.eye(KX.shape[0]) - KX ** 2],
                      [KY ** 2 - np.eye(KX.shape[0]), -KX @ KY]])

        Q = P
        OMEGA_SQ = P @ Q
        W0 = np.block([[np.eye(KX.shape[0]), np.zeros((KX.shape[0], KX.shape[1]))],
                       [np.zeros((KX.shape[0], KX.shape[1])), np.eye(KX.shape[0])]])
        lam = 1j * np.block([[KZ, np.zeros((KZ.shape[0], KZ.shape[1]))],
                             [np.zeros((KZ.shape[0], KZ.shape[1])), KZ]])

        # print('freq', freq[i_freq])
        # print('diag KX', np.diag(KX))
        # print('diag KY', np.diag(KY))
        # print('diag KZ', np.diag(KZ))
        # print('diag lam', np.diag(lam))
        V0 = Q @ LA.inv(lam)

        # === Initialize Device Scattering Matrix
        S11 = np.zeros((P.shape[0], P.shape[1]))
        S12 = np.eye(P.shape[0])
        S21 = S12
        S22 = S11
        SG = {'S11': S11, 'S12': S12, 'S21': S21, 'S22': S22}

        # === Main Loop
        uu, vv, ww = ER.shape
        for ii in range(ww):
            P_ii = np.block([[KX @ LA.inv(ERC[:, :, ii]) @ KY, URC[:, :, ii] - KX @ LA.inv(ERC[:, :, ii]) @ KX],
                             [KY @ LA.inv(ERC[:, :, ii]) @ KY - URC[:, :, ii], -KY @ LA.inv(ERC[:, :, ii]) @ KX]])
            Q_ii = np.block([[KX @ LA.inv(URC[:, :, ii]) @ KY, ERC[:, :, ii] - KX @ LA.inv(URC[:, :, ii]) @ KX],
                             [KY @ LA.inv(URC[:, :, ii]) @ KY - ERC[:, :, ii], -KY @ LA.inv(URC[:, :, ii]) @ KX]])
            OMEGA_SQ_ii = P_ii @ Q_ii
            [lam_sq_ii, W_ii] = LA.eig(OMEGA_SQ_ii)
            # [lam_sq_ii, W_ii] = SLA.eig(OMEGA_SQ_ii,left=False, right=True)
            # lam_sq_ii is the same as matlab, W_ii is different
            lam_sq_ii = np.diag(lam_sq_ii)
            lam_ii = np.sqrt(lam_sq_ii)
            V_ii = Q_ii @ W_ii @ LA.inv(lam_ii)
            A0_ii = LA.inv(W_ii) @ W0 + LA.inv(V_ii) @ V0
            B0_ii = LA.inv(W_ii) @ W0 - LA.inv(V_ii) @ V0

            X_ii = SLA.expm(-lam_ii * k0 * L[ii])
            # X_ii = SLA.expm(float(-lam_ii * k0 * L[ii]))

            S11 = LA.inv(A0_ii - X_ii @ B0_ii @ LA.inv(A0_ii) @ X_ii @ B0_ii) @ \
                  (X_ii @ B0_ii @ LA.inv(A0_ii) @ X_ii @ A0_ii - B0_ii)
            S12 = LA.inv(A0_ii - X_ii @ B0_ii @ LA.inv(A0_ii) @ X_ii @ B0_ii) @ \
                  (X_ii @ (A0_ii - B0_ii @ LA.inv(A0_ii) @ B0_ii))
            S21 = S12
            S22 = S11
            S = {'S11': S11, 'S12': S12, 'S21': S21, 'S22': S22}
            SG = calc_utils.star(SG, S)

        # === Compute Reflection Side Connection S-Matrix
        Q_ref = ur1 ** (-1) * np.block([[KX @ KY, ur1 * er1 * np.eye(KX.shape[0]) - KX ** 2],
                                        [KY ** 2 - ur1 * er1 * np.eye(KY.shape[0]), -KY @ KX]])

        W_ref = np.block([[np.eye(KX.shape[0]), np.zeros((KX.shape[0], KX.shape[1]))],
                          [np.zeros((KY.shape[0], KY.shape[1])), np.eye(KY.shape[0])]])

        lam_ref = np.block([[1j * KZr, np.zeros((KZr.shape[0], KZr.shape[1]))],
                            [np.zeros((KZr.shape[0], KZr.shape[1])), 1j * KZr]])

        V_ref = Q_ref @ LA.inv(lam_ref)
        Ar = LA.inv(W0) @ W_ref + LA.inv(V0) @ V_ref
        Br = LA.inv(W0) @ W_ref - LA.inv(V0) @ V_ref

        S11 = -LA.inv(Ar) @ Br
        S12 = 2 * LA.inv(Ar)
        S21 = 0.5 * (Ar - Br @ LA.inv(Ar) @ Br)
        S22 = Br @ LA.inv(Ar)
        Sref = {'S11': S11, 'S12': S12, 'S21': S21, 'S22': S22}

        # === Compute Transmission Side Connection S-Matrix
        Q_trn = ur2 ** (-1) * np.block([[KX @ KY, ur2 * er2 * np.eye(KX.shape[0]) - KX ** 2],
                                        [KY ** 2 - ur2 * er2 * np.eye(KY.shape[0]), -KY @ KX]])

        W_trn = np.block([[np.eye(KX.shape[0]), np.zeros((KX.shape[0], KX.shape[1]))],
                          [np.zeros((KY.shape[0], KY.shape[1])), np.eye(KY.shape[0])]])

        lam_trn = np.block([[1j * KZt, np.zeros((KZt.shape[0], KZt.shape[1]))],
                            [np.zeros((KZt.shape[0], KZt.shape[1])), 1j * KZt]])

        V_trn = Q_trn @ LA.inv(lam_trn)
        At = LA.inv(W0) @ W_trn + LA.inv(V0) @ V_trn
        Bt = LA.inv(W0) @ W_trn - LA.inv(V0) @ V_trn

        S11 = Bt @ LA.inv(At)
        S12 = 0.5 * (At - Bt @ LA.inv(At) @ Bt)
        S21 = 2 * LA.inv(At)
        S22 = -LA.inv(At) @ Bt
        Strn = {'S11': S11, 'S12': S12, 'S21': S21, 'S22': S22}

        SG = calc_utils.star(Sref, SG)
        SG = calc_utils.star(SG, Strn)

        # === Compute Reflected and Transmitted Fields
        delta = np.concatenate(([np.zeros((1, np.floor(PQ[0] * PQ[1] / 2).astype(int))),
                                 np.array([[1]]),
                                 np.zeros((1, np.floor(PQ[0] * PQ[1] / 2).astype(int)))]), axis=-1).T

        ate = np.array([[0, 1, 0]]).T
        atm = np.array([[1, 0, 0]]).T

        esrc = np.concatenate((EP[0] * delta, EP[1] * delta), axis=0)
        csrc = LA.inv(W_ref) @ esrc

        cref = SG['S11'] @ csrc
        ctrn = SG['S21'] @ csrc

        rall = W_ref @ cref
        tall = W_trn @ ctrn

        nExp = rall.shape[0]
        rx = rall[:int(nExp / 2)]
        ry = rall[int(nExp / 2):]

        tx = tall[:int(nExp / 2)]
        ty = tall[int(nExp / 2):]

        rz = -LA.inv(KZr) @ (KX @ rx + KY @ ry)
        tz = -LA.inv(KZt) @ (KX @ tx + KY @ ty)

        R_ref = np.real(-KZr / kz_inc) * (np.abs(rx) ** 2 + np.abs(ry) ** 2 + np.abs(rz) ** 2)
        R_total[i_freq] = np.sum(np.abs(R_ref))

        T_ref = np.real(ur1 / ur2 * KZr / kz_inc) * (np.abs(tx) ** 2 + np.abs(ty) ** 2 + np.abs(tz) ** 2)
        T_total[i_freq] = np.sum(np.abs(T_ref))

        if use_logger:
            # === print a progressbar
            sys.stdout.write('\r')
            sys.stdout.write("Calculation Progress: %d%%" % ((100 / len(freq)) * (i_freq + 1)))
            sys.stdout.flush()

        # if i_freq==100:
        #     break

    return R_total, T_total



#======================================================================================================================
# 0.2 version, add gpu support and customable in some degree

def rcwa_preprocess_Si_test(eps, L_param, w_param, t_param, PQ_order):
    # === Device Params
    urd = 1.  # permeability of device
    erd = np.conjugate(eps)  # permeability of device

    Lx = L_param * 1e-3 * millimeters  # period along x
    Ly = L_param * 1e-3 * millimeters  # period along y
    d1 = t_param * 1e-3 * millimeters  # thickness of layer 1
    # d2 = 0.0005 * millimeters  # thickness of layer 2
    # d3 = 0.00015 * millimeters  # thickness of layer 3
    w = w_param * 1e-3 * millimeters  # length of one side of square

    # === RCWA Params
    Nx = 512  #512
    Ny = np.round(Nx * Ly / Lx).astype(int)
    PQ = 1 * np.array([PQ_order, PQ_order])  # this must be singular value

    # === Define Structure in Layers
    nxc = np.floor(Nx / 2)
    nyc = np.floor(Ny / 2)
    # ER1 = 1 * np.ones((Nx, Ny))
    # ER2 = erd * np.ones((Nx, Ny))
    # ER3 = np.conjugate(eps_gold[i_freq]) * np.ones((Nx, Ny))
    # ER = np.concatenate([ER1[..., np.newaxis], ER2[..., np.newaxis], ER3[..., np.newaxis]], axis=-1)  # [512,512,3]
    ER = erd * np.ones((Nx, Ny))
    ER = ER[..., np.newaxis]

    # UR1 = urd * np.ones((Nx, Ny))
    # UR2 = urd * np.ones((Nx, Ny))
    # UR3 = urd * np.ones((Nx, Ny))
    # UR = np.concatenate([UR1[..., np.newaxis], UR2[..., np.newaxis], UR3[..., np.newaxis]], axis=-1)  # [512,512,3]
    UR = urd * np.ones((Nx, Ny))
    UR = UR[..., np.newaxis]

    # === Cross Sectional Grid
    ny1 = np.round(nxc - ((w / Ly) * Nx) / 2).astype(int)
    ny2 = np.round(nxc + ((w / Ly) * Nx) / 2).astype(int)
    # === Pattern Structure
    ER[ny1 - 1:ny2, ny1 - 1:ny2, 0] = 1.  # square hole in the middle
    mm, nn, ll = ER.shape
    for i_ll in range(ll):
        URC_i = calc_utils.convmat(UR[..., i_ll][..., np.newaxis], PQ[0], PQ[1])
        ERC_i = calc_utils.convmat(ER[..., i_ll][..., np.newaxis], PQ[0], PQ[1])
        # print('ERC_i',ERC_i)
        # print('ERC_i', ERC_i.shape)
        if i_ll == 0:
            URC = URC_i[..., np.newaxis]
            ERC = ERC_i[..., np.newaxis]
        else:
            URC = np.concatenate((URC, URC_i[..., np.newaxis]), axis=-1)
            ERC = np.concatenate((ERC, ERC_i[..., np.newaxis]), axis=-1)

    return ER, UR, ERC, URC


def rcwa_solver_Si_test(freq, eps_Si, L_param=200, w_param=80, t_param=60, use_logger=False, PQ_order=11):
    # ================= Calculation Start
    R_total = np.zeros((len(freq),))
    T_total = np.zeros((len(freq),))

    # Physics environment define
    ur1 = 1.  # permeability in reflection region
    er1 = 1.  # permeability in reflection region
    ur2 = 1.  # permeability in transmission region
    er2 = 1.  # permeability in transmission region
    PQ = 1 * np.array([PQ_order, PQ_order])  # this must be singular value
    ginc = np.array([0, 0, 1])  # incident light source
    EP = np.array([0, 1, 0])

    # Geometry structure define
    Lx = L_param * 1e-3 * millimeters  # period along x
    Ly = L_param * 1e-3 * millimeters # period along y
    d1 = t_param * 1e-3 * millimeters  # thickness of layer 1
    L = np.array([d1])

    for i_freq in range(len(freq)):
        lam0 = c0 / freq[i_freq]
        # ====== RCWA Preprocessing ======
        ER, UR, ERC, URC = rcwa_preprocess_Si_test(eps_Si[i_freq],
                                           L_param=L_param, w_param=w_param, t_param=t_param, PQ_order=PQ_order)

        # ====== Wave Vector Expansion ======
        nr1 = np.sqrt(ur1 * er1)  # refractive index of reflection medium, eta
        # nr2 = np.sqrt(ur2 * er2)  # refractive index of transmission medium, eta
        k0 = 2 * pi / lam0
        p = np.arange(-(np.floor(PQ[0] / 2)), (np.floor(PQ[0] / 2) + 1))
        q = np.arange(-(np.floor(PQ[1] / 2)), (np.floor(PQ[1] / 2) + 1))
        kx_inc = ginc[0] / LA.norm(ginc) * k0 * nr1  # normal incidence
        ky_inc = ginc[1] / LA.norm(ginc) * k0 * nr1
        kz_inc = ginc[2] / LA.norm(ginc) * k0 * nr1

        KX = (kx_inc - 2 * pi * p / Lx) / k0
        KY = (ky_inc - 2 * pi * q / Ly) / k0

        KY_temp, KX_temp = np.meshgrid(KY, KX)
        KX = np.diag(KX_temp.T.flatten())  # p now is not a flooat, it's a vector, No squeeze
        KY = np.diag(KY_temp.T.flatten())

        if (1 in KX+KY):  # prevent from singular matrix
            print('\n[WARNING] SINGULAR MATRIX!!! freq:', freq[i_freq], 'i_freq', i_freq)
            R_total[i_freq] = R_total[i_freq-1]
            T_total[i_freq] = T_total[i_freq-1]
            continue

        # normalized reflection Kz, no minus sign ahead
        KZr = np.conj(np.sqrt((er1 * ur1 * np.eye(KX.shape[0]) - KX ** 2 - KY ** 2 + 0j).astype(complex)))
        # normalized transmission Kz
        KZt = np.conj(np.sqrt((er2 * ur2 * np.eye(KY.shape[0]) - KX ** 2 - KY ** 2 + 0j).astype(complex)))

        kz_inc = kz_inc / k0

        # === Compute Eigen-modes of Free Space
        KZ = np.conj(np.sqrt((np.eye(KX.shape[0]) - KX ** 2 - KY ** 2 + 0j).astype(complex)))

        P = np.block([[KX @ KY, np.eye(KX.shape[0]) - KX ** 2],
                      [KY ** 2 - np.eye(KX.shape[0]), -KX @ KY]])

        Q = P

        W0 = np.block([[np.eye(KX.shape[0]), np.zeros((KX.shape[0], KX.shape[1]))],
                       [np.zeros((KX.shape[0], KX.shape[1])), np.eye(KX.shape[0])]])
        lam = 1j * np.block([[KZ, np.zeros((KZ.shape[0], KZ.shape[1]))],
                             [np.zeros((KZ.shape[0], KZ.shape[1])), KZ]])

        V0 = Q @ LA.inv(lam)

        # === Initialize Device Scattering Matrix
        S11 = np.zeros((P.shape[0], P.shape[1]))
        S12 = np.eye(P.shape[0])
        S21 = S12
        S22 = S11
        SG = {'S11': S11, 'S12': S12, 'S21': S21, 'S22': S22}

        # === Main Loop
        uu, vv, ww = ER.shape
        for ii in range(ww):
            P_ii = np.block([[KX @ LA.inv(ERC[:, :, ii]) @ KY, URC[:, :, ii] - KX @ LA.inv(ERC[:, :, ii]) @ KX],
                             [KY @ LA.inv(ERC[:, :, ii]) @ KY - URC[:, :, ii], -KY @ LA.inv(ERC[:, :, ii]) @ KX]])
            Q_ii = np.block([[KX @ LA.inv(URC[:, :, ii]) @ KY, ERC[:, :, ii] - KX @ LA.inv(URC[:, :, ii]) @ KX],
                             [KY @ LA.inv(URC[:, :, ii]) @ KY - ERC[:, :, ii], -KY @ LA.inv(URC[:, :, ii]) @ KX]])
            OMEGA_SQ_ii = P_ii @ Q_ii
            [lam_sq_ii, W_ii] = LA.eig(OMEGA_SQ_ii)
            # [lam_sq_ii, W_ii] = SLA.eig(OMEGA_SQ_ii,left=False, right=True)
            # lam_sq_ii is the same as matlab, W_ii is different
            lam_sq_ii = np.diag(lam_sq_ii)
            lam_ii = np.sqrt(lam_sq_ii)
            V_ii = Q_ii @ W_ii @ LA.inv(lam_ii)
            A0_ii = LA.inv(W_ii) @ W0 + LA.inv(V_ii) @ V0
            B0_ii = LA.inv(W_ii) @ W0 - LA.inv(V_ii) @ V0

            X_ii = SLA.expm(-lam_ii * k0 * L[ii])
            # X_ii = SLA.expm(float(-lam_ii * k0 * L[ii]))

            S11 = LA.inv(A0_ii - X_ii @ B0_ii @ LA.inv(A0_ii) @ X_ii @ B0_ii) @ \
                  (X_ii @ B0_ii @ LA.inv(A0_ii) @ X_ii @ A0_ii - B0_ii)
            S12 = LA.inv(A0_ii - X_ii @ B0_ii @ LA.inv(A0_ii) @ X_ii @ B0_ii) @ \
                  (X_ii @ (A0_ii - B0_ii @ LA.inv(A0_ii) @ B0_ii))
            S21 = S12
            S22 = S11
            S = {'S11': S11, 'S12': S12, 'S21': S21, 'S22': S22}
            SG = calc_utils.star(SG, S)

        # === Compute Reflection Side Connection S-Matrix
        Q_ref = ur1 ** (-1) * np.block([[KX @ KY, ur1 * er1 * np.eye(KX.shape[0]) - KX ** 2],
                                        [KY ** 2 - ur1 * er1 * np.eye(KY.shape[0]), -KY @ KX]])

        W_ref = np.block([[np.eye(KX.shape[0]), np.zeros((KX.shape[0], KX.shape[1]))],
                          [np.zeros((KY.shape[0], KY.shape[1])), np.eye(KY.shape[0])]])

        lam_ref = np.block([[1j * KZr, np.zeros((KZr.shape[0], KZr.shape[1]))],
                            [np.zeros((KZr.shape[0], KZr.shape[1])), 1j * KZr]])

        V_ref = Q_ref @ LA.inv(lam_ref)
        Ar = LA.inv(W0) @ W_ref + LA.inv(V0) @ V_ref
        Br = LA.inv(W0) @ W_ref - LA.inv(V0) @ V_ref

        S11 = -LA.inv(Ar) @ Br
        S12 = 2 * LA.inv(Ar)
        S21 = 0.5 * (Ar - Br @ LA.inv(Ar) @ Br)
        S22 = Br @ LA.inv(Ar)
        Sref = {'S11': S11, 'S12': S12, 'S21': S21, 'S22': S22}

        # === Compute Transmission Side Connection S-Matrix
        Q_trn = ur2 ** (-1) * np.block([[KX @ KY, ur2 * er2 * np.eye(KX.shape[0]) - KX ** 2],
                                        [KY ** 2 - ur2 * er2 * np.eye(KY.shape[0]), -KY @ KX]])

        W_trn = np.block([[np.eye(KX.shape[0]), np.zeros((KX.shape[0], KX.shape[1]))],
                          [np.zeros((KY.shape[0], KY.shape[1])), np.eye(KY.shape[0])]])

        lam_trn = np.block([[1j * KZt, np.zeros((KZt.shape[0], KZt.shape[1]))],
                            [np.zeros((KZt.shape[0], KZt.shape[1])), 1j * KZt]])

        V_trn = Q_trn @ LA.inv(lam_trn)
        At = LA.inv(W0) @ W_trn + LA.inv(V0) @ V_trn
        Bt = LA.inv(W0) @ W_trn - LA.inv(V0) @ V_trn

        S11 = Bt @ LA.inv(At)
        S12 = 0.5 * (At - Bt @ LA.inv(At) @ Bt)
        S21 = 2 * LA.inv(At)
        S22 = -LA.inv(At) @ Bt
        Strn = {'S11': S11, 'S12': S12, 'S21': S21, 'S22': S22}

        SG = calc_utils.star(Sref, SG)
        SG = calc_utils.star(SG, Strn)

        # === Compute Reflected and Transmitted Fields
        delta = np.concatenate(([np.zeros((1, np.floor(PQ[0] * PQ[1] / 2).astype(int))),
                                 np.array([[1]]),
                                 np.zeros((1, np.floor(PQ[0] * PQ[1] / 2).astype(int)))]), axis=-1).T

        esrc = np.concatenate((EP[0] * delta, EP[1] * delta), axis=0)
        csrc = LA.inv(W_ref) @ esrc

        cref = SG['S11'] @ csrc
        ctrn = SG['S21'] @ csrc

        rall = W_ref @ cref
        tall = W_trn @ ctrn

        nExp = rall.shape[0]
        rx = rall[:int(nExp / 2)]
        ry = rall[int(nExp / 2):]

        tx = tall[:int(nExp / 2)]
        ty = tall[int(nExp / 2):]

        rz = -LA.inv(KZr) @ (KX @ rx + KY @ ry)
        tz = -LA.inv(KZt) @ (KX @ tx + KY @ ty)

        R_ref = np.real(-KZr / kz_inc) * (np.abs(rx) ** 2 + np.abs(ry) ** 2 + np.abs(rz) ** 2)
        R_total[i_freq] = np.sum(np.abs(R_ref))

        T_ref = np.real(ur1 / ur2 * KZr / kz_inc) * (np.abs(tx) ** 2 + np.abs(ty) ** 2 + np.abs(tz) ** 2)
        T_total[i_freq] = np.sum(np.abs(T_ref))

        if use_logger:
            # === print a progressbar
            sys.stdout.write('\r')
            sys.stdout.write("Calculation Progress: %d%%" % ((100 / len(freq)) * (i_freq + 1)))
            sys.stdout.flush()

    return R_total, T_total


def rcwa_solver_cuda_Si_test(freq, eps_Si, L_param=200, w_param=80, t_param=60, use_logger=False, PQ_order=11):
    # ================= Calculation Start
    R_total = np.zeros((len(freq),))
    T_total = np.zeros((len(freq),))

    # Physics environment define
    ur1 = 1.  # permeability in reflection region
    er1 = 1.  # permeability in reflection region
    ur2 = 1.  # permeability in transmission region
    er2 = 1.  # permeability in transmission region
    PQ = 1 * np.array([PQ_order, PQ_order])  # this must be singular value
    ginc = np.array([0, 0, 1])  # incident light source
    EP = np.array([0, 1, 0])

    # Geometry structure define
    Lx = L_param * 1e-3 * millimeters  # period along x
    Ly = L_param * 1e-3 * millimeters # period along y
    d1 = t_param * 1e-3 * millimeters  # thickness of layer 1
    L = np.array([d1])

    for i_freq in range(len(freq)):
        lam0 = c0 / freq[i_freq]
        # ====== RCWA Preprocessing ======
        ER, UR, ERC, URC = rcwa_preprocess_Si_test(eps_Si[i_freq],
                                           L_param=L_param, w_param=w_param, t_param=t_param, PQ_order=PQ_order)
        # ER, UR, ERC, URC = cp.asarray(ER), cp.asarray(UR), cp.asarray(ERC), cp.asarray(URC)

        # ====== Wave Vector Expansion ======
        nr1 = np.sqrt(ur1 * er1)  # refractive index of reflection medium, eta
        # nr2 = np.sqrt(ur2 * er2)  # refractive index of transmission medium, eta
        k0 = 2 * pi / lam0
        p = np.arange(-(np.floor(PQ[0] / 2)), (np.floor(PQ[0] / 2) + 1))
        q = np.arange(-(np.floor(PQ[1] / 2)), (np.floor(PQ[1] / 2) + 1))
        kx_inc = ginc[0] / LA.norm(ginc) * k0 * nr1  # normal incidence
        ky_inc = ginc[1] / LA.norm(ginc) * k0 * nr1
        kz_inc = ginc[2] / LA.norm(ginc) * k0 * nr1

        KX = (kx_inc - 2 * pi * p / Lx) / k0
        KY = (ky_inc - 2 * pi * q / Ly) / k0

        KY_temp, KX_temp = np.meshgrid(KY, KX)
        KX = np.diag(KX_temp.T.flatten())  # p now is not a flooat, it's a vector, No squeeze
        KY = np.diag(KY_temp.T.flatten())

        if (1 in KX+KY):  # prevent from singular matrix
            print('\n[WARNING] SINGULAR MATRIX!!! freq:', freq[i_freq], 'i_freq', i_freq)
            R_total[i_freq] = R_total[i_freq-1]
            T_total[i_freq] = T_total[i_freq-1]
            continue

        # normalized reflection Kz, no minus sign ahead
        KZr = np.conj(np.sqrt((er1 * ur1 * np.eye(KX.shape[0]) - KX ** 2 - KY ** 2 + 0j).astype(complex)))
        # normalized transmission Kz
        KZt = np.conj(np.sqrt((er2 * ur2 * np.eye(KY.shape[0]) - KX ** 2 - KY ** 2 + 0j).astype(complex)))

        kz_inc = kz_inc / k0

        # === Compute Eigen-modes of Free Space
        KZ = np.conj(np.sqrt((np.eye(KX.shape[0]) - KX ** 2 - KY ** 2 + 0j).astype(complex)))

        P = np.block([[KX @ KY, np.eye(KX.shape[0]) - KX ** 2],
                      [KY ** 2 - np.eye(KX.shape[0]), -KX @ KY]])

        Q = P

        W0 = np.block([[np.eye(KX.shape[0]), np.zeros((KX.shape[0], KX.shape[1]))],
                       [np.zeros((KX.shape[0], KX.shape[1])), np.eye(KX.shape[0])]])
        lam = 1j * np.block([[KZ, np.zeros((KZ.shape[0], KZ.shape[1]))],
                             [np.zeros((KZ.shape[0], KZ.shape[1])), KZ]])

        Q, lam, W0 = cp.asarray(Q), cp.asarray(lam), cp.asarray(W0)
        V0 = Q @ CLA.inv(lam)
        # KX, KY = cp.asarray(KX), cp.asarray(KY)

        # === Initialize Device Scattering Matrix
        S11 = np.zeros((P.shape[0], P.shape[1]))
        S12 = np.eye(P.shape[0])
        S21 = S12
        S22 = S11
        S11, S12, S21, S22 = cp.asarray(S11), cp.asarray(S12), cp.asarray(S21), cp.asarray(S22)
        SG = {'S11': S11, 'S12': S12, 'S21': S21, 'S22': S22}

        # === Main Loop
        uu, vv, ww = ER.shape
        for ii in range(ww):
            P_ii = np.block([[KX @ LA.inv(ERC[:, :, ii]) @ KY, URC[:, :, ii] - KX @ LA.inv(ERC[:, :, ii]) @ KX],
                             [KY @ LA.inv(ERC[:, :, ii]) @ KY - URC[:, :, ii], -KY @ LA.inv(ERC[:, :, ii]) @ KX]])
            Q_ii = np.block([[KX @ LA.inv(URC[:, :, ii]) @ KY, ERC[:, :, ii] - KX @ LA.inv(URC[:, :, ii]) @ KX],
                             [KY @ LA.inv(URC[:, :, ii]) @ KY - ERC[:, :, ii], -KY @ LA.inv(URC[:, :, ii]) @ KX]])

            OMEGA_SQ_ii = P_ii @ Q_ii
            [lam_sq_ii, W_ii] = LA.eig(OMEGA_SQ_ii)
            # [lam_sq_ii, W_ii] = SLA.eig(OMEGA_SQ_ii,left=False, right=True)
            # lam_sq_ii is the same as matlab, W_ii is different
            lam_sq_ii = np.diag(lam_sq_ii)
            lam_ii = np.sqrt(lam_sq_ii)

            Q_ii, W_ii, lam_ii = cp.asarray(Q_ii), cp.asarray(W_ii), cp.asarray(lam_ii)
            V_ii = Q_ii @ W_ii @ LA.inv(lam_ii)
            A0_ii = CLA.inv(W_ii) @ W0 + CLA.inv(V_ii) @ V0
            B0_ii = CLA.inv(W_ii) @ W0 - CLA.inv(V_ii) @ V0

            X_ii = SLA.expm(-cp.asnumpy(lam_ii) * k0 * L[ii])
            # X_ii = SLA.expm(float(-lam_ii * k0 * L[ii]))

            X_ii = cp.asarray(X_ii)
            S11 = CLA.inv(A0_ii - X_ii @ B0_ii @ LA.inv(A0_ii) @ X_ii @ B0_ii) @ \
                  (X_ii @ B0_ii @ LA.inv(A0_ii) @ X_ii @ A0_ii - B0_ii)
            S12 = CLA.inv(A0_ii - X_ii @ B0_ii @ LA.inv(A0_ii) @ X_ii @ B0_ii) @ \
                  (X_ii @ (A0_ii - B0_ii @ LA.inv(A0_ii) @ B0_ii))
            S21 = S12
            S22 = S11
            S = {'S11': S11, 'S12': S12, 'S21': S21, 'S22': S22}
            SG = calc_utils.star(SG, S, device='gpu')

        # === Compute Reflection Side Connection S-Matrix
        Q_ref = ur1 ** (-1) * np.block([[KX @ KY, ur1 * er1 * np.eye(KX.shape[0]) - KX ** 2],
                                        [KY ** 2 - ur1 * er1 * np.eye(KY.shape[0]), -KY @ KX]])

        W_ref = np.block([[np.eye(KX.shape[0]), np.zeros((KX.shape[0], KX.shape[1]))],
                          [np.zeros((KY.shape[0], KY.shape[1])), np.eye(KY.shape[0])]])

        lam_ref = np.block([[1j * KZr, np.zeros((KZr.shape[0], KZr.shape[1]))],
                            [np.zeros((KZr.shape[0], KZr.shape[1])), 1j * KZr]])

        Q_ref, lam_ref = cp.asarray(Q_ref), cp.asarray(lam_ref)
        V_ref = Q_ref @ CLA.inv(lam_ref)

        Ar = CLA.inv(W0) @ cp.asarray(W_ref) + CLA.inv(V0) @ V_ref
        Br = CLA.inv(W0) @ cp.asarray(W_ref) - CLA.inv(V0) @ V_ref

        S11 = -CLA.inv(Ar) @ Br
        S12 = 2 * CLA.inv(Ar)
        S21 = 0.5 * (Ar - Br @ CLA.inv(Ar) @ Br)
        S22 = Br @ CLA.inv(Ar)
        Sref = {'S11': S11, 'S12': S12, 'S21': S21, 'S22': S22}

        # === Compute Transmission Side Connection S-Matrix
        Q_trn = ur2 ** (-1) * np.block([[KX @ KY, ur2 * er2 * np.eye(KX.shape[0]) - KX ** 2],
                                        [KY ** 2 - ur2 * er2 * np.eye(KY.shape[0]), -KY @ KX]])

        W_trn = np.block([[np.eye(KX.shape[0]), np.zeros((KX.shape[0], KX.shape[1]))],
                          [np.zeros((KY.shape[0], KY.shape[1])), np.eye(KY.shape[0])]])

        lam_trn = np.block([[1j * KZt, np.zeros((KZt.shape[0], KZt.shape[1]))],
                            [np.zeros((KZt.shape[0], KZt.shape[1])), 1j * KZt]])

        V_trn = cp.asarray(Q_trn) @ CLA.inv(cp.asarray(lam_trn))
        At = LA.inv(W0) @ cp.asarray(W_trn) + CLA.inv(V0) @ V_trn
        Bt = LA.inv(W0) @ cp.asarray(W_trn) - CLA.inv(V0) @ V_trn

        S11 = Bt @ CLA.inv(At)
        S12 = 0.5 * (At - Bt @ CLA.inv(At) @ Bt)
        S21 = 2 * CLA.inv(At)
        S22 = -CLA.inv(At) @ Bt
        Strn = {'S11': S11, 'S12': S12, 'S21': S21, 'S22': S22}

        SG = calc_utils.star(Sref, SG, device='gpu')
        SG = calc_utils.star(SG, Strn, device='gpu')

        # === Compute Reflected and Transmitted Fields
        delta = np.concatenate(([np.zeros((1, np.floor(PQ[0] * PQ[1] / 2).astype(int))),
                                 np.array([[1]]),
                                 np.zeros((1, np.floor(PQ[0] * PQ[1] / 2).astype(int)))]), axis=-1).T

        esrc = np.concatenate((EP[0] * delta, EP[1] * delta), axis=0)
        csrc = CLA.inv(cp.asarray(W_ref)) @ cp.asarray(esrc)

        cref = SG['S11'] @ csrc
        ctrn = SG['S21'] @ csrc

        rall = cp.asarray(W_ref) @ cref
        tall = cp.asarray(W_trn) @ ctrn

        nExp = rall.shape[0]
        rx = rall[:int(nExp / 2)]
        ry = rall[int(nExp / 2):]

        tx = tall[:int(nExp / 2)]
        ty = tall[int(nExp / 2):]

        rz = -CLA.inv(cp.asarray(KZr)) @ (cp.asarray(KX) @ rx + cp.asarray(KY) @ ry)
        tz = -CLA.inv(cp.asarray(KZt)) @ (cp.asarray(KX) @ tx + cp.asarray(KY) @ ty)

        R_ref = np.real(-KZr / kz_inc) * (np.abs(cp.asnumpy(rx)) ** 2 + np.abs(cp.asnumpy(ry)) ** 2 + np.abs(cp.asnumpy(rz)) ** 2)
        R_total[i_freq] = np.sum(np.abs(R_ref))

        T_ref = np.real(ur1 / ur2 * KZr / kz_inc) * (np.abs(cp.asnumpy(tx)) ** 2 + np.abs(cp.asnumpy(ty)) ** 2 + np.abs(cp.asnumpy(tz)) ** 2)
        T_total[i_freq] = np.sum(np.abs(T_ref))

        if use_logger:
            # === print a progressbar
            sys.stdout.write('\r')
            sys.stdout.write("Calculation Progress: %d%%" % ((100 / len(freq)) * (i_freq + 1)))
            sys.stdout.flush()

    return R_total, T_total



#======================================================================================================================
# official version, use class
'''
For each new layer, add a layerfunc to define the geometry pattern inside.
Places need to remove should be assigned as 1, while places that has material should be assigned as erd.
'''

def layerfunc_Si_square_hole(ER, params_geometry, params_mesh):
    w_param = 160
    # ================= Geometry Params
    Lx = params_geometry[0]
    Ly = params_geometry[1]
    L = params_geometry[2]  # L=[d1,...,dn], thickness of layers

    # ================= Mesh Params
    Nx = params_mesh[0]
    Ny = params_mesh[1]

    # === Define Structure in Layers
    nxc = np.floor(Nx / 2)
    nyc = np.floor(Ny / 2)

    # === Cross Sectional Grid
    w = w_param * 1e-3 * millimeters  # length of one side of square
    ny1 = np.round(nxc - ((w / Ly) * Nx) / 2).astype(int)
    ny2 = np.round(nxc + ((w / Ly) * Nx) / 2).astype(int)
    # === Pattern Structure
    ER[ny1 - 1:ny2, ny1 - 1:ny2, 0] = 1.  # square hole in the middle

    return ER


def layerfunc_absorber_ellipse_hole(ER, params_geometry, params_mesh, layer_params):
    '''
    layer_params: [D1, D2], two axes of the ellipse hole. Already contain the unit.
    '''
    # D1 = 150 * micrometres
    # D2 = 130 * micrometres
    D1 = layer_params[0]
    D2 = layer_params[1]
    r_radius = D1 / 2
    c_radius = D2 / 2
    # ================= Geometry Params
    Lx = params_geometry[0]
    Ly = params_geometry[1]
    L = params_geometry[2]  # L=[d1,...,dn], thickness of layers

    r = Lx / 2
    c = Ly / 2

    # ================= Mesh Params
    Nx = params_mesh[0]
    Ny = params_mesh[1]
    dx = Lx / Nx
    dy = Ly / Ny

    # === Define Structure in Layers
    nxc = np.floor(Nx / 2)
    nyc = np.floor(Ny / 2)

    # === Cross Sectional Grid
    r_coord = Nx // 2
    c_coord = Ny // 2
    r_radius_coord = r_radius // dx
    c_radius_coord = c_radius // dy
    rr, cc = draw.ellipse(r_coord, c_coord, r_radius_coord, c_radius_coord)

    # Visualize pattern
    # img = np.zeros((Nx,Ny))
    # img[rr,cc] = 1
    # # print(img)
    # # print(img[17,238])
    # img = Image.fromarray(img*255)
    # img.show()

    # === Pattern Structure
    ER[rr, cc] = 1.  # ellipse hole in the middle
    # img_ER = Image.fromarray(ER * 255)
    # img_ER.show()
    # print(ER[256,256])

    return ER


def layerfunc_metallic_BIC(ER, params_geometry, params_mesh, layer_params):
    '''
    layer_params: [D1, D2, g1, g2, w]
    D1, D2: outter and inner diameter of the pattern.
    g1: main gap in the middle.
    g2: gap on two sides.
    w: width of the center strip
    '''
    D1, D2, g1, g2, w = layer_params[0], layer_params[1], layer_params[2], layer_params[3], layer_params[4]
    # ================= Geometry Params
    Lx = params_geometry[0]
    Ly = params_geometry[1]
    L = params_geometry[2]  # L=[d1,...,dn], thickness of layers

    r = Lx / 2
    c = Ly / 2

    # ================= Mesh Params
    Nx = params_mesh[0]
    Ny = params_mesh[1]
    dx = Lx / Nx
    dy = Ly / Ny

    # === Cross Sectional Grid
    d1_grid = D1 // dx
    d2_grid = D2 // dx
    # band_width_grid = (D1-D2) // (2*dx)
    band_width_grid = w // dx
    g1_grid = g1 // dx
    g2_grid = g2 // dx

    if D1 <= Lx:
        img_pattern = np.zeros(ER.shape)

        rr_disk_d1, cc_disk_d1 = draw.disk((Nx // 2, Ny // 2), d1_grid//2)
        rr_disk_d2, cc_disk_d2 = draw.disk((Nx // 2, Ny // 2), d2_grid//2)
        # print('rr_disk_d1', rr_disk_d1)
        # print('cc_disk_d1', cc_disk_d1)
        # print('rr_disk_d2', rr_disk_d2)
        # print('cc_disk_d2', cc_disk_d2)
        #
        # print('rr_disk_d1', rr_disk_d1.min())
        # print('cc_disk_d1', cc_disk_d1.min())
        # print('rr_disk_d2', rr_disk_d2.min())
        # print('cc_disk_d2', cc_disk_d2.min())
        #
        # print('rr_disk_d1', rr_disk_d1.shape)

        img_pattern[rr_disk_d1, cc_disk_d1] = 1
        img_pattern[rr_disk_d2, cc_disk_d2] = 0  # finish circle pattern
    else:  # D1>Lx
        img_pattern = np.zeros((int(d1_grid), int(d1_grid)))

        rr_disk_d1, cc_disk_d1 = draw.disk((d1_grid // 2, d1_grid // 2), d1_grid // 2)
        rr_disk_d2, cc_disk_d2 = draw.disk((d1_grid // 2, d1_grid // 2), d2_grid // 2)

        img_pattern[rr_disk_d1, cc_disk_d1] = 1
        img_pattern[rr_disk_d2, cc_disk_d2] = 0  # finish circle pattern

        # central clip img pattern
        img_pattern = img_pattern[int(d1_grid // 2 - Nx // 2): int(d1_grid // 2 + Nx // 2),
                      int(d1_grid // 2 - Nx // 2): int(d1_grid // 2 + Nx // 2)]

    # print('img_pattern', img_pattern.shape)
    assert img_pattern.shape == ER.shape, 'img_pattern.shape != ER.shape !!!'

    img_pattern[int(Ny//2-d2_grid//2) : int(Ny//2+d2_grid//2),
                int(Nx//2-band_width_grid//2) : int(Nx//2+band_width_grid//2)] = 1  # add main pillar

    img_pattern[int(Ny // 2 - g2_grid // 2): int(Ny // 2 + g2_grid // 2), : int((Nx - band_width_grid) // 2)] = 0
    img_pattern[int(Ny // 2 - g2_grid // 2): int(Ny // 2 + g2_grid // 2), -int((Nx - band_width_grid) // 2):] = 0  # finish g2 gap
    img_pattern[int(Ny//2-g1_grid//2) : int(Ny//2+g1_grid//2),
                int(Nx//2-band_width_grid//2) : int(Nx//2+band_width_grid//2)] = 0  # finish g1 gap

    # idx_pattern = np.argwhere(img_pattern)

    # Visualize pattern
    img_ER = np.ones(ER.shape)
    # img_ER[idx_pattern[:, 0], idx_pattern[:, 1]] = 0
    img_ER[img_pattern == 1] = 0

    # plt.figure(1)
    # plt.imshow(img_pattern * 255, cmap='gray')
    # plt.figure(2)
    # plt.imshow(img_ER * 255, cmap='gray')
    # plt.show()

    # === Pattern Structure
    ER[img_pattern == 0] = 1.  # metallic BIC pattern
    return ER


def layerfunc_diatom(ER, params_geometry, params_mesh, layer_params):
    '''
    layer_params: [p, d]
    p: central distance of neighboring cylinders.
    d: didmeter of each cylinder.
    '''
    p, d = layer_params[0], layer_params[1]
    # ================= Geometry Params
    Lx = params_geometry[0]
    Ly = params_geometry[1]
    L = params_geometry[2]  # L=[d1,...,dn], thickness of layers

    a = Lx / 2
    b = Ly / 2

    # ================= Mesh Params
    Nx = params_mesh[0]
    Ny = params_mesh[1]
    dx = Lx / Nx
    dy = Ly / Ny

    # === Cross Sectional Grid
    # p_grid = p // dx
    d_grid = d // dx
    # a_grid = a // dx
    # b_grid = b // dx

    # diatom pattern
    m = p / 2
    n = 3**0.5 / 2 * p
    c0 = (a, b)
    c1 = (a + p, b)
    c2 = (a - p, b)
    c3 = (a +  m, b + n)
    c4 = (a + m, b - n)
    c5 = (a - m, b - n)
    c6 = (a - m, b + n)

    c7 = (Lx, Ly)
    c8 = (Lx - p, Ly)
    c9 = (Lx - m, Ly - n)

    c10 = (Lx, 0)
    c11 = (Lx - p, 0)
    c12 = (Lx - m, n)

    c13 = (0, 0)
    c14 = (p, 0)
    c15 = (m, n)

    c16 = (0, Ly)
    c17 = (p, Ly)
    c18 = (m, Ly - n)

    c_list = [c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18]
    img_pattern = np.zeros(ER.shape)
    for c in c_list:
        c_grid = tuple(ci//dx for ci in c)
        rr_c, cc_c = draw.disk(c_grid, d_grid // 2, shape=img_pattern.shape)
        img_pattern[rr_c, cc_c] = 1

    # Visualize pattern
    # plt.figure(1)
    # plt.imshow(img_pattern * 255, cmap='gray')
    # # plt.figure(2)
    # # plt.imshow(img_ER * 255, cmap='gray')
    # plt.show()

    # === Pattern Structure
    ER[img_pattern == 1] = 1.  # diatom pattern
    return ER



class Material:
    def __init__(self, freq, params_eps, params_geometry, params_mesh, PQ_order, list_layer_funcs, list_layer_params, source, device='cpu', use_logger=True):
        '''
        freq: numpy array of frequencies to solve rcwa, (N_freq,).
        params_eps: list of eps for all layers, each entry in the list is shape of (N_freq,).
        params_geometry: [Lx,Ly,[d1,...,dn]], 2D geometry params and thickness for all layers.
        params_mesh: [Nx,Ny], mesh number for 2D geometry.
        PQ_order: a list of [PQ_x, PQ_y], each entry should be a singular value.
        list_layer_funcs: a list of functions [f1,...,fn] applied to each layer to define patterns inside each layer,
                          deleting materials.
        list_layer_params: a list of params [[l1_p1,...,l1_pm1],...,[ln_p1,...,ln_pmn]], each entry is also a list for
                          the i th layer corresponding to the "list_layer_funcs". They should already contain units
                          inside (eg, millimeters, micrometres, etc).
        source: source of incident light, a list, [ginc, EP], each entry (ginc, EP) is also a list, both ginc and EP
                should be a unit vector.
        device: 'cpu' for CPU using numpy; 'gpu' or 'cuda' for GPU using cupy.
        use_logger: printing solving progress percentage.
        '''
        self.freq = freq
        self.params_eps = params_eps
        self.params_geometry = params_geometry
        self.params_mesh = params_mesh
        self.PQ_order = PQ_order
        self.list_layer_funcs = list_layer_funcs
        self.list_layer_params = list_layer_params
        self.source = source
        self.device = device
        self.use_logger = use_logger

        self.PQ = 1 * np.array(PQ_order)  # this must be singular value

        # layers alignment check
        if len(params_eps) == len(params_geometry[2]) and len(params_eps) == len(list_layer_funcs):
            self.num_layers = len(params_eps)
        else:
            raise ValueError('Make sure layers for \'params_eps\' and \'params_geometry\' are the same!')

        # ================= Geometry Params
        self.Lx = params_geometry[0]
        self.Ly = params_geometry[1]
        self.L = params_geometry[2]  # L=[d1,...,dn], thickness of layers

        # ================= Mesh Params
        self.Nx = params_mesh[0]
        self.Ny = params_mesh[1]

    def rcwa_preprocess(self, params_eps_ifreq):
        '''
        Preprocessing for each freqency.
        '''
        # === Device Params
        # urd = 1.  # permeability of device
        # erd = np.conjugate(eps)  # permeability of device
        list_urd = self.num_layers * [1.]  # list of urd for all layers
        list_erd = np.conjugate(params_eps_ifreq)  # list of erd for all layers at i the freq
        # print(list_erd[0][0])
        # list_erd = np.conjugate([11.13167286+2.058688402j])  # list of erd for all layers

        Lx = self.Lx  # period along x
        Ly = self.Ly  # period along y

        # === RCWA Params
        Nx = self.Nx  # 512
        Ny = np.round(Nx * Ly / Lx).astype(int)
        PQ = self.PQ  # this must be singular value

        # === Define Structure in Layers
        nxc = np.floor(Nx / 2)
        nyc = np.floor(Ny / 2)

        # === Apply Pattern Functions for All Layers (Deleting Materials)
        ER = []
        UR = []
        for idx_layer in range(self.num_layers):
            ERi = list_erd[idx_layer] * np.ones((Nx, Ny))
            ERi = self.list_layer_funcs[idx_layer](ERi, self.params_geometry, self.params_mesh, self.list_layer_params[idx_layer])  # add geometry pattern
            ERi = ERi[..., np.newaxis]
            ER.append(ERi)

            URi = list_urd[idx_layer] * np.ones((Nx, Ny))
            URi = URi[..., np.newaxis]
            UR.append(URi)
        ER = np.concatenate(ER, axis=-1)
        UR = np.concatenate(UR, axis=-1)

        mm, nn, ll = ER.shape
        for i_ll in range(ll):
            URC_i = calc_utils.convmat(UR[..., i_ll][..., np.newaxis], PQ[0], PQ[1])
            ERC_i = calc_utils.convmat(ER[..., i_ll][..., np.newaxis], PQ[0], PQ[1])
            # print('ERC_i',ERC_i)
            # print('ERC_i', ERC_i.shape)
            if i_ll == 0:
                URC = URC_i[..., np.newaxis]
                ERC = ERC_i[..., np.newaxis]
            else:
                URC = np.concatenate((URC, URC_i[..., np.newaxis]), axis=-1)
                ERC = np.concatenate((ERC, ERC_i[..., np.newaxis]), axis=-1)

        return ER, UR, ERC, URC

    def rcwa_solve(self):
        use_logger = self.use_logger
        # ================= Calculation Start
        freq = self.freq
        R_total = np.zeros((len(self.freq),))
        T_total = np.zeros((len(self.freq),))

        # Physics environment define
        ur1 = 1.  # permeability in reflection region
        er1 = 1.  # permeability in reflection region
        ur2 = 1.  # permeability in transmission region
        er2 = 1.  # permeability in transmission region
        PQ = self.PQ  # this must be singular value
        # ginc = np.array([0, 0, 1])  # incident light source
        # EP = np.array([0, 1, 0])  # orig: [0,1,0], source polarization, must be a unit vector
        ginc = np.array(self.source[0])  # orig: [0,0,1], incident source, must be a unit vector
        EP = np.array(self.source[1])  # orig: [0,1,0], source polarization, must be a unit vector

        # Geometry structure define
        Lx = self.Lx  # period along x
        Ly = self.Ly  # period along y
        L = self.L

        if self.device == 'gpu' or self.device == 'cuda':  # GPU solver
            for i_freq in range(len(freq)):
                lam0 = c0 / freq[i_freq]
                # ====== RCWA Preprocessing ======
                # ER, UR, ERC, URC = rcwa_preprocess_Si_test(eps_Si[i_freq],
                #                                            L_param=L_param, w_param=w_param, t_param=t_param,
                #                                            PQ_order=PQ_order)

                params_eps_ifreq = []
                for ele_params_eps in self.params_eps:
                    params_eps_ifreq.append(ele_params_eps[i_freq])
                ER, UR, ERC, URC = self.rcwa_preprocess(params_eps_ifreq)
                # ER, UR, ERC, URC = cp.asarray(ER), cp.asarray(UR), cp.asarray(ERC), cp.asarray(URC)

                # ====== Wave Vector Expansion ======
                nr1 = np.sqrt(ur1 * er1)  # refractive index of reflection medium, eta
                # nr2 = np.sqrt(ur2 * er2)  # refractive index of transmission medium, eta
                k0 = 2 * pi / lam0
                p = np.arange(-(np.floor(PQ[0] / 2)), (np.floor(PQ[0] / 2) + 1))
                q = np.arange(-(np.floor(PQ[1] / 2)), (np.floor(PQ[1] / 2) + 1))
                kx_inc = ginc[0] / LA.norm(ginc) * k0 * nr1  # normal incidence
                ky_inc = ginc[1] / LA.norm(ginc) * k0 * nr1
                kz_inc = ginc[2] / LA.norm(ginc) * k0 * nr1

                KX = (kx_inc - 2 * pi * p / Lx) / k0
                KY = (ky_inc - 2 * pi * q / Ly) / k0

                KY_temp, KX_temp = np.meshgrid(KY, KX)
                KX = np.diag(KX_temp.T.flatten())  # p now is not a flooat, it's a vector, No squeeze
                KY = np.diag(KY_temp.T.flatten())

                if (1 in KX + KY):  # prevent from singular matrix
                    print('\n[WARNING] SINGULAR MATRIX!!! freq:', freq[i_freq], 'i_freq', i_freq)
                    R_total[i_freq] = R_total[i_freq - 1]
                    T_total[i_freq] = T_total[i_freq - 1]
                    continue

                # normalized reflection Kz, no minus sign ahead
                KZr = np.conj(np.sqrt((er1 * ur1 * np.eye(KX.shape[0]) - KX ** 2 - KY ** 2 + 0j).astype(complex)))
                # normalized transmission Kz
                KZt = np.conj(np.sqrt((er2 * ur2 * np.eye(KY.shape[0]) - KX ** 2 - KY ** 2 + 0j).astype(complex)))

                kz_inc = kz_inc / k0

                # === Compute Eigen-modes of Free Space
                KZ = np.conj(np.sqrt((np.eye(KX.shape[0]) - KX ** 2 - KY ** 2 + 0j).astype(complex)))

                P = np.block([[KX @ KY, np.eye(KX.shape[0]) - KX ** 2],
                              [KY ** 2 - np.eye(KX.shape[0]), -KX @ KY]])

                Q = P

                W0 = np.block([[np.eye(KX.shape[0]), np.zeros((KX.shape[0], KX.shape[1]))],
                               [np.zeros((KX.shape[0], KX.shape[1])), np.eye(KX.shape[0])]])
                lam = 1j * np.block([[KZ, np.zeros((KZ.shape[0], KZ.shape[1]))],
                                     [np.zeros((KZ.shape[0], KZ.shape[1])), KZ]])

                Q, lam, W0 = cp.asarray(Q), cp.asarray(lam), cp.asarray(W0)
                V0 = Q @ CLA.inv(lam)
                # KX, KY = cp.asarray(KX), cp.asarray(KY)

                # === Initialize Device Scattering Matrix
                S11 = np.zeros((P.shape[0], P.shape[1]))
                S12 = np.eye(P.shape[0])
                S21 = S12
                S22 = S11
                S11, S12, S21, S22 = cp.asarray(S11), cp.asarray(S12), cp.asarray(S21), cp.asarray(S22)
                SG = {'S11': S11, 'S12': S12, 'S21': S21, 'S22': S22}

                # === Main Loop
                uu, vv, ww = ER.shape
                for ii in range(ww):
                    P_ii = np.block([[KX @ LA.inv(ERC[:, :, ii]) @ KY, URC[:, :, ii] - KX @ LA.inv(ERC[:, :, ii]) @ KX],
                                     [KY @ LA.inv(ERC[:, :, ii]) @ KY - URC[:, :, ii],
                                      -KY @ LA.inv(ERC[:, :, ii]) @ KX]])
                    Q_ii = np.block([[KX @ LA.inv(URC[:, :, ii]) @ KY, ERC[:, :, ii] - KX @ LA.inv(URC[:, :, ii]) @ KX],
                                     [KY @ LA.inv(URC[:, :, ii]) @ KY - ERC[:, :, ii],
                                      -KY @ LA.inv(URC[:, :, ii]) @ KX]])

                    OMEGA_SQ_ii = P_ii @ Q_ii
                    [lam_sq_ii, W_ii] = LA.eig(OMEGA_SQ_ii)
                    # [lam_sq_ii, W_ii] = SLA.eig(OMEGA_SQ_ii,left=False, right=True)
                    # lam_sq_ii is the same as matlab, W_ii is different
                    lam_sq_ii = np.diag(lam_sq_ii)
                    lam_ii = np.sqrt(lam_sq_ii)

                    Q_ii, W_ii, lam_ii = cp.asarray(Q_ii), cp.asarray(W_ii), cp.asarray(lam_ii)
                    V_ii = Q_ii @ W_ii @ LA.inv(lam_ii)
                    A0_ii = CLA.inv(W_ii) @ W0 + CLA.inv(V_ii) @ V0
                    B0_ii = CLA.inv(W_ii) @ W0 - CLA.inv(V_ii) @ V0

                    X_ii = SLA.expm(-cp.asnumpy(lam_ii) * k0 * L[ii])
                    # X_ii = SLA.expm(float(-lam_ii * k0 * L[ii]))

                    X_ii = cp.asarray(X_ii)
                    S11 = CLA.inv(A0_ii - X_ii @ B0_ii @ LA.inv(A0_ii) @ X_ii @ B0_ii) @ \
                          (X_ii @ B0_ii @ LA.inv(A0_ii) @ X_ii @ A0_ii - B0_ii)
                    S12 = CLA.inv(A0_ii - X_ii @ B0_ii @ LA.inv(A0_ii) @ X_ii @ B0_ii) @ \
                          (X_ii @ (A0_ii - B0_ii @ LA.inv(A0_ii) @ B0_ii))
                    S21 = S12
                    S22 = S11
                    S = {'S11': S11, 'S12': S12, 'S21': S21, 'S22': S22}
                    SG = calc_utils.star(SG, S, device='gpu')

                # === Compute Reflection Side Connection S-Matrix
                Q_ref = ur1 ** (-1) * np.block([[KX @ KY, ur1 * er1 * np.eye(KX.shape[0]) - KX ** 2],
                                                [KY ** 2 - ur1 * er1 * np.eye(KY.shape[0]), -KY @ KX]])

                W_ref = np.block([[np.eye(KX.shape[0]), np.zeros((KX.shape[0], KX.shape[1]))],
                                  [np.zeros((KY.shape[0], KY.shape[1])), np.eye(KY.shape[0])]])

                lam_ref = np.block([[1j * KZr, np.zeros((KZr.shape[0], KZr.shape[1]))],
                                    [np.zeros((KZr.shape[0], KZr.shape[1])), 1j * KZr]])

                Q_ref, lam_ref = cp.asarray(Q_ref), cp.asarray(lam_ref)
                V_ref = Q_ref @ CLA.inv(lam_ref)

                Ar = CLA.inv(W0) @ cp.asarray(W_ref) + CLA.inv(V0) @ V_ref
                Br = CLA.inv(W0) @ cp.asarray(W_ref) - CLA.inv(V0) @ V_ref

                S11 = -CLA.inv(Ar) @ Br
                S12 = 2 * CLA.inv(Ar)
                S21 = 0.5 * (Ar - Br @ CLA.inv(Ar) @ Br)
                S22 = Br @ CLA.inv(Ar)
                Sref = {'S11': S11, 'S12': S12, 'S21': S21, 'S22': S22}

                # === Compute Transmission Side Connection S-Matrix
                Q_trn = ur2 ** (-1) * np.block([[KX @ KY, ur2 * er2 * np.eye(KX.shape[0]) - KX ** 2],
                                                [KY ** 2 - ur2 * er2 * np.eye(KY.shape[0]), -KY @ KX]])

                W_trn = np.block([[np.eye(KX.shape[0]), np.zeros((KX.shape[0], KX.shape[1]))],
                                  [np.zeros((KY.shape[0], KY.shape[1])), np.eye(KY.shape[0])]])

                lam_trn = np.block([[1j * KZt, np.zeros((KZt.shape[0], KZt.shape[1]))],
                                    [np.zeros((KZt.shape[0], KZt.shape[1])), 1j * KZt]])

                V_trn = cp.asarray(Q_trn) @ CLA.inv(cp.asarray(lam_trn))
                At = LA.inv(W0) @ cp.asarray(W_trn) + CLA.inv(V0) @ V_trn
                Bt = LA.inv(W0) @ cp.asarray(W_trn) - CLA.inv(V0) @ V_trn

                S11 = Bt @ CLA.inv(At)
                S12 = 0.5 * (At - Bt @ CLA.inv(At) @ Bt)
                S21 = 2 * CLA.inv(At)
                S22 = -CLA.inv(At) @ Bt
                Strn = {'S11': S11, 'S12': S12, 'S21': S21, 'S22': S22}

                SG = calc_utils.star(Sref, SG, device='gpu')
                SG = calc_utils.star(SG, Strn, device='gpu')

                # === Compute Reflected and Transmitted Fields
                delta = np.concatenate(([np.zeros((1, np.floor(PQ[0] * PQ[1] / 2).astype(int))),
                                         np.array([[1]]),
                                         np.zeros((1, np.floor(PQ[0] * PQ[1] / 2).astype(int)))]), axis=-1).T

                esrc = np.concatenate((EP[0] * delta, EP[1] * delta), axis=0)
                csrc = CLA.inv(cp.asarray(W_ref)) @ cp.asarray(esrc)

                cref = SG['S11'] @ csrc
                ctrn = SG['S21'] @ csrc

                rall = cp.asarray(W_ref) @ cref
                tall = cp.asarray(W_trn) @ ctrn

                nExp = rall.shape[0]
                rx = rall[:int(nExp / 2)]
                ry = rall[int(nExp / 2):]

                tx = tall[:int(nExp / 2)]
                ty = tall[int(nExp / 2):]

                rz = -CLA.inv(cp.asarray(KZr)) @ (cp.asarray(KX) @ rx + cp.asarray(KY) @ ry)
                tz = -CLA.inv(cp.asarray(KZt)) @ (cp.asarray(KX) @ tx + cp.asarray(KY) @ ty)

                R_ref = np.real(-KZr / kz_inc) * (
                        np.abs(cp.asnumpy(rx)) ** 2 + np.abs(cp.asnumpy(ry)) ** 2 + np.abs(cp.asnumpy(rz)) ** 2)
                R_total[i_freq] = np.sum(np.abs(R_ref))

                T_ref = np.real(ur1 / ur2 * KZr / kz_inc) * (
                        np.abs(cp.asnumpy(tx)) ** 2 + np.abs(cp.asnumpy(ty)) ** 2 + np.abs(cp.asnumpy(tz)) ** 2)
                T_total[i_freq] = np.sum(np.abs(T_ref))

                if use_logger:
                    # === print a progressbar
                    sys.stdout.write('\r')
                    sys.stdout.write("Calculation Progress: %d%%" % ((100 / len(freq)) * (i_freq + 1)))
                    sys.stdout.flush()
        else:  # CPU solver
            for i_freq in range(len(freq)):
                lam0 = c0 / freq[i_freq]
                # ====== RCWA Preprocessing ======
                # ER, UR, ERC, URC = rcwa_preprocess_Si_test(eps_Si[i_freq],
                #                                            L_param=L_param, w_param=w_param, t_param=t_param,
                #                                            PQ_order=PQ_order)

                params_eps_ifreq = []
                for ele_params_eps in self.params_eps:
                    params_eps_ifreq.append(ele_params_eps[i_freq])
                ER, UR, ERC, URC = self.rcwa_preprocess(params_eps_ifreq)

                # ====== Wave Vector Expansion ======
                nr1 = np.sqrt(ur1 * er1)  # refractive index of reflection medium, eta
                # nr2 = np.sqrt(ur2 * er2)  # refractive index of transmission medium, eta
                k0 = 2 * pi / lam0
                p = np.arange(-(np.floor(PQ[0] / 2)), (np.floor(PQ[0] / 2) + 1))
                q = np.arange(-(np.floor(PQ[1] / 2)), (np.floor(PQ[1] / 2) + 1))
                kx_inc = ginc[0] / LA.norm(ginc) * k0 * nr1  # normal incidence
                ky_inc = ginc[1] / LA.norm(ginc) * k0 * nr1
                kz_inc = ginc[2] / LA.norm(ginc) * k0 * nr1

                KX = (kx_inc - 2 * pi * p / Lx) / k0
                KY = (ky_inc - 2 * pi * q / Ly) / k0

                KY_temp, KX_temp = np.meshgrid(KY, KX)
                KX = np.diag(KX_temp.T.flatten())  # p now is not a flooat, it's a vector, No squeeze
                KY = np.diag(KY_temp.T.flatten())

                if (1 in KX + KY):  # prevent from singular matrix
                    print('\n[WARNING] SINGULAR MATRIX!!! freq:', freq[i_freq], 'i_freq', i_freq)
                    R_total[i_freq] = R_total[i_freq - 1]
                    T_total[i_freq] = T_total[i_freq - 1]
                    continue

                # normalized reflection Kz, no minus sign ahead
                KZr = np.conj(np.sqrt((er1 * ur1 * np.eye(KX.shape[0]) - KX ** 2 - KY ** 2 + 0j).astype(complex)))
                # normalized transmission Kz
                KZt = np.conj(np.sqrt((er2 * ur2 * np.eye(KY.shape[0]) - KX ** 2 - KY ** 2 + 0j).astype(complex)))

                kz_inc = kz_inc / k0

                # === Compute Eigen-modes of Free Space
                KZ = np.conj(np.sqrt((np.eye(KX.shape[0]) - KX ** 2 - KY ** 2 + 0j).astype(complex)))

                P = np.block([[KX @ KY, np.eye(KX.shape[0]) - KX ** 2],
                              [KY ** 2 - np.eye(KX.shape[0]), -KX @ KY]])

                Q = P

                W0 = np.block([[np.eye(KX.shape[0]), np.zeros((KX.shape[0], KX.shape[1]))],
                               [np.zeros((KX.shape[0], KX.shape[1])), np.eye(KX.shape[0])]])
                lam = 1j * np.block([[KZ, np.zeros((KZ.shape[0], KZ.shape[1]))],
                                     [np.zeros((KZ.shape[0], KZ.shape[1])), KZ]])

                V0 = Q @ LA.inv(lam)

                # === Initialize Device Scattering Matrix
                S11 = np.zeros((P.shape[0], P.shape[1]))
                S12 = np.eye(P.shape[0])
                S21 = S12
                S22 = S11
                SG = {'S11': S11, 'S12': S12, 'S21': S21, 'S22': S22}

                # === Main Loop
                uu, vv, ww = ER.shape
                for ii in range(ww):
                    P_ii = np.block([[KX @ LA.inv(ERC[:, :, ii]) @ KY, URC[:, :, ii] - KX @ LA.inv(ERC[:, :, ii]) @ KX],
                                     [KY @ LA.inv(ERC[:, :, ii]) @ KY - URC[:, :, ii],
                                      -KY @ LA.inv(ERC[:, :, ii]) @ KX]])
                    Q_ii = np.block([[KX @ LA.inv(URC[:, :, ii]) @ KY, ERC[:, :, ii] - KX @ LA.inv(URC[:, :, ii]) @ KX],
                                     [KY @ LA.inv(URC[:, :, ii]) @ KY - ERC[:, :, ii],
                                      -KY @ LA.inv(URC[:, :, ii]) @ KX]])
                    OMEGA_SQ_ii = P_ii @ Q_ii
                    [lam_sq_ii, W_ii] = LA.eig(OMEGA_SQ_ii)
                    # [lam_sq_ii, W_ii] = SLA.eig(OMEGA_SQ_ii,left=False, right=True)
                    # lam_sq_ii is the same as matlab, W_ii is different
                    lam_sq_ii = np.diag(lam_sq_ii)
                    lam_ii = np.sqrt(lam_sq_ii)
                    V_ii = Q_ii @ W_ii @ LA.inv(lam_ii)
                    A0_ii = LA.inv(W_ii) @ W0 + LA.inv(V_ii) @ V0
                    B0_ii = LA.inv(W_ii) @ W0 - LA.inv(V_ii) @ V0

                    X_ii = SLA.expm(-lam_ii * k0 * L[ii])
                    # X_ii = SLA.expm(float(-lam_ii * k0 * L[ii]))

                    S11 = LA.inv(A0_ii - X_ii @ B0_ii @ LA.inv(A0_ii) @ X_ii @ B0_ii) @ \
                          (X_ii @ B0_ii @ LA.inv(A0_ii) @ X_ii @ A0_ii - B0_ii)
                    S12 = LA.inv(A0_ii - X_ii @ B0_ii @ LA.inv(A0_ii) @ X_ii @ B0_ii) @ \
                          (X_ii @ (A0_ii - B0_ii @ LA.inv(A0_ii) @ B0_ii))
                    S21 = S12
                    S22 = S11
                    S = {'S11': S11, 'S12': S12, 'S21': S21, 'S22': S22}
                    SG = calc_utils.star(SG, S)

                # === Compute Reflection Side Connection S-Matrix
                Q_ref = ur1 ** (-1) * np.block([[KX @ KY, ur1 * er1 * np.eye(KX.shape[0]) - KX ** 2],
                                                [KY ** 2 - ur1 * er1 * np.eye(KY.shape[0]), -KY @ KX]])

                W_ref = np.block([[np.eye(KX.shape[0]), np.zeros((KX.shape[0], KX.shape[1]))],
                                  [np.zeros((KY.shape[0], KY.shape[1])), np.eye(KY.shape[0])]])

                lam_ref = np.block([[1j * KZr, np.zeros((KZr.shape[0], KZr.shape[1]))],
                                    [np.zeros((KZr.shape[0], KZr.shape[1])), 1j * KZr]])

                V_ref = Q_ref @ LA.inv(lam_ref)
                Ar = LA.inv(W0) @ W_ref + LA.inv(V0) @ V_ref
                Br = LA.inv(W0) @ W_ref - LA.inv(V0) @ V_ref

                S11 = -LA.inv(Ar) @ Br
                S12 = 2 * LA.inv(Ar)
                S21 = 0.5 * (Ar - Br @ LA.inv(Ar) @ Br)
                S22 = Br @ LA.inv(Ar)
                Sref = {'S11': S11, 'S12': S12, 'S21': S21, 'S22': S22}

                # === Compute Transmission Side Connection S-Matrix
                Q_trn = ur2 ** (-1) * np.block([[KX @ KY, ur2 * er2 * np.eye(KX.shape[0]) - KX ** 2],
                                                [KY ** 2 - ur2 * er2 * np.eye(KY.shape[0]), -KY @ KX]])

                W_trn = np.block([[np.eye(KX.shape[0]), np.zeros((KX.shape[0], KX.shape[1]))],
                                  [np.zeros((KY.shape[0], KY.shape[1])), np.eye(KY.shape[0])]])

                lam_trn = np.block([[1j * KZt, np.zeros((KZt.shape[0], KZt.shape[1]))],
                                    [np.zeros((KZt.shape[0], KZt.shape[1])), 1j * KZt]])

                V_trn = Q_trn @ LA.inv(lam_trn)
                At = LA.inv(W0) @ W_trn + LA.inv(V0) @ V_trn
                Bt = LA.inv(W0) @ W_trn - LA.inv(V0) @ V_trn

                S11 = Bt @ LA.inv(At)
                S12 = 0.5 * (At - Bt @ LA.inv(At) @ Bt)
                S21 = 2 * LA.inv(At)
                S22 = -LA.inv(At) @ Bt
                Strn = {'S11': S11, 'S12': S12, 'S21': S21, 'S22': S22}

                SG = calc_utils.star(Sref, SG)
                SG = calc_utils.star(SG, Strn)

                # === Compute Reflected and Transmitted Fields
                delta = np.concatenate(([np.zeros((1, np.floor(PQ[0] * PQ[1] / 2).astype(int))),
                                         np.array([[1]]),
                                         np.zeros((1, np.floor(PQ[0] * PQ[1] / 2).astype(int)))]), axis=-1).T

                esrc = np.concatenate((EP[0] * delta, EP[1] * delta), axis=0)
                csrc = LA.inv(W_ref) @ esrc

                cref = SG['S11'] @ csrc
                ctrn = SG['S21'] @ csrc

                rall = W_ref @ cref
                tall = W_trn @ ctrn

                nExp = rall.shape[0]
                rx = rall[:int(nExp / 2)]
                ry = rall[int(nExp / 2):]

                tx = tall[:int(nExp / 2)]
                ty = tall[int(nExp / 2):]

                rz = -LA.inv(KZr) @ (KX @ rx + KY @ ry)
                tz = -LA.inv(KZt) @ (KX @ tx + KY @ ty)

                R_ref = np.real(-KZr / kz_inc) * (np.abs(rx) ** 2 + np.abs(ry) ** 2 + np.abs(rz) ** 2)
                R_total[i_freq] = np.sum(np.abs(R_ref))

                T_ref = np.real(ur1 / ur2 * KZr / kz_inc) * (np.abs(tx) ** 2 + np.abs(ty) ** 2 + np.abs(tz) ** 2)
                T_total[i_freq] = np.sum(np.abs(T_ref))

                if use_logger:
                    # === print a progressbar
                    sys.stdout.write('\r')
                    sys.stdout.write("Calculation Progress: %d%%" % ((100 / len(freq)) * (i_freq + 1)))
                    sys.stdout.flush()

        sys.stdout.write('\n')
        return R_total, T_total
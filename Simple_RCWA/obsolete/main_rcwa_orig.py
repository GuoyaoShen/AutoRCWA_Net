import numpy as np
from numpy import linalg as LA
from scipy import linalg as SLA
import cmath
import matplotlib
import matplotlib.pyplot as plt

from utils import data_utils
from utils import calc_utils

pi = np.pi

# ================= Unit Define
meters = 1
centimeters = 1e-2 * meters
millimeters = 1e-3 * meters

# ================= Constant Define
c0 = 3e8
e0 = 8.85e-12
u0 = 1.256e-6
yeta0 = np.sqrt(u0/e0)
# print('yeta0',yeta0)

path_gold = 'Au3-Drude.txt'
eps_gold_file = data_utils.load_property_txt(path_gold,2743,3636)
# print('eps_gold',eps_gold.shape)
eps_gold = eps_gold_file[:,1] + eps_gold_file[:,2]*1j
# print('eps_gold',eps_gold)

path_SiNx = 'SiNx_property.mat'
eps_SiNx = data_utils.load_property_mat(path_SiNx)
# print('eps_SiNx REAL',eps_SiNx['eps_SiNx_real'])
# print('eps_SiNx IMAG',eps_SiNx['eps_SiNx_imag'])
eps_SiNx = eps_SiNx['eps_SiNx_real'] + eps_SiNx['eps_SiNx_imag']*1j
# print('eps_SiNx',eps_SiNx.shape)

freq = eps_gold_file[:,0]*1e12
# print(len(freq))
# print(freq.shape[0])

# plot property
n_SiNx = np.sqrt(eps_SiNx)
# print(n_SiNx)
# plt.plot(freq,np.real(n_SiNx))
# plt.plot(freq,np.imag(n_SiNx))
# plt.show()


# ================= Calculation Start
R_total = np.zeros((len(freq),))
T_total = np.zeros((len(freq),))
for i_freq in range(len(freq)):
    lam0 = c0/freq[i_freq]
    # print(lam0)
    ginc = np.array([0,0,1])
    # print(ginc)
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
    w = 0.52 * Ly  # length of one side of square

    # === RCWA Params
    Nx = 512
    Ny = np.round(Nx*Ly/Lx).astype(int)
    PQ = 1 * np.array([1, 31])

    # === Define Structure in Layers
    nxc = np.floor(Nx/2)
    nyc = np.floor(Ny/2)
    ER1 = 1 * np.ones((Nx,Ny))
    ER2 = erd * np.ones((Nx,Ny))
    ER3 = np.conjugate(eps_gold[i_freq]) * np.ones((Nx,Ny))
    ER = np.concatenate([ER1[...,np.newaxis],ER2[...,np.newaxis],ER3[...,np.newaxis]],axis=-1)  # [512,512,3]
    # print(ER)
    # print(ER.shape)
    UR1 = urd * np.ones((Nx,Ny))
    UR2 = urd * np.ones((Nx,Ny))
    UR3 = urd * np.ones((Nx,Ny))
    UR = np.concatenate([UR1[...,np.newaxis],UR2[...,np.newaxis],UR3[...,np.newaxis]],axis=-1)  # [512,512,3]
    # print(UR.shape)
    L = np.array([d1,d2,d3])

    # === Cross Sectional Grid
    dx = Lx/Nx  # grid resolution along x
    dy = Ly/Ny  # grid resolution along y
    xa = np.arange(Nx) * dx  # x axis array
    xa = xa - np.mean(xa)  # center x axis at zero
    ya = np.arange(Ny) * dy  # y axis array
    ya = ya - np.mean(ya)  # center y axis at zero
    x_axis, y_axis = np.meshgrid(xa,ya)

    ny1 = np.round(nxc-((w/Ly)*Nx)/2).astype(int)
    ny2 = np.round(nxc+((w/Ly)*Nx)/2).astype(int)
    ER[ny1-1:ny2,ny1-1:ny2,0] = np.conjugate(eps_gold[i_freq])
    mm, nn, ll = ER.shape
    # print(PQ[0])
    for i_ll in range(ll):
        URC_i = calc_utils.convmat(UR[...,i_ll][...,np.newaxis], PQ[0], PQ[1])
        # print('URC_i',URC_i)
        # print('URC_i', URC_i.shape)
        ERC_i = calc_utils.convmat(ER[..., i_ll][..., np.newaxis], PQ[0], PQ[1])
        # print('ERC_i',ERC_i)
        # print('ERC_i', ERC_i.shape)
        if i_ll == 0:
            URC = URC_i[..., np.newaxis]
            ERC = ERC_i[..., np.newaxis]
        else:
            URC = np.concatenate((URC, URC_i[..., np.newaxis]), axis=-1)
            ERC = np.concatenate((ERC, ERC_i[..., np.newaxis]), axis=-1)

    # print(URC.shape)
    # print(ERC.shape)

    # ====== Wave Vector Expansion ======
    nr1 = np.sqrt(ur1 * er1)  #refractive index of reflection medium, eta
    nr2 = np.sqrt(ur2 * er2)  #refractive index of transmission medium, eta
    k0 = 2 * pi / lam0
    p = np.arange(-(np.floor(PQ[0] / 2)), (np.floor(PQ[0] / 2) + 1))
    q = np.arange(-(np.floor(PQ[1] / 2)), (np.floor(PQ[1] / 2) + 1))
    kx_inc = ginc[0] / LA.norm(ginc) * k0 * nr1  #normal incidence
    ky_inc = ginc[1] / LA.norm(ginc) * k0 * nr1
    kz_inc = ginc[2] / LA.norm(ginc) * k0 * nr1

    KX = (kx_inc - 2 * pi * p / Lx) / k0
    KY = (ky_inc - 2 * pi * q / Ly) / k0
    KY, KX = np.meshgrid(KY, KX)
    # print(KX.shape)
    KX = np.diag(KX.squeeze())
    KY = np.diag(KY.squeeze())
    # print(KX.shape)
    # print(KX)
    # print(KY)
    #normalized reflection Kz, no minus sign ahead
    KZr = np.conj(np.sqrt((er1 * ur1 * np.eye(KX.shape[0]) - KX ** 2 - KY ** 2 + 0j).astype(complex)))
    #normalized transmission Kz
    KZt = np.conj(np.sqrt((er2 * ur2 * np.eye(KY.shape[0]) - KX ** 2 - KY ** 2 + 0j).astype(complex)))
    # print('KZr',KZr.shape)
    # print('KZt',KZt.shape)
    kx_inc = kx_inc / k0
    ky_inc = ky_inc / k0
    kz_inc = kz_inc / k0

    # === Compute Eigen-modes of Free Space
    KZ = np.conj(np.sqrt((np.eye(KX.shape[0]) - KX ** 2 - KY ** 2 + 0j).astype(complex)))
    # print(KZ.shape)
    P = np.block([[KX@KY, np.eye(KX.shape[0])- KX**2],
                  [KY**2 - np.eye(KX.shape[0]), -KX@KY]])
    # print(P[31,0],P[32,1],P[61,30])
    Q = P
    OMEGA_SQ = P@Q
    W0 = np.block([[np.eye(KX.shape[0]), np.zeros((KX.shape[0],KX.shape[1]))],
                   [np.zeros((KX.shape[0],KX.shape[1])), np.eye(KX.shape[0])]])
    lam = 1j*np.block([[KZ, np.zeros((KZ.shape[0],KZ.shape[1]))],
                       [np.zeros((KZ.shape[0],KZ.shape[1])), KZ]])
    # print(lam)
    # print(lam.shape)
    V0 = Q@LA.inv(lam)
    # print(V0[31,0],V0[32,1])
    # print(V0.shape)

    # === Initialize Device Scattering Matrix
    S11 = np.zeros((P.shape[0],P.shape[1]))
    S12 = np.eye(P.shape[0])
    S21 = S12
    S22 = S11
    SG = {'S11':S11, 'S12':S12, 'S21':S21, 'S22':S22}
    # print(SG)

    # === Main Loop
    # print(ER.shape)
    uu, vv, ww = ER.shape
    for ii in range(ww):
        P_ii = np.block([[KX @ LA.inv(ERC[:, :, ii]) @ KY, URC[:, :, ii] - KX @ LA.inv(ERC[:, :, ii]) @ KX],
                         [KY @ LA.inv(ERC[:, :, ii]) @ KY - URC[:, :, ii], -KY @ LA.inv(ERC[:, :, ii]) @ KX]])
        Q_ii = np.block([[KX @ LA.inv(URC[:, :, ii]) @ KY, ERC[:, :, ii] - KX @ LA.inv(URC[:, :, ii]) @ KX],
                         [KY @ LA.inv(URC[:, :, ii]) @ KY - ERC[:, :, ii], -KY @ LA.inv(URC[:, :, ii]) @ KX]])
        OMEGA_SQ_ii = P_ii @ Q_ii
        # print(OMEGA_SQ_ii)
        [lam_sq_ii, W_ii] = LA.eig(OMEGA_SQ_ii)
        # [lam_sq_ii, W_ii] = SLA.eig(OMEGA_SQ_ii,left=False, right=True)
        # lam_sq_ii is the same as matlab, W_ii is different
        # print(lam_sq_ii)
        lam_sq_ii = np.diag(lam_sq_ii)
        # print(W_ii.shape)
        # print(lam_sq_ii.shape)
        # print(lam_sq_ii)
        # print(W_ii)
        lam_ii = np.sqrt(lam_sq_ii)
        # print(lam_ii)
        V_ii = Q_ii @ W_ii @ LA.inv(lam_ii)
        A0_ii = LA.inv(W_ii) @ W0 + LA.inv(V_ii) @ V0
        B0_ii = LA.inv(W_ii) @ W0 - LA.inv(V_ii) @ V0
        # print(A0_ii)
        # print(B0_ii)
        # print('*',np.max(OMEGA_SQ_ii @ W_ii - W_ii @ lam_sq_ii))
        # print(W_ii[:,0])
        # print(lam_sq_ii[0,0])
        # print(np.max(OMEGA_SQ_ii @ W_ii[:,0] - W_ii[:,0] * lam_sq_ii[0,0]))

        X_ii = SLA.expm(-lam_ii*k0*L[ii])
        # print(X_ii)
        S11 = LA.inv(A0_ii - X_ii @ B0_ii @ LA.inv(A0_ii) @ X_ii @ B0_ii) @ \
              (X_ii @ B0_ii @ LA.inv(A0_ii) @ X_ii @ A0_ii - B0_ii)
        S12 = LA.inv(A0_ii - X_ii @ B0_ii @ LA.inv(A0_ii) @ X_ii @ B0_ii) @ \
              (X_ii @ (A0_ii - B0_ii @ LA.inv(A0_ii) @ B0_ii))
        S21 = S12
        S22 = S11
        S = {'S11': S11, 'S12': S12, 'S21': S21, 'S22': S22}
        SG = calc_utils.star(SG,S)
        # print(SG['S11'])

    # === Compute Reflection Side Connection S-Matrix
    Q_ref = ur1 ** (-1) * np.block([[KX @ KY, ur1 * er1 * np.eye(KX.shape[0]) - KX ** 2],
                                    [KY ** 2 - ur1 * er1 * np.eye(KY.shape[0]), -KY @ KX]])
    # print(Q_ref[31,0],Q_ref[32,1])
    W_ref = np.block([[np.eye(KX.shape[0]), np.zeros((KX.shape[0], KX.shape[1]))],
                      [np.zeros((KY.shape[0], KY.shape[1])), np.eye(KY.shape[0])]])
    # print(W_ref)
    lam_ref = np.block([[1j*KZr, np.zeros((KZr.shape[0], KZr.shape[1]))],
                        [np.zeros((KZr.shape[0], KZr.shape[1])), 1j*KZr]])
    # print(lam_ref)
    V_ref = Q_ref @ LA.inv(lam_ref)
    Ar = LA.inv(W0) @ W_ref + LA.inv(V0) @ V_ref
    Br = LA.inv(W0) @ W_ref - LA.inv(V0) @ V_ref
    # print(Ar)
    # print(Br)
    S11 = -LA.inv(Ar) @ Br
    S12 = 2 * LA.inv(Ar)
    S21 = 0.5 * (Ar - Br@LA.inv(Ar)@Br)
    S22 = Br @ LA.inv(Ar)
    Sref = {'S11': S11, 'S12': S12, 'S21': S21, 'S22': S22}
    # print(S22)

    # === Compute Transmission Side Connection S-Matrix
    Q_trn = ur2 ** (-1) * np.block([[KX @ KY, ur2 * er2 * np.eye(KX.shape[0]) - KX ** 2],
                                    [KY ** 2 - ur2 * er2 * np.eye(KY.shape[0]), -KY @ KX]])
    # print(Q_ref[31,0],Q_ref[32,1])
    W_trn = np.block([[np.eye(KX.shape[0]), np.zeros((KX.shape[0], KX.shape[1]))],
                      [np.zeros((KY.shape[0], KY.shape[1])), np.eye(KY.shape[0])]])
    # print(W_ref)
    lam_trn = np.block([[1j * KZt, np.zeros((KZt.shape[0], KZt.shape[1]))],
                        [np.zeros((KZt.shape[0], KZt.shape[1])), 1j * KZt]])
    # print(lam_ref)
    V_trn = Q_trn @ LA.inv(lam_trn)
    At = LA.inv(W0) @ W_trn + LA.inv(V0) @ V_trn
    Bt = LA.inv(W0) @ W_trn - LA.inv(V0) @ V_trn
    # print(Ar)
    # print(Br)
    S11 = Bt @ LA.inv(At)
    S12 = 0.5 * (At - Bt @ LA.inv(At) @ Bt)
    S21 = 2 * LA.inv(At)
    S22 = -LA.inv(At) @ Bt
    Strn = {'S11': S11, 'S12': S12, 'S21': S21, 'S22': S22}
    # print(S22)

    SG = calc_utils.star(Sref, SG)
    SG = calc_utils.star(SG, Strn)
    # print(SG['S12'])

    # === Compute Reflected and Transmitted Fields
    delta = np.concatenate(([np.zeros((1,np.floor(PQ[0]*PQ[1]/2).astype(int))),
                            np.array([[1]]),
                            np.zeros((1,np.floor(PQ[0]*PQ[1]/2).astype(int)))]),axis=-1).T
    # print(delta.shape)
    ate = np.array([[0,1,0]]).T
    atm = np.array([[1,0,0]]).T

    esrc = np.concatenate((EP[0]*delta, EP[1]*delta), axis=0)
    # print(esrc)
    csrc = LA.inv(W_ref) @ esrc
    # print(csrc[46])

    cref = SG['S11'] @ csrc
    ctrn = SG['S21'] @ csrc
    # print(cref)
    # print(ctrn)

    rall = W_ref @ cref
    tall = W_trn @ ctrn
    # print(rall.shape)

    nExp = rall.shape[0]
    rx = rall[:int(nExp/2)]
    ry = rall[int(nExp/2):]
    # print(rx)
    # print(ry)
    tx = tall[:int(nExp / 2)]
    ty = tall[int(nExp / 2):]

    rz = -LA.inv(KZr) @ (KX @ rx + KY @ ry)
    tz = -LA.inv(KZt) @ (KX @ tx + KY @ ty)
    # print(rz)
    # print(tz)

    R_ref = np.real(-KZr / kz_inc) * (np.abs(rx) ** 2 + np.abs(ry) ** 2 + np.abs(rz) ** 2)
    R_total[i_freq] = np.sum(np.abs(R_ref))
    # print(np.sum(np.abs(R_ref)))

    T_ref = np.real(ur1/ur2*KZr / kz_inc) * (np.abs(tx) ** 2 + np.abs(ty) ** 2 + np.abs(tz) ** 2)
    T_total[i_freq] = np.sum(np.abs(T_ref))
    # print(np.sum(np.abs(T_ref)))

    print(i_freq, '||', len(freq))
    # if i_freq==100:
    #     break


plt.figure(1)
plt.plot(freq, R_total)
plt.figure(2)
plt.plot(freq, T_total)
plt.show()

path = 'fRT.npz'
np.savez(path, freq=freq, R=R_total, T=T_total)
print('FILE SAVED')
pass
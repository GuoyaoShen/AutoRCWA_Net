import numpy as np
from numpy import linalg as LA
from scipy import linalg as SLA
import cmath
import matplotlib
import matplotlib.pyplot as plt

from utils import data_utils
from utils import calc_utils
from utils import rcwa_utils

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

path_gold = 'Au3-Drude.txt'
eps_gold_file = data_utils.load_property_txt(path_gold,2743,3636)
eps_gold = eps_gold_file[:,1] + eps_gold_file[:,2]*1j

path_SiNx = 'SiNx_property.mat'
eps_SiNx = data_utils.load_property_mat(path_SiNx)
eps_SiNx = eps_SiNx['eps_SiNx_real'] + eps_SiNx['eps_SiNx_imag']*1j

freq = eps_gold_file[:,0]*1e12

# plot property
n_SiNx = np.sqrt(eps_SiNx)
# print(n_SiNx)
# plt.plot(freq,np.real(n_SiNx))
# plt.plot(freq,np.imag(n_SiNx))
# plt.show()


# # ================= RCWA Solver
# Ly = 0.005 * millimeters  # period along y
# w_weight = 0.48
# w = w_weight * Ly
# R_total, T_total = rcwa_utils.rcwa_solver(freq, eps_gold, eps_SiNx, w=w)
#
#
# # ================= Spectra Plot
# plt.figure(1)
# plt.plot(freq, R_total)
# plt.figure(2)
# plt.plot(freq, T_total)
# plt.show()
#
# path = './data/fRT_w' + str(w_weight) + '.npz'
# np.savez(path, freq=freq, R=R_total, T=T_total)
# print('FILE SAVED')



Ly = 0.005 * millimeters  # period along y

N_w = 10
num_w = 0
w_weight_list = []
# while num_w < N_w:
#     w_weight = np.random.uniform(0.3, 0.6)
#     w_weight = np.around(w_weight, 2)
#     if np.any(np.isin(w_weight_list, w_weight)):
#         pass
#     else:  # not in list, available w
#         w_weight_list = np.append(w_weight_list, w_weight)
#         num_w += 1
#
#     # ================= RCWA Solver
#     w = w_weight * Ly
#     print('[', (num_w), '/', N_w , '] w_weight =', w_weight)
#     R_total, T_total = rcwa_utils.rcwa_solver(freq, eps_gold, eps_SiNx, w=w, use_logger=True)
#
#     # ================= Spectra Plot
#     plt.figure(1)
#     plt.plot(freq, R_total)
#     plt.figure(2)
#     plt.plot(freq, T_total)
#     plt.show()
#
#     path = './data/fRT_w' + str(w_weight) + '.npz'
#     np.savez(path, freq=freq, R=R_total, T=T_total)
#     print('\n')
#     print('FILE SAVED, w_weight =', w_weight)
#     print('----------------')


# ================= RCWA Solver
w_weight = 0.52
w = w_weight * Ly
print('[', (num_w), '/', N_w , '] w_weight =', w_weight)
R_total, T_total = rcwa_utils.rcwa_solver(freq, eps_gold, eps_SiNx, w=w, use_logger=True)

# ================= Spectra Plot
plt.figure(1)
plt.plot(freq, R_total)
plt.figure(2)
plt.plot(freq, T_total)
plt.show()

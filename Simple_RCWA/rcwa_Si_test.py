import numpy as np
from numpy import linalg as LA
from scipy import linalg as SLA
import cmath
import matplotlib
import matplotlib.pyplot as plt
import time

from utils import data_utils
from utils import calc_utils
from utils import rcwa_utils

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

# # ================= Material Property Define
# path_Si = './material_property/Si_Drude_16.txt'
# eps_Si_file = data_utils.load_property_txt(path_Si)
# eps_Si_file = eps_Si_file[1:-1]
# eps_Si = eps_Si_file[:,1] + eps_Si_file[:,2]*1j
#
# freq = eps_Si_file[:,0]*1e12

# plot property
# n_SiNx = np.sqrt(eps_SiNx)
# print(n_SiNx)
# plt.plot(freq,np.real(n_SiNx))
# plt.plot(freq,np.imag(n_SiNx))
# plt.show()


#******************************************************************
'''
Square hole silicone structure test.
'''
# ================= RCWA Solver
# p = 200
# a = 160
# t = 60
# start = time.time()
#
# order = 11
# device = 'CPU'
# R_total, T_total = rcwa_utils.rcwa_solver_Si_test(freq, eps_Si,
#                         L_param=p, w_param=a, t_param=t, use_logger=True, PQ_order=order)  #Solving Time: 57.87713408470154
#
# # device = 'GPU'
# # R_total, T_total = rcwa_utils.rcwa_solver_cuda_Si_test(freq, eps_Si,
# #                         L_param=p, w_param=a, t_param=t, use_logger=True, PQ_order=order)  #Solving Time: 82.6834077835083
#
# end = time.time()
# print('Solving Time:', end - start)

'''
PQ=5
CPU: Solving Time: 57.87713408470154
GPU: Solving Time: 82.6834077835083

PQ=11
CPU: Solving Time: 883.4851624965668
GPU: Solving Time: 752.8701491355896
'''



#******************************************************************
'''
Square hole silicone structure test using class wrapping.
'''
# # ================= Material Structure Define
# p = 200
# a = 160
# t = 60
#
# L_param=p
# w_param=a
# t_param=t
# Lx = L_param * 1e-3 * millimeters  # period along x
# Ly = L_param * 1e-3 * millimeters  # period along y
# d1 = t_param * 1e-3 * millimeters  # thickness of layer 1
#
# params_eps = [eps_Si]
# params_geometry = [Lx, Ly, [d1]]
# params_mesh = [512,512]
# order = 5
# PQ_order = [order,order]
# list_layer_funcs = [rcwa_utils.layerfunc_Si_square_hole]
# device = 'gpu'
#
# # ================= RCWA Solver
# Si_square_hole = rcwa_utils.Material(freq, params_eps, params_geometry, params_mesh, PQ_order, list_layer_funcs,device)
# # ER, UR, ERC, URC = Si_square_hole.rcwa_preprocess()
# start = time.time()
# R_total, T_total = Si_square_hole.rcwa_solve()
# end = time.time()
# print('Solving Time:', end - start)

'''
PQ=5
CPU: Solving Time: 60.17710614204407
GPU: Solving Time: 88.77363109588623
'''



#******************************************************************
'''
Ellipse hole absorber.
'''
# ================= Material Property Define
path_absorber = './material_property/permittivity_absorber.txt'
eps_absorber_file = data_utils.load_property_txt(path_absorber)
eps_absorber_file = eps_absorber_file[::4]
eps_absorber = eps_absorber_file[:,1] + eps_absorber_file[:,2]*1j

freq = eps_absorber_file[:,0]*1e12
print(freq.shape)

# ================= Material Structure Define
a = 160
t = 75

Lx = a * micrometres  # period along x
Ly = a * micrometres  # period along y
d1 = t * micrometres  # thickness of layer 1

params_eps = [eps_absorber]
params_geometry = [Lx, Ly, [d1]]
params_mesh = [512,512]
order = 13
PQ_order = [order,order]
list_layer_funcs = [rcwa_utils.layerfunc_absorber_ellipse_hole]
device = 'gpu'

# ================= RCWA Solver
Si_square_hole = rcwa_utils.Material(freq, params_eps, params_geometry, params_mesh, PQ_order, list_layer_funcs,device)
start = time.time()
R_total, T_total = Si_square_hole.rcwa_solve()
end = time.time()
print('Solving Time:', end - start)

'''
PQ=5
CPU: Solving Time:                    67.06701564788818(iter6)
GPU: Solving Time: 91.44041752815247  91.70914816856384(iter6)

PQ=9
CPU: 
GPU: Solving Time: 385.8824260234833(iter3) 399.268905878067(iter6)

PQ=11
CPU: Solving Time: 925.4946882724762(iter6)
GPU: Solving Time: 764.4430737495422  763.9086158275604(iter3)  795.2279739379883(iter6)  255.897953748703(iter3,1/3)

PQ=13
CPU: 
GPU: Solving Time: 392.55965304374695(iter3,1/4)

PQ=15
CPU: 
GPU: Solving Time: 771.5646812915802(iter3,1/4)

PQ=21
CPU: 
GPU: Solving Time: 2340.4396228790283(iter3,1/6)
'''



# ================= Spectra Plot
plt.figure(1)
plt.plot(freq, R_total)
plt.title('freq-Reflection, PQ='+str(order)+', device='+device)
plt.figure(2)
plt.plot(freq, T_total)
plt.title('freq-Transmission, PQ='+str(order)+', device='+device)
plt.show()

# path = './data/Si_test_p' + str(p) +'a_'+str(a)+'t_'+str(t)+'PQ_'+str(order) + '.npz'
path = './data/absorber_ellipse_hole_PQ_'+str(order)+'.npz'
np.savez(path, freq=freq, R=R_total, T=T_total)
print('FILE SAVED')



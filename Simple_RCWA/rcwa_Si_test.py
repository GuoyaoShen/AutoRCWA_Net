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


# ================= Constant Define
c0 = 3e8
e0 = 8.85e-12
u0 = 1.256e-6
yeta0 = np.sqrt(u0/e0)

path_Si = 'Si_Drude_16.txt'
eps_Si_file = data_utils.load_property_txt(path_Si)
eps_Si_file = eps_Si_file[1:-1]

# eps_Si_file = eps_Si_file[-2:0:-1]
# eps_Si_file = eps_Si_file[1:-2]
# eps_Si_file = eps_Si_file[::2]
eps_Si = eps_Si_file[:,1] + eps_Si_file[:,2]*1j

# path_SiNx = 'SiNx_property.mat'
# eps_SiNx = data_utils.load_property_mat(path_SiNx)
# eps_SiNx = eps_SiNx['eps_SiNx_real'] + eps_SiNx['eps_SiNx_imag']*1j

freq = eps_Si_file[:,0]*1e12

# plot property
# n_SiNx = np.sqrt(eps_SiNx)
# print(n_SiNx)
# plt.plot(freq,np.real(n_SiNx))
# plt.plot(freq,np.imag(n_SiNx))
# plt.show()


# ================= RCWA Solver
p = 200
a = 160
t = 60
start = time.time()
order = 11
device = 'CPU'
R_total, T_total = rcwa_utils.rcwa_solver_Si_test(freq, eps_Si,
                        L_param=p, w_param=a, t_param=t, use_logger=True, PQ_order=order)  #Solving Time: 57.87713408470154

# device = 'GPU'
# R_total, T_total = rcwa_utils.rcwa_solver_cuda_Si_test(freq, eps_Si,
#                         L_param=p, w_param=a, t_param=t, use_logger=True, PQ_order=order)  #Solving Time: 82.6834077835083

end = time.time()
print('Solving Time:', end - start)

# PQ=5
# CPU: Solving Time: 57.87713408470154
# GPU: Solving Time: 82.6834077835083

# PQ=11
# CPU: Solving Time: 883.4851624965668
# GPU: Solving Time: 752.8701491355896


# ================= Spectra Plot
plt.figure(1)
plt.plot(freq, R_total)
plt.title('freq-Reflection, PQ='+str(order)+', device='+device)
plt.figure(2)
plt.plot(freq, T_total)
plt.title('freq-Transmission, PQ='+str(order)+', device='+device)
plt.show()

path = './data/Si_test_p' + str(p) +'a_'+str(a)+'t_'+str(t)+'PQ_'+str(order) + '.npz'
np.savez(path, freq=freq, R=R_total, T=T_total)
print('FILE SAVED')



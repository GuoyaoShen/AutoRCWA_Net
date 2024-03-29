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
nanometres = 1e-9 * meters


# ================= Constant Define
c0 = 3e8
e0 = 8.85e-12
u0 = 1.256e-6
yeta0 = np.sqrt(u0/e0)


#******************************************************************
'''
Metallic BIC.
'''
# ================= Material Property Define
path_absorber = './material_property/permittivity_BIC.txt'
eps_absorber_file = data_utils.load_property_txt(path_absorber)
# print(eps_absorber_file)

# truncate freq over 1.2 THz
freq_truncate = [0.3, 1.2]  # in THz
freq_step = 1
if freq_truncate != 'none' and freq_truncate[0]>eps_absorber_file[0,0] and freq_truncate[1]<eps_absorber_file[-1,0]:
    N_freq_start = np.argmax(eps_absorber_file[:, 0] > freq_truncate[0])
    N_freq_stop = np.argmax(eps_absorber_file[:, 0] > freq_truncate[1])
    eps_absorber_file = eps_absorber_file[N_freq_start: N_freq_stop]

eps_absorber_file = eps_absorber_file[::freq_step]  # solve rcwa with a step size
eps_absorber = eps_absorber_file[:,1] + eps_absorber_file[:,2]*1j

freq = eps_absorber_file[:,0]*1e12
print(freq.min())
print(freq.max())
print(freq.shape)

# ================= Material Structure Define
a = 196
t = 1.5

Lx = a * micrometres  # period along x
Ly = a * micrometres  # period along y
d1 = t * micrometres  # thickness of layer 1

D1 = 100*2 * micrometres  # outter d
D2 = 74*2 * micrometres  # inner d
g1 = 4 * micrometres  # main gap
g2 = 20 * micrometres  # side gap
w = 26 * micrometres  # center strip width

# Lx = a * miumetres  # period along x
# Ly = a * miumetres  # period along y
# d1 = t * miumetres  # thickness of layer 1
#
# D1 = 100*2 * miumetres  # outter d
# D2 = 74*2 * miumetres  # inner d
# g1 = 4 * miumetres  # main gap
# g2 = 20 * miumetres  # side gap
# w = 26 * miumetres  # center strip width

params_eps = [eps_absorber]
params_geometry = [Lx, Ly, [d1]]
params_mesh = [512,512]  #[512,512]
order = 9
PQ_order = [order,order]
list_layer_funcs = [rcwa_utils.layerfunc_metallic_BIC]
list_layer_params = [[D1, D2, g1, g2, w]]
ginc = [0,0,1]  # orig [0,0,1], incident source
EP = [0,1,0]  # orig [0,1,0]
source = [ginc, EP]
device = 'gpu'

# ================= RCWA Solver
Si_square_hole = rcwa_utils.Material(freq, params_eps, params_geometry, params_mesh, PQ_order,
                                     list_layer_funcs, list_layer_params, source, device)
start = time.time()
R_total, T_total = Si_square_hole.rcwa_solve()
end = time.time()
print('Solving Time:', end - start)

# load spectra from real simulation
path_simulation_data_R = './data/BIC/SRR_R_1.txt'
path_simulation_data_T = './data/BIC/SRR_T_1.txt'
simulation_data_R = np.loadtxt(path_simulation_data_R)
simulation_data_T = np.loadtxt(path_simulation_data_T)

# ================= Spectra Plot
plt.figure(1)
plt.plot(freq, R_total, c='b', label='RCWA')
plt.plot(simulation_data_R[:, 0]*1e12, simulation_data_R[:, 1], c='r', label='simulation')
plt.title('freq-Reflection, PQ='+str(order)+', device='+device)
plt.legend()

plt.figure(2)
plt.plot(freq, T_total, c='b', label='RCWA')
plt.plot(simulation_data_T[:, 0]*1e12, simulation_data_T[:, 1], c='r', label='simulation')
plt.title('freq-Transmission, PQ='+str(order)+', device='+device)
plt.legend()
plt.show()

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


# ================= Load True Simulation Spectra
path_true_R = './data/R_absorber.txt'
R_file = data_utils.load_property_txt(path_true_R)
freq_true = R_file[:, 0] * 1e12
R_true = R_file[:, 1]
R_true = R_true**2
print('freq_true.shape', freq_true.shape)
# print(R_true)

path_true_T = './data/T_absorber.txt'
T_file = data_utils.load_property_txt(path_true_T)
T_true = T_file[:, 1]
T_true = T_true**2

# truncate freq
freq_truncate = 1.7  # in THz
N_freq_stop = np.argmax(R_file[:,0]>freq_truncate)
freq_true = freq_true[:N_freq_stop]
R_true = R_true[:N_freq_stop]
T_true = T_true[:N_freq_stop]


# ================= Load RCWA Spectra
path_rcwa = './data/absorber_ellipse_hole_PQ_21.npz'
data = np.load(path_rcwa)
freq_rcwa = data['freq']
R_rcwa = data['R']
T_rcwa = data['T']
print('freq_rcwa.shape', freq_rcwa.shape)
# print(R_rcwa)

path_rcwa = './data/absorber_ellipse_hole_PQ_15.npz'
data = np.load(path_rcwa)
freq_rcwa15 = data['freq']
R_rcwa15 = data['R']
T_rcwa15 = data['T']

path_rcwa = './data/absorber_ellipse_hole_PQ_13.npz'
data = np.load(path_rcwa)
freq_rcwa13 = data['freq']
R_rcwa13 = data['R']
T_rcwa13 = data['T']

path_rcwa = './data/absorber_ellipse_hole_PQ_11.npz'
data = np.load(path_rcwa)
freq_rcwa11 = data['freq']
R_rcwa11 = data['R']
T_rcwa11 = data['T']

path_rcwa = './data/absorber_ellipse_hole_PQ_9.npz'
data = np.load(path_rcwa)
freq_rcwa9 = data['freq']
R_rcwa9 = data['R']
T_rcwa9 = data['T']

path_rcwa = './data/absorber_ellipse_hole_PQ_7.npz'
data = np.load(path_rcwa)
freq_rcwa7 = data['freq']
R_rcwa7 = data['R']
T_rcwa7 = data['T']

path_rcwa = './data/absorber_ellipse_hole_PQ_5.npz'
data = np.load(path_rcwa)
freq_rcwa5 = data['freq']
R_rcwa5 = data['R']
T_rcwa5 = data['T']


# ================= Plot
plt.figure(1)
plt.plot(freq_true, R_true, label='true')
# plt.plot(freq_rcwa, R_rcwa, label='rcwa21')
# plt.plot(freq_rcwa15, R_rcwa15, label='rcwa15')
# plt.plot(freq_rcwa13, R_rcwa13, label='rcwa13')
# plt.plot(freq_rcwa11, R_rcwa11, label='rcwa11')
plt.plot(freq_rcwa9, R_rcwa9, label='rcwa9')
plt.plot(freq_rcwa7, R_rcwa7, label='rcwa7')
# plt.plot(freq_rcwa5, R_rcwa5, label='rcwa5')
plt.title('freq-Reflection')
plt.legend()

plt.figure(2)
plt.plot(freq_true, T_true, label='true')
# plt.plot(freq_rcwa, T_rcwa, label='rcwa21')
# plt.plot(freq_rcwa15, T_rcwa15, label='rcwa15')
# plt.plot(freq_rcwa13, T_rcwa13, label='rcwa13')
# plt.plot(freq_rcwa11, T_rcwa11, label='rcwa11')
plt.plot(freq_rcwa9, T_rcwa9, label='rcwa9')
plt.plot(freq_rcwa7, T_rcwa7, label='rcwa7')
# plt.plot(freq_rcwa5, T_rcwa5, label='rcwa5')
plt.title('freq-Transmission')
plt.legend()
plt.show()
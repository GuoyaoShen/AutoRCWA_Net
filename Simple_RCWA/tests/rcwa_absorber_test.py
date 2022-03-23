import numpy as np
from numpy import linalg as LA
from scipy import linalg as SLA
import cmath
import matplotlib
import matplotlib.pyplot as plt
import time

from Simple_RCWA.utils import data_utils
from Simple_RCWA.utils import calc_utils
from Simple_RCWA.utils import rcwa_utils

# for OMP Error #15
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

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
Absorber structure
'''
# ================= Material Property Define
path_absorber = '../material_property/permittivity_absorber.txt'
eps_absorber_file = data_utils.load_property_txt(path_absorber)
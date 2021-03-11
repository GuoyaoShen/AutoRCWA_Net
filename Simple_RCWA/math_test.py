import numpy as np
import scipy
from numpy import linalg as LA
from scipy import linalg as SLA
import cmath
import matplotlib
import matplotlib.pyplot as plt

from time import sleep
import sys

from Simple_RCWA.utils import data_utils
from Simple_RCWA.utils import calc_utils


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


# === Device Params
ur1 = 1.  # permeability in reflection region
er1 = 1.  # permeability in reflection region
ur2 = 1.  # permeability in transmission region
er2 = 1.  # permeability in transmission region
urd = 1.  # permeability of device
lam0 = 0.0006
ginc = np.array([0, 0, 1])

Lx = 0.0002
Ly = 0.0002

PQ = 1 * np.array([31, 31])


# def func1(x,a):
#     return x+a
#
# def func2(x,a):
#     return x*a
#
# list_func = [func1, func2]
#
# print(list_func[1](10,2))

# a = np.array([3,4,7,9,10])
a = np.array([[0.5,7,8],[0.5,8,9],[0.6,9,10],[0.8,10,11]])
print(a[:,0]>0.6)
print(np.argmax(a[:,0]>0.6))
print(np.argmin(a[:,0]>0.6))

print(a[:,0]<=0.6)
print(np.argmax(a[:,0]<=0.6))
print(np.argmin(a[:,0]<=0.6))

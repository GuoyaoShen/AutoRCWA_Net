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




# [2.64575131+0.j 2.        +0.j 1.73205081+0.j 2.        +0.j
#  2.64575131+0.j 2.        +0.j 1.        +0.j 0.        +0.j
#  1.        +0.j 2.        +0.j 1.73205081+0.j 0.        +0.j
#  0.        +1.j 0.        +0.j 1.73205081+0.j 2.        +0.j
#  1.        +0.j 0.        +0.j 1.        +0.j 2.        +0.j
#  2.64575131+0.j 2.        +0.j 1.73205081+0.j 2.        +0.j
#  2.64575131+0.j 2.64575131+0.j 2.        +0.j 1.73205081+0.j
#  2.        +0.j 2.64575131+0.j 2.        +0.j 1.        +0.j
#  0.        +0.j 1.        +0.j 2.        +0.j 1.73205081+0.j
#  0.        +0.j 0.        +1.j 0.        +0.j 1.73205081+0.j
#  2.        +0.j 1.        +0.j 0.        +0.j 1.        +0.j
#  2.        +0.j 2.64575131+0.j 2.        +0.j 1.73205081+0.j
#  2.        +0.j 2.64575131+0.j]

# A = np.diag(
# [2.64575131+0.j, 2.        +0.j, 1.73205081+0.j, 2.        +0.j,
#  2.64575131+0.j, 2.        +0.j, 1.        +0., 0.        +0.j,
#  1.        +0.j, 2.        +0.j, 1.73205081+0.j, 0.        +0.j,
#  0.        +1.j, 0.        +0.j, 1.73205081+0.j, 2.        +0.j,
#  1.        +0.j, 0.        +0.j, 1.        +0.j, 2.        +0.j,
#  2.64575131+0.j, 2.        +0.j, 1.73205081+0.j, 2.        +0.j,
#  2.64575131+0.j, 2.64575131+0.j, 2.        +0.j, 1.73205081+0.j,
#  2.        +0.j, 2.64575131+0.j, 2.        +0.j, 1.        +0.j,
#  0.        +0.j, 1.        +0.j, 2.        +0.j, 1.73205081+0.j,
#  0.        +0.j, 0.        +1.j, 0.        +0.j, 1.73205081+0.j,
#  2.        +0.j, 1.        +0.j, 0.        +0.j, 1.        +0.j,
#  2.        +0.j, 2.64575131+0.j, 2.        +0.j, 1.73205081+0.j,
#  2.        +0.j, 2.64575131+0.j]
# )
#
# print(A.shape)
# print(LA.inv(A))

A = np.array([1,2,3,4,5,6,7])
print(A[1:-1])
print(A[-1])

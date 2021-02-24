import numpy as np
import scipy.io as sio

def load_property_txt(PATH, line_start=-1, line_end=-1):
    eps_property = np.loadtxt(PATH)
    if line_start==-1 & line_end==-1:
        pass
    else:
        eps_property = eps_property[line_start:line_end,...]

    return eps_property

def load_property_mat(PATH):
    eps_property = sio.loadmat(PATH)

    return eps_property

import numpy as np
import scipy
from numpy import linalg as LA
from scipy import linalg as SLA
import cmath
import matplotlib
import matplotlib.pyplot as plt

def convmat(A,P,Q=1,R=1):
    '''
    % CONVMAT Rectangular Convolution Matrix
    %
    % C = convmat(A,P); for 1D problems
    % C = convmat(A,P,Q); for 2D problems
    % C = convmat(A,P,Q,R); for 3D problems
    %
    % This function constructs convolution matrices
    % from a real space grid.
    '''
    # Initialize C
    C_size = P*Q*R
    C = np.zeros((C_size,C_size)).astype(complex)

    # Determine Size of A
    Nx,Ny,Nz = A.shape

    # Compute Indices of Spatial Harmonics
    NH = P*Q*R  # total number
    p = np.arange(-(np.floor(P / 2)), (np.floor(P / 2) + 1))  # indices along x
    q = np.arange(-(np.floor(Q / 2)), (np.floor(Q / 2) + 1))  # indices along y
    r = np.arange(-(np.floor(R / 2)), (np.floor(R / 2) + 1))  # indices along z

    # Compute Fourier Coefficients of A
    A = np.fft.fftshift(scipy.fft.fftn(A)) / (Nx*Ny*Nz)

    # Compute Array Indices of Center Harmonic
    p0 = 1 + np.floor(Nx / 2)
    q0 = 1 + np.floor(Ny / 2)
    r0 = 1 + np.floor(Nz / 2)

    for rrow in range(R):
        for qrow in range(Q):
            for prow in range(P):
                row = (rrow)*Q*P + (qrow)*P + (prow+1)
                for rcol in range(R):
                    for qcol in range(Q):
                        for pcol in range(P):
                            col = rcol*Q*P + qcol*P + (pcol+1)
                            pfft = p[prow] - p[pcol]
                            qfft = q[qrow] - q[qcol]
                            rfft = r[rrow] - r[rcol]
                            C[(row-1).astype(int),(col-1).astype(int)] = \
                                A[(p0+pfft-1).astype(int),(q0+qfft-1).astype(int),(r0+rfft-1).astype(int)]

    # C = C[...,np.newaxis]
    return C


def star(SA, SB):
    '''
    STAR Redheffer Star Product

    % INPUT ARGUMENTS
    % ================
    % SA First Scattering Matrix
    % .S11 S11 scattering parameter
    % .S12 S12 scattering parameter
    % .S21 S21 scattering parameter
    % .S22 S22 scattering parameter
    %
    % SB Second Scattering Matrix
    % .S11 S11 scattering parameter
    % .S12 S12 scattering parameter
    % .S21 S21 scattering parameter
    % .S22 S22 scattering parameter
    %
    % OUTPUT ARGUMENTS
    % ================
    % S Combined Scattering Matrix
    % .S11 S11 scattering parameter
    % .S12 S12 scattering parameter
    % .S21 S21 scattering parameter
    % .S22 S22 scattering parameter
    '''
    N = SA['S11'].shape[0]
    I = np.eye(N)
    S11 = SA['S11'] + (SA['S12'] @ LA.inv(I - (SB['S11'] @ SA['S22'])) @ SB['S11'] @ SA['S21'])
    S12 = SA['S12'] @ LA.inv(I - (SB['S11'] @ SA['S22'])) @ SB['S12']
    S21 = SB['S21'] @ LA.inv(I - (SA['S22'] @ SB['S11'])) @ SA['S21']
    S22 = SB['S22'] + (SB['S21'] @ LA.inv(I - (SA['S22'] @ SB['S11'])) @ SA['S22'] @ SB['S12'])
    S = {'S11': S11, 'S12': S12, 'S21': S21, 'S22': S22}
    return S
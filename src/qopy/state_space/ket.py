'''

Contains all the operation creating Ket vectors, or acting on ket vectors

'''


import numpy as np
import math

def ket_sqvac(N, r, phi=0):
    ket = np.zeros(N, dtype=complex)
    nmax = int((N+1)/2)
    for n in range(nmax):
        ket[2*n] = (np.tanh(r)/2)**n*np.sqrt(float(math.factorial(2*n)))/math.factorial(n)
    ket = ket/np.sqrt(np.cosh(r))
    if phi != 0:
        ket = ket_rotate(ket, phi)
    return ket

def ket_coh(alpha, N):
    ket = np.zeros(N, dtype=complex)
    ket[0] = np.exp(-np.abs(alpha)**2/2)
    for n in range(1, N):
        ket[n] = ket[n-1]*alpha/np.sqrt(n)
    return ket


def ket_rotate(ket, phi):
    N = len(ket)
    for i in range(N):
        ket[i] = ket[i]*np.exp(-1j*phi*i)
    return ket


def ket_disp(ket, d):
    # d = sqrt(2)*alpha
    if not (isinstance(d, list) or (isinstance(d, np.ndarray))):
        d = [np.real(d), np.imag(d)]
    alpha = (d[0]+1j*d[1])/np.sqrt(2)
    N = len(ket)
    dispket = np.zeros(N, dtype=complex)
    for m in range(N):
        km = 0
        for n in range(N):
            km = km+ket[n]*disp_mat(m, n, alpha)
        dispket[m] = km
    return dispket


def ket_norm(ket, phased=True, length=None):
    # Return a normalized ket
    if type(ket) is not np.array:
        ket = np.array(ket, dtype=complex)
    ket = ket.astype(complex) / np.sqrt(np.sum(np.abs(ket) ** 2))
    if phased:
        nonz = 0
        while ket[nonz] == 0:
            nonz = nonz+1
        th = np.angle(ket[nonz])
        ket = np.exp(-1j*th)*ket
        ket[nonz] = np.real(ket[nonz])
    if length is not None:
        ket = np.concatenate([ket, np.zeros(length-len(ket))])
    return ket
'''

Contains all the operation creating Ket vectors, or acting on ket vectors

'''


import numpy as np
import math
import scipy
from random import random as random_01
from qopy.state_space.density import displacement_matrix


def squeezed_vacuum(N, r, phi=0):
    ket = np.zeros(N, dtype=complex)
    nmax = int((N+1)/2)
    for n in range(nmax):
        ket[2*n] = (np.tanh(r)/2)**n*np.sqrt(float(math.factorial(2*n)))/math.factorial(n)
    ket = ket/np.sqrt(np.cosh(r))
    if phi != 0:
        ket = rotate(ket, phi)
    return ket


def coherent(alpha, N):
    ket = np.zeros(N, dtype=complex)
    ket[0] = np.exp(-np.abs(alpha)**2/2)
    for n in range(1, N):
        ket[n] = ket[n-1]*alpha/np.sqrt(n)
    return ket


def rotate(ket, phi):
    N = len(ket)
    for i in range(N):
        ket[i] = ket[i]*np.exp(-1j*phi*i)
    return ket


def displace(ket, alpha):
    N = len(ket)
    dispket = np.zeros(N, dtype=complex)
    for m in range(N):
        km = 0
        for n in range(N):
            km = km + ket[n] * displacement_matrix(m, n, alpha)
        dispket[m] = km
    return dispket


def normalize(ket, phased=False, length=None):
    # Return a normalized ket
    # If phased is True, then the first non-zero entry of ket is real positive
    # If length=N, zeros are padded to ket in order to make it of length N
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


def random(N):
    if N == 1:
        return np.array([np.exp(1j*random_01()*2*math.pi)])
    return scipy.stats.unitary_group.rvs(N)[0]

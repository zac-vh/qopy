import numpy as np
import scipy
from qopy.state_space.gaussian_unitary.beamsplitter import sigma_mn


def beam_splitter(m, n):
    N = m+n+1
    eigvals = np.zeros(N)
    for k in range(N):
        eigvals[k] = sigma_mn(m, n)
    return np.diag(eigvals)


def binomial(n):
    diag = np.zeros(n+1)
    for k in range(n+1):
        diag[k] = scipy.special.comb(n, k)
    return np.diag(diag/(2**n))


def extreme_passive(n):
    return np.eye(n+1)/(n+1)
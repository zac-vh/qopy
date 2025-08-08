import numpy as np
import scipy
from qopy.state_space.gaussian_unitary.beamsplitter import transition_amplitude


def beam_splitter(m, n, eta=0.5):
    N = m+n+1
    eigvals = np.zeros(N)
    for k in range(N):
        eigvals[k] = transition_amplitude(m, n, k, eta) ** 2
    return np.diag(eigvals)


def binomial(n):
    diag = np.zeros(n+1)
    for k in range(n+1):
        diag[k] = scipy.special.comb(n, k)
    return np.diag(diag/(2**n))


def extreme_passive(n):
    return np.eye(n+1)/(n+1)